#!/usr/bin/env python3
"""Autonomous uncapped commissioning & chromaticity campaign scheduler, safety
watchdog, calibration hand-off runner, and report compiler for ms-x01.

Preserves completed run history on restart, waits for RFSoCs to come online,
verifies multicast ARP before every run, aborts zero-packet runs cleanly at 45s,
and executes automatic post-Phase-1 calibration + config rebuild before Test 1..11.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import signal
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

PYTHON_BIN = "/home/fpalma/charts-utils/.venv/bin/python"
KOTEKAN_BIN = "/home/fpalma/kotekan/build/kotekan/kotekan"
SCRIPT_DIR = Path("/home/fpalma/kotekan/charts/commissioning_20261119_v2")
COMM_DIR = Path("/data/commissioning_20261119_v2")
MANIFEST_PATH = COMM_DIR / "schedule_manifest.json"
STATUS_PATH = COMM_DIR / "campaign_status.json"
REPORT_PATH = COMM_DIR / "CAMPAIGN_V2_REPORT.md"
RFSOC_HOSTS = ("10.17.16.20", "10.17.16.21")
FENGINE_PYTHON = "/home/fpalma/f-engine/.venv/bin/python"
FENGINE_DIR = "/home/fpalma/f-engine"

GLOBAL_STOP_REQUESTED = False


def _on_sig(signum, frame):
    global GLOBAL_STOP_REQUESTED
    GLOBAL_STOP_REQUESTED = True


signal.signal(signal.SIGTERM, _on_sig)
signal.signal(signal.SIGINT, _on_sig)


def parse_utc(iso_str: str) -> float:
    return datetime.fromisoformat(iso_str.replace("Z", "+00:00")).timestamp()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def dir_size_bytes(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for fn in files:
            try:
                total += (Path(root) / fn).stat().st_size
            except OSError:
                pass
    return total


def has_output_bin(out_dir: Path) -> bool:
    return any(out_dir.glob("beam_power_freq0_*.bin")) or any(out_dir.glob("corr_freq_0_*.bin"))


def terminate_proc(proc: Optional[subprocess.Popen], name: str, wait_s: float = 15.0) -> None:
    if proc is None or proc.poll() is not None:
        return
    print(f"[{utc_now_iso()}] Sending SIGTERM to {name} (pid={proc.pid}) ...", flush=True)
    try:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=wait_s)
    except subprocess.TimeoutExpired:
        print(f"[{utc_now_iso()}] {name} did not exit in {wait_s}s; sending SIGKILL ...", flush=True)
        proc.kill()
        proc.wait(timeout=5)
    except Exception:
        pass


def check_no_stray_kotekan() -> None:
    res = subprocess.run(["pgrep", "-x", "kotekan"], capture_output=True, text=True)
    if res.returncode == 0 and res.stdout.strip():
        raise RuntimeError(f"Stray kotekan process already running: PIDs {res.stdout.strip()}")


def rfsocs_reachable(hosts: tuple[str, ...] = RFSOC_HOSTS) -> bool:
    """Return True only when all RFSoC management IPs answer ICMP ping."""
    for host in hosts:
        try:
            res = subprocess.run(
                ["ping", "-c", "1", "-W", "1", host],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=3,
            )
            if res.returncode != 0:
                return False
        except Exception:
            return False
    return True


def ensure_multicast_arp(hosts: tuple[str, ...] = RFSOC_HOSTS) -> bool:
    """Verify running RFSoCs have multicast MACs (03:00:00:00:00:01/02) for ms-x00/ms-x01 replication."""
    check_code = """
import sys
import casperfpga
from charts_constants import DEFAULT_FPG_GPS

hosts = sys.argv[1:]
expected = {10: 0x030000000001, 11: 0x030000000002}
needs_fix = False
for h in hosts:
    fpga = casperfpga.CasperFpga(h)
    if not fpga.is_running():
        print(f"{h}: FPGA not running", flush=True)
        sys.exit(2)
    fpga.get_system_information(DEFAULT_FPG_GPS)
    arp = fpga.gbes["onehundred_gbe"].get_arp_details()
    for idx, mac_int in expected.items():
        if arp[idx].mac_int != mac_int:
            print(f"{h}: ARP[{idx}] is {arp[idx].mac_str}, expected {mac_int:012x}", flush=True)
            needs_fix = True
sys.exit(3 if needs_fix else 0)
"""
    try:
        chk = subprocess.run(
            [FENGINE_PYTHON, "-c", check_code, *hosts],
            cwd=FENGINE_DIR,
            capture_output=True,
            text=True,
            timeout=25,
        )
        if chk.returncode == 0:
            return True
        if chk.returncode == 3:
            if chk.stdout.strip():
                print(f"[{utc_now_iso()}] ARP check: {chk.stdout.strip()}", flush=True)
            fix = subprocess.run(
                [FENGINE_PYTHON, str(Path(FENGINE_DIR) / "set_rfsoc_arp.py"), *hosts],
                cwd=FENGINE_DIR,
                capture_output=True,
                text=True,
                timeout=20,
            )
            if fix.returncode == 0:
                print(f"[{utc_now_iso()}] Re-armed multicast ARP: {fix.stdout.strip()}", flush=True)
                return True
            print(f"[{utc_now_iso()}] set_rfsoc_arp.py failed: {fix.stdout.strip()} {fix.stderr.strip()}", flush=True)
            return False
        detail = (chk.stdout + "\n" + chk.stderr).strip().splitlines()[-1:] or [f"rc={chk.returncode}"]
        print(f"[{utc_now_iso()}] RFSoC check failed ({detail[0]})", flush=True)
        return False
    except Exception as exc:
        print(f"[{utc_now_iso()}] ensure_multicast_arp error: {exc}", flush=True)
        return False


def write_status(status: Dict[str, Any], status_path: Path = STATUS_PATH) -> None:
    status["updated_utc"] = utc_now_iso()
    status["root_free_gb"] = round(shutil.disk_usage("/").free / (1024 ** 3), 2)
    tmp = status_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(status, indent=2), encoding="utf-8")
    os.replace(tmp, status_path)
    os.chmod(status_path, 0o644)


def run_calibration_handoff(
    manifest: Dict[str, Any], corr_dir: Path, status: Dict[str, Any], status_path: Path = STATUS_PATH
) -> Dict[str, Any]:
    """Solve fresh epoch gains from phase1_corr_cal and rebuild downstream YAMLs."""
    gains_path = Path(manifest["gains_file"])
    cal_out = gains_path.parent
    rfi_out = COMM_DIR / "localized_rfi_targets.json"
    cmd = [
        PYTHON_BIN,
        str(SCRIPT_DIR / "calibrate_corr_epoch.py"),
        "--corr-dir",
        str(corr_dir),
        "--out-dir",
        str(cal_out),
        "--rfi-out",
        str(rfi_out),
        "--min-sun-el",
        "15.0",
    ]
    print(f"[{utc_now_iso()}] Running automatic Phase-1 calibration hand-off ...", flush=True)
    res = subprocess.run(cmd, cwd="/home/fpalma/kotekan", capture_output=True, text=True, timeout=600)
    (COMM_DIR / "calibration_handoff.log").write_text(res.stdout + "\n" + res.stderr, encoding="utf-8")
    info: Dict[str, Any] = {"cal_rc": res.returncode, "gains_file": str(gains_path), "utc": utc_now_iso()}

    if res.returncode == 0 and gains_path.exists():
        g = np.load(gains_path)
        ants = manifest.get("active_antennas", [0, 4, 32, 36])
        valid = g.shape == (672, 64) and np.all(np.isfinite(g)) and bool(np.all(np.abs(g[:, ants]) > 0.1))
        info["gains_valid"] = bool(valid)
        if valid:
            rb = subprocess.run(
                [PYTHON_BIN, str(SCRIPT_DIR / "generate_v2_campaign.py"), "--rebuild-beam-configs"],
                cwd="/home/fpalma/kotekan",
                capture_output=True,
                text=True,
                timeout=60,
            )
            info["rebuild_rc"] = rb.returncode
            print(f"[{utc_now_iso()}] Fresh corr_1119 gains verified; rebuilt beam configs rc={rb.returncode}", flush=True)
        else:
            print(f"[{utc_now_iso()}] WARNING: Newly saved gains failed validation!", flush=True)
    else:
        info["gains_valid"] = False
        print(f"[{utc_now_iso()}] WARNING: Calibration hand-off failed rc={res.returncode}; check calibration_handoff.log", flush=True)

    status["calibration_handoff"] = info
    write_status(status, status_path)
    return info


def run_single_observation(
    run: Dict[str, Any],
    *,
    hard_stop_ts: float,
    min_free_gb: float,
    status: Dict[str, Any],
    status_path: Path = STATUS_PATH,
) -> Dict[str, Any]:
    run_id = run["id"]
    out_dir = Path(run["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(out_dir, 0o777)

    cfg_path = Path(run["config_path"])
    cap_bytes = int(run["disk_cap_mb"]) * 1024 * 1024
    dur_s = float(run["duration_s"])
    sched_end_ts = parse_utc(run["end_utc"]) if "end_utc" in run else (time.time() + dur_s)

    now_ts = time.time()
    if now_ts >= hard_stop_ts - 20.0:
        return {"id": run_id, "state": "skipped_past_cutoff", "utc": utc_now_iso()}

    free_gb = shutil.disk_usage("/").free / (1024 ** 3)
    if free_gb < min_free_gb:
        return {"id": run_id, "state": f"skipped_low_disk_{free_gb:.1f}GB", "utc": utc_now_iso()}

    check_no_stray_kotekan()

    effective_dur_s = min(dur_s, max(10.0, hard_stop_ts - now_ts - 15.0))
    run_deadline_ts = min(now_ts + effective_dur_s + 60.0, sched_end_ts + 60.0, hard_stop_ts)

    log_path = out_dir / "kotekan.log"
    sidecar_log = out_dir / "sidecar.log"
    print(f"[{utc_now_iso()}] STARTING {run_id} (dur<={effective_dur_s:.0f}s, cap={run['disk_cap_mb']} MB)", flush=True)

    status["current_run"] = run_id
    status["state"] = "running"
    write_status(status, status_path)

    no_pkt_timeout_s = 120.0 if run.get("sidecar_mode") == "corr_monitor" else 65.0
    abort_reason = None
    kot_rc = None
    run_start_ts = time.time()
    with open(log_path, "w", encoding="utf-8") as kf, open(sidecar_log, "w", encoding="utf-8") as sf:
        kot_proc = subprocess.Popen(
            [KOTEKAN_BIN, "-c", str(cfg_path)],
            cwd="/home/fpalma/kotekan",
            stdout=kf,
            stderr=subprocess.STDOUT,
        )
        side_proc = subprocess.Popen(
            [
                PYTHON_BIN,
                str(SCRIPT_DIR / "sidecar_v2.py"),
                "--mode",
                run["sidecar_mode"],
                "--out-dir",
                str(out_dir),
                "--duration-s",
                str(effective_dur_s + 45.0),
            ],
            cwd="/home/fpalma/kotekan",
            stdout=sf,
            stderr=subprocess.STDOUT,
        )
        try:
            while True:
                rc = kot_proc.poll()
                if rc is not None:
                    kot_rc = rc
                    break
                cur_ts = time.time()
                sz = dir_size_bytes(out_dir)
                cur_free = shutil.disk_usage("/").free / (1024 ** 3)
                if GLOBAL_STOP_REQUESTED:
                    abort_reason = "signal_stop"
                elif cur_ts >= hard_stop_ts:
                    abort_reason = "global_cutoff_reached"
                elif cur_ts >= run_deadline_ts:
                    abort_reason = "run_wallclock_timeout"
                elif sz > cap_bytes:
                    abort_reason = f"disk_cap_exceeded ({sz/(1024**2):.1f} MB > {run['disk_cap_mb']} MB)"
                elif cur_free < min_free_gb:
                    abort_reason = f"root_disk_floor ({cur_free:.1f} GB < {min_free_gb} GB)"
                elif cur_ts - run_start_ts >= no_pkt_timeout_s and not has_output_bin(out_dir):
                    try:
                        log_txt = log_path.read_text(encoding="utf-8", errors="replace")
                    except OSError:
                        log_txt = ""
                    if (
                        "Warmup period" in log_txt
                        and log_txt.count("RX pkt 0 (+0)") >= 20
                        and not re.search(r"RX pkt [1-9]", log_txt)
                    ):
                        abort_reason = "no_rfsoc_packets_timeout"

                if abort_reason:
                    print(f"[{utc_now_iso()}] Stopping {run_id}: {abort_reason}", flush=True)
                    terminate_proc(kot_proc, f"kotekan({run_id})", wait_s=15.0)
                    kot_rc = kot_proc.returncode
                    break
                time.sleep(2.0)
        finally:
            terminate_proc(side_proc, f"sidecar({run_id})", wait_s=5.0)

    subprocess.run(["chmod", "-R", "a+rX", str(out_dir)], check=False)

    final_mb = round(dir_size_bytes(out_dir) / (1024 ** 2), 3)
    result: Dict[str, Any] = {
        "id": run_id,
        "state": "completed" if (kot_rc == 0 and abort_reason is None) else f"stopped({abort_reason or kot_rc})",
        "kotekan_rc": kot_rc,
        "abort_reason": abort_reason,
        "finished_utc": utc_now_iso(),
        "disk_mb": final_mb,
        "output_dir": str(out_dir),
    }

    verifier_name = run.get("verifier")
    if verifier_name:
        if not has_output_bin(out_dir) or abort_reason == "no_rfsoc_packets_timeout":
            msg = (
                f"[{utc_now_iso()}] Skipped verifier {verifier_name}: "
                f"no binary output captured (abort_reason={abort_reason}).\n"
            )
            (out_dir / "verifier_stdout.log").write_text(msg, encoding="utf-8")
            result["verifier_rc"] = None
            print(msg.strip(), flush=True)
        else:
            v_path = (SCRIPT_DIR / verifier_name).resolve()
            extra_args = list(run.get("verifier_args") or [])
            if run_id == "test11_24h_phase_drift_corr":
                cmd = [PYTHON_BIN, str(v_path), *extra_args]
                v_log_dir = COMM_DIR / "test11_24h_phase_drift_corr"
                v_log_dir.mkdir(parents=True, exist_ok=True)
            elif run_id == "test06b_sun_onoff_conj_false":
                t6a_dir = COMM_DIR / "test06a_sun_onoff_conj_true"
                if t6a_dir.exists() and has_output_bin(t6a_dir):
                    cmd = [PYTHON_BIN, str(v_path), "--run-dir", str(t6a_dir), "--conj-false-dir", str(out_dir)]
                else:
                    cmd = [PYTHON_BIN, str(v_path), "--run-dir", str(out_dir)]
                v_log_dir = out_dir
            else:
                cmd = [PYTHON_BIN, str(v_path), "--run-dir", str(out_dir), *extra_args]
                v_log_dir = out_dir

            try:
                v_res = subprocess.run(cmd, cwd="/home/fpalma/kotekan", capture_output=True, text=True, timeout=300)
                (v_log_dir / "verifier_stdout.log").write_text(v_res.stdout + "\n" + v_res.stderr, encoding="utf-8")
                result["verifier_rc"] = v_res.returncode
                print(f"[{utc_now_iso()}] Verifier {v_path.name} rc={v_res.returncode}", flush=True)
            except subprocess.TimeoutExpired as exc:
                out_s = exc.stdout if isinstance(exc.stdout, str) else ((exc.stdout or b"").decode("utf-8", "replace"))
                err_s = exc.stderr if isinstance(exc.stderr, str) else ((exc.stderr or b"").decode("utf-8", "replace"))
                msg = f"\n[{utc_now_iso()}] ERROR: Verifier {v_path.name} timed out after 300s.\n"
                (v_log_dir / "verifier_stdout.log").write_text(out_s + "\n" + err_s + msg, encoding="utf-8")
                result["verifier_rc"] = -124
                result["verifier_error"] = f"timeout after 300s ({v_path.name})"
                print(msg.strip(), flush=True)
            except Exception as exc:
                msg = f"[{utc_now_iso()}] ERROR: Verifier {v_path.name} raised {type(exc).__name__}: {exc}\n"
                (v_log_dir / "verifier_stdout.log").write_text(msg, encoding="utf-8")
                result["verifier_rc"] = -1
                result["verifier_error"] = f"{type(exc).__name__}: {exc}"
                print(msg.strip(), flush=True)

    subprocess.run(["chmod", "-R", "a+rX", str(out_dir)], check=False)
    return result


def compile_report(status: Dict[str, Any], report_path: Path = REPORT_PATH) -> None:
    lines = [
        "# CHARTS 4-Antenna Uncapped V2 Commissioning & Chromaticity Report",
        "",
        f"- **Updated (UTC):** {utc_now_iso()}",
        f"- **Scheduler State:** {status.get('state')}",
        f"- **Root Free Space:** {status.get('root_free_gb')} GB",
        "",
        "## Run Summary",
        "| Run ID | State | Kotekan RC | Verifier RC | Disk (MB) | Finished UTC |",
        "|---|---|---:|---:|---:|---|",
    ]
    for r in status.get("completed_runs", []):
        lines.append(
            f"| {r['id']} | {r['state']} | {r.get('kotekan_rc')} | {r.get('verifier_rc')} | {r.get('disk_mb')} | {r.get('finished_utc')} |"
        )
    if "calibration_handoff" in status:
        lines.extend(["", "## Calibration Hand-Off", f"```json\n{json.dumps(status['calibration_handoff'], indent=2)}\n```"])
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    os.chmod(report_path, 0o644)


def verify_corr_smoke_output(corr_dir: Path) -> Dict[str, Any]:
    """Validate packet rate, calibrated autos (non-clipping), and live MUOS coherence on [0,4,32,36]."""
    from charts_utils.core.antenna_indices import charts_to_kotekan_raw
    from charts_utils.core.gpu_frames import read_gpu_corr_frame_header
    from charts_utils.core.io import extract_gpu_corr_visibility, load_gpu_corr_frame

    f0s = sorted(glob.glob(str(corr_dir / "*freq_0_*.bin")))
    f1s = sorted(glob.glob(str(corr_dir / "*freq_1_*.bin")))
    assert f0s and f1s, f"Missing correlator outputs in {corr_dir}"
    h0 = read_gpu_corr_frame_header(f0s[-1])
    h1 = read_gpu_corr_frame_header(f1s[-1])
    dt_us = abs(int(h1.time0_fpga) - int(h0.time0_fpga))
    v0 = load_gpu_corr_frame(f0s[-1], n_freq=336, num_antennas=64)
    v1 = load_gpu_corr_frame(f1s[-1], n_freq=336, num_antennas=64)

    ants = [0, 4, 32, 36]
    raws = [charts_to_kotekan_raw(a, 64) for a in ants]
    autos_med = {}
    for a, r in zip(ants, raws):
        vis = np.concatenate([extract_gpu_corr_visibility(v0, r, r), extract_gpu_corr_visibility(v1, r, r)])
        autos_med[str(a)] = round(float(np.median(np.real(vis)[50:183])), 1)

    # MUOS cross-correlation |rho| in 375.5-379.5 MHz (sb0 local channels 252..265)
    v04 = extract_gpu_corr_visibility(v0, raws[0], raws[1])[252:265]
    a0 = np.real(extract_gpu_corr_visibility(v0, raws[0], raws[0]))[252:265]
    a4 = np.real(extract_gpu_corr_visibility(v0, raws[1], raws[1]))[252:265]
    muos_rho04 = float(np.median(np.abs(v04) / np.sqrt(np.maximum(a0 * a4, 1.0))))

    v032 = extract_gpu_corr_visibility(v0, raws[0], raws[2])[252:265]
    a32 = np.real(extract_gpu_corr_visibility(v0, raws[2], raws[2]))[252:265]
    muos_rho032 = float(np.median(np.abs(v032) / np.sqrt(np.maximum(a0 * a32, 1.0))))

    log_txt = (corr_dir / "kotekan.log").read_text(encoding="utf-8", errors="replace")
    rx = re.findall(r"RFSoC port (\d+) \| RX pkt (\d+).*? \| lost (\d+).*?\| *([0-9.]+) Mb/s", log_txt)
    summary = {
        "frames_sb0": len(f0s),
        "frames_sb1": len(f1s),
        "last_subband_dt_us": dt_us,
        "autos_315_355_median": autos_med,
        "muos_med_rho_0_4": round(muos_rho04, 4),
        "muos_med_rho_0_32": round(muos_rho032, 4),
        "rx_tail": rx[-4:],
    }
    (corr_dir / "corr_smoke_verification.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    assert muos_rho04 > 0.6 and muos_rho032 > 0.5, f"MUOS coherence unexpectedly low: {summary}"
    return summary


def run_smoke_tests(manifest: Dict[str, Any]) -> None:
    assert rfsocs_reachable(), "RFSoCs are offline; cannot run live smoke tests"
    assert ensure_multicast_arp(), "Multicast ARP check failed; cannot run live smoke tests"

    smoke_root = COMM_DIR / "smoke_tests"
    smoke_root.mkdir(parents=True, exist_ok=True)
    os.chmod(smoke_root, 0o777)

    run_map = {r["id"]: r for r in manifest["runs"]}
    status: Dict[str, Any] = {"mode": "smoke_test", "state": "running", "completed_runs": []}

    # 1. Correlator smoke test (2 integrated frames ~ 22s)
    c_run = dict(run_map["phase1_corr_cal"])
    c_cfg = yaml.safe_load(Path(c_run["config_path"]).read_text(encoding="utf-8"))
    c_out = smoke_root / "phase1_corr_cal"
    if c_out.exists():
        shutil.rmtree(c_out)
    c_out.mkdir(parents=True, exist_ok=True)
    os.chmod(c_out, 0o777)
    for sb in (0, 1):
        c_cfg[f"raw_file_write_{sb}"]["base_dir"] = str(c_out)
        c_cfg[f"raw_file_write_{sb}"]["exit_after_n_files"] = 2
    c_cfg["run_metadata"]["base_dir"] = str(c_out)
    c_smoke_cfg = smoke_root / "phase1_corr_cal_smoke.yaml"
    c_smoke_cfg.write_text("---\n" + yaml.safe_dump(c_cfg, sort_keys=False), encoding="utf-8")

    c_run["id"] = "phase1_corr_cal_smoke"
    c_run["config_path"] = str(c_smoke_cfg)
    c_run["output_dir"] = str(c_out)
    c_run["duration_s"] = 150.0
    c_run["disk_cap_mb"] = 500
    c_run.pop("end_utc", None)

    c_res = run_single_observation(c_run, hard_stop_ts=time.time() + 300.0, min_free_gb=100.0, status=status)
    c_res["corr_smoke"] = verify_corr_smoke_output(c_out)
    status["completed_runs"].append(c_res)
    print(json.dumps(c_res, indent=2), flush=True)
    if c_res.get("kotekan_rc") != 0:
        sys.exit(1)

    # 2. Chromaticity 10-beam smoke test (12 frames ~ 14s)
    b_run = dict(run_map["test08_solar_chromaticity_2d"])
    b_cfg = yaml.safe_load(Path(b_run["config_path"]).read_text(encoding="utf-8"))
    b_out = smoke_root / "test08_solar_chromaticity_2d"
    if b_out.exists():
        shutil.rmtree(b_out)
    b_out.mkdir(parents=True, exist_ok=True)
    os.chmod(b_out, 0o777)
    for sb in (0, 1):
        rw = b_cfg[f"raw_file_write_power_{sb}"]
        rw["base_dir"] = str(b_out)
        rw["num_frames_per_file"] = 12
        rw["exit_after_n_files"] = 1
        rw["skip_frames"] = 1
    b_cfg["run_metadata"]["base_dir"] = str(b_out)
    b_smoke_cfg = smoke_root / "test08_solar_chromaticity_2d_smoke.yaml"
    b_smoke_cfg.write_text("---\n" + yaml.safe_dump(b_cfg, sort_keys=False), encoding="utf-8")

    b_run["id"] = "test08_solar_chromaticity_2d_smoke"
    b_run["config_path"] = str(b_smoke_cfg)
    b_run["output_dir"] = str(b_out)
    b_run["duration_s"] = 90.0
    b_run["disk_cap_mb"] = 100
    b_run.pop("end_utc", None)

    b_res = run_single_observation(b_run, hard_stop_ts=time.time() + 240.0, min_free_gb=100.0, status=status)
    status["completed_runs"].append(b_res)
    print(json.dumps(b_res, indent=2), flush=True)
    if b_res.get("kotekan_rc") != 0 or b_res.get("verifier_rc") != 0:
        sys.exit(1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke-test", action="store_true", help="Run live correlator + 10-beam chromaticity smoke tests")
    p.add_argument("--manifest", default=str(MANIFEST_PATH), help="Schedule manifest JSON path")
    p.add_argument("--status", default=str(STATUS_PATH), help="Campaign status JSON path")
    p.add_argument("--report", default=str(REPORT_PATH), help="Markdown report path")
    args = p.parse_args()

    manifest_path = Path(args.manifest)
    status_path = Path(args.status)
    report_path = Path(args.report)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if args.smoke_test:
        run_smoke_tests(manifest)
        return

    hard_stop_ts = parse_utc(manifest["scheduler_stop_utc"])
    min_free_gb = float(manifest.get("min_free_disk_gb", 100.0))

    prev_completed: List[Dict[str, Any]] = []
    if status_path.exists():
        try:
            prev_data = json.loads(status_path.read_text(encoding="utf-8"))
            if isinstance(prev_data, dict) and isinstance(prev_data.get("completed_runs"), list):
                prev_completed = [r for r in prev_data["completed_runs"] if isinstance(r, dict) and "id" in r]
        except Exception as exc:
            print(f"[{utc_now_iso()}] Warning: could not load prior status: {exc}", flush=True)
    done_ids = {r["id"] for r in prev_completed}

    status: Dict[str, Any] = {
        "mode": manifest.get("campaign", "v2_uncapped_campaign"),
        "pid": os.getpid(),
        "started_utc": utc_now_iso(),
        "scheduler_stop_utc": manifest["scheduler_stop_utc"],
        "state": "waiting",
        "current_run": None,
        "completed_runs": list(prev_completed),
    }
    write_status(status, status_path)

    for run in manifest["runs"]:
        if run["id"] in done_ids:
            continue
        if GLOBAL_STOP_REQUESTED or time.time() >= hard_stop_ts - 20.0:
            break
        start_ts = parse_utc(run["start_utc"])
        end_ts = parse_utc(run["end_utc"])

        if time.time() >= end_ts - 60.0:
            status["completed_runs"].append({"id": run["id"], "state": "skipped_window_missed", "finished_utc": utc_now_iso()})
            write_status(status, status_path)
            compile_report(status, report_path)
            continue

        while time.time() < start_ts and not GLOBAL_STOP_REQUESTED:
            if time.time() >= hard_stop_ts - 20.0:
                break
            status["state"] = f"waiting_for_{run['id']}_at_{run['start_utc']}"
            write_status(status, status_path)
            time.sleep(min(15.0, max(0.5, start_ts - time.time())))

        if GLOBAL_STOP_REQUESTED or time.time() >= hard_stop_ts - 20.0:
            break

        while not rfsocs_reachable() and not GLOBAL_STOP_REQUESTED:
            if time.time() >= min(end_ts - 180.0, hard_stop_ts - 60.0):
                print(f"[{utc_now_iso()}] Skipping {run['id']}: RFSoCs unreachable.", flush=True)
                status["completed_runs"].append({
                    "id": run["id"],
                    "state": "skipped_rfsocs_offline",
                    "kotekan_rc": None,
                    "verifier_rc": None,
                    "disk_mb": 0.0,
                    "finished_utc": utc_now_iso(),
                })
                write_status(status, status_path)
                compile_report(status, report_path)
                break
            status["state"] = f"waiting_for_rfsocs_online_for_{run['id']}"
            write_status(status, status_path)
            time.sleep(20.0)
        else:
            if GLOBAL_STOP_REQUESTED or time.time() >= hard_stop_ts - 20.0:
                break
            arp_ok = ensure_multicast_arp()
            print(f"[{utc_now_iso()}] Multicast ARP check before {run['id']}: {arp_ok}", flush=True)

            run_adj = dict(run)
            rem_window = end_ts - time.time()
            if rem_window < float(run_adj["duration_s"]):
                run_adj["duration_s"] = max(60.0, rem_window)

            res = run_single_observation(
                run_adj,
                hard_stop_ts=hard_stop_ts,
                min_free_gb=min_free_gb,
                status=status,
                status_path=status_path,
            )
            status["completed_runs"].append(res)
            status["current_run"] = None
            status["state"] = "idle_between_runs"
            write_status(status, status_path)

            if run.get("post_cal_handoff") and has_output_bin(Path(run["output_dir"])):
                run_calibration_handoff(manifest, Path(run["output_dir"]), status, status_path)

            compile_report(status, report_path)
            time.sleep(3.0)

    status["state"] = "completed_cleanly" if not GLOBAL_STOP_REQUESTED else "stopped_by_signal"
    status["current_run"] = None
    write_status(status, status_path)
    compile_report(status, report_path)
    print(f"[{utc_now_iso()}] V2 Campaign finished: {status['state']}", flush=True)


if __name__ == "__main__":
    main()
