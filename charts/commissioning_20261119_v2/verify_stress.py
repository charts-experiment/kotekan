#!/usr/bin/env python3
"""Post-test verification for Test 10: 10-Beam Production-Stack + RFI Stress Test.

Verifies:
1. GPU stage & per-command latency vs 51.2 ms real-time budget.
2. Ring-buffer high-water occupancy across all Kotekan buffers.
3. DPDK packet loss / sequence gaps / CLEAN_EXIT from kotekan.log & beam_power headers.
4. Thermal, VRAM, and disk safety (confirming zero baseband/trigger files written).
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from beam_io import atomic_savefig, atomic_write_text, load_dual_subband_beam_power, load_jsonl

BUDGET_MS = 51.2


def parse_kotekan_log(log_path: Path) -> Dict[str, Any]:
    if not log_path.exists():
        return {"exists": False}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    rx_matches = re.findall(r"RFSoC port (\d+) \| RX pkt (\d+) .*? \| lost (\d+) .*? \| Samples (\d+)", text)
    dpdk_matches = re.findall(
        r"DPDK port (\d+) stats \| ipackets (\d+) .*? \| imissed (\d+) .*? \| ierrors (\d+) .*? \| rx_nombuf (\d+)",
        text,
    )
    clean_exit = "kotekan shutdown with status: CLEAN_EXIT" in text
    fatal_lines = [ln for ln in text.splitlines() if "FATAL" in ln or "ERROR" in ln]
    ports = {}
    for p_id, pkt, lost, samp in rx_matches:
        ports[f"rfsoc_port_{p_id}"] = {"rx_pkt": int(pkt), "lost": int(lost), "samples": int(samp)}
    for p_id, ipkt, imiss, ierr, nombuf in dpdk_matches:
        ports[f"dpdk_port_{p_id}"] = {
            "ipackets": int(ipkt),
            "imissed": int(imiss),
            "ierrors": int(ierr),
            "rx_nombuf": int(nombuf),
        }
    return {
        "exists": True,
        "clean_exit": clean_exit,
        "error_count": len(fatal_lines),
        "error_tail": fatal_lines[-10:],
        "final_port_stats": ports,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    telem = load_jsonl(run_dir / "telemetry_log.jsonl")
    log_info = parse_kotekan_log(run_dir / "kotekan.log")

    timing = {}
    if any(run_dir.glob("beam_power_freq0_*.bin")):
        wf, seq, t0_us, timing = load_dual_subband_beam_power(run_dir, n_beams=10)
    else:
        wf = np.empty((0, 10, 672), dtype=np.float32)

    total_bytes = 0
    forbidden_files: List[str] = []
    for root, _, files in os.walk(run_dir):
        for fn in files:
            fp = Path(root) / fn
            try:
                total_bytes += fp.stat().st_size
            except OSError:
                pass
            low = fn.lower()
            if low.endswith(".h5") or "baseband" in low or "trigger" in low:
                forbidden_files.append(str(fp))

    gpu0_ms, gpu1_ms, gpu0_work_ms, gpu1_work_ms, ts_min = [], [], [], [], []
    cmd_latency_samples: Dict[str, List[float]] = {}
    buf_max_occ: Dict[str, float] = {}
    gpu_temps, gpu_vram, gpu_power, gpu_util = [], [], [], []
    rfi_pct = []
    rfi_ch_counts = []

    t0_ref = telem[0]["unix"] if telem else 0.0
    for row in telem:
        gp = row.get("gpu_profile", {})
        g0_tot = float(gp.get("gpu_0", {}).get("total_ms", 0.0) or 0.0)
        g1_tot = float(gp.get("gpu_1", {}).get("total_ms", 0.0) or 0.0)
        if g0_tot > 0.0 or g1_tot > 0.0:
            ts_min.append((row["unix"] - t0_ref) / 60.0)
            if g0_tot > 0.0:
                gpu0_ms.append(g0_tot)
                cmds0 = gp["gpu_0"].get("commands_ms", {})
                gpu0_work_ms.append(sum(float(v) for k, v in cmds0.items() if "sync" not in k.lower()))
                for cname, cval in cmds0.items():
                    cmd_latency_samples.setdefault(f"gpu0/{cname}", []).append(float(cval))
            if g1_tot > 0.0:
                gpu1_ms.append(g1_tot)
                cmds1 = gp["gpu_1"].get("commands_ms", {})
                gpu1_work_ms.append(sum(float(v) for k, v in cmds1.items() if "sync" not in k.lower()))
                for cname, cval in cmds1.items():
                    cmd_latency_samples.setdefault(f"gpu1/{cname}", []).append(float(cval))

        bufs = row.get("buffers", {})
        if isinstance(bufs, dict) and "error" not in bufs:
            for bname, binfo in bufs.items():
                if isinstance(binfo, dict) and "occupancy_pct" in binfo:
                    occ = float(binfo["occupancy_pct"])
                    if occ > buf_max_occ.get(bname, 0.0):
                        buf_max_occ[bname] = occ

        smi = row.get("nvidia_smi", {})
        if isinstance(smi, dict) and "temp_c" in smi and (g0_tot > 0.0 or g1_tot > 0.0):
            gpu_temps.append(float(smi["temp_c"]))
            gpu_vram.append(float(smi["vram_used_mib"]))
            gpu_power.append(float(smi["power_w"]))
            gpu_util.append(float(smi["gpu_util_pct"]))

        rfi = row.get("rfi_status", {})
        if isinstance(rfi, dict) and (g0_tot > 0.0 or g1_tot > 0.0):
            if "overall_flagged_chunk_fraction" in rfi:
                rfi_pct.append(100.0 * float(rfi["overall_flagged_chunk_fraction"]))
            if "flagged_channels" in rfi:
                rfi_ch_counts.append(float(rfi["flagged_channels"]))

    def _stats(vals: List[float]) -> Dict[str, Any]:
        if not vals:
            return {}
        a = np.asarray(vals, dtype=np.float64)
        return {
            "mean": round(float(np.mean(a)), 3),
            "p95": round(float(np.percentile(a, 95)), 3),
            "p99": round(float(np.percentile(a, 99)), 3),
            "max": round(float(np.max(a)), 3),
        }

    cmd_stats = {k: _stats(v) for k, v in sorted(cmd_latency_samples.items())}

    summary = {
        "run_dir": str(run_dir),
        "realtime_budget_ms": BUDGET_MS,
        "telemetry_samples": len(telem),
        "disk_usage_mb": round(total_bytes / (1024.0 ** 2), 3),
        "forbidden_baseband_or_trigger_files": forbidden_files,
        "timing": timing,
        "kotekan_log": log_info,
        "gpu_0_total_ms": _stats(gpu0_ms),
        "gpu_1_total_ms": _stats(gpu1_ms),
        "gpu_0_active_work_ms": _stats(gpu0_work_ms),
        "gpu_1_active_work_ms": _stats(gpu1_work_ms),
        "per_command_ms": cmd_stats,
        "buffer_max_occupancy_pct": {k: round(v, 1) for k, v in sorted(buf_max_occ.items())},
        "gpu_health": {
            "temp_c": _stats(gpu_temps),
            "vram_used_mib": _stats(gpu_vram),
            "power_w": _stats(gpu_power),
            "gpu_util_pct": _stats(gpu_util),
        },
        "rfi_flagged_chunk_pct": _stats(rfi_pct),
        "rfi_flagged_channels": _stats(rfi_ch_counts),
        "mean_power_per_beam": [round(float(np.mean(wf[:, b, :])), 4) for b in range(10)] if wf.shape[0] > 0 else [],
    }
    atomic_write_text(run_dir / "stress_verification.json", json.dumps(summary, indent=2))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    ax00 = axes[0, 0]
    if gpu0_ms:
        ax00.plot(ts_min[: len(gpu0_ms)], gpu0_ms, label="GPU 0 Total (ms)", color="#2563eb")
    if gpu1_ms:
        ax00.plot(ts_min[: len(gpu1_ms)], gpu1_ms, label="GPU 1 Total (ms)", color="#16a34a")
    ax00.axhline(BUDGET_MS, color="#dc2626", ls="--", label="51.2 ms Budget")
    ax00.set_title("10-Beam GPU Pipeline Latency vs Real-Time Budget")
    ax00.set_xlabel("Elapsed Time (min)")
    ax00.set_ylabel("Latency (ms)")
    ax00.grid(True, alpha=0.25)
    ax00.legend(fontsize=8.5)

    ax01 = axes[0, 1]
    if buf_max_occ:
        names = list(sorted(buf_max_occ.keys()))
        vals = [buf_max_occ[n] for n in names]
        ax01.barh(names, vals, color="#0284c7")
        ax01.set_xlim(0, 105)
        ax01.set_title("Ring-Buffer High-Water Occupancy (%)")
        ax01.set_xlabel("Max Occupancy (%)")
        ax01.grid(True, axis="x", alpha=0.25)

    ax10 = axes[1, 0]
    if gpu_temps and gpu_power:
        ax10.plot(ts_min[: len(gpu_temps)], gpu_temps, color="#dc2626", label="GPU Temp (C)")
        ax10_t = ax10.twinx()
        ax10_t.plot(ts_min[: len(gpu_power)], gpu_power, color="#7c3aed", alpha=0.75, label="GPU Power (W)")
        ax10_t.set_ylabel("Power (W)")
        ax10.set_title("RTX 5090 Thermal & Power Stability")
        ax10.set_xlabel("Elapsed Time (min)")
        ax10.set_ylabel("Temp (C)")
        ax10.grid(True, alpha=0.25)

    ax11 = axes[1, 1]
    if cmd_stats:
        cnames = [k for k in cmd_stats if "sync" not in k.lower()]
        cmeans = [cmd_stats[k]["mean"] for k in cnames]
        ax11.barh(cnames, cmeans, color="#f59e0b")
        ax11.set_title("Mean Per-Command GPU Execution Time (ms)")
        ax11.set_xlabel("Time (ms)")
        ax11.grid(True, axis="x", alpha=0.25)

    atomic_savefig(fig, run_dir / "stress_latency_buffers.png", dpi=140)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
