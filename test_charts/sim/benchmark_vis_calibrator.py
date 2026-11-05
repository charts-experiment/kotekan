#!/usr/bin/env python3
"""GPU performance of the correlator chain with and without cudaVisibilityCalibrator.

Replays one simulated baseband frame many times through the real Kotekan binary
(``rawFileRead -> cudaProcess -> restInspectFrame``) and measures two ways:

  * Kotekan's own trackers (``GET /trackers`` on the REST server, enabled by
    ``gpu.profiling``): per-command CUDA-event execution time, and the share of
    the 5.12 ms frame period it uses (the ``_u`` tracker). Kotekan keeps only the
    last 100 samples per tracker, so the endpoint is polled and samples are
    merged by timestamp.
  * ``nvidia-smi`` sampled at 100 ms: GPU utilization, power, SM clock, memory
    and temperature while the replay runs, next to an idle reference.

Variants alternate over several repetitions so that thermal drift affects both.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import tempfile
import threading
import time
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from .corr_cal import create_replay_yaml, kotekan_env, num_blocks_for
from .presets import find_kotekan_binary

REST_URL = "http://127.0.0.1:12048"
SMI_FIELDS = ["utilization.gpu", "power.draw", "clocks.sm", "memory.used", "temperature.gpu"]
SMI_KEYS = ["gpu_util_pct", "power_w", "sm_clock_mhz", "memory_used_mib", "temp_c"]
WARMUP_FRACTION = 0.1
# Commands whose event span is host waiting or a PCIe copy rather than kernel work
NON_COMPUTE = ("Sync", "InputData", "OutputData")


def _get_json(endpoint: str, timeout: float = 5.0) -> Optional[Dict[str, Any]]:
    try:
        with urllib.request.urlopen(f"{REST_URL}/{endpoint}", timeout=timeout) as reply:
            return json.loads(reply.read().decode())
    except (OSError, ValueError):
        return None


class SmiSampler:
    """Background ``nvidia-smi -lms`` sampler that timestamps each reading."""

    def __init__(self, interval_ms: int = 100, gpu_id: int = 0):
        self._cmd = ["nvidia-smi", f"--id={gpu_id}", f"--query-gpu={','.join(SMI_FIELDS)}",
                     "--format=csv,noheader,nounits", f"-lms", str(interval_ms)]
        self.samples: List[Tuple[float, Dict[str, float]]] = []
        self._proc: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "SmiSampler":
        self._proc = subprocess.Popen(self._cmd, stdout=subprocess.PIPE, text=True)
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()
        return self

    def _read(self) -> None:
        assert self._proc and self._proc.stdout
        for line in self._proc.stdout:
            parts = [p.strip() for p in line.split(",")]
            try:
                values = {k: float(v) for k, v in zip(SMI_KEYS, parts)}
            except ValueError:  # "[N/A]" while the driver warms up
                continue
            self.samples.append((time.time(), values))

    def stop(self) -> None:
        if self._proc:
            self._proc.terminate()
            self._proc.wait(timeout=10)
        if self._thread:
            self._thread.join(timeout=5)

    def summarize(self, t0: float = 0.0, t1: float = float("inf")) -> Dict[str, float]:
        window = [v for t, v in self.samples if t0 <= t <= t1]
        out: Dict[str, float] = {"n_samples": float(len(window))}
        for key in SMI_KEYS:
            data = [v[key] for v in window]
            if data:
                out[f"{key}_mean"] = statistics.fmean(data)
                out[f"{key}_max"] = max(data)
        return out


def _stats_us(values: np.ndarray) -> Dict[str, float]:
    us = values * 1e6
    return {"mean_us": float(us.mean()), "p50_us": float(np.percentile(us, 50)),
            "p99_us": float(np.percentile(us, 99)), "max_us": float(us.max())}


def _collect_trackers(proc: subprocess.Popen, expected_frames: int, timeout_s: float,
                      poll_s: float = 0.2, idle_s: float = 2.0
                      ) -> Dict[str, Dict[str, Dict[int, float]]]:
    """Poll /trackers, merging samples per (stage, tracker) keyed by timestamp.

    Returns when kotekan stops producing new samples (replay finished), exits,
    or the timeout passes.
    """
    merged: Dict[str, Dict[str, Dict[int, float]]] = {}
    last_new = time.time()
    deadline = last_new + timeout_s
    while time.time() < deadline and proc.poll() is None:
        reply = _get_json("trackers")
        added = False
        if reply:
            for stage, trackers in reply.items():
                for name, body in trackers.items():
                    store = merged.setdefault(stage, {}).setdefault(name, {})
                    for s in body.get("samples", []):
                        ts = int(s["timestamp"])
                        if ts not in store:
                            store[ts] = float(s["value"])
                            added = True
        if added:
            last_new = time.time()
        elif merged and time.time() - last_new > idle_s:
            break
        time.sleep(poll_s)
    return merged


def run_variant(
    name: str,
    calibrator: Optional[Dict[str, Any]],
    input_dir: Path,
    work_dir: Path,
    repo_root: Path,
    num_frames: int,
    kotekan_bin: Path,
    timeout_s: float,
    smi_interval_ms: int,
) -> Dict[str, Any]:
    """Run one replay and return its tracker and nvidia-smi statistics."""
    yaml_path = create_replay_yaml(
        work_dir / f"bench_{name}.yaml", input_dir, "perf", None, calibrator=calibrator,
        end_interrupt=False,
    )
    commands = yaml.safe_load(yaml_path.read_text())["gpu"]["gpu_0"]["commands"]
    frame_period_s = yaml.safe_load(yaml_path.read_text())["frame_arrival_period"]

    log = open(work_dir / f"bench_{name}.log", "w")
    proc = subprocess.Popen([str(kotekan_bin), "-c", str(yaml_path)], cwd=repo_root,
                            env=kotekan_env(repo_root), stdout=log, stderr=subprocess.STDOUT)
    smi = SmiSampler(smi_interval_ms).start()
    t_start = time.time()
    try:
        trackers = _collect_trackers(proc, num_frames, timeout_s)
    finally:
        t_end = time.time()
        smi.stop()
        if proc.poll() is None:
            proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()

    per_command: Dict[str, Dict[str, float]] = {}
    chain_means: List[float] = []
    compute_means: List[float] = []
    first_ts: List[int] = []
    last_ts: List[int] = []
    for idx, command in enumerate(commands):
        stage = trackers.get(f"/gpu/gpu_0/commands/{idx}", {})
        exec_name = next((n for n in stage if n.endswith("_execute_time")), None)
        if exec_name is None:
            continue
        series = sorted(stage[exec_name].items())
        values = np.array([v for _, v in series])
        skip = int(len(values) * WARMUP_FRACTION)
        values = values[skip:]
        if values.size == 0:
            continue
        entry = _stats_us(values)
        util_name = next((n for n in stage if n.endswith("_u")), None)
        if util_name:
            util = np.array([v for _, v in sorted(stage[util_name].items())][skip:])
            entry["frame_period_share_pct"] = float(util.mean() * 100.0)
        entry["samples"] = int(values.size)
        key = f"{idx}:{command['name']}"
        per_command[key] = entry
        chain_means.append(entry["mean_us"])
        if not any(tag in command["name"] for tag in NON_COMPUTE):
            compute_means.append(entry["mean_us"])
        if idx == 0:
            first_ts.append(series[skip][0])
            last_ts.append(series[-1][0])

    result: Dict[str, Any] = {"variant": name, "commands": per_command,
                              "kernel_time_per_frame_us": float(sum(compute_means)),
                              "all_commands_time_per_frame_us": float(sum(chain_means)),
                              "frame_period_ms": frame_period_s * 1e3}
    if first_ts and last_ts[0] > first_ts[0]:
        frames_measured = per_command[next(iter(per_command))]["samples"]
        span_s = (last_ts[0] - first_ts[0]) / 1e3
        result["pipeline_rate_hz"] = (frames_measured - 1) / span_s
        result["realtime_rate_hz"] = 1.0 / frame_period_s
        result["smi"] = smi.summarize(first_ts[0] / 1e3, last_ts[0] / 1e3)
    else:
        result["smi"] = smi.summarize(t_start, t_end)
    return result


def _average(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Average the numeric leaves of repeated runs of one variant."""
    def avg(items: List[Any]) -> Any:
        if isinstance(items[0], dict):
            keys = set().union(*(i.keys() for i in items))
            return {k: avg([i[k] for i in items if k in i]) for k in sorted(keys)}
        if isinstance(items[0], (int, float)):
            return float(statistics.fmean(items))
        return items[0]
    return avg(runs)


def run_benchmark(
    num_frames: int = 3000,
    repetitions: int = 3,
    calibration_file: Optional[Path] = None,
    work_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    kotekan_bin: Optional[Path] = None,
    timeout_s: float = 180.0,
    smi_interval_ms: int = 100,
    num_elements: int = 64,
    num_local_freq: int = 336,
) -> Dict[str, Any]:
    """Benchmark the correlator chain without and with the calibrator."""
    from .generator import generate_simulation_window
    from .presets import get_preset_config

    repo_root = Path(__file__).resolve().parent.parent.parent
    binary = kotekan_bin or find_kotekan_binary()
    if binary is None:
        raise FileNotFoundError("kotekan binary not found; build it or pass kotekan_bin")
    if shutil.which("nvidia-smi") is None:
        raise RuntimeError("nvidia-smi is required for the GPU metrics")

    own_work = work_dir is None
    work = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="charts_viscal_bench_"))
    work.mkdir(parents=True, exist_ok=True)
    try:
        cfg = get_preset_config(preset="quick", profile="day", antennas=num_elements,
                                num_freq=num_local_freq)
        cfg.duration_s = 0.3
        cfg.window_name = "bench_src"
        cfg.scratch_dir = work
        print("Generating one simulated baseband frame...")
        window = Path(generate_simulation_window(cfg)["target_dir"])
        frame = sorted(window.glob("bench_src_*.bin"))[2]

        input_dir = work / "replay_input"
        input_dir.mkdir(exist_ok=True)
        for i in range(num_frames):
            link = input_dir / f"perf_{i:07d}.bin"
            if not link.exists():
                link.symlink_to(frame)

        if calibration_file is None:
            rng = np.random.default_rng(0)
            shape = (num_local_freq, num_blocks_for(num_elements), 2, 2)
            table = (rng.uniform(0.5, 2.0, shape)
                     * np.exp(1j * rng.uniform(-np.pi, np.pi, shape))).astype(np.complex64)
            calibration_file = work / "bench_table.npy"
            np.save(calibration_file, table)

        variants = {
            "without_calibrator": None,
            "with_calibrator": {"calibration_file": str(calibration_file), "freq_offset": 0},
        }

        print("Sampling idle GPU for 3 s...")
        idle = SmiSampler(smi_interval_ms).start()
        time.sleep(3.0)
        idle.stop()

        runs: Dict[str, List[Dict[str, Any]]] = {name: [] for name in variants}
        names = list(variants)
        for rep in range(repetitions):
            for name in (names if rep % 2 == 0 else names[::-1]):
                print(f"Repetition {rep + 1}/{repetitions}: {name} ({num_frames} frames)...")
                runs[name].append(run_variant(
                    name, variants[name], input_dir, work, repo_root, num_frames, binary,
                    timeout_s, smi_interval_ms))

        report = {
            "num_frames": num_frames, "repetitions": repetitions,
            "gpu_idle": idle.summarize(),
            "variants": {name: _average(r) for name, r in runs.items()},
            "runs": runs,
        }
        base = report["variants"]["without_calibrator"]
        cal = report["variants"]["with_calibrator"]
        added = next((v for k, v in cal["commands"].items() if "VisibilityCalibrator" in k), None)
        correlator = next((v for k, v in cal["commands"].items() if "CorrelatorAstron" in k), None)
        report["overhead"] = {
            "kernel_time_per_frame_delta_us": (cal["kernel_time_per_frame_us"]
                                               - base["kernel_time_per_frame_us"]),
            "calibrator_mean_us": added["mean_us"] if added else None,
            "calibrator_pct_of_correlator": (100.0 * added["mean_us"] / correlator["mean_us"]
                                             if added and correlator else None),
            "calibrator_pct_of_frame_period": added.get("frame_period_share_pct") if added else None,
        }
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "vis_calibrator_benchmark.json").write_text(json.dumps(report, indent=2))
            (output_dir / "vis_calibrator_benchmark.md").write_text(format_markdown(report))
        return report
    finally:
        if own_work:
            shutil.rmtree(work, ignore_errors=True)


def format_markdown(report: Dict[str, Any]) -> str:
    """Render the benchmark report as markdown tables."""
    lines = [f"# cudaVisibilityCalibrator benchmark ({report['num_frames']} frames x "
             f"{report['repetitions']} repetitions)", "",
             "## Kotekan trackers (GPU time per frame, CUDA events)", "",
             "| Variant | Command | mean (us) | p50 (us) | p99 (us) | max (us) | % of frame period |",
             "|---|---|---|---|---|---|---|"]
    for name, variant in report["variants"].items():
        for command, c in variant["commands"].items():
            lines.append(f"| {name} | {command} | {c['mean_us']:.1f} | {c['p50_us']:.1f} | "
                         f"{c['p99_us']:.1f} | {c['max_us']:.1f} | "
                         f"{c.get('frame_period_share_pct', float('nan')):.2f} |")
    lines += ["", "Sync and copy commands (`cudaSync*`, `cudaInputData`, `cudaOutputData`) time host "
              "waits and PCIe transfers, so the kernel total only adds the compute commands.", "",
              "| Variant | Kernel time / frame (us) | All commands / frame (us) | Pipeline rate (Hz) | "
              "Real-time rate (Hz) |", "|---|---|---|---|---|"]
    for name, v in report["variants"].items():
        lines.append(f"| {name} | {v['kernel_time_per_frame_us']:.1f} | "
                     f"{v['all_commands_time_per_frame_us']:.1f} | "
                     f"{v.get('pipeline_rate_hz', float('nan')):.1f} | "
                     f"{v.get('realtime_rate_hz', float('nan')):.1f} |")

    lines += ["", "## nvidia-smi (100 ms samples while replaying)", "",
              "| State | GPU util mean/max (%) | Power mean/max (W) | SM clock mean (MHz) | "
              "Memory max (MiB) | Temp max (C) |", "|---|---|---|---|---|---|"]
    rows = [("idle", report["gpu_idle"])] + [(n, v["smi"]) for n, v in report["variants"].items()]
    for name, s in rows:
        lines.append(f"| {name} | {s.get('gpu_util_pct_mean', float('nan')):.0f} / "
                     f"{s.get('gpu_util_pct_max', float('nan')):.0f} | "
                     f"{s.get('power_w_mean', float('nan')):.0f} / {s.get('power_w_max', float('nan')):.0f} | "
                     f"{s.get('sm_clock_mhz_mean', float('nan')):.0f} | "
                     f"{s.get('memory_used_mib_max', float('nan')):.0f} | "
                     f"{s.get('temp_c_max', float('nan')):.0f} |")

    o = report["overhead"]
    lines += ["", "## Calibrator overhead", ""]
    if o["calibrator_mean_us"] is not None:
        lines += [f"- Kernel time: {o['calibrator_mean_us']:.1f} us/frame "
                  f"({o['calibrator_pct_of_correlator']:.2f}% of the Astron correlator, "
                  f"{o['calibrator_pct_of_frame_period']:.3f}% of the frame period)"]
    lines += [f"- Change in kernel time per frame: {o['kernel_time_per_frame_delta_us']:+.1f} us", ""]
    return "\n".join(lines)
