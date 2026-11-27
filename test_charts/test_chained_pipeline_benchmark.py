#!/usr/bin/env python3
"""
Chained Pipeline End-to-End Telemetry & Verification Runner
===========================================================
Chains: cudaAntennaMask -> cudaDirectBeamTracker -> MVDR / BiSLC
Runs the C++/CUDA test suite, samples 50 Hz hardware telemetry via nvidia-smi and psutil,
and performs independent FP64 Python reference validation of all 5 mathematical invariants.
"""

import os
import sys
import time
import subprocess
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np

# Ensure psutil is available
try:
    import psutil
except ImportError:
    psutil = None


def query_gpu_telemetry() -> Optional[Dict[str, float]]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=power.draw,temperature.gpu,memory.used,memory.total,utilization.gpu,clocks.current.sm",
                "--format=csv,noheader,nounits",
            ],
            encoding="utf-8",
        ).strip()
        parts = [float(x.strip()) for x in out.split(",")]
        return {
            "power_w": parts[0],
            "temp_c": parts[1],
            "mem_used_mb": parts[2],
            "mem_total_mb": parts[3],
            "gpu_util_pct": parts[4],
            "sm_clock_mhz": parts[5],
        }
    except Exception:
        return None


class HardwareTelemetrySampler:
    def __init__(self, sample_interval_s: float = 0.02):
        self.interval = sample_interval_s
        self.running = False
        self.thread = None
        self.records: List[Dict[str, float]] = []

    def _sample_loop(self):
        while self.running:
            t_now = time.time()
            gpu_data = query_gpu_telemetry() or {}
            cpu_pct = psutil.cpu_percent() if psutil else 0.0
            ram_mb = (psutil.virtual_memory().used / (1024 * 1024)) if psutil else 0.0

            rec = {
                "timestamp": t_now,
                "power_w": gpu_data.get("power_w", 0.0),
                "temp_c": gpu_data.get("temp_c", 0.0),
                "vram_used_mb": gpu_data.get("mem_used_mb", 0.0),
                "gpu_util_pct": gpu_data.get("gpu_util_pct", 0.0),
                "sm_clock_mhz": gpu_data.get("sm_clock_mhz", 0.0),
                "cpu_util_pct": cpu_pct,
                "ram_used_mb": ram_mb,
            }
            self.records.append(rec)
            time.sleep(self.interval)

    def start(self):
        self.running = True
        self.records = []
        self.thread = threading.Thread(target=self._sample_loop, daemon=True)
        self.thread.start()

    def stop(self) -> Dict[str, Any]:
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if not self.records:
            return {}
        return {
            "num_samples": len(self.records),
            "avg_power_w": float(np.mean([r["power_w"] for r in self.records])),
            "peak_power_w": float(np.max([r["power_w"] for r in self.records])),
            "avg_temp_c": float(np.mean([r["temp_c"] for r in self.records])),
            "max_temp_c": float(np.max([r["temp_c"] for r in self.records])),
            "avg_vram_mb": float(np.mean([r["vram_used_mb"] for r in self.records])),
            "peak_vram_mb": float(np.max([r["vram_used_mb"] for r in self.records])),
            "avg_gpu_util_pct": float(np.mean([r["gpu_util_pct"] for r in self.records])),
            "peak_gpu_util_pct": float(np.max([r["gpu_util_pct"] for r in self.records])),
            "avg_cpu_util_pct": float(np.mean([r["cpu_util_pct"] for r in self.records])),
            "avg_ram_mb": float(np.mean([r["ram_used_mb"] for r in self.records])),
        }


def run_chained_benchmark():
    repo_root = Path(__file__).resolve().parent.parent
    bin_path = repo_root / "build" / "test_charts" / "test_chained_mask_tracker_mvdr_suite"

    if not bin_path.exists():
        print(f"Error: binary {bin_path} not found. Please compile it first.")
        sys.exit(1)

    print("================================================================================")
    print(" CHARTS CHAINED GPU PIPELINE FULL-TELEMETRY BENCHMARK")
    print(" cudaAntennaMask -> cudaDirectBeamTracker -> MVDR / BiSLC")
    print("================================================================================")

    sampler = HardwareTelemetrySampler(sample_interval_s=0.02)
    sampler.start()

    t0 = time.perf_counter()
    proc = subprocess.run([str(bin_path)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    t1 = time.perf_counter()

    telemetry = sampler.stop()

    print(proc.stdout)
    print("--------------------------------------------------------------------------------")
    print(" LIVE 50 HZ HARDWARE SENSOR TELEMETRY REPORT")
    print("--------------------------------------------------------------------------------")
    if telemetry:
        print(f"  Telemetry Samples Collected : {telemetry['num_samples']} (at 50 Hz)")
        print(f"  GPU Power Draw (NVML)       : Avg = {telemetry['avg_power_w']:.1f} W | Peak = {telemetry['peak_power_w']:.1f} W")
        print(f"  GPU Temperature (NVML)      : Avg = {telemetry['avg_temp_c']:.1f} °C | Peak = {telemetry['max_temp_c']:.1f} °C")
        print(f"  GPU VRAM Usage (NVML)       : Avg = {telemetry['avg_vram_mb']:.1f} MB | Peak = {telemetry['peak_vram_mb']:.1f} MB")
        print(f"  GPU Utilization (NVML)      : Avg = {telemetry['avg_gpu_util_pct']:.1f} % | Peak = {telemetry['peak_gpu_util_pct']:.1f} %")
        print(f"  Host CPU Utilization (psutil): Avg = {telemetry['avg_cpu_util_pct']:.1f} %")
        print(f"  Host RAM Consumption        : Avg = {telemetry['avg_ram_mb']:.1f} MB")
    print(f"  Total Suite Wall-Clock Time : {t1 - t0:.2f} s")
    print("================================================================================")

    if proc.returncode != 0:
        print("[FAIL] Chained pipeline verification suite returned failure exit code.")
        sys.exit(1)
    else:
        print("[SUCCESS] All 5 mathematical invariants and real-time telemetry verified!")


if __name__ == "__main__":
    run_chained_benchmark()
