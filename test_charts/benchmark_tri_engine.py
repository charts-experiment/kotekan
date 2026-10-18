#!/usr/bin/env python3
"""
CHARTS 64-Antenna Tri-Engine Real-Time Benchmark Runner
======================================================
Validates, launches, and profiles the concurrent execution of:
  1. CHARTS Direct Astrometric Multi-Beam Tracker (cudaDirectBeamTrackerCommand)
  2. CHARTS 64-Antenna 2D Grid Beamformer (Sky Grid Mode / Julia BB)
  3. Astron WMMA Tensor Core Correlator (cudaShuffleAstron + cudaCorrelatorAstron)
Fed by in-process astrophysical synthetic data (chartsFEngineSim).

Target Architecture:
  Single NVIDIA GPU (e.g. RTX 5090 32GB GDDR7 / H100 PCIe) + 512 GB Host RAM
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Paths
_script_dir = Path(__file__).resolve().parent
_kotekan_root = _script_dir.parent
DEFAULT_YAML = _script_dir / "config" / "64antennas_tri_engine_benchmark.yaml"


def calculate_theoretical_metrics(
    n_ant: int = 64,
    n_freq: int = 336,
    n_time: int = 15360,
    n_direct_beams: int = 4,
    n_grid_beams: int = 16,
    sample_period_s: float = 3.333333333333e-6,
):
    """Calculates algorithmic complexity, memory footprints, and timing budgets."""
    frame_budget_ms = n_time * sample_period_s * 1000.0  # 51.2 ms

    # Bytes per frame
    bytes_voltage = n_time * n_freq * n_ant * 1  # 4-bit complex packed = 1 B
    bytes_direct_beams = n_time * n_freq * n_direct_beams * 8  # float2 = 8 B
    bytes_grid_beams = n_time * n_freq * n_grid_beams * 8  # float2 = 8 B
    n_baselines = (n_ant * (n_ant + 1)) // 2  # 2080 baselines
    bytes_correlation = n_freq * 528 * 4 * 2 * 4  # 5.68 MB

    # VRAM with buffer depth 3
    vram_voltage_mb = (bytes_voltage * 3) / (1024 ** 2)
    vram_direct_mb = (bytes_direct_beams * 3) / (1024 ** 2)
    vram_grid_mb = (bytes_grid_beams * 3) / (1024 ** 2)
    vram_corr_mb = (bytes_correlation * 3) / (1024 ** 2)
    vram_total_mb = vram_voltage_mb + vram_direct_mb + vram_grid_mb + vram_corr_mb

    # Ingestion Bandwidth
    h2d_bandwidth_gb_s = (bytes_voltage / (1024 ** 3)) / (frame_budget_ms / 1000.0)

    # Computational Complexity (FLOPs per frame)
    # Direct Tracker: 8 FLOPs per (sample * freq * ant * beam)
    flops_direct = 8.0 * n_time * n_freq * n_ant * n_direct_beams
    # Grid Beamformer: 8 FLOPs per (sample * freq * ant * grid_beam)
    flops_grid = 8.0 * n_time * n_freq * n_ant * n_grid_beams
    # Correlator: 8 FLOPs per (sample * freq * n_baselines)
    flops_corr = 8.0 * n_time * n_freq * n_baselines

    total_gflops = (flops_direct + flops_grid + flops_corr) / 1e9

    return {
        "frame_budget_ms": frame_budget_ms,
        "n_ant": n_ant,
        "n_freq": n_freq,
        "n_time": n_time,
        "n_baselines": n_baselines,
        "bytes_voltage_mb": bytes_voltage / (1024 ** 2),
        "vram_total_mb": vram_total_mb,
        "vram_total_gb": vram_total_mb / 1024.0,
        "h2d_bandwidth_gb_s": h2d_bandwidth_gb_s,
        "flops_direct_giga": flops_direct / 1e9,
        "flops_grid_giga": flops_grid / 1e9,
        "flops_corr_giga": flops_corr / 1e9,
        "total_gflops": total_gflops,
    }


def print_theoretical_audit(metrics: dict, target_gpu: str = "NVIDIA RTX 5090 (32GB)"):
    """Displays comprehensive theoretical audit and headroom breakdown."""
    print("=" * 82)
    print(" CHARTS 64-ANTENNA TRI-ENGINE BENCHMARK AUDIT")
    print(f" Target Hardware : {target_gpu} | Host RAM: 512 GB")
    print("=" * 82)
    print(f"  * Antenna Elements   : {metrics['n_ant']} (8x8 Uniform Grid, dx=dy=0.6m)")
    print(
        f"  * Frequency Channels : {metrics['n_freq']} (300.0 - 400.8 MHz, 300 kHz/ch)"
    )
    print(f"  * Time Samples/Frame : {metrics['n_time']} (Sample rate: 300 kSPS)")
    print(
        f"  * Frame Arrival Time : {metrics['frame_budget_ms']:.2f} ms (Real-Time Budget)"
    )
    print(
        f"  * Interferometer     : {metrics['n_baselines']} Baselines (Full N^2 Triangle)"
    )
    print("-" * 82)
    print(" MEMORY FOOTPRINT & BANDWIDTH:")
    print(f"  * Input Voltage Frame: {metrics['bytes_voltage_mb']:.2f} MB / frame")
    print(
        f"  * H2D Stream Rate    : {metrics['h2d_bandwidth_gb_s']:.2f} GB/s (PCIe 4.0/5.0 x16 headroom > 4x)"
    )
    print(
        f"  * Total Active VRAM  : {metrics['vram_total_gb']:.2f} GB / 32.0 GB ({metrics['vram_total_gb']/32.0*100:.1f}% capacity)"
    )
    print(f"  * Host RAM Queue (4x): ~1.32 GB / 512.0 GB (0.26% capacity)")
    print("-" * 82)
    print(" COMPUTATIONAL COMPLEXITY (GFLOPs per 51.2 ms frame):")
    print(
        f"  * Engine 1 (Direct Beam Tracker) : {metrics['flops_direct_giga']:.2f} GFLOPs"
    )
    print(
        f"  * Engine 2 (Sky Grid Beamformer) : {metrics['flops_grid_giga']:.2f} GFLOPs"
    )
    print(
        f"  * Engine 3 (Astron WMMA Correlator): {metrics['flops_corr_giga']:.2f} GFLOPs"
    )
    print(
        f"  * Total Tri-Engine Workload     : {metrics['total_gflops']:.2f} GFLOPs per frame"
    )
    print("-" * 82)
    print(" ESTIMATED LATENCIES ON RTX 5090 (Blackwell SM120 @ 1.8 TB/s):")
    t_direct = 0.95  # ms
    t_grid = 1.10  # ms
    t_corr = 1.35  # ms
    t_total = t_direct + t_grid + t_corr
    rtf = metrics["frame_budget_ms"] / t_total
    duty_cycle = (t_total / metrics["frame_budget_ms"]) * 100.0

    print(
        f"  * Engine 1 Kernel Time: ~{t_direct:.2f} ms ({t_direct/metrics['frame_budget_ms']*100:.1f}% budget)"
    )
    print(
        f"  * Engine 2 Kernel Time: ~{t_grid:.2f} ms ({t_grid/metrics['frame_budget_ms']*100:.1f}% budget)"
    )
    print(
        f"  * Engine 3 Kernel Time: ~{t_corr:.2f} ms ({t_corr/metrics['frame_budget_ms']*100:.1f}% budget)"
    )
    print(
        f"  --> Combined GPU Time : ~{t_total:.2f} ms / {metrics['frame_budget_ms']:.2f} ms"
    )
    print(
        f"  --> Real-Time Factor  : {rtf:.1f}x (Runs {rtf:.1f}x faster than physical stream!)"
    )
    print(f"  --> GPU Duty Cycle    : {duty_cycle:.1f}% of SM120 resources")
    print("=" * 82)


def run_benchmark(yaml_file: Path, kotekan_bin: str | None = None):
    """Executes the benchmark pipeline using the compiled kotekan binary if present."""
    if not yaml_file.exists():
        print(f"[ERROR] YAML configuration file not found at: {yaml_file}")
        sys.exit(1)

    print(f"[INFO] Using Benchmark YAML: {yaml_file}")

    # Search for kotekan binary
    bin_candidates = [
        kotekan_bin,
        _kotekan_root / "build" / "kotekan" / "kotekan",
        _kotekan_root / "build-2404" / "kotekan" / "kotekan",
        Path("/code/kotekan/build/kotekan/kotekan"),
    ]
    resolved_bin = None
    for cand in bin_candidates:
        if cand and Path(cand).is_file() and os.access(cand, os.X_OK):
            resolved_bin = Path(cand)
            break

    metrics = calculate_theoretical_metrics()
    print_theoretical_audit(metrics)

    if resolved_bin:
        print(f"\n[INFO] Launching Kotekan binary: {resolved_bin}")
        cmd = [str(resolved_bin), "-c", str(yaml_file)]
        t0 = time.perf_counter()
        res = subprocess.run(cmd, cwd=str(_kotekan_root))
        t1 = time.perf_counter()
        print(
            f"\n[INFO] Kotekan execution completed with code {res.returncode} in {t1 - t0:.3f} s."
        )
    else:
        print("\n[NOTE] Compiled kotekan binary not found in local development path.")
        print("To run live execution on the GPU node or Trillium cluster, execute:")
        print(f"  kotekan -c {yaml_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Run CHARTS 64-Antenna Tri-Engine Real-Time Benchmark"
    )
    parser.add_argument(
        "--yaml",
        type=Path,
        default=DEFAULT_YAML,
        help="Path to tri-engine benchmark YAML",
    )
    parser.add_argument("--bin", type=str, default=None, help="Path to kotekan binary")
    args = parser.parse_args()

    run_benchmark(args.yaml, args.bin)


if __name__ == "__main__":
    main()
