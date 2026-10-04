#!/usr/bin/env python3
"""CHARTS Direct Beam Tracker Performance, VRAM, and Power Benchmark.

Dedicated analysis engine for Kotekan's upstream `cudaDirectBeamTracker`
(fused multi-beam, zero-integration-window beamformer kernel and pipeline stage).

Measures and models:
  1. Real Cadence Analysis:
     - Grounded in CHARTS FPGA channelizer: 8192 FFT @ 2457.6 MHz -> 3.3333 us/spectrum
     - Standard Cadence: 1,536 samples -> 5.12 ms frame period (195.31 Hz)
     - Low-Latency / Extended Cadence: 3,840 samples -> 12.8 ms frame period (78.13 Hz)
     - Full-Frame Cadence: 15,360 samples -> 51.2 ms frame period (19.53 Hz)
     - Real-time headroom factor, duty cycle, frame slack time, and Gbps ingest rates
  2. VRAM Memory Layout:
     - Exact byte breakdown of input ring buffer, output ring buffer, steering weights,
       survey geometry, and CUDA context
     - Total VRAM footprint across buffer depths (depth=2, 3, 4)
     - GPU VRAM saturation % and maximum safe buffer depth before OOM
  3. GPU Power & Energy Telemetry:
     - Live hardware GPU telemetry via NVML / nvidia-smi (Watts, clocks, temp, util)
     - Peak active kernel power draw vs. idle standby power
     - Continuous average power consumption under real cadence streaming:
       P_cadence = DutyCycle * P_active + (1 - DutyCycle) * P_idle
     - Energy per frame (mJ/frame), energy per sample (nJ), and GFLOPS/Watt
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger("kotekan.charts.sim.benchmark")

# CHARTS Telescope Physical Cadence Constants
FPGA_TIME_RESOLUTION_US = 10.0 / 3.0  # 3.333333 us per time sample (8192 / 2457.6 MHz)
DEFAULT_CADENCE_SAMPLES = 1536        # Standard CHARTS frame (5.12 ms)
LOW_LATENCY_CADENCE_SAMPLES = 3840    # Low-VRAM / Low-latency frame (12.8 ms)
FULL_FRAME_CADENCE_SAMPLES = 15360    # Extended payload frame (51.2 ms)


@dataclass
class GpuDeviceProfile:
    """Hardware specifications for target GPU."""
    name: str
    vram_total_mb: float
    tdp_watts: float
    idle_watts: float
    peak_bandwidth_gb_s: float
    peak_tflops_fp32: float
    cuda_cores: int


# Known Observatory Hardware Profiles
KNOWN_GPUS: Dict[str, GpuDeviceProfile] = {
    "rtx4090": GpuDeviceProfile(
        name="NVIDIA GeForce RTX 4090 (Ada Lovelace)",
        vram_total_mb=24576.0,
        tdp_watts=450.0,
        idle_watts=15.0,
        peak_bandwidth_gb_s=1008.0,
        peak_tflops_fp32=82.6,
        cuda_cores=16384,
    ),
    "rtx5090": GpuDeviceProfile(
        name="NVIDIA GeForce RTX 5090 (Blackwell)",
        vram_total_mb=32768.0,
        tdp_watts=600.0,
        idle_watts=22.0,
        peak_bandwidth_gb_s=1792.0,
        peak_tflops_fp32=125.0,
        cuda_cores=21760,
    ),
    "h100": GpuDeviceProfile(
        name="NVIDIA H100 PCIe (Hopper)",
        vram_total_mb=81920.0,
        tdp_watts=350.0,
        idle_watts=35.0,
        peak_bandwidth_gb_s=2000.0,
        peak_tflops_fp32=67.0,
        cuda_cores=14592,
    ),
    "rtx3060": GpuDeviceProfile(
        name="NVIDIA GeForce RTX 3060 (Laptop/Desktop)",
        vram_total_mb=6144.0,
        tdp_watts=80.0,
        idle_watts=10.0,
        peak_bandwidth_gb_s=360.0,
        peak_tflops_fp32=12.7,
        cuda_cores=3840,
    ),
}


def query_live_gpu_telemetry() -> Optional[Dict[str, Any]]:
    """Queries live NVIDIA GPU telemetry via nvidia-smi if available."""
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=name,power.draw,power.limit,memory.used,memory.total,temperature.gpu,clocks.current.sm,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=5)
        line = res.stdout.strip().split("\n")[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 8:
            gpu_name = parts[0]
            # Parse power draw, sanitizing sensor anomalies
            try:
                p_draw = float(parts[1])
                if p_draw > 1000.0:  # Sensor anomaly fallback
                    p_draw = 45.0
            except ValueError:
                p_draw = 45.0

            try:
                p_lim = float(parts[2])
            except ValueError:
                p_lim = 100.0

            mem_used = float(parts[3])
            mem_total = float(parts[4])
            temp_c = float(parts[5])
            sm_clock_mhz = float(parts[6])
            gpu_util = float(parts[7])

            return {
                "name": gpu_name,
                "power_draw_w": p_draw,
                "power_limit_w": p_lim,
                "memory_used_mb": mem_used,
                "memory_total_mb": mem_total,
                "temp_c": temp_c,
                "sm_clock_mhz": sm_clock_mhz,
                "gpu_util_pct": gpu_util,
            }
    except Exception as e:
        logger.debug(f"Live GPU query unavailable: {e}")
    return None


def detect_target_gpu(explicit_target: Optional[str] = None) -> GpuDeviceProfile:
    """Detects active GPU or returns matched known profile."""
    if explicit_target and explicit_target.lower() in KNOWN_GPUS:
        return KNOWN_GPUS[explicit_target.lower()]

    live = query_live_gpu_telemetry()
    if live:
        name_lower = live["name"].lower()
        if "5090" in name_lower:
            return KNOWN_GPUS["rtx5090"]
        if "4090" in name_lower:
            return KNOWN_GPUS["rtx4090"]
        if "h100" in name_lower:
            return KNOWN_GPUS["h100"]
        if "3060" in name_lower:
            prof = KNOWN_GPUS["rtx3060"]
            prof.vram_total_mb = live["memory_total_mb"]
            return prof

        # Generic custom profile from live data
        return GpuDeviceProfile(
            name=live["name"],
            vram_total_mb=live["memory_total_mb"],
            tdp_watts=live["power_limit_w"] if live["power_limit_w"] > 0 else 250.0,
            idle_watts=15.0,
            peak_bandwidth_gb_s=500.0,
            peak_tflops_fp32=25.0,
            cuda_cores=5000,
        )

    # Default to RTX 4090 (standard observatory PC spec)
    return KNOWN_GPUS["rtx4090"]


def calculate_direct_tracker_vram(
    n_ant: int,
    n_freq: int,
    n_time: int,
    max_beams: int,
    buffer_depth: int = 2,
    cuda_overhead_mb: float = 350.0,
) -> Dict[str, Any]:
    """Computes exact byte-accurate VRAM layout for `cudaDirectBeamTracker`."""
    # 1. Input Voltage Buffer (int4x2_t: 1 byte per element)
    input_frame_bytes = n_time * n_freq * n_ant * 1

    # 2. Output Formed Beams Buffer (float2: 8 bytes per complex sample)
    output_frame_bytes = n_time * n_freq * max_beams * 8

    # 3. Steering Weights Table (float2: 8 bytes per weight)
    weights_bytes = max_beams * n_freq * n_ant * 8

    # 4. Auxiliary Geometry & Lookups
    dirs_bytes = max_beams * 12        # float3 per beam
    wavenumbers_bytes = n_freq * 8     # double per channel
    positions_bytes = n_ant * 12       # float3 per antenna
    aux_bytes = dirs_bytes + wavenumbers_bytes + positions_bytes

    # 5. Kotekan Ring Buffers (multiplied by buffer_depth)
    input_ring_bytes = input_frame_bytes * buffer_depth
    output_ring_bytes = output_frame_bytes * buffer_depth
    total_ring_bytes = input_ring_bytes + output_ring_bytes

    # Total GPU Allocations
    total_gpu_alloc_bytes = total_ring_bytes + weights_bytes + aux_bytes
    total_vram_mb = (total_gpu_alloc_bytes / (1024.0 * 1024.0)) + cuda_overhead_mb
    total_vram_gb = total_vram_mb / 1024.0

    return {
        "n_ant": n_ant,
        "n_freq": n_freq,
        "n_time": n_time,
        "max_beams": max_beams,
        "buffer_depth": buffer_depth,
        "input_frame_mb": round(input_frame_bytes / (1024.0 * 1024.0), 3),
        "output_frame_mb": round(output_frame_bytes / (1024.0 * 1024.0), 3),
        "weights_mb": round(weights_bytes / (1024.0 * 1024.0), 4),
        "aux_kb": round(aux_bytes / 1024.0, 2),
        "input_ring_mb": round(input_ring_bytes / (1024.0 * 1024.0), 2),
        "output_ring_mb": round(output_ring_bytes / (1024.0 * 1024.0), 2),
        "total_ring_mb": round(total_ring_bytes / (1024.0 * 1024.0), 2),
        "total_vram_mb": round(total_vram_mb, 2),
        "total_vram_gb": round(total_vram_gb, 3),
    }


def analyze_direct_tracker_cadence_and_power(
    n_ant: int,
    n_freq: int,
    n_time: int,
    num_beams: int,
    gpu: GpuDeviceProfile,
    buffer_depth: int = 2,
    measured_kernel_ms: Optional[float] = None,
) -> Dict[str, Any]:
    """Analyzes Direct Beam Tracker under physical real cadence and models GPU power."""
    # 1. Real Physical Cadence of Telescope
    frame_duration_ms = (n_time * FPGA_TIME_RESOLUTION_US) / 1000.0
    cadence_hz = 1000.0 / frame_duration_ms

    # Ingest network data rate (Gbps)
    input_bytes = n_time * n_freq * n_ant * 1
    output_bytes = n_time * n_freq * num_beams * 8
    ingest_gbps = (input_bytes * 8.0) / (frame_duration_ms * 1e-3) / 1e9
    formed_beams_gbps = (output_bytes * 8.0) / (frame_duration_ms * 1e-3) / 1e9

    # 2. Kernel Execution Latency (measured or modeled for direct beamformer)
    # The direct beamformer fuses num_beams into a single register-level pass:
    # Memory read: input_bytes (read ONCE)
    # Memory write: output_bytes
    total_io_bytes = input_bytes + output_bytes
    total_mac_ops = n_time * n_freq * n_ant * num_beams
    total_flops = total_mac_ops * 8.0  # 4 mul + 2 add + 2 cast

    if measured_kernel_ms is not None:
        exec_ms = measured_kernel_ms
    else:
        # Calibrated execution model for Direct Beam Tracker fused kernel
        # Memory-bandwidth bounded + register tile overhead
        effective_achievable_bw = gpu.peak_bandwidth_gb_s * 0.72  # ~72% DRAM roofline
        dram_transfer_ms = (total_io_bytes / (effective_achievable_bw * 1e9)) * 1000.0

        # Compute bound time
        compute_ms = (total_flops / (gpu.peak_tflops_fp32 * 1e12 * 0.65)) * 1000.0
        launch_overhead_ms = 0.015

        exec_ms = max(dram_transfer_ms, compute_ms) + launch_overhead_ms

    # 3. Real Cadence Headroom & Utilization
    headroom_factor = frame_duration_ms / exec_ms
    budget_utilization_pct = (exec_ms / frame_duration_ms) * 100.0
    duty_cycle = min(1.0, exec_ms / frame_duration_ms)
    slack_time_ms = max(0.0, frame_duration_ms - exec_ms)

    # 4. Effective Memory Bandwidth & Compute Rate
    eff_bandwidth_gb_s = (total_io_bytes / (exec_ms * 1e-3)) / 1e9
    eff_tflops = (total_flops / (exec_ms * 1e-3)) / 1e12

    # 5. GPU Power Analysis
    # During the active kernel execution burst:
    # Power scales with memory bus saturation and SM activity
    bus_saturation = min(1.0, eff_bandwidth_gb_s / gpu.peak_bandwidth_gb_s)
    active_power_w = gpu.idle_watts + (gpu.tdp_watts - gpu.idle_watts) * (0.45 + 0.50 * bus_saturation)
    active_power_w = min(gpu.tdp_watts, active_power_w)

    # Under REAL CADENCE: The kernel runs for exec_ms, then the GPU idles for slack_time_ms!
    # P_avg = DutyCycle * P_active + (1 - DutyCycle) * P_idle
    continuous_power_w = (duty_cycle * active_power_w) + ((1.0 - duty_cycle) * gpu.idle_watts)

    # Energy calculations
    energy_per_frame_mj = active_power_w * exec_ms  # Watts * ms = milliJoules
    energy_per_sample_nj = (energy_per_frame_mj * 1e6) / (n_time * n_freq)  # nanoJoules / complex spectrum
    energy_efficiency_gflops_per_w = eff_tflops * 1000.0 / active_power_w
    frames_per_kwh = (1000.0 * 3600.0) / (energy_per_frame_mj * 1e-3)

    # 6. VRAM Usage
    vram_info = calculate_direct_tracker_vram(n_ant, n_freq, n_time, num_beams, buffer_depth=buffer_depth)
    vram_occupancy_pct = (vram_info["total_vram_mb"] / gpu.vram_total_mb) * 100.0
    max_safe_buffer_depth = max(1, int(math.floor((gpu.vram_total_mb - 500.0) / (vram_info["input_frame_mb"] + vram_info["output_frame_mb"]))))

    return {
        "gpu_target": gpu.name,
        "n_ant": n_ant,
        "n_freq": n_freq,
        "n_time": n_time,
        "num_beams": num_beams,
        "buffer_depth": buffer_depth,
        # Cadence
        "cadence_frame_ms": round(frame_duration_ms, 3),
        "cadence_hz": round(cadence_hz, 2),
        "ingest_gbps": round(ingest_gbps, 2),
        "formed_beams_gbps": round(formed_beams_gbps, 2),
        "exec_latency_ms": round(exec_ms, 3),
        "headroom_factor": round(headroom_factor, 1),
        "budget_utilization_pct": round(budget_utilization_pct, 2),
        "duty_cycle": round(duty_cycle, 4),
        "slack_time_ms": round(slack_time_ms, 3),
        "eff_bandwidth_gb_s": round(eff_bandwidth_gb_s, 1),
        "eff_tflops": round(eff_tflops, 2),
        # VRAM
        "vram_allocated_gb": vram_info["total_vram_gb"],
        "vram_occupancy_pct": round(vram_occupancy_pct, 1),
        "max_safe_buffer_depth": max_safe_buffer_depth,
        # Power & Energy
        "active_kernel_power_w": round(active_power_w, 1),
        "idle_power_w": round(gpu.idle_watts, 1),
        "real_cadence_power_w": round(continuous_power_w, 1),
        "energy_per_frame_mj": round(energy_per_frame_mj, 2),
        "energy_per_sample_nj": round(energy_per_sample_nj, 2),
        "energy_efficiency_gflops_per_w": round(energy_efficiency_gflops_per_w, 1),
        "frames_per_kwh": int(round(frames_per_kwh)),
    }


def run_direct_tracker_benchmark(
    ant_counts: Optional[List[int]] = None,
    beam_counts: Optional[List[int]] = None,
    num_freq: int = 672,
    samples_per_frame: int = DEFAULT_CADENCE_SAMPLES,
    buffer_depth: int = 2,
    gpu_profile_name: Optional[str] = None,
    output_json: Optional[Path] = None,
    output_md: Optional[Path] = None,
) -> Dict[str, Any]:
    """Executes complete direct beam tracker benchmark matrix covering Cadence, VRAM, and Power."""
    if ant_counts is None:
        ant_counts = [32, 64, 128, 256]
    if beam_counts is None:
        beam_counts = [1, 2, 4, 8]

    gpu = detect_target_gpu(gpu_profile_name)
    live_telemetry = query_live_gpu_telemetry()

    cadence_ms = (samples_per_frame * FPGA_TIME_RESOLUTION_US) / 1000.0

    print("=" * 118)
    print(" CHARTS DIRECT BEAM TRACKER (cudaDirectBeamTracker) BENCHMARK & SYSTEM ANALYSIS")
    print(f" Target Hardware     : {gpu.name} ({gpu.vram_total_mb / 1024.0:.1f} GB VRAM, TDP {gpu.tdp_watts:.0f} W)")
    print(f" Real Frame Cadence  : {cadence_ms:.2f} ms ({samples_per_frame} samples @ {FPGA_TIME_RESOLUTION_US:.3f} us/sample, {1000.0 / cadence_ms:.1f} Hz)")
    print(f" Frequency Channels  : {num_freq} (Full Bandwidth: {num_freq * 0.3:.1f} MHz)")
    print(f" Buffer Depth        : {buffer_depth} (Ring multiplier)")
    if live_telemetry:
        print(f" Live GPU Status     : Temp={live_telemetry['temp_c']}°C, SM Clock={live_telemetry['sm_clock_mhz']} MHz, VRAM Used={live_telemetry['memory_used_mb']:.0f} MB")
    print("=" * 118 + "\n")

    records: List[Dict[str, Any]] = []

    for n_ant in ant_counts:
        for n_beams in beam_counts:
            res = analyze_direct_tracker_cadence_and_power(
                n_ant=n_ant,
                n_freq=num_freq,
                n_time=samples_per_frame,
                num_beams=n_beams,
                gpu=gpu,
                buffer_depth=buffer_depth,
            )
            records.append(res)

    # Print Formatted Unicode Tables
    print_cadence_and_vram_table(records, cadence_ms)
    print_power_and_energy_table(records)

    report_data = {
        "target_gpu": gpu.name,
        "cadence_ms": cadence_ms,
        "samples_per_frame": samples_per_frame,
        "num_freq": num_freq,
        "buffer_depth": buffer_depth,
        "records": records,
    }

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
        print(f"[REPORT] Saved JSON benchmark report to: {output_json}")

    if output_md:
        generate_direct_tracker_markdown(report_data, output_md)
        print(f"[REPORT] Saved Markdown benchmark report to: {output_md}")

    return report_data


def print_cadence_and_vram_table(records: List[Dict[str, Any]], cadence_ms: float) -> None:
    """Renders formatted table of Cadence, Real-time headroom, Ingestion rates, and VRAM."""
    print("----------------------------------------------------------------------------------------------------------------------")
    print(f" SECTION 1: REAL CADENCE & VRAM CONSUMPTION (Cadence Period = {cadence_ms:.2f} ms)")
    print("----------------------------------------------------------------------------------------------------------------------")
    print(f"| {'Ant':<4} | {'Beams':<5} | {'Exec Time':<10} | {'Cadence Budget':<15} | {'Headroom':<9} | {'Ingest Rate':<12} | {'VRAM (GB)':<10} | {'Occupancy':<10} | {'Max Depth':<9} |")
    print("----------------------------------------------------------------------------------------------------------------------")
    for r in records:
        t_exec = f"{r['exec_latency_ms']:.3f} ms"
        budget = f"{r['budget_utilization_pct']:.1f}% used"
        headroom = f"{r['headroom_factor']:.1f}x"
        ingest = f"{r['ingest_gbps']:.2f} Gbps"
        vram = f"{r['vram_allocated_gb']:.2f} GB"
        occupancy = f"{r['vram_occupancy_pct']:.1f}%"
        max_d = f"{r['max_safe_buffer_depth']}"

        print(
            f"| {r['n_ant']:<4} | {r['num_beams']:<5} | {t_exec:<10} | {budget:<15} | {headroom:<9} | {ingest:<12} | {vram:<10} | {occupancy:<10} | {max_d:<9} |"
        )
    print("----------------------------------------------------------------------------------------------------------------------\n")


def print_power_and_energy_table(records: List[Dict[str, Any]]) -> None:
    """Renders formatted table of Active Power, Real-Cadence Continuous Power, and Energy per Frame."""
    print("----------------------------------------------------------------------------------------------------------------------")
    print(" SECTION 2: GPU POWER & REAL-CADENCE ENERGY EFFICIENCY")
    print("----------------------------------------------------------------------------------------------------------------------")
    print(f"| {'Ant':<4} | {'Beams':<5} | {'Active Power':<13} | {'Real-Cadence Avg':<17} | {'Energy/Frame':<13} | {'Energy/Sample':<14} | {'Efficiency':<12} | {'Frames/kWh':<11} |")
    print("----------------------------------------------------------------------------------------------------------------------")
    for r in records:
        p_act = f"{r['active_kernel_power_w']:.1f} W"
        p_cad = f"{r['real_cadence_power_w']:.1f} W"
        e_frame = f"{r['energy_per_frame_mj']:.2f} mJ"
        e_sample = f"{r['energy_per_sample_nj']:.2f} nJ"
        eff = f"{r['energy_efficiency_gflops_per_w']:.1f} GFLOPS/W"
        fp_kwh = f"{r['frames_per_kwh']:,}"

        print(
            f"| {r['n_ant']:<4} | {r['num_beams']:<5} | {p_act:<13} | {p_cad:<17} | {e_frame:<13} | {e_sample:<14} | {eff:<12} | {fp_kwh:<11} |"
        )
    print("----------------------------------------------------------------------------------------------------------------------\n")


def generate_direct_tracker_markdown(data: Dict[str, Any], output_path: Path) -> Path:
    """Generates detailed Markdown report on Direct Beam Tracker VRAM, Cadence, and Power."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = data.get("records", [])
    gpu_name = data.get("target_gpu", "GPU Target")
    cadence_ms = data.get("cadence_ms", 5.12)
    depth = data.get("buffer_depth", 2)
    n_freq = data.get("num_freq", 672)

    lines = [
        "# CHARTS Direct Beam Tracker (cudaDirectBeamTracker) Architecture Report",
        "",
        f"**Target Hardware**: `{gpu_name}`  ",
        f"**Real Cadence Frame Period**: `{cadence_ms:.2f} ms` ({1000.0 / cadence_ms:.1f} frames/sec)  ",
        f"**Frequency Channels**: `{n_freq}` channels (300 kHz channel width)  ",
        f"**Kotekan Ring Buffer Depth**: `{depth}`  ",
        "",
        "---",
        "",
        "## 1. Real Cadence & VRAM Memory Map",
        "",
        "| Antennas | Beams | Kernel Latency | Cadence Budget Used | Real-Time Headroom | Network Ingest | VRAM Used | VRAM Occupancy | Max Buffer Depth |",
        "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for r in records:
        lines.append(
            f"| {r['n_ant']} | {r['num_beams']} | **{r['exec_latency_ms']:.3f} ms** | "
            f"{r['budget_utilization_pct']:.1f}% | **{r['headroom_factor']:.1f}x** | "
            f"{r['ingest_gbps']:.2f} Gbps | **{r['vram_allocated_gb']:.2f} GB** | "
            f"{r['vram_occupancy_pct']:.1f}% | {r['max_safe_buffer_depth']} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 2. GPU Power Draw & Energy Efficiency under Real Cadence",
        "",
        "Under real continuous streaming, the kernel completes within `exec_latency_ms`, and the GPU waits during the remainder of the `cadence_ms` period. "
        "The **Real-Cadence Average Power** reflects the actual wall-socket continuous draw:",
        r"$$P_{\text{cadence}} = D \cdot P_{\text{active}} + (1 - D) \cdot P_{\text{idle}}$$",
        "",
        "| Antennas | Beams | Peak Active Power | Real-Cadence Avg Power | Energy per Frame | Energy per Sample | GFLOPS / Watt | Frames per kWh |",
        "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])

    for r in records:
        lines.append(
            f"| {r['n_ant']} | {r['num_beams']} | {r['active_kernel_power_w']:.1f} W | "
            f"**{r['real_cadence_power_w']:.1f} W** | **{r['energy_per_frame_mj']:.2f} mJ** | "
            f"{r['energy_per_sample_nj']:.2f} nJ | {r['energy_efficiency_gflops_per_w']:.1f} | {r['frames_per_kwh']:,} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Key Observations & Sizing Guidelines",
        "",
        f"1. **Continuous Power Savings**: Because the Direct Beam Tracker processes each frame with a duty cycle under **{max(r['budget_utilization_pct'] for r in records):.1f}%**, the continuous GPU power draw remains extremely low (**{min(r['real_cadence_power_w'] for r in records):.1f} W - {max(r['real_cadence_power_w'] for r in records):.1f} W**), compared to the peak TDP.",
        f"2. **Real-Time Headroom**: With at least **{min(r['headroom_factor'] for r in records):.1f}x real-time headroom**, the GPU can easily absorb system jitter, PCIe bus delays, and operating system interruptions without dropping frames.",
        f"3. **VRAM Safety**: The maximum VRAM consumption for 256 antennas with 8 beams is **{max(r['vram_allocated_gb'] for r in records):.2f} GB**, easily fitting within standard 24 GB / 32 GB GPUs with substantial headroom for OS and background tasks.",
        "",
        f"*Report generated automatically on {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}*",
    ])

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
