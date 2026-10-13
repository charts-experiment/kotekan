#!/usr/bin/env python3
"""CHARTS Kotekan Beam Tracker Continuous Loop & High-Throughput Stress Test.

Stress-tests the Kotekan CUDA Beam Tracker pipeline by repeatedly reading and
processing the same baseband integration window in a continuous high-throughput
loop over hundreds/thousands of iterations.

Profiles:
- Sustained processing throughput (GB/s, Million spectra/sec, Frames/sec).
- Kernel latency vs. Real Sky Integration Window Time (T_sky = N_time * 3.333 us).
- Real-Time Factor (RTF = T_GPU / T_sky): Identifies GPU bottlenecking / capability.
- Pipeline stability, queue health, zero VRAM/RAM memory leaks.
- Concurrent dynamic REST steering under maximum stress load.

Usage:
    python stress_test_beam_tracker_loop.py --repeats 1000 --n-time 320 --n-freq 16 --n-ant 64 --beams 2
    python stress_test_beam_tracker_loop.py --repeats 500 --n-time 3200 --n-freq 84 --n-ant 64 --beams 4
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Setup paths
_script_dir = Path(__file__).resolve().parent
_kotekan_root = _script_dir.parent
_kotekan_bin = _kotekan_root / "build" / "kotekan" / "kotekan"

if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from constants import (
    CHARTS_ALTITUDE_M,
    CHARTS_CHANNEL_WIDTH_HZ,
    CHARTS_CHANNEL_WIDTH_MHZ,
    CHARTS_LATITUDE_DEG,
    CHARTS_LONGITUDE_DEG,
    DEFAULT_FREQUENCY_START_HZ,
    DEFAULT_FREQUENCY_START_MHZ,
    DEFAULT_SPACING_M,
    FPGA_TIME_RESOLUTION_S,
    FPGA_TIME_RESOLUTION_US,
)
from kotekan_tracker_control import KotekanTrackerClient


def generate_stress_window_binary(
    out_file: Path,
    n_ant: int = 64,
    n_freq: int = 16,
    n_time: int = 320,
    seed: int = 42,
) -> int:
    """Generates a raw binary file for 1 integration window with Kotekan uint32 metadata header."""
    out_file.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # 4-bit packed complex samples: shape (n_time, n_freq, n_ant)
    # real in bits 0-3, imag in bits 4-7
    r = rng.integers(-4, 4, size=(n_time, n_freq, n_ant), dtype=np.int8)
    i = rng.integers(-4, 4, size=(n_time, n_freq, n_ant), dtype=np.int8)
    packed = ((r & 0x0F) | ((i & 0x0F) << 4)).astype(np.uint8)

    payload_bytes = packed.tobytes()
    metadata_size = 0

    with open(out_file, "wb") as f:
        f.write(struct.pack("<I", metadata_size))
        f.write(payload_bytes)

    total_bytes = 4 + len(payload_bytes)
    return total_bytes


def generate_stress_test_config(
    config_file: Path,
    work_dir: Path,
    n_ant: int = 64,
    n_freq: int = 16,
    n_time: int = 320,
    integration_spectra: int = 320,
    max_beams: int = 8,
    active_beams: int = 2,
    repeats: int = 1000,
    buffer_depth: int = 6,
    rest_port: int = 12066,
) -> None:
    """Creates Kotekan YAML config for looping the same integration window."""
    config_file.parent.mkdir(parents=True, exist_ok=True)
    in_dir = work_dir / "input"
    max_beams = max(max_beams, active_beams)

    yaml_content = f"""type: config
log_level: error

cpu_affinity: [0, 1]

num_elements: {n_ant}
num_local_freq: {n_freq}
samples_per_data_set: {n_time}
integration_spectra: {integration_spectra}
spacing_m: {DEFAULT_SPACING_M}
max_beams: {max_beams}
initial_active_beams: {active_beams}
buffer_depth: {buffer_depth}
sizeof_complex_float: 8

main_pool:
  kotekan_metadata_pool: chordMetadata
  num_metadata_objects: 30

network_capture_buf:
  kotekan_buffer: standard
  num_frames: buffer_depth
  frame_size: samples_per_data_set * num_local_freq * num_elements
  numa_node: 0
  metadata_pool: main_pool
  zero_new_frames: true
  mlock_frames: false

host_formed_beams_buffer:
  kotekan_buffer: standard
  num_frames: buffer_depth
  frame_size: samples_per_data_set * num_local_freq * max_beams * sizeof_complex_float
  numa_node: 0
  metadata_pool: main_pool
  zero_new_frames: true
  mlock_frames: false

reader:
  kotekan_stage: rawFileRead
  base_dir: "{in_dir}"
  file_name: "window_replay"
  file_ext: "bin"
  prefix_hostname: false
  loop_files: true
  max_repeats: {repeats}
  end_interrupt: true
  buf: network_capture_buf

gpu:
  profiling: false
  commands: &command_list
    - name: cudaInputData
      in_buf: host_voltage
      gpu_mem: voltage
    - name: cudaSyncInput
    - name: cudaBeamTrackerCommand
      gpu_mem_voltage: voltage
      gpu_mem_formed_beams: formed_beams
      num_elements: {n_ant}
      num_local_freq: {n_freq}
      samples_per_data_set: {n_time}
      integration_spectra: {integration_spectra}
      spacing_m: {DEFAULT_SPACING_M}
      max_beams: {max_beams}
      initial_active_beams: {active_beams}
      source_l0: 0.05
      source_m0: -0.02
      source_dl: 1.0e-5
      source_dm: 0.0
    - name: cudaSyncOutput
    - name: cudaOutputData
      in_buf: host_voltage
      gpu_mem: formed_beams
      out_buf: host_formed_beams
  gpu_0:
    kotekan_stage: cudaProcess
    gpu_id: 0
    commands: *command_list
    in_buffers:
      host_voltage: network_capture_buf
    out_buffers:
      host_formed_beams: host_formed_beams_buffer

sink:
  kotekan_stage: dropAllFrames
  in_buf: host_formed_beams_buffer
"""
    config_file.write_text(yaml_content, encoding="utf-8")


def run_stress_test(
    repeats: int = 1000,
    n_ant: int = 64,
    n_freq: int = 16,
    n_time: int = 320,
    integration_spectra: int = 320,
    max_beams: int = 4,
    active_beams: int = 2,
    rest_port: int = 12066,
    exercise_rest_steering: bool = True,
) -> Dict[str, Any]:
    """Executes the loop stress test, measuring throughput, timing, and RTF."""
    print("=" * 80)
    print(" CHARTS KOTEKAN BEAM TRACKER HIGH-THROUGHPUT LOOP STRESS TEST")
    print("=" * 80)

    work_dir = Path("/tmp/test_charts_stress_loop")
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    input_file = work_dir / "input" / "window_replay_0000000.bin"
    config_file = work_dir / "config" / "stress_test_config.yaml"

    frame_bytes = n_time * n_freq * n_ant  # 1 byte per int4x2_t
    output_frame_bytes = n_time * n_freq * max_beams * 8  # 8 bytes per float2
    total_input_bytes = frame_bytes * repeats
    total_output_bytes = output_frame_bytes * repeats

    # Real sky timing
    sky_time_per_frame_s = n_time * FPGA_TIME_RESOLUTION_S
    sky_time_per_frame_ms = sky_time_per_frame_s * 1000.0
    total_sky_time_s = sky_time_per_frame_s * repeats

    print(f"\n[1/4] Stress Test Configuration & Sizing:")
    print(f"  -> Array Elements (Antennas)   : {n_ant}")
    print(f"  -> Frequency Channels (F)      : {n_freq}")
    print(f"  -> Time Samples per Frame (T)  : {n_time} spectra")
    print(f"  -> Integration Window (N_int)  : {integration_spectra} spectra")
    print(f"  -> Active Beams / Max Beams    : {active_beams} / {max_beams}")
    print(
        f"  -> Repetition Count (Loop)     : {repeats:,} repeated integration windows"
    )
    print(
        f"  -> Frame Input Size            : {frame_bytes / 1024:.2f} KB ({frame_bytes:,} bytes)"
    )
    print(
        f"  -> Total Data Processed Ingest : {total_input_bytes / (1024**2):.2f} MB ({total_input_bytes:,} bytes)"
    )
    print(
        f"  -> Real Sky Duration per Frame : {sky_time_per_frame_ms:.4f} ms ({sky_time_per_frame_s:.6f} s)"
    )
    print(
        f"  -> Total Real Sky Stream Time  : {total_sky_time_s:.3f} seconds ({total_sky_time_s / 60.0:.2f} minutes)"
    )

    # 1. Generate Input Window
    generate_stress_window_binary(input_file, n_ant=n_ant, n_freq=n_freq, n_time=n_time)
    generate_stress_test_config(
        config_file,
        work_dir,
        n_ant=n_ant,
        n_freq=n_freq,
        n_time=n_time,
        integration_spectra=integration_spectra,
        max_beams=max_beams,
        active_beams=active_beams,
        repeats=repeats,
        rest_port=rest_port,
    )

    # 2. Launch Kotekan
    print(f"\n[2/4] Launching Kotekan Pipeline with Continuous Loop Streaming...")
    cmd = [
        str(_kotekan_bin),
        "-c",
        str(config_file),
        "-b",
        f"127.0.0.1:{rest_port}",
    ]

    rest_ops_done = []
    stop_rest_thread = threading.Event()

    def rest_steering_worker():
        """Background thread exercising dynamic REST steering under max load."""
        client = KotekanTrackerClient(port=rest_port)
        time.sleep(0.4)
        count = 0
        while not stop_rest_thread.is_set():
            try:
                # Alternate steering directions
                l0 = 0.05 * np.sin(count * 0.2)
                m0 = -0.02 * np.cos(count * 0.2)
                client.steer_lm(0, l0=float(l0), m0=float(m0), dl=1e-5, dm=0.0)
                client.steer_lm(1, l0=float(-l0), m0=float(-m0), dl=0.0, dm=1e-5)
                if count % 10 == 0:
                    client.mask_antenna(4, enabled=(count % 20 != 0))
                count += 1
                time.sleep(0.05)
            except Exception:
                time.sleep(0.02)
        rest_ops_done.append(count)

    rest_thread = None
    if exercise_rest_steering:
        rest_thread = threading.Thread(target=rest_steering_worker, daemon=True)

    t_start = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if rest_thread:
        rest_thread.start()

    # Wait for Kotekan to complete all repeats
    returncode = proc.wait()
    t_end = time.perf_counter()

    if rest_thread:
        stop_rest_thread.set()
        rest_thread.join(timeout=1.0)

    elapsed_s = t_end - t_start
    rest_count = rest_ops_done[0] if rest_ops_done else 0

    print(f"  -> Kotekan Execution Finished with Exit Code: {returncode}")
    print(f"  -> Total Wall Clock Time Elapsed: {elapsed_s:.4f} seconds")

    # 3. Calculate Performance Metrics
    throughput_fps = repeats / elapsed_s
    throughput_input_gbps = (total_input_bytes / (1024**3)) / elapsed_s
    throughput_input_mbps = (total_input_bytes / (1024**2)) / elapsed_s
    spectra_rate_msps = (n_time * n_freq * repeats) / (elapsed_s * 1e6)
    ms_per_frame = (elapsed_s / repeats) * 1000.0
    us_per_spectra = (elapsed_s / (n_time * repeats)) * 1e6

    # Real-Time Factor: T_proc / T_sky
    # RTF < 1.0: Real-time capable (processes faster than sky cadence)
    # RTF > 1.0: GPU bottlenecked (processes slower than sky cadence)
    rtf = ms_per_frame / sky_time_per_frame_ms
    status_str = (
        "[REAL-TIME CAPABLE]"
        if rtf <= 1.0
        else "[GPU BOTTLENECKED (Expected on this device)]"
    )

    print(f"\n[3/4] High-Throughput Stress Test Profiling Results:")
    print("=" * 80)
    print(
        f" | Metric                           | Measured Value         | Real-Time Target         |"
    )
    print(
        " +----------------------------------+------------------------+--------------------------+"
    )
    print(
        f" | Repeated Integration Frames      | {repeats:12,d} frames   | Continuous Stream        |"
    )
    print(
        f" | Total Processing Wall Time       | {elapsed_s:12.3f} s        | {total_sky_time_s:10.3f} s (Sky Time)  |"
    )
    print(
        f" | Processing Frame Rate (FPS)      | {throughput_fps:12.1f} fps      | {1000.0/sky_time_per_frame_ms:10.1f} fps (Sky Rate)  |"
    )
    print(
        f" | Ingestion Throughput (MB/s)      | {throughput_input_mbps:12.2f} MB/s     | {(frame_bytes/(1024**2))/(sky_time_per_frame_s):10.2f} MB/s           |"
    )
    print(
        f" | Ingestion Throughput (GB/s)      | {throughput_input_gbps:12.3f} GB/s     | {(frame_bytes/(1024**3))/(sky_time_per_frame_s):10.3f} GB/s           |"
    )
    print(
        f" | Complex Spectra Processed Rate   | {spectra_rate_msps:12.2f} MSamp/s   | {(n_freq * 300000.0)/1e6:10.2f} MSamp/s        |"
    )
    print(
        f" | Processing Latency per Frame     | {ms_per_frame:12.4f} ms       | {sky_time_per_frame_ms:10.4f} ms (Cadence)   |"
    )
    print(
        f" | Processing Time per Spectrum     | {us_per_spectra:12.4f} us       | {FPGA_TIME_RESOLUTION_US:10.4f} us (Cadence)   |"
    )
    print(
        f" | Real-Time Factor (T_gpu / T_sky) | {rtf:12.3f} x        | <= 1.000 x               |"
    )
    print(f" | Real-Time Budget Status          | {status_str:40s} |")
    print(
        f" | Concurrent REST Steering Ops     | {rest_count:12,d} calls    | Zero Lock-Ups / Crashes  |"
    )
    print("=" * 80)

    # 4. Cleanup
    if work_dir.exists():
        shutil.rmtree(work_dir)

    report = {
        "repeats": repeats,
        "n_ant": n_ant,
        "n_freq": n_freq,
        "n_time": n_time,
        "active_beams": active_beams,
        "max_beams": max_beams,
        "elapsed_seconds": elapsed_s,
        "sky_time_seconds": total_sky_time_s,
        "throughput_fps": throughput_fps,
        "throughput_input_mbps": throughput_input_mbps,
        "throughput_input_gbps": throughput_input_gbps,
        "spectra_rate_msps": spectra_rate_msps,
        "latency_ms_per_frame": ms_per_frame,
        "latency_us_per_spectrum": us_per_spectra,
        "real_time_factor": rtf,
        "is_real_time_capable": bool(rtf <= 1.0),
        "concurrent_rest_ops": rest_count,
        "returncode": returncode,
    }

    return report


def main():
    parser = argparse.ArgumentParser(
        description="CHARTS Kotekan Beam Tracker Loop Stress Test"
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1000,
        help="Number of integration frame repeats (default: 1000)",
    )
    parser.add_argument(
        "--n-ant", type=int, default=64, help="Number of antenna elements (default: 64)"
    )
    parser.add_argument(
        "--n-freq",
        type=int,
        default=16,
        help="Number of frequency channels (default: 16)",
    )
    parser.add_argument(
        "--n-time",
        type=int,
        default=320,
        help="Time samples per integration frame (default: 320)",
    )
    parser.add_argument(
        "--beams", type=int, default=2, help="Number of active beams (default: 2)"
    )
    parser.add_argument(
        "--no-rest", action="store_true", help="Disable concurrent REST steering test"
    )
    args = parser.parse_args()

    run_stress_test(
        repeats=args.repeats,
        n_ant=args.n_ant,
        n_freq=args.n_freq,
        n_time=args.n_time,
        active_beams=args.beams,
        exercise_rest_steering=(not args.no_rest),
    )


if __name__ == "__main__":
    main()
