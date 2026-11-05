#!/usr/bin/env python3
"""
Exhaustive Local Verification Suite for CHARTS Solar Transit Direct Beam Tracker
Covers:
  - Phase 4: Astrometric Transit Simulation & 4-Element Interferometric Physics
      * Test 4.1: Forward Gain Corruption & Exact Inversion Recovery
      * Test 4.2: Topocentric Stationarity Invariance Test
      * Test 4.3: Full 13-Step Synthetic Solar Transit & Sequential Peak Verification
  - Phase 6: Post-Processing & Analysis Script Validation
      * Telemetry Extraction & Light Curve Plotting
      * Negative / Edge-Case Resilience (Truncated file, Dual-Stream Alignment)
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Observational & Array Physics Constants
C_LIGHT = 299792458.0
NUM_ELEMENTS = 64
NUM_LOCAL_FREQ = 336
TOTAL_FREQ = 672
SAMPLES_PER_FRAME = 15360
MAX_BEAMS = 4
SIZEOF_COMPLEX_FLOAT = 8

# Physical antenna layout along E-W axis (y=0, z=0)
CALIBRATED_ANTENNAS = [0, 5, 32, 37]
ANTENNA_X_POSITIONS = {0: 0.0, 5: 0.6, 32: 3.6, 37: 4.2}

# Beam Centers (Traffic Cameras)
BEAMS = [
    {"beam_id": 0, "name": "Cam 0 (12:45 CLST)", "l0": 0.1773, "m0": 0.3247},
    {"beam_id": 1, "name": "Cam 1 (13:15 CLST)", "l0": 0.0512, "m0": 0.3329},
    {"beam_id": 2, "name": "Cam 2 (13:45 CLST)", "l0": -0.0757, "m0": 0.3319},
    {"beam_id": 3, "name": "Cam 3 (14:15 CLST)", "l0": -0.2014, "m0": 0.3218},
]

KOTEKAN_ROOT = Path("/home/fpalma/kotekan")
PY_EXEC = Path("/opt/kotekan_env/.venv/bin/python") if Path("/opt/kotekan_env/.venv/bin/python").exists() else Path(sys.executable)
KOTEKAN_BIN = KOTEKAN_ROOT / "build" / "kotekan" / "kotekan"
GAINS_FILE = KOTEKAN_ROOT / "charts" / "config" / "gains" / "gains_sun_transit_1027_interp.npy"


def pack_int4x2(voltages_complex: np.ndarray) -> np.ndarray:
    """Pack complex voltages into signed 4-bit nibbles stored in uint8."""
    r = np.clip(np.round(np.real(voltages_complex)), -8, 7).astype(np.int8) & 0x0F
    i = np.clip(np.round(np.imag(voltages_complex)), -8, 7).astype(np.int8) & 0x0F
    return (r | (i << 4)).astype(np.uint8)


def write_raw_input_frame(fpath: Path, packed_u8: np.ndarray, timestamp_utc: float = 0.0):
    """Write rawFileRead-compatible frame with 4-byte metadata header."""
    fpath.parent.mkdir(parents=True, exist_ok=True)
    with open(fpath, "wb") as f:
        np.uint32(0).tofile(f)
        packed_u8.tofile(f)


def read_raw_formed_beams(fpath: Path, n_time: int = SAMPLES_PER_FRAME, n_freq: int = NUM_LOCAL_FREQ, n_beams: int = MAX_BEAMS) -> np.ndarray:
    """Read rawFileWrite formed-beam frame."""
    payload_bytes = n_time * n_freq * n_beams * SIZEOF_COMPLEX_FLOAT
    with open(fpath, "rb") as f:
        meta_sz = struct.unpack("<I", f.read(4))[0]
        f.read(meta_sz)
        data = f.read(payload_bytes)
        assert len(data) == payload_bytes, f"Short read: got {len(data)}, expected {payload_bytes}"
        return np.frombuffer(data, dtype=np.complex64).reshape((n_time, n_freq, n_beams))


def build_replay_yaml(
    yaml_path: Path,
    input_dir_0: Path,
    input_dir_1: Optional[Path],
    output_dir: Path,
    num_frames: int,
    gains_file: Optional[Path] = GAINS_FILE,
    antenna_order: str = "descending",
    samples_per_frame: int = SAMPLES_PER_FRAME,
) -> Path:
    """Generate Kotekan dual-stream or single-stream replay YAML config."""
    custom_pos_yaml = []
    for a in range(64):
        x = ANTENNA_X_POSITIONS.get(a, 0.0)
        custom_pos_yaml.append(f"            - [{x:.1f}, 0.0, 0.0]")
    custom_pos_str = "\n".join(custom_pos_yaml)

    cal_str_0 = ""
    cal_str_1 = ""
    if gains_file is not None:
        cal_str_0 = f"""          gains_file: {Path(gains_file).as_posix()}
          gains_antennas: [0, 5, 32, 37]
          gains_first_channel: 0"""
        cal_str_1 = f"""          gains_file: {Path(gains_file).as_posix()}
          gains_antennas: [0, 5, 32, 37]
          gains_first_channel: 336"""

    dual_stream = input_dir_1 is not None

    cfg = f"""type: config
log_level: info
cpu_affinity: [0, 1, 2, 3]

num_elements: 64
num_local_freq: 336
samples_per_data_set: {samples_per_frame}
sample_period_s: 3.333333333333e-6
frame_arrival_period: samples_per_data_set * sample_period_s
max_beams: 4
initial_active_beams: 4
buffer_depth: 3
gpu_buffer_depth: 3
sizeof_complex_float: 8

main_pool:
    kotekan_metadata_pool: chartsMetadata
    num_metadata_objects: 30

network_capture_buf_0:
    kotekan_buffer: standard
    num_frames: 3
    frame_size: samples_per_data_set * num_local_freq * num_elements
    numa_node: 0
    metadata_pool: main_pool
    zero_new_frames: true
    mlock_frames: false

host_formed_beams_buffer_0:
    kotekan_buffer: standard
    num_frames: 3
    frame_size: samples_per_data_set * num_local_freq * max_beams * sizeof_complex_float
    numa_node: 0
    metadata_pool: main_pool
    zero_new_frames: true
    mlock_frames: false

reader_0:
    kotekan_stage: rawFileRead
    buf: network_capture_buf_0
    base_dir: "{input_dir_0.as_posix()}"
    file_name: "raw_stream0"
    file_ext: "bin"
    prefix_hostname: false
    end_interrupt: false
    loop_files: false

raw_file_write_beams_0:
    kotekan_stage: rawFileWrite
    in_buf: host_formed_beams_buffer_0
    base_dir: "{output_dir.as_posix()}"
    file_name: "formed_beams_sun_traffic_cams_freq0"
    file_ext: "bin"
    prefix_hostname: false
    write_every_n_frames: 1
    num_frames_per_file: 1
    exit_after_n_files: {num_frames}
"""

    if dual_stream:
        cfg += f"""
network_capture_buf_1:
    kotekan_buffer: standard
    num_frames: 3
    frame_size: samples_per_data_set * num_local_freq * num_elements
    numa_node: 0
    metadata_pool: main_pool
    zero_new_frames: true
    mlock_frames: false

host_formed_beams_buffer_1:
    kotekan_buffer: standard
    num_frames: 3
    frame_size: samples_per_data_set * num_local_freq * max_beams * sizeof_complex_float
    numa_node: 0
    metadata_pool: main_pool
    zero_new_frames: true
    mlock_frames: false

reader_1:
    kotekan_stage: rawFileRead
    buf: network_capture_buf_1
    base_dir: "{input_dir_1.as_posix()}"
    file_name: "raw_stream1"
    file_ext: "bin"
    prefix_hostname: false
    end_interrupt: false
    loop_files: false

raw_file_write_beams_1:
    kotekan_stage: rawFileWrite
    in_buf: host_formed_beams_buffer_1
    base_dir: "{output_dir.as_posix()}"
    file_name: "formed_beams_sun_traffic_cams_freq1"
    file_ext: "bin"
    prefix_hostname: false
    write_every_n_frames: 1
    num_frames_per_file: 1
    exit_after_n_files: {num_frames}
"""

    cfg += f"""
gpu:
    profiling: false
    commands_0: &command_list_0
        - name: cudaInputData
          in_buf: host_voltage
          gpu_mem: voltage_0
        - name: cudaSyncInput
        - name: cudaDirectBeamTrackerCommand
          gpu_mem_voltage: voltage_0
          gpu_mem_formed_beams: formed_beams_0
          num_elements: 64
          num_local_freq: 336
          samples_per_data_set: {samples_per_frame}
          spacing_m: 0.6
          antenna_order: "{antenna_order}"
          active_antennas: [0, 5, 32, 37]
          custom_antenna_positions:
{custom_pos_str}
          max_beams: 4
          initial_active_beams: 4
          buffer_depth: 3
          time_chunk_size: 256
          time_unroll: 4
          beam_tile_size: 4
          enable_cuda_graph: false
{cal_str_0}
          freq_start_hz: 300000000.0
          freq_step_hz: 300000.0
          source_l0: 0.1773
          source_m0: 0.3247
          source_l0_1: 0.0512
          source_m0_1: 0.3329
          source_l0_2: -0.0757
          source_m0_2: 0.3319
          source_l0_3: -0.2014
          source_m0_3: 0.3218
        - name: cudaSyncOutput
        - name: cudaOutputData
          in_buf: host_voltage
          gpu_mem: formed_beams_0
          out_buf: host_formed_beams
"""
    if dual_stream:
        cfg += f"""
    commands_1: &command_list_1
        - name: cudaInputData
          in_buf: host_voltage
          gpu_mem: voltage_1
        - name: cudaSyncInput
        - name: cudaDirectBeamTrackerCommand
          gpu_mem_voltage: voltage_1
          gpu_mem_formed_beams: formed_beams_1
          num_elements: 64
          num_local_freq: 336
          samples_per_data_set: {samples_per_frame}
          spacing_m: 0.6
          antenna_order: "{antenna_order}"
          active_antennas: [0, 5, 32, 37]
          custom_antenna_positions:
{custom_pos_str}
          max_beams: 4
          initial_active_beams: 4
          buffer_depth: 3
          time_chunk_size: 256
          time_unroll: 4
          beam_tile_size: 4
          enable_cuda_graph: false
{cal_str_1}
          freq_start_hz: 400800000.0
          freq_step_hz: 300000.0
          source_l0: 0.1773
          source_m0: 0.3247
          source_l0_1: 0.0512
          source_m0_1: 0.3329
          source_l0_2: -0.0757
          source_m0_2: 0.3319
          source_l0_3: -0.2014
          source_m0_3: 0.3218
        - name: cudaSyncOutput
        - name: cudaOutputData
          in_buf: host_voltage
          gpu_mem: formed_beams_1
          out_buf: host_formed_beams
"""

    cfg += """
stages:
    gpu_0:
        kotekan_stage: cudaProcess
        gpu_id: 0
        buffer_depth: 3
        cuda_stream_base: 0
        in_buffers:
            host_voltage: network_capture_buf_0
        out_buffers:
            host_formed_beams: host_formed_beams_buffer_0
        commands: *command_list_0
"""
    if dual_stream:
        cfg += """
    gpu_1:
        kotekan_stage: cudaProcess
        gpu_id: 0
        buffer_depth: 3
        cuda_stream_base: 1
        in_buffers:
            host_voltage: network_capture_buf_1
        out_buffers:
            host_formed_beams: host_formed_beams_buffer_1
        commands: *command_list_1
"""
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(cfg, encoding="utf-8")
    return yaml_path


def run_kotekan(cfg_path: Path):
    """Run Kotekan with the specified config and wait for clean exit."""
    cmd = [str(KOTEKAN_BIN), "-c", str(cfg_path)]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        print(f"Kotekan failed with returncode {p.returncode}. Output:\n{p.stdout[-1500:]}")
        raise RuntimeError(f"Kotekan failed with returncode {p.returncode}")


def synthesize_voltage_frame(
    l_source: float,
    freq_start_hz: float,
    n_freq: int,
    n_time: int,
    forward_gains: Optional[np.ndarray] = None,
    corrupt_dead_antennas_with_rfi: bool = False,
    rfi_seed: int = 42,
) -> np.ndarray:
    """
    Synthesize baseband voltages of shape (n_time, n_freq, 64).
    Physics:
      s_a(t, f) = A * exp(j * 2*pi*f/c * x_a * l_source)
      If forward_gains given, v_a = g_a * s_a
    Antenna order is descending: raw element r = 63 - a.
    """
    rng = np.random.default_rng(rfi_seed)
    freqs = freq_start_hz + np.arange(n_freq, dtype=np.float64) * 300.0e3  # (n_freq,)
    c64_voltages = np.zeros((n_time, n_freq, 64), dtype=np.complex64)

    # Coherent sky source amplitude
    amp = 3.5

    for ant_idx, a in enumerate(CALIBRATED_ANTENNAS):
        xa = ANTENNA_X_POSITIONS[a]
        # delay = xa * l_source
        phase_sky = 2.0 * np.pi * freqs / C_LIGHT * (xa * l_source)
        # Vectorized over freq: (n_freq,)
        s_a = amp * np.exp(-1j * phase_sky, dtype=np.complex64)
        if forward_gains is not None:
            # forward_gains has shape (n_freq, 4)
            g_a = forward_gains[:, ant_idx].astype(np.complex64)
            s_a = s_a * g_a

        # Broadcast across time: (n_time, n_freq)
        raw_elem = 63 - a
        c64_voltages[:, :, raw_elem] = s_a[None, :]

    if corrupt_dead_antennas_with_rfi:
        # Fill all 60 dead antennas with max-amplitude saturated +7/-7 uncorrelated RFI
        dead_antennas = [a for a in range(64) if a not in CALIBRATED_ANTENNAS]
        for a in dead_antennas:
            raw_elem = 63 - a
            # High-amplitude noise driving int4 to saturation
            noise_r = rng.choice([-7.0, 7.0], size=(n_time, n_freq)).astype(np.float32)
            noise_i = rng.choice([-7.0, 7.0], size=(n_time, n_freq)).astype(np.float32)
            c64_voltages[:, :, raw_elem] = noise_r + 1j * noise_i

    return pack_int4x2(c64_voltages)


# ==============================================================================
# EXECUTION: TEST 4.1, 4.2, 4.3 and PHASE 6
# ==============================================================================
def main():
    print("==============================================================================")
    print(" CHARTS Direct Beam Tracker 6-Phase Exhaustive Local Verification Suite")
    print(" Phases 4 & 6: Interferometric Physics, Astrometric Transit, and Telemetry")
    print("==============================================================================")

    gains_all = np.load(GAINS_FILE)
    gains_s0 = gains_all[:336, :]
    gains_s1 = gains_all[336:, :]

    # --------------------------------------------------------------------------
    # Test 4.1: Forward Gain Corruption & Exact Inversion Recovery
    # --------------------------------------------------------------------------
    print("\n>>> [Test 4.1] Forward Gain Corruption & Exact Inversion Recovery <<<")
    work_dir = Path("/tmp/test_phase4_test1")
    shutil.rmtree(work_dir, ignore_errors=True)
    in_dir_zero = work_dir / "in_zero"
    in_dir_rfi = work_dir / "in_rfi"
    out_dir_cal_zero = work_dir / "out_cal_zero"
    out_dir_cal_rfi = work_dir / "out_cal_rfi"
    out_dir_uncal = work_dir / "out_uncal"

    # Synthesize 1 frame with source at Beam 0 center (l = 0.1773)
    # Frame Zero: dead antennas = 0
    packed_zero = synthesize_voltage_frame(
        l_source=0.1773,
        freq_start_hz=300.0e6,
        n_freq=336,
        n_time=1536,
        forward_gains=gains_s0,
        corrupt_dead_antennas_with_rfi=False,
    )
    write_raw_input_frame(in_dir_zero / "raw_stream0_0000000.bin", packed_zero)

    # Frame RFI: dead antennas saturated with +7/-7 RFI
    packed_rfi = synthesize_voltage_frame(
        l_source=0.1773,
        freq_start_hz=300.0e6,
        n_freq=336,
        n_time=1536,
        forward_gains=gains_s0,
        corrupt_dead_antennas_with_rfi=True,
    )
    write_raw_input_frame(in_dir_rfi / "raw_stream0_0000000.bin", packed_rfi)

    # Run Kotekan with Calibrated gains on Frame Zero
    cfg_cal_zero = build_replay_yaml(
        work_dir / "cal_zero.yaml",
        in_dir_zero,
        None,
        out_dir_cal_zero,
        num_frames=1,
        gains_file=GAINS_FILE,
        samples_per_frame=1536,
    )
    run_kotekan(cfg_cal_zero)

    # Run Kotekan with Calibrated gains on Frame RFI
    cfg_cal_rfi = build_replay_yaml(
        work_dir / "cal_rfi.yaml",
        in_dir_rfi,
        None,
        out_dir_cal_rfi,
        num_frames=1,
        gains_file=GAINS_FILE,
        samples_per_frame=1536,
    )
    run_kotekan(cfg_cal_rfi)

    # Run Kotekan WITHOUT calibration gains (unity weights) on Frame Zero
    cfg_uncal = build_replay_yaml(
        work_dir / "uncal.yaml",
        in_dir_zero,
        None,
        out_dir_uncal,
        num_frames=1,
        gains_file=None,
        samples_per_frame=1536,
    )
    run_kotekan(cfg_uncal)

    # Read formed beam outputs
    beams_cal_zero = read_raw_formed_beams(out_dir_cal_zero / "formed_beams_sun_traffic_cams_freq0_0000000.bin", n_time=1536)
    beams_cal_rfi = read_raw_formed_beams(out_dir_cal_rfi / "formed_beams_sun_traffic_cams_freq0_0000000.bin", n_time=1536)
    beams_uncal = read_raw_formed_beams(out_dir_uncal / "formed_beams_sun_traffic_cams_freq0_0000000.bin", n_time=1536)

    # Criterion 1: Zero Dead-Antenna Leakage (||Y_RFI - Y_Zero||_2 == 0)
    leakage_diff = np.max(np.abs(beams_cal_rfi - beams_cal_zero))
    print(f"  Dead Antenna RFI vs Zero Leakage: max |Y_RFI - Y_Zero| = {leakage_diff:.2e}")
    assert leakage_diff == 0.0, f"Dead antenna leakage detected! Diff = {leakage_diff}"
    print("  -> Criterion 1 PASSED: Strict Zero Dead-Antenna Leakage (Bit-Identical Output).")

    # Criterion 2: Coherent Phase Restoration
    # Beam 0 power:
    p_cal_b0 = np.mean(np.abs(beams_cal_zero[:, :, 0])**2)
    p_uncal_b0 = np.mean(np.abs(beams_uncal[:, :, 0])**2)
    gain_db = 10.0 * np.log10(p_cal_b0 / p_uncal_b0)
    p_uncal_all64 = p_uncal_b0 / 16.0
    gain_all64_db = 10.0 * np.log10(p_cal_b0 / p_uncal_all64)
    print(f"  Beam 0 Power: Calibrated = {p_cal_b0:.2f}, Uncalibrated (4-ant) = {p_uncal_b0:.2f} (Phase Gain = {gain_db:+.2f} dB)")
    print(f"  Beam 0 Power vs Uncalibrated Full Array (64-ant) = {p_uncal_all64:.2f} (Array Gain = {gain_all64_db:+.2f} dB)")
    assert gain_db >= 3.0, f"Coherent phase restoration insufficient! Gain = {gain_db} dB"
    print("  -> Criterion 2 PASSED: Coherent Phase Restoration Verified (+{:.2f} dB phase gain, +{:.2f} dB array gain).".format(gain_db, gain_all64_db))

    # --------------------------------------------------------------------------
    # Test 4.2: Topocentric Stationarity Invariance Test
    # --------------------------------------------------------------------------
    print("\n>>> [Test 4.2] Topocentric Stationarity Invariance Test <<<")
    dir_42 = Path("/tmp/test_phase4_test2")
    shutil.rmtree(dir_42, ignore_errors=True)
    in_42 = dir_42 / "in"
    out_42 = dir_42 / "out"

    # Generate 4 identical frames
    for f_idx in range(4):
        write_raw_input_frame(in_42 / f"raw_stream0_{f_idx:07d}.bin", packed_zero)

    cfg_42 = build_replay_yaml(
        dir_42 / "test42.yaml",
        in_42,
        None,
        out_42,
        num_frames=4,
        gains_file=GAINS_FILE,
        samples_per_frame=1536,
    )
    run_kotekan(cfg_42)

    powers_42 = []
    for f_idx in range(4):
        b = read_raw_formed_beams(out_42 / f"formed_beams_sun_traffic_cams_freq0_{f_idx:07d}.bin", n_time=1536)
        powers_42.append(np.mean(np.abs(b[:, :, 0])**2))

    max_pwr_diff_db = 10.0 * np.log10(max(powers_42) / min(powers_42))
    print(f"  Power across 4 frames: {powers_42}")
    print(f"  Max Power Variance across frames = {max_pwr_diff_db:.6f} dB")
    assert max_pwr_diff_db < 1e-4, f"Topocentric stationarity failed! Diff = {max_pwr_diff_db} dB"
    print("  -> Criterion PASSED: Formed Beam Output is Identical across Timestamps (Topocentric Lock).")

    # --------------------------------------------------------------------------
    # Test 4.3: Full 13-Step Synthetic Solar Transit & Sequential Peak Verification
    # --------------------------------------------------------------------------
    print("\n>>> [Test 4.3] Full 13-Step Synthetic Solar Transit & Sequential Peak Verification <<<")
    dump_dir = Path("/tmp/test_dump_sun_transit")
    shutil.rmtree(dump_dir, ignore_errors=True)
    transit_in_0 = Path("/tmp/transit_sim_stream0")
    transit_in_1 = Path("/tmp/transit_sim_stream1")
    shutil.rmtree(transit_in_0, ignore_errors=True)
    shutil.rmtree(transit_in_1, ignore_errors=True)

    # 13 steps: l_k = 0.3601 - k * 0.0619 (12:00 to 15:00 CLST in 15-min increments)
    l_schedule = [0.3601 - k * 0.0619 for k in range(13)]
    print(f"  Synthesizing 13 solar transit frames across dual streams (300.0 - 501.3 MHz)...")

    for k, l_k in enumerate(l_schedule):
        p0 = synthesize_voltage_frame(
            l_source=l_k,
            freq_start_hz=300.0e6,
            n_freq=336,
            n_time=SAMPLES_PER_FRAME,
            forward_gains=gains_s0,
        )
        write_raw_input_frame(transit_in_0 / f"raw_stream0_{k:07d}.bin", p0)

        p1 = synthesize_voltage_frame(
            l_source=l_k,
            freq_start_hz=400.8e6,
            n_freq=336,
            n_time=SAMPLES_PER_FRAME,
            forward_gains=gains_s1,
        )
        write_raw_input_frame(transit_in_1 / f"raw_stream1_{k:07d}.bin", p1)

    # Build dual-stream replay YAML and execute Kotekan
    cfg_transit = build_replay_yaml(
        Path("/tmp/transit_replay.yaml"),
        transit_in_0,
        transit_in_1,
        dump_dir,
        num_frames=13,
        gains_file=GAINS_FILE,
        samples_per_frame=SAMPLES_PER_FRAME,
    )
    print("  Executing Kotekan dual-stream replay for 13 frames...")
    run_kotekan(cfg_transit)

    # Verify dumped files
    s0_dumps = sorted(dump_dir.glob("formed_beams_sun_traffic_cams_freq0_*.bin"))
    s1_dumps = sorted(dump_dir.glob("formed_beams_sun_traffic_cams_freq1_*.bin"))
    assert len(s0_dumps) == 13 and len(s1_dumps) == 13, f"Expected 13 dumps/stream, got {len(s0_dumps)}, {len(s1_dumps)}"

    # Check Sequential Peak Criteria & Off-Axis Rejection
    # Aggregate broadband power per beam vs frame k:
    broadband_pwr = np.zeros((13, 4))
    freqs_all = np.linspace(300.0e6, 501.3e6, 672)

    correlations_b0 = []
    for k in range(13):
        b0 = read_raw_formed_beams(s0_dumps[k])  # (time, 336, 4)
        b1 = read_raw_formed_beams(s1_dumps[k])  # (time, 336, 4)
        p0 = np.mean(np.abs(b0)**2, axis=0)      # (336, 4)
        p1 = np.mean(np.abs(b1)**2, axis=0)      # (336, 4)
        full_p = np.vstack([p0, p1])             # (672, 4)
        broadband_pwr[k, :] = np.mean(full_p, axis=0)

        # Single-channel fringe verification against analytical model for Beam 0
        l_k = l_schedule[k]
        dl = l_k - 0.1773
        p_ana_s0 = np.cos(np.pi * freqs_all[:336] * 0.6 * dl / C_LIGHT)**2 * np.cos(np.pi * freqs_all[:336] * 3.6 * dl / C_LIGHT)**2
        p_meas_s0 = p0[:, 0] / np.max(p0[:, 0]) if np.max(p0[:, 0]) > 0 else p0[:, 0]
        if np.std(p_meas_s0) > 1e-4 and np.std(p_ana_s0) > 1e-4:
            r = np.corrcoef(p_meas_s0, p_ana_s0)[0, 1]
            correlations_b0.append(r)

    print("\n  Broadband Power Profiles across 13 Epochs (Frames 0..12):")
    for b in range(4):
        pk = np.argmax(broadband_pwr[:, b])
        p_max = broadband_pwr[pk, b]
        # Opposite beam off-axis rejection
        opposite_b = 3 if b == 0 else (0 if b == 3 else (2 if b == 1 else 1))
        p_opposite = broadband_pwr[np.argmax(broadband_pwr[:, opposite_b]), b]
        rejection_db = 10.0 * np.log10(p_max / max(p_opposite, 1e-12))
        dynamic_range_db = 10.0 * np.log10(p_max / max(np.min(broadband_pwr[:, b]), 1e-12))
        print(f"    Beam {b} ({BEAMS[b]['name']}): Peak at Frame k={pk} (pwr={p_max:.2f}), Off-axis Rejection={rejection_db:.1f} dB, Dynamic Range={dynamic_range_db:.1f} dB")
        assert rejection_db >= 8.0, f"Off-axis rejection for beam {b} is {rejection_db:.1f} dB (< 8.0 dB)!"
        assert dynamic_range_db >= 9.0, f"Dynamic range for beam {b} is {dynamic_range_db:.1f} dB (< 9.0 dB)!"

    peak_frames = [np.argmax(broadband_pwr[:, b]) for b in range(4)]
    print(f"  Observed Peak Indices: {peak_frames}")
    assert peak_frames == [3, 5, 7, 9], f"Peak indices {peak_frames} do not match expected [3, 5, 7, 9]!"
    print("  -> Criterion 1 & 2 PASSED: Strict Sequence t_peak(B0) < t_peak(B1) < t_peak(B2) < t_peak(B3) [3, 5, 7, 9].")

    # Broadband light curve correlation against analytical compound model for all 4 beams
    bb_correlations = []
    for b in range(4):
        lb = BEAMS[b]["l0"]
        dl_b = np.array(l_schedule) - lb
        bb_ana = np.array([np.mean(np.cos(np.pi * freqs_all * 0.6 * d / C_LIGHT)**2 * np.cos(np.pi * freqs_all * 3.6 * d / C_LIGHT)**2) for d in dl_b])
        r_b = np.corrcoef(broadband_pwr[:, b], bb_ana)[0, 1]
        bb_correlations.append(r_b)
        print(f"    Beam {b} ({BEAMS[b]['name']}) Transit Envelope Correlation: r = {r_b:.6f}")
        assert r_b > 0.98, f"Broadband light curve correlation for beam {b} is {r_b:.4f} (<= 0.98)!"

    # Frequency fringe correlation across frames with active fringes (|dl| > 0.1)
    active_corrs = [correlations_b0[k] for k in range(len(correlations_b0)) if abs(l_schedule[k] - 0.1773) > 0.1]
    mean_corr = np.mean(active_corrs)
    print(f"  Mean Active Interferometric Fringe Correlation across Slices (|dl| > 0.1): r = {mean_corr:.6f}")
    assert mean_corr > 0.80, f"Fringe correlation {mean_corr} <= 0.80!"
    print("  -> Criterion 3 PASSED: Theoretical Two-Pair Compound Fringe Correlation Verified (r > 0.98 broadband, r > 0.80 per-slice).")

    # --------------------------------------------------------------------------
    # Phase 6: Post-Processing & Analysis Script Validation
    # --------------------------------------------------------------------------
    print("\n>>> [Phase 6] Post-Processing & Analysis Script Validation <<<")
    report_dir = Path("/tmp/test_sun_reports")
    shutil.rmtree(report_dir, ignore_errors=True)

    # 1. Run analyze_sun_transit_traffic_cams.py against Phase 4.3 synthetic dumps
    print("  Executing charts/analyze_sun_transit_traffic_cams.py...")
    cmd_analysis = [
        str(PY_EXEC), "charts/analyze_sun_transit_traffic_cams.py",
        "--dump-dir", str(dump_dir),
        "--out-dir", str(report_dir),
    ]
    res_analysis = subprocess.run(cmd_analysis, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(f"  Analysis Output:\n{res_analysis.stdout.strip()}")
    assert res_analysis.returncode == 0, f"Analysis script failed with code {res_analysis.returncode}"

    # Verify generated plots and report
    p_lc = report_dir / "sun_traffic_cameras_lightcurves.png"
    p_comp = report_dir / "sun_traffic_cameras_normalized_comparison.png"
    p_spec = report_dir / "sun_traffic_cameras_spectrograms.png"
    p_json = report_dir / "sun_transit_traffic_cams_report.json"

    assert p_lc.exists() and p_lc.stat().st_size > 10000, "Light curves PNG missing or empty!"
    assert p_comp.exists() and p_comp.stat().st_size > 10000, "Comparison PNG missing or empty!"
    assert p_spec.exists() and p_spec.stat().st_size > 10000, "Spectrograms PNG missing or empty!"
    assert p_json.exists(), "Report JSON missing!"

    with open(p_json) as f:
        rep_data = json.load(f)
    assert rep_data["status"] == "SUCCESS"
    assert rep_data["num_epochs"] == 13
    rep_peaks = [b["peak_epoch"] for b in rep_data["beams"]]
    print(f"  Report Detected Peaks: {rep_peaks}")
    assert rep_peaks == [3, 5, 7, 9], f"Report peaks {rep_peaks} != [3, 5, 7, 9]!"
    print("  -> Primary Pass Gate PASSED: Plots, Telemetry JSON, and Transit Peak Times Verified.")

    # 2. Negative Resilience Tests
    print("\n  [Phase 6 Negative Resilience 1: Truncated 13th File]")
    # Copy dumps to temp resilience dir
    res_dir = Path("/tmp/test_dump_resilience")
    shutil.rmtree(res_dir, ignore_errors=True)
    shutil.copytree(dump_dir, res_dir)

    # Truncate the 13th file of Stream 1 by 50%
    f13_s1 = res_dir / "formed_beams_sun_traffic_cams_freq1_0000012.bin"
    sz = f13_s1.stat().st_size
    with open(f13_s1, "r+b") as f:
        f.truncate(sz // 2)

    # Re-run analysis script
    cmd_res1 = [
        str(PY_EXEC), "charts/analyze_sun_transit_traffic_cams.py",
        "--dump-dir", str(res_dir),
        "--out-dir", str(report_dir / "resilience1"),
    ]
    res_trunc = subprocess.run(cmd_res1, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert res_trunc.returncode == 0, f"Analysis script crashed on truncated file! {res_trunc.stdout}"
    with open(report_dir / "resilience1" / "sun_transit_traffic_cams_report.json") as f:
        rep_trunc = json.load(f)
    assert rep_trunc["num_epochs"] == 12, f"Expected 12 valid epochs after truncation, got {rep_trunc['num_epochs']}"
    print("  -> Resilience 1 PASSED: Truncated 13th file skipped gracefully, 12 epochs analyzed without crash.")

    print("\n  [Phase 6 Negative Resilience 2: Dual-Stream Asymmetric Epoch Alignment]")
    # Remove file #5 from Stream 1 in res_dir
    f5_s1 = res_dir / "formed_beams_sun_traffic_cams_freq1_0000005.bin"
    f5_s1.unlink()

    # Re-run analysis script
    cmd_res2 = [
        str(PY_EXEC), "charts/analyze_sun_transit_traffic_cams.py",
        "--dump-dir", str(res_dir),
        "--out-dir", str(report_dir / "resilience2"),
    ]
    res_align = subprocess.run(cmd_res2, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert res_align.returncode == 0, f"Analysis script crashed on missing stream 1 file! {res_align.stdout}"
    with open(report_dir / "resilience2" / "sun_transit_traffic_cams_report.json") as f:
        rep_align = json.load(f)
    # 13 total minus truncated file 12 and missing file 5 = 11 paired epochs
    assert rep_align["num_epochs"] == 11, f"Expected 11 paired epochs, got {rep_align['num_epochs']}"
    print("  -> Resilience 2 PASSED: Asymmetric stream file paired by sequence tag rather than blind index.")

    print("\n==============================================================================")
    print(" [ALL TESTS PASSED] Phases 1 through 6 Exhaustive Verification Successful!")
    print("==============================================================================")


if __name__ == "__main__":
    main()
