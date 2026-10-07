#!/usr/bin/env python3
"""CHARTS Kotekan Pipeline Orchestrator.

Manages automated generation of Kotekan YAML configurations and execution of:
  - Tensor Core Correlator replay (cudaShuffleAstron + cudaCorrelatorAstron)
  - GPU Direct Beam Tracker replay (cudaAntennaMask + cudaDirectBeamTrackerCommand)
  - Native in-kotekan simulation (chartsFEngineSim)
Supports both PC (local desktop/workstation) and Cluster (Trillium / Slurm) environments.
"""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from .astro import resolve_beam_targets
from .constants import (
    CHARTS_LATITUDE_DEG,
    CHARTS_LONGITUDE_DEG,
    DEFAULT_SPACING_M,
)
from .presets import SimulationConfig, find_kotekan_binary

logger = logging.getLogger("kotekan.charts.sim.pipeline")


def parse_beam_targets(
    targets_str: Optional[str] = None,
    max_beams: int = 8,
    default_ra: float = 83.633,
    default_dec: float = 22.014,
    default_lst: float = 5.575,
    default_name: str = "Primary Target",
    obs_time: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Parses beam target definitions using astronomical catalog and dynamic visibility."""
    return resolve_beam_targets(
        targets_spec=targets_str,
        obs_time=obs_time or default_lst,
        max_beams=max_beams,
        lat_deg=CHARTS_LATITUDE_DEG,
        lon_deg=CHARTS_LONGITUDE_DEG,
    )


def create_correlator_yaml(
    yaml_path: Path,
    baseband_dir: Path,
    baseband_name: str,
    correlator_dir: Path,
    num_frames: int,
    num_elements: int = 64,
    num_local_freq: int = 336,
    samples_per_data_set: int = 1536,
    buffer_depth: int = 2,
    cpu_cores: Optional[List[int]] = None,
    cuda_device: int = 0,
) -> Path:
    """Generates Kotekan YAML for rawFileRead -> cudaProcess (correlator) -> rawFileWrite."""
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    correlator_dir.mkdir(parents=True, exist_ok=True)

    block_size = 2
    num_blocks = (num_elements // block_size) * (num_elements // block_size + 1) // 2
    elements_per_thread_block = 32
    cores = cpu_cores or [0, 1, 2, 3]
    cores_str = str(cores)

    content = f"""######################################################################
# CHARTS Replay & Astron Correlator Pipeline (cudaProcess)
# Source: {baseband_dir / baseband_name}_%07d.bin ({num_frames} frames)
######################################################################
type: config
log_level: info

cpu_affinity: {cores_str}

num_elements: {num_elements}
num_local_freq: {num_local_freq}
samples_per_data_set: {samples_per_data_set}
num_data_sets: 1
block_size: {block_size}
num_blocks: {num_blocks}
elements_per_thread_block: {elements_per_thread_block}
sizeof_int: 4
buffer_depth: {buffer_depth}

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

host_correlation_buffer:
  kotekan_buffer: standard
  num_frames: buffer_depth
  frame_size: num_local_freq * num_blocks * (block_size * block_size) * 2 * num_data_sets * sizeof_int
  numa_node: 0
  metadata_pool: main_pool
  zero_new_frames: true
  mlock_frames: false

baseband_reader:
  kotekan_stage: rawFileRead
  out_buf: network_capture_buf
  prefix: "{(baseband_dir / baseband_name).as_posix()}_"
  suffix: ".bin"
  format_length: 7
  start_index: 0
  max_index: {num_frames - 1}
  playback_rate: 0
  end_interrupt: true

gpu:
  profiling: false
  kernel_path: "lib/cuda/kernels"
  commands: &command_list
    - name: cudaInputData
      in_buf: host_voltage
      gpu_mem: voltage
    - name: cudaSyncInput
    - name: cudaShuffleAstron
      gpu_mem_voltage: voltage
      gpu_mem_ordered_voltage: ordered_voltage
    - name: cudaCorrelatorAstron
      gpu_mem_voltage: ordered_voltage
      gpu_mem_correlation_matrix: correlation_matrix
    - name: cudaSyncOutput
    - name: cudaOutputData
      in_buf: host_voltage
      gpu_mem: correlation_matrix
      out_buf: host_correlation
  gpu_{cuda_device}:
    kotekan_stage: cudaProcess
    gpu_id: {cuda_device}
    buffer_depth: {buffer_depth}
    commands: *command_list
    in_buffers:
      host_voltage: network_capture_buf
    out_buffers:
      host_correlation: host_correlation_buffer

correlator_dump:
  kotekan_stage: rawFileWrite
  in_buf: host_correlation_buffer
  prefix: "{(correlator_dir / 'corr_').as_posix()}"
  suffix: ".bin"
  format_length: 7
  start_index: 0
  dump_metadata: false
"""
    yaml_path.write_text(content, encoding="utf-8")
    return yaml_path


def create_beam_tracker_yaml(
    yaml_path: Path,
    baseband_dir: Path,
    baseband_name: str,
    tracker_dir: Path,
    tracker_name: str,
    num_frames: int,
    beam_targets: List[Dict[str, Any]],
    num_elements: int = 64,
    num_local_freq: int = 336,
    samples_per_data_set: int = 1536,
    max_beams: int = 8,
    integration_spectra: int = 320,
    buffer_depth: int = 2,
    cpu_cores: Optional[List[int]] = None,
    stage_type: str = "direct",
    enable_mask: bool = True,
    enable_bislc: bool = False,
    enable_transient_trigger: bool = False,
    cuda_device: int = 0,
    freq_start_hz: float = 300.0e6,
    freq_step_hz: float = 300.0e3,
) -> Path:
    """Generates Kotekan YAML for rawFileRead -> cudaProcess (beam tracker) -> rawFileWrite."""
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    tracker_dir.mkdir(parents=True, exist_ok=True)

    cores = cpu_cores or [0, 1, 2, 3]
    cores_str = str(cores)
    initial_active_beams = len(beam_targets)

    # Format beam definitions depending on stage_type
    command_sections = []

    # 1. Input copy
    command_sections.append("""    - name: cudaInputData
      in_buf: host_voltage
      gpu_mem: voltage
    - name: cudaSyncInput""")

    # 2. Antenna mask (optional)
    if enable_mask:
        command_sections.append(f"""    - name: cudaAntennaMaskCommand
      gpu_mem_voltage: voltage
      num_elements: {num_elements}
      num_local_freq: {num_local_freq}
      samples_per_data_set: {samples_per_data_set}
      buffer_depth: {buffer_depth}
      auto_detect_enabled: true""")

    # 3. Direct Beam Tracker (cudaDirectBeamTrackerCommand)
    # Beam Tracker V5 is legacy and unsupported in CHARTS.
    if stage_type != "direct":
        raise ValueError(
            f"Unsupported stage_type='{stage_type}'. Beam Tracker V5 is legacy and unsupported; "
            "CHARTS exclusively uses the Direct Beam Tracker ('direct' / cudaDirectBeamTrackerCommand)."
        )

    target_lines = []
    for idx, t in enumerate(beam_targets):
        ra = t["ra_deg"]
        dec = t["dec_deg"]
        lst = t["lst_hours"]
        name = t.get("name", f"Beam {idx}")
        prefix = "" if idx == 0 else f"_{idx}"
        target_lines.append(f"      source_ra_deg{prefix}: {ra:.4f}  # Target: {name}")
        target_lines.append(f"      source_dec_deg{prefix}: {dec:.4f}")
        target_lines.append(f"      initial_lst_hours{prefix}: {lst:.4f}")
    targets_str = "\n".join(target_lines)

    command_sections.append(f"""    - name: cudaDirectBeamTrackerCommand
      gpu_mem_voltage: voltage
      gpu_mem_formed_beams: formed_beams
      num_elements: {num_elements}
      num_local_freq: {num_local_freq}
      samples_per_data_set: {samples_per_data_set}
      spacing_m: {DEFAULT_SPACING_M}
      max_beams: {max_beams}
      initial_active_beams: {initial_active_beams}
      buffer_depth: {buffer_depth}
      time_chunk_size: 256
      time_unroll: 4
      beam_tile_size: 4
      site_lat_deg: {CHARTS_LATITUDE_DEG}
      site_lon_deg: {CHARTS_LONGITUDE_DEG}
      freq_start_hz: {freq_start_hz:.1f}
      freq_step_hz: {freq_step_hz:.1f}
{targets_str}""")

    # 4. BiSLC (optional)
    if enable_bislc:
        command_sections.append(f"""    - name: cudaBiSLCCommand
      gpu_mem_formed_beams: formed_beams
      gpu_mem_cleaned_beams: formed_beams
      num_elements: {num_elements}
      num_local_freq: {num_local_freq}
      samples_per_data_set: {samples_per_data_set}
      spacing_m: {DEFAULT_SPACING_M}
      max_beams: {max_beams}
      initial_active_beams: {initial_active_beams}
      buffer_depth: {buffer_depth}
      diagonal_loading: 1.0e-4
      enabled: true
      freq_start_hz: {freq_start_hz:.1f}
      freq_step_hz: {freq_step_hz:.1f}""")

    # 5. Transient trigger (optional)
    if enable_transient_trigger:
        cand_dir = (tracker_dir / "transients").as_posix()
        command_sections.append(f"""    - name: cudaTransientTriggerCommand
      gpu_mem_cleaned_beams: formed_beams
      num_local_freq: {num_local_freq}
      samples_per_data_set: {samples_per_data_set}
      max_beams: {max_beams}
      buffer_depth: {buffer_depth}
      ring_buffer_depth: 16
      pre_trigger_frames: 4
      post_trigger_frames: 4
      sk_threshold: 0.08
      rfi_threshold: 0.30
      min_flagged_channels: 8
      dump_directory: "{cand_dir}"
      auto_dump_enabled: true
      enabled: true""")

    # 6. Output transfer
    command_sections.append("""    - name: cudaSyncOutput
    - name: cudaOutputData
      in_buf: host_voltage
      gpu_mem: formed_beams
      out_buf: host_formed_beams""")

    all_commands_str = "\n".join(command_sections)

    content = f"""######################################################################
# CHARTS Replay & Direct Beam Tracker Pipeline ({max_beams} Beams, cudaProcess)
# Source: {baseband_dir / baseband_name}_%07d.bin ({num_frames} frames)
######################################################################
type: config
log_level: info

cpu_affinity: {cores_str}

num_elements: {num_elements}
num_local_freq: {num_local_freq}
samples_per_data_set: {samples_per_data_set}
integration_spectra: {integration_spectra}
spacing_m: {DEFAULT_SPACING_M}
max_beams: {max_beams}
initial_active_beams: {initial_active_beams}
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

baseband_reader:
  kotekan_stage: rawFileRead
  out_buf: network_capture_buf
  prefix: "{(baseband_dir / baseband_name).as_posix()}_"
  suffix: ".bin"
  format_length: 7
  start_index: 0
  max_index: {num_frames - 1}
  playback_rate: 0
  end_interrupt: true

gpu:
  profiling: false
  kernel_path: "lib/cuda/kernels"
  commands: &command_list
{all_commands_str}
  gpu_{cuda_device}:
    kotekan_stage: cudaProcess
    gpu_id: {cuda_device}
    buffer_depth: {buffer_depth}
    commands: *command_list
    in_buffers:
      host_voltage: network_capture_buf
    out_buffers:
      host_formed_beams: host_formed_beams_buffer

tracker_dump:
  kotekan_stage: rawFileWrite
  in_buf: host_formed_beams_buffer
  prefix: "{(tracker_dir / tracker_name).as_posix()}_"
  suffix: ".bin"
  format_length: 7
  start_index: 0
  dump_metadata: false
"""
    yaml_path.write_text(content, encoding="utf-8")
    return yaml_path


def create_accumulate_yaml(
    yaml_path: Path,
    baseband_dir: Path,
    baseband_name: str,
    output_dir: Path,
    num_frames: int,
    num_elements: int = 64,
    num_local_freq: int = 336,
    samples_per_data_set: int = 1536,
    num_frames_to_accumulate: int = 10,
    buffer_depth: int = 2,
    cpu_cores: Optional[List[int]] = None,
    cuda_device: int = 0,
) -> Path:
    """Generates Kotekan YAML for rawFileRead -> correlator cudaProcess -> chartsAccumulate -> rawFileWrite."""
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    block_size = 2
    num_blocks = (num_elements // block_size) * (num_elements // block_size + 1) // 2
    elements_per_thread_block = 32
    cores = cpu_cores or [0, 1, 2, 3]
    cores_str = str(cores)

    content = f"""######################################################################
# CHARTS Replay & Astron Correlator Pipeline with chartsAccumulate
# Source: {baseband_dir / baseband_name}_%07d.bin ({num_frames} frames)
######################################################################
type: config
log_level: info

cpu_affinity: {cores_str}

num_elements: {num_elements}
num_local_freq: {num_local_freq}
samples_per_data_set: {samples_per_data_set}
num_data_sets: 1
block_size: {block_size}
num_blocks: {num_blocks}
elements_per_thread_block: {elements_per_thread_block}
sizeof_int: 4
buffer_depth: {buffer_depth}

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

host_correlation_buffer:
  kotekan_buffer: standard
  num_frames: buffer_depth
  frame_size: num_local_freq * num_blocks * (block_size * block_size) * 2 * num_data_sets * sizeof_int
  numa_node: 0
  metadata_pool: main_pool
  zero_new_frames: true
  mlock_frames: false

integrated_correlation_buffer:
  kotekan_buffer: standard
  num_frames: buffer_depth
  frame_size: num_local_freq * num_blocks * (block_size * block_size) * 2 * num_data_sets * sizeof_int
  numa_node: 0
  metadata_pool: main_pool
  zero_new_frames: true
  mlock_frames: false

baseband_reader:
  kotekan_stage: rawFileRead
  out_buf: network_capture_buf
  prefix: "{(baseband_dir / baseband_name).as_posix()}_"
  suffix: ".bin"
  format_length: 7
  start_index: 0
  max_index: {num_frames - 1}
  playback_rate: 0
  end_interrupt: true

gpu:
  profiling: false
  kernel_path: "lib/cuda/kernels"
  commands: &command_list
    - name: cudaInputData
      in_buf: host_voltage
      gpu_mem: voltage
    - name: cudaSyncInput
    - name: cudaShuffleAstron
      gpu_mem_voltage: voltage
      gpu_mem_ordered_voltage: ordered_voltage
    - name: cudaCorrelatorAstron
      gpu_mem_voltage: ordered_voltage
      gpu_mem_correlation_matrix: correlation_matrix
    - name: cudaSyncOutput
    - name: cudaOutputData
      in_buf: host_voltage
      gpu_mem: correlation_matrix
      out_buf: host_correlation
  gpu_{cuda_device}:
    kotekan_stage: cudaProcess
    gpu_id: {cuda_device}
    buffer_depth: {buffer_depth}
    commands: *command_list
    in_buffers:
      host_voltage: network_capture_buf
    out_buffers:
      host_correlation: host_correlation_buffer

charts_accumulate:
  kotekan_stage: chartsAccumulate
  in_buf: host_correlation_buffer
  out_buf: integrated_correlation_buffer
  num_frames_to_accumulate: {num_frames_to_accumulate}

accumulate_dump:
  kotekan_stage: rawFileWrite
  in_buf: integrated_correlation_buffer
  prefix: "{(output_dir / 'corr_accum_').as_posix()}"
  suffix: ".bin"
  format_length: 7
  start_index: 0
  dump_metadata: false
"""
    yaml_path.write_text(content, encoding="utf-8")
    return yaml_path


def execute_kotekan(
    config_path: Path,
    kotekan_bin: Optional[Path] = None,
    log_path: Optional[Path] = None,
    dry_run: bool = False,
    timeout_s: Optional[float] = 1800.0,
) -> int:
    """Executes Kotekan pipeline using specified configuration."""
    bin_path = kotekan_bin or find_kotekan_binary()

    if dry_run or bin_path is None or not bin_path.is_file():
        if dry_run:
            print(f"[DRY-RUN] Would execute: {bin_path or 'kotekan'} -c {config_path}")
            return 0
        raise FileNotFoundError(
            f"Kotekan binary not found at '{bin_path}'. "
            "Please build Kotekan with CMake or pass --kotekan-bin <path>."
        )

    repo_root = config_path.resolve().parents[2]
    n2k_lib_dir = repo_root / "build" / "external" / "n2k"

    env = os.environ.copy()
    ld_paths = [str(n2k_lib_dir)]
    if "LD_LIBRARY_PATH" in env:
        ld_paths.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = ":".join(ld_paths)

    cmd = [str(bin_path), "-c", str(config_path)]
    print(f"\n>>> Executing Kotekan: {' '.join(cmd)}")

    t0 = time.perf_counter()
    log_file = None
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "w", encoding="utf-8")

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=log_file or subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            bufsize=1,
        )

        if not log_file:
            for line in proc.stdout:
                print(f"  [kotekan] {line.rstrip()}")

        proc.wait(timeout=timeout_s)
        rc = proc.returncode
    finally:
        if log_file:
            log_file.close()

    elapsed = time.perf_counter() - t0
    print(f">>> Kotekan finished in {elapsed:.2f} s with return code {rc}")
    return rc
