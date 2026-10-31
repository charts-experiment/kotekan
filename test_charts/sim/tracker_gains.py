"""Kotekan config: replay a window through cudaDirectBeamTrackerCommand, optionally calibrated.

Pipeline: rawFileRead -> cudaProcess (cudaInputData, cudaDirectBeamTrackerCommand,
cudaOutputData) -> rawFileWrite of the formed beams (complex float32, frame size
samples_per_data_set * num_local_freq * max_beams * 8 bytes, one file per frame).

Beams are fixed directions (source_l0 / source_m0 keys). gains_file is a charts-utils
complex .npy (n_channels, n_antennas), convention v_obs = g v; the tracker uses the
corrector 1 / g per element. No antenna-mask stage (its automatic detection could mask
different antennas in runs that are compared). Kotekan exits after num_frames output files
(the reader does not end the pipeline), so every frame is on disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

CHARTS_SPACING_M = 0.6


def create_tracker_gains_yaml(
    yaml_path: Path,
    baseband_dir: Path,
    baseband_name: str,
    output_dir: Path,
    output_name: str,
    num_frames: int,
    beams: Sequence[Tuple[float, float]],
    gains_file: Optional[Path] = None,
    gains_first_channel: Optional[int] = None,
    num_elements: int = 64,
    num_local_freq: int = 336,
    samples_per_data_set: int = 15360,
    freq_start_hz: float = 300.0e6,
    freq_step_hz: float = 300.0e3,
    antenna_order: str = "descending",
    buffer_depth: int = 2,
    cpu_cores: Optional[List[int]] = None,
    cuda_device: int = 0,
) -> Path:
    """Write the YAML config and return its path."""
    if not beams:
        raise ValueError("need at least one beam")
    if antenna_order not in ("descending", "ascending"):
        raise ValueError("antenna_order must be descending or ascending")
    yaml_path, output_dir = Path(yaml_path), Path(output_dir)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    max_beams = len(beams)
    cores = cpu_cores or [0, 1, 2, 3]

    beam_lines = []
    for i, (l, m) in enumerate(beams):
        sfx = "" if i == 0 else f"_{i}"
        beam_lines.append(f"      source_l0{sfx}: {float(l)!r}")
        beam_lines.append(f"      source_m0{sfx}: {float(m)!r}")
    beam_str = "\n".join(beam_lines)
    cal = ""
    if gains_file is not None:
        cal = f'\n      gains_file: "{Path(gains_file).as_posix()}"'
        if gains_first_channel is not None:
            cal += f"\n      gains_first_channel: {int(gains_first_channel)}"

    content = f"""######################################################################
# CHARTS replay: Direct Beam Tracker ({max_beams} fixed beams){' with calibration' if cal else ''}
# Source: {Path(baseband_dir) / baseband_name}_%07d.bin ({num_frames} frames)
######################################################################
type: config
log_level: info

cpu_affinity: {cores}

num_elements: {num_elements}
num_local_freq: {num_local_freq}
samples_per_data_set: {samples_per_data_set}
max_beams: {max_beams}
buffer_depth: {buffer_depth}
sizeof_complex_float: 8

main_pool:
  kotekan_metadata_pool: chartsMetadata
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
  buf: network_capture_buf
  base_dir: "{Path(baseband_dir).as_posix()}"
  file_name: "{baseband_name}"
  file_ext: "bin"
  prefix_hostname: false
  end_interrupt: false

gpu:
  profiling: false
  kernel_path: "lib/cuda/kernels"
  commands: &command_list
    - name: cudaInputData
      in_buf: host_voltage
      gpu_mem: voltage
    - name: cudaSyncInput
    - name: cudaDirectBeamTrackerCommand
      gpu_mem_voltage: voltage
      gpu_mem_formed_beams: formed_beams
      num_elements: {num_elements}
      num_local_freq: {num_local_freq}
      samples_per_data_set: {samples_per_data_set}
      spacing_m: {CHARTS_SPACING_M}
      max_beams: {max_beams}
      initial_active_beams: {max_beams}
      buffer_depth: {buffer_depth}
      time_chunk_size: 256
      time_unroll: 4
      beam_tile_size: 4
      antenna_order: {antenna_order}
      freq_start_hz: {float(freq_start_hz):.1f}
      freq_step_hz: {float(freq_step_hz):.1f}
{beam_str}{cal}
    - name: cudaSyncOutput
    - name: cudaOutputData
      in_buf: host_voltage
      gpu_mem: formed_beams
      out_buf: host_formed_beams
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
  base_dir: "{output_dir.as_posix()}"
  file_name: "{output_name}"
  file_ext: "bin"
  num_frames_per_file: 1
  prefix_hostname: false
  skip_zero_frames: false
  exit_after_n_files: {num_frames}
"""
    yaml_path.write_text(content, encoding="utf-8")
    return yaml_path
