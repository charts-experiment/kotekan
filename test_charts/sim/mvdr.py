"""Kotekan config for the correlation-domain MVDR stage (chartsMVDRBeams) on a replayed window.

Pipeline: rawFileRead (baseband frames) -> cudaProcess (cudaShuffleAstron +
cudaCorrelatorAstron) -> chartsMVDRBeams -> rawFileWrite (beam power, weights), plus a
rawFileWrite of the correlation frames so that the stage output can be compared with the
NumPy reference (charts-mvdr, charts_mvdr/stage_reference.py) on exactly the same frames.

Outputs: output_dir/corr/corr_%07d.bin (one correlation frame per file) and, in
output_dir/mvdr, mvdr_power_%07d.bin (float32 [2][num_local_freq][num_beams], DFT and MVDR,
one per frame) and mvdr_weights_%07d.bin (one per weight update), each preceded by the
rawFileWrite metadata header (uint32 size + metadata). Separate directories keep the
correlation dumps readable by tools that take every <prefix>_%07d.bin in a directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


def create_mvdr_yaml(
    yaml_path: Path,
    baseband_dir: Path,
    baseband_name: str,
    output_dir: Path,
    num_frames: int,
    positions_x_m: Sequence[float],
    positions_y_m: Sequence[float],
    beams: Sequence[Tuple[float, float]],
    freq_start_mhz: float = 300.0,
    delta_freq_mhz: float = 0.3,
    num_elements: int = 64,
    num_local_freq: int = 336,
    samples_per_data_set: int = 1536,
    cov_half_window: int = 10,
    cov_guard: int = 1,
    update_every: int = 1,
    loading: float = 10.0,
    loading_ref: str = "noise",
    gains_file: Optional[Path] = None,
    write_correlation: bool = True,
    buffer_depth: int = 2,
    cpu_cores: Optional[List[int]] = None,
    cuda_device: int = 0,
) -> Path:
    """Write the YAML config and return its path.

    The correlation buffer holds 2 * cov_half_window + 4 frames (the stage keeps
    2 * cov_half_window + 1 frames). flush_after_frames = num_frames, so every frame of
    the window gets an output (the last cov_half_window with truncated windows).
    """
    if len(positions_x_m) != num_elements or len(positions_y_m) != num_elements:
        raise ValueError("positions must have num_elements values")
    if not beams:
        raise ValueError("need at least one beam")
    yaml_path = Path(yaml_path)
    output_dir = Path(output_dir)
    corr_dir = output_dir / "corr"
    mvdr_dir = output_dir / "mvdr"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    corr_dir.mkdir(parents=True, exist_ok=True)
    mvdr_dir.mkdir(parents=True, exist_ok=True)

    block_size = 2
    num_blocks = (num_elements // block_size) * (num_elements // block_size + 1) // 2
    num_beams = len(beams)
    corr_frames = 2 * cov_half_window + 4
    weights_frame_size = (
        16
        + num_local_freq * num_beams * num_elements * 8
        + num_local_freq * num_beams * 4
    )
    cores = cpu_cores or [0, 1, 2, 3]
    xs = json.dumps([float(v) for v in positions_x_m])
    ys = json.dumps([float(v) for v in positions_y_m])
    bs = json.dumps([[float(l), float(m)] for l, m in beams])
    gains = f'\n  gains_file: "{Path(gains_file).as_posix()}"' if gains_file else ""

    corr_dump = ""
    if write_correlation:
        corr_dump = f"""
correlator_dump:
  kotekan_stage: rawFileWrite
  in_buf: host_correlation_buffer
  base_dir: "{corr_dir.as_posix()}"
  file_name: "corr"
  file_ext: "bin"
  num_frames_per_file: 1
  prefix_hostname: false
  skip_zero_frames: false
"""

    content = f"""######################################################################
# CHARTS replay: Astron correlator + chartsMVDRBeams (correlation-domain MVDR)
# Source: {Path(baseband_dir) / baseband_name}_%07d.bin ({num_frames} frames)
######################################################################
type: config
log_level: info

cpu_affinity: {cores}

num_elements: {num_elements}
num_local_freq: {num_local_freq}
samples_per_data_set: {samples_per_data_set}
num_data_sets: 1
block_size: {block_size}
num_blocks: {num_blocks}
elements_per_thread_block: 32
sizeof_int: 4
buffer_depth: {buffer_depth}

main_pool:
  kotekan_metadata_pool: chordMetadata
  num_metadata_objects: {2 * corr_frames + 20}

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
  num_frames: {corr_frames}
  frame_size: num_local_freq * num_blocks * (block_size * block_size) * 2 * num_data_sets * sizeof_int
  numa_node: 0
  metadata_pool: main_pool
  zero_new_frames: true
  mlock_frames: false

mvdr_power_buffer:
  kotekan_buffer: standard
  num_frames: 4
  frame_size: {2 * num_local_freq * num_beams * 4}
  numa_node: 0
  metadata_pool: main_pool
  zero_new_frames: true
  mlock_frames: false

mvdr_weights_buffer:
  kotekan_buffer: standard
  num_frames: 4
  frame_size: {weights_frame_size}
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
      num_elements: {num_elements}
      num_local_freq: {num_local_freq}
      samples_per_data_set: {samples_per_data_set}
      num_data_sets: 1
      block_size: {block_size}
      num_blocks: {num_blocks}
      buffer_depth: {buffer_depth}
    - name: cudaCorrelatorAstron
      gpu_mem_voltage: ordered_voltage
      gpu_mem_correlation_matrix: correlation_matrix
      num_elements: {num_elements}
      num_local_freq: {num_local_freq}
      samples_per_data_set: {samples_per_data_set}
      num_data_sets: 1
      block_size: {block_size}
      num_blocks: {num_blocks}
      elements_per_thread_block: 32
      buffer_depth: {buffer_depth}
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

mvdr_beams:
  kotekan_stage: chartsMVDRBeams
  in_buf: host_correlation_buffer
  out_buf: mvdr_power_buffer
  weights_buf: mvdr_weights_buffer
  samples_per_frame: {samples_per_data_set}
  freq_start_mhz: {float(freq_start_mhz)!r}
  delta_freq_mhz: {float(delta_freq_mhz)!r}
  positions_x_m: {xs}
  positions_y_m: {ys}
  beams: {bs}
  cov_half_window: {cov_half_window}
  cov_guard: {cov_guard}
  update_every: {update_every}
  loading: {float(loading)!r}
  loading_ref: {loading_ref}
  flush_after_frames: {num_frames}{gains}

mvdr_power_dump:
  kotekan_stage: rawFileWrite
  in_buf: mvdr_power_buffer
  base_dir: "{mvdr_dir.as_posix()}"
  file_name: "mvdr_power"
  file_ext: "bin"
  num_frames_per_file: 1
  prefix_hostname: false
  skip_zero_frames: false

mvdr_weights_dump:
  kotekan_stage: rawFileWrite
  in_buf: mvdr_weights_buffer
  base_dir: "{mvdr_dir.as_posix()}"
  file_name: "mvdr_weights"
  file_ext: "bin"
  num_frames_per_file: 1
  prefix_hostname: false
  skip_zero_frames: false
{corr_dump}"""
    yaml_path.write_text(content, encoding="utf-8")
    return yaml_path
