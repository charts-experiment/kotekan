#!/usr/bin/env python3
"""
CHARTS 5-Minute Window Correlator Replay Runner
==============================================
Automates correlation replay and validation for realistic 5-minute baseband windows:
  1. Inspects window metadata HDF5 (<name>_meta.h5) to identify written frames
     and key event frames (FRB, pulsar, RFI).
  2. Generates a tailored Kotekan YAML configuration using:
       - rawFileRead with end_interrupt: true streaming baseband frames
       - cudaShuffleAstron + cudaCorrelatorAstron (John Romein's Tensor Core correlator)
       - rawFileWrite writing sequentially numbered correlation matrices (corr_%07d.bin)
  3. Executes Kotekan with LD_LIBRARY_PATH set for n2k Tensor Core libraries.
  4. Inspects correlation dumps via inspect_correlator_dump:
       - Reconstructs 64x64 Hermitian visibility matrix
       - Validates Hermitian symmetry (||V - V^H|| / ||V|| ~ 0.0)
       - Generates 2D heatmap plots (|V_ij| and arg(V_ij)) for baseline, event peak, and final frames.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np

# Setup paths
_test_charts_dir = Path(__file__).resolve().parent
_kotekan_root = _test_charts_dir.parent
if str(_test_charts_dir) not in sys.path:
    sys.path.insert(0, str(_test_charts_dir))

from inspect_correlator_dump import inspect_and_plot


def create_window_replay_yaml(
    yaml_path: Path,
    baseband_dir: Path,
    baseband_name: str,
    correlator_dir: Path,
    correlator_name: str,
    num_frames: int,
    samples_per_data_set: int = 1536,
    num_elements: int = 64,
    num_local_freq: int = 672,
    buffer_depth: int = 4,
):
    """Writes Kotekan YAML configuration for rawFileRead -> cudaCorrelatorAstron -> rawFileWrite."""
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    block_size = 2
    num_blocks = (
        (num_elements // block_size) * (num_elements // block_size + 1) // 2
    )  # 528 for 64 elements
    elements_per_thread_block = 32

    content = f"""######################################################################
# CHARTS 5-Minute Window Replay & Astron Correlator Pipeline
# Source: {baseband_dir / baseband_name}_%07d.bin ({num_frames} frames)
######################################################################
type: config
log_level: info

cpu_affinity: [0, 1, 2, 3]

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

# Input baseband buffer [time][freq][antenna] (int4x2_t)
network_capture_buf:
  kotekan_buffer: standard
  num_frames: buffer_depth
  frame_size: samples_per_data_set * num_local_freq * num_elements
  numa_node: 0
  metadata_pool: main_pool
  zero_new_frames: true
  mlock_frames: false

# Output correlation matrix buffer
host_correlation_buffer:
  kotekan_buffer: standard
  num_frames: buffer_depth
  frame_size: num_local_freq * num_blocks * (block_size * block_size) * 2 * num_data_sets * sizeof_int
  numa_node: 0
  metadata_pool: main_pool
  zero_new_frames: true
  mlock_frames: false

# Stage 1: Read simulated baseband frames from disk
replay_reader:
  kotekan_stage: rawFileRead
  base_dir: {baseband_dir.as_posix()}
  file_name: {baseband_name}
  file_ext: bin
  buf: network_capture_buf
  prefix_hostname: false
  end_interrupt: true

# Stage 2: GPU Tensor Core Correlator (John Romein's Kernel)
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
      buffer_depth: 3
    - name: cudaCorrelatorAstron
      gpu_mem_voltage: ordered_voltage
      gpu_mem_correlation_matrix: correlation_matrix
      num_elements: {num_elements}
      num_local_freq: {num_local_freq}
      samples_per_data_set: {samples_per_data_set}
      num_data_sets: 1
      block_size: {block_size}
      num_blocks: {num_blocks}
      elements_per_thread_block: {elements_per_thread_block}
      buffer_depth: 3
    - name: cudaSyncOutput
    - name: cudaOutputData
      in_buf: host_voltage
      gpu_mem: correlation_matrix
      out_buf: host_correlation
  gpu_0:
    kotekan_stage: cudaProcess
    gpu_id: 0
    buffer_depth: 3
    commands: *command_list
    in_buffers:
      host_voltage: network_capture_buf
    out_buffers:
      host_correlation: host_correlation_buffer

# Stage 3: Write correlation matrix dumps to disk
write_correlator_output:
  kotekan_stage: rawFileWrite
  in_buf: host_correlation_buffer
  base_dir: {correlator_dir.as_posix()}
  file_name: {correlator_name}
  file_ext: bin
  num_frames_per_file: 1
  exit_after_n_files: {num_frames}
  prefix_hostname: false
  skip_zero_frames: false
"""
    with open(yaml_path, "w") as f:
        f.write(content)


def auto_detect_window_info(
    window_dir: Path, base_name: Optional[str] = None
) -> Tuple[str, int, Optional[Path]]:
    """Auto-detects base name, frame count, and metadata file in window directory."""
    if base_name is None:
        meta_files = list(window_dir.glob("*_meta.h5"))
        if meta_files:
            meta_file = meta_files[0]
            name = meta_file.stem.replace("_meta", "")
        else:
            bin_files = sorted(window_dir.glob("*.bin"))
            if not bin_files:
                raise FileNotFoundError(
                    f"No .bin or _meta.h5 files found in {window_dir}"
                )
            # win15UTC_64ant_0000000.bin -> win15UTC_64ant
            first_bin = bin_files[0].stem
            name = "_".join(first_bin.split("_")[:-1])
            meta_file = None
    else:
        name = base_name
        meta_cand = window_dir / f"{name}_meta.h5"
        meta_file = meta_cand if meta_cand.exists() else None

    # Determine frame count
    if meta_file and meta_file.exists():
        with h5py.File(meta_file, "r") as hf:
            num_frames = int(hf.attrs.get("written_frames_count", 0))
            if num_frames == 0 and "written_out_index" in hf:
                num_frames = len(hf["written_out_index"])
    else:
        bin_files = list(window_dir.glob(f"{name}_*.bin"))
        num_frames = len(bin_files)

    return name, num_frames, meta_file


def select_key_inspection_frames(
    num_frames: int,
    meta_file: Optional[Path] = None,
    events_file: Optional[Path] = None,
) -> List[int]:
    """Selects representative frames (first background, event peak, final frame)."""
    selected = [0]

    # Try to locate an event frame from metadata
    event_out_idx = None
    if meta_file and meta_file.exists():
        try:
            with h5py.File(meta_file, "r") as hf:
                if "written_time_s" in hf and "lightcurve_power" in hf:
                    times = np.array(hf["written_time_s"])
                    powers = np.array(hf["written_mean_power"])
                    # Find frame with maximum power (transient peak)
                    if len(powers) > 0:
                        event_out_idx = int(np.argmax(powers))
        except Exception:
            pass

    if (
        event_out_idx is not None
        and event_out_idx not in selected
        and event_out_idx < num_frames
    ):
        selected.append(event_out_idx)
    else:
        mid_idx = num_frames // 2
        if mid_idx not in selected and mid_idx < num_frames:
            selected.append(mid_idx)

    last_idx = max(0, num_frames - 1)
    if last_idx not in selected:
        selected.append(last_idx)

    selected.sort()
    return selected


def run_window_correlator(
    window_dir: Path,
    file_name: Optional[str] = None,
    num_frames: Optional[int] = None,
    kotekan_bin: Optional[Path] = None,
    corr_dir: Optional[Path] = None,
    plots_dir: Optional[Path] = None,
    configs_dir: Optional[Path] = None,
    num_antennas: int = 64,
    num_freq: int = 672,
    samples_per_frame: int = 1536,
    inspect_frames_arg: str = "auto",
):
    """Executes Kotekan correlator replay on 5-minute window frames."""
    window_dir = window_dir.resolve()
    base_name, detected_frames, meta_file = auto_detect_window_info(
        window_dir, file_name
    )
    n_frames = num_frames or detected_frames

    if n_frames <= 0:
        raise ValueError(f"No frames found to correlate in {window_dir}")

    kotekan_executable = (
        kotekan_bin or (_kotekan_root / "build" / "kotekan" / "kotekan")
    ).resolve()
    if not kotekan_executable.exists():
        raise FileNotFoundError(f"Kotekan binary not found at: {kotekan_executable}")

    c_dir = (corr_dir or (window_dir / "correlator")).resolve()
    p_dir = (plots_dir or (window_dir / "plots")).resolve()
    cfg_dir = (configs_dir or (window_dir / "configs")).resolve()

    c_dir.mkdir(parents=True, exist_ok=True)
    p_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir.mkdir(parents=True, exist_ok=True)

    corr_name = f"corr_{base_name}"
    yaml_path = cfg_dir / f"config_corr_{base_name}.yaml"

    print("=" * 78)
    print(" CHARTS 5-MINUTE WINDOW CORRELATOR REPLAY RUNNER")
    print("=" * 78)
    print(f" Window Directory     : {window_dir}")
    print(f" Base Stream Name     : {base_name}_%07d.bin")
    print(f" Total Frames         : {n_frames} frames")
    print(f" Kotekan Binary       : {kotekan_executable}")
    print(f" Correlator Output Dir: {c_dir}")
    print(f" Plots Output Dir     : {p_dir}")
    print(f" YAML Configuration   : {yaml_path}")
    print("=" * 78)

    # 1. Generate YAML Config
    print("\n[1/3] Generating Kotekan Replay YAML Config...")
    create_window_replay_yaml(
        yaml_path=yaml_path,
        baseband_dir=window_dir,
        baseband_name=base_name,
        correlator_dir=c_dir,
        correlator_name=corr_name,
        num_frames=n_frames,
        samples_per_data_set=samples_per_frame,
        num_elements=num_antennas,
        num_local_freq=num_freq,
    )
    print(f"    Saved configuration: {yaml_path.name}")

    # 2. Setup Environment & Execute Kotekan
    print(f"\n[2/3] Executing Kotekan AstronCorrelator over {n_frames} frames...")
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = (
        f"{_kotekan_root / 'build' / 'external' / 'n2k'}:{env.get('LD_LIBRARY_PATH', '')}"
    )

    if "CUDA_HOME" not in env:
        for var in ["CUDA_ROOT", "EBROOTCUDA", "CUDA_PATH"]:
            if var in env:
                env["CUDA_HOME"] = env[var]
                break
        if "CUDA_HOME" not in env:
            import shutil

            nvcc_bin = shutil.which("nvcc")
            if nvcc_bin:
                env["CUDA_HOME"] = str(Path(nvcc_bin).resolve().parent.parent)

    if "CUDA_HOME" in env:
        cuda_inc = f"{env['CUDA_HOME']}/include"
        env["CUDA_PATH"] = env["CUDA_HOME"]
        env["CPATH"] = f"{cuda_inc}:{env.get('CPATH', '')}"

    cmd = [str(kotekan_executable), "--config", str(yaml_path)]
    t0 = time.perf_counter()
    res = subprocess.run(
        cmd, cwd=str(_kotekan_root), env=env, capture_output=True, text=True
    )
    t1 = time.perf_counter()

    if res.returncode != 0:
        print(f"[ERROR] Kotekan execution failed with exit code {res.returncode}!")
        print("STDERR:")
        print(res.stderr[-2000:] if len(res.stderr) > 2000 else res.stderr)
        print("STDOUT:")
        print(res.stdout[-2000:] if len(res.stdout) > 2000 else res.stdout)
        sys.exit(1)

    print(
        f"    Kotekan correlation completed in {(t1 - t0):.2f} s ({(t1 - t0) / max(1, n_frames):.3f} s/frame)"
    )

    # 3. Inspect Selected Frames and Generate Plots
    print("\n[3/3] Inspecting Visibilities & Generating Diagnostic Plots...")
    if inspect_frames_arg == "auto":
        frames_to_inspect = select_key_inspection_frames(n_frames, meta_file=meta_file)
    else:
        frames_to_inspect = [
            int(x.strip()) for x in inspect_frames_arg.split(",") if x.strip().isdigit()
        ]

    print(f"    Selected frames for inspection: {frames_to_inspect}")

    for f_idx in frames_to_inspect:
        corr_file = c_dir / f"{corr_name}_{f_idx:07d}.bin"
        if not corr_file.exists():
            print(
                f"    [WARNING] Correlation frame {corr_file.name} not found, skipping inspection."
            )
            continue

        plot_file = p_dir / f"{corr_name}_frame{f_idx:07d}_inspection.png"
        print(f"\n>>> Inspecting Frame {f_idx:07d} ({corr_file.name}) ...")
        inspect_and_plot(
            bin_path=corr_file,
            output_plot=plot_file,
            num_elements=num_antennas,
            num_channels=num_freq,
            freq_channel_idx=num_freq // 2,
        )
        if plot_file.exists():
            print(f"    Saved diagnostic plot: {plot_file.name}")

    print("\n" + "=" * 78)
    print(f" CORRELATION PIPELINE FOR [{base_name}] COMPLETED SUCCESSFULLY!")
    print(f" Output Correlator Files: {c_dir}")
    print(f" Output Diagnostic Plots: {p_dir}")
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(
        description="CHARTS 5-Minute Window Correlator Replay Runner"
    )
    parser.add_argument(
        "--window-dir",
        type=str,
        required=True,
        help="Directory containing window .bin and _meta.h5",
    )
    parser.add_argument(
        "--file-name", type=str, default=None, help="Base name of stream files"
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=None,
        help="Number of frames to correlate (default: auto)",
    )
    parser.add_argument(
        "--kotekan-bin", type=str, default=None, help="Path to kotekan binary"
    )
    parser.add_argument(
        "--corr-dir",
        type=str,
        default=None,
        help="Output directory for correlation dumps",
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default=None,
        help="Output directory for diagnostic plots",
    )
    parser.add_argument(
        "--configs-dir",
        type=str,
        default=None,
        help="Output directory for generated YAML configs",
    )
    parser.add_argument(
        "--antennas", type=int, default=64, help="Number of antennas (default: 64)"
    )
    parser.add_argument(
        "--num-freq", type=int, default=672, help="Frequency channels (default: 672)"
    )
    parser.add_argument(
        "--samples-per-frame",
        type=int,
        default=1536,
        help="Samples per frame (default: 1536)",
    )
    parser.add_argument(
        "--inspect-frames",
        type=str,
        default="auto",
        help="Comma-separated list of frame indices to inspect or 'auto'",
    )
    args = parser.parse_args()

    run_window_correlator(
        window_dir=Path(args.window_dir),
        file_name=args.file_name,
        num_frames=args.num_frames,
        kotekan_bin=Path(args.kotekan_bin) if args.kotekan_bin else None,
        corr_dir=Path(args.corr_dir) if args.corr_dir else None,
        plots_dir=Path(args.plots_dir) if args.plots_dir else None,
        configs_dir=Path(args.configs_dir) if args.configs_dir else None,
        num_antennas=args.antennas,
        num_freq=args.num_freq,
        samples_per_frame=args.samples_per_frame,
        inspect_frames_arg=args.inspect_frames,
    )


if __name__ == "__main__":
    main()
