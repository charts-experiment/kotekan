#!/usr/bin/env python3
"""
Full Pipeline Runner: CHARTS 64-Antenna F-Engine Baseband & AstronCorrelator
===========================================================================
Automates the full flow on Trillium:
  1. Generates 64-antenna baseband dumps (HDF5 & .bin) for all 5 scenarios:
       - vela_no_noise
       - vela_with_noise
       - sun_no_noise
       - sun_with_noise
       - sun_with_noise_saturated
  2. Runs Kotekan replay pipeline through cudaShuffleAstron + cudaCorrelatorAstron
     to compute the full 64x64 visibility matrix for each scenario.
  3. Inspects correlator outputs, validates Hermitian properties, and generates
     publication-ready 2D heatmap plots.
  4. Stores all baseband, correlation, and plot outputs in /project/def-vanderli/ferpb/.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Setup paths
_test_charts_dir = Path(__file__).resolve().parent
_kotekan_root = _test_charts_dir.parent
if str(_test_charts_dir) not in sys.path:
    sys.path.insert(0, str(_test_charts_dir))

from generate_charts_64ant_dumps import (
    run_all_scenarios,
    get_standard_scenarios_list,
    ASTRONOMICAL_CATALOG,
    parse_scenario_name,
)
from inspect_correlator_dump import inspect_and_plot


def create_scenario_yaml_config(
    yaml_path: Path,
    scenario: str,
    baseband_dir: Path | str,
    baseband_name: str,
    correlator_dir: Path | str,
    correlator_name: str,
    samples_per_data_set: int = 1536,
    num_elements: int = 64,
    num_local_freq: int = 336,
    num_frames: int = 1,
):
    """Writes unified Kotekan YAML config: chartsFEngineSim -> write_baseband + cudaCorrelatorAstron -> write_correlator."""
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    num_blocks = (
        (num_elements // 2) * (num_elements // 2 + 1) // 2
    )  # 528 for 64 elements
    target_key, use_noise, saturate = parse_scenario_name(scenario)

    content = f"""######################################################################
# Unified In-Kotekan Simulation & Correlation Pipeline
# Target: {scenario} (key: {target_key}, noise: {use_noise}, saturate: {saturate})
######################################################################
type: config
log_level: info

cpu_affinity: [0, 1, 2, 3]

num_elements: {num_elements}
num_local_freq: {num_local_freq}
samples_per_data_set: {samples_per_data_set}
num_data_sets: 1
block_size: 2
num_blocks: {num_blocks}
elements_per_thread_block: 32
sizeof_int: 4
buffer_depth: 4

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

# Stage 1: Native In-Process F-Engine Simulator
fengine_sim:
  kotekan_stage: chartsFEngineSim
  out_buf: network_capture_buf
  scenario: "{target_key}"
  use_noise: {str(use_noise).lower()}
  saturate: {str(saturate).lower()}
  num_elements: {num_elements}
  num_local_freq: {num_local_freq}
  samples_per_data_set: {samples_per_data_set}
  num_frames: {num_frames}
  freq_start_mhz: 300.0
  delta_freq_mhz: 0.3
  delta_time_us: 3.333333333333
  spacing_m: 0.6
  site_lat_deg: -33.4211146
  seed: 42

# Stage 2: Write Baseband Dump to Disk
write_baseband:
  kotekan_stage: rawFileWrite
  in_buf: network_capture_buf
  base_dir: {baseband_dir}
  file_name: {baseband_name}
  file_ext: bin
  num_frames_per_file: 1
  exit_after_n_files: {num_frames}
  prefix_hostname: false
  skip_zero_frames: false

# Stage 3: GPU Tensor Core Correlator
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
      block_size: 2
      num_blocks: {num_blocks}
      buffer_depth: 3
    - name: cudaCorrelatorAstron
      gpu_mem_voltage: ordered_voltage
      gpu_mem_correlation_matrix: correlation_matrix
      num_elements: {num_elements}
      num_local_freq: {num_local_freq}
      samples_per_data_set: {samples_per_data_set}
      num_data_sets: 1
      block_size: 2
      num_blocks: {num_blocks}
      elements_per_thread_block: 32
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

# Stage 4: Write Correlator Dump to Disk
write_correlator_output:
  kotekan_stage: rawFileWrite
  in_buf: host_correlation_buffer
  base_dir: {correlator_dir}
  file_name: {correlator_name}
  file_ext: bin
  num_frames_per_file: 1
  exit_after_n_files: {num_frames}
  prefix_hostname: false
  skip_zero_frames: false
"""
    with open(yaml_path, "w") as f:
        f.write(content)


def run_pipeline(
    output_dir: Path,
    kotekan_bin: Path,
    scenario_list: List[str] | None = None,
    num_frames: int = 1,
    samples_per_frame: int = 1536,
    num_antennas: int = 64,
    num_freq: int = 336,
):
    """Executes full simulation, correlation, and diagnostic pipeline."""
    if scenario_list is None:
        scenario_list = get_standard_scenarios_list()

    scenarios = scenario_list

    baseband_dir = output_dir / "baseband"
    correlator_dir = output_dir / "correlator"
    plots_dir = output_dir / "plots"
    configs_dir = output_dir / "configs"

    baseband_dir.mkdir(parents=True, exist_ok=True)
    correlator_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)

    print("======================================================================")
    print(" CHARTS 64-Antenna Full Baseband & AstronCorrelator Pipeline")
    print(f" Target Directory   : {output_dir}")
    print(f" Kotekan Binary     : {kotekan_bin}")
    print(f" Samples Per Window : {samples_per_frame} (5.12 ms)")
    print(f" Total Scenarios    : {len(scenarios)}")
    print("======================================================================")

    # 1. Generate Baseband Dumps (HDF5 & BIN)
    print("\n--- PHASE 1: Generating 64-Antenna Baseband Dumps ---")
    run_all_scenarios(
        output_dir=output_dir,
        scenario_list=scenarios,
        num_frames=num_frames,
        samples_per_frame=samples_per_frame,
        num_antennas=num_antennas,
        num_freq=num_freq,
    )

    # 2. Run Kotekan Correlator for Each Scenario
    print(
        "\n--- PHASE 2: Correlating Baseband Streams via Kotekan AstronCorrelator ---"
    )
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

    for sc in scenarios:
        bb_file_base = f"{sc}_64ant_5ms"
        corr_file_base = f"corr_{sc}"
        yaml_config = configs_dir / f"config_corr_{sc}.yaml"

        create_scenario_yaml_config(
            yaml_path=yaml_config,
            scenario=sc,
            baseband_dir=baseband_dir,
            baseband_name=bb_file_base,
            correlator_dir=correlator_dir,
            correlator_name=corr_file_base,
            samples_per_data_set=samples_per_frame,
            num_elements=num_antennas,
            num_local_freq=num_freq,
            num_frames=num_frames,
        )

        # Expected output binary file
        expected_corr_file = correlator_dir / f"{corr_file_base}_0000000.bin"
        if expected_corr_file.exists():
            expected_corr_file.unlink()

        print(f"\n>>> Running Kotekan AstronCorrelator for [{sc}] ...")
        cmd = [str(kotekan_bin), "--config", str(yaml_config)]
        t0 = time.perf_counter()
        res = subprocess.run(
            cmd, cwd=str(_kotekan_root), env=env, capture_output=True, text=True
        )
        t1 = time.perf_counter()

        if res.returncode != 0:
            print(
                f"[ERROR] Kotekan execution failed for {sc} (exit code {res.returncode})!"
            )
            print("STDERR:")
            print(res.stderr[-2000:] if len(res.stderr) > 2000 else res.stderr)
            print("STDOUT:")
            print(res.stdout[-2000:] if len(res.stdout) > 2000 else res.stdout)
            sys.exit(1)
        else:
            print(f"    Correlation completed in {(t1 - t0):.2f} s")

        if not expected_corr_file.exists():
            print(f"[ERROR] Expected correlation file not found: {expected_corr_file}")
            sys.exit(1)

        corr_size_mb = expected_corr_file.stat().st_size / (1024 * 1024)
        print(
            f"    Saved correlation dump: {expected_corr_file.name} ({corr_size_mb:.2f} MB)"
        )

    # 3. Inspect and Plot Correlation Matrices
    print("\n--- PHASE 3: Inspecting Visibilities & Generating Diagnostic Plots ---")
    for sc in scenarios:
        corr_file = correlator_dir / f"corr_{sc}_0000000.bin"
        plot_file = plots_dir / f"correlation_matrix_{sc}.png"
        print(f"\n>>> Generating diagnostic plot for [{sc}] ...")
        inspect_and_plot(
            bin_path=corr_file,
            output_plot=plot_file,
            num_elements=num_antennas,
            num_channels=num_freq,
            freq_channel_idx=num_freq // 2,
        )

    print("\n" + "=" * 78)
    print(" ALL BASEBAND & CORRELATOR DUMPS SUCCESSFULLY GENERATED AND VERIFIED!")
    print(f" Location: {output_dir}")
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(
        description="Full CHARTS 64-Antenna Baseband & Correlator Pipeline"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="all",
        help="Specific scenario name or 'all' or comma-separated list",
    )
    parser.add_argument(
        "--category",
        choices=[
            "all",
            "calibrator",
            "solar",
            "pulsar",
            "transient",
            "artificial",
            "zenith",
        ],
        default="all",
        help="Filter scenarios by astronomical category",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/project/def-vanderli/ferpb",
        help="Target output directory (default: /project/def-vanderli/ferpb)",
    )
    parser.add_argument(
        "--kotekan-bin",
        type=str,
        default=str(_kotekan_root / "build" / "kotekan" / "kotekan"),
        help="Path to compiled kotekan binary",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=1,
        help="Number of 5.12 ms frames per scenario",
    )
    parser.add_argument(
        "--samples-per-frame",
        type=int,
        default=1536,
        help="Samples per frame (5.12 ms)",
    )
    parser.add_argument("--antennas", type=int, default=64, help="Number of antennas")
    parser.add_argument("--num-freq", type=int, default=336, help="Frequency channels")
    args = parser.parse_args()

    if args.scenario == "all":
        sc_list = get_standard_scenarios_list()
        if args.category != "all":
            filtered = []
            for sc in sc_list:
                t_key, _, _ = parse_scenario_name(sc)
                if ASTRONOMICAL_CATALOG[t_key].get("category") == args.category:
                    filtered.append(sc)
            sc_list = filtered
    elif "," in args.scenario:
        sc_list = [s.strip() for s in args.scenario.split(",")]
    else:
        sc_list = [args.scenario]

    run_pipeline(
        output_dir=Path(args.output_dir),
        kotekan_bin=Path(args.kotekan_bin),
        scenario_list=sc_list,
        num_frames=args.num_frames,
        samples_per_frame=args.samples_per_frame,
        num_antennas=args.antennas,
        num_freq=args.num_freq,
    )


if __name__ == "__main__":
    main()
