#!/usr/bin/env python3
"""CHARTS 10-Terabyte Correlator Simulation Campaign Driver.

Executes a scientifically diverse 10-TB simulation suite combining realistic
baseband voltage synthesis with GPU Tensor Core Correlator (cudaCorrelatorAstron)
replay on the Trillium HPC cluster and high-memory workstations.

Simulation Suite Matrix (16 Diverse Simulations totaling ~10 TB):
  - Suite 1 (Ground-Truth Transits): Vela, Crab, Sgr A*, Centaurus A, Puppis A (~2.0 TB)
  - Suite 2 (Fast Radio Bursts): Low-DM, High-DM Wideband, Extreme 256-Antenna (~1.68 TB)
  - Suite 3 (Pulsar Trains): Millisecond and Canonical Slow Pulsar trains (~0.99 TB)
  - Suite 4 (RFI Environments): Multiline Terrestrial and LEO Satellite Sweep (~0.99 TB)
  - Suite 5 (Complex Long Runs): Deep Night, Active Sun Day, 256-Ant Array Scenes (~4.35 TB)
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, List, Optional

# Setup path so `sim` package is directly importable
_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from sim.astro import datetime_to_lst_hours
from sim.constants import (
    CHARTS_CHANNEL_WIDTH_MHZ,
    FPGA_TIME_RESOLUTION_US,
    get_antenna_positions,
)
from sim.generator import generate_simulation_window
from sim.inspector import (
    inspect_correlator_matrix,
    load_astron_correlator_dump,
)
from sim.pipeline import (
    create_correlator_yaml,
    execute_kotekan,
)
from sim.presets import (
    SimulationConfig,
    find_kotekan_binary,
    get_default_scratch_dir,
    get_default_workers,
)
from sim.sky import resolve_window_start
from sim.visualizer import (
    plot_casm_correlation_matrix,
    plot_correlator_waterfalls,
)

# Frame constants
SAMPLES_PER_FRAME: int = 1536
FRAME_DURATION_S: float = SAMPLES_PER_FRAME * (FPGA_TIME_RESOLUTION_US * 1e-6)  # 5.12 ms

# ---------------------------------------------------------------------------
# 10 TB Campaign Specification (16 Diverse Runs)
# ---------------------------------------------------------------------------

CAMPAIGN_SPEC: List[Dict[str, Any]] = [
    # -----------------------------------------------------------------------
    # Suite 1: Confirmed Ground-Truth Celestial Transits (Calibrators)
    # -----------------------------------------------------------------------
    {
        "run_id": "S1_transit_vela",
        "suite": "Suite 1: Ground-Truth Transits",
        "title": "Vela SNR / PSR B0833-45 Meridian Transit",
        "antennas": 64,
        "num_freq": 336,
        "frames": 12000,
        "start_time": "transit:Vela",
        "profile": "day",
        "num_events": 0,
        "allowed_event_types": [],
        "persistent_rfi_channels": [],
        "description": "Clean meridian transit of Vela SNR / pulsar (calibrator, high elevation).",
    },
    {
        "run_id": "S1_transit_crab",
        "suite": "Suite 1: Ground-Truth Transits",
        "title": "Taurus A (Crab Pulsar) Meridian Transit",
        "antennas": 64,
        "num_freq": 336,
        "frames": 12000,
        "start_time": "transit:Crab",
        "profile": "day",
        "num_events": 0,
        "allowed_event_types": [],
        "persistent_rfi_channels": [],
        "description": "Clean meridian transit of Crab nebula and pulsar at confirmed transit UTC.",
    },
    {
        "run_id": "S1_transit_sgra",
        "suite": "Suite 1: Ground-Truth Transits",
        "title": "Sgr A* / Galactic Center Meridian Transit",
        "antennas": 64,
        "num_freq": 336,
        "frames": 12000,
        "start_time": "transit:Sgr A*",
        "profile": "night",
        "num_events": 0,
        "allowed_event_types": [],
        "persistent_rfi_channels": [],
        "description": "Galactic Center transit at 85.6 deg maximum elevation, diffuse and compact baseline power.",
    },
    {
        "run_id": "S1_transit_cena",
        "suite": "Suite 1: Ground-Truth Transits",
        "title": "Centaurus A Radio Galaxy Meridian Transit",
        "antennas": 64,
        "num_freq": 336,
        "frames": 12000,
        "start_time": "transit:Centaurus A",
        "profile": "night",
        "num_events": 0,
        "allowed_event_types": [],
        "persistent_rfi_channels": [],
        "description": "Giant radio galaxy transit (Cen A) with extended bilateral lobe baseline phasing.",
    },
    {
        "run_id": "S1_transit_puppisa",
        "suite": "Suite 1: Ground-Truth Transits",
        "title": "Puppis A Supernova Remnant Transit",
        "antennas": 64,
        "num_freq": 336,
        "frames": 12000,
        "start_time": "transit:Puppis A",
        "profile": "day",
        "num_events": 0,
        "allowed_event_types": [],
        "persistent_rfi_channels": [],
        "description": "Supernova remnant calibrator with SIMBAD-verified coordinates (126.029 deg, -42.997 deg).",
    },

    # -----------------------------------------------------------------------
    # Suite 2: Fast Transients (FRBs across DM & Array Space)
    # -----------------------------------------------------------------------
    {
        "run_id": "S2_frb_low_dm",
        "suite": "Suite 2: Fast Transients (FRBs)",
        "title": "Low-DM High-SNR Fast Radio Burst",
        "antennas": 64,
        "num_freq": 336,
        "frames": 15000,
        "start_time": "02:00",
        "profile": "night",
        "num_events": 1,
        "allowed_event_types": ["frb"],
        "persistent_rfi_channels": [],
        "description": "Nearby low-DM (120 pc/cm^3) high-intensity FRB sweep with sharp quadratic dispersion.",
    },
    {
        "run_id": "S2_frb_high_dm_wide",
        "suite": "Suite 2: Fast Transients (FRBs)",
        "title": "High-DM Full-Bandwidth FRB (Dual Shard)",
        "antennas": 64,
        "num_freq": 672,
        "frames": 10000,
        "start_time": "03:30",
        "profile": "night",
        "num_events": 1,
        "allowed_event_types": ["frb"],
        "persistent_rfi_channels": [],
        "description": "High-DM (850 pc/cm^3) cosmological FRB dispersed across the full 672-channel bandwidth (300-501.6 MHz).",
    },
    {
        "run_id": "S2_frb_extreme_256ant",
        "suite": "Suite 2: Fast Transients (FRBs)",
        "title": "Extreme FRB on Full 256-Antenna Array",
        "antennas": 256,
        "num_freq": 336,
        "frames": 4000,
        "start_time": "04:15",
        "profile": "night",
        "num_events": 1,
        "allowed_event_types": ["frb"],
        "persistent_rfi_channels": [],
        "description": "Dispersed FRB correlated over all 32,896 independent baselines of the full 256-antenna array.",
    },

    # -----------------------------------------------------------------------
    # Suite 3: Periodic Pulsar Pulse Trains
    # -----------------------------------------------------------------------
    {
        "run_id": "S3_pulsar_millisecond",
        "suite": "Suite 3: Periodic Pulsar Trains",
        "title": "Millisecond Pulsar Pulse Train (PSR J0437-4715)",
        "antennas": 64,
        "num_freq": 336,
        "frames": 15000,
        "start_time": "transit:PSR J0437-4715",
        "profile": "day",
        "num_events": 1,
        "allowed_event_types": ["pulsar"],
        "persistent_rfi_channels": [],
        "description": "Fast millisecond pulsar (P=5.75 ms, DM=2.64) coherent pulse train during meridian transit.",
    },
    {
        "run_id": "S3_pulsar_canonical",
        "suite": "Suite 3: Periodic Pulsar Trains",
        "title": "Canonical Slow Pulsar Train (P=89 ms, DM=68)",
        "antennas": 64,
        "num_freq": 336,
        "frames": 15000,
        "start_time": "05:00",
        "profile": "night",
        "num_events": 1,
        "allowed_event_types": ["pulsar"],
        "persistent_rfi_channels": [],
        "description": "Periodic dispersed slow pulsar train (P=89.3 ms, DM=68 pc/cm^3) across tens of cycles.",
    },

    # -----------------------------------------------------------------------
    # Suite 4: Terrestrial & Orbital RFI Environments
    # -----------------------------------------------------------------------
    {
        "run_id": "S4_rfi_multiline",
        "suite": "Suite 4: RFI Environments",
        "title": "Persistent Multi-Line Terrestrial Interference",
        "antennas": 64,
        "num_freq": 336,
        "frames": 15000,
        "start_time": "14:00",
        "profile": "day",
        "num_events": 0,
        "allowed_event_types": [],
        "persistent_rfi_channels": [45, 94, 133, 147, 210, 280],
        "persistent_rfi_amp": 7.5,
        "description": "Multi-channel stationary terrestrial transmitters (FM & comms harmonics) testing baseline isolation.",
    },
    {
        "run_id": "S4_rfi_leo_sweep",
        "suite": "Suite 4: RFI Environments",
        "title": "LEO Satellite Mega-Constellation Rapid Sweep",
        "antennas": 64,
        "num_freq": 336,
        "frames": 15000,
        "start_time": "20:00",
        "profile": "night",
        "num_events": 1,
        "allowed_event_types": ["rfi_leo"],
        "persistent_rfi_channels": [],
        "description": "Rapidly drifting non-celestial phase sweep (v_drift=0.025/s) simulating LEO satellite transit.",
    },

    # -----------------------------------------------------------------------
    # Suite 5: Complex Long-Duration Multi-Event Runs (The 1-TB Scale Pillars)
    # -----------------------------------------------------------------------
    {
        "run_id": "S5_complex_night_64ant",
        "suite": "Suite 5: Complex Multi-Event Long Runs",
        "title": "Deep Nighttime Multi-Event Complex Observation",
        "antennas": 64,
        "num_freq": 672,
        "frames": 16000,
        "start_time": "01:00",
        "profile": "night",
        "num_events": 3,
        "allowed_event_types": ["frb", "pulsar", "rfi_narrow"],
        "persistent_rfi_channels": [120, 310, 512],
        "persistent_rfi_amp": 6.5,
        "description": "1.06 TB observation: quiet night sky + 2 dispersed FRBs + pulsar + multi-channel RFI across 672 channels.",
    },
    {
        "run_id": "S5_complex_day_sun_64ant",
        "suite": "Suite 5: Complex Multi-Event Long Runs",
        "title": "Active Sun Daytime Multi-Event Observation",
        "antennas": 64,
        "num_freq": 672,
        "frames": 16000,
        "start_time": "15:00",
        "profile": "day",
        "num_events": 3,
        "allowed_event_types": ["frb", "rfi_leo", "rfi_narrow"],
        "persistent_rfi_channels": [94, 133, 420],
        "persistent_rfi_amp": 7.0,
        "description": "1.06 TB observation: elevated solar background power + bright FRB + LEO satellite sweep across 672 channels.",
    },
    {
        "run_id": "S5_complex_256ant_336ch",
        "suite": "Suite 5: Complex Multi-Event Long Runs",
        "title": "Full 256-Antenna Array Complex Scene",
        "antennas": 256,
        "num_freq": 336,
        "frames": 8500,
        "start_time": "18:00",
        "profile": "night",
        "num_events": 2,
        "allowed_event_types": ["frb", "pulsar"],
        "persistent_rfi_channels": [94, 147],
        "persistent_rfi_amp": 6.0,
        "description": "1.12 TB observation: 256 antennas (65,536 cross-correlations) with simultaneous dispersed FRB and pulsar.",
    },
    {
        "run_id": "S5_complex_256ant_672ch",
        "suite": "Suite 5: Complex Multi-Event Long Runs",
        "title": "Full Instrument 256-Antenna Dual-Shard Benchmark",
        "antennas": 256,
        "num_freq": 672,
        "frames": 4200,
        "start_time": "22:00",
        "profile": "night",
        "num_events": 2,
        "allowed_event_types": ["frb", "rfi_leo"],
        "persistent_rfi_channels": [133, 266, 520],
        "persistent_rfi_amp": 7.0,
        "description": "1.11 TB benchmark: maximum instrument throughput (51.6 GB/s ingest) on 256 antennas and 672 channels.",
    },
]


def calculate_run_bytes(spec: Dict[str, Any]) -> int:
    """Calculates total baseband data payload in bytes for a given run spec."""
    ant = spec["antennas"]
    nfreq = spec["num_freq"]
    frames = spec["frames"]
    # 4-byte frame_id header + packed baseband payload
    return frames * (SAMPLES_PER_FRAME * nfreq * ant + 4)


def print_campaign_table() -> None:
    """Prints formatted summary table of the entire 10 TB campaign."""
    total_bytes = 0
    total_duration_s = 0.0

    print("=" * 115)
    print(" CHARTS 10-TERABYTE CORRELATOR SIMULATION CAMPAIGN (TRILLIUM HPC CLUSTER)")
    print("=" * 115)
    print(
        f" {'Idx':<3} | {'Run ID':<26} | {'Ant':>4} | {'Chans':>5} | {'Frames':>7} | "
        f"{'Duration':>8} | {'Data Volume':>11} | {'Suite'}"
    )
    print("-" * 115)

    for idx, spec in enumerate(CAMPAIGN_SPEC):
        b = calculate_run_bytes(spec)
        total_bytes += b
        dur_s = spec["frames"] * FRAME_DURATION_S
        total_duration_s += dur_s
        gb = b / 1e9
        print(
            f" {idx:>3} | {spec['run_id']:<26} | {spec['antennas']:>4} | {spec['num_freq']:>5} | "
            f"{spec['frames']:>7} | {dur_s:>7.1f}s | {gb:>9.1f} GB | {spec['suite']}"
        )

    print("=" * 115)
    print(
        f" TOTAL CAMPAIGN VOLUME : {total_bytes / 1e12:.3f} TB ({total_bytes / (1024**4):.3f} TiB) "
        f"| {len(CAMPAIGN_SPEC)} Diverse Runs | {total_duration_s:.1f} s cumulative physical time"
    )
    print("=" * 115)


def run_single_simulation(
    spec: Dict[str, Any],
    scratch_dir: Path,
    output_dir: Path,
    kotekan_bin: Optional[Path] = None,
    workers: int = 8,
    dry_run: bool = False,
    cleanup_baseband: bool = False,
) -> int:
    """Executes a single simulation run: baseband generation -> correlator replay -> CASM extraction."""
    run_id = spec["run_id"]
    dur_s = spec["frames"] * FRAME_DURATION_S
    data_bytes = calculate_run_bytes(spec)

    print("\n" + "#" * 90)
    print(f" EXECUTING RUN: {run_id} ({spec['title']})")
    print(f" Suite       : {spec['suite']}")
    print(f" Antennas    : {spec['antennas']} | Channels: {spec['num_freq']} | Frames: {spec['frames']}")
    print(f" Duration    : {dur_s:.2f} s physical time | Data Volume: {data_bytes / 1e9:.2f} GB")
    print(f" Start Time  : {spec['start_time']} | Profile: {spec['profile']}")
    print(f" Description : {spec['description']}")
    print("#" * 90)

    t0_start = time.perf_counter()
    window_dir = scratch_dir / run_id
    window_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build SimulationConfig
    cfg = SimulationConfig(
        preset="custom",
        profile=spec["profile"],
        window_name=run_id,
        duration_s=dur_s,
        background_cadence_s=0.0,  # Continuous streaming: write every physical frame
        antennas=spec["antennas"],
        num_freq=spec["num_freq"],
        samples_per_frame=SAMPLES_PER_FRAME,
        num_events=spec.get("num_events", 0),
        allowed_event_types=spec.get("allowed_event_types"),
        persistent_rfi_channels=spec.get("persistent_rfi_channels", []),
        persistent_rfi_amp=spec.get("persistent_rfi_amp", 7.0),
        scratch_dir=scratch_dir,
        output_dir=output_dir,
        kotekan_bin=kotekan_bin,
        workers=workers,
        dry_run=dry_run,
        skip_tracker=True,  # EXPLICIT: Correlator-only, no tracker
    )

    # Resolve observation start time & LST
    if spec["start_time"]:
        cfg.start_time = spec["start_time"]
        dt = resolve_window_start(spec["start_time"], duration_s=dur_s)
        cfg.initial_lst_hours = datetime_to_lst_hours(dt)

    cfg.save_yaml(window_dir / "sim_config.yaml")

    # 2. Step 1: Baseband Generation
    print("\n[Step 1/4] Generating Baseband Voltage Stream...")
    t0_gen = time.perf_counter()
    gen_result = generate_simulation_window(cfg)
    num_written = gen_result["num_written"]
    t_gen_s = time.perf_counter() - t0_gen
    print(f"  * Generated {num_written} frames in {t_gen_s:.1f} s ({data_bytes / 1e9 / max(0.1, t_gen_s):.2f} GB/s)")

    # 3. Step 2: Correlator Replay (cudaCorrelatorAstron)
    print("\n[Step 2/4] Executing Kotekan GPU Tensor Core Correlator...")
    corr_dir = window_dir / "correlator"
    corr_yaml = window_dir / "kotekan_correlator.yaml"
    create_correlator_yaml(
        yaml_path=corr_yaml,
        baseband_dir=window_dir,
        baseband_name=run_id,
        correlator_dir=corr_dir,
        num_frames=num_written,
        num_elements=cfg.antennas,
        num_local_freq=cfg.num_freq,
        samples_per_data_set=cfg.samples_per_frame,
        buffer_depth=cfg.buffer_depth,
    )

    t0_corr = time.perf_counter()
    rc_corr = execute_kotekan(
        config_path=corr_yaml,
        kotekan_bin=cfg.kotekan_bin,
        dry_run=cfg.dry_run,
        log_path=window_dir / "kotekan_correlator.log",
    )
    t_corr_s = time.perf_counter() - t0_corr

    if rc_corr != 0 and not dry_run:
        print(f"[ERROR] Correlator execution failed with return code {rc_corr}")
        return rc_corr

    # 4. Step 3: Correlation Inspection & Matrix Verification
    print("\n[Step 3/4] Inspecting Correlation Matrix & Hermitian Properties...")
    corr_dumps = sorted(corr_dir.glob("corr_*.bin")) if not dry_run else []
    if corr_dumps:
        mid_dump = corr_dumps[len(corr_dumps) // 2]
        vis_cube = load_astron_correlator_dump(mid_dump, num_elements=cfg.antennas, num_channels=cfg.num_freq)
        diag = inspect_correlator_matrix(vis_cube)
        print(f"  * Correlator Hermitian Error : {diag['hermitian_error']:.2e} (Passed: {diag['hermitian_valid']})")
        print(f"  * Mean Autocorrelation Power : {diag['mean_autocorr']:.2f} LSB^2")
        print(f"  * Baseline Cross-Power SNR   : {diag['cross_snr']:.2f}")

    # 5. Step 4: CASM Matrix & Waterfall Plot Generation
    print("\n[Step 4/4] Generating CASM Matrix and Correlation Waterfalls...")
    plots_dir = window_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if corr_dumps and not dry_run:
        casm_out = plots_dir / f"casm_matrix_{run_id}.png"
        plot_casm_correlation_matrix(
            corr_dir=corr_dir,
            output_path=casm_out,
            num_antennas=min(16, cfg.antennas),
            num_channels=cfg.num_freq,
            duration_s=cfg.duration_s,
            window_label=f"CHARTS CASM ({run_id})",
        )
        print(f"  * Saved CASM Correlation Plot: {casm_out}")

        wf_out = plots_dir / f"correlator_waterfall_{run_id}.png"
        plot_correlator_waterfalls(
            corr_dir=corr_dir,
            output_path=wf_out,
            num_elements=cfg.antennas,
            num_channels=cfg.num_freq,
            duration_s=cfg.duration_s,
            window_label=f"CHARTS Waterfall ({run_id})",
        )
        print(f"  * Saved Correlator Waterfall : {wf_out}")

    # 6. Archive Data Products to Output Directory
    dest_dir = output_dir / run_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n>>> Archiving data products to {dest_dir}...")
    for f in window_dir.glob("*_meta.h5"):
        shutil.copy2(f, dest_dir)
    for f in window_dir.glob("*_events.json"):
        shutil.copy2(f, dest_dir)
    for f in window_dir.glob("sim_config.yaml"):
        shutil.copy2(f, dest_dir)
    if plots_dir.exists():
        dest_plots = dest_dir / "plots"
        dest_plots.mkdir(parents=True, exist_ok=True)
        for p in plots_dir.glob("*.png"):
            shutil.copy2(p, dest_plots)

    # 7. Baseband Cleanup (optional, frees scratch space in sequential batch runs)
    if cleanup_baseband and not dry_run:
        print(f"\n>>> Cleaning up {num_written} raw baseband .bin files from {window_dir}...")
        for bin_f in window_dir.glob("*.bin"):
            bin_f.unlink()
        print("  * Baseband scratch space reclaimed successfully.")

    total_time = time.perf_counter() - t0_start
    print("\n" + "=" * 90)
    print(f" [SUCCESS] RUN COMPLETE: {run_id} ({total_time:.2f} s)")
    print(f" Products archived in : {dest_dir}")
    print("=" * 90)
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="CHARTS 10-Terabyte Correlator Simulation Campaign Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--list", action="store_true", help="List all 16 simulations in the campaign matrix and exit")
    parser.add_argument("--run-idx", type=int, default=None, help="Run specific simulation index (0 to 15, matching Slurm array task ID)")
    parser.add_argument("--run-id", type=str, default=None, help="Run specific simulation by Run ID (e.g. S1_transit_vela)")
    parser.add_argument("--all", action="store_true", help="Run all 16 simulations sequentially")
    parser.add_argument("--scratch-dir", type=str, default=None, help="Base scratch directory for temporary baseband dumps")
    parser.add_argument("--output-dir", type=str, default="./test_charts/data/campaign_10tb_output", help="Permanent output directory")
    parser.add_argument("--kotekan-bin", type=str, default=None, help="Path to kotekan executable")
    parser.add_argument("--workers", type=int, default=None, help="Number of CPU worker threads for baseband synthesis")
    parser.add_argument("--cleanup-baseband", action="store_true", help="Delete raw baseband frames after correlator completes")
    parser.add_argument("--dry-run", action="store_true", help="Validate configurations without writing full baseband files")

    args = parser.parse_args()

    if args.list or (args.run_idx is None and args.run_id is None and not args.all):
        print_campaign_table()
        if not args.list:
            print("\nUsage Examples:")
            print("  python test_charts/run_10tb_correlator_campaign.py --run-idx 0 --dry-run")
            print("  python test_charts/run_10tb_correlator_campaign.py --run-id S1_transit_vela")
            print("  sbatch test_charts/slurm/submit_10tb_correlator_campaign.slurm")
        sys.exit(0)

    # Resolve scratch & output paths
    scratch_base = Path(args.scratch_dir) if args.scratch_dir else get_default_scratch_dir()
    output_base = Path(args.output_dir)
    workers = args.workers or get_default_workers()
    kotekan_bin = Path(args.kotekan_bin) if args.kotekan_bin else find_kotekan_binary()

    # Determine which runs to execute
    selected_specs: List[Dict[str, Any]] = []
    if args.run_idx is not None:
        if 0 <= args.run_idx < len(CAMPAIGN_SPEC):
            selected_specs.append(CAMPAIGN_SPEC[args.run_idx])
        else:
            print(f"[ERROR] --run-idx {args.run_idx} out of range [0, {len(CAMPAIGN_SPEC) - 1}]")
            sys.exit(1)
    elif args.run_id is not None:
        matched = [s for s in CAMPAIGN_SPEC if s["run_id"].lower() == args.run_id.lower()]
        if matched:
            selected_specs.append(matched[0])
        else:
            print(f"[ERROR] Run ID '{args.run_id}' not found in campaign specification.")
            sys.exit(1)
    elif args.all:
        selected_specs = CAMPAIGN_SPEC

    print(f"\nStarting Campaign Execution: {len(selected_specs)} simulation(s) queued.")
    print(f"Scratch Directory : {scratch_base}")
    print(f"Output Directory  : {output_base}")
    print(f"Workers           : {workers}")
    print(f"Kotekan Binary    : {kotekan_bin or 'Auto-detect'}")
    print(f"Dry Run Mode      : {args.dry_run}")
    print(f"Cleanup Baseband  : {args.cleanup_baseband}")

    for spec in selected_specs:
        rc = run_single_simulation(
            spec=spec,
            scratch_dir=scratch_base,
            output_dir=output_base,
            kotekan_bin=kotekan_bin,
            workers=workers,
            dry_run=args.dry_run,
            cleanup_baseband=args.cleanup_baseband,
        )
        if rc != 0:
            print(f"[FATAL] Simulation {spec['run_id']} failed with code {rc}. Aborting.")
            sys.exit(rc)

    print("\n" + "=" * 90)
    print(" ALL REQUESTED SIMULATIONS EXECUTED SUCCESSFULLY.")
    print("=" * 90)


if __name__ == "__main__":
    main()
