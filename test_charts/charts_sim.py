#!/usr/bin/env python3
"""CHARTS Unified Simulation & Pipeline Orchestrator CLI.

Centralized entry point for simulating, running, inspecting, and benchmarking
CHARTS radio telescope data with Kotekan on both local workstations (RTX 4090/5090)
and HPC cluster nodes (Trillium).

Commands:
  pipeline   - Run complete end-to-end simulation (Generate -> Correlate -> Direct Track -> Validate -> Visualize)
  generate   - Generate physical baseband data window, celestial targets, and metadata
  reference  - Manage reference baseband library (list, info) for reproducible testing
  correlate  - Replay baseband frames through Kotekan GPU Tensor Core Correlator
  track      - Replay baseband frames through Kotekan Direct Beam Tracker (cudaDirectBeamTracker)
  inspect    - Verify mathematical properties (Hermitian check, SNR, dynamic range, formed beam power)
  visualize  - Produce publication-ready plots (CASM matrix, waterfalls, lightcurves)
  benchmark  - Execute GPU Direct Beam Tracker performance, VRAM, and real-cadence power analysis
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Optional

# Setup path so `sim` package is directly importable
_script_dir = Path(__file__).resolve().parent
_repo_root = _script_dir.parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from sim.astro import (
    datetime_to_lst_hours,
    find_visible_sources,
    parse_observation_time,
    resolve_beam_targets,
)
from sim.sky import (
    catalog_transit_summary,
    load_verified_catalog,
    resolve_window_start,
)
from sim.benchmark import run_direct_tracker_benchmark
from sim.constants import (
    CHARTS_CHANNEL_WIDTH_MHZ,
    CHARTS_LATITUDE_DEG,
    CHARTS_LONGITUDE_DEG,
    DEFAULT_FREQUENCY_START_MHZ,
)
from sim.generator import generate_simulation_window
from sim.inspector import (
    inspect_correlator_matrix,
    inspect_tracker_dump,
    load_astron_correlator_dump,
)
from sim.pipeline import (
    create_beam_tracker_yaml,
    create_correlator_yaml,
    execute_kotekan,
    parse_beam_targets,
)
from sim.presets import (
    SimulationConfig,
    find_kotekan_binary,
    get_preset_config,
)
from sim.reference import (
    create_reference_manifest,
    get_reference_window,
    list_reference_windows,
    save_as_reference_window,
)
from sim.visualizer import (
    plot_casm_correlation_matrix,
    plot_correlator_waterfalls,
    plot_tracker_waterfall,
)


def run_pipeline(cfg: SimulationConfig) -> int:
    """Executes the complete end-to-end CHARTS simulation pipeline."""
    print("=" * 80)
    print(
        f" CHARTS SIMULATION PIPELINE: PRESET={cfg.preset.upper()} PROFILE={cfg.profile.upper()}"
    )
    print("=" * 80)
    print(f" Window Name       : {cfg.window_name}")
    print(f" Start Time (UTC)  : {cfg.start_time or f'{cfg.utc_hour:.1f}h UTC'}")
    print(f" Initial LST       : {cfg.initial_lst_hours:.4f} h")
    print(f" Duration          : {cfg.duration_s:.1f} s")
    print(f" Antennas          : {cfg.antennas}")
    print(f" Frequency Channels: {cfg.num_freq}")
    print(f" Max Beams         : {cfg.max_beams}")
    print(f" Scratch Directory : {cfg.scratch_dir}")
    print(f" Output Directory  : {cfg.output_dir}")
    print(f" Kotekan Binary    : {cfg.kotekan_bin or 'Auto-detect'}")
    print(f" Dry Run Mode      : {cfg.dry_run}")
    print("=" * 80)

    t0_start = time.perf_counter()
    window_dir = Path(cfg.scratch_dir) / cfg.window_name
    window_dir.mkdir(parents=True, exist_ok=True)
    cfg.save_yaml(window_dir / "sim_config.yaml")

    # Step 1: Baseband Generation
    print("\n[Step 1/5] Generating Physical Baseband Data...")
    gen_result = generate_simulation_window(cfg)
    num_frames = gen_result["num_written"]

    # Step 2: Correlator Replay
    print("\n[Step 2/5] Running Kotekan Tensor Core Correlator...")
    corr_dir = window_dir / "correlator"
    corr_yaml = window_dir / "kotekan_correlator.yaml"
    create_correlator_yaml(
        yaml_path=corr_yaml,
        baseband_dir=window_dir,
        baseband_name=cfg.window_name,
        correlator_dir=corr_dir,
        num_frames=num_frames,
        num_elements=cfg.antennas,
        num_local_freq=cfg.num_freq,
        samples_per_data_set=cfg.samples_per_frame,
        buffer_depth=cfg.buffer_depth,
    )

    rc_corr = execute_kotekan(
        config_path=corr_yaml,
        kotekan_bin=cfg.kotekan_bin,
        dry_run=cfg.dry_run,
        log_path=window_dir / "kotekan_correlator.log",
    )
    if rc_corr != 0 and not cfg.dry_run:
        print(f"[ERROR] Correlator execution failed with return code {rc_corr}")
        return rc_corr

    # Step 3: Direct Beam Tracker Replay (cudaDirectBeamTracker)
    if not getattr(cfg, "skip_tracker", False):
        print(
            f"\n[Step 3/5] Running Kotekan Direct Beam Tracker ({cfg.max_beams} Beams)..."
        )
        tracker_dir = window_dir / "tracker"
        tracker_yaml = window_dir / "kotekan_tracker.yaml"
        beam_targets = parse_beam_targets(
            cfg.beam_targets,
            max_beams=cfg.max_beams,
            default_lst=cfg.initial_lst_hours,
            obs_time=cfg.start_time,
        )
        create_beam_tracker_yaml(
            yaml_path=tracker_yaml,
            baseband_dir=window_dir,
            baseband_name=cfg.window_name,
            tracker_dir=tracker_dir,
            tracker_name=f"beams_{cfg.window_name}",
            num_frames=num_frames,
            beam_targets=beam_targets,
            num_elements=cfg.antennas,
            num_local_freq=cfg.num_freq,
            samples_per_data_set=cfg.samples_per_frame,
            max_beams=cfg.max_beams,
            integration_spectra=cfg.integration_spectra,
            buffer_depth=cfg.buffer_depth,
            stage_type="direct",
        )

        rc_track = execute_kotekan(
            config_path=tracker_yaml,
            kotekan_bin=cfg.kotekan_bin,
            dry_run=cfg.dry_run,
            log_path=window_dir / "kotekan_tracker.log",
        )
        if rc_track != 0 and not cfg.dry_run:
            print(f"[ERROR] Beam tracker execution failed with return code {rc_track}")
            return rc_track
    else:
        print("\n[Step 3/5] Skipping Beam Tracker (--no-tracker requested)...")

    # Step 4: Verification & Inspection
    print("\n[Step 4/5] Inspecting & Validating Pipeline Outputs...")
    if not cfg.dry_run:
        corr_dumps = sorted(corr_dir.glob("corr_*.bin"))
        if corr_dumps:
            mid_dump = corr_dumps[len(corr_dumps) // 2]
            vis_cube = load_astron_correlator_dump(
                mid_dump, num_elements=cfg.antennas, num_channels=cfg.num_freq
            )
            diag = inspect_correlator_matrix(vis_cube)
            print(
                f"  * Correlator Hermitian error : {diag['hermitian_error']:.2e} (Valid: {diag['hermitian_valid']})"
            )
            print(f"  * Mean Autocorrelation power : {diag['mean_autocorr']:.2f} LSB^2")
            print(f"  * Baseline cross-power SNR   : {diag['cross_snr']:.2f}")

        if not getattr(cfg, "skip_tracker", False):
            tracker_dir = window_dir / "tracker"
            tracker_dumps = sorted(tracker_dir.glob("*.bin"))
            if tracker_dumps:
                t_diag = inspect_tracker_dump(
                    tracker_dumps[0],
                    num_freq=cfg.num_freq,
                    max_beams=cfg.max_beams,
                    samples_per_data_set=cfg.samples_per_frame,
                )
                print(f"  * Tracker total beam power   : {t_diag['total_power']:.2f}")
                print(f"  * Peak formed beam slot      : Beam {t_diag['peak_beam']}")
    else:
        print("  [DRY-RUN] Skipped binary dump inspection.")

    # Step 5: Visualizations
    print("\n[Step 5/5] Generating Visualizations and Plots...")
    plots_dir = window_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not cfg.dry_run:
        if cfg.generate_casm and corr_dumps:
            casm_out = plots_dir / f"casm_matrix_{cfg.window_name}.png"
            plot_casm_correlation_matrix(
                corr_dir=corr_dir,
                output_path=casm_out,
                num_antennas=min(16, cfg.antennas),
                num_channels=cfg.num_freq,
                duration_s=cfg.duration_s,
                window_label=f"CHARTS CASM ({cfg.window_name})",
            )
            print(f"  * Saved CASM plot            : {casm_out}")

        if cfg.generate_waterfalls and corr_dumps:
            wf_out = plots_dir / f"correlator_waterfall_{cfg.window_name}.png"
            plot_correlator_waterfalls(
                corr_dir=corr_dir,
                output_path=wf_out,
                num_elements=cfg.antennas,
                num_channels=cfg.num_freq,
                duration_s=cfg.duration_s,
                window_label=f"CHARTS ({cfg.window_name})",
            )
            print(f"  * Saved Correlator waterfall : {wf_out}")

        if not getattr(cfg, "skip_tracker", False):
            tracker_dir = window_dir / "tracker"
            tracker_dumps = (
                sorted(tracker_dir.glob("*.bin")) if tracker_dir.exists() else []
            )
            if tracker_dumps:
                tr_out = plots_dir / f"tracker_waterfall_{cfg.window_name}.png"
                plot_tracker_waterfall(
                    tracker_dir=tracker_dir,
                    output_path=tr_out,
                    max_beams=cfg.max_beams,
                    num_freq=cfg.num_freq,
                    samples_per_frame=cfg.samples_per_frame,
                    duration_s=cfg.duration_s,
                    beam_targets_str=cfg.beam_targets,
                )
                print(f"  * Saved Tracker lightcurves  : {tr_out}")
    else:
        print("  [DRY-RUN] Skipped plot generation.")

    # Step 6: Archiving to Permanent Output Directory and Reference Library
    dest_dir = Path(cfg.output_dir) / cfg.window_name
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n>>> Archiving data products to {dest_dir}...")
    for f in window_dir.glob("*_meta.h5"):
        shutil.copy2(f, dest_dir)
    for f in window_dir.glob("*_events.json"):
        shutil.copy2(f, dest_dir)
    if plots_dir.exists():
        dest_plots = dest_dir / "plots"
        dest_plots.mkdir(parents=True, exist_ok=True)
        for p in plots_dir.glob("*.png"):
            shutil.copy2(p, dest_plots)

    if cfg.save_reference:
        save_as_reference_window(
            src_window_dir=window_dir,
            tag=cfg.save_reference,
            description=f"Reference Window: {cfg.window_name} (Preset={cfg.preset})",
            obs_time=cfg.start_time,
            duration_s=cfg.duration_s,
            antennas=cfg.antennas,
            num_freq=cfg.num_freq,
            samples_per_frame=cfg.samples_per_frame,
            active_sources=beam_targets,
        )

    total_time = time.perf_counter() - t0_start
    print("\n" + "=" * 80)
    print(f" CHARTS SIMULATION PIPELINE COMPLETE ({total_time:.2f} s)")
    print(f" Products archived in : {dest_dir}")
    print("=" * 80)
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="CHARTS Unified Simulation, Tracker Replay & Benchmark CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # 1. Pipeline subcommand
    pipe_parser = subparsers.add_parser(
        "pipeline", help="Run end-to-end simulation pipeline"
    )
    pipe_parser.add_argument(
        "--preset",
        type=str,
        default="quick",
        choices=["quick", "1min", "5min"],
        help="Preset profile",
    )
    pipe_parser.add_argument(
        "--profile",
        type=str,
        default="day",
        choices=["day", "night", "both"],
        help="Time/sky profile",
    )
    pipe_parser.add_argument(
        "--start-time",
        type=str,
        default=None,
        help="Observation start time (ISO 8601, 'now', or 'HH:MM')",
    )
    pipe_parser.add_argument(
        "--transit",
        "--transit-target",
        dest="transit_target",
        type=str,
        default=None,
        help="Anchor window at target confirmed transit (e.g. 'Vela', 'Sgr A*', 'Crab')",
    )
    pipe_parser.add_argument(
        "--beam-targets",
        type=str,
        default=None,
        help="Targets: e.g. 'Crab;Vela' or 'auto'",
    )
    pipe_parser.add_argument(
        "--duration-s", type=float, default=None, help="Custom duration (seconds)"
    )
    pipe_parser.add_argument(
        "--antennas", type=int, default=None, help="Number of antennas (e.g. 64 or 256)"
    )
    pipe_parser.add_argument(
        "--num-freq", type=int, default=None, help="Number of frequency channels"
    )
    pipe_parser.add_argument(
        "--max-beams", type=int, default=None, help="Formed beam count"
    )
    pipe_parser.add_argument(
        "--scratch-dir",
        type=str,
        default=None,
        help="Scratch directory for runtime dumps",
    )
    pipe_parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Permanent directory for output artifacts",
    )
    pipe_parser.add_argument(
        "--save-reference",
        type=str,
        default=None,
        help="Save generated dataset to reference library under tag",
    )
    pipe_parser.add_argument(
        "--kotekan-bin", type=str, default=None, help="Path to kotekan executable"
    )
    pipe_parser.add_argument(
        "--workers", type=int, default=None, help="Parallel worker threads"
    )
    pipe_parser.add_argument(
        "--no-tracker",
        "--skip-tracker",
        dest="skip_tracker",
        action="store_true",
        help="Run only baseband generation and correlator (skip beam tracker)",
    )
    pipe_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate configs and simulate without running Kotekan binary",
    )

    # 2. Generate subcommand
    gen_parser = subparsers.add_parser(
        "generate", help="Generate baseband window dataset"
    )
    gen_parser.add_argument(
        "--duration-s", type=float, default=10.0, help="Duration in seconds"
    )
    gen_parser.add_argument(
        "--antennas",
        type=int,
        nargs="+",
        default=[64],
        help="Number of antennas (e.g. 32, 64, or '32 64 256')",
    )
    gen_parser.add_argument(
        "--num-freq", type=int, default=336, help="Frequency channels"
    )
    gen_parser.add_argument(
        "--profile",
        type=str,
        default="day",
        choices=["day", "night"],
        help="Day or Night profile",
    )
    gen_parser.add_argument(
        "--start-time",
        type=str,
        default=None,
        help="Observation start time (ISO 8601, 'now', or 'HH:MM')",
    )
    gen_parser.add_argument(
        "--transit",
        "--transit-target",
        dest="transit_target",
        type=str,
        default=None,
        help="Anchor window at target confirmed transit (e.g. 'Vela', 'Sgr A*', 'Crab')",
    )
    gen_parser.add_argument(
        "--beam-targets",
        type=str,
        default="auto",
        help="Injected celestial sources (e.g. 'Crab;Vela' or 'auto')",
    )
    gen_parser.add_argument(
        "--num-events", type=int, default=2, help="Number of injected transients"
    )
    gen_parser.add_argument(
        "--save-reference",
        type=str,
        default=None,
        help="Save to reference library under tag",
    )
    gen_parser.add_argument(
        "--scratch-dir",
        type=str,
        default="./scratch_charts_sim",
        help="Target output directory",
    )

    # Catalog subcommand
    subparsers.add_parser(
        "catalog", help="List confirmed targets and transit ephemeris"
    )

    # 3. Reference Library subcommand
    ref_parser = subparsers.add_parser(
        "reference", help="Manage reference baseband library"
    )
    ref_parser.add_argument(
        "action", choices=["list", "info"], help="Action: 'list' or 'info'"
    )
    ref_parser.add_argument(
        "--tag", type=str, default=None, help="Reference window tag for 'info'"
    )

    # 4. Correlate subcommand
    corr_parser = subparsers.add_parser(
        "correlate", help="Replay baseband frames through Kotekan correlator"
    )
    corr_parser.add_argument(
        "--window-dir",
        type=str,
        default=None,
        help="Directory containing baseband .bin frames",
    )
    corr_parser.add_argument(
        "--reference", type=str, default=None, help="Reference window tag to correlate"
    )
    corr_parser.add_argument(
        "--kotekan-bin", type=str, default=None, help="Path to kotekan executable"
    )
    corr_parser.add_argument(
        "--dry-run", action="store_true", help="Print command without execution"
    )

    # 5. Track subcommand (cudaDirectBeamTracker)
    track_parser = subparsers.add_parser(
        "track", help="Replay baseband frames through Kotekan Direct Beam Tracker"
    )
    track_parser.add_argument(
        "--window-dir",
        type=str,
        default=None,
        help="Directory containing baseband .bin frames",
    )
    track_parser.add_argument(
        "--reference", type=str, default=None, help="Reference window tag to track"
    )
    track_parser.add_argument(
        "--beam-targets",
        type=str,
        default="auto",
        help="Beam targets: 'Crab;Vela', 'auto', or explicit RA,Dec",
    )
    track_parser.add_argument(
        "--max-beams", type=int, default=4, help="Number of beams"
    )
    track_parser.add_argument(
        "--kotekan-bin", type=str, default=None, help="Path to kotekan executable"
    )
    track_parser.add_argument(
        "--dry-run", action="store_true", help="Print command without execution"
    )

    # 6. Benchmark subcommand (Direct Beam Tracker: Power, VRAM, Real Cadence)
    bench_parser = subparsers.add_parser(
        "benchmark",
        help="Benchmark Direct Beam Tracker under real cadence, VRAM & GPU power",
    )
    bench_parser.add_argument(
        "--antennas",
        type=int,
        nargs="+",
        default=[32, 64, 128, 256],
        help="Antenna counts",
    )
    bench_parser.add_argument(
        "--beams", type=int, nargs="+", default=[1, 4, 8], help="Beam counts"
    )
    bench_parser.add_argument(
        "--num-freq", type=int, default=672, help="Number of frequency channels"
    )
    bench_parser.add_argument(
        "--samples-per-frame",
        type=int,
        default=1536,
        choices=[1536, 3840, 15360],
        help="Samples per frame (1536=5.12ms, 3840=12.8ms, 15360=51.2ms)",
    )
    bench_parser.add_argument(
        "--buffer-depth", type=int, default=2, help="Ring buffer depth"
    )
    bench_parser.add_argument(
        "--target-gpu",
        type=str,
        default=None,
        choices=["rtx4090", "rtx5090", "h100", "rtx3060"],
        help="Target GPU hardware profile",
    )
    bench_parser.add_argument(
        "--out-json", type=str, default=None, help="Save JSON report"
    )
    bench_parser.add_argument(
        "--out-md", type=str, default=None, help="Save Markdown report"
    )

    # 7. Inspect subcommand
    insp_parser = subparsers.add_parser(
        "inspect", help="Inspect correlator or beam tracker binary outputs"
    )
    insp_parser.add_argument(
        "--file", type=str, required=True, help="Path to .bin dump file"
    )
    insp_parser.add_argument(
        "--type",
        type=str,
        default="corr",
        choices=["corr", "tracker"],
        help="Dump file type",
    )

    # 8. Visualize subcommand
    viz_parser = subparsers.add_parser(
        "visualize", help="Generate visualizations from correlator or tracker dumps"
    )
    viz_parser.add_argument(
        "--window-dir", type=str, required=True, help="Window directory"
    )
    viz_parser.add_argument(
        "--type",
        type=str,
        default="waterfall",
        choices=["casm", "waterfall", "tracker", "all"],
    )
    viz_parser.add_argument(
        "--out-file", type=str, default=None, help="Output image file path"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "pipeline":
        profiles = ["day", "night"] if args.profile == "both" else [args.profile]
        for p in profiles:
            cfg = get_preset_config(
                preset=args.preset,
                profile=p,
                antennas=args.antennas,
                num_freq=args.num_freq,
            )
            if args.transit_target:
                cfg.start_time = f"transit:{args.transit_target}"
                dt = resolve_window_start(cfg.start_time, duration_s=cfg.duration_s)
                cfg.initial_lst_hours = datetime_to_lst_hours(dt)
                if not args.beam_targets:
                    cfg.beam_targets = args.transit_target
            elif args.start_time:
                cfg.start_time = args.start_time
                dt = resolve_window_start(args.start_time, duration_s=cfg.duration_s)
                cfg.initial_lst_hours = datetime_to_lst_hours(dt)
            if args.beam_targets and not (
                args.transit_target and not args.beam_targets
            ):
                cfg.beam_targets = args.beam_targets
            if args.duration_s:
                cfg.duration_s = args.duration_s
            if args.max_beams:
                cfg.max_beams = args.max_beams
            if args.scratch_dir:
                cfg.scratch_dir = Path(args.scratch_dir)
            if args.output_dir:
                cfg.output_dir = Path(args.output_dir)
            if args.save_reference:
                cfg.save_reference = args.save_reference
            if args.kotekan_bin:
                cfg.kotekan_bin = Path(args.kotekan_bin)
            if args.workers:
                cfg.workers = args.workers
            cfg.skip_tracker = args.skip_tracker
            cfg.dry_run = args.dry_run

            ret = run_pipeline(cfg)
            if ret != 0:
                sys.exit(ret)

    elif args.command == "generate":
        ant_list = args.antennas if isinstance(args.antennas, list) else [args.antennas]
        for ant in ant_list:
            cfg = get_preset_config(
                preset="quick",
                profile=args.profile,
                antennas=ant,
                num_freq=args.num_freq,
            )
            cfg.duration_s = args.duration_s
            cfg.num_events = args.num_events
            cfg.scratch_dir = Path(args.scratch_dir)
            cfg.window_name = f"charts_sim_{ant}ant"
            if args.transit_target:
                cfg.start_time = f"transit:{args.transit_target}"
                dt = resolve_window_start(cfg.start_time, duration_s=cfg.duration_s)
                cfg.initial_lst_hours = datetime_to_lst_hours(dt)
                if args.beam_targets == "auto":
                    cfg.beam_targets = args.transit_target
            elif args.start_time:
                cfg.start_time = args.start_time
                dt = resolve_window_start(args.start_time, duration_s=cfg.duration_s)
                cfg.initial_lst_hours = datetime_to_lst_hours(dt)
            if args.beam_targets and not (
                args.transit_target and args.beam_targets == "auto"
            ):
                cfg.beam_targets = args.beam_targets

            ref_tag = None
            if args.save_reference:
                ref_tag = (
                    f"{args.save_reference}_{ant}ant"
                    if len(ant_list) > 1
                    else args.save_reference
                )
                cfg.save_reference = ref_tag

            res = generate_simulation_window(cfg)
            print(
                f"\n[DONE] Generated {res['num_written']} frames for {ant} antennas in {res['target_dir']}"
            )

            if ref_tag:
                save_as_reference_window(
                    src_window_dir=res["target_dir"],
                    tag=ref_tag,
                    description=f"CHARTS {ant}-Antenna Reference Baseband Window ({cfg.num_freq} Channels)",
                    obs_time=cfg.start_time,
                    duration_s=cfg.duration_s,
                    antennas=cfg.antennas,
                    num_freq=cfg.num_freq,
                    samples_per_frame=cfg.samples_per_frame,
                )

    elif args.command == "catalog":
        summary = catalog_transit_summary()
        print("=" * 105)
        print(
            " CHARTS CONFIRMED CELESTIAL TARGETS & TRANSIT EPHEMERIS (SIMBAD-VERIFIED)"
        )
        print("=" * 105)
        print(
            f" {'Target Label':<28} {'RA (deg)':>9} {'Dec (deg)':>10} {'Transit UTC':>20} {'Transit Loc':>13} {'Max Alt':>8} {'Vis (h)':>8}"
        )
        print("-" * 105)
        for t in summary:
            utc_str = t["transit_utc"].replace("T", " ")[:19]
            print(
                f" {t['label']:<28} {t['ra_deg']:>9.4f} {t['dec_deg']:>10.4f} "
                f"{utc_str:>20} {t['transit_local']:>13} {t['max_alt_deg']:>6.1f} deg {t['hours_above_mask']:>7.2f}h"
            )
        print("=" * 105)

    elif args.command == "reference":
        if args.action == "list":
            windows = list_reference_windows()
            print("=" * 90)
            print(" CHARTS REFERENCE BASEBAND WINDOW LIBRARY")
            print("=" * 90)
            if not windows:
                print(" No reference windows currently saved. Generate one with:")
                print(
                    "   python test_charts/charts_sim.py generate --save-reference <tag>"
                )
            else:
                for w in windows:
                    print(
                        f" * Tag: {w['tag']:<24} | Ant: {w['antennas']:<3} | Chans: {w['num_freq']:<4} | Frames: {w['total_frames']:<5} | Size: {w['size_mb']:.1f} MB"
                    )
                    print(
                        f"   Start: {w['observation_start']} (LST={w['lst_hours']}h) | Sources: {', '.join(w['active_sources'])}"
                    )
                    print(f"   Path:  {w['path']}\n")
            print("=" * 90)
        elif args.action == "info":
            if not args.tag:
                print("[ERROR] Please provide --tag <name> for reference info.")
                sys.exit(1)
            p, man = get_reference_window(args.tag)
            print(json.dumps(man.to_dict(), indent=2))

    elif args.command == "track":
        # Resolve target directory (direct or reference)
        if args.reference:
            wdir, man = get_reference_window(args.reference)
            antennas = man.antennas
            num_freq = man.num_freq
            samples_per_frame = man.samples_per_frame
            obs_lst = man.observation_start_lst_hours
            window_name = wdir.name
        elif args.window_dir:
            wdir = Path(args.window_dir)
            antennas = 64
            num_freq = 336
            samples_per_frame = 1536
            obs_lst = 5.575
            window_name = wdir.name
        else:
            print("[ERROR] Either --window-dir or --reference must be specified.")
            sys.exit(1)

        bin_frames = sorted(wdir.glob("*.bin"))
        if not bin_frames:
            print(f"[ERROR] No .bin frames found in {wdir}")
            sys.exit(1)

        num_frames = len(bin_frames)
        tracker_dir = wdir / "tracker"
        tracker_yaml = wdir / "kotekan_direct_tracker.yaml"

        beam_targets = parse_beam_targets(
            args.beam_targets,
            max_beams=args.max_beams,
            default_lst=obs_lst,
        )

        create_beam_tracker_yaml(
            yaml_path=tracker_yaml,
            baseband_dir=wdir,
            baseband_name=window_name,
            tracker_dir=tracker_dir,
            tracker_name=f"beams_{window_name}",
            num_frames=num_frames,
            beam_targets=beam_targets,
            num_elements=antennas,
            num_local_freq=num_freq,
            samples_per_data_set=samples_per_frame,
            max_beams=args.max_beams,
            stage_type="direct",
        )

        print(
            f"\n>>> Running Kotekan Direct Beam Tracker on {num_frames} frames in {wdir}..."
        )
        rc = execute_kotekan(
            config_path=tracker_yaml,
            kotekan_bin=Path(args.kotekan_bin) if args.kotekan_bin else None,
            dry_run=args.dry_run,
            log_path=wdir / "kotekan_direct_tracker.log",
        )
        if rc == 0 and not args.dry_run:
            tdumps = sorted(tracker_dir.glob("*.bin"))
            if tdumps:
                diag = inspect_tracker_dump(
                    tdumps[0],
                    num_freq=num_freq,
                    max_beams=args.max_beams,
                    samples_per_data_set=samples_per_frame,
                )
                print(f"[SUCCESS] Direct Beam Tracker executed cleanly.")
                print(f"  * Total Formed Beam Power: {diag['total_power']:.2f}")
                print(f"  * Peak Signal Beam Slot  : Beam {diag['peak_beam']}")

    elif args.command == "benchmark":
        run_direct_tracker_benchmark(
            ant_counts=args.antennas,
            beam_counts=args.beams,
            num_freq=args.num_freq,
            samples_per_frame=args.samples_per_frame,
            buffer_depth=args.buffer_depth,
            gpu_profile_name=args.target_gpu,
            output_json=Path(args.out_json) if args.out_json else None,
            output_md=Path(args.out_md) if args.out_md else None,
        )

    elif args.command == "inspect":
        fpath = Path(args.file)
        if args.type == "corr":
            vis = load_astron_correlator_dump(fpath)
            res = inspect_correlator_matrix(vis)
            print(json.dumps(res, indent=2))
        else:
            res = inspect_tracker_dump(fpath)
            print(json.dumps(res, indent=2))

    elif args.command == "visualize":
        wdir = Path(args.window_dir)
        corr_dir = wdir / "correlator"
        out_p = Path(args.out_file) if args.out_file else (wdir / f"{args.type}.png")

        if args.type in ["casm", "all"]:
            plot_casm_correlation_matrix(corr_dir, out_p)
            print(f"[DONE] Saved CASM plot to {out_p}")
        if args.type in ["waterfall", "all"]:
            plot_correlator_waterfalls(corr_dir, out_p)
            print(f"[DONE] Saved Waterfall plot to {out_p}")
        if args.type in ["tracker", "all"]:
            plot_tracker_waterfall(wdir / "tracker", out_p)
            print(f"[DONE] Saved Tracker plot to {out_p}")


if __name__ == "__main__":
    main()
