#!/usr/bin/env python3
"""CHARTS Realistic Observation Window Generator (Compatibility Layer).

Delegates to the centralized `sim.generator` engine while preserving full CLI argument parity.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

_test_charts_dir = Path(__file__).resolve().parent
if str(_test_charts_dir) not in sys.path:
    sys.path.insert(0, str(_test_charts_dir))

from sim.constants import (
    CHARTS_CHANNEL_WIDTH_MHZ,
    DEFAULT_FREQUENCY_START_MHZ,
)
from sim.generator import (
    SimulatedEvent,
    build_frame_selection_schedule,
    compute_analytic_lightcurve,
    generate_simulation_window,
    schedule_random_events,
)
from sim.presets import SimulationConfig


def generate_5min_window(
    utc_hour: int = 15,
    date_str: str = "2026-03-20",
    duration_s: float = 300.0,
    antennas: int = 64,
    num_freq: int = 672,
    samples_per_frame: int = 1536,
    background_cadence_s: float = 2.0,
    event_dense_s: float = 1.0,
    event_sparse_cadence_ms: float = 51.2,
    num_events: int = 3,
    event_types: list[str] | None = None,
    out_dir: Path | None = None,
    file_name: str | None = None,
    workers: int = 16,
    seed: int | None = None,
    sun_activity: str = "quiet",
    persistent_rfi_channels: list[int] | None = None,
    persistent_rfi_amp: float = 7.0,
):
    """Bridge function invoking centralized generate_simulation_window."""
    cfg = SimulationConfig(
        preset="custom",
        profile="day" if utc_hour == 15 else "night",
        window_name=file_name or f"win{utc_hour:02d}UTC_{antennas}ant",
        duration_s=duration_s,
        antennas=antennas,
        num_freq=num_freq,
        samples_per_frame=samples_per_frame,
        background_cadence_s=background_cadence_s,
        event_dense_s=event_dense_s,
        event_sparse_cadence_ms=event_sparse_cadence_ms,
        num_events=num_events,
        persistent_rfi_channels=persistent_rfi_channels or [],
        persistent_rfi_amp=persistent_rfi_amp,
        utc_hour=float(utc_hour),
        scratch_dir=(out_dir.parent if out_dir else Path(".")).resolve(),
        workers=workers,
    )
    if out_dir:
        cfg.window_name = out_dir.name
        cfg.scratch_dir = out_dir.parent

    return generate_simulation_window(cfg)


def main():
    parser = argparse.ArgumentParser(description="CHARTS Realistic Window Generator")
    parser.add_argument(
        "--utc-hour",
        type=int,
        default=15,
        choices=[15, 3],
        help="UTC hour (15 = Day, 3 = Night)",
    )
    parser.add_argument(
        "--date", type=str, default="2026-03-20", help="Observation date YYYY-MM-DD"
    )
    parser.add_argument(
        "--duration-s", type=float, default=300.0, help="Window duration in seconds"
    )
    parser.add_argument("--antennas", type=int, default=64, help="Number of antennas")
    parser.add_argument("--num-freq", type=int, default=672, help="Frequency channels")
    parser.add_argument(
        "--samples-per-frame", type=int, default=1536, help="Samples per frame"
    )
    parser.add_argument(
        "--background-cadence-s", type=float, default=2.0, help="Background cadence (s)"
    )
    parser.add_argument(
        "--event-dense-s", type=float, default=1.0, help="Dense capture duration (s)"
    )
    parser.add_argument(
        "--event-sparse-cadence-ms",
        type=float,
        default=51.2,
        help="Sparse cadence (ms)",
    )
    parser.add_argument(
        "--num-events", type=int, default=3, help="Number of transient events"
    )
    parser.add_argument(
        "--events", type=str, default=None, help="Comma-separated event types"
    )
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--file-name", type=str, default=None, help="Base file name")
    parser.add_argument("--workers", type=int, default=8, help="Workers")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed")
    parser.add_argument(
        "--sun-activity",
        type=str,
        default="quiet",
        choices=["quiet", "moderate", "active"],
    )
    parser.add_argument(
        "--persistent-rfi-channels",
        type=str,
        default="94,133,147",
        help="Persistent RFI channels",
    )
    parser.add_argument(
        "--persistent-rfi-freqs",
        type=str,
        default=None,
        help="Persistent RFI freqs (MHz)",
    )
    parser.add_argument(
        "--persistent-rfi-amp", type=float, default=7.0, help="Persistent RFI amplitude"
    )
    args = parser.parse_args()

    ev_types = [s.strip() for s in args.events.split(",")] if args.events else None
    persistent_chans = None
    if args.persistent_rfi_freqs:
        freq_list = [
            float(x.strip()) for x in args.persistent_rfi_freqs.split(",") if x.strip()
        ]
        persistent_chans = [
            int(round((f - DEFAULT_FREQUENCY_START_MHZ) / CHARTS_CHANNEL_WIDTH_MHZ))
            for f in freq_list
        ]
    elif (
        args.persistent_rfi_channels and args.persistent_rfi_channels.lower() != "none"
    ):
        persistent_chans = [
            int(x.strip())
            for x in args.persistent_rfi_channels.split(",")
            if x.strip().isdigit()
        ]

    generate_5min_window(
        utc_hour=args.utc_hour,
        date_str=args.date,
        duration_s=args.duration_s,
        antennas=args.antennas,
        num_freq=args.num_freq,
        samples_per_frame=args.samples_per_frame,
        background_cadence_s=args.background_cadence_s,
        event_dense_s=args.event_dense_s,
        event_sparse_cadence_ms=args.event_sparse_cadence_ms,
        num_events=args.num_events,
        event_types=ev_types,
        out_dir=Path(args.out_dir) if args.out_dir else None,
        file_name=args.file_name,
        workers=args.workers,
        seed=args.seed,
        sun_activity=args.sun_activity,
        persistent_rfi_channels=persistent_chans,
        persistent_rfi_amp=args.persistent_rfi_amp,
    )


if __name__ == "__main__":
    main()
