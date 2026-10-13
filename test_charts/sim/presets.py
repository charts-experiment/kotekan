#!/usr/bin/env python3
"""CHARTS Simulation Profiles, Presets, and Configuration Manager.

Provides standard hardware profiles, telescope schedules, and presets
allowing simulations to run seamlessly on both local workstations (e.g. RTX 4090/5090)
and HPC cluster environments (e.g. Trillium).
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

from .constants import (
    CHARTS_N_FREQ,
    DEFAULT_ANTENNAS,
    DEFAULT_FREQUENCY_START_MHZ,
    LOCAL_FREQUENCY_CHANNELS,
)


@dataclass
class SimulationConfig:
    """Master configuration for CHARTS realistic simulation and pipeline."""

    # Identification
    preset: str = "quick"  # 'quick', '1min', '5min', 'custom'
    profile: str = "day"  # 'day', 'night', 'both'
    window_name: str = "charts_sim"

    # Time & Decimation
    duration_s: float = 5.0
    background_cadence_s: float = 1.0
    event_dense_s: float = 0.5
    event_sparse_cadence_ms: float = 50.0

    # Telescope array geometry
    antennas: int = 64
    num_freq: int = 336
    samples_per_frame: int = 1536
    frequency_start_mhz: float = DEFAULT_FREQUENCY_START_MHZ

    # Transients & RFI
    num_events: int = 1
    allowed_event_types: Optional[List[str]] = None
    persistent_rfi_channels: List[int] = field(default_factory=lambda: [94, 133, 147])
    persistent_rfi_freqs: Optional[List[float]] = None
    persistent_rfi_amp: float = 7.0

    # Kotekan Pipeline & GPU Buffer Parameters
    max_beams: int = 4
    integration_spectra: int = 320
    buffer_depth: int = 2  # Conservative default: fits 16-24GB VRAM
    beam_targets: str = ""
    utc_hour: float = 15.0
    initial_lst_hours: float = 5.575
    start_time: Optional[str] = None
    save_reference: Optional[str] = None
    skip_tracker: bool = False

    # System & Execution Paths
    scratch_dir: Path = field(default_factory=lambda: Path("./scratch_charts_sim"))
    output_dir: Path = field(
        default_factory=lambda: Path("./test_charts/data/sim_output")
    )
    kotekan_bin: Optional[Path] = None
    workers: int = 4
    dry_run: bool = False

    # Visualizations & Video
    generate_plots: bool = True
    generate_waterfalls: bool = True
    generate_casm: bool = True
    generate_videos: bool = False
    video_duration_s: float = 5.0
    video_fps: int = 5

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary with stringified paths."""
        res = asdict(self)
        res["scratch_dir"] = str(self.scratch_dir)
        res["output_dir"] = str(self.output_dir)
        res["kotekan_bin"] = str(self.kotekan_bin) if self.kotekan_bin else None
        return res

    def save_yaml(self, path: Path):
        """Save configuration to YAML file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)


# Standard Target Schedules for Daytime (15:00 UTC) and Nighttime (03:00 UTC)
BEAM_TARGETS_DAY = (
    "Crab Pulsar:83.633,22.014;"
    "PSR J0437-4715:69.316,-47.252;"
    "Pictor A:79.957,-45.779;"
    "Orion M42:83.822,-5.391;"
    "Canopus:95.988,-52.696;"
    "Sirius:101.287,-16.716;"
    "PSR B0656+14:104.951,14.238;"
    "LMC J0537:84.440,-69.170"
)

BEAM_TARGETS_NIGHT = (
    "Sgr A* (GC):266.417,-29.008;"
    "Centaurus A:201.365,-43.019;"
    "PSR B1642-03:251.272,-3.371;"
    "Sco X-1:244.979,-15.640;"
    "PSR B1749-28:268.064,-28.106;"
    "PSR B1818-04:275.318,-4.292;"
    "M87 Virgo A:187.706,12.391;"
    "PSR B1937+21:294.911,21.583"
)


def get_default_scratch_dir() -> Path:
    """Intelligently detects scratch storage location across PC and cluster nodes."""
    # 1. Explicit user override
    if "SCRATCH_DIR" in os.environ:
        return Path(os.environ["SCRATCH_DIR"])
    # 2. Local SSD/NVMe $SLURM_TMPDIR (only if NOT on /dev/shm tmpfs)
    slurm_tmp = os.environ.get("SLURM_TMPDIR")
    if slurm_tmp and not slurm_tmp.startswith("/dev/shm"):
        return Path(slurm_tmp)
    # 3. HPC persistent scratch (e.g. $SCRATCH)
    scratch_env = os.environ.get("SCRATCH")
    if scratch_env:
        return Path(scratch_env) / "kotekan_sim_scratch"
    # 4. Standard local fallback for PC
    return Path("./scratch_charts_sim")


def get_default_workers() -> int:
    """Detects available CPU threads appropriately for local PC or Slurm job."""
    if "SLURM_CPUS_PER_TASK" in os.environ:
        try:
            return max(1, int(os.environ["SLURM_CPUS_PER_TASK"]))
        except ValueError:
            pass
    cpu_count = os.cpu_count() or 4
    # On PC, default to min(8, cpu_count) so machine stays responsive
    return max(1, min(8, cpu_count))


def find_kotekan_binary() -> Optional[Path]:
    """Finds kotekan executable in standard build directories or PATH."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    candidates = [
        repo_root / "build" / "kotekan" / "kotekan",
        repo_root / "build" / "kotekan" / "kotekan.exe",
        repo_root.parent / "kotekan" / "build" / "kotekan" / "kotekan",
        Path("./build/kotekan/kotekan"),
    ]
    for cand in candidates:
        if cand.is_file():
            return cand.resolve()
    # Check PATH
    import shutil

    which_bin = shutil.which("kotekan")
    if which_bin:
        return Path(which_bin).resolve()
    return None


def get_preset_config(
    preset: str = "quick",
    profile: str = "day",
    antennas: Optional[int] = None,
    num_freq: Optional[int] = None,
    custom_overrides: Optional[Dict[str, Any]] = None,
) -> SimulationConfig:
    """Builds a fully populated SimulationConfig based on preset and profile."""
    preset = preset.lower()
    profile = profile.lower()

    cfg = SimulationConfig(
        preset=preset,
        profile=profile,
        scratch_dir=get_default_scratch_dir(),
        workers=get_default_workers(),
        kotekan_bin=find_kotekan_binary(),
    )

    # 1. Apply Preset defaults
    if preset == "quick":
        cfg.duration_s = 2.0
        cfg.background_cadence_s = 1.0
        cfg.event_dense_s = 0.02
        cfg.event_sparse_cadence_ms = 50.0
        cfg.num_events = 1
        cfg.num_freq = 336
        cfg.max_beams = 4
        cfg.buffer_depth = 2
        cfg.video_duration_s = 2.0
        cfg.video_fps = 5
        cfg.generate_videos = False

    elif preset == "1min":
        cfg.duration_s = 60.0
        cfg.background_cadence_s = 2.0
        cfg.event_dense_s = 1.0
        cfg.event_sparse_cadence_ms = 100.0
        cfg.num_events = 8
        cfg.num_freq = 672
        cfg.max_beams = 8
        cfg.buffer_depth = 2  # Safe for 24GB RTX 4090
        cfg.video_duration_s = 60.0
        cfg.video_fps = 10
        cfg.generate_videos = True

    elif preset == "5min":
        cfg.duration_s = 300.0
        cfg.background_cadence_s = 2.0
        cfg.event_dense_s = 1.0
        cfg.event_sparse_cadence_ms = 100.0
        cfg.num_events = 8
        cfg.num_freq = 672
        cfg.max_beams = 8
        cfg.buffer_depth = 3
        cfg.video_duration_s = 60.0
        cfg.video_fps = 10
        cfg.generate_videos = True

    # 2. Apply Profile (Day / Night)
    if profile == "day":
        cfg.utc_hour = 15.0
        cfg.initial_lst_hours = 5.575
        cfg.beam_targets = BEAM_TARGETS_DAY
        cfg.window_name = f"win15UTC_{cfg.antennas}ant"
    elif profile == "night":
        cfg.utc_hour = 3.0
        cfg.initial_lst_hours = 17.5
        cfg.beam_targets = BEAM_TARGETS_NIGHT
        cfg.window_name = f"win03UTC_{cfg.antennas}ant"

    # 3. Explicit overrides
    if antennas is not None:
        cfg.antennas = antennas
        cfg.window_name = (
            f"{'win15UTC' if cfg.utc_hour == 15 else 'win03UTC'}_{cfg.antennas}ant"
        )
    if num_freq is not None:
        cfg.num_freq = num_freq

    if custom_overrides:
        for k, v in custom_overrides.items():
            if hasattr(cfg, k) and v is not None:
                setattr(cfg, k, v)

    return cfg
