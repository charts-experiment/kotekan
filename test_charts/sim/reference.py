#!/usr/bin/env python3
"""CHARTS Reference Baseband Window Library & Manifest Generator.

Provides standardized cataloging and storage for reference baseband datasets:
  - Preserves reference observation windows for reproducible testing
  - Records comprehensive, machine-readable `window_manifest.json` metadata
  - Tracks observation start UTC, LST, duration, antenna layout, frequency channels,
    injected transients (FRBs, pulsars), RFI, and frame checksums
  - Enables direct replay into the Kotekan Correlator and Beam Tracker pipelines
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Union

from .astro import (
    datetime_to_lst_hours,
    equatorial_to_direction_cosines,
    equatorial_to_horizontal,
    parse_observation_time,
    resolve_beam_targets,
)
from .constants import (
    CHARTS_CHANNEL_WIDTH_MHZ,
    CHARTS_LATITUDE_DEG,
    CHARTS_LONGITUDE_DEG,
    DEFAULT_FREQUENCY_START_MHZ,
    FPGA_TIME_RESOLUTION_US,
)

logger = logging.getLogger("kotekan.charts.sim.reference")

DEFAULT_REFERENCE_LIBRARY_DIR = Path("test_charts/data/reference_windows")


def get_reference_library_dir() -> Path:
    """Detects default directory for reference baseband datasets."""
    env_dir = os.environ.get("CHARTS_REFERENCE_DIR")
    if env_dir:
        return Path(env_dir)
    return DEFAULT_REFERENCE_LIBRARY_DIR


@dataclass
class ReferenceWindowManifest:
    """Structured, reproducible manifest for a reference baseband observation window."""
    tag: str
    description: str
    created_at_utc: str
    observation_start_utc: str
    observation_start_lst_hours: float
    duration_s: float
    total_frames: int
    samples_per_frame: int
    frame_period_ms: float
    antennas: int
    num_freq: int
    frequency_start_mhz: float
    channel_width_mhz: float
    active_sources: List[Dict[str, Any]] = field(default_factory=list)
    injected_events: List[Dict[str, Any]] = field(default_factory=list)
    persistent_rfi_channels: List[int] = field(default_factory=list)
    frame_files: List[Dict[str, Any]] = field(default_factory=list)
    total_size_bytes: int = 0
    total_size_mb: float = 0.0
    git_commit: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def load(cls, path: Path) -> ReferenceWindowManifest:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


def compute_file_sha256(path: Path, max_bytes: int = 4096 * 1024) -> str:
    """Computes SHA256 checksum for first chunk of file (fast integrity check)."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        chunk = f.read(max_bytes)
        hasher.update(chunk)
    return hasher.hexdigest()[:16]


def create_reference_manifest(
    window_dir: Path,
    tag: str,
    description: str = "",
    obs_time: Optional[Union[str, datetime.datetime]] = None,
    duration_s: float = 0.0,
    antennas: int = 64,
    num_freq: int = 336,
    samples_per_frame: int = 1536,
    frequency_start_mhz: float = DEFAULT_FREQUENCY_START_MHZ,
    channel_width_mhz: float = CHARTS_CHANNEL_WIDTH_MHZ,
    active_sources: Optional[List[Dict[str, Any]]] = None,
    injected_events: Optional[List[Dict[str, Any]]] = None,
    persistent_rfi_channels: Optional[List[int]] = None,
) -> ReferenceWindowManifest:
    """Builds a ReferenceWindowManifest by scanning the window directory."""
    dt = parse_observation_time(obs_time)
    lst_h = datetime_to_lst_hours(dt)

    # Scan binary baseband frames
    bin_files = sorted(window_dir.glob("*.bin"))
    total_size = 0
    frame_entries = []

    for bf in bin_files:
        size = bf.stat().st_size
        total_size += size
        frame_entries.append({
            "name": bf.name,
            "size_bytes": size,
            "sha256_prefix": compute_file_sha256(bf),
        })

    frame_count = len(bin_files)
    frame_period_ms = (samples_per_frame * FPGA_TIME_RESOLUTION_US) / 1000.0

    # Auto-read events JSON if present and not provided
    if injected_events is None:
        injected_events = []
        events_json = list(window_dir.glob("*_events.json"))
        if events_json:
            try:
                with open(events_json[0], "r", encoding="utf-8") as f:
                    injected_events = json.load(f)
            except Exception as e:
                logger.warning(f"Could not read events JSON: {e}")

    # Auto-read sources from beam targets if not provided
    if active_sources is None:
        active_sources = resolve_beam_targets("auto", obs_time=dt, max_beams=4)

    # Determine git commit if in git repo
    git_rev = None
    try:
        import subprocess
        proc = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=window_dir)
        if proc.returncode == 0:
            git_rev = proc.stdout.strip()
    except Exception:
        pass

    manifest = ReferenceWindowManifest(
        tag=tag,
        description=description or f"CHARTS Reference Baseband Window: {tag}",
        created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        observation_start_utc=dt.isoformat(),
        observation_start_lst_hours=float(lst_h),
        duration_s=float(duration_s or (frame_count * frame_period_ms / 1000.0)),
        total_frames=frame_count,
        samples_per_frame=samples_per_frame,
        frame_period_ms=float(frame_period_ms),
        antennas=antennas,
        num_freq=num_freq,
        frequency_start_mhz=float(frequency_start_mhz),
        channel_width_mhz=float(channel_width_mhz),
        active_sources=active_sources,
        injected_events=injected_events,
        persistent_rfi_channels=persistent_rfi_channels or [],
        frame_files=frame_entries,
        total_size_bytes=total_size,
        total_size_mb=round(total_size / (1024.0 * 1024.0), 2),
        git_commit=git_rev,
    )

    return manifest


def save_as_reference_window(
    src_window_dir: Path,
    tag: str,
    description: str = "",
    reference_dir: Optional[Path] = None,
    move_files: bool = False,
    **kwargs: Any,
) -> Path:
    """Stores a generated simulation window into the Reference Window Library.

    Copies (or moves) all binary frames, HDF5 metadata, and event catalogs,
    and writes `window_manifest.json`.
    """
    target_lib = reference_dir or get_reference_library_dir()
    dest_dir = target_lib / tag
    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n>>> Archiving Reference Window '{tag}' to {dest_dir}...")

    # Copy / Move .bin frames
    bin_files = sorted(src_window_dir.glob("*.bin"))
    for bf in bin_files:
        target = dest_dir / bf.name
        if move_files:
            shutil.move(str(bf), str(target))
        else:
            shutil.copy2(bf, target)

    # Copy HDF5 and JSON metadata
    for ext in ("*.h5", "*.json", "*.yaml"):
        for mf in src_window_dir.glob(ext):
            if mf.name != "window_manifest.json":
                target = dest_dir / mf.name
                shutil.copy2(mf, target)

    # Generate and write manifest inside the reference directory
    manifest = create_reference_manifest(
        window_dir=dest_dir,
        tag=tag,
        description=description,
        **kwargs,
    )
    manifest_path = dest_dir / "window_manifest.json"
    manifest.save(manifest_path)

    print(f"  * Created reference manifest : {manifest_path}")
    print(f"  * Total frames stored        : {manifest.total_frames}")
    print(f"  * Total dataset size         : {manifest.total_size_mb} MB")

    return dest_dir


def list_reference_windows(reference_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Discovers and lists all registered reference windows."""
    target_lib = reference_dir or get_reference_library_dir()
    if not target_lib.is_dir():
        return []

    summaries = []
    for item in sorted(target_lib.iterdir()):
        if item.is_dir():
            manifest_file = item / "window_manifest.json"
            if manifest_file.is_file():
                try:
                    man = ReferenceWindowManifest.load(manifest_file)
                    summaries.append({
                        "tag": man.tag,
                        "description": man.description,
                        "observation_start": man.observation_start_utc,
                        "lst_hours": round(man.observation_start_lst_hours, 2),
                        "duration_s": round(man.duration_s, 2),
                        "total_frames": man.total_frames,
                        "antennas": man.antennas,
                        "num_freq": man.num_freq,
                        "size_mb": man.total_size_mb,
                        "active_sources": [s.get("name", "Unknown") for s in man.active_sources[:3]],
                        "events_count": len(man.injected_events),
                        "path": str(item),
                    })
                except Exception as e:
                    logger.warning(f"Error parsing reference manifest at {manifest_file}: {e}")

    return summaries


def get_reference_window(
    tag_or_path: Union[str, Path],
    reference_dir: Optional[Path] = None,
) -> Tuple[Path, ReferenceWindowManifest]:
    """Retrieves path and manifest for a given reference window tag or path."""
    p = Path(tag_or_path)
    if p.is_dir() and (p / "window_manifest.json").is_file():
        return p, ReferenceWindowManifest.load(p / "window_manifest.json")

    target_lib = reference_dir or get_reference_library_dir()
    cand = target_lib / str(tag_or_path)
    if cand.is_dir() and (cand / "window_manifest.json").is_file():
        return cand, ReferenceWindowManifest.load(cand / "window_manifest.json")

    raise FileNotFoundError(
        f"Reference window '{tag_or_path}' not found in {target_lib} or as a filesystem directory."
    )
