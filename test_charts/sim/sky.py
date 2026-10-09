#!/usr/bin/env python3
r"""CHARTS Unified Sky & Time Model.

Single source of truth connecting three previously disconnected systems:

1. The **verified target catalog** (`tools/verified_targets.json`): SIMBAD-verified
   RA/Dec and transit times confirmed operationally by the tracker-viewer routine
   (`direct-beam-tracker-viewer/src/viewer/routine.py`).
2. The **Kotekan X-engine astrometry**: `compute_celestial_direction()` in
   `lib/cuda/cudaDirectBeamTracker.hpp`, which converts (RA, Dec, LST) into
   topocentric direction cosines at command time and re-evaluates them every frame.
3. The **simulation generator** (`sim/generator.py`), which must inject sources with
   the same time-evolving geometry the X-engine tracks.

Continuous Mathematical Model
----------------------------
For a source at equatorial coordinates $(\alpha, \delta)$ observed at local
sidereal time $H_\mathrm{LST}$ from site latitude $\phi$, the hour angle is
$H = \mathrm{LST} - \alpha$ and the topocentric direction cosines are:

.. math::

    l(t) = -\cos\delta \,\sin H(t)
    \qquad
    m(t) = \cos\phi \,\sin\delta - \sin\phi \,\cos\delta \,\cos H(t)

    \phi_a(f, t) = -2\pi f \, \frac{x_a\,l(t) + y_a\,m(t)}{c}

This is exactly the formula Kotekan evaluates per frame (with sub-frame linear
interpolation between frame-start and frame-end directions), so a source injected
with this model is *trackable* by the real X-engine: the tracker's conjugate
weights cohere the sum as the source transits.

Theory & Literature:
  - Buschmann, B. A. P. (2025). "Design and Implementation of the F-Engine for
    CHARTS", §4 (geometric delay and fringe phase conventions).
  - Thompson, A. R., Moran, J. M., Swenson, G. W. (2017). "Interferometry and
    Synthesis in Radio Astronomy", 3rd ed., §4 (hour angle / direction cosines).
"""

from __future__ import annotations

import datetime
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .astro import datetime_to_lst_hours, parse_observation_time
from .constants import (
    C_LIGHT,
    CHARTS_LATITUDE_DEG,
    CHARTS_LONGITUDE_DEG,
)

logger = logging.getLogger("kotekan.charts.sim.sky")

# Repository layout: test_charts/sim/sky.py -> kotekan/test_charts -> kotekan -> charts
_SIM_DIR = Path(__file__).resolve().parent
_VERIFIED_TARGETS_CANDIDATES = [
    _SIM_DIR / "verified_targets.json",
    _SIM_DIR.parent / "verified_targets.json",
    _SIM_DIR.parent.parent.parent / "tools" / "verified_targets.json",
    _SIM_DIR.parent.parent / "tools" / "verified_targets.json",
    Path("tools") / "verified_targets.json",
]


# ---------------------------------------------------------------------------
# Verified target catalog
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VerifiedTarget:
    """One SIMBAD-verified catalog target with its confirmed transit."""
    label: str
    ra_deg: float
    dec_deg: float
    transit_local: str
    transit_utc: str
    max_alt_deg: Optional[float] = None
    rise_local: str = ""
    set_local: str = ""
    hours_above_mask: float = 0.0
    simbad_id: str = ""

    @property
    def transit_dt(self) -> datetime.datetime:
        dt = datetime.datetime.fromisoformat(self.transit_utc.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        return dt

    def direction_cosines_at_transit(
        self,
        lat_deg: float = CHARTS_LATITUDE_DEG,
    ) -> Tuple[float, float]:
        """Direction cosines (l, m) at the confirmed transit time."""
        lst_h = datetime_to_lst_hours(self.transit_dt)
        l, m, _ = direction_cosines_track(
            self.ra_deg, self.dec_deg, np.array([self.transit_dt.timestamp()]),
            lat_deg=lat_deg,
        )
        return float(l[0]), float(m[0])


def _default_catalog_path() -> Optional[Path]:
    for cand in _VERIFIED_TARGETS_CANDIDATES:
        if cand.is_file():
            return cand
    return None


def load_verified_catalog(path: Optional[Union[str, Path]] = None) -> Dict[str, VerifiedTarget]:
    """Loads the SIMBAD-verified target catalog confirmed by the tracker-viewer routine.

    Returns a dict keyed by lowercased label, e.g. ``"vela snr / psr b0833-45"``.
    Use :func:`find_verified_target` for fuzzy name lookup.
    """
    cat_path = Path(path) if path is not None else _default_catalog_path()
    if cat_path is not None and cat_path.is_file():
        with open(cat_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        try:
            from .verified_catalog_data import VERIFIED_TARGETS_RAW
            data = json.loads(VERIFIED_TARGETS_RAW) if isinstance(VERIFIED_TARGETS_RAW, str) else VERIFIED_TARGETS_RAW
        except Exception:
            raise FileNotFoundError(
                "verified_targets.json not found on disk and fallback catalog is unavailable. "
                "Expected at 'tools/verified_targets.json' relative to the charts workspace root."
            )

    catalog: Dict[str, VerifiedTarget] = {}
    for t in data.get("targets", []):
        vt = VerifiedTarget(
            label=str(t["label"]),
            ra_deg=float(t["ra_deg"]),
            dec_deg=float(t["dec_deg"]),
            transit_local=str(t.get("transit_local", "")),
            transit_utc=str(t.get("transit_utc", "")),
            max_alt_deg=float(t["max_alt_deg"]) if t.get("max_alt_deg") is not None else None,
            rise_local=str(t.get("rise_local", "")),
            set_local=str(t.get("set_local", "")),
            hours_above_mask=float(t.get("hours_above_mask", 0.0)),
            simbad_id=str(t.get("simbad_id", "")),
        )
        catalog[vt.label.strip().lower()] = vt
    return catalog


def find_verified_target(
    name: str,
    catalog: Optional[Dict[str, VerifiedTarget]] = None,
) -> VerifiedTarget:
    """Fuzzy-finds a verified target by name (case-insensitive substring match).

    Examples: ``"vela"`` -> Vela SNR / PSR B0833-45, ``"j0437"`` -> PSR J0437-4715,
    ``"crab"`` / ``"taurus"`` -> Taurus A (Crab), ``"fornix"`` -> Fornax A.
    """
    cat = catalog if catalog is not None else load_verified_catalog()
    key = name.strip().lower()

    # Exact label match first
    if key in cat:
        return cat[key]

    # Substring match on label or simbad_id
    matches = [v for k, v in cat.items() if key in k or (v.simbad_id and key in v.simbad_id.lower())]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        # Prefer the match where the query appears at a word start
        for v in matches:
            words = v.label.lower().replace("/", " ").replace("(", " ").replace(")", " ").split()
            if any(w.startswith(key) for w in words):
                return v
        return matches[0]

    # Normalized alphanumeric fallback (e.g. 'sgra' -> 'sgr a* / galactic center')
    norm_key = "".join(c for c in key if c.isalnum())
    if norm_key:
        norm_matches = [
            v for k, v in cat.items()
            if norm_key in "".join(c for c in k if c.isalnum())
        ]
        if norm_matches:
            return norm_matches[0]

    raise KeyError(
        f"No verified target matching '{name}'. Available: {sorted(v.label for v in cat.values())}"
    )


# ---------------------------------------------------------------------------
# Time-evolving direction cosines (Kotekan parity)
# ---------------------------------------------------------------------------

def direction_cosines_track(
    ra_deg: float,
    dec_deg: float,
    unix_times_s: Union[float, np.ndarray, Sequence[float]],
    lat_deg: float = CHARTS_LATITUDE_DEG,
    lon_deg: float = CHARTS_LONGITUDE_DEG,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Vectorized topocentric direction cosines (l, m, n) over absolute UTC times.

    Bit-identical formula to Kotekan's ``compute_celestial_direction``
    (lib/cuda/cudaDirectBeamTracker.hpp):

    .. math::

        H = \mathrm{LST}(t) - \alpha
        \qquad
        l = -\cos\delta \sin H
        \qquad
        m = \cos\phi \sin\delta - \sin\phi \cos\delta \cos H

    Parameters
    ----------
    ra_deg, dec_deg:
        Equatorial source coordinates (J2000, degrees).
    unix_times_s:
        Absolute UTC times as Unix seconds (scalar or array).
    lat_deg, lon_deg:
        Observatorio Carén site coordinates.

    Returns
    -------
    (l, m, n):
        Float64 arrays of the same shape as ``unix_times_s``.
    """
    t = np.atleast_1d(np.asarray(unix_times_s, dtype=np.float64))

    # GMST/LST: same constants as kotekan cudaBeamTrackerCommand.cpp and astro.py
    d = t / 86400.0 - 10957.5
    gmst_hours = (18.697374558 + 24.06570982441908 * d) % 24.0
    lst_hours = (gmst_hours + lon_deg / 15.0) % 24.0

    ha_rad = (lst_hours * (math.pi / 12.0)) - ra_deg * (math.pi / 180.0)
    dec_rad = dec_deg * (math.pi / 180.0)
    lat_rad = lat_deg * (math.pi / 180.0)

    sin_dec = math.sin(dec_rad)
    cos_dec = math.cos(dec_rad)
    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)

    l = -cos_dec * np.sin(ha_rad)
    m = cos_lat * sin_dec - sin_lat * cos_dec * np.cos(ha_rad)
    trans_sq = l * l + m * m
    n = np.sqrt(np.clip(1.0 - trans_sq, 0.0, None))

    return l, m, n


def geometric_phase_track(
    l_track: np.ndarray,
    m_track: np.ndarray,
    pos_x_m: np.ndarray,
    pos_y_m: np.ndarray,
    freqs_hz: np.ndarray,
) -> np.ndarray:
    r"""Geometric phase $\phi_a(f, t) = 2\pi f (x_a l(t) + y_a m(t)) / c$.

    Parameters
    ----------
    l_track, m_track:
        Direction cosines per time sample, shape (n_t,).
    pos_x_m, pos_y_m:
        Antenna positions in meters, shape (n_ant,).
    freqs_hz:
        Channel center frequencies, shape (n_freq,).

    Returns
    -------
    phases:
        Shape (n_t, n_freq, n_ant) float64. The source voltage injected by the
        generator is $A \exp(-j\phi)$ so that the X-engine's conjugate weights
        ($w = \exp(+j\phi)$, see ``generate_steering_weights_kernel``) cohere.
    """
    delays = (np.outer(l_track, pos_x_m) + np.outer(m_track, pos_y_m)) / C_LIGHT  # (n_t, n_ant)
    return (2.0 * math.pi) * (freqs_hz[:, None] * delays[:, None, :])  # (n_t, n_freq, n_ant)


# ---------------------------------------------------------------------------
# Transit-anchored observation time resolution
# ---------------------------------------------------------------------------

def resolve_window_start(
    time_spec: Optional[Union[str, datetime.datetime, float]],
    duration_s: float = 0.0,
    catalog: Optional[Dict[str, VerifiedTarget]] = None,
) -> datetime.datetime:
    """Resolves an observation start time, supporting transit-anchored specs.

    Supported specifications:
      - ``None`` / ``"now"``: current UTC time.
      - ISO 8601 string: ``"2026-10-07T12:15:02Z"``.
      - ``"HH:MM"`` / ``"HH:MM:SS"``: anchored to the verified catalog date.
      - Float UTC hour (e.g. ``15.0``): anchored to the verified catalog date.
      - ``"transit:<Target>"``: window **centered** on the target's confirmed
        transit UTC time (e.g. ``"transit:Vela"``).
      - ``"transit+<delta_s>:<Target>"`` / ``"transit-<delta_s>:<Target>"``:
        transit time plus/minus an offset in seconds (window still centered
        means start = transit - duration/2 + delta).
      - ``"slot:HH:MM"``: local routine set-hour slot (tracker-viewer routine
        convention), converted to UTC via the catalog date and site longitude.

    Parameters
    ----------
    time_spec:
        The time specification (see above).
    duration_s:
        Window duration in seconds; transit-anchored windows start at
        ``transit - duration_s / 2`` so the transit falls at mid-window.
    catalog:
        Pre-loaded verified catalog (loaded on demand otherwise).
    """
    if time_spec is None or (isinstance(time_spec, str) and time_spec.strip().lower() == "now"):
        return datetime.datetime.now(datetime.timezone.utc)

    if isinstance(time_spec, datetime.datetime):
        return parse_observation_time(time_spec)

    if isinstance(time_spec, (int, float)) and not isinstance(time_spec, bool):
        return _anchor_to_catalog_date(float(time_spec), catalog)

    s = str(time_spec).strip()

    # Transit-anchored: "transit:Vela", "transit+30:Crab", "transit-10.5:PSR J0437-4715"
    if s.lower().startswith("transit"):
        cat = catalog if catalog is not None else load_verified_catalog()
        body = s.split(":", 1)[1] if ":" in s else ""
        offset_s = 0.0
        head = s.split(":", 1)[0]
        if head.lower() != "transit":
            offset_s = float(head[len("transit"):])
        if not body:
            raise ValueError(f"Transit spec '{s}' needs a target name: 'transit:<Target>'")
        target = find_verified_target(body, catalog=cat)
        start = target.transit_dt - datetime.timedelta(seconds=duration_s / 2.0 + offset_s)
        return start

    # Routine slot: "slot:09:00" (local solar set-hour, viewer-routine convention)
    if s.lower().startswith("slot:"):
        return _resolve_local_slot(s.split(":", 1)[1], catalog)

    # HH:MM / HH:MM:SS / float hour / ISO
    return _anchor_to_catalog_date(s, catalog)


def _catalog_date(catalog: Optional[Dict[str, VerifiedTarget]] = None) -> datetime.datetime:
    """The catalog reference date (e.g. 2026-10-07) at 00:00 UTC."""
    cat = catalog if catalog is not None else load_verified_catalog()
    any_target = next(iter(cat.values()))
    return any_target.transit_dt.replace(hour=0, minute=0, second=0, microsecond=0)


def _anchor_to_catalog_date(
    hour_spec: Union[str, float],
    catalog: Optional[Dict[str, VerifiedTarget]] = None,
) -> datetime.datetime:
    """Anchors 'HH:MM' / float-hour specs to the verified catalog date."""
    base = _catalog_date(catalog)
    if isinstance(hour_spec, (int, float)):
        h = int(hour_spec) % 24
        m = int((hour_spec - int(hour_spec)) * 60)
        sec = int(round(((hour_spec - int(hour_spec)) * 60 - m) * 60))
        return base.replace(hour=h, minute=m, second=sec, microsecond=0)
    s = str(hour_spec)
    parts = s.split(":")
    if len(parts) in (2, 3):
        try:
            h = int(parts[0])
            m = int(parts[1])
            sec = int(parts[2]) if len(parts) == 3 else 0
            return base.replace(hour=h, minute=m, second=sec, microsecond=0)
        except ValueError:
            pass
    # Fall back to ISO / generic parsing (astro.parse_observation_time)
    return parse_observation_time(s)


def _resolve_local_slot(
    slot_local: str,
    catalog: Optional[Dict[str, VerifiedTarget]] = None,
) -> datetime.datetime:
    """Converts a local routine set-hour slot ('09:00') to UTC on the catalog date.

    The tracker-viewer routine and the verified catalog both work in local
    *clock* time (Chile UTC-3 during the catalog date). The UTC offset is
    derived from the catalog itself (transit_utc vs. transit_local) rather
    than hardcoded, so the routine's set-hour table and the simulation share
    the same convention.
    """
    cat = catalog if catalog is not None else load_verified_catalog()
    base = _catalog_date(cat)
    parts = slot_local.strip().split(":")
    h = int(parts[0])
    m = int(parts[1]) if len(parts) > 1 else 0
    local_dt = base.replace(hour=h % 24, minute=m, second=0, microsecond=0)
    utc_dt = local_dt + _catalog_local_utc_offset(cat)
    return utc_dt


def _catalog_local_utc_offset(
    catalog: Optional[Dict[str, VerifiedTarget]] = None,
) -> datetime.timedelta:
    """Local-clock minus UTC offset, derived from catalog transit entries.

    For each target, transit_utc - transit_local gives the site's clock offset
    (e.g. 3h for Chile DST). Day markers like '(-1d)' in transit_local are
    irrelevant to the offset and ignored by the HH:MM regex.
    """
    import re

    cat = catalog if catalog is not None else load_verified_catalog()
    offsets = []
    for v in cat.values():
        match = re.search(r"(\d{1,2}):(\d{2})", v.transit_local)
        if not match:
            continue
        local_h = int(match.group(1)) + int(match.group(2)) / 60.0
        utc_h = v.transit_dt.hour + v.transit_dt.minute / 60.0 + v.transit_dt.second / 3600.0
        offsets.append((utc_h - local_h) % 24.0)
    if not offsets:
        return datetime.timedelta(hours=-CHARTS_LONGITUDE_DEG / 15.0)
    # transit_local is rounded to minutes, so round the offset to the routine's
    # own granularity to avoid leaking transit seconds into slot times.
    offset_h = float(np.median(offsets))
    offset_min = round(offset_h * 60.0)
    return datetime.timedelta(minutes=offset_min)


def catalog_transit_summary(
    catalog: Optional[Dict[str, VerifiedTarget]] = None,
) -> List[Dict[str, Any]]:
    """Returns the verified catalog as a sorted, JSON-friendly summary list."""
    cat = catalog if catalog is not None else load_verified_catalog()
    entries = []
    for v in cat.values():
        entries.append({
            "label": v.label,
            "ra_deg": v.ra_deg,
            "dec_deg": v.dec_deg,
            "transit_utc": v.transit_utc,
            "transit_local": v.transit_local,
            "max_alt_deg": v.max_alt_deg,
            "hours_above_mask": v.hours_above_mask,
        })
    entries.sort(key=lambda e: e["transit_utc"])
    return entries
