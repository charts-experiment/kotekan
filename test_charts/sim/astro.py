#!/usr/bin/env python3
"""CHARTS Astronomical Coordinates, Ephemeris, and Celestial Target Catalog.

Provides exact astrometric calculations for the CHARTS telescope at Observatorio Carén:
  - Latitude:  -33.4500 deg (-33° 27' 00" S)
  - Longitude: -70.8500 deg (-70° 51' 00" W)
  - Elevation: 458 m

Features:
  - Exact GMST and LST calculation mathematically identical to Kotekan C++ core
  - Topocentric (Azimuth, Elevation) and Direction Cosines (l, m, n) computation
  - Direction rates (dl, dm) per sample for phase-tracking beam steering
  - Standard celestial calibrator, pulsar, and solar catalog
  - Dynamic source visibility ranking and automatic beam target selection
  - Flexible target specification parsing (names, coords, offsets, auto)
"""

from __future__ import annotations

import datetime
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from .constants import (
    CHARTS_LATITUDE_DEG,
    CHARTS_LONGITUDE_DEG,
    SAMPLE_RATE_HZ,
)


@dataclass(frozen=True)
class CelestialSource:
    """Immutable celestial source definition."""

    name: str
    ra_deg: float
    dec_deg: float
    flux_jy_400: float = 0.0  # Estimated flux density at 400 MHz (Jansky)
    source_type: str = "calibrator"  # 'calibrator', 'pulsar', 'frb_host', 'solar'
    description: str = ""


# Comprehensive catalog of standard calibrators and prominent pulsars
CELESTIAL_CATALOG: Dict[str, CelestialSource] = {
    # Primary Southern and Equatorial Radio Calibrators
    "Crab": CelestialSource(
        name="Crab (Tau A / PSR B0531+21)",
        ra_deg=83.6331,
        dec_deg=22.0145,
        flux_jy_400=850.0,
        source_type="calibrator",
        description="Crab Nebula and millisecond pulsar (Taurus A)",
    ),
    "Vela": CelestialSource(
        name="Vela (PSR B0833-45)",
        ra_deg=128.8359,
        dec_deg=-45.1764,
        flux_jy_400=1100.0,
        source_type="pulsar",
        description="Bright southern pulsar in Vela Supernova Remnant",
    ),
    "SgrA": CelestialSource(
        name="Sgr A* (Galactic Center)",
        ra_deg=266.4168,
        dec_deg=-29.0078,
        flux_jy_400=950.0,
        source_type="calibrator",
        description="Supermassive black hole at the Galactic Center",
    ),
    "CenA": CelestialSource(
        name="Centaurus A (NGC 5128)",
        ra_deg=201.3651,
        dec_deg=-43.0191,
        flux_jy_400=650.0,
        source_type="calibrator",
        description="Giant radio galaxy with bright relativistic jets",
    ),
    "PictorA": CelestialSource(
        name="Pictor A",
        ra_deg=79.9572,
        dec_deg=-45.7788,
        flux_jy_400=380.0,
        source_type="calibrator",
        description="Powerful FR II radio galaxy in southern sky",
    ),
    "PuppisA": CelestialSource(
        name="Puppis A",
        ra_deg=125.6875,
        dec_deg=-42.9972,
        flux_jy_400=130.0,
        source_type="calibrator",
        description="Southern galactic supernova remnant",
    ),
    "HydraA": CelestialSource(
        name="Hydra A (3C 218)",
        ra_deg=139.5238,
        dec_deg=-12.0956,
        flux_jy_400=280.0,
        source_type="calibrator",
        description="Standard flux calibrator in Hydra galaxy cluster",
    ),
    "CassiopeiaA": CelestialSource(
        name="Cassiopeia A (3C 461)",
        ra_deg=350.8500,
        dec_deg=58.8150,
        flux_jy_400=2400.0,
        source_type="calibrator",
        description="Brightest extra-solar radio source (Northern low horizon)",
    ),
    "VirgoA": CelestialSource(
        name="Virgo A (M87 / 3C 274)",
        ra_deg=187.7059,
        dec_deg=12.3911,
        flux_jy_400=210.0,
        source_type="calibrator",
        description="Giant elliptical galaxy with relativistic jet in Virgo",
    ),
    # Prominent Southern and Timing Pulsars
    "PSR_J0437-4715": CelestialSource(
        name="PSR J0437-4715",
        ra_deg=69.3162,
        dec_deg=-47.2525,
        flux_jy_400=150.0,
        source_type="pulsar",
        description="Closest and brightest millisecond pulsar in southern sky",
    ),
    "PSR_B1642-03": CelestialSource(
        name="PSR B1642-03",
        ra_deg=251.2721,
        dec_deg=-3.3039,
        flux_jy_400=110.0,
        source_type="pulsar",
        description="Bright, highly stable slow pulsar",
    ),
    "PSR_B0950+08": CelestialSource(
        name="PSR B0950+08",
        ra_deg=148.2888,
        dec_deg=7.9266,
        flux_jy_400=80.0,
        source_type="pulsar",
        description="Nearby bright pulsar with high pulse-to-pulse variability",
    ),
    "PSR_B1929+10": CelestialSource(
        name="PSR B1929+10",
        ra_deg=292.9554,
        dec_deg=10.9899,
        flux_jy_400=65.0,
        source_type="pulsar",
        description="Isolated radio pulsar with strong linear polarization",
    ),
}

# Alias mapping for ease of CLI use
CATALOG_ALIASES: Dict[str, str] = {
    "crab": "Crab",
    "vela": "Vela",
    "sgra": "SgrA",
    "sgr_a": "SgrA",
    "cena": "CenA",
    "cen_a": "CenA",
    "pictora": "PictorA",
    "pictor_a": "PictorA",
    "puppisa": "PuppisA",
    "hydraa": "HydraA",
    "casa": "CassiopeiaA",
    "cas_a": "CassiopeiaA",
    "virgoa": "VirgoA",
    "virgo_a": "VirgoA",
    "m87": "VirgoA",
    "j0437": "PSR_J0437-4715",
    "b1642": "PSR_B1642-03",
    "b0950": "PSR_B0950+08",
    "b1929": "PSR_B1929+10",
}


def parse_observation_time(
    time_spec: Optional[Union[str, datetime.datetime, float]],
) -> datetime.datetime:
    """Parses observation time into an explicit UTC datetime.

    Supports:
      - None / 'now': Current UTC datetime
      - ISO 8601 string: e.g. '2026-10-15T04:20:00Z', '2026-03-20T15:00:00'
      - Time string 'HH:MM' or 'HH:MM:SS': Anchored to equinox (2026-03-20)
      - Float or integer UTC hour (e.g. 15.0): Anchored to equinox (2026-03-20)
    """
    if time_spec is None or time_spec == "now":
        return datetime.datetime.now(datetime.timezone.utc)

    if isinstance(time_spec, datetime.datetime):
        if time_spec.tzinfo is None:
            return time_spec.replace(tzinfo=datetime.timezone.utc)
        return time_spec.astimezone(datetime.timezone.utc)

    if isinstance(time_spec, (int, float)):
        hour = int(time_spec) % 24
        minute = int((time_spec - hour) * 60)
        second = int(((time_spec - hour) * 60 - minute) * 60)
        return datetime.datetime(
            2026, 3, 20, hour, minute, second, tzinfo=datetime.timezone.utc
        )

    s = str(time_spec).strip()
    if s.lower() == "now":
        return datetime.datetime.now(datetime.timezone.utc)

    if s.lower().startswith("transit") or s.lower().startswith("slot:"):
        from .sky import resolve_window_start

        return resolve_window_start(s)

    # Try ISO 8601
    try:
        clean_s = s.replace("Z", "+00:00")
        dt = datetime.datetime.fromisoformat(clean_s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        return dt.astimezone(datetime.timezone.utc)
    except ValueError:
        pass

    # Try HH:MM:SS or HH:MM
    parts = s.split(":")
    if len(parts) in (2, 3):
        try:
            h = int(parts[0])
            m = int(parts[1])
            sec = int(parts[2]) if len(parts) == 3 else 0
            return datetime.datetime(
                2026, 3, 20, h, m, sec, tzinfo=datetime.timezone.utc
            )
        except ValueError:
            pass

    # Fallback: float hour string
    try:
        h_float = float(s)
        return parse_observation_time(h_float)
    except ValueError:
        raise ValueError(f"Unrecognized observation time format: '{time_spec}'")


def datetime_to_julian_date(dt: datetime.datetime) -> float:
    """Computes exact Julian Date from UTC datetime."""
    return 2440587.5 + dt.timestamp() / 86400.0


def datetime_to_lst_hours(
    dt: datetime.datetime,
    longitude_deg: float = CHARTS_LONGITUDE_DEG,
) -> float:
    """Computes Local Sidereal Time (LST) in hours.

    Mathematically identical to Kotekan C++ astrometry formula in:
      - lib/cuda/cudaBeamTrackerCommand.cpp
      - lib/cuda/cudaDirectBeamTrackerCommand.cpp
    """
    unix_s = dt.timestamp()
    d = unix_s / 86400.0 - 10957.5
    gmst_hours = (18.697374558 + 24.06570982441908 * d) % 24.0
    if gmst_hours < 0.0:
        gmst_hours += 24.0

    lst_hours = (gmst_hours + longitude_deg / 15.0) % 24.0
    if lst_hours < 0.0:
        lst_hours += 24.0
    return lst_hours


def equatorial_to_horizontal(
    ra_deg: float,
    dec_deg: float,
    lst_hours: float,
    lat_deg: float = CHARTS_LATITUDE_DEG,
) -> Tuple[float, float]:
    """Converts Equatorial (RA, Dec) to Topocentric Horizontal (Azimuth, Altitude/Elevation).

    Returns:
      (azimuth_deg, elevation_deg) where Azimuth is 0=North, 90=East.
    """
    lst_deg = lst_hours * 15.0
    ha_deg = (lst_deg - ra_deg) % 360.0

    ha_rad = math.radians(ha_deg)
    dec_rad = math.radians(dec_deg)
    lat_rad = math.radians(lat_deg)

    # Elevation: sin(alt) = sin(dec)*sin(lat) + cos(dec)*cos(lat)*cos(ha)
    sin_alt = math.sin(dec_rad) * math.sin(lat_rad) + math.cos(dec_rad) * math.cos(
        lat_rad
    ) * math.cos(ha_rad)
    sin_alt = max(-1.0, min(1.0, sin_alt))
    alt_rad = math.asin(sin_alt)
    alt_deg = math.degrees(alt_rad)

    # Direction cosines in local topocentric horizon:
    # l = -cos(dec) * sin(ha)  (East)
    # m = sin(dec)*cos(lat) - cos(dec)*sin(lat)*cos(ha)  (North)
    l = -math.cos(dec_rad) * math.sin(ha_rad)
    m = math.sin(dec_rad) * math.cos(lat_rad) - math.cos(dec_rad) * math.sin(
        lat_rad
    ) * math.cos(ha_rad)

    az_rad = math.atan2(l, m)
    az_deg = (math.degrees(az_rad) + 360.0) % 360.0

    return az_deg, alt_deg


def equatorial_to_direction_cosines(
    ra_deg: float,
    dec_deg: float,
    lst_hours: float,
    lat_deg: float = CHARTS_LATITUDE_DEG,
) -> Tuple[float, float, float]:
    """Computes Topocentric Direction Cosines (l, m, n) for CHARTS array plane.

    Convention:
      - l: East direction cosine (+x)
      - m: North direction cosine (+y)
      - n: Upward / Zenith cosine (+z = sqrt(1 - l^2 - m^2))
    """
    lst_deg = lst_hours * 15.0
    ha_deg = (lst_deg - ra_deg) % 360.0

    ha_rad = math.radians(ha_deg)
    dec_rad = math.radians(dec_deg)
    lat_rad = math.radians(lat_deg)

    l = -math.cos(dec_rad) * math.sin(ha_rad)
    m = math.sin(dec_rad) * math.cos(lat_rad) - math.cos(dec_rad) * math.sin(
        lat_rad
    ) * math.cos(ha_rad)
    n_sq = 1.0 - (l * l + m * m)
    n = math.sqrt(max(0.0, n_sq)) if n_sq > 0.0 else 0.0

    return float(l), float(m), float(n)


def compute_direction_rates(
    ra_deg: float,
    dec_deg: float,
    lst_hours: float,
    lat_deg: float = CHARTS_LATITUDE_DEG,
    sample_rate_hz: float = SAMPLE_RATE_HZ,
) -> Tuple[float, float]:
    """Computes direction cosine drift rates (dl/dt, dm/dt) in units per sample.

    Used by Kotekan's phase-tracking beam steering kernel.
    """
    ha_rad = math.radians((lst_hours * 15.0 - ra_deg) % 360.0)
    dec_rad = math.radians(dec_deg)
    lat_rad = math.radians(lat_deg)

    # Earth rotation angular velocity: omega = 2*pi / 86164.0905 s (sidereal day)
    omega = 2.0 * math.pi / 86164.090535

    # d(l)/dt = -cos(dec) * cos(ha) * omega
    dl_dt = -math.cos(dec_rad) * math.cos(ha_rad) * omega

    # d(m)/dt = cos(dec) * sin(lat) * sin(ha) * omega
    dm_dt = math.cos(dec_rad) * math.sin(lat_rad) * math.sin(ha_rad) * omega

    # Convert to per-sample rates:
    dl_per_sample = dl_dt / sample_rate_hz
    dm_per_sample = dm_dt / sample_rate_hz

    return float(dl_per_sample), float(dm_per_sample)


def compute_sun_equatorial(dt: datetime.datetime) -> Tuple[float, float]:
    """Computes approximate Solar Right Ascension and Declination for given UTC date.
    Accurate to within ~0.05 degrees.
    """
    doy = dt.timetuple().tm_yday
    gamma = 2.0 * math.pi * (doy - 1) / 365.0

    decl_deg = (
        0.006918
        - 0.399912 * math.cos(gamma)
        + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2.0 * gamma)
        + 0.000907 * math.sin(2.0 * gamma)
        - 0.002697 * math.cos(3.0 * gamma)
        + 0.001480 * math.sin(3.0 * gamma)
    ) * (180.0 / math.pi)

    eot_min = 229.18 * (
        0.000075
        + 0.001868 * math.cos(gamma)
        - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2.0 * gamma)
        - 0.040849 * math.sin(2.0 * gamma)
    )

    # Solar RA from LST and solar time:
    lst_h = datetime_to_lst_hours(dt)
    hour_frac = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
    solar_time_h = (hour_frac + (CHARTS_LONGITUDE_DEG / 15.0) + (eot_min / 60.0)) % 24.0
    ra_hours = (lst_h - (solar_time_h - 12.0)) % 24.0
    ra_deg = (ra_hours * 15.0) % 360.0

    return float(ra_deg), float(decl_deg)


def find_visible_sources(
    obs_time: Optional[Union[str, datetime.datetime, float]] = None,
    min_elevation_deg: float = 15.0,
    lat_deg: float = CHARTS_LATITUDE_DEG,
    lon_deg: float = CHARTS_LONGITUDE_DEG,
) -> List[Dict[str, Any]]:
    """Identifies and ranks all catalog sources visible above the local horizon.

    Sorted in descending order of elevation (highest in the sky first).
    """
    dt = parse_observation_time(obs_time)
    lst_h = datetime_to_lst_hours(dt, longitude_deg=lon_deg)

    results = []

    # Check Sun first
    sun_ra, sun_dec = compute_sun_equatorial(dt)
    sun_az, sun_alt = equatorial_to_horizontal(sun_ra, sun_dec, lst_h, lat_deg=lat_deg)
    if sun_alt >= min_elevation_deg:
        l, m, n = equatorial_to_direction_cosines(
            sun_ra, sun_dec, lst_h, lat_deg=lat_deg
        )
        results.append(
            {
                "key": "Sun",
                "name": "Sun",
                "ra_deg": sun_ra,
                "dec_deg": sun_dec,
                "elevation_deg": sun_alt,
                "azimuth_deg": sun_az,
                "l0": l,
                "m0": m,
                "flux_jy": 1.5e5,
                "source_type": "solar",
                "description": "The Sun (intense radio flux, active/quiet profile)",
            }
        )

    # Check celestial catalog
    for key, src in CELESTIAL_CATALOG.items():
        az, alt = equatorial_to_horizontal(
            src.ra_deg, src.dec_deg, lst_h, lat_deg=lat_deg
        )
        if alt >= min_elevation_deg:
            l, m, n = equatorial_to_direction_cosines(
                src.ra_deg, src.dec_deg, lst_h, lat_deg=lat_deg
            )
            results.append(
                {
                    "key": key,
                    "name": src.name,
                    "ra_deg": src.ra_deg,
                    "dec_deg": src.dec_deg,
                    "elevation_deg": alt,
                    "azimuth_deg": az,
                    "l0": l,
                    "m0": m,
                    "flux_jy": src.flux_jy_400,
                    "source_type": src.source_type,
                    "description": src.description,
                }
            )

    # Sort descending by elevation
    results.sort(key=lambda s: s["elevation_deg"], reverse=True)
    return results


def resolve_beam_targets(
    targets_spec: Optional[str] = None,
    obs_time: Optional[Union[str, datetime.datetime, float]] = None,
    max_beams: int = 8,
    lat_deg: float = CHARTS_LATITUDE_DEG,
    lon_deg: float = CHARTS_LONGITUDE_DEG,
) -> List[Dict[str, Any]]:
    """Resolves target specification string into complete, calibrated beam targets.

    Supports:
      - 'auto' or 'auto:N': Automatically picks the top N visible celestial sources
      - Named sources: 'Crab', 'Vela', 'SgrA', 'Zenith', 'Sun'
      - Semicolon-separated list: 'Crab;Vela;SgrA'
      - Explicit coordinates: '83.633,22.014' or 'MyTarget:83.633,22.014'
      - Direction cosines: 'l=0.1,m=-0.2'
      - Mixed combinations: 'Crab;Zenith;auto:2'

    Returns:
      List of beam dicts ready for Kotekan YAML generation and tracking validation.
    """
    dt = parse_observation_time(obs_time)
    lst_h = datetime_to_lst_hours(dt, longitude_deg=lon_deg)

    targets: List[Dict[str, Any]] = []

    # If no spec or empty, default to auto-ranking
    spec = (targets_spec or "auto").strip()

    items = [s.strip() for s in spec.split(";") if s.strip()]
    auto_count_requested = 0

    for item in items:
        if len(targets) >= max_beams:
            break

        # Check for 'auto' or 'auto:N'
        if item.lower().startswith("auto"):
            count = max_beams - len(targets)
            if ":" in item:
                try:
                    count = int(item.split(":", 1)[1].strip())
                except ValueError:
                    pass
            auto_count_requested += count
            continue

        # Check for 'Zenith'
        if item.lower() == "zenith":
            ra_zenith = (lst_h * 15.0) % 360.0
            dec_zenith = lat_deg
            targets.append(
                {
                    "beam": len(targets),
                    "name": "Zenith (Local Meridian)",
                    "ra_deg": ra_zenith,
                    "dec_deg": dec_zenith,
                    "lst_hours": lst_h,
                    "elevation_deg": 90.0,
                    "azimuth_deg": 0.0,
                    "l0": 0.0,
                    "m0": 0.0,
                }
            )
            continue

        # Check for 'Sun'
        if item.lower() == "sun":
            sun_ra, sun_dec = compute_sun_equatorial(dt)
            az, alt = equatorial_to_horizontal(sun_ra, sun_dec, lst_h, lat_deg)
            l, m, _ = equatorial_to_direction_cosines(sun_ra, sun_dec, lst_h, lat_deg)
            targets.append(
                {
                    "beam": len(targets),
                    "name": "Sun",
                    "ra_deg": sun_ra,
                    "dec_deg": sun_dec,
                    "lst_hours": lst_h,
                    "elevation_deg": alt,
                    "azimuth_deg": az,
                    "l0": l,
                    "m0": m,
                }
            )
            continue

        # Check catalog by name or alias
        lookup_key = item.strip()
        canonical_key = CATALOG_ALIASES.get(lookup_key.lower()) or lookup_key
        if canonical_key in CELESTIAL_CATALOG:
            src = CELESTIAL_CATALOG[canonical_key]
            az, alt = equatorial_to_horizontal(src.ra_deg, src.dec_deg, lst_h, lat_deg)
            l, m, _ = equatorial_to_direction_cosines(
                src.ra_deg, src.dec_deg, lst_h, lat_deg
            )
            targets.append(
                {
                    "beam": len(targets),
                    "name": src.name,
                    "ra_deg": src.ra_deg,
                    "dec_deg": src.dec_deg,
                    "lst_hours": lst_h,
                    "elevation_deg": alt,
                    "azimuth_deg": az,
                    "l0": l,
                    "m0": m,
                }
            )
            continue

        # Check verified targets catalog (SIMBAD parity)
        try:
            from .sky import find_verified_target

            v_src = find_verified_target(lookup_key)
            az, alt = equatorial_to_horizontal(
                v_src.ra_deg, v_src.dec_deg, lst_h, lat_deg
            )
            l, m, _ = equatorial_to_direction_cosines(
                v_src.ra_deg, v_src.dec_deg, lst_h, lat_deg
            )
            targets.append(
                {
                    "beam": len(targets),
                    "name": v_src.label,
                    "ra_deg": v_src.ra_deg,
                    "dec_deg": v_src.dec_deg,
                    "lst_hours": lst_h,
                    "elevation_deg": alt,
                    "azimuth_deg": az,
                    "l0": l,
                    "m0": m,
                }
            )
            continue
        except Exception:
            pass

        # Check if "Name:RA,Dec" or "RA,Dec"
        name = f"Custom Beam {len(targets)}"
        coord_part = item
        if ":" in item:
            name_cand, coord_part = item.split(":", 1)
            name = name_cand.strip()

        coords = [c.strip() for c in coord_part.split(",") if c.strip()]
        if len(coords) >= 2:
            try:
                ra = float(coords[0])
                dec = float(coords[1])
                az, alt = equatorial_to_horizontal(ra, dec, lst_h, lat_deg)
                l, m, _ = equatorial_to_direction_cosines(ra, dec, lst_h, lat_deg)
                targets.append(
                    {
                        "beam": len(targets),
                        "name": name,
                        "ra_deg": ra,
                        "dec_deg": dec,
                        "lst_hours": lst_h,
                        "elevation_deg": alt,
                        "azimuth_deg": az,
                        "l0": l,
                        "m0": m,
                    }
                )
            except ValueError:
                pass

    # If auto count requested, fill remaining beams with visible sources
    if auto_count_requested > 0 or len(targets) == 0:
        visible = find_visible_sources(
            dt, min_elevation_deg=5.0, lat_deg=lat_deg, lon_deg=lon_deg
        )
        already_names = {t["name"] for t in targets}
        for v in visible:
            if len(targets) >= max_beams:
                break
            if v["name"] not in already_names:
                targets.append(
                    {
                        "beam": len(targets),
                        "name": v["name"],
                        "ra_deg": v["ra_deg"],
                        "dec_deg": v["dec_deg"],
                        "lst_hours": lst_h,
                        "elevation_deg": v["elevation_deg"],
                        "azimuth_deg": v["azimuth_deg"],
                        "l0": v["l0"],
                        "m0": v["m0"],
                    }
                )
                already_names.add(v["name"])

    # If still fewer than max_beams, synthesize hexagonal grid offset beams around Zenith or primary
    base_ra = targets[0]["ra_deg"] if targets else (lst_h * 15.0) % 360.0
    base_dec = targets[0]["dec_deg"] if targets else lat_deg
    while len(targets) < max_beams:
        idx = len(targets)
        offset_ra = base_ra + (idx * 0.25)
        offset_dec = base_dec + (idx * 0.25)
        az, alt = equatorial_to_horizontal(offset_ra, offset_dec, lst_h, lat_deg)
        l, m, _ = equatorial_to_direction_cosines(offset_ra, offset_dec, lst_h, lat_deg)
        targets.append(
            {
                "beam": idx,
                "name": f"Hex Offset {idx}",
                "ra_deg": offset_ra,
                "dec_deg": offset_dec,
                "lst_hours": lst_h,
                "elevation_deg": alt,
                "azimuth_deg": az,
                "l0": l,
                "m0": m,
            }
        )

    return targets
