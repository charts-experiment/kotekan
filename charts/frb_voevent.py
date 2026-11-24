#!/usr/bin/env python3
"""IVOA VOEvent v2.0 publisher & NE2025 DM/scattering priors for CHARTS FRB candidates (TRN-5.4)."""

from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from xml.etree import ElementTree as ET

try:
    from charts_constants import (
        CHARTS_ALTITUDE_M,
        CHARTS_LATITUDE_DEG,
        CHARTS_LONGITUDE_DEG,
    )
except Exception:
    CHARTS_LATITUDE_DEG = -33.4211146
    CHARTS_LONGITUDE_DEG = -70.8634710
    CHARTS_ALTITUDE_M = 458.0


def equ_to_gal_deg(ra_deg: float, dec_deg: float) -> tuple[float, float]:
    """Convert ICRS (RA, Dec) in degrees to Galactic (l, b) in degrees (IAU J2000)."""
    ra = math.radians(float(ra_deg))
    dec = math.radians(float(dec_deg))
    ra_gp = math.radians(192.85948)
    dec_gp = math.radians(27.12825)
    l_ncp = math.radians(122.93192)

    sin_b = math.sin(dec) * math.sin(dec_gp) + math.cos(dec) * math.cos(dec_gp) * math.cos(
        ra - ra_gp
    )
    sin_b = max(-1.0, min(1.0, sin_b))
    b = math.asin(sin_b)

    y = math.cos(dec) * math.sin(ra - ra_gp)
    x = math.sin(dec) * math.cos(dec_gp) - math.cos(dec) * math.sin(dec_gp) * math.cos(ra - ra_gp)
    l_rad = (l_ncp - math.atan2(y, x)) % (2.0 * math.pi)
    return math.degrees(l_rad), math.degrees(b)


def estimate_ne2025_galactic_prior(
    ra_deg: float, dec_deg: float, freq_mhz: float = 400.0
) -> Dict[str, Any]:
    """Evaluate analytical NE2025-calibrated Galactic DM & scattering timescale priors (Ocker & Cordes 2026).

    Models the Ocker & Cordes thick disk (~25.5 pc cm^-3 * csc|b| capped along the plane),
    thin spiral disk + Galactic Center component toward inner longitudes |l| < 50 deg,
    plus a CGM/halo baseline of 30 pc cm^-3, and frequency-scaled pulse broadening tau_s.
    """
    l_deg, b_deg = equ_to_gal_deg(ra_deg, dec_deg)
    sin_abs_b = abs(math.sin(math.radians(b_deg)))
    # Regularized cosecant geometry (thick disk scale height ~1.6 kpc, midplane saturation ~900 pc cm^-3)
    csc_reg = 1.0 / math.sqrt(sin_abs_b * sin_abs_b + 0.0008)
    dm_thick = 25.5 * csc_reg

    # Inner-Galaxy thin disk / spiral excess toward |l| < 50 deg and |b| < 12 deg
    l_wrapped = ((l_deg + 180.0) % 360.0) - 180.0
    gc_lon_factor = math.exp(-0.5 * (l_wrapped / 32.0) ** 2)
    gc_lat_factor = math.exp(-0.5 * (b_deg / 4.5) ** 2)
    dm_thin_gc = 420.0 * gc_lon_factor * gc_lat_factor

    dm_ism = dm_thick + dm_thin_gc
    dm_halo = 30.0
    dm_gal_total = dm_ism + dm_halo

    # Empirical Bhat / Ocker & Cordes scattering law scaled to freq_mhz with nu^-4.0 dependence
    nu_ghz = max(0.05, float(freq_mhz) / 1000.0)
    log_dm = math.log10(max(1.0, dm_ism))
    tau_1ghz_ms = 10.0 ** (-6.46 + 0.154 * log_dm + 1.07 * (log_dm**2) - 3.86 * 0.0)
    # Cap unphysical extreme values while preserving scaling monotonically across |b|
    tau_1ghz_ms = min(tau_1ghz_ms, 50.0)
    tau_scatter_ms = tau_1ghz_ms * (nu_ghz ** -4.0)

    return {
        "gal_l_deg": round(l_deg, 4),
        "gal_b_deg": round(b_deg, 4),
        "freq_mhz": float(freq_mhz),
        "dm_gal_ism": round(dm_ism, 3),
        "dm_gal_halo": round(dm_halo, 3),
        "dm_gal_total": round(dm_gal_total, 3),
        "tau_scatter_ms": round(tau_scatter_ms, 6),
    }


def compute_dm_excess(dm_obs: float, prior: Dict[str, Any]) -> Dict[str, Any]:
    """Compute extragalactic DM excess and heuristic probability above NE2025 Galactic total."""
    dm_gal = float(prior["dm_gal_total"])
    dm_excess = float(dm_obs) - dm_gal
    # Smooth logistic extragalactic score centered at +20 pc cm^-3 margin above NE2025 uncertainty (~25%)
    margin = max(15.0, 0.25 * dm_gal)
    z = (dm_excess - 10.0) / margin
    p_extragal = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, z))))
    return {
        "dm_obs": round(float(dm_obs), 3),
        "dm_gal_ne2025": round(dm_gal, 3),
        "dm_excess": round(dm_excess, 3),
        "is_extragalactic": bool(dm_excess > 0.0),
        "p_extragalactic": round(p_extragal, 4),
    }


def build_frb_voevent_json(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Annotate a transient candidate dict with NE2025 DM/scattering priors for JSON/VOEvent export."""
    ra_deg = float(candidate.get("ra_deg", 0.0))
    dec_deg = float(candidate.get("dec_deg", CHARTS_LATITUDE_DEG))
    freq_mhz = float(candidate.get("freq_mhz", 400.0))
    dm_obs = float(candidate.get("dm", candidate.get("dm_pc_cm3", 0.0)))

    prior = estimate_ne2025_galactic_prior(ra_deg, dec_deg, freq_mhz=freq_mhz)
    excess = compute_dm_excess(dm_obs, prior)

    utc_iso = str(
        candidate.get("utc_iso")
        or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    )
    event_id = str(
        candidate.get("event_id")
        or f"charts_{utc_iso.replace(':', '').replace('-', '').replace('.', '')}"
    )

    return {
        "ivorn": f"ivo://charts.cl/frb#{event_id}",
        "event_id": event_id,
        "utc_iso": utc_iso,
        "ra_deg": ra_deg,
        "dec_deg": dec_deg,
        "error_radius_deg": float(candidate.get("error_radius_deg", 3.5)),
        "snr": float(candidate.get("snr", 0.0)),
        "sidelobe_confidence": float(candidate.get("sidelobe_confidence", candidate.get("snr", 0.0))),
        "width_ms": float(candidate.get("width_ms", candidate.get("width_samples", 1) * 0.0512)),
        "beam_id": int(candidate.get("beam_id", 0)),
        "observatory": {
            "name": "CHARTS Laguna Caren",
            "lat_deg": CHARTS_LATITUDE_DEG,
            "lon_deg": CHARTS_LONGITUDE_DEG,
            "alt_m": CHARTS_ALTITUDE_M,
        },
        **prior,
        **excess,
    }


def build_frb_voevent_xml(
    candidate: Dict[str, Any], ivorn_prefix: str = "ivo://charts.cl/frb"
) -> str:
    """Serialize candidate + NE2025 priors into an IVOA VOEvent v2.0 XML packet."""
    payload = build_frb_voevent_json(candidate)
    ivorn = f"{ivorn_prefix}#{payload['event_id']}"

    root = ET.Element(
        "voe:VOEvent",
        {
            "xmlns:voe": "http://www.ivoa.net/xml/VOEvent/v2.0",
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "version": "2.0",
            "role": "observation",
            "ivorn": ivorn,
        },
    )

    who = ET.SubElement(root, "Who")
    ET.SubElement(who, "AuthorIVORN").text = "ivo://charts.cl/contact"
    ET.SubElement(who, "Date").text = payload["utc_iso"]

    what = ET.SubElement(root, "What")
    grp_frb = ET.SubElement(what, "Group", {"name": "frb_parameters"})
    params = [
        ("dm", payload["dm_obs"], "pc/cm^3", "phot.mag;phys.dispMeasure"),
        ("dm_gal_ne2025", payload["dm_gal_ne2025"], "pc/cm^3", "phys.dispMeasure"),
        ("dm_excess", payload["dm_excess"], "pc/cm^3", "phys.dispMeasure"),
        ("tau_scatter_ms", payload["tau_scatter_ms"], "ms", "time.duration"),
        ("snr", payload["snr"], "sigma", "stat.snr"),
        ("sidelobe_confidence", payload["sidelobe_confidence"], "sigma", "stat.snr"),
        ("width_ms", payload["width_ms"], "ms", "time.duration"),
        ("beam_id", payload["beam_id"], "index", "meta.id"),
        ("p_extragalactic", payload["p_extragalactic"], "prob", "stat.probability"),
    ]
    for name, val, unit, ucd in params:
        ET.SubElement(
            grp_frb, "Param", {"name": str(name), "value": str(val), "unit": unit, "ucd": ucd}
        )

    where = ET.SubElement(root, "WhereWhen")
    obs_data = ET.SubElement(where, "ObsDataLocation")
    obs_loc = ET.SubElement(obs_data, "ObservatoryLocation", {"id": "CHARTS_CAREN"})
    astro_loc = ET.SubElement(obs_loc, "AstroCoordSystem", {"id": "UTC-GEOD-TOPO"})
    pos3 = ET.SubElement(ET.SubElement(obs_loc, "AstroCoords"), "Position3D", {"unit": "deg deg m"})
    vals3 = ET.SubElement(pos3, "Value3")
    ET.SubElement(vals3, "C1").text = f"{CHARTS_LONGITUDE_DEG:.7f}"
    ET.SubElement(vals3, "C2").text = f"{CHARTS_LATITUDE_DEG:.7f}"
    ET.SubElement(vals3, "C3").text = f"{CHARTS_ALTITUDE_M:.1f}"

    obs_area = ET.SubElement(obs_data, "ObservationLocation")
    ET.SubElement(obs_area, "AstroCoordSystem", {"id": "UTC-ICRS-TOPO"})
    ac = ET.SubElement(obs_area, "AstroCoords", {"coord_system_id": "UTC-ICRS-TOPO"})
    t_elem = ET.SubElement(ET.SubElement(ac, "Time", {"unit": "s"}), "TimeInstant")
    ET.SubElement(t_elem, "ISOTime").text = payload["utc_iso"]
    pos2 = ET.SubElement(ac, "Position2D", {"unit": "deg"})
    v2 = ET.SubElement(pos2, "Value2")
    ET.SubElement(v2, "C1").text = f"{payload['ra_deg']:.6f}"
    ET.SubElement(v2, "C2").text = f"{payload['dec_deg']:.6f}"
    ET.SubElement(pos2, "Error2Radius").text = f"{payload['error_radius_deg']:.4f}"
    _ = astro_loc

    how = ET.SubElement(root, "How")
    ET.SubElement(how, "Description").text = "CHARTS 64-element phased array direct beamformer & FRB pipeline"

    xml_str = ET.tostring(root, encoding="unicode")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + xml_str


def emit_frb_voevent_alert(
    candidate: Dict[str, Any],
    xml_path: Optional[str | Path] = None,
    json_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Build and optionally write <1s VOEvent XML/JSON packets and return summary + elapsed ms."""
    t0 = time.perf_counter()
    payload = build_frb_voevent_json(candidate)
    xml_text = build_frb_voevent_xml(candidate)
    if xml_path is not None:
        p = Path(xml_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(xml_text, encoding="utf-8")
    if json_path is not None:
        p = Path(json_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "payload": payload,
        "xml": xml_text,
        "latency_ms": round(elapsed_ms, 3),
    }
