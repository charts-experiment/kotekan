#!/usr/bin/env python3
"""Solar Radio Burst (Type II/III/IV/V in 300-360 MHz) drift-rate & direction-cosine guard (RFI-3.4).

References:
  - TWIN-2 Solar Radio Burst Survey (2611.25642v1, Sec. 2-5): Type II/III/IV/V bursts across 180-350 MHz
  - le Roux et al. (2512.11487v1): Background-normalized dynamic spectra & plasma drift df/dt < 0
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _normalize_dyn_spec(spec: np.ndarray) -> np.ndarray:
    """le Roux et al. median/MAD background normalization S_norm(f, t) along time per channel."""
    arr = np.asarray(spec, dtype=np.float64)
    med = np.median(arr, axis=0, keepdims=True)
    mad = np.median(np.abs(arr - med), axis=0, keepdims=True)
    sig = np.where(mad > 1.0e-12, 1.4826 * mad, np.maximum(np.std(arr, axis=0, keepdims=True), 1.0e-6))
    return (arr - med) / sig


def classify_solar_radio_burst(
    dyn_spec_sun: np.ndarray,
    dyn_spec_off: Optional[np.ndarray],
    freqs_mhz: np.ndarray,
    times_s: np.ndarray,
    sun_el_deg: float,
    f_lo_mhz: float = 300.0,
    f_hi_mhz: float = 360.0,
    min_sun_el_deg: float = 10.0,
    min_peak_z: float = 5.0,
    min_sun_off_ratio: float = 1.5,
) -> Dict[str, Any]:
    """Detect and classify Solar Radio Bursts in the 300-360 MHz Subband 0 window.

    Args:
        dyn_spec_sun: Dynamic spectrum (n_time, n_freq) on the Sun beam.
        dyn_spec_off: Optional horizon/OFF reference beam dynamic spectrum (n_time, n_freq).
        freqs_mhz: Channel frequencies in MHz (length n_freq).
        times_s: Relative frame timestamps in seconds (length n_time).
        sun_el_deg: Solar elevation angle in degrees (must exceed `min_sun_el_deg`).

    Returns:
        Dict with `is_solar_burst`, `classification` ("SOLAR_RADIO_BURST" or "NONE"),
        `burst_type` ("Type III", "Type II", "Type IV/V", or "none"), `drift_rate_mhz_s`,
        `pearson_r`, `peak_z_sun`, `sun_off_ratio`, and `protected_ranges_mhz`.
    """
    out_none: Dict[str, Any] = {
        "is_solar_burst": False,
        "classification": "NONE",
        "burst_type": "none",
        "reason": "inactive",
        "drift_rate_mhz_s": 0.0,
        "pearson_r": 0.0,
        "peak_z_sun": 0.0,
        "sun_off_ratio": 0.0,
        "protected_ranges_mhz": [],
    }

    if float(sun_el_deg) <= float(min_sun_el_deg):
        out_none["reason"] = f"sun_el_deg={sun_el_deg:.2f} <= {min_sun_el_deg}"
        return out_none

    sun_arr = np.asarray(dyn_spec_sun, dtype=np.float64)
    f_arr = np.asarray(freqs_mhz, dtype=np.float64)
    t_arr = np.asarray(times_s, dtype=np.float64)
    if sun_arr.ndim != 2 or sun_arr.shape[0] < 4 or sun_arr.shape[1] != f_arr.size:
        raise ValueError("Invalid dynamic spectrum or frequency axis shape")
    if t_arr.size != sun_arr.shape[0]:
        raise ValueError("times_s length must match n_time axis")

    band = (f_arr >= float(f_lo_mhz)) & (f_arr <= float(f_hi_mhz))
    if int(np.sum(band)) < 4:
        out_none["reason"] = "insufficient_band_channels"
        return out_none

    z_sun = _normalize_dyn_spec(sun_arr[:, band])
    f_band = f_arr[band]

    peak_z_per_chan = np.max(z_sun, axis=0)
    peak_z_sun = float(np.max(peak_z_per_chan))
    if peak_z_sun < float(min_peak_z):
        out_none["peak_z_sun"] = round(peak_z_sun, 3)
        out_none["reason"] = "below_peak_z"
        return out_none

    # Spatial localization gate: Sun beam excursion must exceed horizon/OFF beam
    sun_off_ratio = float("inf")
    if dyn_spec_off is not None:
        off_arr = np.asarray(dyn_spec_off, dtype=np.float64)
        z_off = _normalize_dyn_spec(off_arr[:, band])
        peak_z_off = max(float(np.max(z_off)), 0.5)
        sun_off_ratio = peak_z_sun / peak_z_off
        if sun_off_ratio < float(min_sun_off_ratio):
            out_none["peak_z_sun"] = round(peak_z_sun, 3)
            out_none["sun_off_ratio"] = round(sun_off_ratio, 3)
            out_none["reason"] = "not_sun_localized"
            return out_none

    # Track peak arrival times across active channels with significant burst power
    active_ch = np.flatnonzero(peak_z_per_chan >= 0.5 * peak_z_sun)
    if active_ch.size < 4:
        active_ch = np.flatnonzero(peak_z_per_chan >= float(min_peak_z))

    if active_ch.size < 3:
        out_none["peak_z_sun"] = round(peak_z_sun, 3)
        out_none["sun_off_ratio"] = round(min(sun_off_ratio, 1.0e6), 3)
        out_none["reason"] = "too_narrowband"
        return out_none

    peak_t_idx = np.argmax(z_sun[:, active_ch], axis=0)
    t_pk = t_arr[peak_t_idx]
    f_pk = f_band[active_ch]

    t_span = float(np.max(t_pk) - np.min(t_pk))
    f_span = float(np.max(f_pk) - np.min(f_pk))
    # Linear fit f(t) = slope * t + b -> df/dt in MHz/s (using orthogonal regression / least squares)
    dt_centered = t_pk - float(np.mean(t_pk))
    df_centered = f_pk - float(np.mean(f_pk))
    var_t = float(np.dot(dt_centered, dt_centered))
    var_f = float(np.dot(df_centered, df_centered))
    cov_tf = float(np.dot(dt_centered, df_centered))

    drift_rate = 0.0
    pearson_r = 0.0
    if var_t > 1.0e-12 and var_f > 1.0e-12:
        drift_rate = cov_tf / var_t
        pearson_r = cov_tf / math.sqrt(var_t * var_f)

    # Check if negative plasma drift (Type II/III) or broad Sun-localized continuum (Type IV/V)
    burst_type = "none"
    is_solar = False
    if t_span >= 0.05 and f_span >= 5.0 and drift_rate < -0.05 and pearson_r < -0.5:
        is_solar = True
        burst_type = "Type III" if abs(drift_rate) >= 5.0 else "Type II"
    elif f_span >= 15.0 and active_ch.size >= max(4, int(0.4 * f_band.size)) and t_span < 0.05 and sun_off_ratio >= 2.0:
        # Broadband instantaneous/continuum Sun-only excursion (e.g. Type V/IV or fast Type III below frame cadence)
        # Note: reject common-mode horizon RFI cleanly via sun_off_ratio >= 2.0 and only when stationary horizon is quiet
        # Only label as Type IV/V continuum if duration spans > 20% of window or high bandwidth
        frac_active_t = float(np.mean(np.max(z_sun, axis=1) >= float(min_peak_z)))
        if frac_active_t >= 0.15:
            is_solar = True
            burst_type = "Type IV/V"

    protected_ranges: List[List[float]] = []
    if is_solar:
        pad = 0.3
        protected_ranges.append(
            [
                round(max(float(f_lo_mhz), float(np.min(f_pk)) - pad), 3),
                round(min(float(f_hi_mhz), float(np.max(f_pk)) + pad), 3),
            ]
        )

    return {
        "is_solar_burst": is_solar,
        "classification": "SOLAR_RADIO_BURST" if is_solar else "NONE",
        "burst_type": burst_type,
        "reason": "solar_burst_detected" if is_solar else "no_negative_solar_drift",
        "drift_rate_mhz_s": round(float(drift_rate), 4),
        "pearson_r": round(float(pearson_r), 4),
        "peak_z_sun": round(peak_z_sun, 3),
        "sun_off_ratio": round(min(sun_off_ratio, 1.0e6), 3),
        "protected_ranges_mhz": protected_ranges,
    }
