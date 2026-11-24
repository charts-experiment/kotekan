#!/usr/bin/env python3
"""SGP4/TLE non-sidereal LEO tracker, quadratic Doppler filter & beam-envelope holography (TRN-5.3).

References:
  - Vierinen et al. (2611.12158v1, Nov 2026, Eq. 1-12): Quadratic range/Doppler matched filtering,
    8-tap Lanczos fractional-delay resampling, and empirical beam-passage envelope holography.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, Tuple

import numpy as np
from scipy.optimize import least_squares
from sgp4.api import Satrec, jday

try:
    from charts_constants import (
        CHARTS_ALTITUDE_M,
        CHARTS_LATITUDE_DEG,
        CHARTS_LONGITUDE_DEG,
        C_LIGHT,
    )

    _C_LIGHT = float(C_LIGHT.value) if hasattr(C_LIGHT, "value") else float(C_LIGHT)
except Exception:
    CHARTS_LATITUDE_DEG = -33.4211146
    CHARTS_LONGITUDE_DEG = -70.8634710
    CHARTS_ALTITUDE_M = 458.0
    _C_LIGHT = 299792458.0


def _gmst_rad(jd_ut1: float) -> float:
    """Greenwich Mean Sidereal Time (GMST) in radians (IAU 1982 formula)."""
    t_ut1 = (jd_ut1 - 2451545.0) / 36525.0
    gmst_sec = (
        67310.54841
        + (876600.0 * 3600.0 + 8640184.812866) * t_ut1
        + 0.093104 * (t_ut1**2)
        - 6.2e-6 * (t_ut1**3)
    )
    return ((gmst_sec % 86400.0) / 86400.0) * (2.0 * math.pi)


def _geodetic_to_ecef_km(lat_deg: float, lon_deg: float, alt_m: float) -> Tuple[np.ndarray, np.ndarray]:
    """WGS-84 observer ECEF position (km) and ECI->ENU rotation matrix (ECEF frame)."""
    a = 6378.137
    f = 1.0 / 298.257223563
    e2 = f * (2.0 - f)
    lat = math.radians(float(lat_deg))
    lon = math.radians(float(lon_deg))
    sin_lat, cos_lat = math.sin(lat), math.cos(lat)
    sin_lon, cos_lon = math.sin(lon), math.cos(lon)
    n_val = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
    alt_km = float(alt_m) * 1.0e-3
    r_ecef = np.array(
        [
            (n_val + alt_km) * cos_lat * cos_lon,
            (n_val + alt_km) * cos_lat * sin_lon,
            (n_val * (1.0 - e2) + alt_km) * sin_lat,
        ],
        dtype=np.float64,
    )
    rot_ecef_to_enu = np.array(
        [
            [-sin_lon, cos_lon, 0.0],
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
        ],
        dtype=np.float64,
    )
    return r_ecef, rot_ecef_to_enu


class Sgp4TopocentricPropagator:
    """Propagate Two-Line Element (TLE) orbits into CHARTS topocentric ENU (l, m, n) and Doppler."""

    OMEGA_EARTH_RAD_S = 7.2921151467e-5

    def __init__(
        self,
        tle_line1: str,
        tle_line2: str,
        name: str = "",
        lat_deg: float = CHARTS_LATITUDE_DEG,
        lon_deg: float = CHARTS_LONGITUDE_DEG,
        alt_m: float = CHARTS_ALTITUDE_M,
    ) -> None:
        self.tle_line1 = tle_line1.strip()
        self.tle_line2 = tle_line2.strip()
        self.name = name or f"NORAD_{self.tle_line2.split()[1]}"
        self.satrec = Satrec.twoline2rv(self.tle_line1, self.tle_line2)
        self.norad_id = int(self.satrec.satnum)
        self.obs_ecef_km, self.rot_ecef_to_enu = _geodetic_to_ecef_km(lat_deg, lon_deg, alt_m)

    def propagate_unix(self, unix_s: float, carrier_hz: float = 400.0e6) -> Dict[str, float]:
        """Compute topocentric direction cosines, range, velocity, range rate and Doppler at unix_s."""
        dt = datetime.fromtimestamp(float(unix_s), tz=timezone.utc)
        sec_frac = float(dt.second) + float(dt.microsecond) * 1.0e-6
        jd, fr = jday(dt.year, dt.month, dt.day, dt.hour, dt.minute, sec_frac)

        err, r_eci_km, v_eci_km_s = self.satrec.sgp4(jd, fr)
        if err != 0:
            raise RuntimeError(f"SGP4 propagation error code {err}")

        r_eci = np.asarray(r_eci_km, dtype=np.float64)
        v_eci = np.asarray(v_eci_km_s, dtype=np.float64)
        theta = _gmst_rad(jd + fr)
        c, s = math.cos(theta), math.sin(theta)
        rz = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        omega_vec = np.array([0.0, 0.0, self.OMEGA_EARTH_RAD_S], dtype=np.float64)

        r_ecef = rz @ r_eci
        v_ecef = (rz @ v_eci) - np.cross(omega_vec, r_ecef)
        dr_ecef = r_ecef - self.obs_ecef_km
        enu_pos_m = (self.rot_ecef_to_enu @ dr_ecef) * 1.0e3
        enu_vel_m_s = (self.rot_ecef_to_enu @ v_ecef) * 1.0e3

        rng_m = max(float(np.linalg.norm(enu_pos_m)), 1.0)
        l, m, n = (enu_pos_m / rng_m).tolist()
        el_deg = math.degrees(math.asin(max(-1.0, min(1.0, n))))
        az_deg = math.degrees(math.atan2(l, m)) % 360.0

        u_hat = enu_pos_m / rng_m
        r_dot = float(np.dot(u_hat, enu_vel_m_s))
        u_sq = float(np.dot(enu_vel_m_s, enu_vel_m_s))
        # Central-body gravitational acceleration at satellite ECEF radius (Vierinen Eq. 2-4)
        mu_m3_s2 = 3.986004418e14
        r_sat_m = np.linalg.norm(r_ecef * 1.0e3)
        a_grav_ecef = -(mu_m3_s2 / (r_sat_m**3)) * (r_ecef * 1.0e3)
        a_grav_enu = self.rot_ecef_to_enu @ a_grav_ecef
        r_ddot = ((u_sq - r_dot * r_dot) / rng_m) + float(np.dot(u_hat, a_grav_enu))

        # Transversal sky angular rates dl/dt, dm/dt in unit-vector plane
        v_perp = enu_vel_m_s - r_dot * u_hat
        dl_dt = float(v_perp[0] / rng_m)
        dm_dt = float(v_perp[1] / rng_m)

        lam = _C_LIGHT / float(carrier_hz)
        doppler_hz = -r_dot / lam
        doppler_rate_hz_s = -r_ddot / lam

        return {
            "l": float(l),
            "m": float(m),
            "n": float(n),
            "el_deg": float(el_deg),
            "az_deg": float(az_deg),
            "range_m": float(rng_m),
            "range_rate_m_s": float(r_dot),
            "range_accel_m_s2": float(r_ddot),
            "dl_dt": dl_dt,
            "dm_dt": dm_dt,
            "doppler_hz": float(doppler_hz),
            "doppler_rate_hz_s": float(doppler_rate_hz_s),
        }

    def build_tle_target_payload(
        self, beam_id: int, unix_s: float, frame_duration_s: float = 0.0512
    ) -> Dict[str, Any]:
        """Build JSON payload for `POST /direct_tracker/set_tle_target`."""
        s0 = self.propagate_unix(unix_s)
        s1 = self.propagate_unix(unix_s + float(frame_duration_s))
        return {
            "beam_id": int(beam_id),
            "l0": s0["l"],
            "m0": s0["m"],
            "l1": s1["l"],
            "m1": s1["m"],
            "frame_duration_s": float(frame_duration_s),
            "tle_name": self.name,
            "norad_id": self.norad_id,
        }


def lanczos8_resample(signal: np.ndarray, sample_offsets: np.ndarray) -> np.ndarray:
    """8-tap (a=4) Lanczos fractional-delay interpolator (Vierinen et al. 2611.12158v1)."""
    x = np.asarray(signal)
    pos = np.asarray(sample_offsets, dtype=np.float64)
    n = x.shape[0]
    a = 4
    base = np.floor(pos).astype(np.int64)
    frac = pos - base
    offsets = np.arange(-a + 1, a + 1, dtype=np.int64)  # 8 taps: -3 .. +4
    taps_u = frac[:, None] - offsets[None, :]  # distance to each tap

    # Lanczos kernel L(u) = sinc(u) * sinc(u / a) for |u| < a, exact limit 1 at u == 0
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(
            np.abs(taps_u) < 1.0e-12,
            1.0,
            np.sinc(taps_u) * np.sinc(taps_u / float(a)),
        )
    w = np.where(np.abs(taps_u) < float(a), w, 0.0)
    w_sum = np.sum(w, axis=1, keepdims=True)
    w = w / np.where(np.abs(w_sum) > 1.0e-12, w_sum, 1.0)

    idx = base[:, None] + offsets[None, :]
    valid = (idx >= 0) & (idx < n)
    idx_safe = np.clip(idx, 0, max(0, n - 1))
    gathered = x[idx_safe] * valid
    return np.sum(w * gathered, axis=1)


def quadratic_doppler_matched_filter(
    voltages: np.ndarray,
    dt_s: float,
    f_grid_hz: np.ndarray,
    fdot_grid_hz_s: np.ndarray,
) -> Dict[str, Any]:
    """2D quadratic-Doppler matched filter & coherent fraction C (Vierinen Eq. 8-10)."""
    z = np.asarray(voltages, dtype=np.complex128)
    n_p = z.size
    if n_p == 0:
        raise ValueError("Empty voltage array")
    total_power = max(float(np.sum(np.abs(z) ** 2)), 1.0e-18)
    t_axis = (np.arange(n_p, dtype=np.float64) - 0.5 * (n_p - 1)) * float(dt_s)
    f_grid = np.asarray(f_grid_hz, dtype=np.float64)
    fdot_grid = np.asarray(fdot_grid_hz_s, dtype=np.float64)

    coh_map = np.zeros((fdot_grid.size, f_grid.size), dtype=np.float64)
    for i, fdot in enumerate(fdot_grid):
        for j, f0 in enumerate(f_grid):
            phase = -2.0 * math.pi * (f0 * t_axis + 0.5 * fdot * (t_axis * t_axis))
            z_d = z * np.exp(1j * phase)
            coh_sum = np.sum(z_d)
            coh_map[i, j] = float((np.abs(coh_sum) ** 2) / (float(n_p) * total_power))

    best_flat = int(np.argmax(coh_map))
    best_i, best_j = divmod(best_flat, f_grid.size)
    return {
        "best_f_hz": float(f_grid[best_j]),
        "best_fdot_hz_s": float(fdot_grid[best_i]),
        "coherent_fraction": float(coh_map[best_i, best_j]),
        "coh_map": coh_map,
    }


def fit_beam_passage_envelope(
    l_track: np.ndarray,
    m_track: np.ndarray,
    power_series: np.ndarray,
    fwhm_init_rad: float = 0.20,
) -> Dict[str, float]:
    """Fit P(t) = B + A * exp(-4 ln(2) * ((l-dl)^2 + (m-dm)^2) / w^2) across orbital transit."""
    l_arr = np.asarray(l_track, dtype=np.float64)
    m_arr = np.asarray(m_track, dtype=np.float64)
    p_arr = np.asarray(power_series, dtype=np.float64)
    if l_arr.size < 6 or l_arr.shape != m_arr.shape or l_arr.shape != p_arr.shape:
        raise ValueError("Need >= 6 aligned points for beam-passage envelope fit")

    b0 = float(np.percentile(p_arr, 15.0))
    a0 = max(float(np.max(p_arr) - b0), 1.0e-3)
    pk_idx = int(np.argmax(p_arr))
    dl0 = float(l_arr[pk_idx])
    dm0 = float(m_arr[pk_idx])
    x0 = np.array([b0, a0, dl0, dm0, float(fwhm_init_rad)], dtype=np.float64)
    ln2_4 = 4.0 * math.log(2.0)

    def _res(params: np.ndarray) -> np.ndarray:
        b, a, dl, dm, w = params
        r2 = (l_arr - dl) ** 2 + (m_arr - dm) ** 2
        pred = b + a * np.exp(-ln2_4 * r2 / max(w * w, 1.0e-6))
        return pred - p_arr

    sol = least_squares(
        _res,
        x0,
        bounds=([-np.inf, 0.0, -1.0, -1.0, 0.02], [np.inf, np.inf, 1.0, 1.0, 1.5]),
        method="trf",
    )
    b, a, dl, dm, w = sol.x.tolist()
    rmse = float(math.sqrt(float(np.mean(sol.fun * sol.fun))))
    return {
        "baseline": float(b),
        "amplitude": float(a),
        "dl_offset": float(dl),
        "dm_offset": float(dm),
        "fwhm_rad": float(w),
        "fit_rmse": rmse,
    }
