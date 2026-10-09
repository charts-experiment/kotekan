#!/usr/bin/env python3
r"""CHARTS Ground-Truth X-Engine Verification & Reference Modules.

Pure-NumPy reference implementations and equivalence checkers for Kotekan
X-engine pipeline stages:
  - `reference_visibilities`: baseline correlation matrix matching `cudaCorrelatorAstron`
  - `reference_beamform`: coherent formed beam matching `cudaDirectBeamTracker`
  - `verify_visibility_phasing`: visibility fringe phase vs. geometric prediction
  - `verify_pointing`: peak power recovery and pointing offset (< 1e-4 rad)
  - `verify_quantization_snr_loss`: int4x2 quantization SNR loss (< 0.5 dB, AGENTS.md §3)
  - `verify_transit_lightcurve`: formed beam peak aligned with transit time

Theory & Literature Foundations:
  - Thompson, A. R., Moran, J. M., Swenson, G. W. (2017). "Interferometry and
    Synthesis in Radio Astronomy", 3rd ed., §3 & §4 (visibility products & phasing).
  - Buschmann, B. A. P. (2025). "Design and Implementation of the F-Engine for
    the CHARTS Project", Master's Thesis, §4 (delay and fringe phase conventions).
  - AGENTS.md §3: Ground-truth equivalence criteria (pointing < 1e-4 rad, SNR loss < 0.5 dB).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from .constants import C_LIGHT, CHARTS_LATITUDE_DEG, CHARTS_LONGITUDE_DEG
from .fengine import measure_quantization_snr_loss
from .sky import direction_cosines_track


# ---------------------------------------------------------------------------
# 1. Pure-NumPy Reference Implementations
# ---------------------------------------------------------------------------

def reference_visibilities(voltages: np.ndarray) -> np.ndarray:
    r"""Computes time-averaged baseline visibility matrices (cudaCorrelatorAstron parity).

    Mathematical Formulation:
    .. math::

        V_{ij}(f) = \frac{1}{N_\mathrm{samp}} \sum_{t=0}^{N_\mathrm{samp}-1} v_i(t, f)\, v_j^*(t, f)

    Parameters
    ----------
    voltages : np.ndarray
        Complex baseband voltage array of shape ``(n_samples, n_freq, n_ant)``.

    Returns
    -------
    visibilities : np.ndarray
        Complex correlation matrix of shape ``(n_freq, n_ant, n_ant)``.
    """
    v = np.asarray(voltages, dtype=np.complex64)
    if v.ndim != 3:
        raise ValueError(f"Expected voltages with shape (n_samples, n_freq, n_ant), got {v.shape}")

    n_samples = v.shape[0]
    # Einsum: sum over time t for frequency f and antenna pair (i, j)
    vis = np.einsum("tfi,tfj->fij", v, np.conj(v), optimize=True) / float(n_samples)
    return vis


def reference_beamform(
    voltages: np.ndarray,
    pos_x: np.ndarray,
    pos_y: np.ndarray,
    freqs_hz: np.ndarray,
    ra_deg: float,
    dec_deg: float,
    unix_times_s: Union[float, np.ndarray, Sequence[float]],
    active_antennas: Optional[Sequence[int]] = None,
    lat_deg: float = CHARTS_LATITUDE_DEG,
    lon_deg: float = CHARTS_LONGITUDE_DEG,
) -> np.ndarray:
    r"""Coherent direct beamforming (cudaDirectBeamTracker parity).

    Mathematical Formulation:
    .. math::

        \tau_a(t) = \frac{x_a\,l(t) + y_a\,m(t)}{c}
        \qquad
        w_a(f, t) = \frac{1}{\sqrt{N_\mathrm{active}}} \exp\left(+j 2\pi f\,\tau_a(t)\right)

        b(t, f) = \sum_{a \in \mathrm{active}} w_a(f, t)\, v_a(t, f)

    Parameters
    ----------
    voltages : np.ndarray
        Complex baseband array of shape ``(n_samples, n_freq, n_ant)``.
    pos_x, pos_y : np.ndarray
        Antenna positions in meters, shape ``(n_ant,)``.
    freqs_hz : np.ndarray
        Channel center frequencies in Hz, shape ``(n_freq,)``.
    ra_deg, dec_deg : float
        Target celestial coordinates in degrees (J2000).
    unix_times_s : float or np.ndarray
        Unix UTC timestamp for each time sample, shape ``(n_samples,)`` or scalar.
    active_antennas : sequence of int, optional
        List of active antenna indices. Defaults to all antennas.
    lat_deg, lon_deg : float
        Observatory coordinates.

    Returns
    -------
    formed_beams : np.ndarray
        Complex formed beam time series of shape ``(n_samples, n_freq)``.
    """
    v = np.asarray(voltages, dtype=np.complex64)
    n_samples, n_freq, n_ant = v.shape
    t = np.atleast_1d(np.asarray(unix_times_s, dtype=np.float64))
    if t.size == 1 and n_samples > 1:
        t = np.full(n_samples, float(t[0]), dtype=np.float64)

    l_track, m_track, _ = direction_cosines_track(ra_deg, dec_deg, t, lat_deg=lat_deg, lon_deg=lon_deg)
    delays_s = (l_track[:, None] * pos_x[None, :] + m_track[:, None] * pos_y[None, :]) / C_LIGHT  # (n_samples, n_ant)

    # Conjugate steering phase: +2pi * f * delay
    phases = (2.0 * math.pi) * (freqs_hz[None, :, None] * delays_s[:, None, :])  # (n_samples, n_freq, n_ant)

    if active_antennas is not None:
        mask = np.zeros(n_ant, dtype=np.float32)
        for a in active_antennas:
            if 0 <= a < n_ant:
                mask[a] = 1.0
        n_active = max(1, int(np.sum(mask)))
    else:
        mask = np.ones(n_ant, dtype=np.float32)
        n_active = n_ant

    weights = (np.exp(1j * phases) * (mask[None, None, :] / math.sqrt(n_active))).astype(np.complex64)
    formed = np.sum(v * weights, axis=-1)  # (n_samples, n_freq)
    return formed


# ---------------------------------------------------------------------------
# 2. Equivalence Checkers & Ground-Truth Verifiers
# ---------------------------------------------------------------------------

def verify_visibility_phasing(
    voltages: np.ndarray,
    pos_x: np.ndarray,
    pos_y: np.ndarray,
    freqs_hz: np.ndarray,
    ra_deg: float,
    dec_deg: float,
    unix_times_s: Union[float, np.ndarray, Sequence[float]],
    max_phase_err_rad: float = 0.05,
    lat_deg: float = CHARTS_LATITUDE_DEG,
    lon_deg: float = CHARTS_LONGITUDE_DEG,
) -> Dict[str, Any]:
    r"""Verifies measured visibility phase against geometric delay predictions.

    Theoretical fringe phase relation:
    .. math::

        \tau_{ij} = \tau_j - \tau_i
        \qquad
        \psi_{ij}(f) = 2\pi f\,\tau_{ij}
        \qquad
        |\arg(V_{ij}(f)) - \psi_{ij}(f)| \le \epsilon

    Parameters
    ----------
    voltages : np.ndarray
        Complex baseband array of shape ``(n_samples, n_freq, n_ant)``.
    pos_x, pos_y : np.ndarray
        Antenna coordinates in meters.
    freqs_hz : np.ndarray
        Channel frequencies in Hz.
    ra_deg, dec_deg : float
        Injected source equatorial coordinates.
    unix_times_s : array-like
        Sample timestamps.
    max_phase_err_rad : float
        Tolerance in radians (default 0.05 rad ≈ 2.8°).

    Returns
    -------
    dict
        Verification summary with max error, RMS error, and passed status.
    """
    vis = reference_visibilities(voltages)
    n_freq, n_ant, _ = vis.shape

    t = np.atleast_1d(np.asarray(unix_times_s, dtype=np.float64))
    t_mean = float(np.mean(t))
    l_arr, m_arr, _ = direction_cosines_track(ra_deg, dec_deg, np.array([t_mean]), lat_deg=lat_deg, lon_deg=lon_deg)
    l0, m0 = float(l_arr[0]), float(m_arr[0])

    delays = (pos_x * l0 + pos_y * m0) / C_LIGHT

    errors = []
    for f_idx in range(n_freq):
        f_hz = float(freqs_hz[f_idx])
        for i in range(n_ant):
            for j in range(i + 1, n_ant):
                tau_ij = delays[j] - delays[i]
                expected_phase = (2.0 * math.pi * f_hz * tau_ij) % (2.0 * math.pi)
                measured_phase = np.angle(vis[f_idx, i, j])
                diff = np.angle(np.exp(1j * (measured_phase - expected_phase)))
                errors.append(abs(float(diff)))

    errors_arr = np.array(errors, dtype=np.float64)
    max_err = float(np.max(errors_arr)) if len(errors_arr) > 0 else 0.0
    rms_err = float(np.sqrt(np.mean(errors_arr ** 2))) if len(errors_arr) > 0 else 0.0
    passed = max_err <= max_phase_err_rad

    return {
        "max_phase_error_rad": max_err,
        "rms_phase_error_rad": rms_err,
        "max_allowed_rad": max_phase_err_rad,
        "passed": bool(passed),
        "num_baselines_checked": len(errors),
    }


def verify_pointing(
    voltages: np.ndarray,
    pos_x: np.ndarray,
    pos_y: np.ndarray,
    freqs_hz: np.ndarray,
    true_ra: float,
    true_dec: float,
    unix_times_s: Union[float, np.ndarray, Sequence[float]],
    angular_search_radius_deg: float = 0.5,
    num_steps: int = 21,
    max_offset_rad: float = 1e-4,
    lat_deg: float = CHARTS_LATITUDE_DEG,
    lon_deg: float = CHARTS_LONGITUDE_DEG,
) -> Dict[str, Any]:
    r"""Verifies formed-beam pointing recovery against injected ground-truth coordinates.

    Forms beams over a 2D angular grid centered at ``(true_ra, true_dec)`` and
    verifies that peak power occurs at the injected coordinate within ``max_offset_rad``.

    Parameters
    ----------
    voltages : np.ndarray
        Complex baseband array of shape ``(n_samples, n_freq, n_ant)``.
    pos_x, pos_y : np.ndarray
        Antenna positions.
    freqs_hz : np.ndarray
        Channel frequencies.
    true_ra, true_dec : float
        Injected source equatorial coordinates in degrees.
    unix_times_s : array-like
        Sample timestamps.
    angular_search_radius_deg : float
        Search half-width in degrees.
    num_steps : int
        Grid resolution per axis.
    max_offset_rad : float
        Maximum angular error tolerance in radians (default 1e-4 rad).

    Returns
    -------
    dict
        Verification summary with peak coordinates, angular offset, and passed status.
    """
    cos_dec = max(0.1, math.cos(math.radians(true_dec)))
    d_ra = np.linspace(-angular_search_radius_deg / cos_dec, angular_search_radius_deg / cos_dec, num_steps)
    d_dec = np.linspace(-angular_search_radius_deg, angular_search_radius_deg, num_steps)

    power_grid = np.zeros((num_steps, num_steps), dtype=np.float64)

    for i, dra in enumerate(d_ra):
        ra_probe = float(true_ra + dra)
        for j, ddec in enumerate(d_dec):
            dec_probe = float(true_dec + ddec)
            beam = reference_beamform(
                voltages, pos_x, pos_y, freqs_hz, ra_probe, dec_probe,
                unix_times_s, lat_deg=lat_deg, lon_deg=lon_deg,
            )
            power_grid[i, j] = float(np.sum(np.abs(beam) ** 2))

    max_idx = np.unravel_index(np.argmax(power_grid), power_grid.shape)
    peak_ra = float(true_ra + d_ra[max_idx[0]])
    peak_dec = float(true_dec + d_dec[max_idx[1]])

    # Tangent plane angular separation in radians
    sep_deg = math.sqrt(((peak_ra - true_ra) * cos_dec) ** 2 + (peak_dec - true_dec) ** 2)
    offset_rad = math.radians(sep_deg)
    passed = offset_rad <= max_offset_rad

    return {
        "true_ra_deg": float(true_ra),
        "true_dec_deg": float(true_dec),
        "peak_ra_deg": peak_ra,
        "peak_dec_deg": peak_dec,
        "offset_rad": offset_rad,
        "max_allowed_rad": max_offset_rad,
        "passed": bool(passed),
        "peak_power": float(power_grid[max_idx]),
    }


def verify_quantization_snr_loss(
    unquantized_voltages: np.ndarray,
    quantized_voltages: np.ndarray,
    max_loss_db: float = 0.5,
) -> Dict[str, Any]:
    """Verifies that int4x2 quantization SNR loss is within the AGENTS.md §3 bound (< 0.5 dB)."""
    loss_db = measure_quantization_snr_loss(unquantized_voltages, quantized_voltages)
    passed = loss_db <= max_loss_db
    return {
        "snr_loss_db": float(loss_db),
        "max_allowed_db": float(max_loss_db),
        "passed": bool(passed),
    }


def verify_transit_lightcurve(
    timestamps_s: np.ndarray,
    power_series: np.ndarray,
    expected_transit_time_s: float,
    tolerance_s: float = 1.0,
) -> Dict[str, Any]:
    """Verifies that beam power lightcurve peaks at the expected celestial transit time."""
    t = np.asarray(timestamps_s, dtype=np.float64)
    p = np.asarray(power_series, dtype=np.float64)

    if len(t) == 0 or len(p) == 0:
        raise ValueError("Empty timestamps or power series")

    peak_idx = int(np.argmax(p))
    peak_time_s = float(t[peak_idx])
    offset_s = abs(peak_time_s - expected_transit_time_s)
    passed = offset_s <= tolerance_s

    return {
        "peak_time_s": peak_time_s,
        "expected_transit_time_s": float(expected_transit_time_s),
        "offset_s": float(offset_s),
        "tolerance_s": float(tolerance_s),
        "passed": bool(passed),
        "peak_power": float(p[peak_idx]),
    }
