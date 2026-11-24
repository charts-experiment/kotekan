#!/usr/bin/env python3
"""Mutual coupling noise diagnostics & array azimuth/tilt spatial self-calibration (CAL-2.4).

References:
  - Yu et al. (2611.08953v1, Sec. 2.6, Eq. 4-6): Mutual coupling noise fraction p_mc & T_mc
  - BURSTT-256 (2611.07832v1, Sec. 2.3.1): Global array azimuth rotation & planar tilt
  - PR #37 charts/experimental/spatial_calibration.py: BeamSpaceDelayCalibrator & closure
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares

try:
    from charts_constants import C_LIGHT, SUBBAND0_FREQS_MHZ, SUBBAND1_FREQS_MHZ

    _C_LIGHT = float(C_LIGHT.value) if hasattr(C_LIGHT, "value") else float(C_LIGHT)
    DEFAULT_FREQS_MHZ = np.concatenate(
        [np.asarray(SUBBAND0_FREQS_MHZ, dtype=np.float64), np.asarray(SUBBAND1_FREQS_MHZ, dtype=np.float64)]
    )
except Exception:
    _C_LIGHT = 299792458.0
    DEFAULT_FREQS_MHZ = np.concatenate(
        [np.linspace(300.0, 400.5, 336), np.linspace(400.8, 501.3, 336)]
    )


@dataclass
class DelayCalResult:
    """Estimated residual per-antenna delays and pointing/closure decomposition."""

    delays_ns: np.ndarray
    dl_tilt: float
    dm_tilt: float
    closure_delay_ns: float
    before_rmse: float
    after_rmse: float
    mean_abs_residual: float
    n_iterations: int


def decompose_tilt_and_closure(
    positions_m: np.ndarray, delays_ns: np.ndarray
) -> Tuple[float, float, float]:
    """Decompose per-antenna delays into planar tilt (dl, dm) and residual non-planar closure."""
    pos = np.asarray(positions_m, dtype=np.float64)
    tau_s = np.asarray(delays_ns, dtype=np.float64) * 1.0e-9
    design = np.column_stack(
        [np.ones(pos.shape[0]), pos[:, 0] / _C_LIGHT, pos[:, 1] / _C_LIGHT]
    )
    coef, _, _, _ = np.linalg.lstsq(design, tau_s, rcond=None)
    resid_ns = (tau_s - design @ coef) * 1.0e9
    closure_rms = float(np.sqrt(np.mean(resid_ns * resid_ns)))
    return float(coef[1]), float(coef[2]), closure_rms


def array_factor_power(
    positions_m: np.ndarray,
    freqs_mhz: np.ndarray,
    src_l: np.ndarray,
    src_m: np.ndarray,
    beam_l0: np.ndarray,
    beam_m0: np.ndarray,
    antenna_delays_s: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute normalized phased-array beam response |w^H a|^2 across beams and frequencies."""
    pos = np.asarray(positions_m, dtype=np.float64)
    f_hz = np.asarray(freqs_mhz, dtype=np.float64) * 1.0e6
    k = (2.0 * math.pi / _C_LIGHT) * f_hz  # (F,)
    dl = np.asarray(src_l, dtype=np.float64) - np.asarray(beam_l0, dtype=np.float64)
    dm = np.asarray(src_m, dtype=np.float64) - np.asarray(beam_m0, dtype=np.float64)
    # geom shape: (..., A) -> phase (..., A, F)
    geom_m = dl[..., None] * pos[:, 0] + dm[..., None] * pos[:, 1]
    phase = geom_m[..., None] * k
    if antenna_delays_s is not None:
        tau = np.asarray(antenna_delays_s, dtype=np.float64)[:, None]  # (A, 1)
        phase = phase + 2.0 * math.pi * f_hz[None, :] * tau
    resp = np.mean(np.exp(1j * phase), axis=-2)
    return np.real(resp * np.conj(resp))


class BeamSpaceDelayCalibrator:
    """Fit residual element delays from normalized multi-beam calibrator fan ratios."""

    def __init__(
        self,
        positions_m: np.ndarray,
        freqs_mhz: Sequence[float] | np.ndarray = DEFAULT_FREQS_MHZ,
    ) -> None:
        self.positions_m = np.asarray(positions_m, dtype=np.float64)
        self.freqs_mhz = np.asarray(freqs_mhz, dtype=np.float64)
        if self.positions_m.shape[0] < 2:
            raise ValueError("Need at least 2 antennas for delay calibration")

    def _predict_fan_ratios(
        self,
        delays_ns: np.ndarray,
        src_l: np.ndarray,
        src_m: np.ndarray,
        fan_l0: np.ndarray,
        fan_m0: np.ndarray,
        off_l0: np.ndarray,
        off_m0: np.ndarray,
        freqs: np.ndarray,
    ) -> np.ndarray:
        tau_s = np.zeros(self.positions_m.shape[0], dtype=np.float64)
        tau_s[1:] = delays_ns * 1.0e-9

        af_fan = array_factor_power(
            self.positions_m,
            freqs,
            src_l[:, None],
            src_m[:, None],
            fan_l0,
            fan_m0,
            antenna_delays_s=tau_s,
        )
        af_off = array_factor_power(
            self.positions_m,
            freqs,
            src_l,
            src_m,
            off_l0,
            off_m0,
            antenna_delays_s=tau_s,
        )
        diff = af_fan - af_off[:, None, :]
        denom = np.sum(np.abs(diff), axis=1, keepdims=True)
        return diff / np.maximum(denom, 1.0e-9)

    def fit(
        self,
        fan_power: np.ndarray,
        off_power: np.ndarray,
        src_l: np.ndarray,
        src_m: np.ndarray,
        fan_l0: np.ndarray,
        fan_m0: np.ndarray,
        off_l0: np.ndarray,
        off_m0: np.ndarray,
        freq_mask: Optional[np.ndarray] = None,
        bounds_ns: Tuple[float, float] = (-2.5, 2.5),
        reg_weight: float = 1.0e-4,
    ) -> DelayCalResult:
        p_fan = np.asarray(fan_power, dtype=np.float64)
        p_off = np.asarray(off_power, dtype=np.float64)
        s_l = np.atleast_1d(np.asarray(src_l, dtype=np.float64))
        s_m = np.atleast_1d(np.asarray(src_m, dtype=np.float64))
        f_l0 = np.atleast_2d(np.asarray(fan_l0, dtype=np.float64))
        f_m0 = np.atleast_2d(np.asarray(fan_m0, dtype=np.float64))
        o_l0 = np.atleast_1d(np.asarray(off_l0, dtype=np.float64))
        o_m0 = np.atleast_1d(np.asarray(off_m0, dtype=np.float64))

        if p_fan.ndim == 2:
            p_fan = p_fan[None, :, :]
        if p_off.ndim == 1:
            p_off = p_off[None, :]

        if freq_mask is None:
            valid_f = np.all(np.isfinite(p_fan), axis=(0, 1)) & np.all(np.isfinite(p_off), axis=0)
        else:
            valid_f = np.asarray(freq_mask, dtype=bool)
        f_idx = np.flatnonzero(valid_f)
        if f_idx.size < 4:
            raise ValueError("Need at least 4 valid frequency channels for delay fit")

        freqs_sel = self.freqs_mhz[f_idx]
        obs_diff = p_fan[:, :, f_idx] - p_off[:, None, f_idx]
        obs_denom = np.sum(np.abs(obs_diff), axis=1, keepdims=True)
        obs_ratio = obs_diff / np.maximum(obs_denom, 1.0e-9)

        n_free = self.positions_m.shape[0] - 1

        def _residual(x_ns: np.ndarray) -> np.ndarray:
            pred = self._predict_fan_ratios(x_ns, s_l, s_m, f_l0, f_m0, o_l0, o_m0, freqs_sel)
            r_data = (pred - obs_ratio).ravel()
            if reg_weight > 0.0:
                return np.concatenate([r_data, reg_weight * x_ns])
            return r_data

        x0 = np.zeros(n_free, dtype=np.float64)
        before_pred = self._predict_fan_ratios(x0, s_l, s_m, f_l0, f_m0, o_l0, o_m0, freqs_sel)
        before_rmse = float(np.sqrt(np.mean((before_pred - obs_ratio) ** 2)))

        sol = least_squares(
            _residual,
            x0,
            bounds=(bounds_ns[0], bounds_ns[1]),
            method="trf",
            max_nfev=200,
        )
        full_delays = np.zeros(self.positions_m.shape[0], dtype=np.float64)
        full_delays[1:] = sol.x
        full_delays -= np.mean(full_delays)

        after_pred = self._predict_fan_ratios(sol.x, s_l, s_m, f_l0, f_m0, o_l0, o_m0, freqs_sel)
        after_resid = after_pred - obs_ratio
        after_rmse = float(np.sqrt(np.mean(after_resid**2)))
        mean_abs_res = float(np.mean(np.abs(after_resid)))
        dl_tilt, dm_tilt, closure_ns = decompose_tilt_and_closure(self.positions_m, full_delays)

        return DelayCalResult(
            delays_ns=full_delays,
            dl_tilt=dl_tilt,
            dm_tilt=dm_tilt,
            closure_delay_ns=closure_ns,
            before_rmse=before_rmse,
            after_rmse=after_rmse,
            mean_abs_residual=mean_abs_res,
            n_iterations=int(sol.nfev),
        )


def compute_mutual_coupling_noise(
    cov_matrices: np.ndarray,
    weights: Optional[np.ndarray] = None,
    t_rec_k: float = 50.0,
    t_loss_k: float = 5.0,
    t_sky_k: float = 20.0,
    resonant_threshold_k: float = 20.0,
) -> Dict[str, Any]:
    """Compute mutual-coupling power fraction p_mc and coupled noise T_mc (Yu et al. 2611.08953v1 Eq. 4-6).

    p_mc(nu) = Re(sum_{i != k} w_i^* w_k R_{ik}) / (w^H R w)
    T_mc(nu) = (p_mc / (1 - p_mc)) * (T_rec + T_loss + T_sky)
    """
    cov = np.asarray(cov_matrices, dtype=np.complex128)
    if cov.ndim == 2:
        cov = cov[None, :, :]
    if cov.ndim != 3 or cov.shape[1] != cov.shape[2]:
        raise ValueError(f"Expected (n_freq, N, N) or (N, N) covariance, got {cov.shape}")

    n_freq, n_ant, _ = cov.shape
    if weights is None:
        w = np.full((n_freq, n_ant), 1.0 / math.sqrt(n_ant), dtype=np.complex128)
    else:
        w_in = np.asarray(weights, dtype=np.complex128)
        w = np.broadcast_to(w_in, (n_freq, n_ant)).copy()

    t_base = float(t_rec_k) + float(t_loss_k) + float(t_sky_k)
    p_mc = np.zeros(n_freq, dtype=np.float64)
    t_mc_k = np.zeros(n_freq, dtype=np.float64)

    for f in range(n_freq):
        r_mat = cov[f]
        wf = w[f]
        total_p = float(np.real(np.vdot(wf, r_mat @ wf)))
        diag_p = float(np.real(np.sum((np.abs(wf) ** 2) * np.diag(r_mat))))
        cross_p = total_p - diag_p
        if abs(total_p) > 1.0e-18:
            pmc_val = cross_p / total_p
        else:
            pmc_val = 0.0
        # Guard against unphysical division by zero when p_mc -> 1
        pmc_clamped = min(0.95, max(-0.95, pmc_val))
        p_mc[f] = pmc_val
        t_mc_k[f] = (pmc_clamped / (1.0 - pmc_clamped)) * t_base

    spike_indices = np.flatnonzero(t_mc_k > float(resonant_threshold_k)).astype(int).tolist()
    return {
        "p_mc": p_mc,
        "t_mc_k": t_mc_k,
        "mean_p_mc": float(np.mean(p_mc)),
        "mean_t_mc_k": float(np.mean(t_mc_k)),
        "max_t_mc_k": float(np.max(t_mc_k)),
        "resonant_spike_indices": spike_indices,
    }


def fit_array_azimuth_and_tilt(
    positions_m: np.ndarray,
    delays_ns: Optional[np.ndarray] = None,
    measured_lm: Optional[np.ndarray] = None,
    expected_lm: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Recover planar array tilt (dl, dm) and global azimuth rotation dpsi_az (BURSTT-256 Sec. 2.3.1)."""
    dl_tilt = 0.0
    dm_tilt = 0.0
    closure_rms_ns = 0.0
    if delays_ns is not None:
        dl_tilt, dm_tilt, closure_rms_ns = decompose_tilt_and_closure(positions_m, delays_ns)

    azimuth_rot_deg = 0.0
    lm_residual_rms = 0.0
    if measured_lm is not None and expected_lm is not None:
        obs = np.asarray(measured_lm, dtype=np.float64)
        exp = np.asarray(expected_lm, dtype=np.float64)
        if obs.shape != exp.shape or obs.ndim != 2 or obs.shape[1] != 2 or obs.shape[0] < 2:
            raise ValueError("measured_lm and expected_lm must have shape (K >= 2, 2)")

        # Solve rotation R(theta) * expected + shift (dl, dm) ~ measured (Procrustes + shift)
        obs_c = obs - np.mean(obs, axis=0, keepdims=True)
        exp_c = exp - np.mean(exp, axis=0, keepdims=True)
        cross_xy = float(np.sum(exp_c[:, 0] * obs_c[:, 1] - exp_c[:, 1] * obs_c[:, 0]))
        dot_xx = float(np.sum(exp_c[:, 0] * obs_c[:, 0] + exp_c[:, 1] * obs_c[:, 1]))
        theta_rad = math.atan2(cross_xy, dot_xx)
        azimuth_rot_deg = math.degrees(theta_rad)

        c, s = math.cos(theta_rad), math.sin(theta_rad)
        rot = np.array([[c, -s], [s, c]], dtype=np.float64)
        pred_rot = exp @ rot.T
        shift = np.mean(obs - pred_rot, axis=0)
        if delays_ns is None:
            dl_tilt = float(shift[0])
            dm_tilt = float(shift[1])
        resid = obs - (pred_rot + shift[None, :])
        lm_residual_rms = float(math.sqrt(float(np.mean(np.sum(resid * resid, axis=1)))))

    return {
        "azimuth_rot_deg": float(azimuth_rot_deg),
        "dl_tilt": float(dl_tilt),
        "dm_tilt": float(dm_tilt),
        "closure_delay_rms_ns": float(closure_rms_ns),
        "lm_residual_rms": float(lm_residual_rms),
    }
