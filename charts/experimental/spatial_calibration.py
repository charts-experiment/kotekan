"""Software spatial calibration and adaptive sidelobe cancellation for multi-beam CHARTS data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares

from . import C_LIGHT_M_S, FREQS_MHZ, array_factor_power


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


@dataclass
class SidelobeCancelResult:
    """Result of per-channel adaptive auxiliary-beam sidelobe cancellation."""

    cleaned_on: np.ndarray
    weights: np.ndarray
    corr_before: float
    corr_after: float
    rms_before: float
    rms_after: float
    suppression_db: float


def decompose_tilt_and_closure(
    positions_m: np.ndarray, delays_ns: np.ndarray
) -> Tuple[float, float, float]:
    """Decompose per-antenna delays into planar tilt (dl, dm) and residual non-planar closure.

    For planar arrays, a linear phase delay gradient tau_a = (x_a * dl + y_a * dm) / c
    shifts effective beam pointing, whereas non-planar residual tau_res captures hardware
    or cable skew orthogonal to pointing.
    """
    pos = np.asarray(positions_m, dtype=np.float64)
    tau_s = np.asarray(delays_ns, dtype=np.float64) * 1.0e-9
    # Fit tau_s ~ c0 + (x * dl + y * dm) / c
    design = np.column_stack(
        [np.ones(pos.shape[0]), pos[:, 0] / C_LIGHT_M_S, pos[:, 1] / C_LIGHT_M_S]
    )
    coef, _, _, _ = np.linalg.lstsq(design, tau_s, rcond=None)
    resid_ns = (tau_s - design @ coef) * 1.0e9
    closure_rms = float(np.sqrt(np.mean(resid_ns * resid_ns)))
    return float(coef[1]), float(coef[2]), closure_rms


class BeamSpaceDelayCalibrator:
    """Fit residual element delays from normalized multi-beam calibrator fan ratios."""

    def __init__(
        self,
        positions_m: np.ndarray,
        freqs_mhz: Sequence[float] | np.ndarray = FREQS_MHZ,
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
        """Predict normalized fan power ratios (S, B_fan, F) canceling T_sys and S_cal."""
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
        )  # (S, B_fan, F)
        af_off = array_factor_power(
            self.positions_m,
            freqs,
            src_l,
            src_m,
            off_l0,
            off_m0,
            antenna_delays_s=tau_s,
        )  # (S, F)

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
        """Fit antenna delays [tau_1 .. tau_{N-1}] in ns relative to antenna 0."""
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
            valid_f = np.all(np.isfinite(p_fan), axis=(0, 1)) & np.all(
                np.isfinite(p_off), axis=0
            )
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
            pred = self._predict_fan_ratios(
                x_ns, s_l, s_m, f_l0, f_m0, o_l0, o_m0, freqs_sel
            )
            diff_vec = (pred - obs_ratio).reshape(-1)
            if reg_weight > 0.0:
                return np.concatenate([diff_vec, np.sqrt(reg_weight) * x_ns])
            return diff_vec

        r0 = self._predict_fan_ratios(
            np.zeros(n_free), s_l, s_m, f_l0, f_m0, o_l0, o_m0, freqs_sel
        )
        before_rmse = float(np.sqrt(np.mean((r0 - obs_ratio) ** 2)))

        sol = least_squares(
            _residual,
            x0=np.zeros(n_free, dtype=np.float64),
            bounds=(bounds_ns[0], bounds_ns[1]),
            loss="soft_l1",
            f_scale=0.05,
            max_nfev=200,
        )
        full_ns = np.concatenate([[0.0], sol.x])
        r1 = self._predict_fan_ratios(
            sol.x, s_l, s_m, f_l0, f_m0, o_l0, o_m0, freqs_sel
        )
        after_rmse = float(np.sqrt(np.mean((r1 - obs_ratio) ** 2)))
        dl_tilt, dm_tilt, closure_ns = decompose_tilt_and_closure(self.positions_m, full_ns)

        return DelayCalResult(
            delays_ns=full_ns,
            dl_tilt=dl_tilt,
            dm_tilt=dm_tilt,
            closure_delay_ns=closure_ns,
            before_rmse=before_rmse,
            after_rmse=after_rmse,
            mean_abs_residual=float(np.mean(np.abs(r1 - obs_ratio))),
            n_iterations=int(sol.nfev),
        )


def _robust_mad(vec: np.ndarray) -> float:
    """Compute consistency-scaled median absolute deviation."""
    med = float(np.median(vec))
    mad = float(np.median(np.abs(vec - med)))
    return max(1.4826 * mad, 1.0e-9)


class AdaptiveSidelobeCanceller:
    """Per-channel Huber-IRLS beam-space sidelobe and 1/f common-mode canceller."""

    def __init__(self, ridge: float = 1.0e-3, irls_iters: int = 4, huber_k: float = 2.5) -> None:
        self.ridge = float(ridge)
        self.irls_iters = max(1, int(irls_iters))
        self.huber_k = float(huber_k)

    def cancel(
        self,
        beam_power: np.ndarray,
        on_beam_idx: int,
        aux_beam_indices: Sequence[int],
        freq_mask: Optional[np.ndarray] = None,
        fit_sample_mask: Optional[np.ndarray] = None,
        center_output: bool = False,
    ) -> SidelobeCancelResult:
        """Regress out correlated auxiliary/interferer beam fluctuations channel by channel.

        Uses Huber IRLS so positive ON-beam transients do not bias sidelobe regression weights.
        """
        arr = np.asarray(beam_power, dtype=np.float64)
        if arr.ndim != 3:
            raise ValueError(f"Expected (T, B, F) beam_power, got shape {arr.shape}")
        n_t, _, n_f = arr.shape
        aux_idx = [int(b) for b in aux_beam_indices]
        if not aux_idx:
            raise ValueError("aux_beam_indices must not be empty")

        on_ts = arr[:, int(on_beam_idx), :].copy()
        aux_ts = arr[:, aux_idx, :].copy()

        f_mask = np.ones(n_f, dtype=bool) if freq_mask is None else np.asarray(freq_mask, dtype=bool)
        if fit_sample_mask is None:
            s_mask = np.ones(n_t, dtype=bool)
        else:
            s_mask = np.asarray(fit_sample_mask, dtype=bool)

        cleaned = np.empty_like(on_ts)
        weights = np.zeros((n_f, 1 + len(aux_idx)), dtype=np.float64)
        n_cols = 1 + len(aux_idx)

        for f in range(n_f):
            y = on_ts[:, f]
            x_aux = aux_ts[:, :, f]
            if not f_mask[f] or not np.all(np.isfinite(y)) or not np.all(np.isfinite(x_aux)):
                cleaned[:, f] = y
                continue

            # Center auxiliary predictors around median so intercept equals ON baseline
            aux_med = np.median(x_aux[s_mask], axis=0, keepdims=True)
            aux_c = x_aux - aux_med
            aux_scale = np.maximum(np.std(aux_c[s_mask], axis=0, keepdims=True), 1.0e-9)
            x_design = np.column_stack([np.ones(n_t, dtype=np.float64), aux_c / aux_scale])

            y_fit = y[s_mask]
            x_fit = x_design[s_mask]
            w_irls = np.ones(y_fit.size, dtype=np.float64)
            beta = np.zeros(n_cols, dtype=np.float64)
            reg_mat = np.diag([0.0] + [self.ridge] * len(aux_idx))

            for _ in range(self.irls_iters):
                sqrt_w = np.sqrt(w_irls)[:, None]
                xw = x_fit * sqrt_w
                yw = y_fit * sqrt_w[:, 0]
                gram = xw.T @ xw + reg_mat * y_fit.size
                rhs = xw.T @ yw
                beta = np.linalg.solve(gram, rhs)
                resid = y_fit - x_fit @ beta
                scale = _robust_mad(resid)
                thresh = self.huber_k * scale
                abs_r = np.abs(resid)
                w_irls = np.where(abs_r <= thresh, 1.0, thresh / np.maximum(abs_r, 1.0e-12))
                # Stronger attenuation of positive ON-only flares so real pulses remain unsubtracted
                pos_flare = resid > (3.5 * scale)
                if np.any(pos_flare):
                    w_irls[pos_flare] *= 0.1

            # Subtract only auxiliary variation, keeping intercept unless center_output=True
            delta = (x_design[:, 1:] @ beta[1:])
            cleaned[:, f] = (y - x_design @ beta) if center_output else (y - delta)
            weights[f, 0] = beta[0]
            weights[f, 1:] = beta[1:] / aux_scale[0]

        valid_cols = np.flatnonzero(f_mask)
        if valid_cols.size == 0:
            raise ValueError("No valid frequency channels provided to AdaptiveSidelobeCanceller")

        bb_on_before = np.median(on_ts[:, valid_cols], axis=1)
        bb_on_after = np.median(cleaned[:, valid_cols], axis=1)
        bb_aux = np.median(aux_ts[:, 0, valid_cols], axis=1)

        def _corr(a: np.ndarray, b: np.ndarray) -> float:
            ac = a - np.mean(a)
            bc = b - np.mean(b)
            den = np.linalg.norm(ac) * np.linalg.norm(bc)
            return float(np.dot(ac, bc) / den) if den > 1.0e-12 else 0.0

        c_before = _corr(bb_on_before, bb_aux)
        c_after = _corr(bb_on_after, bb_aux)
        rms_before = _robust_mad(bb_on_before)
        rms_after = _robust_mad(bb_on_after)
        suppression_db = float(20.0 * np.log10(max(rms_before, 1.0e-12) / max(rms_after, 1.0e-12)))

        return SidelobeCancelResult(
            cleaned_on=cleaned,
            weights=weights,
            corr_before=c_before,
            corr_after=c_after,
            rms_before=rms_before,
            rms_after=rms_after,
            suppression_db=suppression_db,
        )
