#!/usr/bin/env python3
"""Production RFI comb cancellation, grating-lobe-safe OFF-beam placement & sidelobe cancellation.

Promotes validated algorithms from PR #37 for the CHARTS Laguna Carén (Pudahuel) environment:
- `CoherentCombCanceller` (RFI-3.2): OFF-referenced cancellation of the 9.7656 Hz (102.4 ms)
  timing comb and ~50.1 Hz AC mains harmonics.
- `find_optimal_off_beams` (RFI-3.3): algebraic grating-lobe-safe (l, m) OFF-beam placement
  avoiding main beams and periodic sub-array grating aliases of targets & horizon/GEO RFI.
- `AdaptiveSidelobeCanceller` (RFI-3.3): per-channel Huber-IRLS auxiliary-beam regression
  protecting positive astronomical transients.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import median_filter
from scipy.optimize import nnls

try:
    from charts_constants import C_LIGHT, FREQ

    C_LIGHT_M_S: float = float(getattr(C_LIGHT, "value", C_LIGHT))
    FREQS_MHZ: np.ndarray = np.asarray(FREQ, dtype=np.float64)
except Exception:
    C_LIGHT_M_S = 299792458.0
    FREQS_MHZ = 300.0 + 0.3 * np.arange(672, dtype=np.float64)


def altaz_to_lm(alt_deg: float, az_deg: float) -> Tuple[float, float]:
    """Convert topocentric Alt/Az (deg, N through E) into direction cosines (l=E, m=N)."""
    alt_r = math.radians(float(alt_deg))
    az_r = math.radians(float(az_deg))
    return (
        float(math.cos(alt_r) * math.sin(az_r)),
        float(math.cos(alt_r) * math.cos(az_r)),
    )


# Reference Pudahuel/Laguna Caren terrestrial & GEO RFI directions (topocentric ENU l, m)
PUDAHUEL_KNOWN_INTERFERERS: Dict[str, Tuple[float, float]] = {
    "scl_airport": altaz_to_lm(0.5, 42.0),         # Arturo Merino Benitez airport ENE
    "cerro_san_cristobal": altaz_to_lm(1.0, 102.0), # Santiago basin / Cerro San Cristobal ESE
    "muos_1": altaz_to_lm(42.0, 315.0),             # MUOS-1 WNW
    "muos_3": altaz_to_lm(22.0, 62.0),              # MUOS-3 ENE
}


def array_factor_power(
    positions_m: np.ndarray,
    freqs_mhz: Sequence[float] | np.ndarray,
    l_dir: float | np.ndarray,
    m_dir: float | np.ndarray,
    l0: float | np.ndarray,
    m0: float | np.ndarray,
    antenna_delays_s: Optional[np.ndarray] = None,
    conjugate_sign: float = 1.0,
) -> np.ndarray:
    """Evaluate coherent phased-array normalized power response |AF|^2 <= 1."""
    pos = np.asarray(positions_m, dtype=np.float64)
    freqs_hz = np.asarray(freqs_mhz, dtype=np.float64) * 1.0e6
    l_d = np.asarray(l_dir, dtype=np.float64)
    m_d = np.asarray(m_dir, dtype=np.float64)
    l_0 = np.asarray(l0, dtype=np.float64)
    m_0 = np.asarray(m0, dtype=np.float64)

    n_d = np.sqrt(np.maximum(0.0, 1.0 - l_d * l_d - m_d * m_d))
    n_0 = np.sqrt(np.maximum(0.0, 1.0 - l_0 * l_0 - m_0 * m_0))

    du_l = l_d - conjugate_sign * l_0
    du_m = m_d - conjugate_sign * m_0
    du_n = n_d - conjugate_sign * n_0

    tau_geom = (
        du_l[..., None] * pos[:, 0]
        + du_m[..., None] * pos[:, 1]
        + du_n[..., None] * pos[:, 2]
    ) / C_LIGHT_M_S

    tau_tot = tau_geom + np.asarray(antenna_delays_s, dtype=np.float64) if antenna_delays_s is not None else tau_geom
    phase = (2.0 * np.pi) * tau_tot[..., None] * freqs_hz
    af_complex = np.mean(np.exp(1j * phase), axis=-2)
    return (af_complex.real * af_complex.real + af_complex.imag * af_complex.imag).astype(np.float64)


@dataclass
class CombCancelResult:
    """Result of OFF-beam-referenced periodic harmonic comb subtraction."""

    cleaned_on: np.ndarray
    comb_freqs_hz: List[float]
    comb_bins_mask: np.ndarray
    island_gains: List[float]
    comb_rms_before: float
    comb_rms_after: float
    comb_suppression_db: float


class CoherentCombCanceller:
    """Suppress periodic hardware/mains combs in ON beams using OFF-beam spectral templates."""

    def __init__(
        self,
        detrend_s: float = 1.5,
        line_snr_thresh: float = 18.0,
        min_coherence: float = 0.30,
        min_freq_hz: float = 3.0,
        half_width_bins: int = 2,
        max_islands: int = 24,
        max_comb_fraction: float = 0.06,
    ) -> None:
        self.detrend_s = float(detrend_s)
        self.line_snr_thresh = float(line_snr_thresh)
        self.min_coherence = float(min_coherence)
        self.min_freq_hz = float(min_freq_hz)
        self.half_width_bins = max(0, int(half_width_bins))
        self.max_islands = max(1, int(max_islands))
        self.max_comb_fraction = float(max_comb_fraction)

    def cancel(self, on_ts: np.ndarray, off_ts: np.ndarray, dt_s: float) -> CombCancelResult:
        """Detect common-mode narrowband periodic combs from off_ts and subtract from on_ts."""
        on = np.asarray(on_ts, dtype=np.float64)
        off = np.asarray(off_ts, dtype=np.float64)
        if on.shape != off.shape or on.ndim != 1:
            raise ValueError(f"Expected matching 1D arrays, got {on.shape} and {off.shape}")
        n = on.size
        if n < 32 or dt_s <= 0.0:
            raise ValueError("Invalid time-series length or dt_s")

        detrend_w = max(9, int(round(self.detrend_s / dt_s)) | 1)
        detrend_w = min(detrend_w, max(9, (n // 4) | 1))
        on_base = median_filter(on, size=detrend_w, mode="nearest")
        off_base = median_filter(off, size=detrend_w, mode="nearest")
        on_d = on - on_base
        off_d = off - off_base

        freqs = np.fft.rfftfreq(n, d=dt_s)
        on_fft = np.fft.rfft(on_d)
        off_fft = np.fft.rfft(off_d)

        p_on = (on_fft.real * on_fft.real) + (on_fft.imag * on_fft.imag)
        p_off = (off_fft.real * off_fft.real) + (off_fft.imag * off_fft.imag)
        bg_off = np.maximum(median_filter(p_off, size=61, mode="nearest"), 1.0e-18)
        snr_off = p_off / bg_off

        cross_re = (on_fft.real * off_fft.real) + (on_fft.imag * off_fft.imag)
        coh = cross_re / np.sqrt(np.maximum(p_on * p_off, 1.0e-24))
        peaks = np.flatnonzero(
            (freqs >= self.min_freq_hz) & (snr_off >= self.line_snr_thresh) & (coh >= self.min_coherence)
        )

        if peaks.size == 0:
            return CombCancelResult(
                cleaned_on=on.copy(),
                comb_freqs_hz=[],
                comb_bins_mask=np.zeros(freqs.size, dtype=bool),
                island_gains=[],
                comb_rms_before=0.0,
                comb_rms_after=0.0,
                comb_suppression_db=0.0,
            )

        order = peaks[np.argsort(snr_off[peaks])[::-1]]
        comb_mask = np.zeros(freqs.size, dtype=bool)
        max_bins = max(1, int(self.max_comb_fraction * freqs.size))
        selected_freqs: List[float] = []

        for k in order:
            lo = max(1, int(k) - self.half_width_bins)
            hi = min(freqs.size, int(k) + self.half_width_bins + 1)
            new_bins = int(np.sum(~comb_mask[lo:hi]))
            if int(np.sum(comb_mask)) + new_bins > max_bins:
                continue
            if not np.any(comb_mask[lo:hi]):
                selected_freqs.append(float(freqs[k]))
            comb_mask[lo:hi] = True
            if len(selected_freqs) >= self.max_islands:
                break

        islands: List[Tuple[int, int]] = []
        in_run = False
        run_lo = 0
        for idx, flag in enumerate(comb_mask):
            if flag and not in_run:
                run_lo = idx
                in_run = True
            elif not flag and in_run:
                islands.append((run_lo, idx))
                in_run = False
        if in_run:
            islands.append((run_lo, freqs.size))

        templates = []
        for lo, hi in islands:
            spec_i = np.zeros_like(off_fft)
            spec_i[lo:hi] = off_fft[lo:hi]
            templates.append(np.fft.irfft(spec_i, n=n))

        basis = np.column_stack(templates)
        gains, _ = nnls(basis, on_d)
        comb_est = basis @ gains
        cleaned_on = on - comb_est

        cleaned_fft = np.fft.rfft(cleaned_on - median_filter(cleaned_on, size=detrend_w, mode="nearest"))
        p_before = float(np.sum((on_fft.real[comb_mask] ** 2) + (on_fft.imag[comb_mask] ** 2)))
        p_after = float(np.sum((cleaned_fft.real[comb_mask] ** 2) + (cleaned_fft.imag[comb_mask] ** 2)))
        rms_b = float(np.sqrt(p_before) / n)
        rms_a = float(np.sqrt(p_after) / n)
        supp_db = float(10.0 * np.log10(max(p_before, 1.0e-18) / max(p_after, 1.0e-18)))

        return CombCancelResult(
            cleaned_on=cleaned_on,
            comb_freqs_hz=sorted(selected_freqs),
            comb_bins_mask=comb_mask,
            island_gains=[float(g) for g in gains],
            comb_rms_before=rms_b,
            comb_rms_after=rms_a,
            comb_suppression_db=supp_db,
        )


@dataclass
class OffBeamCandidate:
    """Candidate OFF beam coordinates minimizing grating-lobe sidelobe leakage."""

    l0: float
    m0: float
    zenith_deg: float
    target_sep_deg: float
    mean_leakage: float
    max_leakage: float
    band_power_ratio_db: float


def _sky_grid(grid_n: int, radius_max: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = np.linspace(-radius_max, radius_max, int(grid_n), dtype=np.float64)
    ll, mm = np.meshgrid(axis, axis, indexing="xy")
    valid = (ll * ll + mm * mm) <= (radius_max * radius_max + 1.0e-12)
    return axis, valid, np.column_stack([ll[valid], mm[valid]])


def _spherical_sep_deg(l1: float, m1: float, l2: np.ndarray, m2: np.ndarray) -> np.ndarray:
    n1 = float(np.sqrt(max(0.0, 1.0 - l1 * l1 - m1 * m1)))
    n2 = np.sqrt(np.maximum(0.0, 1.0 - l2 * l2 - m2 * m2))
    dot = np.clip(l1 * l2 + m1 * m2 + n1 * n2, -1.0, 1.0)
    return np.degrees(np.arccos(dot))


def find_optimal_off_beams(
    positions_m: np.ndarray,
    freqs_mhz: Sequence[float] | np.ndarray,
    target_l: float,
    target_m: float,
    interferers: Iterable[Tuple[float, float]] = (),
    min_sep_deg: float = 15.0,
    max_zenith_deg: float = 70.0,
    grid_n: int = 61,
    top_k: int = 3,
) -> List[OffBeamCandidate]:
    """Search (l, m) for OFF beams minimizing grating-lobe leakage from target and interferers."""
    pos = np.asarray(positions_m, dtype=np.float64)
    freqs = np.asarray(freqs_mhz, dtype=np.float64)
    r_max = float(np.sin(np.radians(max_zenith_deg)))
    _, _, cand_lm = _sky_grid(grid_n, radius_max=r_max)

    sep_deg = _spherical_sep_deg(target_l, target_m, cand_lm[:, 0], cand_lm[:, 1])
    keep = sep_deg >= float(min_sep_deg)
    sources = [(float(target_l), float(target_m))]
    for il, im in interferers:
        sources.append((float(il), float(im)))
        sep_i = _spherical_sep_deg(float(il), float(im), cand_lm[:, 0], cand_lm[:, 1])
        keep &= sep_i >= float(min_sep_deg)

    cand_lm = cand_lm[keep]
    sep_deg = sep_deg[keep]
    if cand_lm.shape[0] == 0:
        raise ValueError("No valid OFF beam candidates satisfy angular separation bounds")

    total_score = np.zeros(cand_lm.shape[0], dtype=np.float64)
    for idx_s, (src_l, src_m) in enumerate(sources):
        af = array_factor_power(pos, freqs, src_l, src_m, cand_lm[:, 0], cand_lm[:, 1])
        wt = 1.0 if idx_s == 0 else 0.75
        total_score += wt * (0.65 * np.mean(af, axis=1) + 0.35 * np.max(af, axis=1))

    order = np.argsort(total_score)
    chosen: List[OffBeamCandidate] = []
    min_cand_sep = 12.0
    for idx in order:
        l_c, m_c = float(cand_lm[idx, 0]), float(cand_lm[idx, 1])
        if any(_spherical_sep_deg(l_c, m_c, np.array([c.l0]), np.array([c.m0]))[0] < min_cand_sep for c in chosen):
            continue
        zen_deg = float(np.degrees(np.arcsin(np.clip(np.hypot(l_c, m_c), 0.0, 1.0))))
        af_tgt = array_factor_power(pos, freqs, target_l, target_m, l_c, m_c)
        mean_t = float(np.mean(af_tgt))
        max_t = float(np.max(af_tgt))
        chosen.append(
            OffBeamCandidate(
                l0=l_c,
                m0=m_c,
                zenith_deg=zen_deg,
                target_sep_deg=float(sep_deg[idx]),
                mean_leakage=mean_t,
                max_leakage=max_t,
                band_power_ratio_db=float(10.0 * np.log10(max(mean_t, 1.0e-12))),
            )
        )
        if len(chosen) >= top_k:
            break
    return chosen


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


def _robust_mad(vec: np.ndarray) -> float:
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
        """Regress out correlated auxiliary/interferer beam fluctuations channel by channel."""
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
        s_mask = np.ones(n_t, dtype=bool) if fit_sample_mask is None else np.asarray(fit_sample_mask, dtype=bool)

        cleaned = np.empty_like(on_ts)
        weights = np.zeros((n_f, 1 + len(aux_idx)), dtype=np.float64)
        n_cols = 1 + len(aux_idx)

        for f in range(n_f):
            y = on_ts[:, f]
            x_aux = aux_ts[:, :, f]
            if not f_mask[f] or not np.all(np.isfinite(y)) or not np.all(np.isfinite(x_aux)):
                cleaned[:, f] = y
                continue

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
                pos_flare = resid > (3.5 * scale)
                if np.any(pos_flare):
                    w_irls[pos_flare] *= 0.1

            delta = x_design[:, 1:] @ beta[1:]
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
