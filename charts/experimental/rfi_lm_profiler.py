"""Multi-frequency (l, m) RFI profiling, grating-lobe-safe OFF search, and live REST scanner."""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import nnls

from . import FREQS_MHZ, array_factor_power


@dataclass
class LmMapResult:
    """Deconvolved direction-cosine brightness map and summary peaks."""

    l_grid: np.ndarray
    m_grid: np.ndarray
    valid_mask: np.ndarray
    nnls_map: np.ndarray
    matched_map: np.ndarray
    fit_r2: float
    peaks: List[Dict[str, float]]


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
    """Build 2D (l, m) coordinate grid and unit-disk validity mask."""
    axis = np.linspace(-radius_max, radius_max, int(grid_n), dtype=np.float64)
    ll, mm = np.meshgrid(axis, axis, indexing="xy")
    valid = (ll * ll + mm * mm) <= (radius_max * radius_max + 1.0e-12)
    return axis, valid, np.column_stack([ll[valid], mm[valid]])


def _spherical_sep_deg(l1: float, m1: float, l2: np.ndarray, m2: np.ndarray) -> np.ndarray:
    """Compute great-circle separation in degrees between (l1, m1) and (l2, m2)."""
    n1 = float(np.sqrt(max(0.0, 1.0 - l1 * l1 - m1 * m1)))
    n2 = np.sqrt(np.maximum(0.0, 1.0 - l2 * l2 - m2 * m2))
    dot = np.clip(l1 * l2 + m1 * m2 + n1 * n2, -1.0, 1.0)
    return np.degrees(np.arccos(dot))


class MultiFrequencyLmProfiler:
    """Deconvolve sparse RFI/source brightness in (l, m) across 300-501 MHz beams."""

    def __init__(
        self,
        positions_m: np.ndarray,
        freqs_mhz: Sequence[float] | np.ndarray = FREQS_MHZ,
        grid_n: int = 41,
        radius_max: float = 1.0,
    ) -> None:
        self.positions_m = np.asarray(positions_m, dtype=np.float64)
        self.freqs_mhz = np.asarray(freqs_mhz, dtype=np.float64)
        self.grid_n = int(grid_n)
        self.radius_max = float(radius_max)
        self.axis, self.valid_mask, self.sky_lm = _sky_grid(self.grid_n, self.radius_max)

    def build_differential_design(
        self,
        beam_l0: np.ndarray,
        beam_m0: np.ndarray,
        freq_indices: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Tuple[int, int, int]]:
        """Construct cross-beam mean-subtracted array-factor design matrix across snapshots.

        Args:
            beam_l0: Array of shape (S, B) or (B,) beam l0 coordinates.
            beam_m0: Array of shape (S, B) or (B,) beam m0 coordinates.
            freq_indices: Optional subset of channel indices into self.freqs_mhz.

        Returns:
            design_2d: Shape (S * B * F_sel, K_sky) real matrix.
            dims: (S, B, F_sel).
        """
        l0 = np.atleast_2d(np.asarray(beam_l0, dtype=np.float64))
        m0 = np.atleast_2d(np.asarray(beam_m0, dtype=np.float64))
        if l0.shape != m0.shape:
            raise ValueError(f"beam_l0 shape {l0.shape} != beam_m0 shape {m0.shape}")

        freqs = self.freqs_mhz if freq_indices is None else self.freqs_mhz[freq_indices]
        n_snap, n_beam = l0.shape
        n_freq = freqs.size
        n_sky = self.sky_lm.shape[0]

        # Compute AF power in blocks over sky pixels for low peak RAM:
        # Output before flattening has shape (S, B, F, K)
        design = np.empty((n_snap, n_beam, n_freq, n_sky), dtype=np.float64)
        for s in range(n_snap):
            af_s = array_factor_power(
                self.positions_m,
                freqs,
                self.sky_lm[None, :, 0],
                self.sky_lm[None, :, 1],
                l0[s, :, None],
                m0[s, :, None],
            )  # (B, K, F)
            af_s = np.transpose(af_s, (0, 2, 1))  # (B, F, K)
            af_s -= np.mean(af_s, axis=0, keepdims=True)
            design[s] = af_s

        return design.reshape(n_snap * n_beam * n_freq, n_sky), (n_snap, n_beam, n_freq)

    def reconstruct(
        self,
        beam_l0: np.ndarray,
        beam_m0: np.ndarray,
        beam_power: np.ndarray,
        freq_mask: Optional[np.ndarray] = None,
        ridge_lambda: float = 0.04,
        l1_lambda: float = 0.35,
        normalize_per_freq: bool = True,
        top_peaks: int = 5,
    ) -> LmMapResult:
        """Reconstruct non-negative sparse brightness map in (l, m) using MFS NNLS.

        Args:
            beam_l0: Beam pointing l0, shape (S, B) or (B,).
            beam_m0: Beam pointing m0, shape (S, B) or (B,).
            beam_power: Observed beam powers, shape (S, B, F) or (B, F).
            freq_mask: Boolean mask of clean channels of length F.
            ridge_lambda: Relative L2 Tikhonov regularization strength.
            l1_lambda: Relative L1 sparsity regularization strength (suppresses grating rings).
            normalize_per_freq: Remove passband gain and T_sys via cross-beam normalization.
            top_peaks: Number of top local peaks to extract.
        """
        y_in = np.asarray(beam_power, dtype=np.float64)
        if y_in.ndim == 2:
            y_in = y_in[None, :, :]
        l0 = np.atleast_2d(np.asarray(beam_l0, dtype=np.float64))
        m0 = np.atleast_2d(np.asarray(beam_m0, dtype=np.float64))

        if y_in.shape[:2] != l0.shape or y_in.shape[2] != self.freqs_mhz.size:
            raise ValueError(
                f"Shape mismatch: beam_power={y_in.shape}, l0={l0.shape}, F={self.freqs_mhz.size}"
            )

        if freq_mask is None:
            valid_f = np.all(np.isfinite(y_in), axis=(0, 1)) & (np.min(y_in, axis=(0, 1)) > 0.0)
        else:
            valid_f = np.asarray(freq_mask, dtype=bool)
        freq_idx = np.flatnonzero(valid_f)
        if freq_idx.size == 0:
            raise ValueError("No valid frequency channels after masking")

        y_sub = y_in[:, :, freq_idx].copy()
        y_diff = y_sub - np.mean(y_sub, axis=1, keepdims=True)
        if normalize_per_freq:
            # Divide out passband ripple/system temperature G(f) using cross-beam median
            scale = np.maximum(np.median(y_sub, axis=1, keepdims=True), 1.0e-9)
            y_sub = y_diff / scale
        else:
            y_sub = y_diff

        design, (_, _, _) = self.build_differential_design(l0, m0, freq_indices=freq_idx)
        y_vec = y_sub.reshape(-1)
        valid_rows = np.isfinite(y_vec)
        design = design[valid_rows]
        y_vec = y_vec[valid_rows]

        # Elastic-net (L1 + L2) non-negative least squares via augmented linear system
        col_norm = float(np.sqrt(np.mean(np.sum(design * design, axis=0))))
        reg_l2 = float(ridge_lambda) * max(col_norm, 1.0e-6)
        reg_l1 = float(l1_lambda) * max(col_norm, 1.0e-6)
        n_sky = self.sky_lm.shape[0]
        rows = [design]
        if reg_l2 > 0.0:
            rows.append(reg_l2 * np.eye(n_sky, dtype=np.float64))
        if reg_l1 > 0.0:
            rows.append(reg_l1 * np.ones((1, n_sky), dtype=np.float64))
        a_aug = np.vstack(rows)
        n_extra = a_aug.shape[0] - design.shape[0]
        y_aug = np.concatenate([y_vec, np.zeros(n_extra, dtype=np.float64)])
        x_hat, _ = nnls(a_aug, y_aug)

        pred = design @ x_hat
        ss_res = float(np.sum((y_vec - pred) ** 2))
        ss_tot = float(np.sum((y_vec - np.mean(y_vec)) ** 2))
        fit_r2 = 1.0 - ss_res / ss_tot if ss_tot > 1.0e-15 else 0.0

        # Also evaluate matched-filter correlation map (normalized column inner products)
        y_norm = np.linalg.norm(y_vec)
        d_norms = np.maximum(np.linalg.norm(design, axis=0), 1.0e-12)
        corr_1d = (design.T @ y_vec) / (d_norms * max(y_norm, 1.0e-12))

        nnls_img = np.full((self.grid_n, self.grid_n), np.nan, dtype=np.float64)
        matched_img = np.full((self.grid_n, self.grid_n), np.nan, dtype=np.float64)
        nnls_img[self.valid_mask] = x_hat
        matched_img[self.valid_mask] = corr_1d

        peaks = _extract_lm_peaks(self.axis, nnls_img, matched_img, top_k=top_peaks)
        return LmMapResult(
            l_grid=self.axis,
            m_grid=self.axis,
            valid_mask=self.valid_mask,
            nnls_map=nnls_img,
            matched_map=matched_img,
            fit_r2=fit_r2,
            peaks=peaks,
        )


def _extract_lm_peaks(
    axis: np.ndarray,
    nnls_img: np.ndarray,
    matched_img: np.ndarray,
    top_k: int = 5,
    nms_radius_bins: int = 2,
) -> List[Dict[str, float]]:
    """Extract spatially separated local maxima from the NNLS sky map."""
    work = np.where(np.isfinite(nnls_img), nnls_img, -1.0).copy()
    peaks: List[Dict[str, float]] = []
    n_g = axis.size
    for _ in range(max(1, int(top_k))):
        flat_idx = int(np.argmax(work))
        iy, ix = divmod(flat_idx, n_g)
        val = float(work[iy, ix])
        if val <= 0.0:
            break
        peaks.append(
            {
                "l": float(axis[ix]),
                "m": float(axis[iy]),
                "amplitude": val,
                "matched_corr": float(matched_img[iy, ix]),
            }
        )
        y0 = max(0, iy - nms_radius_bins)
        y1 = min(n_g, iy + nms_radius_bins + 1)
        x0 = max(0, ix - nms_radius_bins)
        x1 = min(n_g, ix + nms_radius_bins + 1)
        work[y0:y1, x0:x1] = -1.0
    return peaks


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

    mean_leak = np.zeros(cand_lm.shape[0], dtype=np.float64)
    max_leak = np.zeros(cand_lm.shape[0], dtype=np.float64)
    for src_l, src_m in sources:
        af = array_factor_power(
            pos,
            freqs,
            src_l,
            src_m,
            cand_lm[:, 0],
            cand_lm[:, 1],
        )  # (N_cand, F)
        mean_af = np.mean(af, axis=1)
        p95_af = np.percentile(af, 95, axis=1)
        mean_leak = np.maximum(mean_leak, mean_af) if src_l != target_l else mean_af
        max_leak = np.maximum(max_leak, p95_af)

    # Combined score prefers low band-averaged leakage and low worst-frequency grating lobe
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


RequesterFn = Callable[[str, str, Optional[Dict[str, Any]]], Dict[str, Any]]


def _default_http_requester(
    method: str, url: str, payload: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Minimal stdlib JSON HTTP client for Kotekan direct_tracker endpoints."""
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=2.5) as resp:
        return json.loads(resp.read().decode("utf-8"))


class LiveRestLmScanner:
    """Drive one spare beam via REST /direct_tracker endpoints to profile live RFI in (l, m)."""

    def __init__(
        self,
        positions_m: np.ndarray,
        freqs_mhz: Sequence[float] | np.ndarray = FREQS_MHZ,
        base_url: str = "http://127.0.0.1:12048",
        prefix: str = "/direct_tracker",
        beam_id: int = 9,
        requester: Optional[RequesterFn] = None,
    ) -> None:
        self.positions_m = np.asarray(positions_m, dtype=np.float64)
        self.freqs_mhz = np.asarray(freqs_mhz, dtype=np.float64)
        self.base_url = base_url.rstrip("/")
        self.prefix = "/" + prefix.strip("/")
        self.beam_id = int(beam_id)
        self._request = requester or _default_http_requester

    def raster_scan(
        self,
        scan_points_lm: Sequence[Tuple[float, float]],
        n_polls_per_point: int = 2,
        grid_n: int = 41,
        ridge_lambda: float = 0.08,
    ) -> Tuple[LmMapResult, np.ndarray, np.ndarray]:
        """Steer spare beam across scan_points_lm, poll beam_power, and reconstruct (l, m)."""
        pts = np.asarray(scan_points_lm, dtype=np.float64)
        spectra: List[np.ndarray] = []

        for l0, m0 in pts:
            self._request(
                "POST",
                f"{self.base_url}{self.prefix}/set_target",
                {"beam_id": self.beam_id, "l0": float(l0), "m0": float(m0)},
            )
            acc = np.zeros(self.freqs_mhz.size, dtype=np.float64)
            for _ in range(max(1, int(n_polls_per_point))):
                bp = self._request("GET", f"{self.base_url}{self.prefix}/beam_power", None)
                acc += np.asarray(bp["beam_power"][self.beam_id], dtype=np.float64)
            spectra.append(acc / max(1, int(n_polls_per_point)))

        meas = np.vstack(spectra)  # (M_points, F)
        profiler = MultiFrequencyLmProfiler(
            self.positions_m, freqs_mhz=self.freqs_mhz, grid_n=grid_n
        )
        # Treat scan raster as S=1 snapshot with M virtual simultaneous beams sharing T_sys(f)
        res = profiler.reconstruct(pts[:, 0], pts[:, 1], meas, ridge_lambda=ridge_lambda)
        return res, pts, meas
