"""Unsupervised RFI regime classification, OFF-referenced comb cancellation, and transient scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.optimize import nnls

from . import FREQS_MHZ, K_DM_PC_CM3_MHZ2_S


@dataclass
class RfiClusteringResult:
    """Cluster assignments, anomaly scores, and per-cluster feature centroids."""

    labels: np.ndarray
    cluster_names: Dict[int, str]
    mahalanobis_distance: np.ndarray
    features: np.ndarray
    pca_2d: np.ndarray
    cluster_summaries: List[Dict[str, Any]]
    meta: Dict[str, np.ndarray]


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


@dataclass
class TransientCandidate:
    """Detected dispersed transient candidate with spatial OFF coincidence check."""

    sample_idx: int
    dm_pc_cm3: float
    width_samples: int
    snr_on: float
    snr_off_at_peak: float
    snr_off_dm0: float
    rfi_vetoed: bool


def extract_rfi_snapshot_features(
    snapshots: Sequence[Dict[str, Any]],
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Build standardized physical feature vectors from rfi_channel_snapshots.jsonl rows."""
    feats: List[np.ndarray] = []
    meta_snap: List[int] = []
    meta_freq: List[float] = []
    meta_bits: List[int] = []
    meta_frac: List[float] = []
    meta_fz: List[float] = []

    for s_idx, row in enumerate(snapshots):
        chans = row.get("data", {}).get("channels", [])
        if not chans:
            continue
        p = np.asarray([float(c.get("mean_power", 0.0)) for c in chans], dtype=np.float64)
        fz = np.asarray([float(c.get("freq_mad_z", 0.0)) for c in chans], dtype=np.float64)
        msk = np.asarray([float(c.get("mean_sk", 1.0)) for c in chans], dtype=np.float64)
        xsk = np.asarray([float(c.get("max_sk", 1.0)) for c in chans], dtype=np.float64)
        frac = np.asarray(
            [float(c.get("flagged_chunk_fraction", 0.0)) for c in chans], dtype=np.float64
        )
        freqs = np.asarray([float(c.get("freq_mhz", 0.0)) for c in chans], dtype=np.float64)
        bits = np.asarray([int(c.get("flag_bits", 0)) for c in chans], dtype=np.int64)

        valid = p > 0.0
        if not np.any(valid):
            continue

        order = np.argsort(freqs[valid])
        pv = p[valid][order]
        fzv = fz[valid][order]
        mskv = msk[valid][order]
        xskv = xsk[valid][order]
        fracv = frac[valid][order]
        freqsv = freqs[valid][order]
        bitsv = bits[valid][order]

        local_med = np.maximum(median_filter(pv, size=17, mode="nearest"), 1.0e-6)
        rel_pwr = np.clip((pv - local_med) / local_med, -1.0, 50.0)
        sk_spread = np.clip(xskv - mskv, 0.0, 100.0)

        mat = np.column_stack(
            [
                np.log10(np.maximum(pv, 1.0e-6)),
                np.clip(fzv, -30.0, 200.0),
                np.log10(np.maximum(mskv, 1.0e-3)),
                np.log10(np.maximum(xskv, 1.0e-3)),
                fracv,
                rel_pwr,
                sk_spread,
            ]
        )
        feats.append(mat)
        meta_snap.extend([s_idx] * pv.size)
        meta_freq.extend(freqsv.tolist())
        meta_bits.extend(bitsv.tolist())
        meta_frac.extend(fracv.tolist())
        meta_fz.extend(fzv.tolist())

    if not feats:
        raise ValueError("No valid non-zero RFI channel snapshots found")

    return np.vstack(feats), {
        "snapshot_idx": np.asarray(meta_snap, dtype=np.int64),
        "freq_mhz": np.asarray(meta_freq, dtype=np.float64),
        "flag_bits": np.asarray(meta_bits, dtype=np.int64),
        "flagged_fraction": np.asarray(meta_frac, dtype=np.float64),
        "freq_mad_z": np.asarray(meta_fz, dtype=np.float64),
    }


def _kmeans_pp_init(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Deterministic k-means++ centroid initialization."""
    n = x.shape[0]
    centers = np.empty((k, x.shape[1]), dtype=np.float64)
    centers[0] = x[int(rng.integers(0, n))]
    d2 = np.sum((x - centers[0]) ** 2, axis=1)
    for c_idx in range(1, k):
        tot = float(np.sum(d2))
        probs = np.full(n, 1.0 / n) if tot <= 1.0e-15 else (d2 / tot)
        chosen = int(rng.choice(n, p=probs))
        centers[c_idx] = x[chosen]
        d2 = np.minimum(d2, np.sum((x - centers[c_idx]) ** 2, axis=1))
    return centers


class UnsupervisedRfiClassifier:
    """Standardized k-means++ and Mahalanobis RFI regime classifier for telemetry streams."""

    def __init__(self, n_clusters: int = 5, max_iter: int = 50, seed: int = 2026) -> None:
        self.n_clusters = max(2, int(n_clusters))
        self.max_iter = max(5, int(max_iter))
        self.seed = int(seed)
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._centers: Optional[np.ndarray] = None
        self._cluster_names: Dict[int, str] = {}
        self._pca_basis: Optional[np.ndarray] = None
        self._prec_mats: List[np.ndarray] = []

    def fit_predict(
        self,
        features: np.ndarray,
        meta: Optional[Dict[str, np.ndarray]] = None,
    ) -> RfiClusteringResult:
        """Fit standardized k-means++ clusters, Mahalanobis anomaly distances, and regime names."""
        x_raw = np.asarray(features, dtype=np.float64)
        if x_raw.ndim != 2 or x_raw.shape[0] < self.n_clusters:
            raise ValueError(f"Need at least {self.n_clusters} rows, got {x_raw.shape}")

        self._mean = np.mean(x_raw, axis=0)
        self._std = np.maximum(np.std(x_raw, axis=0), 1.0e-6)
        x = (x_raw - self._mean) / self._std

        rng = np.random.default_rng(self.seed)
        centers = _kmeans_pp_init(x, self.n_clusters, rng)
        labels = np.zeros(x.shape[0], dtype=np.int64)

        for _ in range(self.max_iter):
            dists = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            new_labels = np.argmin(dists, axis=1)
            new_centers = centers.copy()
            for cid in range(self.n_clusters):
                mask = new_labels == cid
                if np.any(mask):
                    new_centers[cid] = np.mean(x[mask], axis=0)
            shift = float(np.max(np.linalg.norm(new_centers - centers, axis=1)))
            centers = new_centers
            labels = new_labels
            if shift < 1.0e-5:
                break

        self._centers = centers

        # PCA 2D projection via SVD on standardized feature matrix
        _, _, vt = np.linalg.svd(x, full_matrices=False)
        self._pca_basis = vt[:2].T
        pca_2d = x @ self._pca_basis

        # Per-cluster regularized inverse covariance for Mahalanobis anomaly distance
        maha = np.zeros(x.shape[0], dtype=np.float64)
        self._prec_mats = []
        reg_eye = 0.05 * np.eye(x.shape[1], dtype=np.float64)
        for cid in range(self.n_clusters):
            mask = labels == cid
            if np.sum(mask) > 2:
                cov = np.cov(x[mask], rowvar=False) + reg_eye
            else:
                cov = np.eye(x.shape[1], dtype=np.float64)
            prec = np.linalg.inv(cov)
            self._prec_mats.append(prec)
            delta = x[mask] - centers[cid]
            if np.any(mask):
                maha[mask] = np.sqrt(np.maximum(0.0, np.sum((delta @ prec) * delta, axis=1)))

        self._cluster_names = self._name_clusters(x_raw, labels)
        summaries: List[Dict[str, Any]] = []
        for cid in range(self.n_clusters):
            mask = labels == cid
            summaries.append(
                {
                    "cluster_id": cid,
                    "name": self._cluster_names[cid],
                    "count": int(np.sum(mask)),
                    "fraction": float(np.mean(mask)),
                    "mean_pwr_log10": float(np.mean(x_raw[mask, 0])) if np.any(mask) else 0.0,
                    "mean_freq_mad_z": float(np.mean(x_raw[mask, 1])) if np.any(mask) else 0.0,
                    "mean_sk": float(np.mean(10.0 ** x_raw[mask, 2])) if np.any(mask) else 0.0,
                    "mean_max_sk": float(np.mean(10.0 ** x_raw[mask, 3])) if np.any(mask) else 0.0,
                    "mean_flagged_frac": float(np.mean(x_raw[mask, 4])) if np.any(mask) else 0.0,
                }
            )

        return RfiClusteringResult(
            labels=labels,
            cluster_names=self._cluster_names,
            mahalanobis_distance=maha,
            features=x_raw,
            pca_2d=pca_2d,
            cluster_summaries=summaries,
            meta=meta or {},
        )

    def _name_clusters(self, x_raw: np.ndarray, labels: np.ndarray) -> Dict[int, str]:
        """Interpret cluster centroids physically without hardcoding dataset labels."""
        names: Dict[int, str] = {}
        stats = []
        for cid in range(self.n_clusters):
            m = labels == cid
            if not np.any(m):
                stats.append((cid, 0.0, 0.0, 1.0, 1.0, 0.0))
                continue
            stats.append(
                (
                    cid,
                    float(np.mean(x_raw[m, 4])),  # flagged_frac
                    float(np.mean(np.abs(x_raw[m, 1]))),  # |freq_mad_z|
                    float(np.mean(10.0 ** x_raw[m, 2])),  # mean_sk
                    float(np.mean(10.0 ** x_raw[m, 3])),  # max_sk
                    float(np.mean(x_raw[m, 5])),  # rel_pwr
                )
            )

        used_counts: Dict[str, int] = {}
        for cid, f_frac, abs_fz, mean_sk, max_sk, rel_pwr in stats:
            if f_frac < 0.15 and abs_fz < 3.0 and max_sk < 2.2:
                base = "clean_thermal"
            elif max_sk >= 2.2 and (abs_fz >= 8.0 or mean_sk > 5.0):
                base = "extreme_burst_dropout" if rel_pwr < -0.2 else "impulsive_burst"
            elif (abs_fz >= 12.0 or (mean_sk < 0.55 and rel_pwr > 0.25)) and max_sk < 5.0:
                base = "cw_or_narrowband"
            elif max_sk >= 2.2:
                base = "impulsive_burst"
            elif f_frac >= 0.45 and 0.60 <= mean_sk <= 1.35:
                base = "narrowband_or_latch"
            else:
                base = "broadband_rfi"

            idx = used_counts.get(base, 0)
            used_counts[base] = idx + 1
            names[cid] = base if idx == 0 else f"{base}_{idx + 1}"
        return names


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

    def cancel(
        self,
        on_ts: np.ndarray,
        off_ts: np.ndarray,
        dt_s: float,
    ) -> CombCancelResult:
        """Detect common-mode narrowband periodic combs from off_ts and subtract from on_ts.

        Only spectral bins with strong narrowband power in off_ts AND non-negative cross-beam
        coherence enter the subtraction template, preserving isolated ON-beam pulsars.
        """
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

        # Common-mode consistency: real part of ON * conj(OFF) must exceed coherence threshold
        cross_re = (on_fft.real * off_fft.real) + (on_fft.imag * off_fft.imag)
        coh = cross_re / np.sqrt(np.maximum(p_on * p_off, 1.0e-24))
        peaks = np.flatnonzero(
            (freqs >= self.min_freq_hz)
            & (snr_off >= self.line_snr_thresh)
            & (coh >= self.min_coherence)
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

        # Rank peak bins by SNR and group into contiguous spectral islands
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

        # Extract contiguous bin islands so each harmonic family gets its own non-negative transfer gain
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

        basis = np.column_stack(templates)  # (T, n_islands)
        gains, _ = nnls(basis, on_d)
        comb_est = basis @ gains
        cleaned_on = on - comb_est

        # Measure narrowband comb RMS inside comb_mask before & after
        cleaned_fft = np.fft.rfft(cleaned_on - median_filter(cleaned_on, size=detrend_w, mode="nearest"))
        p_before = float(np.sum((on_fft.real[comb_mask] ** 2) + (on_fft.imag[comb_mask] ** 2)))
        p_after = float(
            np.sum((cleaned_fft.real[comb_mask] ** 2) + (cleaned_fft.imag[comb_mask] ** 2))
        )
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


def fold_profile(
    ts: np.ndarray,
    dt_s: float,
    period_s: float,
    n_bins: int = 32,
) -> Tuple[np.ndarray, float]:
    """Epoch-fold a detrended or raw 1D time series at period_s and return profile + peak SNR."""
    y = np.asarray(ts, dtype=np.float64)
    y = y - np.median(y)
    phase = np.mod(np.arange(y.size, dtype=np.float64) * dt_s / period_s, 1.0)
    bin_idx = np.minimum(n_bins - 1, (phase * n_bins).astype(np.int64))
    sums = np.bincount(bin_idx, weights=y, minlength=n_bins)
    counts = np.maximum(np.bincount(bin_idx, minlength=n_bins), 1)
    prof = sums / counts
    # Leave-one-third-out robust off-pulse noise estimate around profile median
    med = float(np.median(prof))
    mad = max(1.4826 * float(np.median(np.abs(prof - med))), 1.0e-12)
    snr = float((np.max(prof) - med) / mad)
    return prof, snr


class MatchedFilterTransientScorer:
    """DM-trial + multi-width boxcar matcher with ON-vs-OFF spatial coincidence RFI veto."""

    def __init__(
        self,
        freqs_mhz: Sequence[float] | np.ndarray = FREQS_MHZ,
        dm_trials: Sequence[float] = (0.0, 20.0, 50.0, 100.0, 150.0),
        widths_samples: Sequence[int] = (1, 2, 4, 8),
        snr_threshold: float = 6.0,
        off_veto_ratio: float = 0.45,
    ) -> None:
        self.freqs_mhz = np.asarray(freqs_mhz, dtype=np.float64)
        self.dm_trials = [float(d) for d in dm_trials]
        self.widths_samples = [max(1, int(w)) for w in widths_samples]
        self.snr_threshold = float(snr_threshold)
        self.off_veto_ratio = float(off_veto_ratio)

    def _dedisperse(
        self, dyn_spec: np.ndarray, dt_s: float, dm: float, freq_mask: np.ndarray
    ) -> np.ndarray:
        """Incoherent dedispersion relative to top frequency channel in freq_mask."""
        freqs = self.freqs_mhz[freq_mask]
        spec = dyn_spec[:, freq_mask]
        if dm == 0.0:
            return np.mean(spec, axis=1)

        f_ref = float(np.max(freqs))
        delays_s = K_DM_PC_CM3_MHZ2_S * dm * ((freqs ** -2) - (f_ref ** -2))
        shifts = np.rint(delays_s / dt_s).astype(np.int64)
        n_t = spec.shape[0]
        acc = np.zeros(n_t, dtype=np.float64)
        counts = np.zeros(n_t, dtype=np.float64)
        for ch in range(freqs.size):
            sh = int(shifts[ch])
            if sh == 0:
                acc += spec[:, ch]
                counts += 1.0
            elif 0 < sh < n_t:
                acc[: n_t - sh] += spec[sh:, ch]
                counts[: n_t - sh] += 1.0
            elif -n_t < sh < 0:
                acc[-sh:] += spec[: n_t + sh, ch]
                counts[-sh:] += 1.0
        return acc / np.maximum(counts, 1.0)

    def _best_boxcar_snr(self, ts: np.ndarray) -> Tuple[np.ndarray, int, int, float]:
        """Compute MAD-normalized SNR trace across configured boxcar widths."""
        med = float(np.median(ts))
        centered = ts - med
        # Baseline single-sample noise sigma from robust MAD
        best_snr_trace = np.zeros_like(centered)
        best_w = self.widths_samples[0]
        best_idx = 0
        best_snr = -np.inf

        for w in self.widths_samples:
            smoothed = uniform_filter1d(centered, size=w, mode="nearest")
            sigma_w = max(1.4826 * float(np.median(np.abs(smoothed - np.median(smoothed)))), 1.0e-12)
            snr_w = smoothed / sigma_w
            pk_idx = int(np.argmax(snr_w))
            pk_val = float(snr_w[pk_idx])
            if pk_val > best_snr:
                best_snr = pk_val
                best_idx = pk_idx
                best_w = w
                best_snr_trace = snr_w
        return best_snr_trace, best_idx, best_w, best_snr

    def score(
        self,
        on_dyn_spec: np.ndarray,
        off_dyn_spec: np.ndarray,
        dt_s: float,
        freq_mask: Optional[np.ndarray] = None,
    ) -> Optional[TransientCandidate]:
        """Search DM and pulse-width grid on ON spectrum while vetoing common-mode OFF triggers."""
        on = np.asarray(on_dyn_spec, dtype=np.float64)
        off = np.asarray(off_dyn_spec, dtype=np.float64)
        if on.shape != off.shape or on.ndim != 2:
            raise ValueError(f"Expected matching (T, F) arrays, got {on.shape} and {off.shape}")

        f_mask = (
            np.all(np.isfinite(on), axis=0) & np.all(np.isfinite(off), axis=0)
            if freq_mask is None
            else np.asarray(freq_mask, dtype=bool)
        )
        # Channel median subtraction removes bandpass baseline
        on_c = on - np.median(on, axis=0, keepdims=True)
        off_c = off - np.median(off, axis=0, keepdims=True)

        off_dm0_trace, _, _, _ = self._best_boxcar_snr(
            self._dedisperse(off_c, dt_s, 0.0, f_mask)
        )

        best: Optional[TransientCandidate] = None
        for dm in self.dm_trials:
            ts_on = self._dedisperse(on_c, dt_s, dm, f_mask)
            ts_off = self._dedisperse(off_c, dt_s, dm, f_mask)
            _, idx, width, snr_on = self._best_boxcar_snr(ts_on)
            off_pk_trace, _, _, _ = self._best_boxcar_snr(ts_off)

            w_lo = max(0, idx - width)
            w_hi = min(on.shape[0], idx + width + 1)
            snr_off_dm = float(np.max(off_pk_trace[w_lo:w_hi]))
            snr_off_zero = float(np.max(off_dm0_trace[w_lo:w_hi]))
            vetoed = max(snr_off_dm, snr_off_zero) >= (self.off_veto_ratio * max(snr_on, 1.0e-9))

            cand = TransientCandidate(
                sample_idx=idx,
                dm_pc_cm3=dm,
                width_samples=width,
                snr_on=snr_on,
                snr_off_at_peak=snr_off_dm,
                snr_off_dm0=snr_off_zero,
                rfi_vetoed=vetoed,
            )
            # Rank by dominant event SNR so strong zero-DM RFI cannot hide behind smeared halos
            if best is None or cand.snr_on > best.snr_on:
                best = cand
        return best
