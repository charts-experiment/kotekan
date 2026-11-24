#!/usr/bin/env python3
"""CHARTS Fast Radio Burst (FRB) & Transient Event Detector.

Provides:
1. De-dispersion Search: Scans candidate DM trials (DM in [0, 2000] pc/cm^3) using cold-plasma dispersion delay.
2. Boxcar Matched Filtering: Multi-scale temporal width convolution (1..64 samples).
3. SNR Peak Candidate Flagging: Detects events exceeding detection threshold (e.g. SNR >= 7.0).
4. RFI Discrimination: Distinguishes zero-DM terrestrial interference vs. true astrophysical sweeps.
5. Injected & Real Baseband Validation: Evaluates detection on site baseband snapshots with synthetic/real burst injection.
6. Event Candidate Visualization: Generates 4-panel diagnostic plot (Dispersed Waterfall, Dedispersed Waterfall, S(t) Pulse Profile, and DM-Time "Bowtie" SNR Heatmap).

Usage:
    # Scan baseband dataset for events
    python detect_frb_events.py --h5-path /path/to/baseband_virtual.h5

    # Inject a test FRB (DM=350 pc/cm^3, SNR=15) into real baseband and detect it
    python detect_frb_events.py --inject --dm 350.0 --snr 18.0 --out-plot frb_event_candidate.png
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Setup paths
_script_dir = Path(__file__).resolve().parent
_kotekan_root = _script_dir.parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from constants import (
    C_LIGHT,
    CHARTS_CHANNEL_WIDTH_MHZ,
    CHARTS_LATITUDE_DEG,
    CHARTS_LONGITUDE_DEG,
    DEFAULT_FREQUENCY_START_MHZ,
    DEFAULT_SPACING_M,
    FPGA_TIME_RESOLUTION_US,
    K_DM,
    get_default_charts_h5_path,
)
from sim.writer import unpack_int4x2

# Canonical alias
unpack_4bit_complex = unpack_int4x2


def compute_dispersion_delay_samples(
    freqs_mhz: np.ndarray, dm: float, sample_time_s: float, f_ref_mhz: float,
) -> np.ndarray:
    """Computes dispersion delay in integer/float time samples relative to f_ref."""
    if dm == 0.0:
        return np.zeros_like(freqs_mhz, dtype=np.int64)
    delays_s = K_DM * dm * ((freqs_mhz ** -2.0) - (f_ref_mhz ** -2.0))
    return np.round(delays_s / sample_time_s).astype(np.int64)


def dedisperse_waterfall(
    waterfall: np.ndarray, freqs_mhz: np.ndarray, dm: float, sample_time_s: float,
) -> np.ndarray:
    """De-disperses a 2D (n_freq, n_time) dynamic spectrum waterfall by shifting frequency channels."""
    n_freq, n_time = waterfall.shape
    f_ref_mhz = np.max(freqs_mhz)
    delay_samples = compute_dispersion_delay_samples(
        freqs_mhz, dm, sample_time_s, f_ref_mhz
    )

    dedisp = np.zeros_like(waterfall)
    for f in range(n_freq):
        shift = delay_samples[f]
        if 0 <= shift < n_time:
            dedisp[f, : n_time - shift] = waterfall[f, shift:]
        elif shift < 0:
            pos_shift = -shift
            if pos_shift < n_time:
                dedisp[f, pos_shift:] = waterfall[f, : n_time - pos_shift]
        else:
            dedisp[f, :] = np.median(waterfall[f, :])
    return dedisp


def search_transient_events(
    waterfall: np.ndarray,
    freqs_mhz: np.ndarray,
    sample_time_s: float,
    dm_min: float = 0.0,
    dm_max: float = 1200.0,
    dm_step: float = 10.0,
    boxcar_widths: Tuple[int, ...] = (1, 2, 4, 8, 16, 32),
    threshold_snr: float = 7.0,
) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    """Runs a dedispersion search over DM trials and boxcar filters.

    Returns:
        candidates: List of detected event metadata
        dm_trials: Array of tested DMs
        time_series_best: Time series at optimal DM
        dm_time_snr_map: 2D SNR heatmap (n_dm, n_time)
    """
    n_freq, n_time = waterfall.shape
    dm_trials = np.arange(dm_min, dm_max + dm_step, dm_step, dtype=np.float32)
    dm_time_snr_map = np.zeros((len(dm_trials), n_time), dtype=np.float32)

    candidates = []
    best_overall_snr = 0.0
    best_dm = 0.0
    best_dedisp_ts = np.zeros(n_time, dtype=np.float32)

    # Baseline statistics per frequency
    f_medians = np.median(waterfall, axis=1, keepdims=True)
    f_stds = np.std(waterfall, axis=1, keepdims=True)
    f_stds[f_stds <= 0] = 1.0
    norm_wf = (waterfall - f_medians) / f_stds

    for i, dm in enumerate(dm_trials):
        dedisp_wf = dedisperse_waterfall(norm_wf, freqs_mhz, dm, sample_time_s)
        # Sum over frequency to form frequency-integrated time series S(t)
        ts = np.sum(dedisp_wf, axis=0)  # (n_time,)

        # Compute robust baseline mean and standard deviation (sigma-clipping)
        med = np.median(ts)
        mad = np.median(np.abs(ts - med))
        std = 1.4826 * mad if mad > 0 else (np.std(ts) or 1.0)

        # Multi-scale boxcar convolution
        for w in boxcar_widths:
            if w == 1:
                smoothed_ts = (ts - med) / std
            else:
                kernel = np.ones(w, dtype=np.float32) / math.sqrt(w)
                smoothed_ts = (np.convolve(ts - med, kernel, mode="same")) / std

            peak_idx = int(np.argmax(smoothed_ts))
            peak_snr = float(smoothed_ts[peak_idx])

            if peak_snr > dm_time_snr_map[i, peak_idx]:
                dm_time_snr_map[i, peak_idx] = peak_snr

            if peak_snr >= threshold_snr:
                candidates.append(
                    {
                        "dm": float(dm),
                        "time_sample": peak_idx,
                        "time_ms": float(peak_idx * sample_time_s * 1000.0),
                        "boxcar_width_samples": int(w),
                        "snr": peak_snr,
                    }
                )

            if peak_snr > best_overall_snr:
                best_overall_snr = peak_snr
                best_dm = float(dm)
                best_dedisp_ts = smoothed_ts

    # Cluster/deduplicate candidate detections via DBSCAN in (DM, t, beam)
    if len(candidates) >= 2:
        clustered = cluster_candidates_dbscan(
            candidates, sigma_t_s=0.01, sigma_dm=15.0, eps=1.5, min_samples=1
        )
        return clustered, dm_trials, best_dedisp_ts, dm_time_snr_map

    candidates.sort(key=lambda c: c["snr"], reverse=True)
    return candidates, dm_trials, best_dedisp_ts, dm_time_snr_map


def two_stage_iqr_normalize(
    waterfall: np.ndarray, short_win: int = 64, long_win: int = 512
) -> np.ndarray:
    """Two-stage running IQR/median normalization (BLINK/STRIDE & Northern Cross).

    Expects shape (n_freq, n_time). Removes slow long-timescale baseline drifts across
    `long_win` samples and normalizes locally via robust IQR-scaled noise `sigma = IQR / 1.349`.
    """
    wf = np.asarray(waterfall, dtype=np.float64)
    if wf.ndim != 2 or wf.shape[1] == 0:
        raise ValueError(f"Expected 2D (n_freq, n_time) array, got {wf.shape}")

    n_freq, n_time = wf.shape
    l_win = max(8, min(int(long_win), n_time))
    s_win = max(4, min(int(short_win), n_time))

    # Stage 1: coarse block medians across long_win interpolated smoothly across time
    n_long = max(1, (n_time + l_win - 1) // l_win)
    long_centers = np.zeros(n_long, dtype=np.float64)
    long_meds = np.zeros((n_freq, n_long), dtype=np.float64)
    for b in range(n_long):
        t0 = b * l_win
        t1 = min(n_time, t0 + l_win)
        long_centers[b] = 0.5 * (t0 + t1 - 1)
        long_meds[:, b] = np.median(wf[:, t0:t1], axis=1)

    t_axis = np.arange(n_time, dtype=np.float64)
    baseline = np.empty_like(wf)
    if n_long == 1:
        baseline[:] = long_meds[:, :1]
    else:
        for f in range(n_freq):
            baseline[f] = np.interp(t_axis, long_centers, long_meds[f])
    detrended = wf - baseline

    # Stage 2: short-window local median subtraction & IQR robust scaling
    n_short = max(1, (n_time + s_win - 1) // s_win)
    short_centers = np.zeros(n_short, dtype=np.float64)
    short_meds = np.zeros((n_freq, n_short), dtype=np.float64)
    short_sigs = np.zeros((n_freq, n_short), dtype=np.float64)
    for b in range(n_short):
        t0 = b * s_win
        t1 = min(n_time, t0 + s_win)
        short_centers[b] = 0.5 * (t0 + t1 - 1)
        blk = detrended[:, t0:t1]
        short_meds[:, b] = np.median(blk, axis=1)
        q25, q75 = np.percentile(blk, [25.0, 75.0], axis=1)
        iqr_sig = (q75 - q25) / 1.3489795
        chan_std = np.std(blk, axis=1)
        short_sigs[:, b] = np.where(iqr_sig > 1.0e-9, iqr_sig, np.maximum(chan_std, 1.0e-9))

    loc_med = np.empty_like(wf)
    loc_sig = np.empty_like(wf)
    if n_short == 1:
        loc_med[:] = short_meds[:, :1]
        loc_sig[:] = short_sigs[:, :1]
    else:
        for f in range(n_freq):
            loc_med[f] = np.interp(t_axis, short_centers, short_meds[f])
            loc_sig[f] = np.interp(t_axis, short_centers, short_sigs[f])

    return (detrended - loc_med) / np.maximum(loc_sig, 1.0e-9)


def compute_sidelobe_confidence(
    snr: float, beam_powers: np.ndarray, peak_beam: Optional[int] = None
) -> Tuple[float, float, int]:
    """Compute SPOTLIGHT sidelobe discrimination index C = SNR * (1 - P_sidelobe / P_peak)."""
    pw = np.asarray(beam_powers, dtype=np.float64)
    if pw.ndim != 1 or pw.size == 0:
        raise ValueError("beam_powers must be a non-empty 1D array")
    b_peak = int(np.argmax(pw)) if peak_beam is None else int(peak_beam)
    p_peak = float(pw[b_peak])
    if pw.size == 1 or p_peak <= 0.0:
        return float(snr), 0.0, b_peak
    p_side = float(np.max(np.delete(pw, b_peak)))
    ratio = min(1.0, max(0.0, p_side / p_peak))
    conf = float(snr) * max(0.0, 1.0 - ratio)
    return conf, ratio, b_peak


def cluster_candidates_dbscan(
    candidates: List[Dict[str, Any]],
    sigma_t_s: float = 0.01,
    sigma_dm: float = 15.0,
    eps: float = 1.5,
    min_samples: int = 2,
) -> List[Dict[str, Any]]:
    """Pure-NumPy DBSCAN in normalized SPOTLIGHT (dt/sigma_t, dDM/sigma_dm, beam) space.

    Points belong to different clusters if they originate from distinct `beam_id` (when set
    and disagreeing) or if `D = sqrt((dt/sigma_t)^2 + (dDM/sigma_dm)^2) > eps`.
    Returns representative candidates (one per cluster plus noise outliers when min_samples<=1),
    each annotated with `cluster_id`, `cluster_size`, and ranked by `sidelobe_confidence` / `snr`.
    """
    n = len(candidates)
    if n == 0:
        return []

    t_s = np.array(
        [float(c.get("time_s", float(c.get("time_ms", 0.0)) * 1.0e-3)) for c in candidates],
        dtype=np.float64,
    )
    dm = np.array([float(c.get("dm", c.get("dm_pc_cm3", 0.0))) for c in candidates], dtype=np.float64)
    beams = np.array([int(c.get("beam_id", 0)) for c in candidates], dtype=np.int64)

    sig_t = max(float(sigma_t_s), 1.0e-6)
    sig_dm = max(float(sigma_dm), 1.0e-3)
    dt_mat = (t_s[:, None] - t_s[None, :]) / sig_t
    ddm_mat = (dm[:, None] - dm[None, :]) / sig_dm
    d_mat = np.sqrt(dt_mat * dt_mat + ddm_mat * ddm_mat)
    same_beam = beams[:, None] == beams[None, :]
    adj = (d_mat <= eps) & same_beam

    UNVISITED = -2
    NOISE = -1
    labels = np.full(n, UNVISITED, dtype=np.int64)
    cluster_id = 0

    for i in range(n):
        if labels[i] != UNVISITED:
            continue
        nbrs = list(np.flatnonzero(adj[i]))
        if len(nbrs) < min_samples:
            labels[i] = NOISE
            continue
        labels[i] = cluster_id
        queue = list(nbrs)
        idx_q = 0
        while idx_q < len(queue):
            q = queue[idx_q]
            idx_q += 1
            if labels[q] == NOISE:
                labels[q] = cluster_id
            if labels[q] != UNVISITED:
                continue
            labels[q] = cluster_id
            q_nbrs = np.flatnonzero(adj[q])
            if len(q_nbrs) >= min_samples:
                for nb in q_nbrs:
                    if labels[nb] in (UNVISITED, NOISE):
                        queue.append(int(nb))
        cluster_id += 1

    representatives: List[Dict[str, Any]] = []
    for cid in range(cluster_id):
        members_idx = np.flatnonzero(labels == cid)
        members = [candidates[int(m)] for m in members_idx]
        best = max(
            members,
            key=lambda c: (
                float(c.get("sidelobe_confidence", c.get("snr", 0.0))),
                float(c.get("snr", 0.0)),
            ),
        )
        rep = dict(best)
        rep["cluster_id"] = int(cid)
        rep["cluster_size"] = int(len(members))
        representatives.append(rep)

    representatives.sort(
        key=lambda c: (
            float(c.get("sidelobe_confidence", c.get("snr", 0.0))),
            float(c.get("snr", 0.0)),
        ),
        reverse=True,
    )
    return representatives


class MatchedFilterMultiBeamScorer:
    """Multi-beam dedispersion & multi-width boxcar scorer with sidelobe confidence C veto."""

    def __init__(
        self,
        freqs_mhz: np.ndarray,
        dm_trials: Tuple[float, ...] = (0.0, 20.0, 50.0, 100.0, 200.0, 350.0),
        widths_samples: Tuple[int, ...] = (1, 2, 4, 8, 16),
        snr_threshold: float = 6.0,
        max_sidelobe_ratio: float = 0.70,
        min_sidelobe_confidence: float = 3.0,
    ) -> None:
        self.freqs_mhz = np.asarray(freqs_mhz, dtype=np.float64)
        self.dm_trials = [float(d) for d in dm_trials]
        self.widths_samples = [max(1, int(w)) for w in widths_samples]
        self.snr_threshold = float(snr_threshold)
        self.max_sidelobe_ratio = float(max_sidelobe_ratio)
        self.min_sidelobe_confidence = float(min_sidelobe_confidence)

    @staticmethod
    def _best_boxcar(ts: np.ndarray, widths: List[int]) -> Tuple[int, int, float, np.ndarray]:
        med = float(np.median(ts))
        centered = ts - med
        best_snr = -np.inf
        best_idx = 0
        best_w = widths[0]
        best_trace = np.zeros_like(centered)
        for w in widths:
            if w == 1:
                sm = centered
            else:
                k = np.ones(w, dtype=np.float64) / math.sqrt(w)
                sm = np.convolve(centered, k, mode="same")
            sig = max(1.4826 * float(np.median(np.abs(sm - np.median(sm)))), 1.0e-9)
            snr_t = sm / sig
            idx = int(np.argmax(snr_t))
            val = float(snr_t[idx])
            if val > best_snr:
                best_snr = val
                best_idx = idx
                best_w = w
                best_trace = snr_t
        return best_idx, best_w, best_snr, best_trace

    def score_multibeam(
        self, dyn_specs: np.ndarray, dt_s: float
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Score dynamic spectra of shape (n_beams, n_freq, n_time).

        Returns (accepted_candidates, all_candidates).
        """
        arr = np.asarray(dyn_specs, dtype=np.float64)
        if arr.ndim != 3:
            raise ValueError(f"Expected (n_beams, n_freq, n_time), got {arr.shape}")
        n_beams, _, n_time = arr.shape
        norm_specs = np.stack(
            [two_stage_iqr_normalize(arr[b]) for b in range(n_beams)], axis=0
        )

        all_cands: List[Dict[str, Any]] = []
        for dm in self.dm_trials:
            ts_beams = np.stack(
                [dedisperse_waterfall(norm_specs[b], self.freqs_mhz, dm, dt_s).sum(axis=0) for b in range(n_beams)],
                axis=0,
            )
            beam_traces = []
            peak_b = 0
            best_idx = 0
            best_w = self.widths_samples[0]
            best_snr = -np.inf
            for b in range(n_beams):
                idx, w, snr, trace = self._best_boxcar(ts_beams[b], self.widths_samples)
                beam_traces.append(trace)
                if snr > best_snr:
                    best_snr = snr
                    peak_b = b
                    best_idx = idx
                    best_w = w

            if best_snr < self.snr_threshold:
                continue

            lo = max(0, best_idx - best_w)
            hi = min(n_time, best_idx + best_w + 1)
            beam_pw = np.array([max(0.0, float(np.max(beam_traces[b][lo:hi]))) for b in range(n_beams)])
            conf, sl_ratio, _ = compute_sidelobe_confidence(best_snr, beam_pw, peak_beam=peak_b)
            vetoed = (n_beams >= 2) and (
                sl_ratio > self.max_sidelobe_ratio or conf < self.min_sidelobe_confidence
            )
            all_cands.append(
                {
                    "dm": float(dm),
                    "time_sample": best_idx,
                    "time_ms": float(best_idx * dt_s * 1000.0),
                    "time_s": float(best_idx * dt_s),
                    "boxcar_width_samples": best_w,
                    "snr": best_snr,
                    "beam_id": peak_b,
                    "sidelobe_ratio": sl_ratio,
                    "sidelobe_confidence": conf,
                    "rfi_vetoed": bool(vetoed),
                }
            )

        accepted = [c for c in all_cands if not c["rfi_vetoed"]]
        accepted.sort(key=lambda c: c["sidelobe_confidence"], reverse=True)
        all_cands.sort(key=lambda c: c["snr"], reverse=True)
        return accepted, all_cands


def inject_synthetic_frb(
    baseband_voltages: np.ndarray,
    freqs_mhz: np.ndarray,
    dm: float = 350.0,
    pulse_width_ms: float = 2.0,
    target_snr: float = 15.0,
    sample_time_s: float = FPGA_TIME_RESOLUTION_US * 1e-6,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Injects a physical cold-plasma dispersed FRB burst into raw baseband complex voltages."""
    n_ant, n_freq, n_time = baseband_voltages.shape
    f_ref_mhz = np.max(freqs_mhz)
    delays_samples = compute_dispersion_delay_samples(
        freqs_mhz, dm, sample_time_s, f_ref_mhz
    )

    pulse_width_samples = max(1, int(round(pulse_width_ms * 1e-3 / sample_time_s)))
    t_center_samples = n_time // 4  # place burst near first quarter of stream

    # Noise baseline standard deviation
    noise_sigma = float(np.std(baseband_voltages[:8, :, :]))
    amplitude = (target_snr / math.sqrt(n_freq * 8)) * noise_sigma

    injected = baseband_voltages.copy()
    time_indices = np.arange(n_time)

    for f in range(n_freq):
        t0 = t_center_samples + delays_samples[f]
        if t0 < n_time:
            # Gaussian pulse profile
            pulse_env = amplitude * np.exp(
                -0.5 * ((time_indices - t0) / (pulse_width_samples / 2.355)) ** 2
            )
            for a in range(min(8, n_ant)):
                injected[a, f, :] += pulse_env.astype(np.complex64)

    meta = {
        "true_dm": dm,
        "true_center_sample": t_center_samples,
        "true_center_time_ms": t_center_samples * sample_time_s * 1000.0,
        "true_width_ms": pulse_width_ms,
        "true_snr": target_snr,
    }
    return injected, meta


def plot_frb_candidate_diagnostic(
    raw_waterfall: np.ndarray,
    dedisp_waterfall: np.ndarray,
    time_series_snr: np.ndarray,
    dm_time_snr_map: np.ndarray,
    dm_trials: np.ndarray,
    freqs_mhz: np.ndarray,
    best_cand: Dict[str, Any],
    out_plot_path: Path,
    sample_time_s: float,
) -> None:
    """Generates publication-quality 4-panel FRB candidate diagnostic figure."""
    n_freq, n_time = raw_waterfall.shape
    time_ms = np.arange(n_time) * sample_time_s * 1000.0

    fig, axes = plt.subplots(2, 2, figsize=(16, 11), facecolor="#0E1117")
    fig.suptitle(
        f"CHARTS Fast Radio Burst (FRB) Transient Detection Diagnostic\n"
        f"Detected Candidate: Peak SNR = {best_cand.get('snr', 0.0):.1f}σ | Optimal DM = {best_cand.get('dm', 0.0):.1f} pc/cm³ | Arrival = {best_cand.get('time_ms', 0.0):.2f} ms",
        color="white",
        fontsize=14,
        fontweight="bold",
        y=0.97,
    )

    # Panel 1: Dispersed Raw Dynamic Spectrum Waterfall
    ax1 = axes[0, 0]
    ax1.set_facecolor("#161B22")
    ax1.set_title(
        "1. Dispersed Dynamic Spectrum Waterfall (Raw Ingest)",
        color="white",
        fontsize=11,
        fontweight="bold",
    )
    raw_med = np.maximum(1e-6, np.median(raw_waterfall, axis=1, keepdims=True))
    wf_norm = raw_waterfall / raw_med
    im1 = ax1.imshow(
        10.0 * np.log10(np.maximum(wf_norm, 1e-4)),
        aspect="auto",
        origin="lower",
        extent=[time_ms[0], time_ms[-1], freqs_mhz[0], freqs_mhz[-1]],
        cmap="inferno",
        vmin=-2.0,
        vmax=8.0,
    )
    # Overlay theoretical dispersion sweep curve
    f_ref = np.max(freqs_mhz)
    t_sweep_ms = (
        best_cand.get("time_ms", 0.0)
        + (K_DM * best_cand.get("dm", 0.0) * ((freqs_mhz ** -2.0) - (f_ref ** -2.0)))
        * 1000.0
    )
    ax1.plot(
        t_sweep_ms,
        freqs_mhz,
        "--",
        color="#00FFCC",
        linewidth=1.8,
        label=f"Dispersion Sweep (DM={best_cand.get('dm', 0):.0f})",
    )
    ax1.set_xlabel("Time (ms)", color="#C9D1D9")
    ax1.set_ylabel("Frequency (MHz)", color="#C9D1D9")
    ax1.tick_params(colors="#8B949E")
    for sp in ax1.spines.values():
        sp.set_color("#30363D")
    ax1.legend(
        loc="upper right",
        facecolor="#161B22",
        edgecolor="#30363D",
        labelcolor="#C9D1D9",
        fontsize=8,
    )
    cb1 = fig.colorbar(im1, ax=ax1, pad=0.02)
    cb1.set_label("Relative Power (dB)", color="#C9D1D9")
    cb1.ax.tick_params(colors="#8B949E")

    # Panel 2: De-dispersed Dynamic Spectrum Waterfall (Aligned in Time)
    ax2 = axes[0, 1]
    ax2.set_facecolor("#161B22")
    ax2.set_title(
        f"2. De-dispersed Waterfall at Optimal DM = {best_cand.get('dm', 0.0):.1f} pc/cm³",
        color="white",
        fontsize=11,
        fontweight="bold",
    )
    dedisp_med = np.maximum(1e-6, np.median(dedisp_waterfall, axis=1, keepdims=True))
    dedisp_norm = dedisp_waterfall / dedisp_med
    im2 = ax2.imshow(
        10.0 * np.log10(np.maximum(dedisp_norm, 1e-4)),
        aspect="auto",
        origin="lower",
        extent=[time_ms[0], time_ms[-1], freqs_mhz[0], freqs_mhz[-1]],
        cmap="inferno",
        vmin=-2.0,
        vmax=8.0,
    )
    ax2.axvline(
        best_cand.get("time_ms", 0.0),
        color="#00FFCC",
        linestyle=":",
        linewidth=1.5,
        label="Aligned Pulse Time",
    )
    ax2.set_xlabel("Time (ms)", color="#C9D1D9")
    ax2.set_ylabel("Frequency (MHz)", color="#C9D1D9")
    ax2.tick_params(colors="#8B949E")
    for sp in ax2.spines.values():
        sp.set_color("#30363D")
    ax2.legend(
        loc="upper right",
        facecolor="#161B22",
        edgecolor="#30363D",
        labelcolor="#C9D1D9",
        fontsize=8,
    )
    cb2 = fig.colorbar(im2, ax=ax2, pad=0.02)
    cb2.set_label("Relative Power (dB)", color="#C9D1D9")
    cb2.ax.tick_params(colors="#8B949E")

    # Panel 3: Frequency-Integrated Time Series S(t) Pulse Profile
    ax3 = axes[1, 0]
    ax3.set_facecolor("#161B22")
    ax3.set_title(
        f"3. De-dispersed Frequency-Integrated Time Series S(t)",
        color="white",
        fontsize=11,
        fontweight="bold",
    )
    ax3.plot(
        time_ms,
        time_series_snr,
        color="#00FFCC",
        linewidth=1.5,
        label="De-dispersed S(t)",
    )
    ax3.axhline(
        7.0,
        color="#FF5252",
        linestyle="--",
        linewidth=1.2,
        label="Detection Threshold (7.0σ)",
    )
    ax3.scatter(
        [best_cand.get("time_ms", 0.0)],
        [best_cand.get("snr", 0.0)],
        color="#FFEE58",
        marker="*",
        s=160,
        zorder=5,
        label=f"Peak: {best_cand.get('snr', 0):.1f}σ",
    )
    ax3.set_xlabel("Time (ms)", color="#C9D1D9")
    ax3.set_ylabel("Detection S/N Ratio (σ)", color="#C9D1D9")
    ax3.tick_params(colors="#8B949E")
    for sp in ax3.spines.values():
        sp.set_color("#30363D")
    ax3.legend(
        loc="upper right",
        facecolor="#161B22",
        edgecolor="#30363D",
        labelcolor="#C9D1D9",
        fontsize=8,
    )
    ax3.grid(True, color="#21262D", linestyle="--", alpha=0.6)

    # Panel 4: DM vs. Time SNR Heatmap ("Bowtie" Signature)
    ax4 = axes[1, 1]
    ax4.set_facecolor("#161B22")
    ax4.set_title(
        "4. DM vs. Time SNR Matrix (Astrophysical 'Bowtie' Signature)",
        color="white",
        fontsize=11,
        fontweight="bold",
    )
    im4 = ax4.imshow(
        dm_time_snr_map,
        aspect="auto",
        origin="lower",
        extent=[time_ms[0], time_ms[-1], dm_trials[0], dm_trials[-1]],
        cmap="magma",
        vmin=0.0,
        vmax=max(8.0, best_cand.get("snr", 10.0)),
    )
    ax4.scatter(
        [best_cand.get("time_ms", 0.0)],
        [best_cand.get("dm", 0.0)],
        marker="x",
        color="#00FFCC",
        s=100,
        linewidth=2,
        label=f"Best Candidate (DM={best_cand.get('dm', 0):.0f})",
    )
    ax4.set_xlabel("Time (ms)", color="#C9D1D9")
    ax4.set_ylabel("Dispersion Measure (pc/cm³)", color="#C9D1D9")
    ax4.tick_params(colors="#8B949E")
    for sp in ax4.spines.values():
        sp.set_color("#30363D")
    ax4.legend(
        loc="upper right",
        facecolor="#161B22",
        edgecolor="#30363D",
        labelcolor="#C9D1D9",
        fontsize=8,
    )
    cb4 = fig.colorbar(im4, ax=ax4, pad=0.02)
    cb4.set_label("Detection S/N (σ)", color="#C9D1D9")
    cb4.ax.tick_params(colors="#8B949E")

    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.94])
    out_plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_plot_path, dpi=180, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)


def run_frb_detection(
    h5_path: Path,
    inject: bool = False,
    inject_dm: float = 350.0,
    inject_snr: float = 18.0,
    dm_min: float = 0.0,
    dm_max: float = 1000.0,
    dm_step: float = 10.0,
    threshold_snr: float = 7.0,
    out_plot: Optional[Path] = None,
    voevent_xml: Optional[Path] = None,
    voevent_json: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Runs complete FRB detection pipeline on real or injected baseband data."""
    print("=" * 80)
    print(" CHARTS FAST RADIO BURST (FRB) & TRANSIENT EVENT DETECTION ENGINE")
    print("=" * 80)

    # 1. Ingest Baseband Data
    if not h5_path.exists():
        print(f"[ERROR] Baseband dataset not found at {h5_path}")
        return []

    print(f"[1/4] Ingesting Site Baseband Data Snapshot: {h5_path}...")
    with h5py.File(h5_path, "r") as f:
        dset = f["baseband"]
        # Ingest 8 active antennas, 84 frequency channels, 15,360 time samples (~51.2 ms)
        n_time_slice = min(15360, dset.shape[2])
        raw_packed = dset[:8, :84, :n_time_slice]
        cdata = unpack_4bit_complex(raw_packed)

    n_ant, n_freq, n_time = cdata.shape
    sample_time_s = FPGA_TIME_RESOLUTION_US * 1e-6
    total_time_ms = n_time * sample_time_s * 1000.0
    freqs_mhz = (
        DEFAULT_FREQUENCY_START_MHZ + np.arange(n_freq) * CHARTS_CHANNEL_WIDTH_MHZ
    )

    print(
        f"  -> Array Shape: {n_ant} Antennas x {n_freq} Channels x {n_time} Samples ({total_time_ms:.2f} ms sky time)"
    )
    print(
        f"  -> Frequency Band: {freqs_mhz[0]:.2f} MHz -> {freqs_mhz[-1]:.2f} MHz (Δf = {CHARTS_CHANNEL_WIDTH_MHZ:.3f} MHz)"
    )

    # 2. Injection Step (if enabled)
    if inject:
        print(
            f"\n[2/4] Injecting Synthetic FRB (DM={inject_dm:.1f} pc/cm³, Target SNR={inject_snr:.1f}σ)..."
        )
        cdata, inject_meta = inject_synthetic_frb(
            cdata,
            freqs_mhz,
            dm=inject_dm,
            target_snr=inject_snr,
            sample_time_s=sample_time_s,
        )
        print(
            f"  -> Injected arrival time: {inject_meta['true_center_time_ms']:.2f} ms"
        )
    else:
        print(
            f"\n[2/4] Searching for native natural transient events (No Injection)..."
        )

    # 3. Form Array Coherent Power Waterfall
    # Beamform to zenith (l0=0, m0=0) or sum active dipoles
    formed_voltages = np.sum(cdata, axis=0)  # (n_freq, n_time)
    raw_waterfall = np.abs(formed_voltages) ** 2  # (n_freq, n_time)

    # 4. Dedispersion & Matched Filtering Search
    print(
        f"\n[3/4] Running Dedispersion Search (DM: {dm_min:.0f} -> {dm_max:.0f} pc/cm³, step={dm_step:.0f})..."
    )
    candidates, dm_trials, best_ts, dm_snr_map = search_transient_events(
        raw_waterfall,
        freqs_mhz,
        sample_time_s,
        dm_min=dm_min,
        dm_max=dm_max,
        dm_step=dm_step,
        threshold_snr=threshold_snr,
    )

    # 5. Report Candidate Detections
    print(f"\n[4/4] Detection Search Results:")
    if candidates:
        print(
            f"  *** FOUND {len(candidates)} EVENT CANDIDATES (SNR >= {threshold_snr:.1f}σ) ***"
        )
        for i, c in enumerate(candidates):
            print(
                f"  [{i+1}] SNR = {c['snr']:.2f}σ | DM = {c['dm']:.1f} pc/cm³ | Time = {c['time_ms']:.2f} ms | Boxcar = {c['boxcar_width_samples']} smp"
            )
    else:
        print(
            f"  -> No transient events detected above threshold ({threshold_snr:.1f}σ). Baseline noise is clean."
        )

    # 6. Generate Candidate Diagnostic Plot & Optional VOEvent Alert
    if candidates and (voevent_xml or voevent_json):
        if str(_kotekan_root) not in sys.path:
            sys.path.insert(0, str(_kotekan_root))
        from charts.frb_voevent import emit_frb_voevent_alert

        top = dict(candidates[0])
        top.setdefault("freq_mhz", float(np.mean(freqs_mhz)))
        voe_res = emit_frb_voevent_alert(top, xml_path=voevent_xml, json_path=voevent_json)
        print(
            f"  -> Emitted frb-voe alert in {voe_res['latency_ms']:.2f} ms "
            f"(DM_gal={voe_res['payload']['dm_gal_ne2025']:.1f}, "
            f"DM_ex={voe_res['payload']['dm_excess']:.1f})"
        )

    if out_plot:
        best_cand = (
            candidates[0]
            if candidates
            else {"dm": 0.0, "time_ms": 0.0, "snr": float(np.max(best_ts))}
        )
        dedisp_wf = dedisperse_waterfall(
            raw_waterfall, freqs_mhz, best_cand.get("dm", 0.0), sample_time_s
        )
        plot_frb_candidate_diagnostic(
            raw_waterfall,
            dedisp_wf,
            best_ts,
            dm_snr_map,
            dm_trials,
            freqs_mhz,
            best_cand,
            out_plot,
            sample_time_s,
        )
        print(f"\n  -> Generated Event Candidate Diagnostic Plot: {out_plot}")

    print("=" * 80 + "\n")
    return candidates


def main():
    parser = argparse.ArgumentParser(
        description="CHARTS FRB & Transient Event Detector"
    )
    default_h5 = get_default_charts_h5_path()
    default_plot = _kotekan_root / "test_charts" / "charts_frb_event_candidate.png"

    parser.add_argument(
        "--h5-path",
        type=Path,
        default=default_h5,
        help="Path to baseband virtual HDF5 dataset",
    )
    parser.add_argument(
        "--inject",
        action="store_true",
        help="Inject a synthetic FRB into real baseband data for validation",
    )
    parser.add_argument(
        "--dm",
        type=float,
        default=350.0,
        help="Injection DM in pc/cm^3 (default: 350.0)",
    )
    parser.add_argument(
        "--snr", type=float, default=18.0, help="Injection target SNR (default: 18.0)"
    )
    parser.add_argument(
        "--dm-min", type=float, default=0.0, help="Minimum search DM (default: 0.0)"
    )
    parser.add_argument(
        "--dm-max",
        type=float,
        default=1000.0,
        help="Maximum search DM (default: 1000.0)",
    )
    parser.add_argument(
        "--dm-step", type=float, default=10.0, help="Search DM step (default: 10.0)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=7.0,
        help="SNR detection threshold (default: 7.0)",
    )
    parser.add_argument(
        "--out-plot", type=Path, default=default_plot, help="Output candidate plot path"
    )
    parser.add_argument(
        "--voevent-xml",
        type=Path,
        default=None,
        help="Optional output path for IVOA VOEvent v2.0 XML alert",
    )
    parser.add_argument(
        "--voevent-json",
        type=Path,
        default=None,
        help="Optional output path for NE2025-annotated candidate JSON alert",
    )
    args = parser.parse_args()

    run_frb_detection(
        h5_path=args.h5_path,
        inject=args.inject,
        inject_dm=args.dm,
        inject_snr=args.snr,
        dm_min=args.dm_min,
        dm_max=args.dm_max,
        dm_step=args.dm_step,
        threshold_snr=args.threshold,
        out_plot=args.out_plot,
        voevent_xml=args.voevent_xml,
        voevent_json=args.voevent_json,
    )


if __name__ == "__main__":
    main()
