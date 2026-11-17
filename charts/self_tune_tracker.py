#!/usr/bin/env python3
"""Closed-loop self-tuning and waterfall verification tool for the CHARTS Direct Beam Tracker.

Supports:
  --analyze-run <dir>        Quantify k/20 accumulation bias, bandpass flatness (interp vs freqflt),
                             and improved RFI mitigation across captured beam_power_freq*.bin files.
  --push-rfi-gains           Compute smooth passband pre-whitening gains g(f) from raw RFI
                             channel power (with fallback) and push to POST /rfi_mitigation/set_gains.
  --reload-cal-gains <npy>   Hot-reload complex calibration gains via POST /direct_tracker/reload_gains.
  --check-antennas           Inspect GET /antenna_mask/status and GET /antenna_mask/channel_power.
  --daemon                   Run the full closed-loop self-tuning sidecar (raw pre-whitening gains,
                             selective latched-EMA recovery, incremental 1.024s RFI/bandpass audit,
                             conservative static-notch hysteresis, and calibration hot-reload watch).
  --monitor                  Display live pipeline, closed-loop daemon, and output QA telemetry.
"""

from __future__ import annotations

import argparse
from collections import deque
import datetime
import glob
import json
import os
import re
import struct
import sys
import time
import urllib.error
import urllib.request
import warnings
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

SUBBAND0_START_HZ = 300.0e6
SUBBAND1_START_HZ = 400.8e6
CHANNEL_WIDTH_HZ = 300.0e3
N_CHAN_SUBBAND = 336
N_CHAN_TOTAL = 672
DEFAULT_SIGNAL_ANTS = [0, 4, 32, 36]
DEFAULT_OLD_GAINS = "/data/calibration/corr_1113/gains_corr_1113_0_4_32_36_interp_672x64.npy"
DEFAULT_NEW_GAINS = "/data/calibration/corr_1113/gains_corr_1113_0_4_32_36_freqflt_672x64.npy"
EXPECTED_ACCUM_DSEQ = 307200
MAX_AUTO_NOTCH_CHANNELS = 24


def http_get_json(base_url: str, path: str, timeout: float = 5.0) -> Any:
    url = base_url.rstrip("/") + path
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_post_json(base_url: str, path: str, payload: Dict[str, Any], timeout: float = 5.0) -> Any:
    url = base_url.rstrip("/") + path
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8").strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"message": raw}


def read_beam_power_file(path: str, n_freq: int = N_CHAN_SUBBAND, n_beams: int = 1) -> np.ndarray:
    """Read a rawFileWrite binary beam power file into shape (n_frames, n_freq)."""
    payload_floats = n_freq * n_beams
    payload_bytes = payload_floats * 4
    frames: List[np.ndarray] = []
    with open(path, "rb") as f:
        while True:
            sz_buf = f.read(4)
            if len(sz_buf) < 4:
                break
            (meta_size,) = struct.unpack("<I", sz_buf)
            if meta_size > 1_000_000:
                raise ValueError(f"Unreasonable metadata_size={meta_size} in {path}")
            if meta_size > 0:
                skipped = f.read(meta_size)
                if len(skipped) < meta_size:
                    break
            raw = f.read(payload_bytes)
            if len(raw) < payload_bytes:
                break
            arr = np.frombuffer(raw, dtype="<f4").copy()
            frames.append(arr[:n_freq])
    if not frames:
        return np.empty((0, n_freq), dtype=np.float32)
    return np.vstack(frames)


def load_run_waterfalls(run_dir: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load stitched subband 0 & 1 beam power waterfalls (T, 672) and run metadata."""
    rdir = Path(run_dir)
    f0_list = sorted(glob.glob(str(rdir / "beam_power_freq0_*.bin")))
    f1_list = sorted(glob.glob(str(rdir / "beam_power_freq1_*.bin")))
    if not f0_list or not f1_list:
        raise FileNotFoundError(
            f"Missing beam_power_freq0_*.bin or beam_power_freq1_*.bin in {run_dir}"
        )

    sb0 = np.vstack([read_beam_power_file(p) for p in f0_list])
    sb1 = np.vstack([read_beam_power_file(p) for p in f1_list])
    n_t = min(sb0.shape[0], sb1.shape[0])
    if n_t == 0:
        raise ValueError(f"No complete beam power frames found in {run_dir}")

    waterfall = np.hstack([sb0[:n_t], sb1[:n_t]]).astype(np.float64)

    meta: Dict[str, Any] = {"signal_antennas": list(DEFAULT_SIGNAL_ANTS)}
    meta_files = sorted(glob.glob(str(rdir / "run_metadata_*.json")))
    if meta_files:
        try:
            with open(meta_files[-1], "r", encoding="utf-8") as mf:
                meta = json.load(mf)
        except Exception:
            pass
    cfg_files = sorted(glob.glob(str(rdir / "kotekan_config_*.json")))
    if cfg_files:
        try:
            cfg_text = Path(cfg_files[-1]).read_text(encoding="utf-8", errors="ignore")
            matches = re.findall(r'"gains_file"\s*:\s*"([^"]+)"', cfg_text)
            if matches:
                meta["active_gains_file"] = matches[-1]
        except Exception:
            pass
    return waterfall, meta


def _rescale_bandpass_if_needed(
    waterfall: np.ndarray,
    meta: Dict[str, Any],
    old_gains_path: str,
    new_gains_path: str,
) -> Tuple[np.ndarray, bool]:
    """Scale old interp runs -> new_gains_path; skip if run already used new_gains_path."""
    active_gains = meta.get("active_gains_file")
    if active_gains and Path(active_gains).name == Path(new_gains_path).name:
        return waterfall.copy(), False
    signal_ants = meta.get("signal_antennas", DEFAULT_SIGNAL_ANTS)
    s_old = gains_power_scale(old_gains_path, signal_ants)
    s_new = gains_power_scale(new_gains_path, signal_ants)
    if s_old is None or s_new is None:
        return waterfall.copy(), False
    valid_s = s_old > 0.0
    bp_ratio = np.where(valid_s, s_new / np.where(valid_s, s_old, 1.0), 1.0)
    return waterfall * bp_ratio[None, :], True


def running_median_1d(arr: np.ndarray, half_width: int = 16) -> np.ndarray:
    """Sliding median envelope (~10 MHz wide) avoiding narrowband RFI spikes."""
    n = arr.shape[0]
    out = np.empty_like(arr, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - half_width)
        hi = min(n, i + half_width + 1)
        out[i] = float(np.median(arr[lo:hi]))
    return out


def compute_smooth_rfi_gains(
    spectrum: np.ndarray, clean_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """Compute smooth normalized RFI pre-whitening gains g(f) (~sqrt(env / median(env)))."""
    valid = np.isfinite(spectrum) & (spectrum > 0.0)
    if clean_mask is not None and len(clean_mask) == len(spectrum):
        valid = valid & np.asarray(clean_mask, dtype=bool)
    if not np.any(valid):
        valid = np.isfinite(spectrum) & (spectrum > 0.0)
    if not np.any(valid):
        return np.ones(len(spectrum), dtype=np.float32)

    idx = np.arange(len(spectrum), dtype=np.float64)
    clean_spec = np.interp(idx, idx[valid], spectrum[valid])

    env = running_median_1d(clean_spec, half_width=16)
    ref = float(np.median(env))
    if ref <= 0.0:
        return np.ones(len(spectrum), dtype=np.float32)

    gains = np.sqrt(np.clip(env / ref, 0.0625, 16.0))
    return gains.astype(np.float32)


def gains_power_scale(
    gains_path: str, signal_ants: List[int] = DEFAULT_SIGNAL_ANTS
) -> Optional[np.ndarray]:
    """Compute incoherent beam power weight factor sum(1/|g_a(f)|^2) across active antennas."""
    if not os.path.exists(gains_path):
        return None
    g = np.load(gains_path)
    if g.ndim != 2 or g.shape[0] < N_CHAN_TOTAL:
        return None
    sub = g[:N_CHAN_TOTAL, signal_ants]
    mag2 = np.abs(sub) ** 2
    inv_mag2 = np.where(np.isfinite(mag2) & (mag2 > 1e-12), 1.0 / mag2, 0.0)
    return np.sum(inv_mag2, axis=1).astype(np.float64)


def detect_and_correct_k20_dilution(
    waterfall: np.ndarray, n_accum: int = 20
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Detect quantized k/20 dilution steps in old chartsAccumulate output and reconstruct mean."""
    n_t, n_f = waterfall.shape
    chan_med = np.median(waterfall, axis=0)
    valid_chan = chan_med > 0.0
    corrected = waterfall.copy()

    diluted_cells = 0
    zero_cells = 0
    for f in range(n_f):
        med = chan_med[f]
        if med <= 0.0:
            continue
        col = waterfall[:, f]
        ratio = col / med
        mad_rel = max(float(1.4826 * np.median(np.abs(ratio - 1.0))), 0.015)

        zero_mask = col <= 0.0
        zero_cells += int(np.sum(zero_mask))

        # Flag cells quantized cleanly below 1.0 by at least 4 sigma (typical partial-frame blanking)
        dip_thresh = min(0.95, 1.0 - max(4.0 * mad_rel, 0.06))
        dip_mask = (~zero_mask) & (ratio < dip_thresh)
        if np.any(dip_mask):
            k_est = np.rint(ratio[dip_mask] * n_accum)
            near_quant = (k_est >= 1.0) & (k_est < float(n_accum)) & (
                np.abs(ratio[dip_mask] * n_accum - k_est) < 0.35
            )
            idx = np.where(dip_mask)[0][near_quant]
            if len(idx) > 0:
                diluted_cells += len(idx)
                corrected[idx, f] = col[idx] * (float(n_accum) / k_est[near_quant])

    return corrected, {
        "total_cells": int(n_t * n_f),
        "zero_cells": zero_cells,
        "diluted_k_of_20_cells": diluted_cells,
    }


def simulate_improved_rfi_excision(
    waterfall: np.ndarray,
    ema_alpha: float = 0.02,
    burst_sigma_thresh: float = 5.0,
    mad_sigma_thresh: float = 6.0,
    freq_win_half: int = 8,
    edge_ignore: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate persistent-EMA + edge-enabled + pre-whitened RFI detection at integrated cadence."""
    n_t, n_f = waterfall.shape
    chan_med = np.median(waterfall, axis=0)
    rfi_gains = compute_smooth_rfi_gains(chan_med).astype(np.float64)

    # 1. Vectorized frequency-domain local MAD outlier check across full band (edge_ignore=0)
    row_norm = waterfall / np.maximum(rfi_gains[None, :] ** 2, 1e-6)
    padded = np.pad(row_norm, ((0, 0), (freq_win_half, freq_win_half)), mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(padded, 2 * freq_win_half + 1, axis=1)
    w_med = np.median(win, axis=2)
    w_dev = np.abs(win - w_med[:, :, None])
    w_mad = np.maximum(1.4826 * np.median(w_dev, axis=2), 1e-9)
    inlier_mask = w_dev < (3.5 * w_mad[:, :, None])
    inlier_dev = np.where(inlier_mask, w_dev, np.nan)
    with np.errstate(all="ignore"):
        refined_mad = 1.4826 * np.nanmedian(inlier_dev, axis=2)
    local_mad = np.where(np.sum(inlier_mask, axis=2) >= 3, refined_mad, w_mad)
    freq_sigma = np.maximum(local_mad, 0.05 * np.maximum(w_med, 1e-3))
    freq_flagged = ((row_norm - w_med) / freq_sigma) >= mad_sigma_thresh
    if edge_ignore > 0:
        freq_flagged[:, :edge_ignore] = False
        freq_flagged[:, -edge_ignore:] = False

    # 2. Temporal persistent EMA burst check across consecutive integrated frames
    flag_mask = np.zeros((n_t, n_f), dtype=bool)
    ema_p = np.percentile(waterfall[: min(30, n_t)], 35, axis=0)
    ema_mad = 0.015 * ema_p

    for t in range(n_t):
        p = waterfall[t]
        valid = p > 0.0
        has_ema = valid & (ema_p > 0.0)
        t_sigma = np.minimum(
            np.maximum(ema_mad, 0.015 * np.maximum(ema_p, 1e-3)),
            0.025 * np.maximum(ema_p, 1e-3),
        )
        time_flag = has_ema & (((p - ema_p) / np.maximum(t_sigma, 1e-9)) >= burst_sigma_thresh)
        frame_flag = freq_flagged[t] | time_flag
        flag_mask[t] = frame_flag

        need_init = valid & (ema_p <= 0.0) & (~frame_flag)
        ema_p[need_init] = p[need_init]
        ema_mad[need_init] = 0.015 * p[need_init]

        update_mask = has_ema & (
            ((~frame_flag) & ((p - ema_p) < (2.5 * t_sigma))) | (p < ema_p)
        )
        dev = np.abs(p - ema_p)
        ema_p[update_mask] = (1.0 - ema_alpha) * ema_p[update_mask] + ema_alpha * p[update_mask]
        ema_mad[update_mask] = (1.0 - ema_alpha) * ema_mad[update_mask] + ema_alpha * np.minimum(
            1.4826 * dev[update_mask], 0.025 * ema_p[update_mask]
        )

    cleaned = waterfall.copy()
    cleaned[flag_mask] = np.nan
    return cleaned, flag_mask


def summarize_band_excursion(
    data: np.ndarray, freqs_mhz: np.ndarray, f_lo: float, f_hi: float
) -> Dict[str, Any]:
    """Compute peak excess % above per-channel median and RMS % for a frequency slice."""
    sel = (freqs_mhz >= f_lo) & (freqs_mhz <= f_hi)
    sub = data[:, sel]
    n_t, n_c = sub.shape
    rel_vals = []
    for c in range(n_c):
        col = sub[:, c]
        fin = col[np.isfinite(col) & (col > 0.0)]
        if len(fin) >= 5:
            med = float(np.median(fin))
            if med > 0.0:
                rel_vals.append(fin / med - 1.0)
    total = int(sub.size)
    kept = sum(len(v) for v in rel_vals)
    if kept == 0:
        return {"n_chan": int(np.sum(sel)), "blanked_pct": 100.0, "peak_pct": 0.0, "rms_pct": 0.0}
    finite = np.concatenate(rel_vals)
    return {
        "n_chan": int(np.sum(sel)),
        "blanked_pct": round(100.0 * (total - kept) / total, 2),
        "peak_pct": round(100.0 * float(np.max(finite)), 2),
        "rms_pct": round(100.0 * float(np.std(finite)), 3),
    }


def analyze_run(
    run_dir: str,
    old_gains_path: str = DEFAULT_OLD_GAINS,
    new_gains_path: str = DEFAULT_NEW_GAINS,
    out_json: Optional[str] = None,
) -> Dict[str, Any]:
    waterfall_raw, meta = load_run_waterfalls(run_dir)
    n_t, n_f = waterfall_raw.shape
    freqs_mhz = (SUBBAND0_START_HZ + np.arange(n_f) * CHANNEL_WIDTH_HZ) * 1.0e-6
    signal_ants = meta.get("signal_antennas", DEFAULT_SIGNAL_ANTS)

    # 1. Reconstruct undiluted frames (simulating chartsAccumulate fix)
    waterfall_undiluted, dil_stats = detect_and_correct_k20_dilution(waterfall_raw)

    # 2. Apply bandpass ratio correction from old -> freq-filtered gains (only if not already active)
    waterfall_bp, bp_rescaled = _rescale_bandpass_if_needed(
        waterfall_undiluted, meta, old_gains_path, new_gains_path
    )

    # 3. Apply improved RFI excision simulation
    waterfall_clean, sim_mask = simulate_improved_rfi_excision(waterfall_bp)

    # Bandpass flatness metrics (across active valid channels 300-501.3 MHz)
    raw_spec = np.median(waterfall_raw, axis=0)
    valid_raw_ch = np.isfinite(raw_spec) & (raw_spec > 0.0)
    bp_spec = np.full(n_f, np.nan, dtype=np.float64)
    for f in range(n_f):
        col = waterfall_clean[:, f]
        fin = col[np.isfinite(col) & (col > 0.0)]
        if len(fin) >= 10:
            bp_spec[f] = float(np.median(fin))
    valid_bp_ch = np.isfinite(bp_spec) & (bp_spec > 0.0)

    raw_vals = raw_spec[valid_raw_ch]
    bp_vals = bp_spec[valid_bp_ch]
    raw_flat_std_pct = 100.0 * float(np.std(raw_vals) / np.median(raw_vals))
    bp_flat_std_pct = 100.0 * float(np.std(bp_vals) / np.median(bp_vals))
    raw_p90_p10 = float(np.percentile(raw_vals, 90) / np.percentile(raw_vals, 10))
    bp_p90_p10 = float(np.percentile(bp_vals, 90) / np.percentile(bp_vals, 10))

    # Residual dynamic spectrum RMS on identical active non-blanked channels
    common_ch = valid_raw_ch & valid_bp_ch
    with np.errstate(all="ignore"):
        raw_nz = np.where(waterfall_raw[:, common_ch] > 0.0, waterfall_raw[:, common_ch], np.nan)
        raw_rel = raw_nz / raw_spec[common_ch][None, :] - 1.0
        raw_cm = np.nanmedian(raw_rel, axis=1, keepdims=True)
        raw_res_rms_pct = 100.0 * float(np.nanstd(raw_rel - raw_cm))

        clean_nz = np.where(waterfall_clean[:, common_ch] > 0.0, waterfall_clean[:, common_ch], np.nan)
        clean_rel = clean_nz / bp_spec[common_ch][None, :] - 1.0
        clean_cm = np.nanmedian(clean_rel, axis=1, keepdims=True)
        clean_res = clean_rel - clean_cm
        clean_res_rms_pct = 100.0 * float(np.nanstd(clean_res))

    families = [
        ("edge_401_406", 401.7, 405.6),
        ("cw_414_0", 413.85, 414.15),
        ("burst_381_384", 381.0, 384.0),
        ("comb_362_378", 362.0, 378.0),
    ]

    rfi_comparison: Dict[str, Any] = {}
    for name, f_lo, f_hi in families:
        rfi_comparison[name] = {
            "band_mhz": [f_lo, f_hi],
            "before": summarize_band_excursion(waterfall_raw, freqs_mhz, f_lo, f_hi),
            "after": summarize_band_excursion(waterfall_clean, freqs_mhz, f_lo, f_hi),
        }

    report: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "frames": int(n_t),
        "channels": int(n_f),
        "signal_antennas": signal_ants,
        "dilution_audit": dil_stats,
        "bandpass_flatness": {
            "before_std_pct": round(raw_flat_std_pct, 2),
            "after_std_pct": round(bp_flat_std_pct, 2),
            "before_p90_p10_ratio": round(raw_p90_p10, 3),
            "after_p90_p10_ratio": round(bp_p90_p10, 3),
        },
        "dynamic_spectrum_rms": {
            "before_residual_rms_pct": round(raw_res_rms_pct, 3),
            "after_residual_rms_pct": round(clean_res_rms_pct, 3),
            "extra_flagged_pct": round(100.0 * float(np.mean(sim_mask)), 2),
        },
        "rfi_families": rfi_comparison,
    }

    print(f"=== CHARTS Closed-Loop Run Analysis: {run_dir} ({n_t} frames x {n_f} ch) ===")
    print(
        f"[chartsAccumulate] Zeroed cells: {dil_stats['zero_cells']}, "
        f"Quantized k/20 diluted cells corrected: {dil_stats['diluted_k_of_20_cells']} "
        f"/ {dil_stats['total_cells']}"
    )
    bp_tag = "interp->freqflt" if bp_rescaled else "freqflt active"
    print(
        f"[Bandpass {bp_tag}] Spectral CV: "
        f"{report['bandpass_flatness']['before_std_pct']}% -> "
        f"{report['bandpass_flatness']['after_std_pct']}% | "
        f"P90/P10: {report['bandpass_flatness']['before_p90_p10_ratio']} -> "
        f"{report['bandpass_flatness']['after_p90_p10_ratio']}"
    )
    print(
        f"[Dynamic Spectrum] Global residual RMS: "
        f"{report['dynamic_spectrum_rms']['before_residual_rms_pct']}% -> "
        f"{report['dynamic_spectrum_rms']['after_residual_rms_pct']}% "
        f"(+{report['dynamic_spectrum_rms']['extra_flagged_pct']}% extra RFI samples excised)"
    )
    for name, info in rfi_comparison.items():
        b = info["before"]
        a = info["after"]
        print(
            f"  [{name} {info['band_mhz'][0]:.1f}-{info['band_mhz'][1]:.1f} MHz] "
            f"Peak: +{b['peak_pct']}% -> +{a['peak_pct']}% | "
            f"RMS: {b['rms_pct']}% -> {a['rms_pct']}% | "
            f"Flagged: {a['blanked_pct']}%"
        )

    if out_json:
        with open(out_json, "w", encoding="utf-8") as jf:
            json.dump(report, jf, indent=2)
        print(f"Saved report JSON to {out_json}")

    return report


RFI_SK_IMPULSIVE = 1 << 0
RFI_SK_CW_TONE = 1 << 1
RFI_TIME_BURST = 1 << 2
RFI_FREQ_MAD = 1 << 3
RFI_STATIC_NOTCH = 1 << 4
RFI_SATURATION_CLIP = 1 << 5
RFI_MORPHOLOGY_SIR = 1 << 6


def _parse_frame_metadata(meta_bytes: bytes) -> Tuple[Optional[int], Optional[float]]:
    """Extract (fpga_seq_num, unix_timestamp_s) from serialized chartsMetadata if present."""
    seq: Optional[int] = None
    ts: Optional[float] = None
    if len(meta_bytes) >= 16:
        (seq,) = struct.unpack_from("<Q", meta_bytes, 8)
    if len(meta_bytes) >= 3024:
        sec, usec = struct.unpack_from("<QQ", meta_bytes, 3008)
        if sec > 1_000_000_000:
            ts = float(sec) + float(usec) * 1.0e-6
    return seq, ts


class _SubbandTailCursor:
    """Incremental reader cursor across hourly beam_power_freq{sb}_*.bin files."""

    def __init__(self, subband: int, n_freq: int = N_CHAN_SUBBAND, n_beams: int = 1) -> None:
        self.subband = subband
        self.n_freq = n_freq
        self.payload_bytes = n_freq * n_beams * 4
        self.active_path: Optional[str] = None
        self.offset = 0
        self.frames: Deque[np.ndarray] = deque()
        self.times: Deque[Optional[float]] = deque()
        self.seqs: Deque[Optional[int]] = deque()

    def reset(self) -> None:
        self.active_path = None
        self.offset = 0
        self.frames.clear()
        self.times.clear()
        self.seqs.clear()

    def _read_available_from(self, path: str) -> None:
        file_size = os.path.getsize(path)
        if self.offset > file_size:
            self.offset = 0
        with open(path, "rb") as f:
            f.seek(self.offset)
            while True:
                sz_buf = f.read(4)
                if len(sz_buf) < 4:
                    break
                (meta_size,) = struct.unpack("<I", sz_buf)
                if meta_size > 1_000_000:
                    raise ValueError(f"Unreasonable metadata_size={meta_size} in {path}")
                meta_bytes = f.read(meta_size) if meta_size > 0 else b""
                if len(meta_bytes) < meta_size:
                    break
                raw = f.read(self.payload_bytes)
                if len(raw) < self.payload_bytes:
                    break
                arr = np.frombuffer(raw, dtype="<f4", count=self.n_freq).copy()
                seq, ts = _parse_frame_metadata(meta_bytes)
                self.frames.append(arr)
                self.times.append(ts)
                self.seqs.append(seq)
                self.offset = f.tell()

    def poll(self, run_dir: str) -> int:
        files = sorted(glob.glob(os.path.join(run_dir, f"beam_power_freq{self.subband}_*.bin")))
        if not files:
            return 0
        if self.active_path is None or self.active_path not in files:
            start_idx = max(0, len(files) - 2)
            self.active_path = files[start_idx]
            self.offset = 0

        added = 0
        idx = files.index(self.active_path)
        while idx < len(files):
            cur_path = files[idx]
            before = len(self.frames)
            self._read_available_from(cur_path)
            added += len(self.frames) - before
            if idx + 1 < len(files):
                idx += 1
                self.active_path = files[idx]
                self.offset = 0
            else:
                break
        return added


class WaterfallTailer:
    """Incrementally tails SB0 + SB1 beam_power binary files and computes live QA metrics."""

    def __init__(self, run_dir: Optional[str], max_frames: int = 300) -> None:
        self.run_dir = run_dir
        self.max_frames = max(30, int(max_frames))
        self._resolved_dir: Optional[str] = None
        self._cursors = (_SubbandTailCursor(0), _SubbandTailCursor(1))
        self.aligned_frames: Deque[np.ndarray] = deque(maxlen=self.max_frames)
        self.aligned_times: Deque[Optional[float]] = deque(maxlen=self.max_frames)
        self.aligned_seqs: Deque[Optional[int]] = deque(maxlen=self.max_frames)
        self.total_aligned = 0
        self.seq_gaps = 0
        self._last_seq: Optional[int] = None
        self.expected_dseq: Optional[int] = EXPECTED_ACCUM_DSEQ
        self.active_gains_file: Optional[str] = None

    def reset(self) -> None:
        for cur in self._cursors:
            cur.reset()
        self.aligned_frames.clear()
        self.aligned_times.clear()
        self.aligned_seqs.clear()
        self.total_aligned = 0
        self.seq_gaps = 0
        self._last_seq = None
        self.active_gains_file = None

    def append_live_spectrum(self, spectrum: np.ndarray, ts: Optional[float] = None) -> None:
        if spectrum.shape[0] < N_CHAN_TOTAL:
            return
        self.aligned_frames.append(np.asarray(spectrum[:N_CHAN_TOTAL], dtype=np.float64))
        self.aligned_times.append(ts if ts is not None else time.time())
        self.aligned_seqs.append(None)
        self.total_aligned += 1

    def poll(self) -> int:
        if not self.run_dir:
            return 0
        resolved = os.path.realpath(self.run_dir)
        if not os.path.isdir(resolved):
            return 0
        if resolved != self._resolved_dir:
            self.reset()
            self._resolved_dir = resolved

        cfg_files = sorted(glob.glob(os.path.join(resolved, "kotekan_config_*.json")))
        if cfg_files and self.active_gains_file is None:
            try:
                txt = Path(cfg_files[-1]).read_text(encoding="utf-8", errors="ignore")
                matches = re.findall(r'"gains_file"\s*:\s*"([^"]+)"', txt)
                if matches:
                    self.active_gains_file = matches[-1]
            except Exception:
                pass

        for cur in self._cursors:
            cur.poll(resolved)

        paired = 0
        c0, c1 = self._cursors
        while c0.frames and c1.frames:
            s0 = c0.seqs[0]
            s1 = c1.seqs[0]
            if s0 is not None and s1 is not None and s0 != s1:
                if s0 < s1:
                    c0.frames.popleft()
                    c0.times.popleft()
                    c0.seqs.popleft()
                else:
                    c1.frames.popleft()
                    c1.times.popleft()
                    c1.seqs.popleft()
                continue

            f0 = c0.frames.popleft()
            t0 = c0.times.popleft()
            s0 = c0.seqs.popleft()
            f1 = c1.frames.popleft()
            _t1 = c1.times.popleft()
            _s1 = c1.seqs.popleft()

            if s0 is not None and self._last_seq is not None and self.expected_dseq:
                if (s0 - self._last_seq) != self.expected_dseq:
                    self.seq_gaps += 1
            if s0 is not None:
                self._last_seq = s0

            self.aligned_frames.append(np.concatenate([f0, f1]).astype(np.float64))
            self.aligned_times.append(t0)
            self.aligned_seqs.append(s0)
            self.total_aligned += 1
            paired += 1
        return paired

    def compute_qa(self) -> Dict[str, Any]:
        if len(self.aligned_frames) < 5:
            return {"window_frames": len(self.aligned_frames), "total_frames": self.total_aligned}

        wf = np.vstack(list(self.aligned_frames))
        n_t, _ = wf.shape
        zero_mask = (wf <= 0.0) | (~np.isfinite(wf))
        _, dil_stats = detect_and_correct_k20_dilution(wf)

        raw_med = np.median(wf, axis=0)
        valid = np.isfinite(raw_med) & (raw_med > 0.0)
        cv_pct = (
            100.0 * float(np.std(raw_med[valid]) / np.median(raw_med[valid]))
            if np.any(valid)
            else 0.0
        )
        p90_p10 = (
            float(np.percentile(raw_med[valid], 90) / np.percentile(raw_med[valid], 10))
            if np.any(valid)
            else 1.0
        )
        sb0_med = float(np.median(raw_med[:N_CHAN_SUBBAND][valid[:N_CHAN_SUBBAND]])) if np.any(valid[:N_CHAN_SUBBAND]) else 0.0
        sb1_med = float(np.median(raw_med[N_CHAN_SUBBAND:][valid[N_CHAN_SUBBAND:]])) if np.any(valid[N_CHAN_SUBBAND:]) else 0.0

        wf_clean, sim_mask = simulate_improved_rfi_excision(wf)
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore", category=RuntimeWarning)
            raw_nz = np.where(wf[:, valid] > 0.0, wf[:, valid], np.nan)
            raw_rel = raw_nz / raw_med[valid][None, :] - 1.0
            raw_rms_pct = 100.0 * float(np.nanstd(raw_rel - np.nanmedian(raw_rel, axis=1, keepdims=True)))

            bp_spec = np.nanmedian(np.where(wf_clean > 0.0, wf_clean, np.nan), axis=0)
            clean_nz = np.where(wf_clean[:, valid] > 0.0, wf_clean[:, valid], np.nan)
            clean_rel = clean_nz / bp_spec[valid][None, :] - 1.0
            clean_rms_pct = 100.0 * float(
                np.nanstd(clean_rel - np.nanmedian(clean_rel, axis=1, keepdims=True))
            )

        times = [t for t in self.aligned_times if t is not None]
        dt_med = float(np.median(np.diff(times))) if len(times) >= 2 else None
        last_ts = times[-1] if times else None

        return {
            "resolved_run_dir": self._resolved_dir,
            "total_frames": int(self.total_aligned),
            "window_frames": int(n_t),
            "seq_gaps": int(self.seq_gaps),
            "dt_med_s": round(dt_med, 6) if dt_med is not None else None,
            "last_frame_utc": (
                datetime.datetime.fromtimestamp(last_ts, datetime.timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                )
                if last_ts is not None
                else None
            ),
            "last_frame_age_s": round(time.time() - last_ts, 1) if last_ts is not None else None,
            "sb0_zero_pct": round(100.0 * float(np.mean(zero_mask[:, :N_CHAN_SUBBAND])), 3),
            "sb1_zero_pct": round(100.0 * float(np.mean(zero_mask[:, N_CHAN_SUBBAND:])), 3),
            "overall_zero_pct": round(100.0 * float(np.mean(zero_mask)), 3),
            "k20_diluted_recent": int(dil_stats["diluted_k_of_20_cells"]),
            "sb0_median_power": round(sb0_med, 3),
            "sb1_median_power": round(sb1_med, 3),
            "bandpass_cv_pct": round(cv_pct, 2),
            "bandpass_p90_p10": round(p90_p10, 3),
            "raw_residual_rms_pct": round(raw_rms_pct, 3),
            "slow_rfi_cleaned_rms_pct": round(clean_rms_pct, 3),
            "slow_extra_flagged_pct": round(100.0 * float(np.mean(sim_mask)), 2),
            "slow_channel_occupancy": np.mean(sim_mask, axis=0),
        }


def fetch_raw_rfi_channels(base_url: str) -> Tuple[List[Dict[str, Any]], bool]:
    """Fetch per-channel raw RFI stats across subbands from GET /rfi_mitigation/channels."""
    try:
        data = http_get_json(base_url, "/rfi_mitigation/channels")
        chans = sorted(data.get("channels", []), key=lambda c: float(c.get("freq_mhz", 0.0)))
        if chans:
            return chans, True
    except urllib.error.HTTPError as err:
        if err.code != 404:
            raise
    return [], False


def compute_raw_rfi_gains_from_channels(
    channels: List[Dict[str, Any]],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    """Compute smooth SB0 & SB1 pre-whitening gains g(f) using unflagged raw channel powers."""
    if len(channels) < N_CHAN_SUBBAND:
        return None, None, {}
    pwr = np.array([float(c.get("mean_power", 0.0)) for c in channels], dtype=np.float64)
    frac = np.array([float(c.get("flagged_chunk_fraction", 0.0)) for c in channels], dtype=np.float64)
    bits = np.array([int(c.get("flag_bits", 0)) for c in channels], dtype=np.int64)
    mad_z = np.array([float(c.get("freq_mad_z", 0.0)) for c in channels], dtype=np.float64)
    clean = (
        np.isfinite(pwr)
        & (pwr > 0.0)
        & (frac < 0.25)
        & ((bits & ~(RFI_TIME_BURST | RFI_MORPHOLOGY_SIR)) == 0)
        & (np.abs(mad_z) <= 4.0)
    )

    g_sb0 = compute_smooth_rfi_gains(pwr[:N_CHAN_SUBBAND], clean[:N_CHAN_SUBBAND])
    g_sb1: Optional[np.ndarray] = None
    if len(pwr) >= N_CHAN_TOTAL:
        g_sb1 = compute_smooth_rfi_gains(
            pwr[N_CHAN_SUBBAND:N_CHAN_TOTAL], clean[N_CHAN_SUBBAND:N_CHAN_TOTAL]
        )

    meta = {
        "source": "rfi_mitigation/channels",
        "clean_raw_channels": int(np.sum(clean)),
        "raw_median_power": round(float(np.median(pwr[pwr > 0])) if np.any(pwr > 0) else 0.0, 3),
        "g_sb0_minmax": [round(float(g_sb0.min()), 4), round(float(g_sb0.max()), 4)],
        "g_sb1_minmax": (
            [round(float(g_sb1.min()), 4), round(float(g_sb1.max()), 4)]
            if g_sb1 is not None
            else None
        ),
    }
    return g_sb0, g_sb1, meta


def detect_latched_ema_channels(
    channels: List[Dict[str, Any]],
    latch_streaks: Dict[float, int],
    last_reset_ts: Dict[float, float],
    now: float,
    streak_threshold: int = 2,
    cooldown_s: float = 180.0,
) -> List[float]:
    """Identify channels stuck in TIME_BURST/SIR at Gaussian SK & normal neighborhood power."""
    to_reset: List[float] = []
    seen_freqs = set()
    for ch in channels:
        f_mhz = round(float(ch.get("freq_mhz", 0.0)), 3)
        seen_freqs.add(f_mhz)
        frac = float(ch.get("flagged_chunk_fraction", 0.0))
        bits = int(ch.get("flag_bits", 0))
        sk = float(ch.get("mean_sk", 1.0))
        mad_z = float(ch.get("freq_mad_z", 0.0))

        pure_burst_latch = (
            frac >= 0.85
            and (bits & (RFI_TIME_BURST | RFI_MORPHOLOGY_SIR)) != 0
            and (bits & ~(RFI_TIME_BURST | RFI_MORPHOLOGY_SIR)) == 0
            and (0.75 <= sk <= 1.25)
            and (abs(mad_z) <= 2.5)
        )
        if pure_burst_latch:
            latch_streaks[f_mhz] = latch_streaks.get(f_mhz, 0) + 1
            if latch_streaks[f_mhz] >= streak_threshold:
                if (now - last_reset_ts.get(f_mhz, 0.0)) >= cooldown_s:
                    to_reset.append(f_mhz)
                    last_reset_ts[f_mhz] = now
                    latch_streaks[f_mhz] = 0
        else:
            latch_streaks[f_mhz] = 0

    for old_f in list(latch_streaks.keys()):
        if old_f not in seen_freqs:
            del latch_streaks[old_f]
    return to_reset


def update_managed_notches(
    channels: List[Dict[str, Any]],
    slow_occ: Optional[np.ndarray],
    existing_notches: List[List[float]],
    bad_streaks: Dict[float, int],
    clean_streaks: Dict[float, int],
    managed_freqs: set[float],
    promote_streak: int = 5,
    demote_streak: int = 5,
    max_managed: int = MAX_AUTO_NOTCH_CHANNELS,
) -> Tuple[List[List[float]], bool, Dict[str, Any]]:
    """Conservative hysteresis manager for persistent physical RFI narrowband static notches."""
    # Preserve user/external notches outside daemon-managed +-0.05 MHz bands
    manual_notches: List[List[float]] = []
    for pair in existing_notches:
        if len(pair) < 2:
            continue
        lo, hi = float(pair[0]), float(pair[1])
        mid = round(0.5 * (lo + hi), 3)
        if any(abs(mid - mf) <= 0.06 and abs((hi - lo) - 0.10) <= 0.05 for mf in managed_freqs):
            continue
        manual_notches.append([lo, hi])

    promoted: List[float] = []
    demoted: List[float] = []
    next_managed = set(managed_freqs)

    for idx, ch in enumerate(channels):
        f_mhz = round(float(ch.get("freq_mhz", 0.0)), 3)
        raw_occ = float(ch.get("flagged_chunk_fraction", 0.0))
        slow_val = float(slow_occ[idx]) if (slow_occ is not None and idx < len(slow_occ)) else 0.0
        bits = int(ch.get("flag_bits", 0))
        sk = float(ch.get("mean_sk", 1.0))
        mad_z = float(ch.get("freq_mad_z", 0.0))

        physical_rfi = (
            abs(mad_z) >= 6.0
            or sk >= 2.2
            or sk <= 0.65
            or (bits & RFI_SATURATION_CLIP) != 0
        )
        severe_persistent = physical_rfi and (raw_occ >= 0.90 or slow_val >= 0.80)
        clean_raw = (abs(mad_z) < 3.0) and (0.80 <= sk <= 1.20) and ((bits & RFI_SATURATION_CLIP) == 0)

        if severe_persistent:
            bad_streaks[f_mhz] = bad_streaks.get(f_mhz, 0) + 1
            clean_streaks[f_mhz] = 0
            if (
                f_mhz not in next_managed
                and bad_streaks[f_mhz] >= promote_streak
                and len(next_managed) < max_managed
            ):
                next_managed.add(f_mhz)
                promoted.append(f_mhz)
        elif clean_raw:
            bad_streaks[f_mhz] = 0
            if f_mhz in next_managed:
                clean_streaks[f_mhz] = clean_streaks.get(f_mhz, 0) + 1
                if clean_streaks[f_mhz] >= demote_streak:
                    next_managed.remove(f_mhz)
                    demoted.append(f_mhz)
                    clean_streaks[f_mhz] = 0
        else:
            bad_streaks[f_mhz] = max(0, bad_streaks.get(f_mhz, 0) - 1)
            clean_streaks[f_mhz] = 0

    changed = next_managed != managed_freqs
    managed_freqs.clear()
    managed_freqs.update(next_managed)

    combined = list(manual_notches)
    for mf in sorted(managed_freqs):
        band = [round(mf - 0.05, 3), round(mf + 0.05, 3)]
        if not any(abs(b[0] - band[0]) <= 0.01 and abs(b[1] - band[1]) <= 0.01 for b in combined):
            combined.append(band)
    return combined, changed, {"promoted": promoted, "demoted": demoted, "managed_count": len(managed_freqs)}


def validate_gains_file(
    gains_path: str, signal_ants: List[int] = DEFAULT_SIGNAL_ANTS
) -> Tuple[bool, str]:
    """Validate candidate complex calibration gains table before live tracker hot-reload."""
    if not os.path.isfile(gains_path):
        return False, "file not found"
    try:
        g = np.load(gains_path)
    except Exception as exc:
        return False, f"load error: {exc}"
    if g.ndim != 2 or g.shape[0] < N_CHAN_TOTAL or g.shape[1] <= max(signal_ants):
        return False, f"unexpected shape {g.shape}"
    sub = g[:N_CHAN_TOTAL, signal_ants]
    if not np.all(np.isfinite(sub)):
        return False, "contains non-finite gain entries on active antennas"
    mag = np.abs(sub)
    if np.any(mag <= 1e-6):
        return False, "contains near-zero gain magnitude on active antennas"
    inv_pow = np.sum(1.0 / (mag**2), axis=1)
    med = float(np.median(inv_pow))
    if med <= 0.0 or (float(np.max(inv_pow)) / med) > 25.0:
        return False, f"excessive dynamic range in 1/|g|^2 (max/med={float(np.max(inv_pow))/max(med,1e-12):.2f})"
    return True, f"ok shape={g.shape} norm_med={med:.3f}"


def _write_json_atomic(path: str, payload: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, p)


def push_rfi_gains(
    base_url: str,
    run_dir: Optional[str] = None,
    new_gains_path: str = DEFAULT_NEW_GAINS,
    prev_gains: Optional[Tuple[np.ndarray, Optional[np.ndarray]]] = None,
    min_delta: float = 0.0,
    verbose: bool = True,
) -> Tuple[bool, Optional[Tuple[np.ndarray, Optional[np.ndarray]]], Dict[str, Any]]:
    """Compute smooth RFI pre-whitening gains (preferring raw /rfi_mitigation/channels) and push."""
    g_sb0: Optional[np.ndarray] = None
    g_sb1: Optional[np.ndarray] = None
    info: Dict[str, Any] = {}

    channels, has_raw = fetch_raw_rfi_channels(base_url)
    if has_raw:
        g_sb0, g_sb1, info = compute_raw_rfi_gains_from_channels(channels)

    if g_sb0 is None:
        spectrum: Optional[np.ndarray] = None
        if run_dir:
            try:
                wf, meta = load_run_waterfalls(run_dir)
                wf_u, _ = detect_and_correct_k20_dilution(wf)
                wf_u, _ = _rescale_bandpass_if_needed(
                    wf_u, meta, DEFAULT_OLD_GAINS, new_gains_path
                )
                spectrum = np.median(wf_u[-min(len(wf_u), 180) :], axis=0)
                info["source"] = "beam_power_bin_fallback"
            except Exception as exc:
                print(f"[warn] Could not read run_dir {run_dir}: {exc}", file=sys.stderr)
        if spectrum is None:
            bp = http_get_json(base_url, "/direct_tracker/beam_power")
            powers = bp.get("beam_power", [])
            if not powers or len(powers[0]) < N_CHAN_SUBBAND:
                if verbose:
                    print("[info] No live non-zero spectrum available to fit RFI gains.")
                return False, prev_gains, {"source": "none"}
            arr = np.asarray(powers[0], dtype=np.float64)
            if np.max(arr) <= 0.0:
                if verbose:
                    print("[info] Live pipeline is idling at 0 power; skipping RFI gain push.")
                return False, prev_gains, {"source": "idle"}
            spectrum = arr
            info["source"] = "beam_power_rest_fallback"

        g_sb0 = compute_smooth_rfi_gains(spectrum[:N_CHAN_SUBBAND])
        if len(spectrum) >= N_CHAN_TOTAL:
            g_sb1 = compute_smooth_rfi_gains(spectrum[N_CHAN_SUBBAND:N_CHAN_TOTAL])

    if prev_gains is not None:
        prev0, prev1 = prev_gains
        if prev0 is not None and g_sb0 is not None:
            g_sb0 = (0.5 * prev0 + 0.5 * g_sb0).astype(np.float32)
        if prev1 is not None and g_sb1 is not None:
            g_sb1 = (0.5 * prev1 + 0.5 * g_sb1).astype(np.float32)

        d0 = float(np.max(np.abs(g_sb0 - prev0))) if (prev0 is not None and g_sb0 is not None) else 1.0
        d1 = float(np.max(np.abs(g_sb1 - prev1))) if (prev1 is not None and g_sb1 is not None) else 0.0
        if max(d0, d1) < min_delta:
            info["skipped_small_delta"] = round(max(d0, d1), 5)
            return False, prev_gains, info

    res0 = http_post_json(
        base_url,
        "/rfi_mitigation/set_gains",
        {"freq_start_hz": SUBBAND0_START_HZ, "gains": g_sb0.tolist()},
    )
    if verbose:
        print(f"Pushed SB0 RFI gains ({info.get('source', '?')}): {res0}")
    if g_sb1 is not None:
        res1 = http_post_json(
            base_url,
            "/rfi_mitigation/set_gains",
            {"freq_start_hz": SUBBAND1_START_HZ, "gains": g_sb1.tolist()},
        )
        if verbose:
            print(f"Pushed SB1 RFI gains ({info.get('source', '?')}): {res1}")
    return True, (g_sb0, g_sb1), info


class ClosedLoopDaemon:
    """Stateful closed-loop supervisor bridging GPU telemetry, waterfall QA, and calibration."""

    def __init__(
        self,
        base_url: str,
        run_dir: Optional[str] = None,
        status_json: Optional[str] = None,
        watch_gains: Optional[str] = None,
        auto_notch: bool = True,
        new_gains_path: str = DEFAULT_NEW_GAINS,
    ) -> None:
        self.base_url = base_url
        self.run_dir = run_dir
        if status_json:
            self.status_json = status_json
        elif run_dir:
            self.status_json = os.path.join(run_dir, "self_tune_status.json")
        else:
            self.status_json = "/tmp/charts_self_tune_status.json"
        self.watch_gains = watch_gains
        self.auto_notch = auto_notch
        self.new_gains_path = new_gains_path

        self.tailer = WaterfallTailer(run_dir=run_dir, max_frames=300)
        self.prev_rfi_gains: Optional[Tuple[np.ndarray, Optional[np.ndarray]]] = None
        self.latch_streaks: Dict[float, int] = {}
        self.last_ema_reset_ts: Dict[float, float] = {}
        self.notch_bad_streaks: Dict[float, int] = {}
        self.notch_clean_streaks: Dict[float, int] = {}
        self.managed_notch_freqs: set[float] = set()
        if os.path.isfile(self.status_json):
            try:
                prev = json.loads(Path(self.status_json).read_text(encoding="utf-8"))
                saved_notches = prev.get("rfi", {}).get("managed_notch_freqs_mhz", [])
                self.managed_notch_freqs.update(round(float(x), 3) for x in saved_notches)
            except Exception:
                pass

        self.last_good_gains_file: Optional[str] = None
        self.last_gains_sig: Optional[Tuple[str, int, int]] = None
        self.cycle_count = 0
        self.action_log: Deque[str] = deque(maxlen=25)
        self.last_status: Dict[str, Any] = {}

    def _record_action(self, msg: str) -> None:
        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.action_log.append(f"{ts} {msg}")

    def _check_and_reload_gains(self, trk: Dict[str, Any]) -> Dict[str, Any]:
        active_file = trk.get("gains_file") or ""
        if active_file and self.last_good_gains_file is None:
            self.last_good_gains_file = active_file

        candidate = active_file
        if self.watch_gains:
            matches = sorted(glob.glob(self.watch_gains), key=os.path.getmtime)
            if matches:
                candidate = matches[-1]
        if not candidate or not os.path.isfile(candidate):
            return {"checked": False}

        st = os.stat(candidate)
        sig = (os.path.realpath(candidate), int(st.st_mtime_ns), int(st.st_size))
        if self.last_gains_sig is None:
            self.last_gains_sig = sig
            return {"checked": True, "path": candidate, "event": "baseline"}

        if sig == self.last_gains_sig:
            return {"checked": True, "path": candidate, "event": "unchanged"}

        sig_ants = trk.get("active_physical_antennas") or DEFAULT_SIGNAL_ANTS
        sig_ants = sorted([int(a) for a in sig_ants])
        ok, reason = validate_gains_file(candidate, sig_ants)
        if not ok:
            self._record_action(f"cal_reload rejected {candidate}: {reason}")
            return {"checked": True, "path": candidate, "event": "rejected", "reason": reason}

        old_gen = int(trk.get("gains_generation", 0))
        resolved = str(Path(candidate).resolve())
        try:
            reply = http_post_json(
                self.base_url, "/direct_tracker/reload_gains", {"gains_file": resolved}
            )
            new_trk = http_get_json(self.base_url, "/direct_tracker/status")
            new_gen = int(new_trk.get("gains_generation", old_gen))
            self.last_gains_sig = sig
            self.last_good_gains_file = resolved
            msg = f"cal_reload {Path(resolved).name} gen {old_gen}->{new_gen}: {reply}"
            self._record_action(msg)
            return {
                "checked": True,
                "path": resolved,
                "event": "reloaded",
                "gains_generation": new_gen,
            }
        except Exception as exc:
            self._record_action(f"cal_reload error on {candidate}: {exc}")
            return {"checked": True, "path": candidate, "event": "error", "reason": str(exc)}

    def _fetch_gpu_util(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for idx, ep in enumerate(("/gpu_profile/stages/gpu_0", "/gpu_profile/stages/gpu_1")):
            try:
                gp = http_get_json(self.base_url, ep, timeout=3.0)
                cin = sum(float(x.get("time", 0.0)) for x in gp.get("copy_in", []))
                cout = sum(float(x.get("time", 0.0)) for x in gp.get("copy_out", []))
                out[f"gpu_{idx}"] = {
                    "copy_in_ms": round(cin * 1e3, 2),
                    "copy_out_ms": round(cout * 1e3, 2),
                }
            except Exception:
                pass
        return out

    def step(self) -> Dict[str, Any]:
        now = time.time()
        self.cycle_count += 1
        cycle_actions: List[str] = []

        rfi = http_get_json(self.base_url, "/rfi_mitigation/status")
        trk = http_get_json(self.base_url, "/direct_tracker/status")
        ant = http_get_json(self.base_url, "/antenna_mask/status")
        channels, has_raw_channels = fetch_raw_rfi_channels(self.base_url)

        # 1. Incremental 1.024 s waterfall QA (or REST beam_power fallback)
        if self.run_dir:
            self.tailer.poll()
        else:
            bp = http_get_json(self.base_url, "/direct_tracker/beam_power")
            powers = bp.get("beam_power", [])
            if powers and len(powers[0]) >= N_CHAN_TOTAL and max(powers[0]) > 0.0:
                self.tailer.append_live_spectrum(
                    np.asarray(powers[0], dtype=np.float64), ts=bp.get("timestamp_utc")
                )

        qa = self.tailer.compute_qa()
        slow_occ = qa.pop("slow_channel_occupancy", None)

        # 2. Loop A: Pre-whitening RFI gain update (raw channel power domain when exposed)
        min_delta = 0.0 if self.prev_rfi_gains is None else 0.005
        pushed, self.prev_rfi_gains, rfi_gain_info = push_rfi_gains(
            self.base_url,
            run_dir=self.run_dir,
            new_gains_path=self.new_gains_path,
            prev_gains=self.prev_rfi_gains,
            min_delta=min_delta,
            verbose=False,
        )
        if pushed:
            msg = f"set_gains ({rfi_gain_info.get('source', '?')})"
            cycle_actions.append(msg)
            self._record_action(msg)

        # 3. Loop B: Selective latched-EMA recovery (only TIME_BURST/SIR at Gaussian SK & normal MAD)
        ema_reset_freqs: List[float] = []
        if has_raw_channels:
            ema_reset_freqs = detect_latched_ema_channels(
                channels, self.latch_streaks, self.last_ema_reset_ts, now
            )
            if ema_reset_freqs:
                try:
                    res = http_post_json(
                        self.base_url, "/rfi_mitigation/reset_ema", {"freq_mhz": ema_reset_freqs}
                    )
                    msg = f"reset_ema {ema_reset_freqs} -> {res.get('mode', 'ok')}"
                    cycle_actions.append(msg)
                    self._record_action(msg)
                except urllib.error.HTTPError as err:
                    if err.code != 404:
                        raise

        # 4. Loop C: Conservative narrowband persistent-RFI static-notch hysteresis
        notch_info: Dict[str, Any] = {"managed_count": len(self.managed_notch_freqs)}
        if self.auto_notch and has_raw_channels:
            existing_notches = rfi.get("static_notch_bands_mhz", [])
            new_notches, notch_changed, notch_info = update_managed_notches(
                channels,
                slow_occ,
                existing_notches,
                self.notch_bad_streaks,
                self.notch_clean_streaks,
                self.managed_notch_freqs,
            )
            if notch_changed:
                http_post_json(
                    self.base_url,
                    "/rfi_mitigation/configure",
                    {"static_notch_bands_mhz": new_notches},
                )
                msg = f"notches promoted={notch_info['promoted']} demoted={notch_info['demoted']}"
                cycle_actions.append(msg)
                self._record_action(msg)

        # 5. Loop D: Calibration gains file watch & validated hot-reload
        cal_info = self._check_and_reload_gains(trk)
        if cal_info.get("event") in ("reloaded", "rejected", "error"):
            cycle_actions.append(f"cal:{cal_info['event']}")

        # 6. Loop E: Hardware/pipeline QA snapshot
        ants = ant.get("antennas", [])
        active_ant_rows = [a for a in ants if a.get("active")]
        min_healthy = (
            min(int(a.get("consecutive_healthy", 0)) for a in active_ant_rows)
            if active_ant_rows
            else 0
        )
        max_clip = (
            max(float(a.get("clip_fraction", 0.0)) for a in active_ant_rows)
            if active_ant_rows
            else 0.0
        )

        status_payload = {
            "timestamp_utc": datetime.datetime.fromtimestamp(
                now, datetime.timezone.utc
            ).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "cycle": self.cycle_count,
            "url": self.base_url,
            "run_dir": self.run_dir,
            "pipeline": {
                "gains_file": trk.get("gains_file"),
                "gains_generation": trk.get("gains_generation"),
                "apply_calibration_gains": trk.get("apply_calibration_gains"),
                "current_lst_hours": round(float(trk.get("current_lst_hours", 0.0)), 4),
                "active_antennas": ant.get("active_physical_antennas", []),
                "dead_antennas": ant.get("dead_antennas", 0),
                "saturated_antennas": ant.get("saturated_antennas", 0),
                "min_consecutive_healthy": min_healthy,
                "max_clip_fraction": round(max_clip, 8),
                "gpu_profile": self._fetch_gpu_util(),
            },
            "rfi": {
                "enabled": rfi.get("enabled"),
                "raw_channels_api": has_raw_channels,
                "flagged_channels": rfi.get("flagged_channels", 0),
                "total_channels": rfi.get("total_channels", 0),
                "overall_flagged_pct": round(
                    100.0 * float(rfi.get("overall_flagged_chunk_fraction", 0.0)), 2
                ),
                "static_notch_count": len(rfi.get("static_notch_bands_mhz", [])),
                "managed_notch_freqs_mhz": sorted(self.managed_notch_freqs),
                "rfi_gains": rfi_gain_info,
                "ema_reset_freqs_mhz": ema_reset_freqs,
                "notch_update": notch_info,
            },
            "waterfall_qa": qa,
            "calibration_watch": cal_info,
            "cycle_actions": cycle_actions,
            "recent_actions": list(self.action_log),
        }
        self.last_status = status_payload
        _write_json_atomic(self.status_json, status_payload)
        return status_payload


def run_daemon(
    base_url: str,
    run_dir: Optional[str],
    interval_s: float,
    status_json: Optional[str] = None,
    watch_gains: Optional[str] = None,
    auto_notch: bool = True,
    once: bool = False,
) -> None:
    print(
        f"Starting closed-loop tracker daemon (url={base_url}, run_dir={run_dir}, "
        f"interval={interval_s:.1f}s, auto_notch={auto_notch}, status_json={status_json or 'auto'})",
        flush=True,
    )
    daemon = ClosedLoopDaemon(
        base_url=base_url,
        run_dir=run_dir,
        status_json=status_json,
        watch_gains=watch_gains,
        auto_notch=auto_notch,
    )
    while True:
        try:
            st = daemon.step()
            rfi = st["rfi"]
            qa = st["waterfall_qa"]
            pl = st["pipeline"]
            acts = ",".join(st["cycle_actions"]) if st["cycle_actions"] else "steady"
            print(
                f"[{st['timestamp_utc']}] cycle={st['cycle']} "
                f"ants={len(pl['active_antennas'])}(dead={pl['dead_antennas']}) "
                f"RFI={rfi['flagged_channels']}/{rfi['total_channels']}({rfi['overall_flagged_pct']:.2f}%) "
                f"frames={qa.get('total_frames', 0)}(gaps={qa.get('seq_gaps', 0)}) "
                f"zero={qa.get('sb0_zero_pct', '?')}/{qa.get('sb1_zero_pct', '?')}% "
                f"CV={qa.get('bandpass_cv_pct', '?')}% RMS={qa.get('raw_residual_rms_pct', '?')}% "
                f"cal_gen={pl['gains_generation']} acts={acts}",
                flush=True,
            )
        except Exception as exc:
            print(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Daemon poll error: {exc}",
                file=sys.stderr,
                flush=True,
            )
        if once:
            break
        time.sleep(max(5.0, interval_s))


def print_monitor(
    base_url: str,
    status_json: Optional[str],
    run_dir: Optional[str],
    follow: bool = False,
    interval_s: float = 15.0,
) -> None:
    """Print live pipeline + closed-loop daemon summary from status JSON and Kotekan REST."""
    path = status_json or (
        os.path.join(run_dir, "self_tune_status.json")
        if run_dir
        else "/tmp/charts_self_tune_status.json"
    )
    while True:
        snap: Dict[str, Any] = {}
        if os.path.isfile(path):
            try:
                snap = json.loads(Path(path).read_text(encoding="utf-8"))
            except Exception as exc:
                snap = {"error": str(exc)}
        try:
            rfi = http_get_json(base_url, "/rfi_mitigation/status")
            trk = http_get_json(base_url, "/direct_tracker/status")
            ant = http_get_json(base_url, "/antenna_mask/status")
            live_line = (
                f"LIVE REST: ants={ant.get('active_antennas', '?')} dead={ant.get('dead_antennas', '?')} "
                f"sat={ant.get('saturated_antennas', '?')} | "
                f"RFI={rfi.get('flagged_channels', 0)}/{rfi.get('total_channels', 0)} "
                f"({100.0 * float(rfi.get('overall_flagged_chunk_fraction', 0.0)):.2f}%) "
                f"notches={len(rfi.get('static_notch_bands_mhz', []))} | "
                f"cal_gen={trk.get('gains_generation')} lst={float(trk.get('current_lst_hours', 0.0)):.3f}h"
            )
        except Exception as exc:
            live_line = f"LIVE REST ERROR: {exc}"

        print(live_line)
        if snap and "waterfall_qa" in snap:
            qa = snap["waterfall_qa"]
            drfi = snap.get("rfi", {})
            print(
                f"DAEMON SNAPSHOT ({snap.get('timestamp_utc')} cycle={snap.get('cycle')}): "
                f"frames={qa.get('total_frames')} gaps={qa.get('seq_gaps')} "
                f"dt={qa.get('dt_med_s')}s age={qa.get('last_frame_age_s')}s | "
                f"SB0/SB1 med={qa.get('sb0_median_power')}/{qa.get('sb1_median_power')} "
                f"zero={qa.get('sb0_zero_pct')}/{qa.get('sb1_zero_pct')}% | "
                f"k20={qa.get('k20_diluted_recent')} CV={qa.get('bandpass_cv_pct')}% "
                f"RMS={qa.get('raw_residual_rms_pct')}%->{qa.get('slow_rfi_cleaned_rms_pct')}% | "
                f"raw_api={drfi.get('raw_channels_api')} managed_notches={drfi.get('managed_notch_freqs_mhz')}"
            )
            recent = snap.get("recent_actions", [])[-5:]
            if recent:
                print("  Recent loop actions: " + " ; ".join(recent))
        elif os.path.isfile(path):
            print(f"DAEMON SNAPSHOT ({path}): {snap}")
        else:
            print(f"DAEMON SNAPSHOT: not found at {path}")

        if not follow:
            break
        time.sleep(max(5.0, interval_s))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Closed-loop self-tuning and verification utility for CHARTS Kotekan tracker."
    )
    parser.add_argument("--url", default="http://127.0.0.1:12048", help="Kotekan REST base URL")
    parser.add_argument("--analyze-run", metavar="DIR", help="Analyze beam_power_freq*.bin in DIR")
    parser.add_argument("--run-dir", metavar="DIR", help="Live run directory to tail in --daemon/--monitor")
    parser.add_argument("--old-gains", default=DEFAULT_OLD_GAINS, help="Baseline gains .npy path")
    parser.add_argument("--new-gains", default=DEFAULT_NEW_GAINS, help="Improved gains .npy path")
    parser.add_argument("--out-json", help="Optional JSON output path for --analyze-run")
    parser.add_argument(
        "--push-rfi-gains",
        action="store_true",
        help="Push smooth passband pre-whitening gains to /rfi_mitigation/set_gains",
    )
    parser.add_argument(
        "--reload-cal-gains",
        metavar="NPY_PATH",
        help="Hot-reload tracker complex gains via POST /direct_tracker/reload_gains",
    )
    parser.add_argument(
        "--check-antennas",
        action="store_true",
        help="Query /antenna_mask/status and /antenna_mask/channel_power",
    )
    parser.add_argument("--daemon", action="store_true", help="Run periodic closed-loop sidecar")
    parser.add_argument("--once", action="store_true", help="Run one daemon step and exit")
    parser.add_argument("--interval", type=float, default=30.0, help="Daemon poll interval (s)")
    parser.add_argument("--status-json", help="Atomic JSON telemetry output path for --daemon/--monitor")
    parser.add_argument("--watch-gains", help="Optional path or glob for new calibration .npy hot-reload")
    parser.add_argument(
        "--no-auto-notch",
        action="store_true",
        help="Disable automatic persistent-RFI narrowband static notch promotion",
    )
    parser.add_argument("--monitor", action="store_true", help="Print live pipeline + daemon QA status")
    parser.add_argument("--monitor-follow", action="store_true", help="Continuously refresh --monitor")

    args = parser.parse_args()
    effective_run_dir = args.run_dir or args.analyze_run

    if not any(
        [
            args.analyze_run,
            args.push_rfi_gains,
            args.reload_cal_gains,
            args.check_antennas,
            args.daemon,
            args.monitor,
            args.monitor_follow,
        ]
    ):
        parser.print_help()
        return 1

    if args.analyze_run:
        analyze_run(
            args.analyze_run,
            old_gains_path=args.old_gains,
            new_gains_path=args.new_gains,
            out_json=args.out_json,
        )

    if args.reload_cal_gains:
        resolved = str(Path(args.reload_cal_gains).resolve())
        reply = http_post_json(
            args.url, "/direct_tracker/reload_gains", {"gains_file": resolved}
        )
        print(f"Reloaded calibration gains: {reply}")

    if args.check_antennas:
        check_antennas(args.url)

    if args.push_rfi_gains:
        push_rfi_gains(args.url, run_dir=effective_run_dir, new_gains_path=args.new_gains)

    if args.daemon:
        run_daemon(
            args.url,
            run_dir=effective_run_dir,
            interval_s=args.interval,
            status_json=args.status_json,
            watch_gains=args.watch_gains,
            auto_notch=not args.no_auto_notch,
            once=args.once,
        )

    if args.monitor or args.monitor_follow:
        print_monitor(
            args.url,
            status_json=args.status_json,
            run_dir=effective_run_dir,
            follow=args.monitor_follow,
            interval_s=args.interval,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
