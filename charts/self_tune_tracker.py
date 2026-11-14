#!/usr/bin/env python3
"""Closed-loop self-tuning and waterfall verification tool for the CHARTS Direct Beam Tracker.

Supports:
  --analyze-run <dir>        Quantify k/20 accumulation bias, bandpass flatness (interp vs freqflt),
                             and improved RFI mitigation across captured beam_power_freq*.bin files.
  --push-rfi-gains           Compute smooth passband pre-whitening gains g(f) and push them to
                             POST /rfi_mitigation/set_gains for subbands 0 and 1.
  --reload-cal-gains <npy>   Hot-reload complex calibration gains via POST /direct_tracker/reload_gains.
  --check-antennas           Inspect GET /antenna_mask/status and GET /antenna_mask/channel_power.
  --daemon                   Periodically poll live telemetry / latest beam_power frames and refresh
                             RFI passband pre-whitening gains safely during runs.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import struct
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

SUBBAND0_START_HZ = 300.0e6
SUBBAND1_START_HZ = 400.8e6
CHANNEL_WIDTH_HZ = 300.0e3
N_CHAN_SUBBAND = 336
N_CHAN_TOTAL = 672
DEFAULT_SIGNAL_ANTS = [0, 4, 32, 36]
DEFAULT_OLD_GAINS = "/data/calibration/corr_1113/gains_corr_1113_0_4_32_36_interp_672x64.npy"
DEFAULT_NEW_GAINS = "/data/calibration/corr_1113/gains_corr_1113_0_4_32_36_freqflt_672x64.npy"


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


def compute_smooth_rfi_gains(spectrum: np.ndarray) -> np.ndarray:
    """Compute smooth normalized RFI pre-whitening gains g(f) (~sqrt(env / median(env)))."""
    valid = np.isfinite(spectrum) & (spectrum > 0.0)
    if not np.any(valid):
        return np.ones(len(spectrum), dtype=np.float32)

    clean_spec = spectrum.copy()
    global_med = float(np.median(clean_spec[valid]))
    clean_spec[~valid] = global_med

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


def push_rfi_gains(
    base_url: str, run_dir: Optional[str] = None, new_gains_path: str = DEFAULT_NEW_GAINS
) -> bool:
    """Compute smooth RFI pre-whitening gains and push to /rfi_mitigation/set_gains."""
    spectrum: Optional[np.ndarray] = None

    if run_dir:
        try:
            wf, meta = load_run_waterfalls(run_dir)
            wf_u, _ = detect_and_correct_k20_dilution(wf)
            wf_u, _ = _rescale_bandpass_if_needed(wf_u, meta, DEFAULT_OLD_GAINS, new_gains_path)
            spectrum = np.median(wf_u[-min(len(wf_u), 180) :], axis=0)
        except Exception as exc:
            print(f"[warn] Could not read run_dir {run_dir}: {exc}", file=sys.stderr)

    if spectrum is None:
        bp = http_get_json(base_url, "/direct_tracker/beam_power")
        powers = bp.get("beam_power", [])
        if not powers or len(powers[0]) < N_CHAN_SUBBAND:
            print("[info] No live non-zero beam_power spectrum available to fit RFI gains.")
            return False
        arr = np.asarray(powers[0], dtype=np.float64)
        if np.max(arr) <= 0.0:
            print("[info] Live pipeline is idling at 0 power; skipping RFI gain push.")
            return False
        spectrum = arr

    if len(spectrum) >= N_CHAN_TOTAL:
        g_sb0 = compute_smooth_rfi_gains(spectrum[:N_CHAN_SUBBAND]).tolist()
        g_sb1 = compute_smooth_rfi_gains(spectrum[N_CHAN_SUBBAND:N_CHAN_TOTAL]).tolist()
        res0 = http_post_json(
            base_url,
            "/rfi_mitigation/set_gains",
            {"freq_start_hz": SUBBAND0_START_HZ, "gains": g_sb0},
        )
        res1 = http_post_json(
            base_url,
            "/rfi_mitigation/set_gains",
            {"freq_start_hz": SUBBAND1_START_HZ, "gains": g_sb1},
        )
        print(f"Pushed SB0 RFI gains: {res0}")
        print(f"Pushed SB1 RFI gains: {res1}")
        return True

    g_sb0 = compute_smooth_rfi_gains(spectrum[:N_CHAN_SUBBAND]).tolist()
    res0 = http_post_json(
        base_url,
        "/rfi_mitigation/set_gains",
        {"freq_start_hz": SUBBAND0_START_HZ, "gains": g_sb0},
    )
    print(f"Pushed SB0 RFI gains: {res0}")
    return True


def check_antennas(base_url: str) -> None:
    st = http_get_json(base_url, "/antenna_mask/status")
    print(
        f"Antenna Mask: active={st.get('active_antennas', st.get('active_count', '?'))}/{st.get('num_elements', '?')} "
        f"auto_detect={st.get('auto_detect_enabled')}"
    )
    try:
        cp = http_get_json(base_url, "/antenna_mask/channel_power")
        for sub in cp.get("subbands", []):
            mean_p = sub.get("mean_active_power", [])
            med_p = float(np.median(mean_p)) if mean_p else 0.0
            print(
                f"  Subband {sub.get('subband')} ({sub.get('freq_start_hz', 0)/1e6:.1f} MHz): "
                f"has_channel_power={sub.get('has_channel_power')}, "
                f"active_ants={sub.get('active_physical_antennas')}, "
                f"median_active_ch_power={med_p:.3f}"
            )
    except urllib.error.HTTPError as err:
        print(f"  [warn] /antenna_mask/channel_power HTTP {err.code}")


def run_daemon(base_url: str, run_dir: Optional[str], interval_s: float) -> None:
    print(f"Starting closed-loop tracker sidecar (url={base_url}, interval={interval_s:.1f}s)")
    while True:
        try:
            rfi = http_get_json(base_url, "/rfi_mitigation/status")
            trk = http_get_json(base_url, "/direct_tracker/status")
            print(
                f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] "
                f"RFI flagged={rfi.get('flagged_channels', 0)}/{rfi.get('total_channels', 0)} "
                f"({rfi.get('mean_flagged_percent', 0.0):.2f}%), "
                f"gains_gen={trk.get('gains_generation', '?')}"
            )
            push_rfi_gains(base_url, run_dir=run_dir)
        except Exception as exc:
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Daemon poll error: {exc}", file=sys.stderr)
        time.sleep(max(5.0, interval_s))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Closed-loop self-tuning and verification utility for CHARTS Kotekan tracker."
    )
    parser.add_argument("--url", default="http://127.0.0.1:12048", help="Kotekan REST base URL")
    parser.add_argument("--analyze-run", metavar="DIR", help="Analyze beam_power_freq*.bin in DIR")
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
    parser.add_argument("--interval", type=float, default=60.0, help="Daemon poll interval (s)")

    args = parser.parse_args()

    if not any(
        [
            args.analyze_run,
            args.push_rfi_gains,
            args.reload_cal_gains,
            args.check_antennas,
            args.daemon,
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
        push_rfi_gains(args.url, run_dir=args.analyze_run, new_gains_path=args.new_gains)

    if args.daemon:
        run_daemon(args.url, run_dir=args.analyze_run, interval_s=args.interval)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
