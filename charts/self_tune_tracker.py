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
import csv
from collections import deque
import datetime
import glob
import json
import os
import re
import socket
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import warnings
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

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
MAX_AUTO_NOTCH_CHANNELS = 48
VALID_DETREND_MODES = (
    "none",
    "scalar",
    "subband_median",
    "subband_linear",
    "subband_quad",
    "global_linear",
    "global_quad",
)


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
    frames, _, _ = read_beam_power_file_with_meta(path, n_freq=n_freq, n_beams=n_beams)
    return frames


def load_run_waterfalls(run_dir: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load stitched subband 0 & 1 beam power waterfalls (T, 672) and run metadata."""
    waterfall, _, meta = load_run_waterfalls_with_times(run_dir)
    return waterfall, meta


def read_beam_power_file_with_meta(
    path: str, n_freq: int = N_CHAN_SUBBAND, n_beams: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read rawFileWrite beam power frames plus (fpga_seq_num, unix_timestamp_s)."""
    payload_floats = n_freq * n_beams
    payload_bytes = payload_floats * 4
    frames: List[np.ndarray] = []
    seqs: List[float] = []
    times: List[float] = []
    with open(path, "rb") as f:
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
            raw = f.read(payload_bytes)
            if len(raw) < payload_bytes:
                break
            arr = np.frombuffer(raw, dtype="<f4", count=n_freq).copy()
            seq, ts = _parse_frame_metadata(meta_bytes)
            frames.append(arr)
            seqs.append(float(seq) if seq is not None else np.nan)
            times.append(float(ts) if ts is not None else np.nan)
    if not frames:
        return (
            np.empty((0, n_freq), dtype=np.float32),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )
    return np.vstack(frames), np.asarray(seqs, dtype=np.float64), np.asarray(times, dtype=np.float64)


def load_run_waterfalls_with_times(
    run_dir: str,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load stitched SB0+SB1 beam power waterfalls (T, 672), UNIX timestamps (T,), and metadata."""
    rdir = Path(run_dir)
    f0_list = sorted(glob.glob(str(rdir / "beam_power_freq0_*.bin")))
    f1_list = sorted(glob.glob(str(rdir / "beam_power_freq1_*.bin")))
    if not f0_list or not f1_list:
        raise FileNotFoundError(
            f"Missing beam_power_freq0_*.bin or beam_power_freq1_*.bin in {run_dir}"
        )

    sb0_parts, seq0_parts, ts0_parts = zip(*(read_beam_power_file_with_meta(p) for p in f0_list))
    sb1_parts, seq1_parts, ts1_parts = zip(*(read_beam_power_file_with_meta(p) for p in f1_list))
    sb0 = np.vstack(sb0_parts)
    sb1 = np.vstack(sb1_parts)
    seq0 = np.concatenate(seq0_parts)
    seq1 = np.concatenate(seq1_parts)
    ts0 = np.concatenate(ts0_parts)
    ts1 = np.concatenate(ts1_parts)

    # Align by FPGA sequence when valid metadata is present and lengths differ or start offsets drift.
    valid_s0 = np.all(np.isfinite(seq0)) and len(seq0) > 0
    valid_s1 = np.all(np.isfinite(seq1)) and len(seq1) > 0
    if valid_s0 and valid_s1 and (len(seq0) != len(seq1) or not np.array_equal(seq0, seq1)):
        common, i0, i1 = np.intersect1d(seq0.astype(np.int64), seq1.astype(np.int64), return_indices=True)
        if len(common) == 0:
            raise ValueError(f"No sequence-aligned SB0/SB1 frames found in {run_dir}")
        waterfall = np.hstack([sb0[i0], sb1[i1]]).astype(np.float64)
        times = ts0[i0]
        seqs = common
    else:
        n_t = min(sb0.shape[0], sb1.shape[0])
        if n_t == 0:
            raise ValueError(f"No complete beam power frames found in {run_dir}")
        waterfall = np.hstack([sb0[:n_t], sb1[:n_t]]).astype(np.float64)
        times = np.where(np.isfinite(ts0[:n_t]), ts0[:n_t], ts1[:n_t])
        seqs = np.where(np.isfinite(seq0[:n_t]), seq0[:n_t], seq1[:n_t])

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
    finite_seq = seqs[np.isfinite(seqs)]
    seq_gaps = 0
    if len(finite_seq) >= 2:
        diffs = np.diff(finite_seq)
        seq_gaps = int(np.sum(diffs != EXPECTED_ACCUM_DSEQ))
    meta["seq_gaps"] = seq_gaps
    return waterfall, times, meta


def _robust_poly_fit_slice(
    rel_slice: np.ndarray,
    freqs_slice: np.ndarray,
    deg: int,
    outlier_sigma: float = 3.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized outlier-trimmed polynomial fit per row on shape (T, M).

    Returns (fitted_2d, c0_rel, tilt_rel_per_100mhz).
    """
    n_t, n_m = rel_slice.shape
    k_dim = deg + 1
    c0 = np.zeros(n_t, dtype=np.float64)
    tilt = np.zeros(n_t, dtype=np.float64)
    fitted = np.zeros_like(rel_slice, dtype=np.float64)
    if n_m == 0 or n_t == 0:
        return fitted, c0, tilt

    f_lo = float(freqs_slice[0])
    f_hi = float(freqs_slice[-1])
    f_span = max(f_hi - f_lo, 1.0e-3)
    f_mid = 0.5 * (f_lo + f_hi)
    u = 2.0 * (freqs_slice - f_mid) / f_span
    design = np.vstack([u**p for p in range(k_dim)]).T  # (M, K)

    finite_mask = np.isfinite(rel_slice)
    y0 = np.where(finite_mask, rel_slice, 0.0)
    with np.errstate(all="ignore"):
        row_med = np.nanmedian(np.where(finite_mask, rel_slice, np.nan), axis=1)
    row_med = np.where(np.isfinite(row_med), row_med, 0.0)

    if deg == 0:
        fitted[:] = row_med[:, None]
        return fitted, row_med, tilt

    w = finite_mask.copy()
    eye_k = np.eye(k_dim, dtype=np.float64)
    coef = np.zeros((n_t, k_dim), dtype=np.float64)

    for it in range(3):
        wf = w.astype(np.float64)
        n_valid = np.sum(w, axis=1)
        enough = n_valid >= (k_dim + 3)

        gram = np.einsum("tm,mj,mk->tjk", wf, design, design)
        rhs = np.einsum("tm,mj->tj", wf * y0, design)
        gram_safe = np.where(enough[:, None, None], gram + 1e-12 * eye_k[None, :, :], eye_k[None, :, :])
        rhs_safe = np.where(enough[:, None], rhs, 0.0)
        coef_new = np.linalg.solve(gram_safe, rhs_safe[:, :, None])[:, :, 0]
        coef = np.where(enough[:, None], coef_new, coef)
        if not np.all(enough):
            coef[~enough, 0] = row_med[~enough]
            coef[~enough, 1:] = 0.0

        fitted_cur = coef @ design.T
        if it < 2:
            with np.errstate(all="ignore"):
                res = np.where(finite_mask, rel_slice - fitted_cur, np.nan)
                res_med = np.nanmedian(res, axis=1, keepdims=True)
                mad = 1.4826 * np.nanmedian(np.abs(res - res_med), axis=1, keepdims=True)
            mad = np.where(np.isfinite(mad), np.maximum(mad, 0.003), 0.005)
            w_next = finite_mask & (np.abs(res - np.where(np.isfinite(res_med), res_med, 0.0)) <= outlier_sigma * mad)
            # Only tighten weights on rows that still have enough points
            keep_update = np.sum(w_next, axis=1) >= (k_dim + 3)
            w = np.where(keep_update[:, None], w_next, w)
        else:
            fitted = fitted_cur

    c0 = row_med
    tilt = coef[:, 1] * (200.0 / f_span)  # fractional change per 100 MHz
    return fitted, c0, tilt


def detrend_dynamic_spectrum(
    rel_spec: np.ndarray,
    freqs_mhz: Optional[np.ndarray] = None,
    mode: str = "subband_linear",
    sb_split: int = N_CHAN_SUBBAND,
    outlier_sigma: float = 3.5,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Remove low-order foreground / thermal spectral tilt from relative beam power (ΔP/P).

    Supported modes:
      - none            : identity (no per-frame subtraction)
      - scalar          : single whole-band per-frame median subtracted
      - subband_median  : independent per-frame median per 100.8 MHz subband
      - subband_linear  : per-subband robust deg-1 spectral tilt + offset (default)
      - subband_quad    : per-subband robust deg-2 spectral curvature + tilt + offset
      - global_linear   : whole-band robust deg-1 spectral tilt + offset
      - global_quad     : whole-band robust deg-2 polynomial
    """
    if mode not in VALID_DETREND_MODES:
        raise ValueError(f"Unsupported detrend mode '{mode}', expected one of {VALID_DETREND_MODES}")

    arr = np.asarray(rel_spec, dtype=np.float64)
    squeeze_1d = arr.ndim == 1
    if squeeze_1d:
        arr = arr[None, :]
    n_t, n_f = arr.shape
    if freqs_mhz is None or len(freqs_mhz) != n_f:
        if n_f == N_CHAN_TOTAL and sb_split == N_CHAN_SUBBAND:
            f0 = SUBBAND0_START_HZ * 1e-6 + np.arange(N_CHAN_SUBBAND) * (CHANNEL_WIDTH_HZ * 1e-6)
            f1 = SUBBAND1_START_HZ * 1e-6 + np.arange(N_CHAN_SUBBAND) * (CHANNEL_WIDTH_HZ * 1e-6)
            freqs = np.concatenate([f0, f1])
        else:
            freqs = SUBBAND0_START_HZ * 1e-6 + np.arange(n_f) * (CHANNEL_WIDTH_HZ * 1e-6)
    else:
        freqs = np.asarray(freqs_mhz, dtype=np.float64)

    fitted = np.zeros_like(arr, dtype=np.float64)
    deg = 0
    if mode.endswith("_linear"):
        deg = 1
    elif mode.endswith("_quad"):
        deg = 2

    split = max(0, min(sb_split, n_f))
    has_two_sb = split > 0 and split < n_f

    # Always extract SB0/SB1 common-mode & linear tilt diagnostics for telemetry regardless of mode
    _, sb0_c, sb0_t = (
        _robust_poly_fit_slice(arr[:, :split], freqs[:split], 1, outlier_sigma)
        if has_two_sb
        else _robust_poly_fit_slice(arr, freqs, 1, outlier_sigma)
    )
    if has_two_sb:
        _, sb1_c, sb1_t = _robust_poly_fit_slice(arr[:, split:], freqs[split:], 1, outlier_sigma)
    else:
        sb1_c = np.zeros(n_t, dtype=np.float64)
        sb1_t = np.zeros(n_t, dtype=np.float64)

    if mode == "none":
        pass
    elif mode in ("scalar", "global_linear", "global_quad"):
        fitted, _, _ = _robust_poly_fit_slice(arr, freqs, deg, outlier_sigma)
    else:
        # subband_median, subband_linear, subband_quad
        if has_two_sb:
            fit0, _, _ = _robust_poly_fit_slice(arr[:, :split], freqs[:split], deg, outlier_sigma)
            fit1, _, _ = _robust_poly_fit_slice(arr[:, split:], freqs[split:], deg, outlier_sigma)
            fitted[:, :split] = fit0
            fitted[:, split:] = fit1
        else:
            fitted, _, _ = _robust_poly_fit_slice(arr, freqs, deg, outlier_sigma)

    detrended = np.where(np.isfinite(arr), arr - fitted, np.nan)
    fin_vals = detrended[np.isfinite(detrended)]
    if len(fin_vals) > 0:
        rms_pct = 100.0 * float(np.std(fin_vals))
        abs_pct = 100.0 * np.abs(fin_vals)
        sat_1 = 100.0 * float(np.mean(abs_pct > 1.0))
        sat_2 = 100.0 * float(np.mean(abs_pct > 2.0))
        sat_3 = 100.0 * float(np.mean(abs_pct > 3.0))
    else:
        rms_pct = sat_1 = sat_2 = sat_3 = 0.0

    diag: Dict[str, Any] = {
        "mode": mode,
        "detrended_rms_pct": round(rms_pct, 3),
        "sat_1pct": round(sat_1, 2),
        "sat_2pct": round(sat_2, 2),
        "sat_3pct": round(sat_3, 2),
        "sb0_common_pct": 100.0 * sb0_c,
        "sb1_common_pct": 100.0 * sb1_c,
        "sb0_tilt_pct_per_100mhz": 100.0 * sb0_t,
        "sb1_tilt_pct_per_100mhz": 100.0 * sb1_t,
        "fitted_baseline": fitted[0] if squeeze_1d else fitted,
    }
    return (detrended[0] if squeeze_1d else detrended), diag


def _downsample_waterfall_for_plot(
    arr: np.ndarray, times: Optional[np.ndarray] = None, max_rows: int = 900
) -> Tuple[np.ndarray, np.ndarray]:
    """Bin (T, F) array with NaNs into <= max_rows along time for clean waterfall plotting."""
    n_t, _ = arr.shape
    t_vals = (
        np.asarray(times, dtype=np.float64)
        if (times is not None and len(times) == n_t)
        else np.arange(n_t, dtype=np.float64)
    )
    if n_t <= max_rows or max_rows <= 0:
        return arr, t_vals
    step = int(np.ceil(n_t / float(max_rows)))
    n_bins = n_t // step
    trimmed = arr[: n_bins * step].reshape(n_bins, step, arr.shape[1])
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", category=RuntimeWarning)
        binned = np.nanmean(trimmed, axis=1)
        t_binned = np.nanmean(t_vals[: n_bins * step].reshape(n_bins, step), axis=1)
    return binned, t_binned


def find_active_transits(
    t0_unix: float, t1_unix: float, min_alt_deg: float = 10.0
) -> List[Tuple[datetime.datetime, str, float]]:
    """Find bright radio sources and planets reaching peak transit inside (t0_unix, t1_unix)."""
    if not (np.isfinite(t0_unix) and np.isfinite(t1_unix)) or t1_unix <= t0_unix or t0_unix < 1e9:
        return []
    try:
        from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_body
        from astropy.time import Time
        import astropy.units as u
    except ImportError:
        return []

    try:
        from charts_constants import CHARTS_ANTENNA_REF_LOCATION

        lat, lon, alt = CHARTS_ANTENNA_REF_LOCATION
    except ImportError:
        lat, lon, alt = (-33.4212167, -70.8637972, 458.0)

    loc = EarthLocation(lat=lat * u.deg, lon=lon * u.deg, height=alt * u.m)
    n_grid = min(2000, max(300, int((t1_unix - t0_unix) / 10.0)))
    times = Time(np.linspace(t0_unix, t1_unix, n_grid), format="unix")
    frame = AltAz(obstime=times, location=loc)

    candidates: Dict[str, Any] = {}
    try:
        from charts_utils.cal.catalog import RADIO_CALIBRATORS

        for name, star in RADIO_CALIBRATORS.items():
            pretty = name.upper() if name.startswith("3c") else name.replace("_", " ").title()
            candidates[pretty] = SkyCoord(
                ra=star.ra.hours * u.hourangle, dec=star.dec.degrees * u.deg
            )
    except ImportError:
        pass

    extra = {
        "Cygnus A": ("19h59m28.3s", "+40d44m02s"),
        "Cassiopeia A": ("23h23m24.0s", "+58d48m54s"),
        "Taurus A": ("05h34m31.9s", "+22d00m52s"),
        "Virgo A": ("12h30m49.4s", "+12d23m28s"),
        "Hydra A": ("09h18m05.7s", "-12d05m44s"),
        "Centaurus A": ("13h25m27.6s", "-43d01m09s"),
        "Pictor A": ("05h19m49.7s", "-45d46m44s"),
        "Fornax A": ("03h22m41.7s", "-37d12m30s"),
        "Vela": ("08h35m20.6s", "-45d10m35s"),
        "PKS 2032-35": ("20h35m47.6s", "-34d54m08s"),
        "PKS 2104-25": ("21h07m25.7s", "-25d25m46s"),
        "PKS 2153-69": ("21h57m06.0s", "-69d41m24s"),
        "PKS 2250-41": ("22h53m03.2s", "-40d57m46s"),
    }
    for label, (ra_s, dec_s) in extra.items():
        if label not in candidates:
            candidates[label] = SkyCoord(ra_s, dec_s)

    active: List[Tuple[datetime.datetime, str, float]] = []
    for label, sc in candidates.items():
        aa = sc.transform_to(frame)
        alts = aa.alt.deg
        imax = int(np.argmax(alts))
        if 0 < imax < len(alts) - 1 and alts[imax] > min_alt_deg:
            t_pk = times[imax].to_datetime(datetime.timezone.utc)
            active.append((t_pk, label, float(alts[imax])))

    for body in ("sun", "moon", "jupiter", "saturn"):
        try:
            sc = get_body(body, times, loc)
            aa = sc.transform_to(frame)
            alts = aa.alt.deg
            imax = int(np.argmax(alts))
            if 0 < imax < len(alts) - 1 and alts[imax] > min_alt_deg:
                t_pk = times[imax].to_datetime(datetime.timezone.utc)
                active.append((t_pk, body.title(), float(alts[imax])))
        except Exception:
            continue

    active.sort(key=lambda x: x[0])
    return active


def render_waterfall_figure(
    wf_clean: np.ndarray,
    freqs_mhz: np.ndarray,
    out_png: str,
    times: Optional[np.ndarray] = None,
    detrend_mode: str = "subband_linear",
    clim_pct: float = 2.5,
    title: Optional[str] = None,
    telemetry_rows: Optional[List[Dict[str, Any]]] = None,
    wf_raw: Optional[np.ndarray] = None,
    transits: Optional[Sequence[Tuple[Any, str, float]]] = None,
) -> str:
    """Render plot_waterfall-style summary (Time X, Freq Y) with active transit markers & tilt detrend."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.dates as mdates
        import matplotlib.patheffects as pe
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for --plot-waterfall/--live-waterfall "
            "(available in /home/fpalma/charts-utils/.venv/bin/python)"
        ) from exc

    n_t, n_f = wf_clean.shape
    wf_rec = wf_raw if (wf_raw is not None and wf_raw.shape == wf_clean.shape) else wf_clean
    wf_rec_u, _ = detect_and_correct_k20_dilution(wf_rec)

    max_cols = 1500
    bin_t = max(1, int(np.ceil(n_t / float(max_cols))))
    n_use = max(1, (n_t // bin_t) * bin_t)
    t_arr = (
        np.asarray(times, dtype=np.float64)
        if (times is not None and len(times) == n_t)
        else np.arange(n_t, dtype=np.float64)
    )
    ts_b = t_arr[:n_use].reshape(-1, bin_t).mean(axis=1)

    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", category=RuntimeWarning)
        wf_rec_pos = np.where(wf_rec[:n_use] > 0.0, wf_rec[:n_use], np.nan)
        wf_rec_bin = np.nanmedian(wf_rec_pos.reshape(-1, bin_t, n_f), axis=1)
        wf_db = 10.0 * np.log10(np.where(wf_rec_bin > 0.0, wf_rec_bin, np.nan))

        ch_med_rec = np.nanmedian(np.where(wf_rec_u > 0.0, wf_rec_u, np.nan), axis=0)
        dyn_raw_pct = np.full((n_t, n_f), np.nan, dtype=np.float64)
        ok_rec = np.isfinite(ch_med_rec) & (ch_med_rec > 0.0)
        dyn_raw_pct[:, ok_rec] = 100.0 * (
            np.where(wf_rec_u[:, ok_rec] > 0.0, wf_rec_u[:, ok_rec], np.nan)
            / ch_med_rec[ok_rec][None, :]
            - 1.0
        )
        dyn_raw_bin = np.nanmedian(dyn_raw_pct[:n_use].reshape(-1, bin_t, n_f), axis=1)

        wf_clean_pos = np.where(np.isfinite(wf_clean) & (wf_clean > 0.0), wf_clean, np.nan)
        ch_med_c = np.nanmedian(wf_clean_pos, axis=0)
        ok_c = np.isfinite(ch_med_c) & (ch_med_c > 0.0)
        rel_clean = np.full((n_t, n_f), np.nan, dtype=np.float64)
        rel_clean[:, ok_c] = wf_clean_pos[:, ok_c] / ch_med_c[ok_c][None, :] - 1.0

        rel_det, diag_det = detrend_dynamic_spectrum(rel_clean, freqs_mhz, mode=detrend_mode)
        dyn_det_pct = 100.0 * rel_det
        dyn_det_bin = np.nanmedian(dyn_det_pct[:n_use].reshape(-1, bin_t, n_f), axis=1)

        gpu_zero_pct_ch = 100.0 * np.mean((wf_rec <= 0.0) | (~np.isfinite(wf_rec)), axis=0)
        tot_blank_pct_ch = 100.0 * np.mean((~np.isfinite(wf_clean)) | (wf_clean <= 0.0), axis=0)
        rms_raw_ch = np.nanstd(
            dyn_raw_pct - np.nanmedian(dyn_raw_pct, axis=1, keepdims=True), axis=0
        )
        rms_det_ch = np.nanstd(dyn_det_pct, axis=0)

    has_utc = len(ts_b) > 0 and np.all(np.isfinite(ts_b)) and np.nanmin(ts_b) > 1e9
    if has_utc:
        t_dt = [datetime.datetime.fromtimestamp(float(t), datetime.timezone.utc) for t in ts_b]
        t0_num = float(mdates.date2num(t_dt[0]))
        t1_num = float(mdates.date2num(t_dt[-1]))
        x_label = f"Time (UTC, {t_dt[0].strftime('%Y-%m-%d')} to {t_dt[-1].strftime('%Y-%m-%d')})"
        if transits is None:
            transits = find_active_transits(float(t_arr[0]), float(t_arr[-1]))
    else:
        t0_num = float(ts_b[0]) if len(ts_b) else 0.0
        t1_num = float(ts_b[-1]) if len(ts_b) else 1.0
        x_label = "Elapsed Time / Frame (s)"
        transits = transits or []

    if t1_num <= t0_num:
        t1_num = t0_num + 1.0
    extent = [t0_num, t1_num, float(freqs_mhz[0]), float(freqs_mhz[-1])]

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10.5, "axes.labelsize": 9.5})
    fig = plt.figure(figsize=(13.5, 10.5), dpi=112, facecolor="white")
    dense_transits = len(transits) > 5
    top_margin = 0.824 if dense_transits else (0.842 if transits else 0.905)
    gs = fig.add_gridspec(
        3,
        2,
        width_ratios=[4.6, 1.15],
        hspace=0.28,
        wspace=0.08,
        left=0.065,
        right=0.96,
        top=top_margin,
        bottom=0.058,
    )

    cmap_pwr = plt.get_cmap("inferno").copy()
    cmap_pwr.set_bad("#404040")
    cmap_div = plt.get_cmap("coolwarm").copy()
    cmap_div.set_bad("#404040")

    # Panel 1: Recorded Beam Power (dB) + Median Bandpass
    ax0 = fig.add_subplot(gs[0, 0])
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", category=RuntimeWarning)
        wf_clean_bin = np.nanmedian(wf_clean_pos[:n_use].reshape(-1, bin_t, n_f), axis=1)
        wf_clean_db = 10.0 * np.log10(np.where(wf_clean_bin > 0.0, wf_clean_bin, np.nan))
    finite_unflagged_db = wf_clean_db[np.isfinite(wf_clean_db)]
    finite_db = finite_unflagged_db if len(finite_unflagged_db) >= 2 else wf_db[np.isfinite(wf_db)]
    if len(finite_db) >= 2:
        vlo, vhi = np.percentile(finite_db, [5.0, 95.0])
        if vhi <= vlo:
            vlo, vhi = float(finite_db.min()) - 1.0, float(finite_db.max()) + 1.0
    else:
        vlo, vhi = -10.0, 10.0
    im0 = ax0.imshow(
        wf_db.T,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap=cmap_pwr,
        vmin=vlo,
        vmax=vhi,
        interpolation="nearest",
    )
    ax0.axhline(400.65, color="white", linestyle=":", linewidth=0.8, alpha=0.65)
    title_pad = 58 if dense_transits else (46 if transits else 8)
    ax0.set_title(
        "Recorded Beam Power $10\\log_{10} P$ [dB]  (gray = GPU RFI-blanked)",
        fontweight="bold",
        loc="left",
        pad=title_pad,
    )
    ax0.set_ylabel("Frequency [MHz]")
    cb0 = fig.colorbar(im0, ax=ax0, pad=0.012, aspect=26)
    cb0.set_label("Power [dB]")

    ax0r = fig.add_subplot(gs[0, 1], sharey=ax0)
    ok_med = np.isfinite(ch_med_rec) & (ch_med_rec > 0.0)
    if np.any(ok_med):
        ax0r.plot(ch_med_rec[ok_med], freqs_mhz[ok_med], color="#1f4e79", linewidth=0.8)
        ax0r.set_xlim(0.0, max(1e-3, float(np.nanpercentile(ch_med_rec[ok_med], 99.5) * 1.1)))
    ax0r.set_title("Median Bandpass", fontweight="bold", fontsize=9.5)
    ax0r.set_xlabel("Power [arb.]")
    ax0r.grid(True, linestyle="--", alpha=0.35)
    plt.setp(ax0r.get_yticklabels(), visible=False)

    # Panel 2: Un-detrended Bandpass-Normalized Dynamic Spectrum (%) + Blanked %
    norm_clim = max(3.0, clim_pct)
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0, sharey=ax0)
    im1 = ax1.imshow(
        dyn_raw_bin.T,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap=cmap_div,
        vmin=-norm_clim,
        vmax=norm_clim,
        interpolation="nearest",
    )
    ax1.axhline(400.65, color="#333333", linestyle=":", linewidth=0.8, alpha=0.55)
    ax1.set_title(
        "Bandpass-Normalized Dynamic Spectrum $\\Delta P / P_{\\mathrm{med}}(f)$ [%]  (raw thermal/sky drift)",
        fontweight="bold",
        loc="left",
    )
    ax1.set_ylabel("Frequency [MHz]")
    cb1 = fig.colorbar(im1, ax=ax1, pad=0.012, aspect=26)
    cb1.set_label("$\\Delta P / P_{\\mathrm{med}}$ [%]")

    ax1r = fig.add_subplot(gs[1, 1], sharey=ax0)
    if wf_raw is not None:
        ax1r.barh(freqs_mhz, tot_blank_pct_ch, height=0.32, color="#0d9488", alpha=0.65, linewidth=0, label="+Shldr")
    ax1r.barh(freqs_mhz, gpu_zero_pct_ch, height=0.32, color="#475569", alpha=0.85, linewidth=0, label="GPU")
    ax1r.set_title("RFI Blanked %", fontweight="bold", fontsize=9.5)
    ax1r.set_xlabel("Blanked [%]")
    ax1r.set_xlim(0.0, 105.0)
    ax1r.grid(True, linestyle="--", alpha=0.35)
    if wf_raw is not None:
        ax1r.legend(loc="upper right", fontsize=7.0, framealpha=0.8)
    plt.setp(ax1r.get_yticklabels(), visible=False)

    # Panel 3: Slow-RFI Masked + Spectral-Tilt Detrended Residuals (%) + Channel RMS %
    ax2 = fig.add_subplot(gs[2, 0], sharex=ax0, sharey=ax0)
    im2 = ax2.imshow(
        dyn_det_bin.T,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap=cmap_div,
        vmin=-clim_pct,
        vmax=clim_pct,
        interpolation="nearest",
    )
    ax2.axhline(400.65, color="#333333", linestyle=":", linewidth=0.8, alpha=0.55)
    det_fin = dyn_det_pct[np.isfinite(dyn_det_pct)]
    sat_clim = float(100.0 * np.mean(np.abs(det_fin) > clim_pct)) if len(det_fin) else 0.0
    ax2.set_title(
        f"Slow-RFI + `{detrend_mode}` Detrended [%] (±{clim_pct:.1f}%) — "
        f"RMS={diag_det['detrended_rms_pct']:.2f}%, Sat>{clim_pct:.1f}%={sat_clim:.1f}%",
        fontweight="bold",
        loc="left",
    )
    ax2.set_ylabel("Frequency [MHz]")
    ax2.set_xlabel(x_label)
    cb2 = fig.colorbar(im2, ax=ax2, pad=0.012, aspect=26)
    cb2.set_label("Residual [%]")

    ax2r = fig.add_subplot(gs[2, 1], sharey=ax0)
    ax2r.plot(np.clip(rms_raw_ch, 0.0, 12.0), freqs_mhz, color="#9e9e9e", linewidth=0.7, alpha=0.85, label="Raw")
    ax2r.plot(np.clip(rms_det_ch, 0.0, 12.0), freqs_mhz, color="#0d6efd", linewidth=0.85, label=detrend_mode)
    ax2r.set_title("Channel RMS %", fontweight="bold", fontsize=9.5)
    ax2r.set_xlabel("RMS [%]")
    ax2r.set_xlim(0.0, 6.0)
    ax2r.grid(True, linestyle="--", alpha=0.35)
    ax2r.legend(loc="upper right", fontsize=7.0, framealpha=0.8)
    plt.setp(ax2r.get_yticklabels(), visible=False)

    # Vertical dashed target transit lines across all 3 waterfalls + staggered top pills
    row_offsets = [1.010, 1.064, 1.118, 1.172] if dense_transits else [1.012, 1.072, 1.132]
    placed_rows: List[List[float]] = [[] for _ in row_offsets]
    pill_fontsize = 6.9 if dense_transits else 7.3
    for item in transits:
        t_pk, label, alt_deg = item[0], str(item[1]), float(item[2])
        if isinstance(t_pk, datetime.datetime):
            tx = float(mdates.date2num(t_pk))
            t_lbl = t_pk.strftime("%H:%M")
        else:
            tx = float(t_pk)
            if has_utc and tx > 1e9:
                dt_pk = datetime.datetime.fromtimestamp(tx, datetime.timezone.utc)
                tx = float(mdates.date2num(dt_pk))
                t_lbl = dt_pk.strftime("%H:%M")
            else:
                t_lbl = f"{tx:.0f}s"
        if tx < t0_num or tx > t1_num:
            continue

        halo = [pe.Stroke(linewidth=2.5, foreground="white", alpha=0.85), pe.Normal()]
        ax0.axvline(tx, color="#0e7490", linestyle="--", linewidth=1.2, alpha=0.95, zorder=4, path_effects=halo)
        ax1.axvline(tx, color="#1f2937", linestyle="--", linewidth=1.1, alpha=0.85, zorder=4, path_effects=halo)
        ax2.axvline(tx, color="#1f2937", linestyle="--", linewidth=1.1, alpha=0.85, zorder=4, path_effects=halo)

        rel_x = (tx - t0_num) / (t1_num - t0_num)
        ha = "left" if rel_x < 0.06 else ("right" if rel_x > 0.94 else "center")
        chosen_row = None
        for r_idx in range(len(row_offsets)):
            if all(abs(rel_x - px) >= 0.22 for px in placed_rows[r_idx]):
                chosen_row = r_idx
                break
        if chosen_row is None:
            chosen_row = max(
                range(len(row_offsets)),
                key=lambda r_idx: min((abs(rel_x - px) for px in placed_rows[r_idx]), default=1.0),
            )
        placed_rows[chosen_row].append(rel_x)
        ax0.text(
            tx,
            row_offsets[chosen_row],
            f"{label} ({t_lbl}, {alt_deg:.0f}°)",
            transform=ax0.get_xaxis_transform(),
            ha=ha,
            va="bottom",
            fontsize=pill_fontsize,
            fontweight="bold",
            color="#0b3c5d",
            bbox=dict(boxstyle="round,pad=0.14", fc="#e8f6ff", ec="#38bdf8", lw=0.6, alpha=0.92),
            clip_on=False,
        )

    for ax in (ax0, ax1, ax2):
        ax.set_xlim(t0_num, t1_num)
        ax.set_ylim(float(freqs_mhz[0]), float(freqs_mhz[-1]))
        if has_utc:
            ax.xaxis_date()
            span_s = float(t_arr[-1] - t_arr[0])
            if span_s >= 5400.0:
                ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 30]))
            elif span_s >= 1500.0:
                ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=range(0, 60, 10)))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.setp(ax0.get_xticklabels(), visible=False)
    plt.setp(ax1.get_xticklabels(), visible=False)

    env_note = ""
    if telemetry_rows:
        temps = [r["ambient_temp_c"] for r in telemetry_rows if "ambient_temp_c" in r and r["ambient_temp_c"] is not None]
        lna_is = [r["lna_i"] for r in telemetry_rows if "lna_i" in r and r["lna_i"] is not None]
        bits = []
        if temps:
            bits.append(f"Tamb={min(temps):.1f}-{max(temps):.1f}C")
        if lna_is:
            bits.append(f"ILNA={min(lna_is):.3f}-{max(lna_is):.3f}A")
        if bits:
            env_note = " | " + " ".join(bits)

    sb0_z = float(np.mean(gpu_zero_pct_ch[: min(N_CHAN_SUBBAND, n_f)]))
    sb1_z = float(np.mean(gpu_zero_pct_ch[N_CHAN_SUBBAND:])) if n_f > N_CHAN_SUBBAND else 0.0
    dt_est = float(np.median(np.diff(t_arr))) if len(t_arr) >= 2 else 1.024
    base_title = title or "CHARTS Zenith Dynamic Spectrum Summary"
    if has_utc:
        t0_s = datetime.datetime.fromtimestamp(float(t_arr[0]), datetime.timezone.utc).strftime("%H:%M:%S")
        t1_s = datetime.datetime.fromtimestamp(float(t_arr[-1]), datetime.timezone.utc).strftime("%H:%M:%S")
        time_note = f"{t0_s}–{t1_s} UTC ({n_t} frames)"
    else:
        time_note = f"{n_t} frames"
    fig.suptitle(
        f"{base_title}: {time_note}\n"
        f"dt={dt_est:.3f}s | SB0/SB1 blanked={sb0_z:.2f}%/{sb1_z:.2f}% | "
        f"{bin_t * dt_est:.1f}s time bin | dashed = active target peak transits{env_note}",
        fontsize=10.6,
        fontweight="bold",
        y=0.978,
    )

    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = str(out_path) + ".tmp.png"
    fig.savefig(tmp_path, dpi=112)
    plt.close(fig)
    os.replace(tmp_path, str(out_path))
    return str(out_path)


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


def fit_standing_wave_ripple(
    clean_spec: np.ndarray,
    env: np.ndarray,
    valid_mask: np.ndarray,
    channel_bw_mhz: float = CHANNEL_WIDTH_HZ * 1.0e-6,
    tau_range_us: Tuple[float, float] = (0.85, 1.15),
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Fit dominant coax standing-wave sinusoid (e.g. 1.029 MHz / 0.972 us) on clean residuals."""
    n_ch = len(clean_spec)
    ones = np.ones(n_ch, dtype=np.float64)
    if n_ch < 32 or np.sum(valid_mask) < 24:
        return ones, {"applied": False}

    ratio = clean_spec / np.maximum(env, 1e-12) - 1.0
    v = np.asarray(valid_mask, dtype=bool) & np.isfinite(ratio)
    if np.sum(v) < 24:
        return ones, {"applied": False}

    med = float(np.median(ratio[v]))
    mad = max(float(1.4826 * np.median(np.abs(ratio[v] - med))), 1e-4)
    inliers = v & (np.abs(ratio - med) <= 3.5 * mad)
    if np.sum(inliers) < 24:
        return ones, {"applied": False}

    f_mhz = np.arange(n_ch, dtype=np.float64) * channel_bw_mhz
    f_in = f_mhz[inliers]
    y_in = ratio[inliers]
    var0 = float(np.var(y_in))
    if var0 <= 1e-8:
        return ones, {"applied": False}

    best_sse = np.inf
    best_tau = 0.0
    best_coef = np.zeros(3, dtype=np.float64)

    def _eval_grid(taus: np.ndarray) -> None:
        nonlocal best_sse, best_tau, best_coef
        for tau in taus:
            ang = 2.0 * np.pi * tau * f_in
            design = np.column_stack((np.ones_like(f_in), np.cos(ang), np.sin(ang)))
            coef, _, _, _ = np.linalg.lstsq(design, y_in, rcond=None)
            res = y_in - design @ coef
            sse = float(np.dot(res, res))
            if sse < best_sse:
                best_sse = sse
                best_tau = float(tau)
                best_coef = coef

    _eval_grid(np.linspace(tau_range_us[0], tau_range_us[1], 121))
    _eval_grid(np.linspace(max(tau_range_us[0], best_tau - 0.006), min(tau_range_us[1], best_tau + 0.006), 49))

    amp = float(np.hypot(best_coef[1], best_coef[2]))
    var_ratio = (best_sse / max(len(y_in), 1)) / var0
    if amp < 0.01 or amp > 0.35 or var_ratio > 0.90:
        return ones, {"applied": False, "amp": amp, "tau_us": best_tau, "var_ratio": var_ratio}

    full_ang = 2.0 * np.pi * best_tau * f_mhz
    ripple = np.clip(
        1.0 + best_coef[1] * np.cos(full_ang) + best_coef[2] * np.sin(full_ang),
        0.65,
        1.35,
    )
    return ripple, {
        "applied": True,
        "tau_us": round(best_tau, 5),
        "period_mhz": round(1.0 / max(best_tau, 1e-9), 4),
        "amp": round(amp, 4),
        "var_ratio": round(var_ratio, 4),
    }


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
    ripple, _ = fit_standing_wave_ripple(clean_spec, env, valid)
    env_whitened = env * ripple
    ref = float(np.median(env_whitened))
    if ref <= 0.0:
        return np.ones(len(spectrum), dtype=np.float32)

    gains = np.sqrt(np.clip(env_whitened / ref, 0.0625, 16.0))
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


def _compute_freq_mad_block(
    block: np.ndarray, freq_win_half: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute outlier-rejected local passband median, refined MAD, and raw dev on (T, F_sb)."""
    n_t, n_f = block.shape
    hw = min(freq_win_half, max(1, (n_f - 1) // 2))
    padded = np.pad(block, ((0, 0), (hw, hw)), mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(padded, 2 * hw + 1, axis=1)
    w_med0 = np.median(win, axis=2)
    w_dev0 = np.abs(win - w_med0[:, :, None])
    w_mad0 = np.maximum(1.4826 * np.median(w_dev0, axis=2), 1e-9)
    inlier_mask = w_dev0 < (3.5 * w_mad0[:, :, None])
    inlier_vals = np.where(inlier_mask, win, np.nan)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", category=RuntimeWarning)
        w_med_in = np.nanmedian(inlier_vals, axis=2)
        n_in = np.sum(inlier_mask, axis=2)
        w_med = np.where(n_in >= 3, w_med_in, w_med0)
        inlier_dev = np.where(inlier_mask, np.abs(win - w_med[:, :, None]), np.nan)
        refined_mad = 1.4826 * np.nanmedian(inlier_dev, axis=2)
    local_mad = np.where((n_in >= 3) & np.isfinite(refined_mad), refined_mad, w_mad0)
    return block - w_med, w_med, local_mad


def simulate_improved_rfi_excision(
    waterfall: np.ndarray,
    ema_alpha: float = 0.02,
    burst_sigma_thresh: float = 5.0,
    mad_sigma_thresh: float = 6.0,
    freq_win_half: int = 8,
    edge_ignore: int = 0,
    enable_freq_shoulder: bool = True,
    shoulder_half_width: int = 2,
    shoulder_sigma_thresh: float = 2.5,
    sb_split: int = N_CHAN_SUBBAND,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate GPU RFI excision: pre-whitened subband MAD, persistent EMA, & conditional shoulders."""
    n_t, n_f = waterfall.shape
    chan_med = np.median(waterfall, axis=0)

    # 1. Subband-isolated frequency-domain outlier-rejected MAD passband check
    has_split = 0 < sb_split < n_f
    if has_split:
        g0 = compute_smooth_rfi_gains(chan_med[:sb_split]).astype(np.float64)
        g1 = compute_smooth_rfi_gains(chan_med[sb_split:]).astype(np.float64)
        rfi_gains = np.concatenate([g0, g1])
        row_norm = waterfall / np.maximum(rfi_gains[None, :] ** 2, 1e-6)
        exc0, med0, mad0 = _compute_freq_mad_block(row_norm[:, :sb_split], freq_win_half)
        exc1, med1, mad1 = _compute_freq_mad_block(row_norm[:, sb_split:], freq_win_half)
        freq_excess = np.hstack([exc0, exc1])
        w_med = np.hstack([med0, med1])
        local_mad = np.hstack([mad0, mad1])
    else:
        rfi_gains = compute_smooth_rfi_gains(chan_med).astype(np.float64)
        row_norm = waterfall / np.maximum(rfi_gains[None, :] ** 2, 1e-6)
        freq_excess, w_med, local_mad = _compute_freq_mad_block(row_norm, freq_win_half)

    freq_sigma = np.maximum(local_mad, 0.05 * np.maximum(w_med, 1e-3))
    freq_flagged = (freq_excess / freq_sigma) >= mad_sigma_thresh
    shoulder_sigma = np.maximum(local_mad, 0.04 * np.maximum(w_med, 1e-3))
    freq_shoulder_z = freq_excess / shoulder_sigma
    if edge_ignore > 0:
        freq_flagged[:, :edge_ignore] = False
        freq_flagged[:, -edge_ignore:] = False
        if has_split:
            freq_flagged[:, max(0, sb_split - edge_ignore) : min(n_f, sb_split + edge_ignore)] = False

    def _dilate_1d_subband(core: np.ndarray, hw: int) -> np.ndarray:
        out = np.zeros_like(core, dtype=bool)
        if hw <= 0:
            return out
        slices = ((0, sb_split), (sb_split, n_f)) if has_split else ((0, n_f),)
        for lo, hi in slices:
            sub = core[lo:hi]
            m = sub.shape[0]
            acc = np.zeros(m, dtype=bool)
            for d in range(1, min(hw, m - 1) + 1):
                acc[d:] |= sub[:-d]
                acc[:-d] |= sub[d:]
            out[lo:hi] = acc & (~sub)
        return out

    # 2. Temporal persistent EMA burst check + conditional frequency-shoulder dilation
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
        time_z = np.where(has_ema, (p - ema_p) / np.maximum(t_sigma, 1e-9), 0.0)
        time_flag = has_ema & (time_z >= burst_sigma_thresh)
        core_flag = freq_flagged[t] | time_flag
        frame_flag = core_flag.copy()

        if enable_freq_shoulder and shoulder_half_width > 0:
            adj_core = _dilate_1d_subband(core_flag, shoulder_half_width)
            adj_time = _dilate_1d_subband(time_flag, shoulder_half_width)
            sh_flag = adj_core & (
                (freq_shoulder_z[t] >= shoulder_sigma_thresh)
                | (adj_time & (time_z >= shoulder_sigma_thresh))
            )
            frame_flag |= sh_flag

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


def _channel_freqs_mhz(n_f: int = N_CHAN_TOTAL) -> np.ndarray:
    if n_f <= N_CHAN_SUBBAND:
        return (SUBBAND0_START_HZ + np.arange(n_f) * CHANNEL_WIDTH_HZ) * 1.0e-6
    f0 = (SUBBAND0_START_HZ + np.arange(N_CHAN_SUBBAND) * CHANNEL_WIDTH_HZ) * 1.0e-6
    f1 = (SUBBAND1_START_HZ + np.arange(n_f - N_CHAN_SUBBAND) * CHANNEL_WIDTH_HZ) * 1.0e-6
    return np.concatenate([f0, f1])


def analyze_run(
    run_dir: str,
    old_gains_path: str = DEFAULT_OLD_GAINS,
    new_gains_path: str = DEFAULT_NEW_GAINS,
    out_json: Optional[str] = None,
    detrend_mode: str = "subband_linear",
    plot_png: Optional[str] = None,
    clim_pct: float = 2.0,
) -> Dict[str, Any]:
    waterfall_raw, times, meta = load_run_waterfalls_with_times(run_dir)
    n_t, n_f = waterfall_raw.shape
    freqs_mhz = _channel_freqs_mhz(n_f)
    signal_ants = meta.get("signal_antennas", DEFAULT_SIGNAL_ANTS)

    # 1. Reconstruct undiluted frames (simulating chartsAccumulate fix)
    waterfall_undiluted, dil_stats = detect_and_correct_k20_dilution(waterfall_raw)

    # 2. Apply bandpass ratio correction from old -> freq-filtered gains (only if not already active)
    waterfall_bp, bp_rescaled = _rescale_bandpass_if_needed(
        waterfall_undiluted, meta, old_gains_path, new_gains_path
    )

    # 3. Apply improved RFI excision simulation (subband MAD + persistent EMA + conditional shoulders)
    waterfall_clean, sim_mask = simulate_improved_rfi_excision(waterfall_bp)

    # Bandpass flatness metrics (across active valid channels 300-501.3 MHz)
    raw_spec = np.median(waterfall_raw, axis=0)
    valid_raw_ch = np.isfinite(raw_spec) & (raw_spec > 0.0)
    bp_spec = np.full(n_f, np.nan, dtype=np.float64)
    min_bp_frames = max(10, n_t // 2)
    for f in range(n_f):
        col = waterfall_clean[:, f]
        fin = col[np.isfinite(col) & (col > 0.0)]
        if len(fin) >= min_bp_frames:
            bp_spec[f] = float(np.median(fin))
    valid_bp_ch = np.isfinite(bp_spec) & (bp_spec > 0.0)
    common_ch = valid_raw_ch & valid_bp_ch

    raw_vals = raw_spec[common_ch] if np.any(common_ch) else raw_spec[valid_raw_ch]
    bp_vals = bp_spec[common_ch] if np.any(common_ch) else bp_spec[valid_bp_ch]
    raw_flat_std_pct = 100.0 * float(np.std(raw_vals) / np.median(raw_vals))
    bp_flat_std_pct = 100.0 * float(np.std(bp_vals) / np.median(bp_vals))
    raw_p90_p10 = float(np.percentile(raw_vals, 90) / np.percentile(raw_vals, 10))
    bp_p90_p10 = float(np.percentile(bp_vals, 90) / np.percentile(bp_vals, 10))

    # Residual dynamic spectrum RMS on identical active non-blanked channels
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", category=RuntimeWarning)
        raw_nz = np.where(waterfall_raw[:, common_ch] > 0.0, waterfall_raw[:, common_ch], np.nan)
        raw_rel = raw_nz / raw_spec[common_ch][None, :] - 1.0
        raw_cm = np.nanmedian(raw_rel, axis=1, keepdims=True)
        raw_res_rms_pct = 100.0 * float(np.nanstd(raw_rel - raw_cm))

        clean_full_rel = np.full((n_t, n_f), np.nan, dtype=np.float64)
        clean_nz_full = np.where(
            waterfall_clean[:, valid_bp_ch] > 0.0, waterfall_clean[:, valid_bp_ch], np.nan
        )
        clean_full_rel[:, valid_bp_ch] = clean_nz_full / bp_spec[valid_bp_ch][None, :] - 1.0
        clean_res_rms_pct = detrend_dynamic_spectrum(
            clean_full_rel[:, common_ch], freqs_mhz[common_ch], mode="scalar", sb_split=0
        )[1]["detrended_rms_pct"]

    detrend_summary: Dict[str, Any] = {}
    for dmode in (
        "none",
        "scalar",
        "subband_median",
        "subband_linear",
        "subband_quad",
        "global_linear",
    ):
        _, dinfo = detrend_dynamic_spectrum(clean_full_rel, freqs_mhz, mode=dmode)
        detrend_summary[dmode] = {
            "rms_pct": dinfo["detrended_rms_pct"],
            "sat_1pct": dinfo["sat_1pct"],
            "sat_2pct": dinfo["sat_2pct"],
            "sat_3pct": dinfo["sat_3pct"],
        }
    _, det_diag = detrend_dynamic_spectrum(clean_full_rel, freqs_mhz, mode=detrend_mode)
    sb0_c = det_diag["sb0_common_pct"]
    sb1_c = det_diag["sb1_common_pct"]
    sb0_t = det_diag["sb0_tilt_pct_per_100mhz"]
    sb1_t = det_diag["sb1_tilt_pct_per_100mhz"]

    families = [
        ("edge_401_406", 401.7, 405.6),
        ("cw_414_0", 413.85, 414.15),
        ("burst_381_384", 381.0, 384.0),
        ("comb_362_378", 362.0, 378.0),
        ("tx_shoulder_431_435", 430.8, 435.6),
        ("band_425_450", 425.0, 450.0),
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
        "seq_gaps": int(meta.get("seq_gaps", 0)),
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
        "detrending": {
            "selected_mode": detrend_mode,
            "modes": detrend_summary,
            "sb0_common_range_pct": [round(float(np.nanmin(sb0_c)), 2), round(float(np.nanmax(sb0_c)), 2)],
            "sb1_common_range_pct": [round(float(np.nanmin(sb1_c)), 2), round(float(np.nanmax(sb1_c)), 2)],
            "sb0_tilt_range_pct_per_100mhz": [
                round(float(np.nanmin(sb0_t)), 2),
                round(float(np.nanmax(sb0_t)), 2),
            ],
            "sb1_tilt_range_pct_per_100mhz": [
                round(float(np.nanmin(sb1_t)), 2),
                round(float(np.nanmax(sb1_t)), 2),
            ],
        },
        "rfi_families": rfi_comparison,
    }

    print(
        f"=== CHARTS Closed-Loop Run Analysis: {run_dir} "
        f"({n_t} frames x {n_f} ch, seq_gaps={report['seq_gaps']}) ==="
    )
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
    sel_det = detrend_summary[detrend_mode]
    scl_det = detrend_summary["scalar"]
    print(
        f"[Detrending] scalar RMS={scl_det['rms_pct']}% (sat>2%={scl_det['sat_2pct']}%) -> "
        f"{detrend_mode} RMS={sel_det['rms_pct']}% (sat>2%={sel_det['sat_2pct']}%) | "
        f"SB0 drift={report['detrending']['sb0_common_range_pct']}% "
        f"SB1 drift={report['detrending']['sb1_common_range_pct']}%"
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

    tlog_path = os.path.join(run_dir, "telemetry_log.jsonl")
    if os.path.isfile(tlog_path):
        rows = load_telemetry_records(tlog_path)
        report["telemetry_log"] = {"path": tlog_path, "records": len(rows)}
        print(f"[Telemetry] Loaded {len(rows)} records from {tlog_path}")

    if plot_png:
        trows = load_telemetry_records(tlog_path) if os.path.isfile(tlog_path) else None
        out_path = render_waterfall_figure(
            waterfall_clean,
            freqs_mhz,
            plot_png,
            times=times,
            detrend_mode=detrend_mode,
            clim_pct=clim_pct,
            title=f"CHARTS {Path(run_dir).name} — {detrend_mode} (±{clim_pct:.1f}%)",
            telemetry_rows=trows,
            wf_raw=waterfall_raw,
        )
        report["waterfall_plot_png"] = out_path
        print(f"Saved detrended waterfall PNG to {out_path}")

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
RFI_FREQ_SHOULDER = 1 << 7


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

    def compute_qa(self, detrend_mode: str = "subband_linear") -> Dict[str, Any]:
        if len(self.aligned_frames) < 5:
            return {"window_frames": len(self.aligned_frames), "total_frames": self.total_aligned}

        wf = np.vstack(list(self.aligned_frames))
        n_t, n_f = wf.shape
        freqs_mhz = _channel_freqs_mhz(n_f)
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
            valid_bp = np.isfinite(bp_spec) & (bp_spec > 0.0)
            clean_rel_full = np.full((n_t, n_f), np.nan, dtype=np.float64)
            clean_nz = np.where(wf_clean[:, valid_bp] > 0.0, wf_clean[:, valid_bp], np.nan)
            clean_rel_full[:, valid_bp] = clean_nz / bp_spec[valid_bp][None, :] - 1.0
            clean_rms_pct = detrend_dynamic_spectrum(
                clean_rel_full[:, valid], freqs_mhz[valid], mode="scalar", sb_split=0
            )[1]["detrended_rms_pct"]
            det_res, det_diag = detrend_dynamic_spectrum(
                clean_rel_full, freqs_mhz, mode=detrend_mode
            )

        times = [t for t in self.aligned_times if t is not None]
        dt_med = float(np.median(np.diff(times))) if len(times) >= 2 else None
        last_ts = times[-1] if times else None

        sb0_ts = np.nanmean(clean_rel_full[:, :N_CHAN_SUBBAND], axis=1)
        sb1_ts = np.nanmean(clean_rel_full[:, N_CHAN_SUBBAND:], axis=1)
        both_ok = np.isfinite(sb0_ts) & np.isfinite(sb1_ts)
        sb0_sb1_r: Optional[float] = None
        if int(np.sum(both_ok)) >= 4:
            s0, s1 = sb0_ts[both_ok], sb1_ts[both_ok]
            if float(np.std(s0)) > 1e-12 and float(np.std(s1)) > 1e-12:
                sb0_sb1_r = round(float(np.corrcoef(s0, s1)[0, 1]), 4)

        bb_norm = 1.0 + np.nanmean(clean_rel_full, axis=1)
        dt_eff = float(dt_med) if (dt_med is not None and dt_med > 0.0) else 1.024
        adev_hdev = compute_adev_hdev_summary(bb_norm, dt_eff)

        qa_out: Dict[str, Any] = {
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
            "detrend_mode": detrend_mode,
            "detrended_rms_pct": det_diag["detrended_rms_pct"],
            "detrended_sat_1pct": det_diag["sat_1pct"],
            "detrended_sat_2pct": det_diag["sat_2pct"],
            "detrended_sat_3pct": det_diag["sat_3pct"],
            "sb0_common_pct": round(float(det_diag["sb0_common_pct"][-1]), 3),
            "sb1_common_pct": round(float(det_diag["sb1_common_pct"][-1]), 3),
            "sb0_tilt_pct_per_100mhz": round(float(det_diag["sb0_tilt_pct_per_100mhz"][-1]), 3),
            "sb1_tilt_pct_per_100mhz": round(float(det_diag["sb1_tilt_pct_per_100mhz"][-1]), 3),
            "slow_extra_flagged_pct": round(100.0 * float(np.mean(sim_mask)), 2),
            "slow_channel_occupancy": np.mean(sim_mask, axis=0),
            "_recent_clean_waterfall": wf_clean,
            "_recent_freqs_mhz": freqs_mhz,
            "_recent_times": np.asarray(
                [float(t) if t is not None else np.nan for t in self.aligned_times], dtype=np.float64
            ),
            "_recent_detrended_rel": det_res,
            "sb0_sb1_pearson_r": sb0_sb1_r,
        }
        qa_out.update(adev_hdev)
        return qa_out


def compute_adev_hdev_summary(
    series: np.ndarray,
    dt_s: float,
    taus_s: Sequence[float] = (1.0, 8.0, 16.0, 60.0, 120.0, 300.0),
) -> Dict[str, float]:
    """Compute overlapping Allan (ADEV) and linear-drift-invariant Hadamard (HDEV) deviation."""
    y = np.asarray(series, dtype=np.float64)
    y = y[np.isfinite(y)]
    n = int(y.size)
    if n < 5 or dt_s <= 0.0:
        return {}
    csum = np.concatenate([[0.0], np.cumsum(y)])
    x = np.concatenate([[0.0], np.cumsum(y) * float(dt_s)])
    out: Dict[str, float] = {}
    for target_tau in taus_s:
        m = max(1, int(round(float(target_tau) / float(dt_s))))
        tau = m * float(dt_s)
        tag = f"{int(round(float(target_tau)))}s"
        if 2 * m < n:
            ybar = (csum[m:] - csum[:-m]) / float(m)
            diffs = ybar[m:] - ybar[:-m]
            if diffs.size >= 2:
                avar = 0.5 * float(np.mean(diffs * diffs))
                out[f"adev_tau_{tag}"] = round(float(np.sqrt(max(avar, 0.0))), 6)
        if 3 * m < x.size:
            d3 = x[3 * m :] - 3.0 * x[2 * m : -m] + 3.0 * x[m : -2 * m] - x[: -3 * m]
            if d3.size >= 2:
                hvar = float(np.mean(d3 * d3)) / (6.0 * tau * tau)
                out[f"hdev_tau_{tag}"] = round(float(np.sqrt(max(hvar, 0.0))), 6)
    return out


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


def check_antennas(base_url: str, verbose: bool = True) -> Dict[str, Any]:
    """Inspect live antenna mask status and optional per-channel antenna power telemetry."""
    status = http_get_json(base_url, "/antenna_mask/status")
    chan_power: Dict[str, Any] = {}
    try:
        chan_power = http_get_json(base_url, "/antenna_mask/channel_power")
    except urllib.error.HTTPError as err:
        if err.code != 404:
            raise
    except Exception:
        pass

    ants = status.get("antennas", [])
    active_ids = status.get("active_physical_antennas", [])
    dead = status.get("dead_antennas", 0)
    sat = status.get("saturated_antennas", 0)
    if verbose:
        print(
            f"Antenna Mask Status: active={active_ids} "
            f"(count={len(active_ids)}, dead={dead}, saturated={sat})"
        )
        for row in ants:
            if not row.get("active", True) or float(row.get("clip_fraction", 0.0)) > 0.0:
                print(
                    f"  ant={row.get('antenna_id')} active={row.get('active')} "
                    f"healthy={row.get('consecutive_healthy')} clip={row.get('clip_fraction')}"
                )
        if chan_power:
            chans = chan_power.get("channels", chan_power.get("channel_power", []))
            if chans:
                print(f"Antenna channel_power entries: {len(chans)}")
    return {"status": status, "channel_power": chan_power}


def _channel_prewhite_power(ch: Dict[str, Any]) -> float:
    p = float(ch.get("mean_power", 0.0))
    if p <= 0.0:
        return 0.0
    g = float(ch.get("gain", 1.0))
    return p / (g * g) if g > 0.0 else p


def detect_latched_ema_channels(
    channels: List[Dict[str, Any]],
    latch_streaks: Dict[float, int],
    last_reset_ts: Dict[float, float],
    now: float,
    streak_threshold: int = 2,
    cooldown_s: float = 180.0,
    reset_counts: Optional[Dict[float, int]] = None,
) -> List[float]:
    """Identify channels stuck in TIME_BURST/SIR at Gaussian SK & truly clean neighborhood power."""
    to_reset: List[float] = []
    seen_freqs = set()
    n_ch = len(channels)
    prewhite = [_channel_prewhite_power(ch) for ch in channels]
    sbs = [int(ch.get("subband", 0 if i < N_CHAN_SUBBAND else 1)) for i, ch in enumerate(channels)]

    for i, ch in enumerate(channels):
        f_mhz = round(float(ch.get("freq_mhz", 0.0)), 3)
        seen_freqs.add(f_mhz)
        frac = float(ch.get("flagged_chunk_fraction", 0.0))
        bits = int(ch.get("flag_bits", 0))
        sk = float(ch.get("mean_sk", 1.0))
        mad_z = float(ch.get("freq_mad_z", 0.0))

        # Guard 1: adjacent channels actively bursting or notched indicate a shoulder, not a latch
        neighbor_bursting = False
        clean_nbr_pws: List[float] = []
        for j in range(max(0, i - 6), min(n_ch, i + 7)):
            if j == i or sbs[j] != sbs[i]:
                continue
            nb_frac = float(channels[j].get("flagged_chunk_fraction", 0.0))
            nb_bits = int(channels[j].get("flag_bits", 0))
            if abs(j - i) <= 2 and (
                nb_frac >= 0.50
                or (
                    nb_bits
                    & (RFI_FREQ_MAD | RFI_SK_CW_TONE | RFI_SATURATION_CLIP | RFI_STATIC_NOTCH)
                )
                != 0
            ):
                neighbor_bursting = True
            if nb_frac < 0.25 and prewhite[j] > 0.0:
                clean_nbr_pws.append(prewhite[j])

        # Guard 2: true excess > +6% above clean neighbors means real unexcised RFI, not a dip latch
        excess_above_nbr = 0.0
        if len(clean_nbr_pws) >= 2 and prewhite[i] > 0.0:
            nbr_med = float(np.median(clean_nbr_pws))
            if nbr_med > 0.0:
                excess_above_nbr = prewhite[i] / nbr_med - 1.0

        pure_burst_latch = (
            frac >= 0.85
            and (bits & (RFI_TIME_BURST | RFI_MORPHOLOGY_SIR)) != 0
            and (bits & ~(RFI_TIME_BURST | RFI_MORPHOLOGY_SIR)) == 0
            and (0.75 <= sk <= 1.25)
            and (abs(mad_z) <= 2.5)
            and (not neighbor_bursting)
            and (excess_above_nbr <= 0.06)
        )
        if pure_burst_latch:
            latch_streaks[f_mhz] = latch_streaks.get(f_mhz, 0) + 1
            if latch_streaks[f_mhz] >= streak_threshold:
                prior = reset_counts.get(f_mhz, 0) if reset_counts is not None else 0
                eff_cd = cooldown_s * float(2 ** min(max(0, prior - 1), 3))
                if (now - last_reset_ts.get(f_mhz, 0.0)) >= eff_cd:
                    to_reset.append(f_mhz)
                    last_reset_ts[f_mhz] = now
                    latch_streaks[f_mhz] = 0
                    if reset_counts is not None:
                        reset_counts[f_mhz] = prior + 1
        else:
            latch_streaks[f_mhz] = 0
            if reset_counts is not None and frac < 0.10:
                reset_counts[f_mhz] = 0

    for old_f in list(latch_streaks.keys()):
        if old_f not in seen_freqs:
            del latch_streaks[old_f]
    if reset_counts is not None:
        for old_f in list(reset_counts.keys()):
            if old_f not in seen_freqs:
                del reset_counts[old_f]
    return to_reset


def _in_manual_notch(f_mhz: float, manual_notches: List[List[float]], margin: float = 0.0) -> bool:
    return any((pair[0] - margin) <= f_mhz <= (pair[1] + margin) for pair in manual_notches)


def _in_protected_solar_range(
    f_mhz: float, solar_protected_ranges_mhz: Optional[Sequence[Sequence[float]]]
) -> bool:
    if not solar_protected_ranges_mhz:
        return False
    return any(len(r) >= 2 and float(r[0]) <= f_mhz <= float(r[1]) for r in solar_protected_ranges_mhz)


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
    solar_protected_ranges_mhz: Optional[Sequence[Sequence[float]]] = None,
) -> Tuple[List[List[float]], bool, Dict[str, Any]]:
    """Cluster- and shoulder-aware hysteresis manager for persistent RFI static notches."""
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
    n_ch = len(channels)
    freqs = [round(float(ch.get("freq_mhz", 0.0)), 3) for ch in channels]
    sbs = [int(ch.get("subband", 0 if i < N_CHAN_SUBBAND else 1)) for i, ch in enumerate(channels)]

    # Pass 1: identify core severe RFI channels (lowered 0.90 -> 0.70 gate catches 431.7-434.4 bursts)
    core_active = [False] * n_ch
    for i, ch in enumerate(channels):
        f_mhz = freqs[i]
        raw_occ = float(ch.get("flagged_chunk_fraction", 0.0))
        slow_val = float(slow_occ[i]) if (slow_occ is not None and i < len(slow_occ)) else 0.0
        bits = int(ch.get("flag_bits", 0))
        sk = float(ch.get("mean_sk", 1.0))
        mad_z = float(ch.get("freq_mad_z", 0.0))
        physical_core = (
            abs(mad_z) >= 6.0
            or sk >= 2.2
            or sk <= 0.65
            or (bits & RFI_SATURATION_CLIP) != 0
        )
        severe_core = physical_core and (raw_occ >= 0.70 or slow_val >= 0.65)
        if (
            severe_core
            or (f_mhz in managed_freqs)
            or _in_manual_notch(f_mhz, manual_notches, margin=0.05)
        ):
            core_active[i] = True

    # Pass 2: promote/demote both core RFI channels and active +-2-channel spectral shoulders
    for idx, ch in enumerate(channels):
        f_mhz = freqs[idx]
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
        severe_core = physical_rfi and (raw_occ >= 0.70 or slow_val >= 0.65)

        adj_core = False
        for nb in range(max(0, idx - 2), min(n_ch, idx + 3)):
            if nb == idx or sbs[nb] != sbs[idx]:
                continue
            if abs(freqs[nb] - f_mhz) <= 0.65 and core_active[nb]:
                adj_core = True
                break
        shoulder_rfi = adj_core and (
            (abs(mad_z) >= 2.5 or (bits & RFI_FREQ_SHOULDER) != 0)
            and (raw_occ >= 0.65 or slow_val >= 0.55)
        )

        severe_persistent = severe_core or shoulder_rfi
        clean_raw = (
            (abs(mad_z) < 2.5)
            and (0.80 <= sk <= 1.20)
            and ((bits & (RFI_SATURATION_CLIP | RFI_FREQ_SHOULDER)) == 0)
            and (raw_occ < 0.30)
            and (slow_val < 0.30)
        )

        if severe_persistent:
            if _in_protected_solar_range(f_mhz, solar_protected_ranges_mhz):
                bad_streaks[f_mhz] = 0
                clean_streaks[f_mhz] = 0
                continue
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


def _parse_timestamp_any(val: Any) -> Optional[float]:
    """Parse UNIX epoch (s or ms) or ISO/datetime timestamp string to UTC UNIX seconds."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        v = float(val)
        if not np.isfinite(v):
            return None
        return v * 1e-3 if v > 1e11 else v
    s = str(val).strip()
    if not s:
        return None
    try:
        v = float(s)
        if np.isfinite(v):
            return v * 1e-3 if v > 1e11 else v
    except ValueError:
        pass
    s_norm = s.replace("Z", "+00:00")
    for fmt in (None, "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S"):
        try:
            if fmt is None:
                dt = datetime.datetime.fromisoformat(s_norm)
            else:
                dt = datetime.datetime.strptime(s, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=datetime.timezone.utc)
            return dt.timestamp()
        except ValueError:
            continue
    return None


def _parse_numeric_any(val: Any) -> Optional[float]:
    if val is None or isinstance(val, bool):
        return None
    if isinstance(val, (int, float)):
        v = float(val)
        return v if np.isfinite(v) else None
    s = str(val).strip()
    if not s or s.lower() in ("nan", "null", "none", "na", "-"):
        return None
    m = re.match(r"^([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", s)
    if not m:
        return None
    v = float(m.group(1))
    return v if np.isfinite(v) else None


def _canonical_telemetry_key(raw_key: str) -> str:
    k = raw_key.strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "temp": "ambient_temp_c",
        "temp_c": "ambient_temp_c",
        "temperature": "ambient_temp_c",
        "temperature_c": "ambient_temp_c",
        "outdoor_temp": "ambient_temp_c",
        "outdoor_temp_c": "ambient_temp_c",
        "aht10_temp": "ambient_temp_c",
        "pequen_temp": "ambient_temp_c",
        "lna_temp": "lna_temp_c",
        "lna_voltage": "lna_v",
        "lna_voltage_v": "lna_v",
        "psu_ch1_v": "lna_v",
        "v_ch1": "lna_v",
        "lna_current": "lna_i",
        "lna_current_a": "lna_i",
        "psu_ch1_i": "lna_i",
        "i_ch1": "lna_i",
        "lna_power_w": "lna_w",
        "humidity": "humidity_pct",
        "rh": "humidity_pct",
    }
    return aliases.get(k, k)


def load_telemetry_records(path_or_glob: str) -> List[Dict[str, Any]]:
    """Load timestamped environmental / LNA telemetry from JSONL or CSV (incl. Grafana exports)."""
    paths = sorted(glob.glob(path_or_glob))
    if not paths and os.path.isfile(path_or_glob):
        paths = [path_or_glob]
    records: List[Dict[str, Any]] = []

    for p in paths:
        if p.endswith(".jsonl") or p.endswith(".json"):
            txt = Path(p).read_text(encoding="utf-8", errors="ignore").splitlines()
            for line in txt:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                items = obj if isinstance(obj, list) else ([obj] if isinstance(obj, dict) else [])
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    ts = _parse_timestamp_any(
                        item.get("timestamp_unix") or item.get("timestamp_utc") or item.get("time")
                    )
                    if ts is None:
                        continue
                    rec = {"timestamp_unix": ts}
                    for k, v in item.items():
                        if k in ("timestamp_unix", "timestamp_utc", "time", "series"):
                            continue
                        num = _parse_numeric_any(v)
                        if num is not None:
                            rec[_canonical_telemetry_key(k)] = num
                    records.append(rec)
        else:
            with open(p, "r", encoding="utf-8", errors="ignore", newline="") as cf:
                reader = csv.DictReader(cf)
                if not reader.fieldnames:
                    continue
                ts_col = None
                for cand in reader.fieldnames:
                    cl = cand.strip().lower()
                    if cl in ("timestamp_unix", "timestamp_utc", "timestamp", "time", "date", "utc"):
                        ts_col = cand
                        break
                if ts_col is None:
                    ts_col = reader.fieldnames[0]
                is_long_grafana = any(f.strip().lower() == "value" for f in reader.fieldnames) and any(
                    f.strip().lower() in ("series", "metric", "field", "name") for f in reader.fieldnames
                )
                series_col = next(
                    (f for f in reader.fieldnames if f.strip().lower() in ("series", "metric", "field", "name")),
                    None,
                )
                for row in reader:
                    ts = _parse_timestamp_any(row.get(ts_col))
                    if ts is None:
                        continue
                    rec: Dict[str, Any] = {"timestamp_unix": ts}
                    if is_long_grafana and series_col:
                        sname = _canonical_telemetry_key(str(row.get(series_col) or "value"))
                        val = _parse_numeric_any(row.get("Value") or row.get("value"))
                        if val is not None:
                            rec[sname] = val
                    else:
                        for k, v in row.items():
                            if k == ts_col or not k:
                                continue
                            ck = _canonical_telemetry_key(k)
                            if ck == "value" and series_col is None:
                                ck = "ambient_temp_c"
                            num = _parse_numeric_any(v)
                            if num is not None:
                                rec[ck] = num
                    if len(rec) > 1:
                        records.append(rec)

    records.sort(key=lambda r: float(r["timestamp_unix"]))
    merged: List[Dict[str, Any]] = []
    for rec in records:
        ts_val = float(rec["timestamp_unix"])
        if merged and abs(float(merged[-1]["timestamp_unix"]) - ts_val) <= 1e-3:
            merged[-1].update(rec)
        else:
            merged.append(dict(rec))
    return merged


def _read_local_hwmon_temps(base_path: Path = Path("/sys/class/hwmon")) -> Dict[str, float]:
    """Read Linux sysfs hwmon temperature sensors in Celsius, skipping alarmed/unpopulated channels."""
    out: Dict[str, float] = {}
    if not base_path.is_dir():
        return out
    temps: List[float] = []
    for hw in sorted(base_path.iterdir()):
        name_p = hw / "name"
        hw_name = name_p.read_text(encoding="utf-8", errors="ignore").strip() if name_p.is_file() else hw.name
        for tp in sorted(hw.glob("temp*_input")):
            # Skip disconnected/faulted headers (e.g. unpopulated NCT6798 AUXTIN4 reading 98-100C with alarm=1).
            alarm_p = hw / tp.name.replace("_input", "_alarm")
            fault_p = hw / tp.name.replace("_input", "_fault")
            try:
                if alarm_p.is_file() and alarm_p.read_text(encoding="utf-8").strip() == "1":
                    continue
                if fault_p.is_file() and fault_p.read_text(encoding="utf-8").strip() == "1":
                    continue
                raw_mc = int(tp.read_text(encoding="utf-8").strip())
                if raw_mc == 0:
                    continue
                val_c = float(raw_mc) / 1000.0
            except Exception:
                continue
            if -50.0 < val_c < 150.0:
                temps.append(val_c)
                label_p = hw / tp.name.replace("_input", "_label")
                lbl = (
                    label_p.read_text(encoding="utf-8", errors="ignore").strip()
                    if label_p.is_file()
                    else tp.name.replace("_input", "")
                )
                key = f"hwmon_{re.sub(r'[^a-z0-9_]+', '_', (hw_name + '_' + lbl).lower()).strip('_')}_c"
                out[key] = round(val_c, 2)
    if temps:
        out["host_max_temp_c"] = round(max(temps), 2)
    return out


class EnvironmentTelemetryCollector:
    """Multi-source collector for outdoor ambient temperature and LNA bias telemetry."""

    def __init__(
        self,
        prometheus_url: Optional[str] = None,
        prom_queries: Optional[Dict[str, str]] = None,
        lna_json_url: Optional[str] = None,
        rigol_host: Optional[str] = None,
        rigol_port: int = 5555,
        telemetry_csv: Optional[str] = None,
        include_local_hwmon: bool = True,
        timeout_s: float = 1.5,
        alpha_t_db_per_c: float = 0.0,
        ref_temp_c: float = 20.0,
    ) -> None:
        self.prometheus_url = prometheus_url or os.environ.get("CHARTS_PROMETHEUS_URL")
        self.prom_queries = prom_queries or {
            "ambient_temp_c": 'aht10_temperature_c or pequen_temperature_c or node_hwmon_temp_celsius{sensor=~".*ambient.*"}',
            "humidity_pct": "aht10_humidity_pct or pequen_humidity_pct",
        }
        self.lna_json_url = lna_json_url or os.environ.get("CHARTS_LNA_JSON_URL")
        rh = rigol_host or os.environ.get("CHARTS_RIGOL_HOST")
        if rh and ":" in rh:
            host_part, port_part = rh.rsplit(":", 1)
            self.rigol_host = host_part
            self.rigol_port = int(port_part)
        else:
            self.rigol_host = rh
            self.rigol_port = int(rigol_port)
        self.telemetry_csv = telemetry_csv or os.environ.get("CHARTS_TELEMETRY_CSV")
        self.include_local_hwmon = include_local_hwmon
        self.timeout_s = float(timeout_s)
        self.alpha_t_db_per_c = float(alpha_t_db_per_c)
        self.ref_temp_c = float(ref_temp_c)

    def _poll_prometheus(self) -> Tuple[Dict[str, float], str]:
        if not self.prometheus_url:
            return {}, "disabled"
        vals: Dict[str, float] = {}
        try:
            base = self.prometheus_url.rstrip("/")
            if base.endswith("/metrics"):
                req = urllib.request.Request(base, method="GET")
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    text = resp.read().decode("utf-8", errors="ignore")
                for line in text.splitlines():
                    if not line or line.startswith("#"):
                        continue
                    m = re.match(r"^([a-zA-Z0-9_:]+)(?:\{[^}]*\})?\s+([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", line)
                    if not m:
                        continue
                    metric, raw_v = m.group(1).lower(), float(m.group(2))
                    if "ambient" in metric or "aht10" in metric or "pequen" in metric:
                        if "temp" in metric:
                            vals["ambient_temp_c"] = round(raw_v, 3)
                        elif "hum" in metric:
                            vals["humidity_pct"] = round(raw_v, 2)
            else:
                for key, query in self.prom_queries.items():
                    q_url = f"{base}/api/v1/query?query={urllib.parse.quote(query)}"
                    with urllib.request.urlopen(q_url, timeout=self.timeout_s) as resp:
                        payload = json.loads(resp.read().decode("utf-8"))
                    res = payload.get("data", {}).get("result", [])
                    if res:
                        val = _parse_numeric_any(res[0].get("value", [None, None])[1])
                        if val is not None:
                            vals[key] = round(val, 3)
            return vals, "ok" if vals else "empty"
        except Exception as exc:
            return {}, f"offline: {exc}"

    def _poll_lna_json(self) -> Tuple[Dict[str, float], str]:
        if not self.lna_json_url:
            return {}, "disabled"
        try:
            with urllib.request.urlopen(self.lna_json_url, timeout=self.timeout_s) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            vals: Dict[str, float] = {}
            if isinstance(data, dict):
                for k, v in data.items():
                    num = _parse_numeric_any(v)
                    if num is not None:
                        vals[_canonical_telemetry_key(k)] = round(num, 4)
            return vals, "ok" if vals else "empty"
        except Exception as exc:
            return {}, f"offline: {exc}"

    def _poll_rigol_dp8xx(self) -> Tuple[Dict[str, float], str]:
        if not self.rigol_host:
            return {}, "disabled"
        vals: Dict[str, float] = {}
        try:
            with socket.create_connection((self.rigol_host, self.rigol_port), timeout=self.timeout_s) as sock:
                sock.settimeout(self.timeout_s)
                for ch_idx in (1, 2, 3):
                    sock.sendall(f":MEAS:ALL:DC? CH{ch_idx}\n".encode("ascii"))
                    line = sock.recv(256).decode("ascii", errors="ignore").strip()
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 3:
                        v, i_a, w = (_parse_numeric_any(parts[0]), _parse_numeric_any(parts[1]), _parse_numeric_any(parts[2]))
                        if ch_idx == 1:
                            if v is not None:
                                vals["lna_v"] = round(v, 4)
                            if i_a is not None:
                                vals["lna_i"] = round(i_a, 5)
                            if w is not None:
                                vals["lna_w"] = round(w, 4)
                        else:
                            if v is not None:
                                vals[f"psu_ch{ch_idx}_v"] = round(v, 4)
                            if i_a is not None:
                                vals[f"psu_ch{ch_idx}_i"] = round(i_a, 5)
            return vals, "ok" if vals else "empty"
        except Exception as exc:
            return {}, f"offline: {exc}"

    def _poll_csv(self) -> Tuple[Dict[str, float], str]:
        if not self.telemetry_csv:
            return {}, "disabled"
        try:
            recs = load_telemetry_records(self.telemetry_csv)
            if not recs:
                return {}, "empty"
            latest: Dict[str, Any] = {}
            for r in recs[-min(len(recs), 16) :]:
                for k, v in r.items():
                    if k != "timestamp_unix" and isinstance(v, (int, float)):
                        latest[k] = float(v)
            age_s = time.time() - float(recs[-1]["timestamp_unix"])
            return latest, f"ok(age={age_s:.0f}s)"
        except Exception as exc:
            return {}, f"offline: {exc}"

    def sample(self) -> Dict[str, Any]:
        """Collect one aligned telemetry snapshot across all enabled environmental sources."""
        now = time.time()
        metrics: Dict[str, Any] = {}
        sources_status: Dict[str, str] = {}

        if self.include_local_hwmon:
            hw = _read_local_hwmon_temps()
            metrics.update(hw)
            sources_status["local_hwmon"] = "ok" if hw else "empty"

        for name, fn in (
            ("prometheus", self._poll_prometheus),
            ("lna_json", self._poll_lna_json),
            ("rigol_dp8xx", self._poll_rigol_dp8xx),
            ("telemetry_csv", self._poll_csv),
        ):
            vals, st = fn()
            if st != "disabled":
                sources_status[name] = st
            metrics.update(vals)

        lna_temp_c = metrics.get("ambient_temp_c")
        if lna_temp_c is None:
            for k, v in metrics.items():
                if ("systin" in k or "lna_temp" in k) and isinstance(v, (int, float)):
                    lna_temp_c = float(v)
                    break
        if lna_temp_c is None and isinstance(metrics.get("host_max_temp_c"), (int, float)):
            lna_temp_c = float(metrics["host_max_temp_c"])

        if lna_temp_c is not None:
            metrics["lna_temp_c"] = round(float(lna_temp_c), 3)
            if abs(self.alpha_t_db_per_c) > 0.0:
                drift_db = self.alpha_t_db_per_c * (float(lna_temp_c) - self.ref_temp_c)
                metrics["thermal_gain_drift_db"] = round(drift_db, 4)
                metrics["thermal_gain_scale"] = round(float(10.0 ** (-drift_db / 10.0)), 6)

        return {
            "timestamp_unix": round(now, 3),
            "timestamp_utc": datetime.datetime.fromtimestamp(
                now, datetime.timezone.utc
            ).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "metrics": metrics,
            "sources": sources_status,
        }


def detect_gain_steps(
    times_s: np.ndarray,
    values_pct: np.ndarray,
    window_frames: int = 45,
    min_step_pct: float = 0.25,
    min_z: float = 4.0,
) -> List[Dict[str, Any]]:
    """Detect significant broadband step changes via split-window median difference with NMS."""
    t_arr = np.asarray(times_s, dtype=np.float64)
    y_arr = np.asarray(values_pct, dtype=np.float64)
    n = len(y_arr)
    w = max(5, int(window_frames))
    if n < 2 * w + 1:
        return []

    raw_cands: List[Tuple[int, float, float, float, float]] = []
    for i in range(w, n - w):
        left = y_arr[i - w : i]
        right = y_arr[i : i + w]
        left = left[np.isfinite(left)]
        right = right[np.isfinite(right)]
        if len(left) < max(3, w // 2) or len(right) < max(3, w // 2):
            continue
        m0 = float(np.median(left))
        m1 = float(np.median(right))
        step = m1 - m0
        mad0 = 1.4826 * float(np.median(np.abs(left - m0)))
        mad1 = 1.4826 * float(np.median(np.abs(right - m1)))
        sigma = max(np.sqrt(mad0 * mad0 + mad1 * mad1) / np.sqrt(max(1.0, float(min(len(left), len(right))) * 0.25)), 0.05)
        z = step / sigma
        if abs(step) >= min_step_pct and abs(z) >= min_z:
            raw_cands.append((i, step, z, m0, m1))

    # Non-maximum suppression within +/- w frames
    selected: List[Dict[str, Any]] = []
    raw_cands.sort(key=lambda x: abs(x[1]), reverse=True)
    used: List[int] = []
    for i, step, z, m0, m1 in raw_cands:
        if any(abs(i - u) < w for u in used):
            continue
        used.append(i)
        ts = float(t_arr[i])
        ts_utc = (
            datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            if ts > 1e9
            else None
        )
        selected.append(
            {
                "index": int(i),
                "timestamp_unix": round(ts, 3),
                "timestamp_utc": ts_utc,
                "step_pct": round(float(step), 3),
                "z_score": round(float(z), 2),
                "before_pct": round(float(m0), 3),
                "after_pct": round(float(m1), 3),
            }
        )
    selected.sort(key=lambda d: d["index"])
    return selected


def correlate_environment_telemetry(
    run_or_log_path: str,
    telemetry_path: Optional[str] = None,
    out_json: Optional[str] = None,
    step_window_frames: int = 45,
    min_step_pct: float = 0.25,
) -> Dict[str, Any]:
    """Correlate broadband SB0/SB1 continuum steps & drifts with outdoor/LNA telemetry logs."""
    p = Path(run_or_log_path)
    beam_times: np.ndarray
    sb0_pct: np.ndarray
    sb1_pct: np.ndarray
    tel_rows: List[Dict[str, Any]] = []

    if p.is_dir():
        tlog = p / "telemetry_log.jsonl"
        if telemetry_path:
            tel_rows = load_telemetry_records(telemetry_path)
        elif tlog.is_file():
            tel_rows = load_telemetry_records(str(tlog))

        wf_raw, times, meta = load_run_waterfalls_with_times(str(p))
        if not np.all(np.isfinite(times)):
            raise ValueError(f"Run {run_or_log_path} lacks valid UNIX frame timestamps")
        wf_u, _ = detect_and_correct_k20_dilution(wf_raw)
        wf_clean, _ = simulate_improved_rfi_excision(wf_u)
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore", category=RuntimeWarning)
            nz = np.where(wf_clean > 0.0, wf_clean, np.nan)
            bp = np.nanmedian(nz, axis=0)
            ok = np.isfinite(bp) & (bp > 0.0)
            rel = np.full_like(nz, np.nan)
            rel[:, ok] = nz[:, ok] / bp[ok][None, :] - 1.0
        _, diag = detrend_dynamic_spectrum(rel, _channel_freqs_mhz(wf_raw.shape[1]), mode="subband_linear")
        beam_times = times
        sb0_pct = diag["sb0_common_pct"]
        sb1_pct = diag["sb1_common_pct"]
    else:
        rows = load_telemetry_records(str(p))
        if not rows:
            raise ValueError(f"No valid telemetry records found in {run_or_log_path}")
        beam_times = np.array([r["timestamp_unix"] for r in rows], dtype=np.float64)
        sb0_pct = np.array([r.get("sb0_common_pct", np.nan) for r in rows], dtype=np.float64)
        sb1_pct = np.array([r.get("sb1_common_pct", np.nan) for r in rows], dtype=np.float64)
        tel_rows = load_telemetry_records(telemetry_path) if telemetry_path else rows

    steps_sb0 = detect_gain_steps(beam_times, sb0_pct, window_frames=step_window_frames, min_step_pct=min_step_pct)
    steps_sb1 = detect_gain_steps(beam_times, sb1_pct, window_frames=step_window_frames, min_step_pct=min_step_pct)

    sensor_keys = sorted(
        {
            k
            for r in tel_rows
            for k in r.keys()
            if k
            not in (
                "timestamp_unix",
                "sb0_common_pct",
                "sb1_common_pct",
                "sb0_median_power",
                "sb1_median_power",
                "sb0_tilt_pct_per_100mhz",
                "sb1_tilt_pct_per_100mhz",
                "detrended_rms_pct",
                "raw_residual_rms_pct",
                "rfi_flagged_pct",
            )
        }
    )

    correlations: Dict[str, Any] = {}
    if tel_rows and len(beam_times) >= 4:
        t_tel = np.array([r["timestamp_unix"] for r in tel_rows], dtype=np.float64)
        for key in sensor_keys:
            pairs = [(r["timestamp_unix"], float(r[key])) for r in tel_rows if key in r]
            if len(pairs) < 4:
                continue
            tx = np.array([x[0] for x in pairs], dtype=np.float64)
            vx = np.array([x[1] for x in pairs], dtype=np.float64)
            ov = (beam_times >= tx[0]) & (beam_times <= tx[-1])
            n_ov = int(np.sum(ov))
            if n_ov < 4:
                correlations[key] = {"overlap_samples": n_ov, "sensor_samples": len(pairs)}
                continue
            v_interp = np.interp(beam_times[ov], tx, vx)
            s0 = sb0_pct[ov]
            s1 = sb1_pct[ov]
            m0 = np.isfinite(s0) & np.isfinite(v_interp)
            m1 = np.isfinite(s1) & np.isfinite(v_interp)

            def _safe_r(a: np.ndarray, b: np.ndarray) -> Optional[float]:
                if len(a) < 4 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
                    return None
                return round(float(np.corrcoef(a, b)[0, 1]), 4)

            step_alignments: List[Dict[str, Any]] = []
            for st in steps_sb0 + steps_sb1:
                ts_st = float(st["timestamp_unix"])
                pre = vx[(tx >= ts_st - 120.0) & (tx < ts_st)]
                post = vx[(tx >= ts_st) & (tx <= ts_st + 120.0)]
                if len(pre) >= 1 and len(post) >= 1:
                    step_alignments.append(
                        {
                            "timestamp_utc": st.get("timestamp_utc"),
                            "beam_step_pct": st["step_pct"],
                            "sensor_delta": round(float(np.median(post) - np.median(pre)), 4),
                        }
                    )

            correlations[key] = {
                "overlap_samples": n_ov,
                "sensor_range": [round(float(vx.min()), 4), round(float(vx.max()), 4)],
                "r_level_sb0": _safe_r(v_interp[m0], s0[m0]),
                "r_level_sb1": _safe_r(v_interp[m1], s1[m1]),
                "r_diff_sb0": _safe_r(np.diff(v_interp[m0]), np.diff(s0[m0])),
                "r_diff_sb1": _safe_r(np.diff(v_interp[m1]), np.diff(s1[m1])),
                "step_alignments": step_alignments,
            }

    summary: Dict[str, Any] = {
        "source": str(run_or_log_path),
        "telemetry_path": telemetry_path,
        "beam_samples": int(len(beam_times)),
        "telemetry_samples": int(len(tel_rows)),
        "detected_steps_sb0": steps_sb0,
        "detected_steps_sb1": steps_sb1,
        "sensor_correlations": correlations,
    }

    print(
        f"=== Telemetry Step & Correlation Audit ({len(beam_times)} beam samples, "
        f"{len(tel_rows)} env samples) ==="
    )
    print(f"Detected gain steps: SB0={len(steps_sb0)} SB1={len(steps_sb1)}")
    for st in steps_sb0[:8]:
        print(f"  [SB0 step] {st.get('timestamp_utc') or st['timestamp_unix']} Δ={st['step_pct']:+.3f}% (z={st['z_score']})")
    for st in steps_sb1[:8]:
        print(f"  [SB1 step] {st.get('timestamp_utc') or st['timestamp_unix']} Δ={st['step_pct']:+.3f}% (z={st['z_score']})")
    for k, info in correlations.items():
        if "r_level_sb0" in info:
            print(
                f"  [{k}] overlap={info['overlap_samples']} range={info['sensor_range']} "
                f"r_lvl=({info['r_level_sb0']},{info['r_level_sb1']}) "
                f"r_diff=({info['r_diff_sb0']},{info['r_diff_sb1']}) "
                f"step_hits={len(info['step_alignments'])}"
            )
        else:
            print(f"  [{k}] no overlap ({info['overlap_samples']} samples)")

    if out_json:
        _write_json_atomic(out_json, summary)
        print(f"Saved telemetry correlation JSON to {out_json}")
    return summary


def _default_rfsoc_probe(host: str) -> Dict[str, Any]:
    """Run non-blocking 1s ping and ARP neighbor probe for an RFSoC host."""
    ping_ok = False
    arp_state = "unknown"
    try:
        pr = subprocess.run(
            ["ping", "-c", "1", "-W", "1", host],
            capture_output=True,
            text=True,
            timeout=2.5,
            check=False,
        )
        ping_ok = pr.returncode == 0
    except Exception:
        ping_ok = False
    try:
        ar = subprocess.run(
            ["ip", "neigh", "show", host],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        txt = (ar.stdout or "").strip()
        if txt:
            arp_state = txt.splitlines()[0].split()[-1]
    except Exception:
        pass
    return {"host": host, "ping_ok": ping_ok, "arp_state": arp_state}


class RfsocRecoveryWatchdog:
    """Automated RFSoC ping/ARP & DPDK sequence-staleness recovery watchdog (OPS-6.2)."""

    def __init__(
        self,
        rfsoc_hosts: Sequence[str] = (),
        stale_threshold_s: float = 10.0,
        cooldown_s: float = 120.0,
        recovery_cmd: Optional[str] = None,
        probe_fn: Optional[Callable[[str], Dict[str, Any]]] = None,
        recovery_fn: Optional[Callable[[List[str]], Tuple[bool, str]]] = None,
    ) -> None:
        self.rfsoc_hosts = [str(h).strip() for h in rfsoc_hosts if str(h).strip()]
        self.stale_threshold_s = float(stale_threshold_s)
        self.cooldown_s = float(cooldown_s)
        self.recovery_cmd = recovery_cmd
        self.probe_fn = probe_fn or _default_rfsoc_probe
        self.recovery_fn = recovery_fn
        self.last_recovery_ts: float = -1.0e18

    def check_and_recover(self, now: float, frame_age_s: Optional[float]) -> Dict[str, Any]:
        """Check whether stream staleness exceeds threshold and run RFSoC recovery probes."""
        if not self.rfsoc_hosts:
            return {"enabled": False, "triggered": False}
        if frame_age_s is None or float(frame_age_s) <= self.stale_threshold_s:
            return {"enabled": True, "triggered": False, "frame_age_s": frame_age_s}
        if (now - self.last_recovery_ts) < self.cooldown_s:
            return {
                "enabled": True,
                "triggered": False,
                "cooldown_active": True,
                "frame_age_s": float(frame_age_s),
            }

        probes = [self.probe_fn(h) for h in self.rfsoc_hosts]
        unhealthy = [p["host"] for p in probes if not p.get("ping_ok")]
        recovery_ok = True
        message = "all_hosts_ping_ok"
        if self.recovery_fn is not None:
            recovery_ok, message = self.recovery_fn(unhealthy or list(self.rfsoc_hosts))
        elif self.recovery_cmd:
            try:
                res = subprocess.run(
                    self.recovery_cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=15.0,
                    check=False,
                )
                recovery_ok = res.returncode == 0
                message = f"cmd_rc={res.returncode}"
            except Exception as exc:
                recovery_ok = False
                message = f"cmd_error:{exc}"
        elif unhealthy:
            recovery_ok = False
            message = f"unreachable:{','.join(unhealthy)}"

        self.last_recovery_ts = now
        return {
            "enabled": True,
            "triggered": True,
            "frame_age_s": float(frame_age_s),
            "probes": probes,
            "unhealthy_hosts": unhealthy,
            "recovery_ok": bool(recovery_ok),
            "message": message,
        }


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
        detrend_mode: str = "subband_linear",
        env_collector: Optional[EnvironmentTelemetryCollector] = None,
        telemetry_jsonl: Optional[str] = None,
        live_waterfall_png: Optional[str] = None,
        clim_pct: float = 2.5,
        rfsoc_watchdog: Optional[RfsocRecoveryWatchdog] = None,
    ) -> None:
        self.base_url = base_url
        self.run_dir = run_dir
        if status_json:
            self.status_json = status_json
        elif run_dir:
            self.status_json = os.path.join(run_dir, "self_tune_status.json")
        else:
            self.status_json = "/tmp/charts_self_tune_status.json"
        if telemetry_jsonl:
            self.telemetry_jsonl = telemetry_jsonl
        elif run_dir:
            self.telemetry_jsonl = os.path.join(run_dir, "telemetry_log.jsonl")
        else:
            self.telemetry_jsonl = "/tmp/charts_telemetry_log.jsonl"
        self.watch_gains = watch_gains
        self.auto_notch = auto_notch
        self.new_gains_path = new_gains_path
        self.detrend_mode = detrend_mode
        self.env_collector = env_collector or EnvironmentTelemetryCollector()
        self.live_waterfall_png = live_waterfall_png
        self.clim_pct = float(clim_pct)
        self.rfsoc_watchdog = rfsoc_watchdog or RfsocRecoveryWatchdog()

        self.tailer = WaterfallTailer(run_dir=run_dir, max_frames=300)
        self.prev_rfi_gains: Optional[Tuple[np.ndarray, Optional[np.ndarray]]] = None
        self.latch_streaks: Dict[float, int] = {}
        self.last_ema_reset_ts: Dict[float, float] = {}
        self.ema_reset_counts: Dict[float, int] = {}
        self.notch_bad_streaks: Dict[float, int] = {}
        self.notch_clean_streaks: Dict[float, int] = {}
        self.managed_notch_freqs: set[float] = set()
        self.solar_protected_ranges_mhz: List[List[float]] = []
        self.last_solar_burst_info: Dict[str, Any] = {"classification": "NONE"}
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
        self._last_live_wf_frames = -1
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
        """Fetch copy/kernel timings from Kotekan's /gpu_profile/* endpoints."""
        endpoints: List[str] = []
        try:
            ep_reply = http_get_json(self.base_url, "/endpoints", timeout=3.0)
            endpoints = sorted(
                e
                for e in (ep_reply.get("GET") or [])
                if isinstance(e, str) and e.startswith("/gpu_profile/")
            )
        except Exception:
            endpoints = []

        out: Dict[str, Any] = {}
        if endpoints:
            for ep in endpoints:
                key = ep.rsplit("/", 1)[-1]
                if key in out:
                    key = ep.removeprefix("/gpu_profile/").replace("/", "_")
                try:
                    gp = http_get_json(self.base_url, ep, timeout=3.0)
                    cin = sum(float(x.get("time") or 0.0) for x in gp.get("copy_in", []))
                    cout = sum(float(x.get("time") or 0.0) for x in gp.get("copy_out", []))
                    kern = sum(float(x.get("time") or 0.0) for x in gp.get("kernel", []))
                    out[key] = {
                        "copy_in_ms": round(cin * 1e3, 2),
                        "copy_out_ms": round(cout * 1e3, 2),
                        "kernel_ms": round(kern * 1e3, 2),
                    }
                except Exception:
                    pass
            return out

        for idx in (0, 1):
            for ep in (f"/gpu_profile/gpu/gpu_{idx}", f"/gpu_profile/stages/gpu_{idx}"):
                try:
                    gp = http_get_json(self.base_url, ep, timeout=3.0)
                    cin = sum(float(x.get("time") or 0.0) for x in gp.get("copy_in", []))
                    cout = sum(float(x.get("time") or 0.0) for x in gp.get("copy_out", []))
                    kern = sum(float(x.get("time") or 0.0) for x in gp.get("kernel", []))
                    out[f"gpu_{idx}"] = {
                        "copy_in_ms": round(cin * 1e3, 2),
                        "copy_out_ms": round(cout * 1e3, 2),
                        "kernel_ms": round(kern * 1e3, 2),
                    }
                    break
                except Exception:
                    continue
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

        qa = self.tailer.compute_qa(detrend_mode=self.detrend_mode)
        slow_occ = qa.pop("slow_channel_occupancy", None)
        recent_wf = qa.pop("_recent_clean_waterfall", None)
        recent_freqs = qa.pop("_recent_freqs_mhz", None)
        recent_times = qa.pop("_recent_times", None)
        qa.pop("_recent_detrended_rel", None)
        age_s = qa.get("last_frame_age_s")
        is_stale = age_s is not None and float(age_s) > 120.0
        qa["stale"] = is_stale
        if is_stale:
            cycle_actions.append(f"stale_stream(age={float(age_s):.0f}s)")

        watchdog_info = self.rfsoc_watchdog.check_and_recover(now, age_s)
        if watchdog_info.get("triggered"):
            wmsg = (
                f"rfsoc_watchdog(age={float(age_s):.1f}s,"
                f"ok={watchdog_info.get('recovery_ok')},{watchdog_info.get('message')})"
            )
            cycle_actions.append(wmsg)
            self._record_action(wmsg)

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

        # 3. Loop B: Selective latched-EMA recovery (only TIME_BURST/SIR at Gaussian SK & clean neighbors)
        ema_reset_freqs: List[float] = []
        if has_raw_channels:
            ema_reset_freqs = detect_latched_ema_channels(
                channels,
                self.latch_streaks,
                self.last_ema_reset_ts,
                now,
                reset_counts=self.ema_reset_counts,
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

        # 4. Loop C: Cluster/shoulder-aware persistent-RFI static-notch hysteresis (protected from Solar Radio Bursts)
        notch_info: Dict[str, Any] = {"managed_count": len(self.managed_notch_freqs)}
        if recent_wf is not None and recent_freqs is not None and recent_times is not None and recent_wf.shape[0] >= 4:
            try:
                from solar_burst_guard import classify_solar_radio_burst

                sun_el = float(env_snap.get("metrics", {}).get("sun_el_deg", qa.get("sun_el_deg", 0.0)) or 0.0) if "env_snap" in locals() else 0.0
                if sun_el > 10.0 and recent_wf.ndim == 3 and recent_wf.shape[1] >= 1:
                    sun_spec = recent_wf[:, 0, :]
                    off_spec = recent_wf[:, -1, :] if recent_wf.shape[1] >= 2 else None
                    sb_res = classify_solar_radio_burst(
                        sun_spec, off_spec, recent_freqs, recent_times, sun_el_deg=sun_el
                    )
                    self.last_solar_burst_info = sb_res
                    self.solar_protected_ranges_mhz = list(sb_res.get("protected_ranges_mhz", []))
                    if sb_res.get("is_solar_burst"):
                        msg = (
                            f"SOLAR_RADIO_BURST ({sb_res['burst_type']}, "
                            f"df/dt={sb_res['drift_rate_mhz_s']:.2f} MHz/s, "
                            f"protected={self.solar_protected_ranges_mhz})"
                        )
                        cycle_actions.append(msg)
                        self._record_action(msg)
                        http_post_json(
                            self.base_url,
                            "/rfi_mitigation/solar_burst_exempt",
                            {"ranges_mhz": self.solar_protected_ranges_mhz},
                        )
            except Exception:
                pass
        if self.auto_notch and has_raw_channels:
            existing_notches = rfi.get("static_notch_bands_mhz", [])
            new_notches, notch_changed, notch_info = update_managed_notches(
                channels,
                slow_occ,
                existing_notches,
                self.notch_bad_streaks,
                self.notch_clean_streaks,
                self.managed_notch_freqs,
                solar_protected_ranges_mhz=self.solar_protected_ranges_mhz,
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

        # 6. Loop E: Hardware/pipeline & outdoor environmental/LNA telemetry snapshot
        try:
            env_snap = self.env_collector.sample()
        except Exception as exc:
            env_snap = {
                "timestamp_unix": round(now, 3),
                "timestamp_utc": datetime.datetime.fromtimestamp(
                    now, datetime.timezone.utc
                ).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "metrics": {},
                "sources": {"collector": f"error: {exc}"},
            }

        overall_flagged_pct = round(
            100.0 * float(rfi.get("overall_flagged_chunk_fraction", 0.0)), 2
        )
        telemetry_row: Dict[str, Any] = {
            "timestamp_unix": env_snap.get("timestamp_unix", round(now, 3)),
            "timestamp_utc": env_snap.get("timestamp_utc"),
            "cycle": self.cycle_count,
            "sb0_median_power": qa.get("sb0_median_power"),
            "sb1_median_power": qa.get("sb1_median_power"),
            "sb0_common_pct": qa.get("sb0_common_pct"),
            "sb1_common_pct": qa.get("sb1_common_pct"),
            "sb0_tilt_pct_per_100mhz": qa.get("sb0_tilt_pct_per_100mhz"),
            "sb1_tilt_pct_per_100mhz": qa.get("sb1_tilt_pct_per_100mhz"),
            "detrended_rms_pct": qa.get("detrended_rms_pct"),
            "raw_residual_rms_pct": qa.get("raw_residual_rms_pct"),
            "rfi_flagged_pct": overall_flagged_pct,
            "active_antennas_count": len(ant.get("active_physical_antennas", [])),
            "gains_generation": trk.get("gains_generation"),
            "sb0_sb1_pearson_r": qa.get("sb0_sb1_pearson_r"),
            "watchdog_triggered": bool(watchdog_info.get("triggered", False)),
        }
        for k_qa, v_qa in qa.items():
            if k_qa.startswith(("adev_tau_", "hdev_tau_")):
                telemetry_row[k_qa] = v_qa
        telemetry_row.update(env_snap.get("metrics", {}))
        try:
            tpath = Path(self.telemetry_jsonl)
            tpath.parent.mkdir(parents=True, exist_ok=True)
            with open(tpath, "a", encoding="utf-8") as tf:
                tf.write(json.dumps(telemetry_row) + "\n")
                tf.flush()
        except Exception as exc:
            self._record_action(f"telemetry_log write error: {exc}")

        cur_total_frames = int(qa.get("total_frames") or 0)
        if (
            self.live_waterfall_png
            and recent_wf is not None
            and recent_freqs is not None
            and cur_total_frames > self._last_live_wf_frames
        ):
            try:
                trows = load_telemetry_records(self.telemetry_jsonl) if os.path.isfile(self.telemetry_jsonl) else None
                render_waterfall_figure(
                    recent_wf,
                    recent_freqs,
                    self.live_waterfall_png,
                    times=recent_times,
                    detrend_mode=self.detrend_mode,
                    clim_pct=self.clim_pct,
                    title=f"CHARTS Live ({qa.get('last_frame_utc') or 'now'}) — {self.detrend_mode}",
                    telemetry_rows=trows,
                )
                self._last_live_wf_frames = cur_total_frames
            except Exception as exc:
                self._record_action(f"live_waterfall error: {exc}")

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
            "telemetry_jsonl": self.telemetry_jsonl,
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
                "overall_flagged_pct": overall_flagged_pct,
                "static_notch_count": len(rfi.get("static_notch_bands_mhz", [])),
                "managed_notch_freqs_mhz": sorted(self.managed_notch_freqs),
                "rfi_gains": rfi_gain_info,
                "ema_reset_freqs_mhz": ema_reset_freqs,
                "notch_update": notch_info,
            },
            "waterfall_qa": qa,
            "environment": env_snap,
            "calibration_watch": cal_info,
            "rfsoc_watchdog": watchdog_info,
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
    detrend_mode: str = "subband_linear",
    env_collector: Optional[EnvironmentTelemetryCollector] = None,
    telemetry_jsonl: Optional[str] = None,
    live_waterfall_png: Optional[str] = None,
    clim_pct: float = 2.5,
    rfsoc_watchdog: Optional[RfsocRecoveryWatchdog] = None,
) -> None:
    print(
        f"Starting closed-loop tracker daemon (url={base_url}, run_dir={run_dir}, "
        f"interval={interval_s:.1f}s, detrend={detrend_mode}, auto_notch={auto_notch}, "
        f"status_json={status_json or 'auto'})",
        flush=True,
    )
    daemon = ClosedLoopDaemon(
        base_url=base_url,
        run_dir=run_dir,
        status_json=status_json,
        watch_gains=watch_gains,
        auto_notch=auto_notch,
        detrend_mode=detrend_mode,
        env_collector=env_collector,
        telemetry_jsonl=telemetry_jsonl,
        live_waterfall_png=live_waterfall_png,
        clim_pct=clim_pct,
        rfsoc_watchdog=rfsoc_watchdog,
    )
    while True:
        try:
            st = daemon.step()
            rfi = st["rfi"]
            qa = st["waterfall_qa"]
            pl = st["pipeline"]
            env_m = st.get("environment", {}).get("metrics", {})
            env_bits = []
            if "ambient_temp_c" in env_m:
                env_bits.append(f"Tamb={env_m['ambient_temp_c']}C")
            if "lna_v" in env_m:
                env_bits.append(f"LNA={env_m['lna_v']}V/{env_m.get('lna_i', '?')}A")
            elif "host_max_temp_c" in env_m:
                env_bits.append(f"Thost={env_m['host_max_temp_c']}C")
            env_tag = (" " + " ".join(env_bits)) if env_bits else ""
            acts = ",".join(st["cycle_actions"]) if st["cycle_actions"] else "steady"
            print(
                f"[{st['timestamp_utc']}] cycle={st['cycle']} "
                f"ants={len(pl['active_antennas'])}(dead={pl['dead_antennas']}) "
                f"RFI={rfi['flagged_channels']}/{rfi['total_channels']}({rfi['overall_flagged_pct']:.2f}%) "
                f"frames={qa.get('total_frames', 0)}(gaps={qa.get('seq_gaps', 0)}) "
                f"zero={qa.get('sb0_zero_pct', '?')}/{qa.get('sb1_zero_pct', '?')}% "
                f"CV={qa.get('bandpass_cv_pct', '?')}% detRMS={qa.get('detrended_rms_pct', '?')}% "
                f"cal_gen={pl['gains_generation']}{env_tag} acts={acts}",
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
                f"RMS={qa.get('raw_residual_rms_pct')}%->{qa.get('slow_rfi_cleaned_rms_pct')}% "
                f"det[{qa.get('detrend_mode', 'scalar')}]={qa.get('detrended_rms_pct')}% | "
                f"raw_api={drfi.get('raw_channels_api')} managed_notches={drfi.get('managed_notch_freqs_mhz')}"
            )
            env = snap.get("environment", {})
            if env:
                print(
                    f"  ENV ({env.get('timestamp_utc')}): metrics={env.get('metrics')} "
                    f"sources={env.get('sources')}"
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


def evaluate_mutual_coupling_diagnostics(
    cov_matrices: np.ndarray,
    positions_m: np.ndarray,
    weights: Optional[np.ndarray] = None,
    delays_ns: Optional[np.ndarray] = None,
    measured_lm: Optional[np.ndarray] = None,
    expected_lm: Optional[np.ndarray] = None,
    t_rec_k: float = 50.0,
    t_loss_k: float = 5.0,
    t_sky_k: float = 20.0,
) -> Dict[str, Any]:
    """Run CAL-2.4 mutual coupling noise (p_mc, T_mc) & azimuth/tilt self-calibration diagnostics."""
    from spatial_calibration import compute_mutual_coupling_noise, fit_array_azimuth_and_tilt

    mc = compute_mutual_coupling_noise(
        cov_matrices, weights=weights, t_rec_k=t_rec_k, t_loss_k=t_loss_k, t_sky_k=t_sky_k
    )
    geom: Dict[str, float] = {}
    if delays_ns is not None or (measured_lm is not None and expected_lm is not None):
        geom = fit_array_azimuth_and_tilt(
            positions_m, delays_ns=delays_ns, measured_lm=measured_lm, expected_lm=expected_lm
        )
    return {
        "mean_p_mc": mc["mean_p_mc"],
        "mean_t_mc_k": mc["mean_t_mc_k"],
        "max_t_mc_k": mc["max_t_mc_k"],
        "resonant_spike_indices": mc["resonant_spike_indices"],
        "geometry": geom,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Closed-loop self-tuning and verification utility for CHARTS Kotekan tracker."
    )
    parser.add_argument("--url", default="http://127.0.0.1:12048", help="Kotekan REST base URL")
    parser.add_argument("--analyze-run", metavar="DIR", help="Analyze beam_power_freq*.bin in DIR")
    parser.add_argument("--run-dir", metavar="DIR", help="Live run directory to tail in --daemon/--monitor")
    parser.add_argument("--old-gains", default=DEFAULT_OLD_GAINS, help="Baseline gains .npy path")
    parser.add_argument("--new-gains", default=DEFAULT_NEW_GAINS, help="Improved gains .npy path")
    parser.add_argument("--out-json", help="Optional JSON output path for --analyze-run/--correlate-telemetry")
    parser.add_argument(
        "--detrend-mode",
        choices=VALID_DETREND_MODES,
        default="subband_linear",
        help="Waterfall spectral-tilt detrending mode",
    )
    parser.add_argument(
        "--plot-waterfall",
        metavar="PNG_PATH",
        help="Render raw vs scalar vs detrended waterfall PNG during --analyze-run",
    )
    parser.add_argument(
        "--clim",
        type=float,
        default=2.5,
        help="Symmetric percent color limit (+/-CLIM%%) for detrended waterfall plots",
    )
    parser.add_argument(
        "--live-waterfall",
        metavar="PNG_PATH",
        help="Periodically render live detrended waterfall PNG during --daemon",
    )
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
    parser.add_argument("--telemetry-jsonl", help="Append aligned beam+env telemetry JSONL path in --daemon")
    parser.add_argument("--prometheus-url", help="Prometheus/node_exporter URL for outdoor temp telemetry")
    parser.add_argument("--lna-json-url", help="HTTP JSON endpoint for outdoor LNA bias/temp telemetry")
    parser.add_argument("--rigol-host", help="Rigol DP8xx host[:5555] for live SCPI CH1..CH3 V/I/W polling")
    parser.add_argument("--telemetry-csv", help="CSV/JSONL file or glob (e.g. Grafana pequen-aht10 export)")
    parser.add_argument(
        "--correlate-telemetry",
        nargs="?",
        const="__AUTO__",
        metavar="RUN_OR_LOG",
        help="Detect beam gain steps and correlate against environmental/LNA telemetry",
    )
    parser.add_argument("--watch-gains", help="Optional path or glob for new calibration .npy hot-reload")
    parser.add_argument(
        "--no-auto-notch",
        action="store_true",
        help="Disable automatic persistent-RFI narrowband static notch promotion",
    )
    parser.add_argument("--monitor", action="store_true", help="Print live pipeline + daemon QA status")
    parser.add_argument("--monitor-follow", action="store_true", help="Continuously refresh --monitor")
    parser.add_argument("--lna-alpha-db-per-c", type=float, default=0.0, help="LNA thermal gain-drift slope alpha_T (dB/C)")
    parser.add_argument("--lna-ref-temp-c", type=float, default=20.0, help="Reference ambient temperature T_0 (C)")
    parser.add_argument("--rfsoc-hosts", default="", help="Comma-separated RFSoC management IPs for staleness watchdog")
    parser.add_argument("--stale-recovery-s", type=float, default=10.0, help="Frame staleness (s) triggering RFSoC ping/ARP check")
    parser.add_argument("--suggest-off-beams", metavar="L,M", help="Suggest grating-lobe-safe OFF (l, m) beams for target L,M")

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
            args.correlate_telemetry,
            args.suggest_off_beams,
        ]
    ):
        parser.print_help()
        return 1

    if args.suggest_off_beams:
        from rfi_mitigation_tools import (
            FREQS_MHZ,
            PUDAHUEL_KNOWN_INTERFERERS,
            find_optimal_off_beams,
        )

        parts = [float(x.strip()) for x in args.suggest_off_beams.split(",")]
        if len(parts) != 2:
            print("--suggest-off-beams expects 'L,M' floats", file=sys.stderr)
            return 2
        cols, rows = np.meshgrid(np.arange(8), np.arange(8))
        pos_64 = np.column_stack([-cols.ravel() * 0.6, rows.ravel() * 0.6, np.zeros(64)])
        cands = find_optimal_off_beams(
            pos_64,
            FREQS_MHZ[::4],
            target_l=parts[0],
            target_m=parts[1],
            interferers=tuple(PUDAHUEL_KNOWN_INTERFERERS.values()),
            top_k=3,
        )
        print(json.dumps([c.__dict__ for c in cands], indent=2))
        if not any(
            [
                args.analyze_run,
                args.push_rfi_gains,
                args.reload_cal_gains,
                args.check_antennas,
                args.daemon,
                args.monitor,
                args.monitor_follow,
                args.correlate_telemetry,
            ]
        ):
            return 0

    env_collector = EnvironmentTelemetryCollector(
        prometheus_url=args.prometheus_url,
        lna_json_url=args.lna_json_url,
        rigol_host=args.rigol_host,
        telemetry_csv=args.telemetry_csv,
        alpha_t_db_per_c=args.lna_alpha_db_per_c,
        ref_temp_c=args.lna_ref_temp_c,
    )
    rfsoc_hosts = [h.strip() for h in args.rfsoc_hosts.split(",") if h.strip()]
    watchdog = RfsocRecoveryWatchdog(rfsoc_hosts=rfsoc_hosts, stale_threshold_s=args.stale_recovery_s)

    if args.analyze_run:
        analyze_run(
            args.analyze_run,
            old_gains_path=args.old_gains,
            new_gains_path=args.new_gains,
            out_json=args.out_json if not args.correlate_telemetry else None,
            detrend_mode=args.detrend_mode,
            plot_png=args.plot_waterfall,
            clim_pct=args.clim,
        )

    if args.correlate_telemetry:
        target = effective_run_dir if args.correlate_telemetry == "__AUTO__" else args.correlate_telemetry
        if not target:
            print("--correlate-telemetry requires a path or --run-dir/--analyze-run", file=sys.stderr)
            return 2
        correlate_environment_telemetry(
            target,
            telemetry_path=args.telemetry_csv,
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
            detrend_mode=args.detrend_mode,
            env_collector=env_collector,
            telemetry_jsonl=args.telemetry_jsonl,
            live_waterfall_png=args.live_waterfall,
            clim_pct=args.clim,
            rfsoc_watchdog=watchdog,
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
