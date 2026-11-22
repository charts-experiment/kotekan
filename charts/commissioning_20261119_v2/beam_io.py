#!/usr/bin/env python3
"""Shared binary reader, timing auditor, RFI masker, and Allan deviation helpers
for CHARTS Direct Beam Tracker commissioning verification scripts."""

from __future__ import annotations

import glob
import json
import os
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from charts_constants import C_LIGHT, DELTA_TIME, FREQ

FREQS_MHZ = np.asarray(FREQ, dtype=np.float64)
N_CHAN_SB = 336
N_CHAN_TOTAL = 672
FPGA_DT_S = float(DELTA_TIME)  # 10/3 us
C_LIGHT_MPS = float(C_LIGHT.value) if hasattr(C_LIGHT, "value") else float(C_LIGHT)

# Local ENU coordinates (m) of the 4 commissioned CHARTS antennas [0, 4, 32, 36]
POS4 = np.array(
    [
        [0.0, 0.0, 0.0],
        [-4.24775, 0.0275, 0.0],
        [0.15, 4.30625, 0.0],
        [-4.12725, 4.42725, 0.0],
    ],
    dtype=np.float64,
)


def load_beam_power_subband(
    run_dir: Path,
    sb_idx: int,
    n_beams: int,
    n_freq: int = N_CHAN_SB,
    max_frames: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read rawFileWrite beam_power_freq{sb_idx}_*.bin into (T, n_beams, n_freq), seq, time0_us."""
    files = sorted(glob.glob(str(run_dir / f"beam_power_freq{sb_idx}_*.bin")))
    if not files:
        raise FileNotFoundError(f"No beam_power_freq{sb_idx}_*.bin found in {run_dir}")

    payload_floats = n_beams * n_freq
    payload_bytes = payload_floats * 4

    # Probe first frame to verify uniform frame size for fast vectorized reading
    frames_list: List[np.ndarray] = []
    seq_list: List[int] = []
    t0_list: List[int] = []

    for fpath in files:
        fsize = os.path.getsize(fpath)
        if fsize < 8:
            continue
        with open(fpath, "rb") as f:
            hdr_sz_buf = f.read(4)
            if len(hdr_sz_buf) < 4:
                continue
            (meta_size,) = struct.unpack("<I", hdr_sz_buf)
            if meta_size < 16 or meta_size > 1_000_000:
                raise ValueError(f"Unexpected meta_size={meta_size} in {fpath}")
            rec_bytes = 4 + meta_size + payload_bytes
            if fsize % rec_bytes == 0 and meta_size >= 16:
                n_rec = fsize // rec_bytes
                if max_frames is not None:
                    rem = max_frames - len(seq_list)
                    if rem <= 0:
                        break
                    n_rec = min(n_rec, rem)
                f.seek(0)
                raw = np.fromfile(f, dtype=np.uint8, count=n_rec * rec_bytes)
                if raw.size < rec_bytes:
                    continue
                n_actual = raw.size // rec_bytes
                recs = raw[: n_actual * rec_bytes].reshape(n_actual, rec_bytes)
                # Verify all meta_sizes match
                msizes = recs[:, :4].view("<u4")[:, 0]
                if np.all(msizes == meta_size):
                    seqs = recs[:, 12:20].view("<i8")[:, 0].copy()
                    t0s = recs[:, 4 + meta_size - 8 : 4 + meta_size].view("<u8")[:, 0].copy()
                    arr = recs[:, 4 + meta_size :].view("<f4").reshape(n_actual, n_beams, n_freq).copy()
                    frames_list.append(arr)
                    seq_list.extend(seqs.tolist())
                    t0_list.extend(t0s.tolist())
                    if max_frames is not None and len(seq_list) >= max_frames:
                        break
                    continue

            # Fallback frame-by-frame reader if record sizes vary
            f.seek(0)
            while True:
                if max_frames is not None and len(seq_list) >= max_frames:
                    break
                sz_buf = f.read(4)
                if len(sz_buf) < 4:
                    break
                (msz,) = struct.unpack("<I", sz_buf)
                if msz < 16 or msz > 1_000_000:
                    break
                meta = f.read(msz)
                if len(meta) < msz:
                    break
                pay = f.read(payload_bytes)
                if len(pay) < payload_bytes:
                    break
                seq_num = struct.unpack_from("<q", meta, 8)[0]
                t0_us = struct.unpack_from("<Q", meta, msz - 8)[0]
                arr = np.frombuffer(pay, dtype="<f4").reshape(n_beams, n_freq).copy()
                frames_list.append(arr[None, :, :])
                seq_list.append(seq_num)
                t0_list.append(t0_us)

    if not frames_list:
        raise ValueError(f"No readable frames in {run_dir} for subband {sb_idx}")

    data = np.concatenate(frames_list, axis=0)
    return data, np.asarray(seq_list, dtype=np.int64), np.asarray(t0_list, dtype=np.uint64)


def load_dual_subband_beam_power(
    run_dir: Path,
    n_beams: int,
    max_frames: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load and seq-align subband 0 & 1 beam power waterfalls into (T, n_beams, 672)."""
    d0, seq0, t0_0 = load_beam_power_subband(run_dir, 0, n_beams, max_frames=max_frames)
    d1, seq1, t0_1 = load_beam_power_subband(run_dir, 1, n_beams, max_frames=max_frames)

    # Align by exact fpga_seq_num intersection
    common_seq, idx0, idx1 = np.intersect1d(seq0, seq1, return_indices=True)
    if len(common_seq) == 0:
        # Fallback if seq offsets differ by a fixed startup frame offset
        n_t = min(len(seq0), len(seq1))
        wf = np.concatenate([d0[:n_t], d1[:n_t]], axis=2)
        seq_out = seq0[:n_t]
        t_out = t0_0[:n_t]
        dt_sb_us = t0_1[:n_t].astype(np.int64) - t0_0[:n_t].astype(np.int64)
        aligned_by_seq = False
    else:
        wf = np.concatenate([d0[idx0], d1[idx1]], axis=2)
        seq_out = common_seq
        t_out = t0_0[idx0]
        dt_sb_us = t0_1[idx1].astype(np.int64) - t0_0[idx0].astype(np.int64)
        aligned_by_seq = True

    dseq0 = np.diff(seq0)
    dseq1 = np.diff(seq1)
    med_dseq = int(np.median(dseq0)) if len(dseq0) > 0 else 0
    gaps0 = int(np.sum(dseq0 != med_dseq)) if len(dseq0) > 0 else 0
    gaps1 = int(np.sum(dseq1 != med_dseq)) if len(dseq1) > 0 else 0
    dt_frame_s = med_dseq * FPGA_DT_S if med_dseq > 0 else 0.0
    dt0_us = np.diff(t0_0.astype(np.int64))

    timing_stats = {
        "n_frames_sb0": int(len(seq0)),
        "n_frames_sb1": int(len(seq1)),
        "n_frames_aligned": int(len(seq_out)),
        "aligned_by_exact_fpga_seq": aligned_by_seq,
        "median_dseq": med_dseq,
        "cadence_s": round(dt_frame_s, 6),
        "seq_gaps_sb0": gaps0,
        "seq_gaps_sb1": gaps1,
        "max_abs_subband_dt_us": int(np.max(np.abs(dt_sb_us))) if len(dt_sb_us) else 0,
        "median_subband_dt_us": float(np.median(dt_sb_us)) if len(dt_sb_us) else 0.0,
        "median_time0_step_us": float(np.median(dt0_us)) if len(dt0_us) else 0.0,
        "start_time0_unix": float(t_out[0]) * 1e-6 if len(t_out) else 0.0,
        "end_time0_unix": float(t_out[-1]) * 1e-6 if len(t_out) else 0.0,
    }
    return wf, seq_out, t_out, timing_stats


def clean_channel_mask(
    wf_2d: np.ndarray,
    *,
    max_zero_frac: float = 0.05,
    mad_sigma: float = 5.0,
) -> np.ndarray:
    """Identify clean spectral channels in a (T, 672) waterfall."""
    # Exclude wholly-dropped startup/shutdown frames before assessing per-channel dropout
    good_t = np.mean(np.isfinite(wf_2d) & (wf_2d > 0.0), axis=1) > 0.30
    sub = wf_2d[good_t] if np.any(good_t) else wf_2d

    finite_ok = np.mean(np.isfinite(sub), axis=0) > 0.95
    zero_frac = np.mean(sub <= 0.0, axis=0)

    # Subsample long high-rate waterfalls (>8k frames) so nanmedian/std finish in <0.3 s
    stride = max(1, sub.shape[0] // 4000) if sub.shape[0] > 8000 else 1
    sub_stat = sub[::stride]

    with np.errstate(all="ignore"):
        med_spec = np.nanmedian(np.where(sub_stat > 0.0, sub_stat, np.nan), axis=0)
    valid = finite_ok & (zero_frac <= max_zero_frac) & np.isfinite(med_spec) & (med_spec > 0.0)
    if int(np.sum(valid)) < 20:
        return valid

    # Exclude known MUOS band 356-384 MHz from clean radiometer baseline
    valid &= ~((FREQS_MHZ >= 355.5) & (FREQS_MHZ <= 384.5))

    # Spectral MAD relative to sliding local median (width 21 ch ~ 6.3 MHz)
    n_f = sub_stat.shape[1]
    env = np.copy(med_spec)
    for f in range(n_f):
        lo, hi = max(0, f - 10), min(n_f, f + 11)
        vals = med_spec[lo:hi]
        ok_vals = vals[np.isfinite(vals) & valid[lo:hi]]
        if len(ok_vals) > 0:
            env[f] = float(np.median(ok_vals))
    ratio = med_spec / np.maximum(env, 1e-12)
    r_valid = ratio[valid]
    r_med = float(np.median(r_valid))
    r_mad = 1.4826 * float(np.median(np.abs(r_valid - r_med)))
    valid &= np.abs(ratio - r_med) <= mad_sigma * max(r_mad, 1e-3)

    # Temporal stability check (relative RMS of normalized lightcurve on non-zero samples)
    norm_ts = np.where(sub_stat > 0.0, sub_stat, med_spec[None, :]) / np.maximum(med_spec[None, :], 1e-12)
    t_rms = np.std(norm_ts, axis=0)
    rms_v = t_rms[valid]
    if len(rms_v) > 10:
        rms_med = float(np.median(rms_v))
        rms_mad = 1.4826 * float(np.median(np.abs(rms_v - rms_med)))
        valid &= t_rms <= (rms_med + mad_sigma * max(rms_mad, 1e-3))
    return valid


def atomic_write_text(path: Path, text: str) -> None:
    """Write text via temp file + os.replace so 0777 dirs can replace root-owned outputs."""
    tmp = path.with_name(f".tmp.{os.getpid()}.{path.name}")
    tmp.write_text(text, encoding="utf-8")
    os.chmod(tmp, 0o644)
    os.replace(tmp, path)


def atomic_savefig(fig: Any, path: Path, dpi: int = 140) -> None:
    """Save a matplotlib figure via temp file + os.replace."""
    tmp = path.with_name(f".tmp.{os.getpid()}.{path.name}")
    fig.savefig(tmp, dpi=dpi)
    os.chmod(tmp, 0o644)
    os.replace(tmp, path)


def overlapping_allan_deviation(
    y: np.ndarray,
    dt_s: float,
    taus_s: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute overlapping Allan deviation sigma_y(tau) in O(N) per lag."""
    y = np.asarray(y, dtype=np.float64)
    y = y[np.isfinite(y)]
    n = len(y)
    if taus_s is None:
        max_m = max(1, n // 3)
        ms = np.unique(np.round(np.geomspace(1, max_m, 30)).astype(int))
    else:
        ms = np.unique(np.clip(np.round(np.asarray(taus_s) / dt_s).astype(int), 1, max(1, n // 3)))

    csum = np.concatenate([[0.0], np.cumsum(y)])
    out_taus: List[float] = []
    out_adev: List[float] = []
    for m in ms:
        if 2 * m >= n:
            continue
        # m-sample averages: ybar[k] = (csum[k+m] - csum[k]) / m for k = 0 .. N-m
        ybar = (csum[m:] - csum[:-m]) / float(m)
        diffs = ybar[m:] - ybar[:-m]
        if len(diffs) < 2:
            continue
        avar = 0.5 * float(np.mean(diffs * diffs))
        out_taus.append(m * dt_s)
        out_adev.append(float(np.sqrt(max(avar, 0.0))))
    return np.asarray(out_taus), np.asarray(out_adev)


def overlapping_hadamard_deviation(
    y: np.ndarray,
    dt_s: float,
    taus_s: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute overlapping Hadamard deviation (invariant to linear drift slope) in O(N) per lag."""
    y = np.asarray(y, dtype=np.float64)
    y = y[np.isfinite(y)]
    n = len(y)
    if taus_s is None:
        max_m = max(1, (n - 1) // 3)
        ms = np.unique(np.round(np.geomspace(1, max_m, 30)).astype(int))
    else:
        ms = np.unique(np.clip(np.round(np.asarray(taus_s) / dt_s).astype(int), 1, max(1, (n - 1) // 3)))

    x = np.concatenate([[0.0], np.cumsum(y) * dt_s])
    out_taus: List[float] = []
    out_hdev: List[float] = []
    for m in ms:
        if 3 * m >= len(x):
            continue
        tau = m * dt_s
        d3 = x[3 * m :] - 3.0 * x[2 * m : -m] + 3.0 * x[m : -2 * m] - x[: -3 * m]
        if len(d3) < 2:
            continue
        hvar = float(np.mean(d3 * d3)) / (6.0 * tau * tau)
        out_taus.append(tau)
        out_hdev.append(float(np.sqrt(max(hvar, 0.0))))
    return np.asarray(out_taus), np.asarray(out_hdev)


def theoretical_array_factor_sq(
    l_off: float,
    m_off: float,
    l_src: float,
    m_src: float,
    freqs_mhz: np.ndarray,
    pos: np.ndarray = POS4,
) -> np.ndarray:
    """Normalized power array factor |mean(exp(i k r.(s_off - s_src)))|^2 vs frequency."""
    n_off = float(np.sqrt(max(0.0, 1.0 - l_off * l_off - m_off * m_off)))
    n_src = float(np.sqrt(max(0.0, 1.0 - l_src * l_src - m_src * m_src)))
    ds = np.array([l_off - l_src, m_off - m_src, n_off - n_src], dtype=np.float64)
    delays = np.asarray(pos, dtype=np.float64) @ ds
    k = 2.0 * np.pi * np.asarray(freqs_mhz, dtype=np.float64) * 1e6 / C_LIGHT_MPS
    af = np.mean(np.exp(1j * k[:, None] * delays[None, :]), axis=1)
    return np.abs(af) ** 2


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                try:
                    rows.append(json.loads(s))
                except json.JSONDecodeError:
                    pass
    return rows
