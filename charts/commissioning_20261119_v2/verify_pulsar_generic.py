#!/usr/bin/env python3
"""Post-test verification for bright pulsars (Vela PSR B0833-45,
PSR B0950+08, PSR B1749-28): GPSDO/1PPS subband sync, 2D (P, DM) folding,
300-501.3 MHz channel-order validation, and 4-subband chromatic profiles."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import astropy.units as u
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time

from beam_io import (
    FREQS_MHZ,
    atomic_savefig,
    atomic_write_text,
    clean_channel_mask,
    load_dual_subband_beam_power,
    load_jsonl,
)
from charts_constants import CHARTS_ALTITUDE_M, CHARTS_LATITUDE_DEG, CHARTS_LONGITUDE_DEG, K_DM

K_DM_VAL = float(K_DM.value)

PULSAR_CATALOG = {
    "vela": {
        "name": "PSR B0833-45 (Vela)",
        "ra": "08h35m20.61149s",
        "dec": "-45d10m34.8751s",
        "p0_bary_s": 0.089328385024,
        "pepoch_mjd": 51559.319,
        "pdot": 1.25008e-13,
        "dm_cat": 67.771,
        "dm_search_span": (52.0, 84.0),
    },
    "b0950": {
        "name": "PSR B0950+08",
        "ra": "09h53m09.3097s",
        "dec": "+07d55m35.75s",
        "p0_bary_s": 0.253065164948,
        "pepoch_mjd": 46375.0,
        "pdot": 2.297e-16,
        "dm_cat": 2.972,
        "dm_search_span": (0.0, 15.0),
    },
    "b1749": {
        "name": "PSR B1749-28",
        "ra": "17h52m58.6896s",
        "dec": "-28d06m37.3s",
        "p0_bary_s": 0.56255794132,
        "pepoch_mjd": 46483.0,
        "pdot": 8.128e-15,
        "dm_cat": 50.88,
        "dm_search_span": (35.0, 68.0),
    },
}


def normalize_clean_beam(wf_b_clean: np.ndarray) -> np.ndarray:
    """Normalize a single beam's clean channels (T, C) in float32 zero-safe form."""
    v = wf_b_clean.astype(np.float32, copy=False)
    pos = np.isfinite(v) & (v > 0.0)
    cnt = np.maximum(pos.sum(axis=0), 1)
    v_pos = np.where(pos, v, 0.0)
    mu = v_pos.sum(axis=0) / cnt
    var = np.where(pos, (v - mu[None, :]) ** 2, 0.0).sum(axis=0) / cnt
    sd = np.maximum(np.sqrt(var), 1e-6)
    return np.where(pos, np.clip((v - mu[None, :]) / sd[None, :], -8.0, 8.0), 0.0).astype(np.float32)


def dedisperse_1d(
    norm_tc: np.ndarray,
    freqs_ch: np.ndarray,
    dm: float,
    cadence_s: float,
    ch_submask: np.ndarray | None = None,
) -> np.ndarray:
    """Dedisperse normalized (T, C) waterfall into a 1D time series at integer sample shifts."""
    src = norm_tc if ch_submask is None else norm_tc[:, ch_submask]
    f_sub = freqs_ch if ch_submask is None else freqs_ch[ch_submask]
    n_t, n_c = src.shape
    if n_c == 0 or n_t == 0:
        return np.zeros(n_t, dtype=np.float64)
    f_ref = float(np.max(FREQS_MHZ))
    shifts = np.round(K_DM_VAL * dm * (f_sub ** -2 - f_ref ** -2) / cadence_s).astype(np.int64)
    ts = np.zeros(n_t, dtype=np.float64)
    for k, sh in enumerate(shifts):
        if sh == 0:
            ts += src[:, k]
        elif 0 < sh < n_t:
            ts[:-sh] += src[sh:, k]
        elif -n_t < sh < 0:
            ts[-sh:] += src[:sh, k]
    return ts / np.sqrt(float(n_c))


def fold_1d(ts: np.ndarray, t_rel_s: np.ndarray, period_s: float, n_bins: int = 32) -> Tuple[np.ndarray, float]:
    """Fold 1D dedispersed time series at period_s and return (profile, off-pulse-rms SNR)."""
    bins = np.clip((np.mod(t_rel_s / period_s, 1.0) * n_bins).astype(np.int64), 0, n_bins - 1)
    s = np.bincount(bins, weights=ts, minlength=n_bins)
    c = np.bincount(bins, minlength=n_bins)
    prof = s / np.maximum(c, 1.0)
    med = float(np.median(prof))
    off = np.sort(prof)[: max(4, int(0.70 * n_bins))]
    off_rms = float(np.std(off)) if np.std(off) > 0 else float(np.std(prof))
    snr = float((np.max(prof) - med) / max(off_rms, 1e-12))
    return prof, snr


def fold_dedispersed_channels(
    norm_tc: np.ndarray,
    t_rel_s: np.ndarray,
    freqs_ch: np.ndarray,
    period_s: float,
    dm: float,
    n_bins: int = 32,
) -> np.ndarray:
    """Compute (C, n_bins) dedispersed phase-vs-frequency array for plotting once at best (P, DM)."""
    n_c = norm_tc.shape[1]
    subband_prof = np.zeros((n_c, n_bins), dtype=np.float64)
    f_ref = float(np.max(FREQS_MHZ))
    delays_s = K_DM_VAL * dm * (freqs_ch ** -2 - f_ref ** -2)
    ph0 = t_rel_s / period_s
    for k in range(n_c):
        bk = np.clip((np.mod(ph0 - delays_s[k] / period_s, 1.0) * n_bins).astype(np.int64), 0, n_bins - 1)
        ck = np.bincount(bk, minlength=n_bins)
        sk = np.bincount(bk, weights=norm_tc[:, k], minlength=n_bins)
        subband_prof[k] = np.where(ck > 0, sk / np.maximum(ck, 1.0), 0.0)
    return subband_prof


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--target", default="auto", choices=["auto", "vela", "b0950", "b1749"])
    p.add_argument("--n-bins", type=int, default=32)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    tgt_key = args.target
    if tgt_key == "auto":
        nm = run_dir.name.lower()
        if "0950" in nm:
            tgt_key = "b0950"
        elif "1749" in nm:
            tgt_key = "b1749"
        else:
            tgt_key = "vela"
    cat = PULSAR_CATALOG[tgt_key]

    wf, seq, t0_us, timing = load_dual_subband_beam_power(run_dir, n_beams=3)
    n_t_raw = wf.shape[0]
    cadence_s = timing["cadence_s"] if timing["cadence_s"] > 0 else 0.00256

    good_t = np.mean(np.isfinite(wf[:, 0, :]) & (wf[:, 0, :] > 0.0), axis=1) > 0.30
    dropped_whole_frames = int(n_t_raw - np.sum(good_t))
    if np.sum(good_t) > 10 and dropped_whole_frames > 0:
        wf = wf[good_t]
        seq_valid = seq[good_t]
    else:
        seq_valid = seq
    n_t = wf.shape[0]
    t_rel_s = (seq_valid - seq[0]).astype(np.float64) * (10.0 / 3.0) * 1e-6

    telem = load_jsonl(run_dir / "telemetry_log.jsonl")
    mask0 = clean_channel_mask(wf[:, 0, :], max_zero_frac=0.10, mad_sigma=5.0)
    mask2 = clean_channel_mask(wf[:, 2, :], max_zero_frac=0.10, mad_sigma=5.0)
    clean_mask = mask0 & mask2
    if int(np.sum(clean_mask)) < 30:
        clean_mask = np.mean(np.isfinite(wf[:, 0, :]) & (wf[:, 0, :] > 0.0), axis=0) > 0.85

    ch_idx = np.where(clean_mask)[0]
    f_ch = FREQS_MHZ[ch_idx]
    norm0 = normalize_clean_beam(wf[:, 0, ch_idx])
    norm1 = normalize_clean_beam(wf[:, 1, ch_idx])
    norm2 = normalize_clean_beam(wf[:, 2, ch_idx])

    mid_unix = 0.5 * (timing["start_time0_unix"] + timing["end_time0_unix"])
    if mid_unix <= 0:
        mid_unix = datetime.now(timezone.utc).timestamp()
    t_mid = Time(mid_unix, format="unix")
    loc = EarthLocation(lat=CHARTS_LATITUDE_DEG * u.deg, lon=CHARTS_LONGITUDE_DEG * u.deg, height=CHARTS_ALTITUDE_M * u.m)
    sc = SkyCoord(cat["ra"], cat["dec"], frame="icrs")
    v_corr_mps = float(sc.radial_velocity_correction(obstime=t_mid, location=loc).to(u.m / u.s).value)
    p_bary_obs_s = cat["p0_bary_s"] + cat["pdot"] * (float(t_mid.mjd) - cat["pepoch_mjd"]) * 86400.0
    p_topo_pred_s = p_bary_obs_s * (1.0 - v_corr_mps / 299792458.0)

    dm_cat = cat["dm_cat"]
    dur_s = float(n_t * cadence_s)
    is_smoke = dur_s < 30.0
    ts_dm_cache: Dict[float, np.ndarray] = {}
    if is_smoke:
        best_p = p_topo_pred_s
        best_dm = dm_cat
        dm_grid = np.array([0.0, dm_cat])
        p_grid = np.array([p_topo_pred_s])
        snr_grid = np.zeros((2, 1))
        for di, dm in enumerate(dm_grid):
            ts_dm = dedisperse_1d(norm0, f_ch, float(dm), cadence_s)
            ts_dm_cache[round(float(dm), 3)] = ts_dm
            _, snr_grid[di, 0] = fold_1d(ts_dm, t_rel_s, best_p, n_bins=args.n_bins)
    else:
        dm_lo, dm_hi = cat["dm_search_span"]
        dm_coarse = np.linspace(0.0, max(100.0, dm_hi + 20.0), 11)
        dm_fine = np.linspace(dm_lo, dm_hi, 17)
        dm_grid = np.unique(np.round(np.concatenate([dm_coarse, dm_fine, [dm_cat]]), 3))
        p_span = max(3.0e-5, 3.0e-4 * p_bary_obs_s)
        coarse_bins = 16
        # Nyquist-safe period step: <= 0.45 * P^2 / (coarse_bins * T_obs)
        dp_coarse = min(6.0e-7, max(2.0e-7, 0.45 * (p_bary_obs_s ** 2) / (coarse_bins * max(dur_s, 10.0))))
        n_p = int(np.clip(round(2.0 * p_span / dp_coarse) + 1, 41, 201))
        if n_p % 2 == 0:
            n_p += 1
        p_grid = p_topo_pred_s + np.linspace(-p_span, p_span, n_p)
        snr_grid = np.zeros((len(dm_grid), len(p_grid)), dtype=np.float64)
        for di, dm in enumerate(dm_grid):
            dm_key = round(float(dm), 3)
            ts_dm = dedisperse_1d(norm0, f_ch, float(dm), cadence_s)
            ts_dm_cache[dm_key] = ts_dm
            for pi, per in enumerate(p_grid):
                _, snr_grid[di, pi] = fold_1d(ts_dm, t_rel_s, float(per), n_bins=coarse_bins)

        # Fine refinement around top coarse candidates + catalog prediction at full n_bins
        cand_list = []
        flat_top = np.argsort(snr_grid.ravel())[::-1][:8]
        for idx_f in flat_top:
            ci, cj = np.unravel_index(int(idx_f), snr_grid.shape)
            cand_list.append((float(dm_grid[ci]), float(p_grid[cj])))
        cand_list.append((float(dm_cat), float(p_topo_pred_s)))

        best_snr = -1.0
        best_dm = float(dm_cat)
        best_p = float(p_topo_pred_s)
        dp_fine = max(5.0e-8, 0.15 * (p_bary_obs_s ** 2) / (args.n_bins * max(dur_s, 10.0)))
        for c_dm, c_p in cand_list:
            ts_c = ts_dm_cache[round(c_dm, 3)]
            for pf in c_p + np.linspace(-3.0 * dp_coarse, 3.0 * dp_coarse, max(25, int(round(6.0 * dp_coarse / dp_fine)) + 1)):
                _, sn_f = fold_1d(ts_c, t_rel_s, float(pf), n_bins=args.n_bins)
                if sn_f > best_snr:
                    best_snr = sn_f
                    best_dm = c_dm
                    best_p = float(pf)

    ts0_best = ts_dm_cache.get(round(best_dm, 3))
    if ts0_best is None:
        ts0_best = dedisperse_1d(norm0, f_ch, best_dm, cadence_s)
    ts1_best = dedisperse_1d(norm1, f_ch, best_dm, cadence_s)
    ts2_best = dedisperse_1d(norm2, f_ch, best_dm, cadence_s)
    prof_cal, snr_cal = fold_1d(ts0_best, t_rel_s, best_p, n_bins=args.n_bins)
    prof_uncal, snr_uncal = fold_1d(ts1_best, t_rel_s, best_p, n_bins=args.n_bins)
    prof_off, snr_off = fold_1d(ts2_best, t_rel_s, best_p, n_bins=args.n_bins)
    subband_cal = fold_dedispersed_channels(norm0, t_rel_s, f_ch, best_p, best_dm, n_bins=args.n_bins)

    # Channel-order & dispersion sign tests evaluated at best_p and dm_cat
    ts_norm = ts_dm_cache.get(round(float(dm_cat), 3))
    if ts_norm is None:
        ts_norm = dedisperse_1d(norm0, f_ch, dm_cat, cadence_s)
        ts_dm_cache[round(float(dm_cat), 3)] = ts_norm
    _, snr_cat_normal = fold_1d(ts_norm, t_rel_s, best_p, n_bins=args.n_bins)
    f_rev_all = FREQS_MHZ[::-1]
    _, snr_cat_rev = fold_1d(dedisperse_1d(norm0, f_rev_all[ch_idx], dm_cat, cadence_s), t_rel_s, best_p, n_bins=args.n_bins)
    f_swap_all = np.concatenate([FREQS_MHZ[336:], FREQS_MHZ[:336]])
    _, snr_cat_swap = fold_1d(dedisperse_1d(norm0, f_swap_all[ch_idx], dm_cat, cadence_s), t_rel_s, best_p, n_bins=args.n_bins)
    ts_dm0 = ts_dm_cache.get(0.0)
    if ts_dm0 is None:
        ts_dm0 = dedisperse_1d(norm0, f_ch, 0.0, cadence_s)
        ts_dm_cache[0.0] = ts_dm0
    _, snr_dm0 = fold_1d(ts_dm0, t_rel_s, best_p, n_bins=args.n_bins)

    # DM curve at best_p reusing cached dedispersed time series
    dm_curve: Dict[str, float] = {}
    for dm_pt in sorted(set(ts_dm_cache.keys())):
        _, sn_dm = fold_1d(ts_dm_cache[dm_pt], t_rel_s, best_p, n_bins=args.n_bins)
        dm_curve[str(dm_pt)] = round(sn_dm, 3)

    # 2-subband & 4-subband chromatic alignment evaluated at (best_p, dm_cat)
    subband_alignment: Dict[str, Any] = {}
    for sname, s_mask in [("low_300_400", f_ch < 400.65), ("high_400_501", f_ch >= 400.65)]:
        if np.any(s_mask):
            pr_s, sn_s = fold_1d(dedisperse_1d(norm0, f_ch, dm_cat, cadence_s, s_mask), t_rel_s, best_p, n_bins=args.n_bins)
            subband_alignment[sname] = {"n_ch": int(np.sum(s_mask)), "snr": round(sn_s, 3), "peak_bin": int(np.argmax(pr_s))}

    chromatic_4sb: Dict[str, Any] = {}
    sb_edges = [(300.0, 350.0), (350.0, 400.65), (400.65, 450.0), (450.0, 502.0)]
    profs_4sb = {}
    for f0, f1 in sb_edges:
        lbl = f"{int(f0)}_{int(f1)}MHz"
        m_sb = (f_ch >= f0) & (f_ch < f1)
        if np.any(m_sb):
            pr, sn = fold_1d(dedisperse_1d(norm0, f_ch, dm_cat, cadence_s, m_sb), t_rel_s, best_p, n_bins=args.n_bins)
            profs_4sb[lbl] = (pr, sn)
            chromatic_4sb[lbl] = {"n_ch": int(np.sum(m_sb)), "snr": round(sn, 3), "peak_bin": int(np.argmax(pr))}

    summary = {
        "run_dir": str(run_dir),
        "target": cat["name"],
        "target_key": tgt_key,
        "is_smoke_test": bool(is_smoke),
        "duration_s": round(float(n_t * cadence_s), 3),
        "dropped_whole_frames": dropped_whole_frames,
        "clean_channels": int(np.sum(clean_mask)),
        "timing_gpsdo": timing,
        "predicted_bary_period_s": round(p_bary_obs_s, 9),
        "predicted_topo_period_s": round(p_topo_pred_s, 9),
        "best_period_s": round(best_p, 9),
        "best_dm_pc_cm3": round(best_dm, 3),
        "catalog_dm_pc_cm3": dm_cat,
        "snr_beam0_cal_on": round(snr_cal, 3),
        "snr_beam1_uncal_on": round(snr_uncal, 3),
        "snr_beam2_off_cal": round(snr_off, 3),
        "subband_alignment_at_cat_dm": subband_alignment,
        "channel_order_check_at_cat_dm": {
            "normal_300_to_501_snr": round(snr_cat_normal, 3),
            "reversed_501_to_300_snr": round(snr_cat_rev, 3),
            "subband_swapped_snr": round(snr_cat_swap, 3),
            "dm0_unsmeared_snr": round(snr_dm0, 3),
        },
        "dm_curve_snr": dm_curve,
        "chromatic_4_subbands": chromatic_4sb,
        "mean_band_power": {
            "beam0_cal": round(float(np.mean(wf[:, 0, clean_mask])), 4),
            "beam1_uncal": round(float(np.mean(wf[:, 1, clean_mask])), 4),
            "beam2_off": round(float(np.mean(wf[:, 2, clean_mask])), 4),
        },
        "telemetry_samples": len(telem),
    }
    atomic_write_text(run_dir / "pulsar_verification.json", json.dumps(summary, indent=2))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    phase_axis = np.linspace(0, 1, args.n_bins, endpoint=False)
    axes[0, 0].plot(phase_axis, prof_cal, lw=2.0, color="#2563eb", label=f"Beam 0 Cal (SNR={snr_cal:.2f})")
    axes[0, 0].plot(phase_axis, prof_uncal, lw=1.4, color="#f59e0b", label=f"Beam 1 Uncal (SNR={snr_uncal:.2f})")
    axes[0, 0].plot(phase_axis, prof_off, lw=1.4, color="#64748b", ls="--", label=f"Beam 2 OFF (SNR={snr_off:.2f})")
    axes[0, 0].set_title(f"{cat['name']} Folded Profile (P={best_p*1e3:.4f} ms, DM={best_dm:.2f})")
    axes[0, 0].set_xlabel("Pulse Phase")
    axes[0, 0].set_ylabel("Normalized Amplitude")
    axes[0, 0].grid(True, alpha=0.25)
    axes[0, 0].legend(fontsize=8.5)

    if snr_grid.shape[1] > 1:
        im = axes[0, 1].imshow(
            snr_grid,
            aspect="auto",
            origin="lower",
            extent=[p_grid[0] * 1e3, p_grid[-1] * 1e3, dm_grid[0], dm_grid[-1]],
            cmap="viridis",
        )
        fig.colorbar(im, ax=axes[0, 1], label="Fold SNR")
        axes[0, 1].axhline(dm_cat, color="white", ls="--", lw=1.0, label=f"Cat DM={dm_cat}")
        axes[0, 1].axvline(best_p * 1e3, color="#fde047", ls=":", lw=1.0, label=f"Best P={best_p*1e3:.4f} ms")
        axes[0, 1].set_title("2D (P, DM) Search SNR")
        axes[0, 1].set_xlabel("Period (ms)")
        axes[0, 1].set_ylabel("DM (pc cm^-3)")
        axes[0, 1].legend(fontsize=8.5)
    else:
        axes[0, 1].bar(["Normal", "Reversed", "Swapped", "DM=0"], [snr_cat_normal, snr_cat_rev, snr_cat_swap, snr_dm0])
        axes[0, 1].set_title("Channel-Order & Dedispersion Check")

    axes[1, 0].imshow(
        subband_cal,
        aspect="auto",
        origin="lower",
        extent=[0, 1, float(f_ch.min()), float(f_ch.max())],
        cmap="magma",
        vmin=-0.3,
        vmax=0.3,
    )
    axes[1, 0].set_title("Dedispersed Phase vs Frequency (Beam 0)")
    axes[1, 0].set_xlabel("Pulse Phase")
    axes[1, 0].set_ylabel("Frequency (MHz)")

    for lbl, (pr, sn) in profs_4sb.items():
        axes[1, 1].plot(phase_axis, pr, lw=1.3, label=f"{lbl} (SNR={sn:.2f})")
    axes[1, 1].set_title("4-Subband Chromatic Pulse Profile Alignment")
    axes[1, 1].set_xlabel("Pulse Phase")
    axes[1, 1].set_ylabel("Normalized Amplitude")
    axes[1, 1].grid(True, alpha=0.25)
    axes[1, 1].legend(fontsize=8.0)

    atomic_savefig(fig, run_dir / "pulsar_verification.png", dpi=140)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
