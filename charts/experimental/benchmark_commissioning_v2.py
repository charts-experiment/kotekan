"""Benchmark experimental modules on real commissioning_20261119_v2 captures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import welch

from . import (
    FREQS_MHZ,
    K_DM_PC_CM3_MHZ2_S,
    N_CHAN_SB,
    N_CHAN_TOTAL,
    load_antenna_positions_from_yaml,
    load_beam_power_subband,
    load_dual_subband_beam_power,
    load_pointing_log,
    load_rfi_snapshots,
)
from .auto_rfi_transient_detector import (
    CoherentCombCanceller,
    MatchedFilterTransientScorer,
    UnsupervisedRfiClassifier,
    extract_rfi_snapshot_features,
    fold_profile,
)
from .rfi_lm_profiler import MultiFrequencyLmProfiler, find_optimal_off_beams
from .spatial_calibration import AdaptiveSidelobeCanceller, BeamSpaceDelayCalibrator


def _nearest_pointing_targets(
    pointing_rows: Sequence[Dict[str, Any]], unix_s: float
) -> Dict[int, Dict[str, Any]]:
    """Return {beam_id: target_dict} for the pointing row closest to unix_s."""
    times = np.asarray([float(r["unix"]) for r in pointing_rows], dtype=np.float64)
    idx = int(np.argmin(np.abs(times - unix_s)))
    return {int(t["beam_id"]): t for t in pointing_rows[idx].get("targets", [])}


def _clean_channel_mask(
    run_dir: Path,
    median_spec: Optional[np.ndarray] = None,
    fmin_mhz: float = 306.0,
    fmax_mhz: float = 496.0,
    max_flagged_frac: float = 0.25,
) -> np.ndarray:
    """Build conservative 672-channel boolean mask avoiding RFI notches and roll-off edges."""
    mask = (FREQS_MHZ >= fmin_mhz) & (FREQS_MHZ <= fmax_mhz)
    snaps = load_rfi_snapshots(run_dir)
    if snaps:
        pwr = np.full((len(snaps), N_CHAN_TOTAL), np.nan, dtype=np.float64)
        frac = np.ones((len(snaps), N_CHAN_TOTAL), dtype=np.float64)
        for s_idx, row in enumerate(snaps):
            for c in row.get("data", {}).get("channels", []):
                sb = int(c.get("subband", 0))
                lc = int(c.get("local_channel", -1))
                ch = sb * N_CHAN_SB + lc
                if 0 <= ch < N_CHAN_TOTAL:
                    pwr[s_idx, ch] = float(c.get("mean_power", 0.0))
                    frac[s_idx, ch] = float(c.get("flagged_chunk_fraction", 1.0))
        med_p = np.nanmedian(pwr, axis=0)
        med_f = np.nanmedian(frac, axis=0)
        mask &= np.isfinite(med_p) & (med_p > 0.05) & (med_f <= max_flagged_frac)

    if median_spec is not None:
        sp = np.asarray(median_spec, dtype=np.float64)
        valid_vals = np.isfinite(sp) & (sp > 0.0)
        mask &= valid_vals
        if np.any(mask):
            base = median_filter(np.where(valid_vals, sp, np.nanmedian(sp)), size=21, mode="nearest")
            ratio = sp / np.maximum(base, 1.0e-9)
            mask &= (ratio > 0.45) & (ratio < 2.2)

    return mask


def _dedisperse_chunk(
    wf_beams_freq: np.ndarray,
    dt_s: float,
    dm_pc_cm3: float,
    chan_mask: np.ndarray,
    beam_indices: Sequence[int] = (0, 2),
) -> np.ndarray:
    """Incoherently dedisperse selected beams using per-beam channel median normalization."""
    idx_f = np.flatnonzero(chan_mask)
    freqs = FREQS_MHZ[idx_f]
    f_ref = float(np.max(freqs))
    delays_s = K_DM_PC_CM3_MHZ2_S * dm_pc_cm3 * ((freqs ** -2) - (f_ref ** -2))
    shifts = np.rint(delays_s / dt_s).astype(np.int64)

    sub_raw = wf_beams_freq[:, list(beam_indices), :][:, :, idx_f].astype(np.float64)
    n_t, n_b, _ = sub_raw.shape
    ch_med = np.maximum(np.median(sub_raw, axis=0, keepdims=True), 1.0e-6)
    rel = (sub_raw / ch_med) - 1.0

    # Reject transient-burst channels within each chunk before inverse-variance weighting
    ch_mad = 1.4826 * np.median(np.abs(rel[:, 0, :] - np.median(rel[:, 0, :], axis=0)), axis=0)
    med_mad = max(float(np.median(ch_mad)), 1.0e-4)
    keep_ch = (ch_mad > 0.25 * med_mad) & (ch_mad < 1.8 * med_mad)
    inv_var = np.where(keep_ch, 1.0 / np.maximum(ch_mad * ch_mad, (0.5 * med_mad) ** 2), 0.0)
    inv_var /= max(float(np.sum(inv_var)), 1.0e-12)

    acc = np.zeros((n_t, n_b), dtype=np.float64)
    w_sum = np.zeros(n_t, dtype=np.float64)
    for k, sh in enumerate(shifts):
        s_k = int(sh)
        wk = float(inv_var[k])
        if s_k == 0:
            acc += rel[:, :, k] * wk
            w_sum += wk
        elif 0 < s_k < n_t:
            acc[: n_t - s_k] += rel[s_k:, :, k] * wk
            w_sum[: n_t - s_k] += wk
        elif -n_t < s_k < 0:
            acc[-s_k:] += rel[: n_t + s_k, :, k] * wk
            w_sum[-s_k:] += wk

    return acc / np.maximum(w_sum[:, None], 1.0e-12)


def benchmark_test08_sun_and_cal(
    data_dir: Path, positions_m: np.ndarray
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run (l, m) profiler, optimal OFF finder, and beam-space delay self-cal on test08."""
    run_dir = data_dir / "test08_solar_chromaticity_2d"
    wf, _, t0_us, _ = load_dual_subband_beam_power(run_dir, n_beams=10, max_frames=16000)
    pt_rows = load_pointing_log(run_dir)
    cal_beams = [0, 1, 2, 3, 4, 5, 6, 7, 9]

    clean_mask = _clean_channel_mask(run_dir, median_spec=np.median(wf[:, 9, :], axis=0))
    sel_idx = np.flatnonzero(clean_mask)
    step = max(1, sel_idx.size // 64)
    freq_mask_sub = np.zeros(N_CHAN_TOTAL, dtype=bool)
    freq_mask_sub[sel_idx[::step][:64]] = True

    n_snap = 5
    edges = np.linspace(0, wf.shape[0], n_snap + 1, dtype=int)
    p_snaps = np.zeros((n_snap, len(cal_beams), N_CHAN_TOTAL), dtype=np.float64)
    l0_mat = np.zeros((n_snap, len(cal_beams)), dtype=np.float64)
    m0_mat = np.zeros((n_snap, len(cal_beams)), dtype=np.float64)
    sun_l = np.zeros(n_snap, dtype=np.float64)
    sun_m = np.zeros(n_snap, dtype=np.float64)

    for s in range(n_snap):
        a, b = edges[s], edges[s + 1]
        p_snaps[s] = np.median(wf[a:b, cal_beams, :], axis=0)
        mid_unix = float(np.median(t0_us[a:b])) * 1.0e-6
        tgts = _nearest_pointing_targets(pt_rows, mid_unix)
        sun_l[s] = float(tgts[0]["l0"])
        sun_m[s] = float(tgts[0]["m0"])
        for j, bid in enumerate(cal_beams):
            l0_mat[s, j] = float(tgts[bid]["l0"])
            m0_mat[s, j] = float(tgts[bid]["m0"])

    profiler = MultiFrequencyLmProfiler(positions_m, freqs_mhz=FREQS_MHZ, grid_n=41)
    lm_res = profiler.reconstruct(
        l0_mat, m0_mat, p_snaps, freq_mask=freq_mask_sub, ridge_lambda=0.05, l1_lambda=0.25
    )

    mid_l_sun = float(np.median(sun_l))
    mid_m_sun = float(np.median(sun_m))
    peak0 = lm_res.peaks[0]
    sun_loc_err = float(np.hypot(peak0["l"] - mid_l_sun, peak0["m"] - mid_m_sun))

    # Optimal OFF beam vs deployed Deep_South_OFF (0.0, -0.7)
    clean_freqs = FREQS_MHZ[freq_mask_sub]
    opt_cands = find_optimal_off_beams(
        positions_m, clean_freqs, mid_l_sun, mid_m_sun, min_sep_deg=20.0, top_k=3
    )
    from . import array_factor_power

    dep_af = float(
        np.mean(array_factor_power(positions_m, clean_freqs, mid_l_sun, mid_m_sun, 0.0, -0.7))
    )
    dep_db = float(10.0 * np.log10(max(dep_af, 1.0e-12)))

    # Delay self-calibration using calibrated Sun fan [0, 2, 3, 4, 5, 6, 7] minus beam 9 (index 8)
    fan_rel = [0, 2, 3, 4, 5, 6, 7]
    cal = BeamSpaceDelayCalibrator(positions_m, freqs_mhz=FREQS_MHZ)
    cal_res = cal.fit(
        fan_power=p_snaps[:, fan_rel, :],
        off_power=p_snaps[:, 8, :],  # Deep_South_OFF is cal_beams[8]
        src_l=sun_l,
        src_m=sun_m,
        fan_l0=l0_mat[:, fan_rel],
        fan_m0=m0_mat[:, fan_rel],
        off_l0=l0_mat[:, 8],
        off_m0=m0_mat[:, 8],
        freq_mask=freq_mask_sub,
    )

    # Observed vs predicted fan ratio profiles across beams [0, 2, 3, 4, 5, 6, 7]
    obs_diff = p_snaps[:, fan_rel, :][:, :, freq_mask_sub] - p_snaps[:, 8, :][:, None, freq_mask_sub]
    obs_ratio = obs_diff / np.maximum(np.sum(np.abs(obs_diff), axis=1, keepdims=True), 1.0e-9)
    pred0 = cal._predict_fan_ratios(
        np.zeros(positions_m.shape[0] - 1),
        sun_l,
        sun_m,
        l0_mat[:, fan_rel],
        m0_mat[:, fan_rel],
        l0_mat[:, 8],
        m0_mat[:, 8],
        clean_freqs,
    )
    pred1 = cal._predict_fan_ratios(
        cal_res.delays_ns[1:],
        sun_l,
        sun_m,
        l0_mat[:, fan_rel],
        m0_mat[:, fan_rel],
        l0_mat[:, 8],
        m0_mat[:, 8],
        clean_freqs,
    )

    summary = {
        "lm_peak": peak0,
        "sun_true_mid_lm": [mid_l_sun, mid_m_sun],
        "sun_localization_err_lm": sun_loc_err,
        "lm_fit_r2": lm_res.fit_r2,
        "deployed_off_leakage_db": dep_db,
        "optimal_off_best": {
            "l0": opt_cands[0].l0,
            "m0": opt_cands[0].m0,
            "leakage_db": opt_cands[0].band_power_ratio_db,
            "sep_deg": opt_cands[0].target_sep_deg,
        },
        "delay_cal_ns": [round(float(x), 4) for x in cal_res.delays_ns],
        "delay_dl_tilt": cal_res.dl_tilt,
        "delay_dm_tilt": cal_res.dm_tilt,
        "delay_closure_ns": cal_res.closure_delay_ns,
        "delay_before_rmse": cal_res.before_rmse,
        "delay_after_rmse": cal_res.after_rmse,
    }
    artifacts = {
        "lm_res": lm_res,
        "sun_mid": (mid_l_sun, mid_m_sun),
        "opt_off": (opt_cands[0].l0, opt_cands[0].m0),
        "cal_res": cal_res,
        "fan_labels": ["ON", "E+2", "E+4", "E+7", "W-4", "N+4", "S-7"],
        "obs_beam_ratio": np.mean(obs_ratio, axis=(0, 2)),
        "pred0_beam_ratio": np.mean(pred0, axis=(0, 2)),
        "pred1_beam_ratio": np.mean(pred1, axis=(0, 2)),
    }
    return summary, artifacts


def benchmark_test12b_horizon(
    data_dir: Path, positions_m: np.ndarray
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Profile urban/airport horizon RFI brightness in (l, m) from test12b."""
    run_dir = data_dir / "test12b_airport_santiago_horizon"
    wf, _, t0_us, _ = load_dual_subband_beam_power(run_dir, n_beams=10, max_frames=3500)
    pt_rows = load_pointing_log(run_dir)
    cal_beams = [0, 1, 2, 3, 4, 5, 6, 7, 9]

    mid_unix = float(np.median(t0_us)) * 1.0e-6
    tgts = _nearest_pointing_targets(pt_rows, mid_unix)
    l0_vec = np.asarray([float(tgts[b]["l0"]) for b in cal_beams], dtype=np.float64)
    m0_vec = np.asarray([float(tgts[b]["m0"]) for b in cal_beams], dtype=np.float64)
    p_med = np.median(wf[:, cal_beams, :], axis=0)

    # Include broad spectrum across 315..495 MHz avoiding extreme passband edges/zeros
    valid_f = (FREQS_MHZ >= 315.0) & (FREQS_MHZ <= 495.0) & np.all(p_med > 0.05, axis=0)
    idx_f = np.flatnonzero(valid_f)
    step = max(1, idx_f.size // 96)
    f_mask = np.zeros(N_CHAN_TOTAL, dtype=bool)
    f_mask[idx_f[::step][:96]] = True

    profiler = MultiFrequencyLmProfiler(positions_m, freqs_mhz=FREQS_MHZ, grid_n=41)
    lm_res = profiler.reconstruct(
        l0_vec, m0_vec, p_med, freq_mask=f_mask, ridge_lambda=0.06, l1_lambda=0.22
    )
    return {
        "fit_r2": lm_res.fit_r2,
        "top_peaks": lm_res.peaks[:3],
    }, {
        "lm_res": lm_res,
        "targets": {b: tgts[b] for b in cal_beams},
    }


def benchmark_sidelobe_cancel(
    data_dir: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluate Huber-IRLS adaptive sidelobe cancellation on test09a and test07."""
    canceller = AdaptiveSidelobeCanceller(ridge=1.0e-3, irls_iters=4, huber_k=2.5)
    results: Dict[str, Any] = {}
    plot_data: Dict[str, Any] = {}

    for run_name, n_b, on_b, aux_bs in [
        ("test09a_galactic_center_plane", 10, 0, [8, 9]),
        ("test07_cena_dual_source", 6, 0, [4, 5]),
    ]:
        run_dir = data_dir / run_name
        wf, _, t0_us, _ = load_dual_subband_beam_power(run_dir, n_beams=n_b, max_frames=12000)
        f_mask = _clean_channel_mask(run_dir, median_spec=np.median(wf[:, on_b, :], axis=0))
        res = canceller.cancel(wf, on_beam_idx=on_b, aux_beam_indices=aux_bs, freq_mask=f_mask)

        idx_valid = np.flatnonzero(f_mask)
        t_rel_s = (t0_us.astype(np.float64) - float(t0_us[0])) * 1.0e-6
        before_bb = np.median(wf[:, on_b, idx_valid], axis=1)
        after_bb = np.median(res.cleaned_on[:, idx_valid], axis=1)
        results[run_name] = {
            "corr_before_sun_mon": res.corr_before,
            "corr_after_sun_mon": res.corr_after,
            "rms_before": res.rms_before,
            "rms_after": res.rms_after,
            "suppression_db": res.suppression_db,
        }
        if run_name == "test07_cena_dual_source":
            plot_data = {
                "run_name": run_name,
                "t_s": t_rel_s,
                "before_bb": before_bb,
                "after_bb": after_bb,
                "sun_bb": np.median(wf[:, aux_bs[0], idx_valid], axis=1),
                "summary": results[run_name],
            }
    return results, plot_data


def benchmark_ml_rfi_clustering(data_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Cluster RFI telemetry regimes across 5 commissioning runs without supervised labels."""
    run_names = [
        "test01_muos_rfi_425_475",
        "test08_solar_chromaticity_2d",
        "test09a_galactic_center_plane",
        "test10_production_stress_10beams",
        "test12b_airport_santiago_horizon",
    ]
    all_snaps: List[Dict[str, Any]] = []
    snap_to_run: List[str] = []
    for name in run_names:
        rows = load_rfi_snapshots(data_dir / name)[::2]
        for r in rows:
            all_snaps.append(r)
            snap_to_run.append(name.split("_")[0])

    feats, meta = extract_rfi_snapshot_features(all_snaps)
    clf = UnsupervisedRfiClassifier(n_clusters=5, max_iter=50, seed=2026)
    res = clf.fit_predict(feats, meta=meta)

    snap_arr = np.asarray(snap_to_run)
    sample_runs = snap_arr[res.meta["snapshot_idx"]]
    run_breakdown: Dict[str, Dict[str, float]] = {}
    for rn in sorted(set(snap_to_run)):
        m_r = sample_runs == rn
        run_breakdown[rn] = {}
        for s in res.cluster_summaries:
            cid = s["cluster_id"]
            nm = s["name"]
            run_breakdown[rn][nm] = float(np.mean(res.labels[m_r] == cid))

    summary = {
        "total_channel_samples": int(feats.shape[0]),
        "clusters": res.cluster_summaries,
        "run_breakdown": run_breakdown,
    }
    return summary, {
        "res": res,
        "sample_runs": sample_runs,
        "run_labels": [r.split("_")[0] for r in run_names],
        "run_breakdown": run_breakdown,
    }


def benchmark_comb_and_pulsars(data_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluate OFF-referenced comb cancellation on test05 and Vela folding on test04."""
    canceller = CoherentCombCanceller(line_snr_thresh=18.0, min_coherence=0.30)

    # 1) Test 05 PSR B0950: measure broadband PSD suppression of 9.7656 Hz & 50.1 Hz combs
    r5 = data_dir / "test05_psr_b0950_transit"
    d0_5, seq0_5, _ = load_beam_power_subband(r5, 0, n_beams=3, max_frames=40000)
    d1_5, seq1_5, _ = load_beam_power_subband(r5, 1, n_beams=3, max_frames=40000)
    common5, i0_5, i1_5 = np.intersect1d(seq0_5, seq1_5, return_indices=True)
    wf5 = np.concatenate([d0_5[i0_5], d1_5[i1_5]], axis=2)
    dt5_s = float(np.median(np.diff(common5))) * (10.0 / 3.0 * 1.0e-6)
    fmask5 = _clean_channel_mask(r5, median_spec=np.median(wf5[:, 0, :], axis=0))
    idx5 = np.flatnonzero(fmask5)
    on5 = np.median(wf5[:, 0, idx5], axis=1)
    off5 = np.median(wf5[:, 2, idx5], axis=1)
    comb5 = canceller.cancel(on5, off5, dt_s=dt5_s)

    on5_d = comb5.cleaned_on - median_filter(comb5.cleaned_on, size=585, mode="nearest")
    raw5_d = on5 - median_filter(on5, size=585, mode="nearest")
    f_psd, p_before = welch(raw5_d, fs=1.0 / dt5_s, nperseg=min(8192, raw5_d.size // 4))
    _, p_after = welch(on5_d, fs=1.0 / dt5_s, nperseg=min(8192, on5_d.size // 4))

    # 2) Test 04 Vela Pulsar: dedisperse full run with low-variance clean channels
    r4 = data_dir / "test04_vela_pulsar_transit"
    # Filter transient/bursting channels using full-run strided probe across all files
    probe_parts = []
    for st in range(0, 360000, 90000):
        try:
            pw_s, _, _, _ = load_dual_subband_beam_power(r4, n_beams=3, max_frames=3000, start_frame=st)
            probe_parts.append(pw_s[:, 0, :])
        except ValueError:
            break
    b0_probe = np.concatenate(probe_parts, axis=0).astype(np.float64)
    p_med = np.median(b0_probe, axis=0)
    p_mad = np.median(np.abs(b0_probe - p_med[None, :]), axis=0)
    rel_var = p_mad / np.maximum(p_med, 1.0e-6)
    med_rv = float(np.median(rel_var[(FREQS_MHZ > 306.0) & (FREQS_MHZ < 495.0)]))
    fmask4 = (
        (FREQS_MHZ > 306.0)
        & (FREQS_MHZ < 495.0)
        & (p_med > 1.0)
        & (rel_var > 0.25 * med_rv)
        & (rel_var < 1.8 * med_rv)
    )

    on4_parts: List[np.ndarray] = []
    off4_parts: List[np.ndarray] = []
    dt4_s = 0.00256
    chunk_size = 90000
    for start in range(0, 360000, chunk_size):
        try:
            wf4, _, _, info4 = load_dual_subband_beam_power(
                r4, n_beams=3, max_frames=chunk_size, start_frame=start
            )
        except ValueError:
            break
        if info4["dt_s"] > 0.0:
            dt4_s = float(info4["dt_s"])
        dd = _dedisperse_chunk(wf4, dt4_s, dm_pc_cm3=67.771, chan_mask=fmask4, beam_indices=(0, 2))
        on4_parts.append(dd[:, 0])
        off4_parts.append(dd[:, 1])

    on4 = np.concatenate(on4_parts)
    off4 = np.concatenate(off4_parts)
    comb4 = canceller.cancel(on4, off4, dt_s=dt4_s)

    p_vela_obs = 0.089428056
    on4_det = on4 - median_filter(on4, size=79, mode="nearest")
    c4_det = comb4.cleaned_on - median_filter(comb4.cleaned_on, size=79, mode="nearest")
    prof_b, snr_b = fold_profile(on4_det, dt4_s, p_vela_obs, n_bins=32)
    prof_a, snr_a = fold_profile(c4_det, dt4_s, p_vela_obs, n_bins=32)

    # 3) Run MatchedFilterTransientScorer on a 25,000-frame slice of test04 (32 rebinned channels)
    wf4_s, _, _, _ = load_dual_subband_beam_power(r4, n_beams=3, max_frames=24000, start_frame=120000)
    idx4 = np.flatnonzero(fmask4)
    n_reb = 32
    bins = np.array_split(idx4, n_reb)
    reb_freqs = np.asarray([float(np.mean(FREQS_MHZ[b])) for b in bins], dtype=np.float64)
    on_reb = np.column_stack([np.mean(wf4_s[:, 0, b], axis=1) for b in bins])
    off_reb = np.column_stack([np.mean(wf4_s[:, 2, b], axis=1) for b in bins])
    scorer = MatchedFilterTransientScorer(
        freqs_mhz=reb_freqs, dm_trials=[0.0, 35.0, 67.8, 120.0], widths_samples=[1, 2, 4, 8], snr_threshold=6.0
    )
    cand4 = scorer.score(on_reb, off_reb, dt_s=dt4_s)

    summary = {
        "test05_comb_suppression_db": comb5.comb_suppression_db,
        "test05_top_comb_freqs_hz": [round(f, 3) for f in comb5.comb_freqs_hz[:10]],
        "test04_frames": int(on4.size),
        "test04_comb_suppression_db": comb4.comb_suppression_db,
        "test04_vela_snr_before": snr_b,
        "test04_vela_snr_after": snr_a,
        "test04_peak_bin_before": int(np.argmax(prof_b)),
        "test04_peak_bin_after": int(np.argmax(prof_a)),
        "test04_transient_check": {
            "dm": cand4.dm_pc_cm3 if cand4 else None,
            "snr_on": cand4.snr_on if cand4 else None,
            "rfi_vetoed": cand4.rfi_vetoed if cand4 else None,
        },
    }
    plot_data = {
        "f_psd": f_psd,
        "p_before": p_before,
        "p_after": p_after,
        "comb5": comb5,
        "prof_b": prof_b,
        "prof_a": prof_a,
        "snr_b": snr_b,
        "snr_a": snr_a,
        "comb4_supp_db": comb4.comb_suppression_db,
    }
    return summary, plot_data


def _plot_fig14(
    out_path: Path,
    t08_art: Dict[str, Any],
    t08_sum: Dict[str, Any],
    t12b_art: Dict[str, Any],
    t09_art: Dict[str, Any],
) -> None:
    """Render 14_exp_lm_rfi_and_spatial_cal.png cleanly without label overlap."""
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 11.5), dpi=160)
    fig.subplots_adjust(left=0.06, right=0.96, top=0.90, bottom=0.07, wspace=0.25, hspace=0.30)

    # Panel (a): test08 Sun (l, m) reconstruction
    ax = axes[0, 0]
    lm_res = t08_art["lm_res"]
    im = ax.imshow(
        lm_res.nnls_map,
        origin="lower",
        extent=[-1.0, 1.0, -1.0, 1.0],
        cmap="inferno",
        aspect="equal",
    )
    theta = np.linspace(0.0, 2.0 * np.pi, 256)
    for deg, ls in [(30, ":"), (60, "--"), (90, "-")]:
        r = np.sin(np.radians(deg))
        ax.plot(r * np.cos(theta), r * np.sin(theta), color="white", alpha=0.35, lw=0.9, ls=ls)
    sl, sm = t08_art["sun_mid"]
    ol, om = t08_art["opt_off"]
    pk = t08_sum["lm_peak"]
    ax.scatter([sl], [sm], marker="*", s=220, color="cyan", edgecolor="black", lw=0.7, label="True Sun (l,m)")
    ax.scatter([pk["l"]], [pk["m"]], marker="+", s=140, color="lime", lw=2.0, label="MFS NNLS Peak")
    ax.scatter([0.0], [-0.7], marker="x", s=100, color="orange", lw=2.0, label=f"Deployed OFF ({t08_sum['deployed_off_leakage_db']:.1f} dB)")
    ax.scatter([ol], [om], marker="D", s=85, color="deepskyblue", edgecolor="white", lw=0.8, label=f"Optimal OFF ({t08_sum['optimal_off_best']['leakage_db']:.1f} dB)")
    ax.set_title(f"(a) test08 Solar MFS Deconvolution (err={t08_sum['sun_localization_err_lm']:.3f}, R²={lm_res.fit_r2:.3f})")
    ax.set_xlabel("Direction cosine l (East)")
    ax.set_ylabel("Direction cosine m (North)")
    ax.legend(loc="upper left", fontsize=8.0, framealpha=0.85)
    fig.colorbar(im, ax=ax, fraction=0.044, pad=0.02, label="Relative excess")

    # Panel (b): test12b urban/airport horizon map
    ax = axes[0, 1]
    lm12 = t12b_art["lm_res"]
    im2 = ax.imshow(
        lm12.nnls_map,
        origin="lower",
        extent=[-1.05, 1.05, -1.05, 1.05],
        cmap="magma",
        aspect="equal",
    )
    ax.plot(np.cos(theta), np.sin(theta), color="white", alpha=0.45, lw=1.0)
    highlight_beams = {
        0: ("SCL Airport", (-0.42, 0.14)),
        2: ("Santiago Centro", (-0.58, -0.18)),
        5: ("Quiet W", (0.08, 0.06)),
        6: ("Quiet S", (0.08, 0.08)),
        9: ("Zenith", (0.07, 0.07)),
    }
    tgts = t12b_art["targets"]
    for bid, (lbl, (dx, dy)) in highlight_beams.items():
        tl, tm = float(tgts[bid]["l0"]), float(tgts[bid]["m0"])
        ax.scatter([tl], [tm], s=60, color="cyan", edgecolor="black", lw=0.6, zorder=4)
        ax.annotate(
            lbl,
            xy=(tl, tm),
            xytext=(tl + dx, tm + dy),
            color="white",
            fontsize=8.0,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.18", fc="black", alpha=0.55, lw=0),
            arrowprops=dict(arrowstyle="->", color="cyan", lw=0.8),
        )
    ax.set_title(f"(b) test12b Horizon RFI (l,m) Profile (R²={lm12.fit_r2:.3f})")
    ax.set_xlabel("Direction cosine l (East)")
    ax.set_ylabel("Direction cosine m (North)")
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-1.08, 1.08)
    fig.colorbar(im2, ax=ax, fraction=0.044, pad=0.02, label="Relative excess")

    # Panel (c): Beam-Space Delay Self-Calibration
    ax = axes[1, 0]
    x_pos = np.arange(len(t08_art["fan_labels"]))
    w = 0.26
    ax.bar(x_pos - w, t08_art["obs_beam_ratio"], width=w, label="Observed (Sun - OFF)", color="#1f77b4")
    ax.bar(x_pos, t08_art["pred0_beam_ratio"], width=w, label=f"Nominal tau=0 (RMSE={t08_sum['delay_before_rmse']:.4f})", color="#ff7f0e", alpha=0.85)
    ax.bar(x_pos + w, t08_art["pred1_beam_ratio"], width=w, label=f"Calibrated tau (RMSE={t08_sum['delay_after_rmse']:.4f})", color="#2ca02c")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(t08_art["fan_labels"], fontsize=9)
    ax.set_ylabel("Normalized differential fan fraction")
    ax.set_title("(c) test08 Beam-Space Delay Self-Calibration")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8.0)
    ax.set_ylim(-0.08, 0.39)
    tau_txt = "\n".join([f"Ant {i}: {d:+.3f} ns" for i, d in enumerate(t08_sum["delay_cal_ns"])])
    tau_txt += f"\nClosure: {t08_sum['delay_closure_ns']:.3f} ns"
    ax.text(
        0.24,
        0.96,
        tau_txt,
        transform=ax.transAxes,
        va="top",
        fontsize=8.3,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.92),
    )

    # Panel (d): Adaptive Sidelobe Canceller on test07 CenA + Sun_Monitor
    ax = axes[1, 1]
    t_s = t09_art["t_s"]
    bef = t09_art["before_bb"]
    aft = t09_art["after_bb"]
    s09 = t09_art["summary"]
    stride = max(1, t_s.size // 1200)
    ax.plot(t_s[::stride], bef[::stride], lw=1.0, alpha=0.8, color="#d62728", label=f"Before (|r_Sun|={abs(s09['corr_before_sun_mon']):.3f})")
    ax.plot(t_s[::stride], aft[::stride], lw=1.1, color="#1f77b4", label=f"After Huber-IRLS (|r_Sun|={abs(s09['corr_after_sun_mon']):.3f})")
    ax.set_xlabel("Elapsed time [s]")
    ax.set_ylabel("Clean-band median power [arb]")
    ax.set_title(f"(d) test07 CenA Sidelobe Cancel ({s09['suppression_db']:+.2f} dB RMS)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8.5)

    fig.suptitle(
        "CHARTS Experimental Prototypes I: (l,m) Sky/Horizon RFI Profiling & Beam-Space Calibration",
        fontsize=13.5,
        fontweight="bold",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_fig15(
    out_path: Path,
    ml_art: Dict[str, Any],
    comb_art: Dict[str, Any],
    comb_sum: Dict[str, Any],
) -> None:
    """Render 15_exp_ml_rfi_and_comb_cancel.png cleanly without overlapping text."""
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 11.5), dpi=160)
    fig.subplots_adjust(left=0.06, right=0.96, top=0.90, bottom=0.07, wspace=0.24, hspace=0.30)

    # Panel (a): PCA 2D scatter colored by unsupervised cluster
    ax = axes[0, 0]
    res = ml_art["res"]
    palette = ["#2ca02c", "#d62728", "#9467bd", "#ff7f0e", "#1f77b4"]
    rng = np.random.default_rng(7)
    for cid, name in res.cluster_names.items():
        mask = np.flatnonzero(res.labels == cid)
        if mask.size == 0:
            continue
        take = mask if mask.size <= 1400 else rng.choice(mask, size=1400, replace=False)
        ax.scatter(
            res.pca_2d[take, 0],
            res.pca_2d[take, 1],
            s=7,
            alpha=0.55,
            color=palette[cid % len(palette)],
            label=f"{name} ({100.0 * mask.size / res.labels.size:.1f}%)",
        )
    ax.set_xlabel("Telemetry PCA Component 1")
    ax.set_ylabel("Telemetry PCA Component 2")
    ax.set_title("(a) Unsupervised k-means++ Telemetry Clusters (5 Runs)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8.0, markerscale=2.0)

    # Panel (b): Stacked bar of regime composition per commissioning test
    ax = axes[0, 1]
    runs = sorted(ml_art["run_breakdown"].keys())
    c_items = sorted(res.cluster_names.items())
    left = np.zeros(len(runs), dtype=np.float64)
    y_pos = np.arange(len(runs))
    for cid, cname in c_items:
        vals = np.asarray([100.0 * ml_art["run_breakdown"][r].get(cname, 0.0) for r in runs])
        ax.barh(y_pos, vals, left=left, height=0.58, color=palette[cid % len(palette)], label=cname)
        left += vals
    ax.set_yticks(y_pos)
    ax.set_yticklabels(runs, fontsize=9.5)
    ax.set_xlim(0.0, 100.0)
    ax.set_xlabel("Channel-snapshot fraction [%]")
    ax.set_title("(b) Telemetry Cluster Composition by Test Run")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower right", fontsize=7.8, framealpha=0.9)

    # Panel (c): test05 Welch PSD BEFORE vs AFTER CoherentCombCanceller
    ax = axes[1, 0]
    f_psd = comb_art["f_psd"]
    sel = f_psd >= 1.5
    ax.semilogy(f_psd[sel], comb_art["p_before"][sel], lw=1.0, alpha=0.85, color="#d62728", label="Beam 0 Before")
    ax.semilogy(f_psd[sel], comb_art["p_after"][sel], lw=1.0, alpha=0.9, color="#1f77b4", label="Beam 0 After OFF Comb Cancel")
    ax.axvline(9.7656, color="#444444", ls=":", lw=1.0)
    ax.axvline(50.10, color="#444444", ls="--", lw=1.0)
    ax.text(10.8, np.max(comb_art["p_before"][sel]) * 0.45, "9.77 Hz comb", fontsize=8.5, color="#222222")
    ax.text(51.2, np.max(comb_art["p_before"][sel]) * 0.25, "~50.1 Hz mains", fontsize=8.5, color="#222222")
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD [arb^2 / Hz]")
    ax.set_title(f"(c) test05 Comb Cancellation ({comb_sum['test05_comb_suppression_db']:.1f} dB line reduction)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper right", fontsize=8.5)

    # Panel (d): test04 Vela 89.43 ms folded profile BEFORE vs AFTER CombCancel
    ax = axes[1, 1]
    bins = np.arange(comb_art["prof_b"].size)
    pb_norm = (comb_art["prof_b"] - np.median(comb_art["prof_b"])) / max(
        1.4826 * float(np.median(np.abs(comb_art["prof_b"] - np.median(comb_art["prof_b"])))), 1.0e-12
    )
    pa_norm = (comb_art["prof_a"] - np.median(comb_art["prof_a"])) / max(
        1.4826 * float(np.median(np.abs(comb_art["prof_a"] - np.median(comb_art["prof_a"])))), 1.0e-12
    )
    ax.step(bins, pb_norm, where="mid", lw=1.4, color="#ff7f0e", label=f"Before (Vela SNR={comb_art['snr_b']:.2f} sigma)")
    ax.step(bins, pa_norm, where="mid", lw=1.6, color="#1f77b4", label=f"After CombCancel (SNR={comb_art['snr_a']:.2f} sigma)")
    ax.set_xlabel("Pulse phase bin (32 bins across P = 89.428 ms)")
    ax.set_ylabel("Normalized SNR above off-pulse median")
    ax.set_title(f"(d) test04 Vela Pulse Preservation ({comb_art['comb4_supp_db']:.1f} dB common comb)")
    ax.set_ylim(-2.2, 7.6)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8.5)

    fig.suptitle(
        "CHARTS Experimental Prototypes II: Unsupervised RFI ML Clustering & Comb Cancellation",
        fontsize=13.5,
        fontweight="bold",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> int:
    """Run commissioning v2 experimental benchmarks and write JSON + summary plots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("/data/commissioning_20261119_v2"))
    parser.add_argument(
        "--config-yaml",
        type=Path,
        default=Path(
            "/home/fpalma/kotekan/charts/commissioning_20261119_v2/configs_20261121_full/09_test08_solar_chromaticity_2d.yaml"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/data/commissioning_20261119_v2/plots"),
    )
    args = parser.parse_args()

    fallback_cfg = Path(__file__).resolve().parents[1] / "config" / "64antennas_direct_tracker_10beams.yaml"
    cfg_path = args.config_yaml if args.config_yaml.exists() else fallback_cfg
    positions_m, active_ids, _ = load_antenna_positions_from_yaml(cfg_path)

    t08_sum, t08_art = benchmark_test08_sun_and_cal(args.data_dir, positions_m)
    t12b_sum, t12b_art = benchmark_test12b_horizon(args.data_dir, positions_m)
    sl_sum, t09_art = benchmark_sidelobe_cancel(args.data_dir)
    ml_sum, ml_art = benchmark_ml_rfi_clustering(args.data_dir)
    comb_sum, comb_art = benchmark_comb_and_pulsars(args.data_dir)

    fig14_path = args.out_dir / "14_exp_lm_rfi_and_spatial_cal.png"
    fig15_path = args.out_dir / "15_exp_ml_rfi_and_comb_cancel.png"
    _plot_fig14(fig14_path, t08_art, t08_sum, t12b_art, t09_art)
    _plot_fig15(fig15_path, ml_art, comb_art, comb_sum)

    report = {
        "active_antennas": active_ids,
        "test08_sun_and_cal": t08_sum,
        "test12b_horizon_lm": t12b_sum,
        "sidelobe_cancellation": sl_sum,
        "unsupervised_rfi_clustering": ml_sum,
        "comb_and_pulsars": comb_sum,
        "plots": [str(fig14_path), str(fig15_path)],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "experimental_benchmarks_summary.json"
    summary_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
