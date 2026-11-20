#!/usr/bin/env python3
"""Post-test verification for Test 2 (Nighttime Fornax A) & Test 8 (2D Solar
Chromaticity): drift-scan FWHM(nu) power-law beta, 2D spatial-fan chromaticity,
frequency-dependent beam squint, offset spectral-index tilt d_alpha(theta), and
Cal vs Uncal broadband spectra."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

from beam_io import (
    FREQS_MHZ,
    atomic_savefig,
    atomic_write_text,
    clean_channel_mask,
    load_dual_subband_beam_power,
    load_jsonl,
)


def measure_fwhm_and_peak(t_s: np.ndarray, y: np.ndarray, smooth_sigma: float = 4.0):
    """Return (peak_t_s, fwhm_s, snr, y_smooth) on a baseline-subtracted drift curve."""
    if len(y) < 12:
        return None, None, 0.0, y
    y_sm = gaussian_filter1d(y, sigma=max(0.8, min(smooth_sigma, len(y) / 6.0)), mode="nearest")
    edge = max(2, min(len(y_sm) // 8, 40))
    edges = np.concatenate([y_sm[:edge], y_sm[-edge:]])
    base = float(np.median(edges))
    rms = float(np.std(edges)) if np.std(edges) > 0 else float(np.std(y_sm))
    y_sub = y_sm - base
    pk_i = int(np.argmax(y_sub))
    pk_val = float(y_sub[pk_i])
    snr = pk_val / max(rms, 1e-12)
    if pk_val <= 0 or snr < 3.0:
        return float(t_s[pk_i]), None, float(snr), y_sub

    half = 0.5 * pk_val
    t_left, t_right = None, None
    for i in range(pk_i, 0, -1):
        if y_sub[i - 1] <= half <= y_sub[i]:
            frac = (half - y_sub[i - 1]) / max(y_sub[i] - y_sub[i - 1], 1e-12)
            t_left = float(t_s[i - 1] + frac * (t_s[i] - t_s[i - 1]))
            break
    for i in range(pk_i, len(y_sub) - 1):
        if y_sub[i] >= half >= y_sub[i + 1]:
            frac = (y_sub[i] - half) / max(y_sub[i] - y_sub[i + 1], 1e-12)
            t_right = float(t_s[i] + frac * (t_s[i + 1] - t_s[i]))
            break
    fwhm_s = (t_right - t_left) if (t_left is not None and t_right is not None and t_right > t_left) else None

    # Weighted parabolic centroid near top 80% for sub-sample peak time
    hi_mask = y_sub >= 0.8 * pk_val
    if np.sum(hi_mask) >= 3:
        w = np.maximum(y_sub[hi_mask], 0.0)
        t_peak = float(np.sum(t_s[hi_mask] * w) / max(np.sum(w), 1e-12))
    else:
        t_peak = float(t_s[pk_i])
    return t_peak, fwhm_s, float(snr), y_sub


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--mode", default="auto", choices=["auto", "solar", "fornax"])
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    mode = args.mode
    if mode == "auto":
        mode = "fornax" if "fornax" in run_dir.name.lower() else "solar"

    wf, seq, t0_us, timing = load_dual_subband_beam_power(run_dir, n_beams=10)
    pointing = load_jsonl(run_dir / "beam_pointing_log.jsonl")
    t_rel_s = (seq - seq[0]).astype(np.float64) * (10.0 / 3.0) * 1e-6
    dur_s = float(t_rel_s[-1] - t_rel_s[0]) if len(t_rel_s) > 1 else 0.0
    is_smoke = dur_s < 120.0

    if mode == "solar":
        dec_deg = -22.023
        drift_beam, uncal_beam, off_beam = 1, 8, 9
        fan_specs = [
            (0, "ON (0 deg)", 0.0),
            (2, "E +2 deg", 2.0),
            (3, "E +4 deg", 4.0),
            (4, "E +7 deg", 7.0),
            (5, "W -4 deg", -4.0),
            (6, "N +4 deg", 4.0),
            (7, "S -7 deg", -7.0),
        ]
    else:
        dec_deg = -37.2083
        drift_beam, uncal_beam, off_beam = 1, 6, 9
        fan_specs = [
            (0, "ON (0 deg)", 0.0),
            (2, "E +3 deg", 3.0),
            (3, "W -3 deg", -3.0),
            (4, "E +6 deg", 6.0),
            (5, "W -6 deg", -6.0),
        ]

    sky_rate_deg_per_s = (15.0 / 3600.0) * float(np.cos(np.deg2rad(dec_deg)))

    clean = clean_channel_mask(wf[:, 0, :], max_zero_frac=0.08, mad_sigma=5.0) & clean_channel_mask(
        wf[:, off_beam, :], max_zero_frac=0.08, mad_sigma=5.0
    )
    if int(np.sum(clean)) < 30:
        clean = np.mean(np.isfinite(wf[:, 0, :]) & (wf[:, 0, :] > 0.0), axis=0) > 0.85

    # Group clean channels into ~18 ~10-MHz spectral bins across 300-501.3 MHz
    edges = np.linspace(300.0, 501.5, 19)
    subbands: List[Dict[str, Any]] = []
    sb_freqs, sb_fwhm_deg, sb_tpeak_s, sb_squint_deg = [], [], [], []
    sb_cal_excess, sb_uncal_excess = [], []

    for bi in range(len(edges) - 1):
        m_b = clean & (FREQS_MHZ >= edges[bi]) & (FREQS_MHZ < edges[bi + 1])
        if int(np.sum(m_b)) < 3:
            continue
        f_mid = float(np.mean(FREQS_MHZ[m_b]))
        drift_ts = np.mean(wf[:, drift_beam, m_b] - wf[:, off_beam, m_b], axis=1)
        t_pk, fwhm_s, snr, _ = measure_fwhm_and_peak(t_rel_s, drift_ts)
        fwhm_deg = (fwhm_s * sky_rate_deg_per_s) if (fwhm_s is not None and not is_smoke) else None

        p_on = float(np.median(np.mean(wf[:, 0, m_b] - wf[:, off_beam, m_b], axis=1)))
        p_un = float(np.median(np.mean(wf[:, uncal_beam, m_b] - wf[:, off_beam, m_b], axis=1)))
        p_off = float(np.median(np.mean(wf[:, off_beam, m_b], axis=1)))

        sb_rec: Dict[str, Any] = {
            "freq_mhz": round(f_mid, 2),
            "n_clean_ch": int(np.sum(m_b)),
            "drift_snr": round(snr, 2),
            "drift_peak_s": round(t_pk, 2) if t_pk is not None else None,
            "fwhm_deg": round(fwhm_deg, 3) if fwhm_deg is not None else None,
            "cal_excess_frac": round(p_on / max(p_off, 1e-12), 4),
            "uncal_excess_frac": round(p_un / max(p_off, 1e-12), 4),
        }
        subbands.append(sb_rec)
        sb_freqs.append(f_mid)
        sb_cal_excess.append(p_on)
        sb_uncal_excess.append(p_un)
        # A ~4.4 m 4-antenna array has mainlobe width >> 1 deg across 300-501.3 MHz;
        # narrower peaks are noise spikes rather than physical sky drift transits.
        if fwhm_deg is not None and fwhm_deg >= 1.0 and snr >= 5.0:
            sb_fwhm_deg.append((f_mid, fwhm_deg))
        if t_pk is not None and fwhm_deg is not None and fwhm_deg >= 1.0 and snr >= 5.0 and not is_smoke:
            sb_tpeak_s.append((f_mid, t_pk))

    # 1. Fit power-law FWHM(nu) = A * (nu/400)^(-beta)
    beta_fit, a_400_fit = None, None
    if len(sb_fwhm_deg) >= 4:
        fs = np.array([x[0] for x in sb_fwhm_deg])
        ws = np.array([x[1] for x in sb_fwhm_deg])
        slope, intercept = np.polyfit(np.log(fs / 400.0), np.log(ws), 1)
        beta_fit = round(float(-slope), 4)
        a_400_fit = round(float(np.exp(intercept)), 3)

    # 2. Squint vs frequency relative to median transit time
    squint_slope_deg_per_100mhz = None
    if len(sb_tpeak_s) >= 4:
        t_med = float(np.median([x[1] for x in sb_tpeak_s]))
        for rec in subbands:
            if rec["drift_peak_s"] is not None:
                sq = (rec["drift_peak_s"] - t_med) * sky_rate_deg_per_s
                rec["squint_deg"] = round(sq, 4)
        fs_sq = np.array([x[0] for x in sb_tpeak_s])
        sq_vals = np.array([(x[1] - t_med) * sky_rate_deg_per_s for x in sb_tpeak_s])
        squint_slope_deg_per_100mhz = round(float(np.polyfit(fs_sq / 100.0, sq_vals, 1)[0]), 4)

    # 3. Spatial-fan beam profile B(offset, nu) & offset spectral-index tilt delta_alpha
    fan_results: Dict[str, Any] = {}
    sb_freqs_arr = np.asarray(sb_freqs, dtype=np.float64)
    on_off_sb = np.asarray(sb_cal_excess, dtype=np.float64)
    for bid, lbl, off_deg in fan_specs:
        b_ratios = []
        raw_fracs = []
        for bi in range(len(edges) - 1):
            m_b = clean & (FREQS_MHZ >= edges[bi]) & (FREQS_MHZ < edges[bi + 1])
            if int(np.sum(m_b)) < 3:
                continue
            p_k = float(np.median(np.mean(wf[:, bid, m_b] - wf[:, off_beam, m_b], axis=1)))
            p_off = float(np.median(np.mean(wf[:, off_beam, m_b], axis=1)))
            raw_fracs.append(p_k / max(p_off, 1e-12))
            idx_sb = len(raw_fracs) - 1
            on_v = on_off_sb[idx_sb] if idx_sb < len(on_off_sb) else 0.0
            b_ratios.append(p_k / max(on_v, 1e-12) if on_v > 0 else np.nan)

        b_ratios_arr = np.asarray(b_ratios, dtype=np.float64)
        valid_r = np.isfinite(b_ratios_arr) & (b_ratios_arr > 0.02)
        d_alpha = None
        if bid != 0 and int(np.sum(valid_r)) >= 4 and len(sb_freqs_arr) == len(b_ratios_arr):
            d_alpha = round(float(np.polyfit(np.log(sb_freqs_arr[valid_r] / 400.0), np.log(b_ratios_arr[valid_r]), 1)[0]), 4)
        fan_results[f"beam_{bid}"] = {
            "label": lbl,
            "offset_deg": off_deg,
            "mean_excess_frac_vs_off": round(float(np.mean(raw_fracs)) if raw_fracs else 0.0, 4),
            "mean_rel_to_on": round(float(np.nanmean(b_ratios_arr)) if len(b_ratios_arr) else 0.0, 4),
            "delta_spectral_index_alpha": d_alpha,
            "per_subband_rel_to_on": [round(float(x), 4) if np.isfinite(x) else None for x in b_ratios_arr],
        }

    # Global Cal vs Uncal broadband comparison
    p_on_tot = float(np.mean(wf[:, 0, clean]))
    p_uncal_tot = float(np.mean(wf[:, uncal_beam, clean]))
    p_off_tot = float(np.mean(wf[:, off_beam, clean]))

    summary = {
        "run_dir": str(run_dir),
        "mode": mode,
        "is_smoke_test": bool(is_smoke),
        "duration_s": round(dur_s, 2),
        "timing": timing,
        "clean_channels": int(np.sum(clean)),
        "pointing_samples": len(pointing),
        "chromaticity_power_law_fwhm": {
            "beta_measured": beta_fit,
            "beta_expected_chromatic_array": 1.0,
            "fwhm_at_400mhz_deg": a_400_fit,
            "n_subbands_fitted": len(sb_fwhm_deg),
        },
        "beam_squint_deg_per_100mhz": squint_slope_deg_per_100mhz,
        "cal_vs_uncal": {
            "mean_power_cal_on": round(p_on_tot, 4),
            "mean_power_uncal_on": round(p_uncal_tot, 4),
            "mean_power_off": round(p_off_tot, 4),
            "cal_to_uncal_power_ratio": round(p_on_tot / max(p_uncal_tot, 1e-12), 4),
            "cal_excess_vs_off_frac": round((p_on_tot - p_off_tot) / max(p_off_tot, 1e-12), 5),
        },
        "spatial_fan": fan_results,
        "subbands": subbands,
    }
    atomic_write_text(run_dir / "chromaticity_verification.json", json.dumps(summary, indent=2))

    # 2x2 Diagnostic Figure
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5), layout="constrained")
    ax00, ax01 = axes[0, 0], axes[0, 1]
    ax10, ax11 = axes[1, 0], axes[1, 1]

    # Drift-scan curves at low/mid/high frequencies
    t_min = t_rel_s / 60.0
    for lbl, f_lo, f_hi, col in [("320-355 MHz", 320, 355, "#dc2626"), ("395-415 MHz", 395, 415, "#2563eb"), ("440-465 MHz", 440, 465, "#16a34a")]:
        m_b = clean & (FREQS_MHZ >= f_lo) & (FREQS_MHZ <= f_hi)
        if np.any(m_b):
            dc = np.mean(wf[:, drift_beam, m_b] - wf[:, off_beam, m_b], axis=1)
            ax00.plot(t_min, gaussian_filter1d(dc, sigma=max(1.0, min(4.0, len(dc) / 10.0))), color=col, lw=1.4, label=lbl)
    ax00.set_title(f"{mode.capitalize()} Parked Meridian Drift-Scan by Frequency (Beam {drift_beam} - OFF)")
    ax00.set_xlabel("Elapsed Time (min)")
    ax00.set_ylabel("Excess Power")
    ax00.grid(True, alpha=0.25)
    ax00.legend(fontsize=8.5)

    if len(sb_fwhm_deg) >= 2:
        fs = np.array([x[0] for x in sb_fwhm_deg])
        ws = np.array([x[1] for x in sb_fwhm_deg])
        ax01.plot(fs, ws, "o-", color="#0f172a", label="Measured FWHM(nu)")
        if beta_fit is not None and a_400_fit is not None:
            fg = np.linspace(300, 500, 100)
            ax01.plot(fg, a_400_fit * (fg / 400.0) ** (-beta_fit), "--", color="#2563eb", label=f"Fit beta={beta_fit:.2f} (A={a_400_fit:.2f} deg)")
            ax01.plot(fg, a_400_fit * (fg / 400.0) ** (-1.0), ":", color="#dc2626", label="Theory beta=1.0")
        ax01.legend(fontsize=8.5)
    else:
        msg = (
            "Smoke test (<120s):\nFull drift FWHM fit requires transit pass"
            if is_smoke
            else "No valid >=1 deg FWHM detections\n(compact source diluted vs sky/horizon gradient)"
        )
        ax01.text(0.5, 0.5, msg, ha="center", va="center", transform=ax01.transAxes)
    ax01.set_title("Beamwidth Chromatic Scaling FWHM(nu)")
    ax01.set_xlabel("Frequency (MHz)")
    ax01.set_ylabel("FWHM (deg)")
    ax01.grid(True, alpha=0.25)

    for bid, lbl, _ in fan_specs:
        if bid == 0:
            continue
        vals = fan_results[f"beam_{bid}"]["per_subband_rel_to_on"]
        if len(sb_freqs) == len(vals):
            ax10.plot(sb_freqs, vals, "o-", lw=1.2, ms=3.5, label=f"B{bid} {lbl} (da={fan_results[f'beam_{bid}']['delta_spectral_index_alpha']})")
    ax10.set_title("Spatial Offset Chromaticity: (Fan - OFF) / (ON - OFF) vs Frequency")
    ax10.set_xlabel("Frequency (MHz)")
    ax10.set_ylabel("Relative Response to ON")
    ax10.grid(True, alpha=0.25)
    ax10.legend(fontsize=7.5, ncol=2)

    if sb_freqs:
        ax11.plot(sb_freqs, [r["cal_excess_frac"] for r in subbands], "o-", color="#2563eb", label=f"Beam 0 Calibrated")
        ax11.plot(sb_freqs, [r["uncal_excess_frac"] for r in subbands], "s--", color="#f59e0b", label=f"Beam {uncal_beam} Uncalibrated")
    ax11.set_title("Calibrated vs Uncalibrated Excess Spectrum (Isolating Element vs Array)")
    ax11.set_xlabel("Frequency (MHz)")
    ax11.set_ylabel("(ON - OFF) / OFF")
    ax11.grid(True, alpha=0.25)
    ax11.legend(fontsize=8.5)

    atomic_savefig(fig, run_dir / "chromaticity_verification.png", dpi=140)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
