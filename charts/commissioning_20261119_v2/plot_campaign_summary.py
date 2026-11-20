#!/usr/bin/env python3
"""Generate 5 campaign summary figures across corr_1119 calibration and
commissioning v2 runs (test01-test04) plus pipeline telemetry improvements."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from beam_io import (
    FREQS_MHZ,
    atomic_savefig,
    clean_channel_mask,
    load_dual_subband_beam_power,
    load_jsonl,
)

V2_ROOT = Path("/data/commissioning_20261119_v2")
CAL_1119_DIR = Path("/data/calibration/corr_1119")
CAL_1118_DIR = Path("/data/calibration/corr_1118")
ANTS = [0, 4, 32, 36]
ANT_NAMES = ["Ant 0 (RFSoC 0, SE)", "Ant 4 (RFSoC 0, SW)", "Ant 32 (RFSoC 1, NE)", "Ant 36 (RFSoC 1, NW)"]
ANT_COLS = ["#0f172a", "#2563eb", "#16a34a", "#dc2626"]


def plot_01_calibration(out_dir: Path) -> Path:
    g19 = np.load(CAL_1119_DIR / "gains_corr_1119_0_4_32_36_fringehp_672x64.npy")[:, ANTS]
    g18 = np.load(CAL_1118_DIR / "gains_corr_1118_0_4_32_36_fringehp_672x64.npy")[:, ANTS]
    rep19 = json.loads((CAL_1119_DIR / "calibration_report.json").read_text(encoding="utf-8"))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    ax00, ax01, ax10, ax11 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    for k in range(4):
        ax00.plot(FREQS_MHZ, np.abs(g19[:, k]), color=ANT_COLS[k], lw=1.3, label=ANT_NAMES[k])
    ax00.axvspan(356.0, 384.0, color="#fef08a", alpha=0.35, label="MUOS 356-384 MHz")
    ax00.set_title("corr_1119 Gain Amplitudes |g(nu)| (300-501.3 MHz)")
    ax00.set_xlabel("Frequency (MHz)")
    ax00.set_ylabel("|g|")
    ax00.grid(True, alpha=0.25)
    ax00.legend(fontsize=8.0, loc="lower right")

    for k in range(1, 4):
        uph = np.rad2deg(np.unwrap(np.angle(g19[:, k])))
        ax01.plot(FREQS_MHZ, uph, color=ANT_COLS[k], lw=1.4, label=ANT_NAMES[k])
    ax01.axvspan(356.0, 384.0, color="#fef08a", alpha=0.35)
    ax01.set_title("Unwrapped Gain Phase vs Ant 0 (0 Phase Jumps >1.5 rad)")
    ax01.set_xlabel("Frequency (MHz)")
    ax01.set_ylabel("Unwrapped Phase (deg)")
    ax01.grid(True, alpha=0.25)
    ax01.legend(fontsize=8.5)

    for k in range(1, 4):
        diff = g19[:, k] * np.conj(g18[:, k])
        ph_deg = np.rad2deg(np.mod(np.angle(diff), 2.0 * np.pi))
        uph_rad = np.unwrap(np.angle(diff))
        slope = float(np.polyfit(FREQS_MHZ, uph_rad, 1)[0])
        dtau_ns = slope / (-2.0 * np.pi) * 1e3
        med_deg = float(np.rad2deg(np.angle(np.mean(diff / np.maximum(np.abs(diff), 1e-12)))))
        ax10.plot(
            FREQS_MHZ,
            ph_deg,
            color=ANT_COLS[k],
            lw=1.1,
            label=f"Ant {ANTS[k]} (mean={med_deg:+.1f} deg, d_tau={dtau_ns:+.2f} ns)",
        )
    ax10.set_title("Post-Reboot Phase Step & Inter-Board Delay Shift (corr_1119 - corr_1118)")
    ax10.set_xlabel("Frequency (MHz)")
    ax10.set_ylabel("Phase Diff Mod 360 (deg)")
    ax10.set_ylim(-10, 370)
    ax10.grid(True, alpha=0.25)
    ax10.legend(fontsize=8.0, loc="lower left")

    prov = rep19.get("channel_provenance", {})
    labels = list(prov.keys())
    counts = [prov[k] for k in labels]
    ax11.barh(labels, counts, color=["#0284c7", "#2563eb", "#16a34a", "#f59e0b", "#64748b"][: len(labels)])
    for idx, c in enumerate(counts):
        ax11.text(c + 4, idx, f"{c} ch", va="center", fontsize=9.0)
    ax11.set_xlim(0, max(counts) * 1.18 if counts else 250)
    ax11.set_title(f"corr_1119 Calibration Channel Provenance ({sum(counts)} Total)")
    ax11.set_xlabel("Number of Channels")
    ax11.grid(True, axis="x", alpha=0.25)

    p_out = out_dir / "01_calibration_epoch_comparison.png"
    atomic_savefig(fig, p_out, dpi=140)
    plt.close(fig)
    return p_out


def plot_02_test01_rfi(out_dir: Path) -> Path:
    rdir = V2_ROOT / "test01_muos_rfi_425_475"
    snaps = load_jsonl(rdir / "rfi_channel_snapshots.jsonl")
    wf, _, _, _ = load_dual_subband_beam_power(rdir, n_beams=8)

    flag_sum = np.zeros(672, dtype=np.float64)
    sk_max = np.zeros(672, dtype=np.float64)
    n_s = 0
    for snap in snaps:
        chs = snap.get("data", {}).get("channels", [])
        if len(chs) >= 672:
            n_s += 1
            for e in chs:
                gi = int(e.get("global_channel", int(e.get("subband", 0)) * 336 + int(e.get("local_channel", -1))))
                if 0 <= gi < 672:
                    flag_sum[gi] += float(e.get("flagged_chunk_fraction", e.get("flag_fraction", 0.0)))
                    sk_max[gi] += float(e.get("max_sk", e.get("mean_sk", 0.0)))
    flag_pct = 100.0 * flag_sum / max(n_s, 1)
    sk_mean_max = sk_max / max(n_s, 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    ax00, ax01, ax10, ax11 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    ax00.plot(FREQS_MHZ, flag_pct, color="#0f172a", lw=0.9)
    for lo, hi, col, lbl in [
        (356.0, 385.0, "#f59e0b", f"MUOS 356-385 ({np.mean(flag_pct[(FREQS_MHZ>=356)&(FREQS_MHZ<=385)]):.1f}%)"),
        (415.0, 438.0, "#dc2626", f"425 Band 415-438 ({np.mean(flag_pct[(FREQS_MHZ>=415)&(FREQS_MHZ<=438)]):.1f}%)"),
        (464.0, 486.0, "#2563eb", f"475 Band 464-486 ({np.mean(flag_pct[(FREQS_MHZ>=464)&(FREQS_MHZ<=486)]):.1f}%)"),
    ]:
        ax00.axvspan(lo, hi, color=col, alpha=0.22, label=lbl)
    ax00.set_title(f"Test 01 RFI Flagged Fraction Across 300-501.3 MHz ({n_s} Snapshots)")
    ax00.set_xlabel("Frequency (MHz)")
    ax00.set_ylabel("Mean Flagged Chunks (%)")
    ax00.set_ylim(-2, 105)
    ax00.grid(True, alpha=0.25)
    ax00.legend(fontsize=8.0, loc="upper left")

    ax01.plot(FREQS_MHZ, np.maximum(sk_mean_max, 0.5), color="#7c3aed", lw=0.9)
    ax01.set_yscale("log")
    for lo, hi, col in [(356.0, 385.0, "#f59e0b"), (415.0, 438.0, "#dc2626"), (464.0, 486.0, "#2563eb")]:
        ax01.axvspan(lo, hi, color=col, alpha=0.18)
    ax01.set_title("Test 01 Mean Per-Channel Max Spectral Kurtosis (SK)")
    ax01.set_xlabel("Frequency (MHz)")
    ax01.set_ylabel("Mean max_sk (log scale)")
    ax01.grid(True, which="both", alpha=0.25)

    med_spec = np.median(wf, axis=0)
    m_muos = (FREQS_MHZ >= 355.0) & (FREQS_MHZ <= 386.0)
    ref_m = np.maximum(med_spec[3, m_muos], 1e-12)
    for b, lbl, col in [(1, "Beam 1: MUOS-5", "#dc2626"), (0, "Beam 0: MUOS-1", "#f59e0b"), (2, "Beam 2: MUOS-3", "#2563eb"), (6, "Beam 6: Loc 472.5", "#16a34a")]:
        ax10.plot(FREQS_MHZ[m_muos], med_spec[b, m_muos] / ref_m, color=col, lw=1.2, label=lbl)
    ax10.axhline(1.0, color="#0f172a", ls=":", lw=1.0)
    ax10.set_title("MUOS Band Spatial Localization Contrast (vs Beam 3 OFF-MUOS)")
    ax10.set_xlabel("Frequency (MHz)")
    ax10.set_ylabel("Beam / OFF-MUOS Power Ratio")
    ax10.grid(True, alpha=0.25)
    ax10.legend(fontsize=8.0)

    m_425 = (FREQS_MHZ >= 414.0) & (FREQS_MHZ <= 440.0)
    ref_425 = np.maximum(med_spec[7, m_425], 1e-12)
    for b, lbl, col in [(0, "Beam 0", "#dc2626"), (5, "Beam 5: Loc 425.4", "#2563eb"), (4, "Beam 4: LEO/GOES", "#16a34a"), (3, "Beam 3: OFF-MUOS", "#f59e0b")]:
        ax11.plot(FREQS_MHZ[m_425], med_spec[b, m_425] / ref_425, color=col, lw=1.2, label=lbl)
    ax11.axhline(1.0, color="#0f172a", ls=":", lw=1.0)
    ax11.set_title("425 MHz Band Spatial Localization Contrast (vs Beam 7 Zenith)")
    ax11.set_xlabel("Frequency (MHz)")
    ax11.set_ylabel("Beam / Zenith Power Ratio")
    ax11.grid(True, alpha=0.25)
    ax11.legend(fontsize=8.0)

    p_out = out_dir / "02_test01_rfi_and_muos.png"
    atomic_savefig(fig, p_out, dpi=140)
    plt.close(fig)
    return p_out


def plot_03_test02_test03(out_dir: Path) -> Path:
    r2 = V2_ROOT / "test02_fornax_a_night_chromaticity"
    r3 = V2_ROOT / "test03_pictor_orion_lmc_crab"
    wf2, seq2, _, _ = load_dual_subband_beam_power(r2, n_beams=10)
    wf3, _, _, _ = load_dual_subband_beam_power(r3, n_beams=10)
    ms3 = json.loads((r3 / "multi_source_verification.json").read_text(encoding="utf-8"))

    cm2 = clean_channel_mask(wf2[:, 0, :], max_zero_frac=0.08, mad_sigma=5.0) & clean_channel_mask(wf2[:, 9, :], max_zero_frac=0.08, mad_sigma=5.0)
    cm3 = clean_channel_mask(wf3[:, 0, :], max_zero_frac=0.08, mad_sigma=5.0) & clean_channel_mask(wf3[:, 9, :], max_zero_frac=0.08, mad_sigma=5.0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    ax00, ax01, ax10, ax11 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    t2_min = (seq2 - seq2[0]).astype(np.float64) * (10.0 / 3.0) * 1e-6 / 60.0
    n_bins_t = 25
    edges_i = np.linspace(0, len(t2_min), n_bins_t + 1, dtype=int)
    t_mid_m = np.array([np.mean(t2_min[edges_i[i] : edges_i[i + 1]]) for i in range(n_bins_t)])
    med2 = np.median(wf2[:, :, cm2], axis=2)  # (T, 10)
    for b, lbl, col in [
        (0, "B0: Fornax A Tracked", "#dc2626"),
        (2, "B2: E +3 deg Offset", "#f59e0b"),
        (4, "B4: E +6 deg Offset", "#7c3aed"),
        (1, "B1: Parked Meridian Drift", "#2563eb"),
        (8, "B8: Zenith", "#0f172a"),
    ]:
        rat = np.array([np.median(med2[edges_i[i] : edges_i[i + 1], b] / np.maximum(med2[edges_i[i] : edges_i[i + 1], 9], 1e-12)) for i in range(n_bins_t)])
        ax00.plot(t_mid_m, rat, "o-", ms=3.0, lw=1.3, color=col, label=lbl)
    ax00.axhline(1.0, color="#64748b", ls=":", lw=1.0)
    ax00.set_title("Test 02 Clean-Band Power Relative to Deep OFF (El 45 deg)")
    ax00.set_xlabel("Elapsed Time (min)")
    ax00.set_ylabel("Beam / Deep OFF Ratio")
    ax00.set_ylim(0.985, 1.016)
    ax00.grid(True, alpha=0.25)
    ax00.legend(fontsize=7.5, loc="lower center", ncol=2)

    r_0_9 = np.median(wf2[:, 0, :], axis=0) / np.maximum(np.median(wf2[:, 9, :], axis=0), 1e-9)
    r_8_9 = np.median(wf2[:, 8, :], axis=0) / np.maximum(np.median(wf2[:, 9, :], axis=0), 1e-9)
    bins_f = np.arange(300, 501, 25)
    fc, v09, v89 = [], [], []
    for f0 in bins_f:
        m = cm2 & (FREQS_MHZ >= f0) & (FREQS_MHZ < f0 + 25)
        if np.any(m):
            fc.append(f0 + 12.5)
            v09.append(float(np.median(r_0_9[m])))
            v89.append(float(np.median(r_8_9[m])))
    ax01.plot(fc, v09, "o-", lw=1.6, color="#dc2626", label="Fornax A (B0) / Deep OFF (B9)")
    ax01.plot(fc, v89, "s--", lw=1.6, color="#0f172a", label="Zenith (B8) / Deep OFF (B9)")
    ax01.axhline(1.0, color="#64748b", ls=":", lw=1.0)
    ax01.set_title("Test 02 Spectral Ratio Across 25 MHz Clean Bins")
    ax01.set_xlabel("Frequency (MHz)")
    ax01.set_ylabel("Median Power Ratio")
    ax01.grid(True, alpha=0.25)
    ax01.legend(fontsize=8.5)

    cal_sources = [s for s in ms3.get("sources", []) if "Uncal" not in s.get("name", "")]
    names3 = [s["name"] for s in cal_sources]
    excs3 = [100.0 * float(s["mean_excess_frac"]) for s in cal_sources]
    snrs3 = [float(s["snr_mean_over_se"]) for s in cal_sources]
    bar_cols = ["#16a34a" if v >= 0 else "#dc2626" for v in excs3]
    x3 = np.arange(len(names3))
    ax10.bar(x3, excs3, color=bar_cols)
    ax10.axhline(0.0, color="#0f172a", lw=1.0)
    for i, (ev, sv) in enumerate(zip(excs3, snrs3)):
        va = "bottom" if ev >= 0 else "top"
        dy = 0.02 if ev >= 0 else -0.02
        ax10.text(i, ev + dy, f"{sv:+.1f} sig", ha="center", va=va, fontsize=8.5, fontweight="bold")
    ax10.set_xticks(x3)
    ax10.set_xticklabels([n.replace(" vs ", "\nvs ").replace("_Cal", "") for n in names3], fontsize=8.0)
    ax10.set_ylim(min(excs3) - 0.14, max(excs3) + 0.14)
    ax10.set_title("Test 03 Multi-Source Calibrated ON-OFF Excess (%)")
    ax10.set_ylabel("Fractional Excess (%)")
    ax10.grid(True, axis="y", alpha=0.25)

    cal_pic = next((s for s in ms3.get("sources", []) if s.get("name") == "PictorA_Cal vs OFF_Pictor"), None)
    uncal_pic = next((s for s in ms3.get("sources", []) if s.get("name") == "PictorA_Uncal vs OFF_Pictor"), None)
    cal_ratio = (
        float(cal_pic["mean_on_power"]) / max(float(uncal_pic["mean_on_power"]), 1e-12)
        if cal_pic and uncal_pic
        else float(np.median(wf3[:, 0, cm3]) / max(np.median(wf3[:, 8, cm3]), 1e-12))
    )

    spec3 = np.median(wf3, axis=0)
    ax11.plot(FREQS_MHZ[cm3], spec3[1, cm3], color="#16a34a", lw=1.1, label="Beam 1: Orion A Cal")
    ax11.plot(FREQS_MHZ[cm3], spec3[0, cm3], color="#2563eb", lw=1.0, label="Beam 0: Pictor A Cal")
    ax11.plot(FREQS_MHZ[cm3], spec3[8, cm3], color="#f59e0b", lw=1.1, label=f"Beam 8: Pictor A Uncal ({cal_ratio:.2f}x below Cal)")
    ax11.set_title("Test 03 Clean-Channel Median Spectra (Calibrated vs Uncalibrated)")
    ax11.set_xlabel("Frequency (MHz)")
    ax11.set_ylabel("Median Beam Power")
    ax11.grid(True, alpha=0.25)
    ax11.legend(fontsize=8.0)

    p_out = out_dir / "03_test02_test03_sky_and_gain.png"
    atomic_savefig(fig, p_out, dpi=140)
    plt.close(fig)
    return p_out


def plot_04_test04_vela(out_dir: Path) -> Path:
    from charts_constants import K_DM

    r4 = V2_ROOT / "test04_vela_pulsar_transit"
    pv = json.loads((r4 / "pulsar_verification.json").read_text(encoding="utf-8"))
    wf, seq, _, _ = load_dual_subband_beam_power(r4, n_beams=3)
    t_rel_s = (seq - seq[0]).astype(np.float64) * (10.0 / 3.0) * 1e-6
    cadence_s = float(pv["timing_gpsdo"].get("cadence_s") or 0.00256)
    best_p = float(pv["best_period_s"])
    dm_cat = float(pv["catalog_dm_pc_cm3"])

    m0 = clean_channel_mask(wf[:, 0, :], max_zero_frac=0.10, mad_sigma=5.0)
    m2 = clean_channel_mask(wf[:, 2, :], max_zero_frac=0.10, mad_sigma=5.0)
    ch_idx = np.where(m0 & m2)[0]
    f_ch = FREQS_MHZ[ch_idx]
    f_ref = float(np.max(FREQS_MHZ))
    k_dm_val = float(K_DM.value)
    shifts = np.round(k_dm_val * dm_cat * (f_ch ** -2 - f_ref ** -2) / cadence_s).astype(np.int64)

    def _fold(beam_i: int, ch_submask=None):
        sel_idx = ch_idx if ch_submask is None else ch_idx[ch_submask]
        sel_sh = shifts if ch_submask is None else shifts[ch_submask]
        v = wf[:, beam_i, sel_idx].astype(np.float32)
        pos = np.isfinite(v) & (v > 0.0)
        cnt = np.maximum(pos.sum(axis=0), 1)
        mu = np.where(pos, v, 0.0).sum(axis=0) / cnt
        sd = np.maximum(np.sqrt(np.where(pos, (v - mu[None, :]) ** 2, 0.0).sum(axis=0) / cnt), 1e-6)
        nb = np.where(pos, np.clip((v - mu[None, :]) / sd[None, :], -8.0, 8.0), 0.0)
        ts = np.zeros(wf.shape[0], dtype=np.float64)
        for k, sh in enumerate(sel_sh):
            if sh == 0:
                ts += nb[:, k]
            elif 0 < sh < wf.shape[0]:
                ts[:-sh] += nb[sh:, k]
            elif -wf.shape[0] < sh < 0:
                ts[-sh:] += nb[:sh, k]
        ts /= np.sqrt(float(len(sel_idx)))
        bins = np.clip((np.mod(t_rel_s / best_p, 1.0) * 32).astype(np.int64), 0, 31)
        prof = np.bincount(bins, weights=ts, minlength=32) / np.maximum(np.bincount(bins, minlength=32), 1.0)
        off = np.sort(prof)[:22]
        snr = float((np.max(prof) - np.median(prof)) / max(np.std(off), 1e-12))
        return prof, snr

    pr0, sn0 = _fold(0)
    pr1, sn1 = _fold(1)
    pr2, sn2 = _fold(2)
    pr_lo, sn_lo = _fold(0, f_ch < 400.65)
    pr_hi, sn_hi = _fold(0, f_ch >= 400.65)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    ax00, ax01, ax10, ax11 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    ph = np.linspace(0, 1, 32, endpoint=False)

    ax00.plot(ph, pr0, lw=2.0, color="#2563eb", label=f"Beam 0: Vela Cal ON (SNR={sn0:.2f})")
    ax00.plot(ph, pr1, lw=1.4, color="#f59e0b", label=f"Beam 1: Vela Uncal (SNR={sn1:.2f})")
    ax00.plot(ph, pr2, lw=1.3, color="#64748b", ls="--", label=f"Beam 2: OFF Vela (SNR={sn2:.2f})")
    ax00.set_title(f"Test 04 Vela Folded Profile (P={best_p*1e3:.4f} ms, DM={dm_cat})")
    ax00.set_xlabel("Pulse Phase")
    ax00.set_ylabel("Normalized Amplitude")
    ax00.grid(True, alpha=0.25)
    ax00.legend(fontsize=8.5)

    dm_map = {float(k): float(v) for k, v in pv.get("dm_curve_snr", {}).items()}
    dm_xs = sorted(dm_map.keys())
    dm_ys = [dm_map[x] for x in dm_xs]
    ax01.plot(dm_xs, dm_ys, "o-", color="#0f172a", ms=4, lw=1.5)
    ax01.axvline(dm_cat, color="#dc2626", ls="--", lw=1.2, label=f"Catalog DM={dm_cat}")
    ax01.set_title(f"Dispersion Measure Trial Curve at P={best_p*1e3:.4f} ms")
    ax01.set_xlabel("DM (pc cm^-3)")
    ax01.set_ylabel("Folded SNR")
    ax01.grid(True, alpha=0.25)
    ax01.legend(fontsize=8.5)

    ax10.plot(ph, pr_lo, lw=1.8, color="#dc2626", label=f"LOW 300-400 MHz (SNR={sn_lo:.2f}, bin {int(np.argmax(pr_lo))})")
    ax10.plot(ph, pr_hi, lw=1.8, color="#16a34a", label=f"HIGH 400-501 MHz (SNR={sn_hi:.2f}, bin {int(np.argmax(pr_hi))})")
    ax10.set_title("Subband Phase Alignment & GPSDO 0 us Sync Check")
    ax10.set_xlabel("Pulse Phase")
    ax10.set_ylabel("Normalized Amplitude")
    ax10.grid(True, alpha=0.25)
    ax10.legend(fontsize=8.5)

    chk = pv.get("channel_order_check_at_cat_dm", {})
    bar_lbls = ["Normal\n(300->501)", "Reversed\n(501->300)", "Subbands\nSwapped", "DM=0\n(Unsmeared)"]
    bar_vals = [
        chk.get("normal_300_to_501_snr", 0.0),
        chk.get("reversed_501_to_300_snr", 0.0),
        chk.get("subband_swapped_snr", 0.0),
        chk.get("dm0_unsmeared_snr", 0.0),
    ]
    ax11.bar(bar_lbls, bar_vals, color=["#16a34a", "#dc2626", "#f59e0b", "#64748b"])
    for idx, v in enumerate(bar_vals):
        ax11.text(idx, v + 0.18, f"{v:.2f}", ha="center", fontsize=9.0, fontweight="bold")
    ax11.set_ylim(0, max(bar_vals) * 1.2 + 0.5)
    ax11.set_title(
        f"Channel Order Check ({pv['timing_gpsdo']['n_frames_aligned']} frames, "
        f"gaps={pv['timing_gpsdo']['seq_gaps_sb0']}/{pv['timing_gpsdo']['seq_gaps_sb1']}, "
        f"dt={pv['timing_gpsdo']['max_abs_subband_dt_us']} us)"
    )
    ax11.set_ylabel("Folded SNR")
    ax11.grid(True, axis="y", alpha=0.25)

    p_out = out_dir / "04_test04_vela_pulsar.png"
    atomic_savefig(fig, p_out, dpi=140)
    plt.close(fig)
    return p_out


def plot_05_telemetry(out_dir: Path) -> Path:
    runs = [
        ("test01_muos_rfi_425_475", "Test 01\n(8b, 51.2ms)"),
        ("test02_fornax_a_night_chromaticity", "Test 02\n(10b, 51.2ms)"),
        ("test03_pictor_orion_lmc_crab", "Test 03\n(10b, 51.2ms)"),
        ("test04_vela_pulsar_transit", "Test 04\n(3b, 2.56ms)"),
    ]
    cmd_cats = [
        ("copy_in:input: voltage_0", "CopyIn Voltage", "#64748b"),
        ("kernel:cudaRfiMitigationCommand", "RFI Mitigation", "#f59e0b"),
        ("kernel:cudaDirectBeamTrackerCommand", "Beamformer", "#2563eb"),
        ("copy_out:sync_output", "CopyOut Sync (formed_beams)", "#dc2626"),
        ("copy_out:output: formed_beams_0", "CopyOut formed_beams", "#991b1b"),
        ("copy_out:output: beam_power_0", "CopyOut beam_power", "#16a34a"),
    ]

    summary: Dict[str, Dict[str, Any]] = {}
    for rid, _ in runs:
        rows = load_jsonl(V2_ROOT / rid / "telemetry_log.jsonl")
        cmds_vals: Dict[str, List[float]] = {k: [] for k, _, _ in cmd_cats}
        tot_vals, temps, pwrs, buf0_occ = [], [], [], []
        for r in rows:
            gp = r.get("gpu_profile", {}).get("gpu_0", {})
            t_ms = float(gp.get("total_ms", 0.0) or 0.0)
            if t_ms > 0.0:
                tot_vals.append(t_ms)
                cdict = gp.get("commands_ms", {})
                for k, _, _ in cmd_cats:
                    if k in cdict:
                        cmds_vals[k].append(float(cdict[k]))
                smi = r.get("nvidia_smi", {})
                if "temp_c" in smi:
                    temps.append(float(smi["temp_c"]))
                if "power_w" in smi:
                    pwrs.append(float(smi["power_w"]))
            bufs = r.get("buffers", {})
            if isinstance(bufs, dict) and "host_beam_power_buffer_0" in bufs:
                buf0_occ.append(float(bufs["host_beam_power_buffer_0"].get("occupancy_pct", 0.0)))
        log_mb = (V2_ROOT / rid / "kotekan.log").stat().st_size / (1024 ** 2)
        summary[rid] = {
            "cmd_mean": {k: (float(np.mean(v)) if v else 0.0) for k, v in cmds_vals.items()},
            "tot": tot_vals,
            "temps": temps,
            "pwrs": pwrs,
            "max_buf0_occ": max(buf0_occ) if buf0_occ else 0.0,
            "log_mb": log_mb,
        }

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    ax00, ax01, ax10, ax11 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    x = np.arange(len(runs))
    bottom = np.zeros(len(runs))
    for k, lbl, col in cmd_cats:
        vals = np.array([summary[rid]["cmd_mean"][k] for rid, _ in runs])
        ax00.bar(x, vals, bottom=bottom, label=lbl, color=col)
        bottom += vals
    ax00.set_xticks(x)
    ax00.set_xticklabels([s for _, s in runs], fontsize=8.5)
    ax00.set_title("GPU0 Mean Per-Command Execution Breakdown (formed_beams D2H Overhead)")
    ax00.set_ylabel("Mean Duration (ms)")
    ax00.grid(True, axis="y", alpha=0.25)
    ax00.legend(fontsize=7.5, loc="upper right")

    meds = [float(np.median(summary[rid]["tot"])) for rid, _ in runs]
    maxs = [float(np.max(summary[rid]["tot"])) for rid, _ in runs]
    w = 0.35
    ax01.bar(x - w / 2, meds, w, color="#2563eb", label="Median gpu_0 total_ms")
    ax01.bar(x + w / 2, maxs, w, color="#f59e0b", label="Max gpu_0 total_ms")
    ax01.axhline(51.2, color="#dc2626", ls="--", lw=1.2, label="51.2 ms Budget (Test 01-03)")
    ax01.axhline(2.56, color="#16a34a", ls=":", lw=1.4, label="2.56 ms Budget (Test 04)")
    for i, mx in enumerate(maxs):
        ax01.text(i + w / 2, mx + 1.0, f"{mx:.1f}", ha="center", fontsize=8.0)
    ax01.set_xticks(x)
    ax01.set_xticklabels([s for _, s in runs], fontsize=8.5)
    ax01.set_ylim(0, max(maxs) * 1.28)
    ax01.set_title("GPU0 Total Latency vs Frame Period Budget (Test 01-04)")
    ax01.set_ylabel("Time (ms)")
    ax01.grid(True, axis="y", alpha=0.25)
    ax01.legend(fontsize=8.0, loc="upper right", ncol=2)

    t_maxs = [max(summary[rid]["temps"]) for rid, _ in runs]
    p_meds = [float(np.median(summary[rid]["pwrs"])) for rid, _ in runs]
    b1 = ax10.bar(x - w / 2, t_maxs, w, color="#dc2626", label="Max GPU Temp (C)")
    ax10_t = ax10.twinx()
    b2 = ax10_t.bar(x + w / 2, p_meds, w, color="#7c3aed", alpha=0.8, label="Median GPU Power (W)")
    ax10.set_xticks(x)
    ax10.set_xticklabels([s for _, s in runs], fontsize=8.5)
    ax10.set_ylim(0, 78)
    ax10_t.set_ylim(0, 260)
    ax10.set_title("RTX 5090 Thermal & Power Stability Across Campaign")
    ax10.set_ylabel("Max Temp (C)")
    ax10_t.set_ylabel("Median Power (W)")
    ax10.grid(True, axis="y", alpha=0.25)
    h1, l1 = ax10.get_legend_handles_labels()
    h2, l2 = ax10_t.get_legend_handles_labels()
    ax10.legend(h1 + h2, l1 + l2, fontsize=8.0, loc="upper left", ncol=2)

    log_mbs = [summary[rid]["log_mb"] for rid, _ in runs]
    ax11.bar(x, log_mbs, color=["#64748b", "#64748b", "#64748b", "#dc2626"])
    for i, mb in enumerate(log_mbs):
        ax11.text(i, mb + 2.0, f"{mb:.2f} MB", ha="center", fontsize=8.5, fontweight="bold")
    ax11.set_xticks(x)
    ax11.set_xticklabels([s for _, s in runs], fontsize=8.5)
    ax11.set_ylim(0, max(log_mbs) * 1.18)
    ax11.set_title("kotekan.log Size (Test 04 Per-Frame Info Spam Fixed in rawFileWrite.cpp)")
    ax11.set_ylabel("Log File Size (MB)")
    ax11.grid(True, axis="y", alpha=0.25)

    p_out = out_dir / "05_pipeline_telemetry_and_improvements.png"
    atomic_savefig(fig, p_out, dpi=140)
    plt.close(fig)
    return p_out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default="/data/commissioning_20261119_v2/plots")
    p.add_argument("--copy-to", default=None, help="Optional secondary directory to copy generated PNGs into")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(out_dir, 0o777)

    generated = [
        plot_01_calibration(out_dir),
        plot_02_test01_rfi(out_dir),
        plot_03_test02_test03(out_dir),
        plot_04_test04_vela(out_dir),
        plot_05_telemetry(out_dir),
    ]
    if args.copy_to:
        dst = Path(args.copy_to)
        dst.mkdir(parents=True, exist_ok=True)
        for p_img in generated:
            shutil.copy2(p_img, dst / p_img.name)

    print("Generated plots:")
    for p_img in generated:
        print(f"  {p_img}")


if __name__ == "__main__":
    main()
