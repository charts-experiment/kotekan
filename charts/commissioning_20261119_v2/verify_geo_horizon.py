#!/usr/bin/env python3
"""Post-test verification for Test 12A (GEO Satellites Survey) and Test 12B
(SCL Airport & Santiago / Cerro San Cristobal Horizon Beams)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from beam_io import (
    FREQS_MHZ,
    atomic_savefig,
    atomic_write_text,
    clean_channel_mask,
    load_dual_subband_beam_power,
    load_jsonl,
)

GEO_BEAM_LABELS = [
    "B0: MUOS-1 (El42 W)",
    "B1: MUOS-3 (El22 E)",
    "B2: GOES-19 (El51 N)",
    "B3: GOES-18 (El11 W)",
    "B4: UFO-10 (El31 E)",
    "B5: SKYNET-5C (El25 E)",
    "B6: USA-115 (El59 NW)",
    "B7: INMARSAT-3F5 (El57 NE)",
    "B8: OFF-South (El35)",
    "B9: Zenith (El90)",
]

HORIZON_BEAM_LABELS = [
    "B0: SCL Airport (0.5°)",
    "B1: SCL Approach (10°)",
    "B2: Santiago Centro (0.5°)",
    "B3: San Cristobal (1.0°)",
    "B4: Santiago El10 (10°)",
    "B5: Quiet West (0.5°)",
    "B6: Quiet South (0.5°)",
    "B7: Quiet NW (0.5°)",
    "B8: San Cristobal Uncal",
    "B9: Zenith (90°)",
]


def safe_db(num: Any, den: Any) -> float:
    return round(float(10.0 * np.log10(np.maximum(num, 1e-12) / np.maximum(den, 1e-12))), 3)


def summarize_rfi_snapshots(rfi_snaps: List[Dict[str, Any]], bands: List[Tuple[str, float, float]]) -> Dict[str, Any]:
    if not rfi_snaps:
        return {}
    flag_accum = np.zeros(len(FREQS_MHZ), dtype=np.float64)
    sk_accum = np.zeros(len(FREQS_MHZ), dtype=np.float64)
    n_s = 0
    for snap in rfi_snaps:
        ch_list = snap.get("data", {}).get("channels", [])
        if len(ch_list) >= len(FREQS_MHZ):
            n_s += 1
            for entry in ch_list:
                if "global_channel" in entry:
                    idx = int(entry["global_channel"])
                elif "subband" in entry and "local_channel" in entry:
                    idx = int(entry["subband"]) * 336 + int(entry["local_channel"])
                else:
                    idx = -1
                if 0 <= idx < len(FREQS_MHZ):
                    flag_accum[idx] += float(
                        entry.get("flagged_chunk_fraction", entry.get("flag_fraction", entry.get("flagged_pct", 0.0)))
                    )
                    sk_accum[idx] += float(entry.get("max_sk", entry.get("mean_sk", entry.get("sk_median", 0.0))))
    if n_s == 0:
        return {}
    flag_accum /= n_s
    sk_accum /= n_s
    out = {}
    for bname, f0, f1 in bands:
        m = (FREQS_MHZ >= f0) & (FREQS_MHZ <= f1)
        if not np.any(m):
            continue
        out[bname] = {
            "mean_flag_metric": round(float(np.mean(flag_accum[m])), 4),
            "max_flag_metric": round(float(np.max(flag_accum[m])), 4),
            "peak_sk_freq_mhz": round(float(FREQS_MHZ[m][np.argmax(sk_accum[m])]), 2),
            "max_sk_metric": round(float(np.max(sk_accum[m])), 3),
        }
    return out


def analyze_band_vs_ref(
    wf: np.ndarray,
    mask: np.ndarray,
    ref_beam: int,
    zen_beam: int,
    labels: List[str],
    f0: float,
    f1: float,
) -> Dict[str, Any]:
    sub_f = FREQS_MHZ[mask]
    sub_wf = wf[:, :, mask]
    med_spec = np.median(sub_wf, axis=0)
    p95_spec = np.percentile(sub_wf, 95, axis=0)
    std_spec = np.std(sub_wf, axis=0)

    ref_med = np.maximum(med_spec[ref_beam], 1e-12)
    beam_stats = {}
    for b in range(wf.shape[1]):
        ratio = med_spec[b] / ref_med
        peak_i = int(np.argmax(med_spec[b]))
        c_i = int(np.argmax(ratio))
        beam_stats[f"beam_{b}"] = {
            "label": labels[b] if b < len(labels) else f"Beam {b}",
            "band_median_power": round(float(np.median(med_spec[b])), 3),
            "band_p95_power": round(float(np.percentile(p95_spec[b], 95)), 3),
            "peak_freq_mhz": round(float(sub_f[peak_i]), 2),
            "peak_power": round(float(med_spec[b, peak_i]), 3),
            "max_contrast_vs_ref": round(float(ratio[c_i]), 3),
            "max_contrast_db": safe_db(ratio[c_i], 1.0),
            "max_contrast_freq_mhz": round(float(sub_f[c_i]), 2),
            "median_contrast_vs_ref": round(float(np.median(ratio)), 3),
            "median_contrast_db": safe_db(np.median(ratio), 1.0),
            "temporal_cv_median": round(float(np.median(std_spec[b] / np.maximum(med_spec[b], 1e-12))), 4),
        }

    max_across = np.max(med_spec, axis=0)
    top_idx = np.argsort(max_across)[::-1][:10]
    top_ch = [
        {
            "freq_mhz": round(float(sub_f[i]), 2),
            "max_med_power": round(float(max_across[i]), 3),
            "best_beam": int(np.argmax(med_spec[:, i])),
            "best_label": labels[int(np.argmax(med_spec[:, i]))],
            "ref_power": round(float(med_spec[ref_beam, i]), 3),
            "zenith_power": round(float(med_spec[zen_beam, i]), 3),
            "contrast_vs_ref": round(float(max_across[i] / max(med_spec[ref_beam, i], 1e-12)), 3),
        }
        for i in top_idx
    ]
    return {
        "fmin_mhz": float(f0),
        "fmax_mhz": float(f1),
        "n_channels": int(np.sum(mask)),
        "beams": beam_stats,
        "top_channels": top_ch,
    }


def run_geo_mode(run_dir: Path) -> Dict[str, Any]:
    wf, seq, _, timing = load_dual_subband_beam_power(run_dir, n_beams=10)
    pointing = load_jsonl(run_dir / "beam_pointing_log.jsonl")
    rfi_snaps = load_jsonl(run_dir / "rfi_channel_snapshots.jsonl")
    clean_mask = clean_channel_mask(wf[:, 9, :])

    band_defs = [
        ("band_300_355_uhf_milsatcom", 300.0, 355.0),
        ("band_356_385_muos", 356.0, 385.0),
        ("band_399_405_dcs", 399.0, 405.0),
        ("band_415_440_425mhz", 415.0, 440.0),
        ("band_464_486_goes_475mhz", 464.0, 486.0),
    ]
    bands_out: Dict[str, Any] = {}
    for bname, f0, f1 in band_defs:
        m = (FREQS_MHZ >= f0) & (FREQS_MHZ <= f1)
        bands_out[bname] = analyze_band_vs_ref(wf, m, ref_beam=8, zen_beam=9, labels=GEO_BEAM_LABELS, f0=f0, f1=f1)
    bands_out["band_full_clean_300_501"] = analyze_band_vs_ref(
        wf, clean_mask, ref_beam=8, zen_beam=9, labels=GEO_BEAM_LABELS, f0=float(FREQS_MHZ[0]), f1=float(FREQS_MHZ[-1])
    )

    # Top 15 contrast peaks across all 672 channels vs OFF-South (beam 8)
    med_all = np.median(wf, axis=0)
    sat_over_off = med_all[:8] / np.maximum(med_all[8][None, :], 1e-12)
    best_sat_per_ch = np.argmax(sat_over_off, axis=0)
    best_ratio_per_ch = np.max(sat_over_off, axis=0)
    top15 = np.argsort(best_ratio_per_ch)[::-1][:15]
    top_spectral_excess = [
        {
            "freq_mhz": round(float(FREQS_MHZ[c]), 2),
            "best_beam": int(best_sat_per_ch[c]),
            "best_label": GEO_BEAM_LABELS[int(best_sat_per_ch[c])],
            "ratio_vs_off_south": round(float(best_ratio_per_ch[c]), 3),
            "db_vs_off_south": safe_db(best_ratio_per_ch[c], 1.0),
            "sat_med_power": round(float(med_all[best_sat_per_ch[c], c]), 3),
            "off_south_power": round(float(med_all[8, c]), 3),
            "zenith_power": round(float(med_all[9, c]), 3),
        }
        for c in top15
    ]

    latest_targets = pointing[-1].get("targets", []) if pointing else []
    rfi_summary = summarize_rfi_snapshots(
        rfi_snaps,
        [(k.replace("band_", ""), f0, f1) for k, f0, f1 in band_defs] + [("468_470_goes", 468.0, 470.0)],
    )

    summary = {
        "mode": "geo_satellites",
        "run_dir": str(run_dir),
        "timing": timing,
        "clean_channel_count": int(np.sum(clean_mask)),
        "latest_pointing_targets": latest_targets,
        "bands": bands_out,
        "top_15_contrast_peaks_vs_off_south": top_spectral_excess,
        "rfi_telemetry_summary": rfi_summary,
    }
    atomic_write_text(run_dir / "geo_satellites_verification.json", json.dumps(summary, indent=2))

    # 2x2 Diagnostic Figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), layout="constrained")
    colors = ["#dc2626", "#2563eb", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#ca8a04", "#db2777", "#0f172a", "#64748b"]

    ax00 = axes[0, 0]
    m00 = (FREQS_MHZ >= 355.0) & (FREQS_MHZ <= 386.0)
    for b in (0, 1, 4, 5, 6, 8, 9):
        ax00.plot(FREQS_MHZ[m00], med_all[b, m00], color=colors[b], lw=1.2, label=GEO_BEAM_LABELS[b])
    ax00.axvline(380.0, color="k", ls=":", lw=1.0, label="380 MHz")
    ax00.set_title("355-386 MHz: MUOS-1/3, UFO-10, SKYNET-5C, USA-115 vs Controls")
    ax00.set_xlabel("Frequency (MHz)")
    ax00.set_ylabel("Median Beam Power")
    ax00.set_yscale("log")
    ax00.grid(True, alpha=0.25)
    ax00.legend(fontsize=7.5, ncol=2)

    ax01 = axes[0, 1]
    m01 = (FREQS_MHZ >= 300.0) & (FREQS_MHZ <= 410.0)
    for b in (2, 3, 4, 5, 6, 7, 8):
        ratio_db = 10.0 * np.log10(np.maximum(med_all[b, m01], 1e-12) / np.maximum(med_all[8, m01], 1e-12))
        ax01.plot(FREQS_MHZ[m01], ratio_db, color=colors[b], lw=1.0, alpha=0.85, label=GEO_BEAM_LABELS[b])
    ax01.axhline(0.0, color="k", ls="--", lw=0.9)
    ax01.axvspan(356.0, 384.0, color="#fef08a", alpha=0.25, label="MUOS 356-384")
    ax01.axvspan(399.0, 405.0, color="#bfdbfe", alpha=0.30, label="DCS 399-405")
    ax01.set_title("300-410 MHz GEO Excess Relative to OFF-South (dB)")
    ax01.set_xlabel("Frequency (MHz)")
    ax01.set_ylabel("Power / OFF-South (dB)")
    ax01.grid(True, alpha=0.25)
    ax01.legend(fontsize=7.5, ncol=2)

    ax10 = axes[1, 0]
    m10 = (FREQS_MHZ >= 464.0) & (FREQS_MHZ <= 486.0)
    for b in (2, 3, 4, 7, 8, 9):
        ax10.plot(FREQS_MHZ[m10], med_all[b, m10], color=colors[b], lw=1.2, label=GEO_BEAM_LABELS[b])
    ax10.axvline(468.83, color="#16a34a", ls="--", lw=1.0, label="GOES DCP 468.83 MHz")
    ax10.axvline(475.0, color="k", ls=":", lw=1.0, label="475 MHz")
    ax10.set_title("464-486 MHz: GOES-19 / GOES-18 & 475 MHz Comparison")
    ax10.set_xlabel("Frequency (MHz)")
    ax10.set_ylabel("Median Beam Power")
    ax10.set_yscale("log")
    ax10.grid(True, alpha=0.25)
    ax10.legend(fontsize=7.5, ncol=2)

    ax11 = axes[1, 1]
    band_keys = [
        ("300-355", "band_300_355_uhf_milsatcom"),
        ("356-385", "band_356_385_muos"),
        ("399-405", "band_399_405_dcs"),
        ("415-440", "band_415_440_425mhz"),
        ("464-486", "band_464_486_goes_475mhz"),
        ("Full Clean", "band_full_clean_300_501"),
    ]
    mat_db = np.zeros((8, len(band_keys)), dtype=np.float64)
    for j, (_, bk) in enumerate(band_keys):
        for i in range(8):
            mat_db[i, j] = bands_out[bk]["beams"][f"beam_{i}"]["median_contrast_db"]
    vmax = max(3.0, float(np.max(np.abs(mat_db))))
    im = ax11.imshow(mat_db, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax11.set_xticks(range(len(band_keys)))
    ax11.set_xticklabels([k[0] for k in band_keys], fontsize=8.5)
    ax11.set_yticks(range(8))
    ax11.set_yticklabels([GEO_BEAM_LABELS[i] for i in range(8)], fontsize=8.0)
    for i in range(8):
        for j in range(len(band_keys)):
            ax11.text(j, i, f"{mat_db[i, j]:+.1f}", ha="center", va="center", fontsize=7.5, color="black")
    ax11.set_title("GEO Satellite Median Contrast vs OFF-South (dB)")
    fig.colorbar(im, ax=ax11, label="Median Ratio (dB)")

    atomic_savefig(fig, run_dir / "geo_satellites_verification.png", dpi=140)
    plt.close(fig)
    return summary


def run_horizon_mode(run_dir: Path) -> Dict[str, Any]:
    wf, seq, _, timing = load_dual_subband_beam_power(run_dir, n_beams=10)
    rfi_snaps = load_jsonl(run_dir / "rfi_channel_snapshots.jsonl")

    # Quiet zenith channels give clean broadband continuum; also evaluate raw all-channel power because DTV/airport tones live in RFI bands
    clean_mask = clean_channel_mask(wf[:, 9, :])
    med_all = np.median(wf, axis=0)  # (10, 672)
    p95_all = np.percentile(wf, 95, axis=0)
    quiet_hor_mean_spec = np.mean(med_all[[5, 6, 7], :], axis=0)

    dt_s = float(timing.get("cadence_s") or 1.0)
    t_rel_min = (seq - seq[0]).astype(np.float64) * (10.0 / 3.0) * 1e-6 / 60.0

    # Broadband & targeted band definitions
    subbands = [
        ("all_300_501", np.ones(len(FREQS_MHZ), dtype=bool)),
        ("clean_continuum", clean_mask),
        ("300_325", (FREQS_MHZ >= 300.0) & (FREQS_MHZ < 325.0)),
        ("325_350", (FREQS_MHZ >= 325.0) & (FREQS_MHZ < 350.0)),
        ("350_375_muos", (FREQS_MHZ >= 350.0) & (FREQS_MHZ < 375.0)),
        ("375_400_muos_edge", (FREQS_MHZ >= 375.0) & (FREQS_MHZ < 400.0)),
        ("400_425", (FREQS_MHZ >= 400.0) & (FREQS_MHZ < 425.0)),
        ("425_450_425band", (FREQS_MHZ >= 425.0) & (FREQS_MHZ < 450.0)),
        ("450_470_pmr", (FREQS_MHZ >= 450.0) & (FREQS_MHZ < 470.0)),
        ("470_501_dtv_uhf", (FREQS_MHZ >= 470.0) & (FREQS_MHZ <= 501.5)),
    ]

    band_excess_db: Dict[str, Dict[str, Any]] = {}
    for bname, bmask in subbands:
        if int(np.sum(bmask)) == 0:
            continue
        q_mean_band = float(np.median(quiet_hor_mean_spec[bmask]))
        q_west_band = float(np.median(med_all[5, bmask]))
        zen_band = float(np.median(med_all[9, bmask]))
        per_beam = {}
        for b in range(10):
            bm = float(np.median(med_all[b, bmask]))
            bp95 = float(np.percentile(p95_all[b, bmask], 95))
            per_beam[f"beam_{b}"] = {
                "label": HORIZON_BEAM_LABELS[b],
                "median_power": round(bm, 3),
                "p95_power": round(bp95, 3),
                "db_vs_quiet_mean": safe_db(bm, q_mean_band),
                "db_vs_quiet_west": safe_db(bm, q_west_band),
                "db_vs_zenith": safe_db(bm, zen_band),
            }
        band_excess_db[bname] = {
            "n_channels": int(np.sum(bmask)),
            "quiet_horizon_mean_power": round(q_mean_band, 3),
            "zenith_power": round(zen_band, 3),
            "beams": per_beam,
        }

    # Key physics / geometry comparisons
    def ratio_stats(b_num: int, b_den: int, bmask: np.ndarray) -> Dict[str, Any]:
        r_ch = med_all[b_num, bmask] / np.maximum(med_all[b_den, bmask], 1e-12)
        return {
            "median_ratio": round(float(np.median(r_ch)), 4),
            "median_db": safe_db(float(np.median(r_ch)), 1.0),
            "p95_ratio": round(float(np.percentile(r_ch, 95)), 4),
            "p95_db": safe_db(float(np.percentile(r_ch, 95)), 1.0),
        }

    discrimination = {
        "scl_airport_el05_vs_el10": {
            "clean": ratio_stats(0, 1, clean_mask),
            "dtv_470_501": ratio_stats(0, 1, (FREQS_MHZ >= 470.0) & (FREQS_MHZ <= 501.5)),
            "all_300_501": ratio_stats(0, 1, np.ones(len(FREQS_MHZ), dtype=bool)),
        },
        "san_cristobal_el1_vs_santiago_el10": {
            "clean": ratio_stats(3, 4, clean_mask),
            "dtv_470_501": ratio_stats(3, 4, (FREQS_MHZ >= 470.0) & (FREQS_MHZ <= 501.5)),
            "all_300_501": ratio_stats(3, 4, np.ones(len(FREQS_MHZ), dtype=bool)),
        },
        "san_cristobal_cal_vs_uncal": {
            "clean": ratio_stats(3, 8, clean_mask),
            "muos_356_385": ratio_stats(3, 8, (FREQS_MHZ >= 356.0) & (FREQS_MHZ <= 385.0)),
            "415_440": ratio_stats(3, 8, (FREQS_MHZ >= 415.0) & (FREQS_MHZ <= 440.0)),
            "dtv_470_501": ratio_stats(3, 8, (FREQS_MHZ >= 470.0) & (FREQS_MHZ <= 501.5)),
        },
    }

    # Broadband time-series bursts / aircraft transients relative to Quiet_West (B5)
    ts_mask = clean_mask if int(np.sum(clean_mask)) >= 50 else np.ones(len(FREQS_MHZ), dtype=bool)
    qwest_ts = np.maximum(np.median(wf[:, 5, ts_mask], axis=1), 1e-12)
    burst_summary = {}
    for b, key in [
        (0, "scl_airport_horizon"),
        (1, "scl_approach_el10"),
        (2, "santiago_centro"),
        (3, "san_cristobal"),
        (6, "quiet_south"),
        (7, "quiet_nw"),
        (9, "zenith"),
    ]:
        rts = np.median(wf[:, b, ts_mask], axis=1) / qwest_ts
        med_r = float(np.median(rts))
        mad_r = 1.4826 * float(np.median(np.abs(rts - med_r)))
        thresh = med_r + 5.0 * max(mad_r, 1e-4)
        burst_idx = np.where(rts > thresh)[0]
        burst_summary[key] = {
            "median_vs_quiet_west": round(med_r, 4),
            "median_db_vs_quiet_west": safe_db(med_r, 1.0),
            "p99_over_median": round(float(np.percentile(rts, 99) / max(med_r, 1e-12)), 4),
            "burst_frames_gt5mad": int(len(burst_idx)),
            "peak_time_min": round(float(t_rel_min[int(np.argmax(rts))]), 2) if len(t_rel_min) else 0.0,
            "peak_ratio_vs_quiet_west": round(float(np.max(rts)), 4),
        }

    # Top 15 excess spectral peaks in urban+airport (B0..B3) over quiet horizon mean
    urban_max = np.max(med_all[[0, 2, 3], :], axis=0)
    urban_winner = np.argmax(med_all[[0, 2, 3], :], axis=0)
    urban_map = [0, 2, 3]
    excess_ratio = urban_max / np.maximum(quiet_hor_mean_spec, 1e-12)
    top15 = np.argsort(excess_ratio)[::-1][:15]
    top_urban_peaks = [
        {
            "freq_mhz": round(float(FREQS_MHZ[c]), 2),
            "best_beam": urban_map[int(urban_winner[c])],
            "best_label": HORIZON_BEAM_LABELS[urban_map[int(urban_winner[c])]],
            "excess_vs_quiet_mean": round(float(excess_ratio[c]), 3),
            "excess_db": safe_db(excess_ratio[c], 1.0),
            "urban_power": round(float(urban_max[c]), 3),
            "quiet_mean_power": round(float(quiet_hor_mean_spec[c]), 3),
            "zenith_power": round(float(med_all[9, c]), 3),
        }
        for c in top15
    ]

    rfi_summary = summarize_rfi_snapshots(
        rfi_snaps,
        [("300_355", 300.0, 355.0), ("356_385", 356.0, 385.0), ("415_440", 415.0, 440.0), ("450_470", 450.0, 470.0), ("470_501", 470.0, 501.5)],
    )

    summary = {
        "mode": "airport_santiago_horizon",
        "run_dir": str(run_dir),
        "timing": timing,
        "cadence_s": round(dt_s, 3),
        "clean_channel_count": int(np.sum(clean_mask)),
        "band_excess_vs_controls": band_excess_db,
        "discrimination_tests": discrimination,
        "temporal_burst_stats_vs_quiet_west": burst_summary,
        "top_15_urban_excess_peaks": top_urban_peaks,
        "rfi_telemetry_summary": rfi_summary,
    }
    atomic_write_text(run_dir / "airport_santiago_verification.json", json.dumps(summary, indent=2))

    # 2x2 Diagnostic Figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), layout="constrained")
    ax00 = axes[0, 0]
    for b, col, lw in [
        (0, "#dc2626", 1.2),
        (2, "#ea580c", 1.1),
        (3, "#7c3aed", 1.2),
        (8, "#c084fc", 0.9),
        (9, "#0f172a", 1.1),
    ]:
        ax00.plot(FREQS_MHZ, med_all[b], color=col, lw=lw, label=HORIZON_BEAM_LABELS[b], alpha=0.9)
    ax00.plot(FREQS_MHZ, quiet_hor_mean_spec, color="#16a34a", lw=1.2, ls="--", label="Mean(Quiet W/S/NW El 0.5°)")
    ax00.set_title("300-501.3 MHz Horizon Beams vs Quiet Horizon & Zenith")
    ax00.set_xlabel("Frequency (MHz)")
    ax00.set_ylabel("Median Beam Power")
    ax00.set_yscale("log")
    ax00.grid(True, alpha=0.25)
    ax00.legend(fontsize=7.5, ncol=2)

    ax01 = axes[0, 1]
    bar_bands = [
        ("Clean", "clean_continuum"),
        ("300-350", "300_325"),
        ("375-400", "375_400_muos_edge"),
        ("425-450", "425_450_425band"),
        ("450-470", "450_470_pmr"),
        ("470-501 DTV", "470_501_dtv_uhf"),
    ]
    bar_beams = [
        (0, "SCL Airport 0.5°", "#dc2626"),
        (1, "SCL App 10°", "#f87171"),
        (2, "Stgo Centro 0.5°", "#ea580c"),
        (3, "San Cristobal 1°", "#7c3aed"),
        (4, "Stgo El10 10°", "#0284c7"),
        (9, "Zenith", "#64748b"),
    ]
    x = np.arange(len(bar_bands))
    w = 0.13
    for k, (bid, blbl, bcol) in enumerate(bar_beams):
        vals = [
            band_excess_db[bk]["beams"][f"beam_{bid}"]["db_vs_quiet_mean"] if bk in band_excess_db else 0.0
            for _, bk in bar_bands
        ]
        ax01.bar(x + (k - 2.5) * w, vals, width=w, color=bcol, label=blbl)
    ax01.axhline(0.0, color="#16a34a", ls="--", lw=1.0, label="Quiet Horizon Mean (0 dB)")
    ax01.set_xticks(x)
    ax01.set_xticklabels([b[0] for b in bar_bands], fontsize=8.5)
    ax01.set_title("Urban & Airport Excess Power vs Quiet Horizon Controls (dB)")
    ax01.set_ylabel("Excess over Quiet Horizon Mean (dB)")
    ax01.grid(True, axis="y", alpha=0.25)
    ax01.legend(fontsize=7.5, ncol=2)

    ax10 = axes[1, 0]
    ax10.plot(
        FREQS_MHZ,
        10.0 * np.log10(np.maximum(med_all[0], 1e-12) / np.maximum(med_all[1], 1e-12)),
        color="#dc2626",
        lw=1.1,
        label="SCL El 0.5° / El 10°",
    )
    ax10.plot(
        FREQS_MHZ,
        10.0 * np.log10(np.maximum(med_all[3], 1e-12) / np.maximum(med_all[4], 1e-12)),
        color="#7c3aed",
        lw=1.1,
        label="San Cristobal El 1° / Stgo El 10°",
    )
    ax10.plot(
        FREQS_MHZ,
        10.0 * np.log10(np.maximum(med_all[3], 1e-12) / np.maximum(med_all[8], 1e-12)),
        color="#0284c7",
        lw=1.0,
        alpha=0.85,
        label="San Cristobal Cal / Uncal",
    )
    ax10.axhline(0.0, color="k", ls="--", lw=0.8)
    ax10.set_title("Elevation & Calibration Discrimination Ratios (dB)")
    ax10.set_xlabel("Frequency (MHz)")
    ax10.set_ylabel("Power Ratio (dB)")
    ax10.grid(True, alpha=0.25)
    ax10.legend(fontsize=8.0)

    ax11 = axes[1, 1]
    for b, lbl, col in [
        (0, "SCL Airport 0.5° / Quiet W", "#dc2626"),
        (1, "SCL Approach 10° / Quiet W", "#f59e0b"),
        (2, "Stgo Centro 0.5° / Quiet W", "#ea580c"),
        (3, "San Cristobal 1° / Quiet W", "#7c3aed"),
        (6, "Quiet South 0.5° / Quiet W", "#16a34a"),
    ]:
        ts_r = np.median(wf[:, b, ts_mask], axis=1) / qwest_ts
        ax11.plot(t_rel_min, ts_r, color=col, lw=1.0, label=lbl, alpha=0.9)
    ax11.axhline(1.0, color="k", ls="--", lw=0.8)
    ax11.set_title("Clean-Band Horizon Time Series Relative to Quiet West")
    ax11.set_xlabel("Elapsed Time (min)")
    ax11.set_ylabel("Clean Median Power / Quiet West")
    ax11.grid(True, alpha=0.25)
    ax11.legend(fontsize=8.0)

    atomic_savefig(fig, run_dir / "airport_santiago_verification.png", dpi=140)
    plt.close(fig)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--mode", required=True, choices=["geo", "horizon"])
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    if args.mode == "geo":
        summary = run_geo_mode(run_dir)
    else:
        summary = run_horizon_mode(run_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
