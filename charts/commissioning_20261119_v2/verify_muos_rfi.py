#!/usr/bin/env python3
"""Post-test verification for Test 1: MUOS (360-385 MHz) & 425 / 475 MHz
satellite + terrestrial RFI diagnostics (monitor-only, unblanked voltages)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from beam_io import FREQS_MHZ, atomic_savefig, atomic_write_text, load_dual_subband_beam_power, load_jsonl

BEAM_NAMES_DEFAULT = [
    "Beam 0: MUOS-1",
    "Beam 1: MUOS-5",
    "Beam 2: MUOS-3",
    "Beam 3: OFF-MUOS",
    "Beam 4: LEO/GOES",
    "Beam 5: Loc 425.4",
    "Beam 6: Loc 472.5",
    "Beam 7: Zenith",
]


def analyze_band(wf: np.ndarray, fmin: float, fmax: float, ref_beam: int = 3) -> Dict[str, Any]:
    mask = (FREQS_MHZ >= fmin) & (FREQS_MHZ <= fmax)
    sub_f = FREQS_MHZ[mask]
    sub_wf = wf[:, :, mask]  # (T, B, F_sub)
    med_spec = np.median(sub_wf, axis=0)  # (B, F_sub)
    p95_spec = np.percentile(sub_wf, 95, axis=0)
    std_spec = np.std(sub_wf, axis=0)

    ref_med = np.maximum(med_spec[ref_beam], 1e-12)
    beam_stats = {}
    for b in range(wf.shape[1]):
        ratio = med_spec[b] / ref_med
        peak_i = int(np.argmax(med_spec[b]))
        c_ratio_i = int(np.argmax(ratio))
        beam_stats[f"beam_{b}"] = {
            "band_median_power": round(float(np.median(med_spec[b])), 3),
            "band_p95_power": round(float(np.percentile(p95_spec[b], 95)), 3),
            "peak_freq_mhz": round(float(sub_f[peak_i]), 2),
            "peak_power": round(float(med_spec[b, peak_i]), 3),
            "max_contrast_vs_ref": round(float(ratio[c_ratio_i]), 3),
            "max_contrast_freq_mhz": round(float(sub_f[c_ratio_i]), 2),
            "median_contrast_vs_ref": round(float(np.median(ratio)), 3),
            "temporal_cv_median": round(float(np.median(std_spec[b] / np.maximum(med_spec[b], 1e-12))), 4),
        }

    max_across_beams = np.max(med_spec, axis=0)
    top_idx = np.argsort(max_across_beams)[::-1][:10]
    top_ch = [
        {
            "freq_mhz": round(float(sub_f[i]), 2),
            "max_med_power": round(float(max_across_beams[i]), 3),
            "best_beam": int(np.argmax(med_spec[:, i])),
            "off_power": round(float(med_spec[ref_beam, i]), 3),
            "zenith_power": round(float(med_spec[7, i]), 3),
        }
        for i in top_idx
    ]
    return {"fmin_mhz": fmin, "fmax_mhz": fmax, "n_channels": int(mask.sum()), "beams": beam_stats, "top_channels": top_ch}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    wf, seq, t0_us, timing = load_dual_subband_beam_power(run_dir, n_beams=8)
    pointing = load_jsonl(run_dir / "beam_pointing_log.jsonl")
    rfi_snaps = load_jsonl(run_dir / "rfi_channel_snapshots.jsonl")

    t_rel_min = (seq - seq[0]).astype(np.float64) * (10.0 / 3.0) * 1e-6 / 60.0

    # 1. Analyze target frequency bands
    band_muos_all = analyze_band(wf, 356.0, 385.0, ref_beam=3)
    band_muos_380 = analyze_band(wf, 375.0, 385.0, ref_beam=3)
    band_425 = analyze_band(wf, 415.0, 440.0, ref_beam=7)
    band_475 = analyze_band(wf, 464.0, 486.0, ref_beam=7)

    # 2. Extract RFI telemetry flag statistics across 360-385 / 415-440 / 464-486 MHz
    rfi_band_summary = {}
    if rfi_snaps:
        flag_accum = np.zeros(672, dtype=np.float64)
        sk_accum = np.zeros(672, dtype=np.float64)
        n_s = 0
        for snap in rfi_snaps:
            ch_list = snap.get("data", {}).get("channels", [])
            if len(ch_list) >= 672:
                n_s += 1
                for entry in ch_list:
                    if "global_channel" in entry:
                        idx = int(entry["global_channel"])
                    elif "subband" in entry and "local_channel" in entry:
                        idx = int(entry["subband"]) * 336 + int(entry["local_channel"])
                    else:
                        idx = -1
                    if 0 <= idx < 672:
                        flag_accum[idx] += float(
                            entry.get("flagged_chunk_fraction", entry.get("flag_fraction", entry.get("flagged_pct", 0.0)))
                        )
                        sk_accum[idx] += float(entry.get("max_sk", entry.get("mean_sk", entry.get("sk_median", 0.0))))
        if n_s > 0:
            flag_accum /= n_s
            sk_accum /= n_s
            for bname, f0, f1 in [("360_385", 360.0, 385.0), ("415_440", 415.0, 440.0), ("464_486", 464.0, 486.0)]:
                m = (FREQS_MHZ >= f0) & (FREQS_MHZ <= f1)
                rfi_band_summary[bname] = {
                    "mean_flag_metric": round(float(np.mean(flag_accum[m])), 4),
                    "max_flag_metric": round(float(np.max(flag_accum[m])), 4),
                    "peak_sk_freq_mhz": round(float(FREQS_MHZ[m][np.argmax(sk_accum[m])]), 2),
                    "max_sk_metric": round(float(np.max(sk_accum[m])), 3),
                }

    # 3. Track Beam 4 LEO targets over time from pointing log
    b4_targets = []
    if pointing:
        last_lbl = None
        for row in pointing:
            tgts = row.get("targets", [])
            if len(tgts) > 4:
                lbl = tgts[4].get("label", "")
                if lbl != last_lbl:
                    b4_targets.append(
                        {"utc": row["utc"], "label": lbl, "el_deg": tgts[4].get("el_deg"), "az_deg": tgts[4].get("az_deg")}
                    )
                    last_lbl = lbl

    summary = {
        "run_dir": str(run_dir),
        "timing": timing,
        "beam_4_tracked_targets": b4_targets,
        "band_356_385_muos": band_muos_all,
        "band_375_385_edge": band_muos_380,
        "band_415_440_425mhz": band_425,
        "band_464_486_475mhz": band_475,
        "rfi_telemetry_summary": rfi_band_summary,
    }
    atomic_write_text(run_dir / "muos_425_475_verification.json", json.dumps(summary, indent=2))

    # Diagnostic 2x2 Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    m_muos = (FREQS_MHZ >= 355.0) & (FREQS_MHZ <= 386.0)
    ax00 = axes[0, 0]
    for b, col in [(0, "#dc2626"), (1, "#f59e0b"), (2, "#2563eb"), (3, "#0f172a"), (7, "#64748b")]:
        ax00.plot(FREQS_MHZ[m_muos], np.median(wf[:, b, m_muos], axis=0), color=col, lw=1.2, label=BEAM_NAMES_DEFAULT[b])
    ax00.axvline(380.0, color="k", ls=":", lw=1.0, label="380 MHz")
    ax00.set_title("355-386 MHz MUOS Downlink & 380 MHz Edge Spectrum")
    ax00.set_xlabel("Frequency (MHz)")
    ax00.set_ylabel("Median Beam Power")
    ax00.set_yscale("log")
    ax00.grid(True, alpha=0.25)
    ax00.legend(fontsize=8.0)

    ax01 = axes[0, 1]
    m_425 = (FREQS_MHZ >= 415.0) & (FREQS_MHZ <= 440.0)
    for b, col in [(5, "#dc2626"), (4, "#16a34a"), (3, "#2563eb"), (7, "#64748b")]:
        ax01.plot(FREQS_MHZ[m_425], np.median(wf[:, b, m_425], axis=0), color=col, lw=1.2, label=BEAM_NAMES_DEFAULT[b])
    ax01.axvline(425.0, color="k", ls=":", lw=1.0, label="425 MHz")
    ax01.set_title("415-440 MHz Spectrum (425 MHz + 70cm Satellite Band)")
    ax01.set_xlabel("Frequency (MHz)")
    ax01.set_ylabel("Median Beam Power")
    ax01.set_yscale("log")
    ax01.grid(True, alpha=0.25)
    ax01.legend(fontsize=8.0)

    ax10 = axes[1, 0]
    m_475 = (FREQS_MHZ >= 464.0) & (FREQS_MHZ <= 486.0)
    for b, col in [(6, "#dc2626"), (4, "#16a34a"), (3, "#2563eb"), (7, "#64748b")]:
        ax10.plot(FREQS_MHZ[m_475], np.median(wf[:, b, m_475], axis=0), color=col, lw=1.2, label=BEAM_NAMES_DEFAULT[b])
    ax10.axvline(465.988, color="#16a34a", ls="--", lw=0.9, label="465.988 ARGOS-3")
    ax10.axvline(475.0, color="k", ls=":", lw=1.0, label="475 MHz")
    ax10.set_title("464-486 MHz Spectrum (475 MHz + ARGOS-3/GOES/ISS)")
    ax10.set_xlabel("Frequency (MHz)")
    ax10.set_ylabel("Median Beam Power")
    ax10.set_yscale("log")
    ax10.grid(True, alpha=0.25)
    ax10.legend(fontsize=8.0)

    ax11 = axes[1, 1]
    m_380 = (FREQS_MHZ >= 375.5) & (FREQS_MHZ <= 381.0)
    m_argos = (FREQS_MHZ >= 465.5) & (FREQS_MHZ <= 469.5)
    ax11.plot(
        t_rel_min,
        np.mean(wf[:, 0, m_380], axis=1) / np.maximum(np.mean(wf[:, 3, m_380], axis=1), 1e-12),
        label="MUOS-1 / OFF (375.5-381 MHz)",
        color="#dc2626",
    )
    ax11.plot(
        t_rel_min,
        np.mean(wf[:, 4, m_argos], axis=1) / np.maximum(np.mean(wf[:, 7, m_argos], axis=1), 1e-12),
        label="Beam 4 Track / Zenith (465.5-469.5 MHz)",
        color="#16a34a",
    )
    ax11.set_title("Normalized Target-to-Control Contrast Over Time")
    ax11.set_xlabel("Elapsed Time (min)")
    ax11.set_ylabel("ON / Control Power Ratio")
    ax11.grid(True, alpha=0.25)
    ax11.legend(fontsize=8.0)

    atomic_savefig(fig, run_dir / "muos_425_475_diagnostics.png", dpi=140)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
