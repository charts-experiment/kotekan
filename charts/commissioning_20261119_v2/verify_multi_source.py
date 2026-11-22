#!/usr/bin/env python3
"""Post-test verification for Test 3 (Deep-Night Multi-Source: Pictor A,
Orion A, LMC, Crab) and Test 7 (Cen A Dual-Source + Sun Monitor)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

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
    theoretical_array_factor_sq,
)


def eval_pair(wf: np.ndarray, clean: np.ndarray, on_b: int, off_b: int, name: str) -> Dict[str, Any]:
    on_ts = np.mean(wf[:, on_b, clean], axis=1)
    off_ts = np.mean(wf[:, off_b, clean], axis=1)
    exc = (on_ts - off_ts) / np.maximum(off_ts, 1e-12)
    mean_exc = float(np.mean(exc))
    se_exc = float(np.std(exc) / np.sqrt(max(len(exc), 1)))
    mid_i = max(1, len(exc) // 2)
    h1_exc = float(np.mean(exc[:mid_i]))
    h2_exc = float(np.mean(exc[mid_i:])) if mid_i < len(exc) else h1_exc
    # Fit spectral contrast slope across ~14 frequency bins
    edges = np.linspace(300.0, 501.5, 15)
    fs, rs = [], []
    for i in range(len(edges) - 1):
        mb = clean & (FREQS_MHZ >= edges[i]) & (FREQS_MHZ < edges[i + 1])
        if np.sum(mb) < 3:
            continue
        p_on = float(np.median(np.mean(wf[:, on_b, mb], axis=1)))
        p_off = float(np.median(np.mean(wf[:, off_b, mb], axis=1)))
        r = (p_on - p_off) / max(p_off, 1e-12)
        if r > 0.005:
            fs.append(float(np.mean(FREQS_MHZ[mb])))
            rs.append(r)
    alpha_slope = round(float(np.polyfit(np.log(np.array(fs) / 400.0), np.log(rs), 1)[0]), 3) if len(fs) >= 4 else None
    return {
        "name": name,
        "on_beam": on_b,
        "off_beam": off_b,
        "mean_on_power": round(float(np.mean(on_ts)), 4),
        "mean_off_power": round(float(np.mean(off_ts)), 4),
        "mean_excess_frac": round(mean_exc, 5),
        "first_half_excess_frac": round(h1_exc, 5),
        "second_half_excess_frac": round(h2_exc, 5),
        "std_err_frac": round(se_exc, 5),
        "snr_mean_over_se": round(mean_exc / max(se_exc, 1e-12), 2),
        "excess_spectral_slope": alpha_slope,
        "subband_freqs_mhz": [round(x, 2) for x in fs],
        "subband_excess_frac": [round(float(x), 4) for x in rs],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--mode", default="auto", choices=["auto", "multi_source", "cena"])
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    mode = args.mode
    if mode == "auto":
        mode = "cena" if "cena" in run_dir.name.lower() else "multi_source"

    n_beams = 6 if mode == "cena" else 10
    wf, seq, t0_us, timing = load_dual_subband_beam_power(run_dir, n_beams=n_beams)
    pointing = load_jsonl(run_dir / "beam_pointing_log.jsonl")
    t_rel_min = (seq - seq[0]).astype(np.float64) * (10.0 / 3.0) * 1e-6 / 60.0

    clean = clean_channel_mask(wf[:, 0, :], max_zero_frac=0.08, mad_sigma=5.0)
    if int(np.sum(clean)) < 30:
        clean = np.mean(np.isfinite(wf[:, 0, :]) & (wf[:, 0, :] > 0.0), axis=0) > 0.85

    pairs: List[Dict[str, Any]] = []
    if mode == "cena":
        pairs.append(eval_pair(wf, clean, 0, 5, "CenA_Cal_ON vs Deep_OFF"))
        pairs.append(eval_pair(wf, clean, 1, 5, "CenA_Uncal_ON vs Deep_OFF"))
        pairs.append(eval_pair(wf, clean, 2, 5, "CenA_E+5deg vs Deep_OFF"))
        pairs.append(eval_pair(wf, clean, 3, 5, "CenA_W-5deg vs Deep_OFF"))
        pairs.append(eval_pair(wf, clean, 4, 5, "Sun_Monitor vs Deep_OFF"))
    else:
        pairs.append(eval_pair(wf, clean, 0, 4, "PictorA_Cal vs OFF_Pictor"))
        pairs.append(eval_pair(wf, clean, 1, 5, "OrionA_Cal vs OFF_Orion"))
        pairs.append(eval_pair(wf, clean, 2, 6, "LMC_30Dor_Cal vs OFF_LMC"))
        pairs.append(eval_pair(wf, clean, 3, 7, "Crab_TauA_Cal vs OFF_Crab"))
        pairs.append(eval_pair(wf, clean, 8, 4, "PictorA_Uncal vs OFF_Pictor"))
        pairs.append(eval_pair(wf, clean, 9, 4, "Zenith vs OFF_Pictor"))

    sun_af_sidelobe_check: Dict[str, Any] = {}
    if mode == "cena" and pointing:
        sun_bid = 4
        check_bids = [0, 2, 3, 5]
        t_rel_s = (seq - seq[0]).astype(np.float64) * (10.0 / 3.0) * 1e-6
        t0_unix = float(pointing[0]["unix"])
        p_mat = np.zeros((len(pointing), n_beams), dtype=np.float64)
        af_mat = np.zeros((len(pointing), n_beams), dtype=np.float64)
        for si, snap in enumerate(pointing):
            idx_t = int(np.argmin(np.abs(t_rel_s - (float(snap["unix"]) - t0_unix))))
            tg = {int(x["beam_id"]): x for x in snap.get("targets", [])}
            if sun_bid not in tg:
                continue
            s_tgt = tg[sun_bid]
            for bid in range(n_beams):
                p_mat[si, bid] = float(np.mean(wf[idx_t, bid, clean]))
                if bid in tg:
                    af_mat[si, bid] = float(
                        np.mean(
                            theoretical_array_factor_sq(
                                float(tg[bid]["l0"]),
                                float(tg[bid]["m0"]),
                                float(s_tgt["l0"]),
                                float(s_tgt["m0"]),
                                FREQS_MHZ[clean],
                            )
                        )
                    )
        obs_p = p_mat[:, check_bids].ravel()
        obs_af = af_mat[:, check_bids].ravel()
        if np.std(obs_af) > 1e-6 and np.std(obs_p) > 1e-12:
            af_slope, af_intercept = np.polyfit(obs_af, obs_p, 1)
            r_corr = float(np.corrcoef(obs_af, obs_p)[0, 1])
            resid0 = p_mat[:, 0] - (af_intercept + af_slope * af_mat[:, 0])
            resid5 = p_mat[:, 5] - (af_intercept + af_slope * af_mat[:, 5])
            sun_af_sidelobe_check = {
                "pearson_r": round(r_corr, 4),
                "slope_per_af": round(float(af_slope), 4),
                "intercept_power": round(float(af_intercept), 4),
                "mean_af_sun_by_beam": {f"beam_{b}": round(float(np.mean(af_mat[:, b])), 4) for b in check_bids},
                "sun_af_subtracted_cena_excess_frac": round(float(np.mean(resid0 - resid5) / max(abs(af_intercept), 1e-12)), 5),
            }

    summary = {
        "run_dir": str(run_dir),
        "mode": mode,
        "timing": timing,
        "clean_channels": int(np.sum(clean)),
        "pointing_samples": len(pointing),
        "sources": pairs,
        "sun_af_sidelobe_check": sun_af_sidelobe_check,
    }
    atomic_write_text(run_dir / "multi_source_verification.json", json.dumps(summary, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), layout="constrained")
    for s in pairs:
        b_on, b_off = s["on_beam"], s["off_beam"]
        ts = np.mean(wf[:, b_on, clean] - wf[:, b_off, clean], axis=1) / np.maximum(np.mean(wf[:, b_off, clean], axis=1), 1e-12)
        axes[0].plot(t_rel_min, ts, lw=1.1, label=f"{s['name']} ({s['mean_excess_frac']:+.4f})")
        if s["subband_freqs_mhz"]:
            axes[1].plot(s["subband_freqs_mhz"], s["subband_excess_frac"], "o-", ms=3.5, label=s["name"])

    axes[0].set_title(f"{mode}: ON - OFF Fractional Lightcurves")
    axes[0].set_xlabel("Elapsed Time (min)")
    axes[0].set_ylabel("(ON - OFF) / OFF")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=7.5)

    axes[1].set_title(f"{mode}: Subband Excess Spectra")
    axes[1].set_xlabel("Frequency (MHz)")
    axes[1].set_ylabel("(ON - OFF) / OFF")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=7.5)

    atomic_savefig(fig, run_dir / "multi_source_verification.png", dpi=140)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
