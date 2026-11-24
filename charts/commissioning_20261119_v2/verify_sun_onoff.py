#!/usr/bin/env python3
"""Post-test verification for Test 6A/6B: Morning Sun multi-beam ON/OFF test.

Checks:
1. Off-zenith steering geometry & antenna coordinate consistency (comparing Sun ON
   vs OFF-N / OFF-S excess against theoretical 4-antenna array factor).
2. Coherent phasing & conjugate_gains sign convention (Beam 0 Cal vs Beam 3 Uncal,
   plus Phase 6A conjugate_gains=true vs Phase 6B conjugate_gains=false).
3. Overlapping Allan deviation sigma_y(tau) from 1.024 s toward 60-300 s on ON,
   OFF-N, OFF-S, and ON-OFF differenced power to quantify 1/f drift cancellation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from beam_io import (
    FREQS_MHZ,
    POS4,
    atomic_savefig,
    atomic_write_text,
    clean_channel_mask,
    effective_beam_response_sq,
    load_dual_subband_beam_power,
    load_jsonl,
    overlapping_allan_deviation,
    overlapping_hadamard_deviation,
    primary_beam_gain,
    theoretical_array_factor_sq,
)


def summarize_sun_run(run_dir: Path) -> Dict[str, Any]:
    wf, seq, t0_us, timing = load_dual_subband_beam_power(run_dir, n_beams=4)
    pointing = load_jsonl(run_dir / "beam_pointing_log.jsonl")

    clean = clean_channel_mask(wf[:, 0, :], max_zero_frac=0.05, mad_sigma=5.0)
    if int(np.sum(clean)) < 30:
        clean = np.all(np.isfinite(wf[:, 0, :]) & (wf[:, 0, :] > 0.0), axis=0)

    ts_clean = np.mean(wf[:, :, clean], axis=2)  # [Sun_Cal, OFF_N, OFF_S, Sun_Uncal]
    ts_sb0 = np.mean(wf[:, :, clean & (FREQS_MHZ < 400.65)], axis=2) if np.any(clean & (FREQS_MHZ < 400.65)) else ts_clean
    ts_sb1 = np.mean(wf[:, :, clean & (FREQS_MHZ >= 400.65)], axis=2) if np.any(clean & (FREQS_MHZ >= 400.65)) else ts_clean

    p_on = ts_clean[:, 0]
    p_off_n = ts_clean[:, 1]
    p_off_s = ts_clean[:, 2]
    p_uncal = ts_clean[:, 3]

    exc_n = (p_on - p_off_n) / np.maximum(p_off_n, 1e-12)
    exc_s = (p_on - p_off_s) / np.maximum(p_off_s, 1e-12)
    cal_vs_uncal = p_on / np.maximum(p_uncal, 1e-12)

    af_n_mean, af_s_mean, sep_n_deg, sep_s_deg = None, None, None, None
    eff_n_mean, eff_s_mean, b_sun_mean = None, None, None
    if pointing:
        mid = pointing[len(pointing) // 2]["targets"]
        b0, b1, b2 = mid[0], mid[1], mid[2]
        n0 = np.sqrt(max(0.0, 1.0 - b0["l0"] ** 2 - b0["m0"] ** 2))
        n1 = np.sqrt(max(0.0, 1.0 - b1["l0"] ** 2 - b1["m0"] ** 2))
        n2 = np.sqrt(max(0.0, 1.0 - b2["l0"] ** 2 - b2["m0"] ** 2))
        sep_n_deg = float(np.rad2deg(np.arccos(np.clip(b0["l0"] * b1["l0"] + b0["m0"] * b1["m0"] + n0 * n1, -1.0, 1.0))))
        sep_s_deg = float(np.rad2deg(np.arccos(np.clip(b0["l0"] * b2["l0"] + b0["m0"] * b2["m0"] + n0 * n2, -1.0, 1.0))))
        af_n_mean = float(np.mean(theoretical_array_factor_sq(b1["l0"], b1["m0"], b0["l0"], b0["m0"], FREQS_MHZ[clean])))
        af_s_mean = float(np.mean(theoretical_array_factor_sq(b2["l0"], b2["m0"], b0["l0"], b0["m0"], FREQS_MHZ[clean])))
        b_sun_mean = float(np.mean(primary_beam_gain(b0["l0"], b0["m0"], FREQS_MHZ[clean])))
        eff_n_mean = float(np.mean(effective_beam_response_sq(b1["l0"], b1["m0"], b0["l0"], b0["m0"], FREQS_MHZ[clean])))
        eff_s_mean = float(np.mean(effective_beam_response_sq(b2["l0"], b2["m0"], b0["l0"], b0["m0"], FREQS_MHZ[clean])))

    dt_s = timing["cadence_s"] if timing["cadence_s"] > 0 else 1.024
    taus_target = np.geomspace(dt_s, min(300.0, max(dt_s * 4.0, (len(p_on) * dt_s) / 3.0)), 25)

    y_on = p_on / np.mean(p_on)
    y_off_n = p_off_n / np.mean(p_off_n)
    y_ratio_n = p_on / np.maximum(p_off_n, 1e-12)
    y_ratio_n /= np.mean(y_ratio_n)
    y_diff_n = (p_on - p_off_n) / np.mean(p_off_n)
    y_diff_n = y_diff_n - np.mean(y_diff_n) + 1.0

    y_ratio_s = p_on / np.maximum(p_off_s, 1e-12)
    y_ratio_s /= np.mean(y_ratio_s)

    taus_on, adev_on = overlapping_allan_deviation(y_on, dt_s, taus_target)
    _, adev_off_n = overlapping_allan_deviation(y_off_n, dt_s, taus_target)
    _, adev_ratio_n = overlapping_allan_deviation(y_ratio_n, dt_s, taus_target)
    _, adev_diff_n = overlapping_allan_deviation(y_diff_n, dt_s, taus_target)

    th_on, hdev_on = overlapping_hadamard_deviation(y_on, dt_s, taus_target)
    _, hdev_off_n = overlapping_hadamard_deviation(y_off_n, dt_s, taus_target)
    _, hdev_ratio_n = overlapping_hadamard_deviation(y_ratio_n, dt_s, taus_target)
    _, hdev_ratio_s = overlapping_hadamard_deviation(y_ratio_s, dt_s, taus_target)

    def _opt_tau(taus, adev):
        if len(taus) == 0:
            return None, None
        i = int(np.argmin(adev))
        return round(float(taus[i]), 2), round(float(adev[i]), 6)

    tau_opt_on, min_adev_on = _opt_tau(taus_on, adev_on)
    tau_opt_off, _ = _opt_tau(taus_on, adev_off_n)
    tau_opt_ratio, min_adev_ratio = _opt_tau(taus_on, adev_ratio_n)
    tau_opt_diff, min_adev_diff = _opt_tau(taus_on, adev_diff_n)
    tau_opt_hratio_n, min_hdev_ratio_n = _opt_tau(th_on, hdev_ratio_n)

    def _at_tau(taus, adev, target):
        if len(taus) == 0:
            return None
        return round(float(adev[int(np.argmin(np.abs(taus - target)))]), 6)

    return {
        "run_dir": str(run_dir),
        "timing": timing,
        "clean_channels": int(np.sum(clean)),
        "mean_power_clean": {
            "beam0_sun_cal": round(float(np.mean(p_on)), 4),
            "beam1_off_n": round(float(np.mean(p_off_n)), 4),
            "beam2_off_s": round(float(np.mean(p_off_s)), 4),
            "beam3_sun_uncal": round(float(np.mean(p_uncal)), 4),
        },
        "on_off_excess_median": {
            "sun_vs_off_n": round(float(np.median(exc_n)), 4),
            "sun_vs_off_s": round(float(np.median(exc_s)), 4),
            "sun_cal_over_uncal": round(float(np.median(cal_vs_uncal)), 4),
            "sb0_cal_over_uncal": round(float(np.median(ts_sb0[:, 0] / np.maximum(ts_sb0[:, 3], 1e-12))), 4),
            "sb1_cal_over_uncal": round(float(np.median(ts_sb1[:, 0] / np.maximum(ts_sb1[:, 3], 1e-12))), 4),
        },
        "geometry_check": {
            "sep_off_n_deg": round(sep_n_deg, 2) if sep_n_deg is not None else None,
            "sep_off_s_deg": round(sep_s_deg, 2) if sep_s_deg is not None else None,
            "theoretical_af_sq_off_n": round(af_n_mean, 4) if af_n_mean is not None else None,
            "theoretical_af_sq_off_s": round(af_s_mean, 4) if af_s_mean is not None else None,
            "sun_primary_beam_gain": round(b_sun_mean, 4) if b_sun_mean is not None else None,
            "effective_beam_sq_off_n": round(eff_n_mean, 4) if eff_n_mean is not None else None,
            "effective_beam_sq_off_s": round(eff_s_mean, 4) if eff_s_mean is not None else None,
        },
        "allan_variance": {
            "taus_s": [round(float(x), 3) for x in taus_on],
            "adev_on": [round(float(x), 6) for x in adev_on],
            "adev_off_n": [round(float(x), 6) for x in adev_off_n],
            "adev_on_div_off": [round(float(x), 6) for x in adev_ratio_n],
            "adev_on_minus_off": [round(float(x), 6) for x in adev_diff_n],
            "tau_opt_on_s": tau_opt_on,
            "min_adev_on": min_adev_on,
            "tau_opt_off_s": tau_opt_off,
            "tau_opt_ratio_s": tau_opt_ratio,
            "min_adev_ratio": min_adev_ratio,
            "tau_opt_diff_s": tau_opt_diff,
            "min_adev_diff": min_adev_diff,
            "adev_on_16s": _at_tau(taus_on, adev_on, 16.0),
            "adev_on_120s": _at_tau(taus_on, adev_on, 120.0),
            "adev_ratio_16s": _at_tau(taus_on, adev_ratio_n, 16.0),
            "adev_ratio_120s": _at_tau(taus_on, adev_ratio_n, 120.0),
        },
        "hadamard_variance": {
            "taus_s": [round(float(x), 3) for x in th_on],
            "hdev_on": [round(float(x), 6) for x in hdev_on],
            "hdev_off_n": [round(float(x), 6) for x in hdev_off_n],
            "hdev_on_div_off_n": [round(float(x), 6) for x in hdev_ratio_n],
            "hdev_on_div_off_s": [round(float(x), 6) for x in hdev_ratio_s],
            "tau_opt_ratio_n_s": tau_opt_hratio_n,
            "min_hdev_ratio_n": min_hdev_ratio_n,
            "hdev_ratio_16s": _at_tau(th_on, hdev_ratio_n, 16.0),
            "hdev_ratio_65s": _at_tau(th_on, hdev_ratio_n, 65.0),
            "hdev_ratio_120s": _at_tau(th_on, hdev_ratio_n, 120.0),
            "hdev_ratio_300s": _at_tau(th_on, hdev_ratio_n, 300.0),
        },
        "_arrays": {
            "t_s": (seq - seq[0]).astype(np.float64) * (10.0 / 3.0) * 1e-6,
            "ts_clean": ts_clean,
            "clean": clean,
            "wf": wf,
            "taus": taus_on,
            "adev_on": adev_on,
            "adev_off_n": adev_off_n,
            "adev_ratio_n": adev_ratio_n,
            "adev_diff_n": adev_diff_n,
            "taus_hdev": th_on,
            "hdev_on": hdev_on,
            "hdev_ratio_n": hdev_ratio_n,
        },
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--conj-false-dir", default=None)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    res_a = summarize_sun_run(run_dir)
    arr_a = res_a.pop("_arrays")

    conj_false_dir = Path(args.conj_false_dir) if args.conj_false_dir else run_dir.parent / "test06b_sun_onoff_conj_false"
    res_b = None
    if conj_false_dir.exists() and any(conj_false_dir.glob("beam_power_freq0_*.bin")) and conj_false_dir.resolve() != run_dir.resolve():
        res_b = summarize_sun_run(conj_false_dir)
        res_b.pop("_arrays", None)

    out = {"primary_run": res_a, "conj_false_control": res_b}
    if res_b is not None:
        out["conjugate_gains_ab_comparison"] = {
            "conj_true_cal_over_uncal": res_a["on_off_excess_median"]["sun_cal_over_uncal"],
            "conj_false_cal_over_uncal": res_b["on_off_excess_median"]["sun_cal_over_uncal"],
            "conj_true_on_off_n_excess": res_a["on_off_excess_median"]["sun_vs_off_n"],
            "conj_false_on_off_n_excess": res_b["on_off_excess_median"]["sun_vs_off_n"],
        }

    atomic_write_text(run_dir / "sun_onoff_verification.json", json.dumps(out, indent=2))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    ax00 = axes[0, 0]
    t_min = arr_a["t_s"] / 60.0
    ts_c = arr_a["ts_clean"]
    ax00.plot(t_min, ts_c[:, 0], label="Beam 0: Sun Cal ON", lw=1.4, color="#dc2626")
    ax00.plot(t_min, ts_c[:, 1], label="Beam 1: OFF-N (same El)", lw=1.2, color="#2563eb")
    ax00.plot(t_min, ts_c[:, 2], label="Beam 2: OFF-S (same El)", lw=1.2, color="#16a34a")
    ax00.plot(t_min, ts_c[:, 3], label="Beam 3: Sun Uncal ON", lw=1.2, color="#f59e0b", ls="--")
    ax00.set_title("Clean-Band Multi-Beam Lightcurves")
    ax00.set_xlabel("Elapsed Time (min)")
    ax00.set_ylabel("Mean Beam Power")
    ax00.grid(True, alpha=0.25)
    ax00.legend(fontsize=8.5)

    ax01 = axes[0, 1]
    taus = arr_a["taus"]
    if len(taus) > 1:
        ax01.loglog(taus, arr_a["adev_on"], "o-", ms=3, color="#dc2626", label=f"Raw Sun ON (opt={res_a['allan_variance']['tau_opt_on_s']}s)")
        ax01.loglog(taus, arr_a["adev_off_n"], "s-", ms=3, color="#2563eb", label=f"Raw OFF-N (opt={res_a['allan_variance']['tau_opt_off_s']}s)")
        ax01.loglog(taus, arr_a["adev_ratio_n"], "^-", ms=4, color="#16a34a", lw=2.0, label=f"ADEV ON/OFF-N (opt={res_a['allan_variance']['tau_opt_ratio_s']}s)")
        ax01.loglog(taus, arr_a["adev_diff_n"], "v--", ms=3, color="#7c3aed", label=f"ADEV (ON-OFF-N) (opt={res_a['allan_variance']['tau_opt_diff_s']}s)")
        if len(arr_a["taus_hdev"]) > 1:
            ax01.loglog(arr_a["taus_hdev"], arr_a["hdev_ratio_n"], "d-", ms=4, color="#0f766e", lw=1.8, label=f"HDEV ON/OFF-N (opt={res_a['hadamard_variance']['tau_opt_ratio_n_s']}s)")
        ref = arr_a["adev_ratio_n"][0] * np.sqrt(taus[0] / taus)
        ax01.loglog(taus, ref, "k:", lw=1.2, label="Radiometer tau^-1/2")
    ax01.set_title("Allan & Hadamard Deviation (1/f Drift Cancellation)")
    ax01.set_xlabel("Averaging Time tau (s)")
    ax01.set_ylabel("Deviation sigma_y(tau)")
    ax01.grid(True, which="both", alpha=0.25)
    ax01.legend(fontsize=7.5)

    ax10 = axes[1, 0]
    clean = arr_a["clean"]
    spec_med = np.median(arr_a["wf"], axis=0)
    ax10.plot(FREQS_MHZ[clean], spec_med[0, clean], lw=1.1, color="#dc2626", label="Beam 0: Sun Cal ON")
    ax10.plot(FREQS_MHZ[clean], spec_med[1, clean], lw=1.0, color="#2563eb", label="Beam 1: OFF-N")
    ax10.plot(FREQS_MHZ[clean], spec_med[3, clean], lw=1.0, color="#f59e0b", alpha=0.85, label="Beam 3: Sun Uncal")
    ax10.set_title("Median Clean-Channel Spectrum")
    ax10.set_xlabel("Frequency (MHz)")
    ax10.set_ylabel("Beam Power")
    ax10.grid(True, alpha=0.25)
    ax10.legend(fontsize=8.5)

    ax11 = axes[1, 1]
    if res_b is not None:
        cats = ["conj=true\nCal/Uncal", "conj=false\nCal/Uncal", "conj=true\n(ON-OFF)/OFF", "conj=false\n(ON-OFF)/OFF"]
        vals = [
            res_a["on_off_excess_median"]["sun_cal_over_uncal"],
            res_b["on_off_excess_median"]["sun_cal_over_uncal"],
            res_a["on_off_excess_median"]["sun_vs_off_n"],
            res_b["on_off_excess_median"]["sun_vs_off_n"],
        ]
        ax11.bar(cats, vals, color=["#16a34a", "#dc2626", "#2563eb", "#f59e0b"])
        ax11.set_title("conjugate_gains A/B Sign Validation")
    else:
        ratio_ts = ts_c[:, 0] / np.maximum(ts_c[:, 3], 1e-12)
        ax11.plot(t_min, ratio_ts, color="#16a34a", lw=1.3)
        ax11.set_title("Coherent Phasing Gain Ratio (Sun Cal / Sun Uncal)")
        ax11.set_xlabel("Elapsed Time (min)")
        ax11.set_ylabel("Power Ratio")
    ax11.grid(True, alpha=0.25)

    atomic_savefig(fig, run_dir / "sun_onoff_allan_diagnostics.png", dpi=140)
    plt.close(fig)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
