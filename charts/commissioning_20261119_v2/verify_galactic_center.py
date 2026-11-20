#!/usr/bin/env python3
"""Post-test verification for Test 9A: Galactic Center (Sgr A*, l=0 deg)
Galactic latitude cut, synchrotron spectral index, parked zenith drift scan,
and simultaneous Sun sidelobe isolation check."""

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

from beam_io import FREQS_MHZ, clean_channel_mask, load_dual_subband_beam_power, load_jsonl


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    wf, seq, t0_us, timing = load_dual_subband_beam_power(run_dir, n_beams=10)
    pointing = load_jsonl(run_dir / "beam_pointing_log.jsonl")
    t_rel_s = (seq - seq[0]).astype(np.float64) * (10.0 / 3.0) * 1e-6

    off_beam = 9
    drift_beam = 7
    sun_beam = 8
    lat_map = [
        (0, 0.0, "b=0 deg (Sgr A*)"),
        (1, 4.0, "b=+4 deg"),
        (2, -4.0, "b=-4 deg"),
        (3, 8.0, "b=+8 deg"),
        (4, -8.0, "b=-8 deg"),
        (5, 15.0, "b=+15 deg"),
        (6, -15.0, "b=-15 deg"),
    ]

    clean = clean_channel_mask(wf[:, 0, :], max_zero_frac=0.08, mad_sigma=5.0) & clean_channel_mask(
        wf[:, off_beam, :], max_zero_frac=0.08, mad_sigma=5.0
    )
    if int(np.sum(clean)) < 30:
        clean = np.mean(np.isfinite(wf[:, 0, :]) & (wf[:, 0, :] > 0.0), axis=0) > 0.85

    # Latitude cut fractional excess & SNR
    off_ts = np.mean(wf[:, off_beam, clean], axis=1)
    lat_profile: List[Dict[str, Any]] = []
    for bid, bdeg, lbl in sorted(lat_map, key=lambda x: x[1]):
        b_ts = np.mean(wf[:, bid, clean], axis=1)
        exc = (b_ts - off_ts) / np.maximum(off_ts, 1e-12)
        exc_low = np.mean((wf[:, bid, clean & (FREQS_MHZ < 400.65)] - wf[:, off_beam, clean & (FREQS_MHZ < 400.65)]) / np.maximum(wf[:, off_beam, clean & (FREQS_MHZ < 400.65)], 1e-12)) if np.any(clean & (FREQS_MHZ < 400.65)) else 0.0
        exc_hi = np.mean((wf[:, bid, clean & (FREQS_MHZ >= 400.65)] - wf[:, off_beam, clean & (FREQS_MHZ >= 400.65)]) / np.maximum(wf[:, off_beam, clean & (FREQS_MHZ >= 400.65)], 1e-12)) if np.any(clean & (FREQS_MHZ >= 400.65)) else 0.0
        lat_profile.append({
            "beam_id": bid,
            "b_gal_deg": bdeg,
            "label": lbl,
            "mean_power": round(float(np.mean(b_ts)), 4),
            "mean_excess_vs_cold_off": round(float(np.mean(exc)), 4),
            "std_excess": round(float(np.std(exc)), 4),
            "excess_300_400": round(float(exc_low), 4),
            "excess_400_501": round(float(exc_hi), 4),
        })

    # Spectral index beta_synch on b=0 excess normalized by cold OFF spectral template
    edges = np.linspace(300.0, 501.5, 17)
    spec_freqs, spec_exc = [], []
    for bi in range(len(edges) - 1):
        m_b = clean & (FREQS_MHZ >= edges[bi]) & (FREQS_MHZ < edges[bi + 1])
        if int(np.sum(m_b)) < 3:
            continue
        p0 = float(np.median(np.mean(wf[:, 0, m_b], axis=1)))
        p9 = float(np.median(np.mean(wf[:, off_beam, m_b], axis=1)))
        # Excess normalized by cold OFF removes bandpass; slope reflects T_gal(nu)/T_sys(nu)
        spec_freqs.append(float(np.mean(FREQS_MHZ[m_b])))
        spec_exc.append((p0 - p9) / max(p9, 1e-12))

    slope_contrast = None
    valid_s = [k for k, v in enumerate(spec_exc) if v > 0.01]
    if len(valid_s) >= 4:
        fs = np.array([spec_freqs[k] for k in valid_s])
        vs = np.array([spec_exc[k] for k in valid_s])
        slope_contrast = round(float(np.polyfit(np.log(fs / 400.0), np.log(vs), 1)[0]), 4)

    # Parked zenith drift transit check
    drift_ts = np.mean(wf[:, drift_beam, clean] - wf[:, off_beam, clean], axis=1)
    drift_sm = gaussian_filter1d(drift_ts, sigma=max(1.0, min(4.0, len(drift_ts) / 8.0)))
    drift_pk_s = float(t_rel_s[int(np.argmax(drift_sm))]) if len(drift_sm) else 0.0

    # Sun sidelobe check
    sun_ts = np.mean(wf[:, sun_beam, clean], axis=1)
    sun_exc = float(np.mean((sun_ts - off_ts) / np.maximum(off_ts, 1e-12)))

    summary = {
        "run_dir": str(run_dir),
        "duration_s": round(float(t_rel_s[-1] - t_rel_s[0]) if len(t_rel_s) > 1 else 0.0, 2),
        "timing": timing,
        "clean_channels": int(np.sum(clean)),
        "pointing_samples": len(pointing),
        "latitude_profile": lat_profile,
        "b0_excess_spectral_slope": slope_contrast,
        "drift_peak_elapsed_s": round(drift_pk_s, 2),
        "sun_monitor_excess_vs_cold_off": round(sun_exc, 4),
        "subband_freqs_mhz": [round(x, 2) for x in spec_freqs],
        "subband_b0_excess_frac": [round(float(x), 4) for x in spec_exc],
    }
    (run_dir / "galactic_center_verification.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # 2x2 Diagnostic Figure
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9), layout="constrained")
    bs = [r["b_gal_deg"] for r in lat_profile]
    excs = [r["mean_excess_vs_cold_off"] for r in lat_profile]
    axes[0, 0].plot(bs, excs, "o-", color="#2563eb", lw=2.0)
    axes[0, 0].set_title("Galactic Center Cut (l=0 deg): (Beam - Cold OFF) / Cold OFF")
    axes[0, 0].set_xlabel("Galactic Latitude b (deg)")
    axes[0, 0].set_ylabel("Fractional Excess")
    axes[0, 0].grid(True, alpha=0.25)

    t_min = t_rel_s / 60.0
    axes[0, 1].plot(t_min, gaussian_filter1d(np.mean(wf[:, 0, clean] - wf[:, off_beam, clean], axis=1), sigma=2.0), label="b=0 tracked - OFF", color="#dc2626")
    axes[0, 1].plot(t_min, drift_sm, label="Beam 7 parked drift - OFF", color="#0f172a")
    axes[0, 1].set_title("Galactic Center Tracked vs Parked Zenith Drift Lightcurve")
    axes[0, 1].set_xlabel("Elapsed Time (min)")
    axes[0, 1].set_ylabel("Excess Power")
    axes[0, 1].grid(True, alpha=0.25)
    axes[0, 1].legend(fontsize=8.5)

    if spec_freqs:
        axes[1, 0].plot(spec_freqs, spec_exc, "o-", color="#16a34a", label=f"slope={slope_contrast}")
        axes[1, 0].legend(fontsize=8.5)
    axes[1, 0].set_title("Sgr A* (b=0) Excess Spectrum vs Frequency")
    axes[1, 0].set_xlabel("Frequency (MHz)")
    axes[1, 0].set_ylabel("(b=0 - Cold OFF) / Cold OFF")
    axes[1, 0].grid(True, alpha=0.25)

    axes[1, 1].plot(bs, [r["excess_300_400"] for r in lat_profile], "o-", color="#dc2626", label="300-400 MHz")
    axes[1, 1].plot(bs, [r["excess_400_501"] for r in lat_profile], "s--", color="#2563eb", label="400-501 MHz")
    axes[1, 1].set_title("Galactic Latitude Profile: Subband 0 vs Subband 1 Chromatic Width")
    axes[1, 1].set_xlabel("Galactic Latitude b (deg)")
    axes[1, 1].set_ylabel("Fractional Excess")
    axes[1, 1].grid(True, alpha=0.25)
    axes[1, 1].legend(fontsize=8.5)

    fig.savefig(run_dir / "galactic_center_verification.png", dpi=140)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
