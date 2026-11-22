#!/usr/bin/env python3
"""Generate publication-quality Phase 2 summary plots for tonight's 4 commissioning
runs (test11 48h phase drift, test12a GEO satellite survey, test12b SCL Airport &
Santiago horizon beams, and test10 60-min stress + power-only DMA savings)."""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from beam_io import (
    FREQS_MHZ,
    atomic_savefig,
    clean_channel_mask,
    load_dual_subband_beam_power,
    load_jsonl,
)
from calibrate_corr_epoch import (
    ANTS,
    CHARTS_ANTENNA_REF_LOCATION,
    CHARTSArray,
    POS4,
    geometric_visibility_phase,
    load_aligned_corr_pair,
)
from charts_utils.core.antenna_indices import charts_to_kotekan_raw
from charts_utils.core.gpu_frames import read_gpu_corr_frame_header

V2_DATA_ROOT = Path("/data/commissioning_20261119_v2")
PLOTS_DIR = V2_DATA_ROOT / "plots"
CORR_DRIFT_DIR = Path("/data/corr/corr_1121_drift")
GAINS_FILE = Path("/data/calibration/corr_1119/gains_corr_1119_0_4_32_36_fringehp_672x64.npy")


def array_factor_map(pos_xy: np.ndarray, l_src: float, m_src: float, freq_mhz: float, grid_n: int = 181):
    """Evaluate 4-antenna 2x2 array power response |AF(l, m)|^2 to a unit source at (l_src, m_src)."""
    ax = np.linspace(-1.0, 1.0, grid_n)
    ll, mm = np.meshgrid(ax, ax)
    lam = 299.792458 / freq_mhz
    k = 2.0 * np.pi / lam
    dl = l_src - ll
    dm = m_src - mm
    phasor = np.zeros_like(ll, dtype=np.complex128)
    for x, y in pos_xy:
        phasor += np.exp(1j * k * (x * dl + y * dm))
    af = np.abs(phasor / len(pos_xy)) ** 2
    af[(ll * ll + mm * mm) > 1.0] = np.nan
    return ll, mm, af


def plot_test11_drift(out_path: Path) -> None:
    drift_json = json.loads((V2_DATA_ROOT / "test11_24h_phase_drift_corr" / "drift_verification.json").read_text())
    f0s = sorted(glob.glob(str(CORR_DRIFT_DIR / "corr_freq_0_*.bin")))
    f1s = sorted(glob.glob(str(CORR_DRIFT_DIR / "corr_freq_1_*.bin")))
    gains_sub = np.load(GAINS_FILE)[:, ANTS]
    raws = [charts_to_kotekan_raw(a, 64) for a in ANTS]
    triu_i, triu_j = np.triu_indices(4)

    t_list, v_list = [], []
    for p0, p1 in zip(f0s, f1s):
        h0 = read_gpu_corr_frame_header(p0)
        v_pair = load_aligned_corr_pair(p0, p1, raws, triu_i, triu_j)
        if np.median(np.real(v_pair[:, 0])) < 1e5:
            continue
        t_list.append(h0.time0_fpga * 1e-6)
        v_list.append(v_pair)

    t_arr = np.asarray(t_list, dtype=np.float64)
    V_raw = np.transpose(np.asarray(v_list, dtype=np.complex128), (1, 2, 0))
    arr = CHARTSArray(*CHARTS_ANTENNA_REF_LOCATION, POS4)
    geom = -geometric_visibility_phase(t_arr, FREQS_MHZ, arr, triu_i, triu_j, src_name="sun")
    V_fs = V_raw * np.exp(-1j * geom)
    V_sub = (V_raw - np.mean(V_raw, axis=-1, keepdims=True)) * np.exp(-1j * geom)
    V_mean_fs = np.mean(V_fs, axis=-1)
    V_mean_sub = np.mean(V_sub, axis=-1)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(15, 6.2), layout="constrained")
    pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    pair_idx_map = {}
    for p_idx, (i, j) in enumerate(zip(triu_i, triu_j)):
        if i != j:
            pair_idx_map[(i, j)] = p_idx
    cols = ["#dc2626", "#2563eb", "#16a34a", "#9333ea", "#ea580c", "#0891b2"]

    m_plot = ((FREQS_MHZ >= 312.0) & (FREQS_MHZ <= 355.0)) | ((FREQS_MHZ >= 440.0) & (FREQS_MHZ <= 462.0))
    for c_i, (i, j) in enumerate(pairs):
        p_idx = pair_idx_map[(i, j)]
        gprod = gains_sub[:, i] * np.conj(gains_sub[:, j])
        res_ph = np.rad2deg(np.angle(V_mean_sub[:, p_idx] / gprod))
        lbl = f"{ANTS[i]}-{ANTS[j]}"
        res_m = np.where(m_plot, res_ph, np.nan)
        ax0.plot(FREQS_MHZ, res_m, color=cols[c_i], lw=1.2, label=lbl, alpha=0.9)

    ax0.axvspan(315.0, 355.0, color="#dbeafe", alpha=0.4, label="Clean Low (315-355)")
    ax0.axvspan(440.0, 462.0, color="#dcfce7", alpha=0.4, label="Clean High (440-462)")
    ax0.axhline(0.0, color="k", ls="--", lw=0.9)
    ax0.set_ylim(-100, 100)
    ax0.set_xlim(310, 468)
    ax0.set_title("48h Post-Epoch Residual Phase Spectrum (corr_1121 vs corr_1119)")
    ax0.set_xlabel("Frequency (MHz)")
    ax0.set_ylabel("Residual Phase (deg)")
    ax0.grid(True, alpha=0.25)
    ax0.legend(fontsize=8.0, ncol=2, loc="upper right")

    bl_labels = [f"{ANTS[i]}-{ANTS[j]}" for i, j in pairs]
    x = np.arange(len(bl_labels))
    w = 0.25
    u_low_d = [drift_json["residuals_unfilt_315_355"][b]["delay_err_ns"] for b in bl_labels]
    h_low_d = [drift_json["residuals_hp_315_355"][b]["delay_err_ns"] for b in bl_labels]
    h_hi_d = [drift_json["residuals_hp_440_462"][b]["delay_err_ns"] for b in bl_labels]
    u_low_r = [drift_json["residuals_unfilt_315_355"][b]["circular_R"] for b in bl_labels]
    h_low_r = [drift_json["residuals_hp_315_355"][b]["circular_R"] for b in bl_labels]

    ax1.bar(x - w, u_low_d, width=w, color="#2563eb", label="315-355 MHz Unfilt")
    ax1.bar(x, h_low_d, width=w, color="#16a34a", label="315-355 MHz Mean-Sub")
    ax1.bar(x + w, h_hi_d, width=w, color="#ea580c", label="440-462 MHz Mean-Sub")
    ax1.axhline(0.0, color="k", ls="--", lw=0.9)
    ax1.axhspan(-2.0, 2.0, color="#f0fdf4", alpha=0.6, label="+-2 ns target")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{b}\nR={u_low_r[k]:.2f}/{h_low_r[k]:.2f}" for k, b in enumerate(bl_labels)], fontsize=8.5)
    ax1.set_title("48h + Post-Reboot Residual Delay Error & Circular Coherence R")
    ax1.set_ylabel("Delay Error (ns)")
    ax1.set_ylim(-4.5, 11.0)
    ax1.grid(True, axis="y", alpha=0.25)
    ax1.legend(fontsize=8.0, loc="upper left")

    fig.suptitle("Test 11: 48-Hour Clock & Cable Phase Drift Audit (120 Frames, Sun El 39.5°-43.7°)", fontsize=13, fontweight="bold")
    atomic_savefig(fig, out_path, dpi=150)
    plt.close(fig)


def plot_test12a_geo(out_path: Path) -> None:
    run_dir = V2_DATA_ROOT / "test12a_geo_satellites_survey"
    wf, _, _, _ = load_dual_subband_beam_power(run_dir, n_beams=10)
    vjson = json.loads((run_dir / "geo_satellites_verification.json").read_text())
    pts = vjson["latest_pointing_targets"]
    med = np.median(wf, axis=0)

    labels = [
        "B0: MUOS-1 (El42° W)",
        "B1: MUOS-3 (El22° E)",
        "B2: GOES-19 (El51° N)",
        "B3: GOES-18 (El11° W)",
        "B4: UFO-10 (El31° E)",
        "B5: SKYNET-5C (El25° E)",
        "B6: USA-115 (El59° NW)",
        "B7: INMARSAT-3F5 (El57° NE)",
        "B8: OFF-South (El35°)",
        "B9: Zenith (El90°)",
    ]
    colors = ["#dc2626", "#2563eb", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#ca8a04", "#db2777", "#0f172a", "#64748b"]

    fig = plt.figure(figsize=(15.5, 10.5), layout="constrained")
    gs = gridspec.GridSpec(2, 2, figure=fig)

    ax00 = fig.add_subplot(gs[0, 0])
    m00 = (FREQS_MHZ >= 355.0) & (FREQS_MHZ <= 386.0)
    for b in (1, 5, 7, 0, 4, 8, 9):
        ax00.plot(FREQS_MHZ[m00], med[b, m00], color=colors[b], lw=1.3, label=labels[b])
    ax00.axvline(380.0, color="k", ls=":", lw=1.0, label="380 MHz")
    ax00.set_title("MUOS WCDMA Downlink: MUOS-3 (El 22° E) at +9.7 dB vs Zenith")
    ax00.set_xlabel("Frequency (MHz)")
    ax00.set_ylabel("Median Beam Power")
    ax00.set_yscale("log")
    ax00.grid(True, alpha=0.25)
    ax00.legend(fontsize=7.5, ncol=2)

    ax01 = fig.add_subplot(gs[0, 1])
    m01 = (FREQS_MHZ >= 300.0) & (FREQS_MHZ <= 410.0)
    for b in (2, 6, 4, 5, 3, 7):
        rdb = 10.0 * np.log10(np.maximum(med[b, m01], 1e-12) / np.maximum(med[8, m01], 1e-12))
        ax01.plot(FREQS_MHZ[m01], rdb, color=colors[b], lw=1.1, label=labels[b], alpha=0.9)
    ax01.axhline(0.0, color="k", ls="--", lw=0.8)
    ax01.axvspan(356.0, 384.0, color="#fef08a", alpha=0.25, label="MUOS 356-384")
    ax01.axvspan(399.0, 405.0, color="#bfdbfe", alpha=0.35, label="DCS 399-405 (GOES-19)")
    ax01.set_title("300-410 MHz Contrast vs OFF-South (GOES-19 Leads 318.3 & DCS)")
    ax01.set_xlabel("Frequency (MHz)")
    ax01.set_ylabel("Excess vs OFF-South (dB)")
    ax01.grid(True, alpha=0.25)
    ax01.legend(fontsize=7.5, ncol=2)

    sub_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 0], hspace=0.32)
    ax10a = fig.add_subplot(sub_gs[0])
    m10a = (FREQS_MHZ >= 413.0) & (FREQS_MHZ <= 435.0)
    for b in (0, 6, 4, 1, 8, 9):
        ax10a.plot(FREQS_MHZ[m10a], med[b, m10a], color=colors[b], lw=1.2, label=labels[b])
    ax10a.set_title("413-435 MHz: MUOS-1 (415.5 +15.7 dB) & USA-115 (427.8 +8.0 dB)")
    ax10a.set_ylabel("Median Power")
    ax10a.set_yscale("log")
    ax10a.set_ylim(0.5, 65.0)
    ax10a.grid(True, alpha=0.25)
    ax10a.legend(fontsize=7.0, ncol=3, loc="lower left")

    ax10b = fig.add_subplot(sub_gs[1])
    m10b = (FREQS_MHZ >= 464.0) & (FREQS_MHZ <= 486.0)
    for b in (2, 3, 4, 7, 8, 9):
        ax10b.plot(FREQS_MHZ[m10b], med[b, m10b], color=colors[b], lw=1.2, label=labels[b])
    ax10b.axvline(468.83, color="#16a34a", ls="--", lw=1.0, label="GOES 468.83")
    ax10b.axvline(472.5, color="#dc2626", ls=":", lw=1.0, label="472.5 MHz")
    ax10b.set_title("464-486 MHz: GOES-19 / GOES-18 & 472.5 MHz Peak")
    ax10b.set_xlabel("Frequency (MHz)")
    ax10b.set_ylabel("Median Power")
    ax10b.set_yscale("log")
    ax10b.grid(True, alpha=0.25)
    ax10b.legend(fontsize=7.0, ncol=3)

    ax11 = fig.add_subplot(gs[1, 1])
    mu3 = pts[1]
    ll, mm, af = array_factor_map(POS4[:, :2], mu3["l0"], mu3["m0"], 377.4, grid_n=181)
    cf = ax11.contourf(ll, mm, af, levels=np.linspace(0.0, 1.0, 11), cmap="YlOrRd", alpha=0.42)
    plt.colorbar(cf, ax=ax11, label="2x2 Array Factor from MUOS-3 @ 377.4 MHz")
    theta = np.linspace(0, 2 * np.pi, 360)
    ax11.plot(np.cos(theta), np.sin(theta), "k-", lw=1.0)
    ax11.axhline(0.0, color="gray", ls=":", lw=0.7)
    ax11.axvline(0.0, color="gray", ls=":", lw=0.7)
    m_mu = (FREQS_MHZ >= 375.5) & (FREQS_MHZ <= 379.5)
    mu_pwr = np.mean(med[:, m_mu], axis=1)
    sc = ax11.scatter(
        [p["l0"] for p in pts],
        [p["m0"] for p in pts],
        c=mu_pwr,
        s=110,
        cmap="viridis",
        edgecolors="black",
        zorder=5,
    )
    plt.colorbar(sc, ax=ax11, label="Measured 375.5-379.5 MHz Power")
    offsets = [
        (-0.08, 0.14),   # MUOS-1
        (-0.36, -0.22),  # MUOS-3
        (0.06, 0.08),    # GOES-19
        (0.06, 0.10),    # GOES-18
        (-0.42, 0.24),   # UFO-10
        (-0.56, 0.02),   # SKYNET-5C
        (-0.04, -0.22),  # USA-115
        (-0.14, 0.24),   # INMARSAT
        (0.08, 0.05),    # OFF-S
        (0.06, 0.06),    # Zenith
    ]
    short = ["MUOS-1", "MUOS-3", "GOES-19", "GOES-18", "UFO-10", "SKYNET-5C", "USA-115", "INMARSAT", "OFF-S", "Zenith"]
    for idx, p in enumerate(pts):
        dx, dy = offsets[idx]
        ax11.annotate(
            f"{short[idx]}\n({mu_pwr[idx]:.1f})",
            (p["l0"], p["m0"]),
            xytext=(p["l0"] + dx, p["m0"] + dy),
            fontsize=7.5,
            fontweight="bold" if idx == 1 else "normal",
            arrowprops={"arrowstyle": "-", "lw": 0.6, "color": "#475569"},
        )
    ax11.set_xlim(-1.08, 1.08)
    ax11.set_ylim(-1.08, 1.08)
    ax11.set_aspect("equal")
    ax11.set_title("Sky (l, m) + 2x2 Grating Lobe Coupling Explain SKYNET-5C & INMARSAT")
    ax11.set_xlabel("Direction Cosine l (East +)")
    ax11.set_ylabel("Direction Cosine m (North +)")

    fig.suptitle("Test 12A: Multi-GEO Satellite UHF Survey (30 Min Unblanked, 1755 Frames)", fontsize=13.5, fontweight="bold")
    atomic_savefig(fig, out_path, dpi=150)
    plt.close(fig)


def plot_test12b_horizon(out_path: Path) -> None:
    run_dir = V2_DATA_ROOT / "test12b_airport_santiago_horizon"
    wf, seq, _, _ = load_dual_subband_beam_power(run_dir, n_beams=10)
    med = np.median(wf, axis=0)
    quiet_mean = np.mean(med[[5, 6, 7], :], axis=0)
    cmask = clean_channel_mask(wf[:, 9, :])
    t_min = (seq - seq[0]).astype(np.float64) * (10.0 / 3.0) * 1e-6 / 60.0

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.2), layout="constrained")
    ax00 = axes[0, 0]
    specs = [
        (0, "B0: SCL Airport (0.5°)", "#dc2626", 1.2),
        (2, "B2: Santiago Centro (0.5°)", "#ea580c", 1.1),
        (3, "B3: San Cristobal (1.0° Cal)", "#7c3aed", 1.1),
        (4, "B4: Santiago El 10°", "#2563eb", 1.0),
        (8, "B8: San Cristobal Uncal", "#c084fc", 0.9),
        (9, "B9: Zenith (90°)", "#0f172a", 1.0),
    ]
    for b, lbl, col, lw in specs:
        ax00.plot(FREQS_MHZ, med[b], color=col, lw=lw, label=lbl, alpha=0.9)
    ax00.plot(FREQS_MHZ, quiet_mean, color="#16a34a", lw=1.2, ls="--", label="Mean(Quiet W/S/NW 0.5°)")
    ax00.set_title("300-501.3 MHz Urban/Airport Horizon vs Quiet Controls & Zenith")
    ax00.set_xlabel("Frequency (MHz)")
    ax00.set_ylabel("Median Beam Power")
    ax00.set_yscale("log")
    ax00.grid(True, alpha=0.25)
    ax00.legend(fontsize=7.5, ncol=2)

    ax01 = axes[0, 1]
    nb_bands = [
        ("361.2-364.2\nMUOS/SCL", (FREQS_MHZ >= 361.2) & (FREQS_MHZ <= 364.2)),
        ("414.3-414.9\nSCL CW", (FREQS_MHZ >= 414.3) & (FREQS_MHZ <= 414.9)),
        ("415.2-416.1\nStgo TETRA", (FREQS_MHZ >= 415.2) & (FREQS_MHZ <= 416.1)),
        ("429.0\nNarrowband", (FREQS_MHZ >= 428.8) & (FREQS_MHZ <= 429.2)),
        ("470.1\nStgo DTV", (FREQS_MHZ >= 469.9) & (FREQS_MHZ <= 470.3)),
    ]
    bar_beams = [
        (0, "SCL Airport 0.5°", "#dc2626"),
        (1, "SCL App 10°", "#f87171"),
        (2, "Stgo Centro 0.5°", "#ea580c"),
        (3, "San Cristobal 1°", "#7c3aed"),
        (4, "Stgo El 10°", "#2563eb"),
        (9, "Zenith 90°", "#475569"),
    ]
    x = np.arange(len(nb_bands))
    w = 0.13
    for k, (bid, blbl, bcol) in enumerate(bar_beams):
        vals = []
        for _, m in nb_bands:
            p_b = float(np.median(med[bid, m]))
            p_q = float(np.median(quiet_mean[m]))
            vals.append(10.0 * np.log10(max(p_b, 1e-12) / max(p_q, 1e-12)))
        ax01.bar(x + (k - 2.5) * w, vals, width=w, color=bcol, label=blbl)
    ax01.axhline(0.0, color="#16a34a", ls="--", lw=1.0, label="Quiet Horizon Mean (0 dB)")
    ax01.set_xticks(x)
    ax01.set_xticklabels([b[0] for b in nb_bands], fontsize=8.5)
    ax01.set_title("Target Narrowband Excess Over Quiet Horizon Mean (dB)")
    ax01.set_ylabel("Excess Power (dB)")
    ax01.grid(True, axis="y", alpha=0.25)
    ax01.legend(fontsize=7.5, ncol=2)

    ax10 = axes[1, 0]
    ax10.plot(FREQS_MHZ, 10.0 * np.log10(np.maximum(med[0], 1e-12) / np.maximum(med[1], 1e-12)), color="#dc2626", lw=1.2, label="SCL El 0.5° / SCL El 10°")
    ax10.plot(FREQS_MHZ, 10.0 * np.log10(np.maximum(med[2], 1e-12) / np.maximum(quiet_mean, 1e-12)), color="#ea580c", lw=1.1, label="Stgo Centro 0.5° / Quiet Mean", alpha=0.85)
    ax10.plot(FREQS_MHZ, 10.0 * np.log10(np.maximum(med[3], 1e-12) / np.maximum(med[8], 1e-12)), color="#7c3aed", lw=1.1, label="San Cristobal Cal / Uncal (+4.1..+11.9 dB)", alpha=0.85)
    ax10.axhline(0.0, color="k", ls="--", lw=0.8)
    ax10.set_title("Elevation Selectivity & Calibration Coherent Gain vs Frequency")
    ax10.set_xlabel("Frequency (MHz)")
    ax10.set_ylabel("Power Ratio (dB)")
    ax10.grid(True, alpha=0.25)
    ax10.legend(fontsize=8.0)

    ax11 = axes[1, 1]
    for b, lbl, col in [
        (0, "B0: SCL Airport (0.5°)", "#dc2626"),
        (2, "B2: Santiago Centro (0.5°)", "#ea580c"),
        (3, "B3: San Cristobal (1.0° Cal)", "#7c3aed"),
        (5, "B5: Quiet West (0.5°)", "#16a34a"),
        (9, "B9: Zenith (90°)", "#0f172a"),
    ]:
        ts = np.median(wf[:, b, cmask], axis=1)
        ax11.plot(t_min, ts, color=col, lw=1.0, label=f"{lbl} (CV={np.std(ts)/np.mean(ts)*100:.2f}%)", alpha=0.9)
    ax11.set_title("Raw Clean-Band Continuum Time Series (Setting Sun El 31.9°→25.8°)")
    ax11.set_xlabel("Elapsed Time (min)")
    ax11.set_ylabel("Median Clean Power (Raw Scale)")
    ax11.grid(True, alpha=0.25)
    ax11.legend(fontsize=7.8, loc="lower right")

    fig.suptitle("Test 12B: SCL Airport, Santiago Centro & Cerro San Cristobal Horizon Study (30 Min)", fontsize=13.5, fontweight="bold")
    atomic_savefig(fig, out_path, dpi=150)
    plt.close(fig)


def plot_test10_stress(out_path: Path) -> None:
    d10 = V2_DATA_ROOT / "test10_production_stress_10beams"
    d12b = V2_DATA_ROOT / "test12b_airport_santiago_horizon"
    sjson = json.loads((d10 / "stress_verification.json").read_text())
    t10 = load_jsonl(d10 / "telemetry_log.jsonl")
    t12b = load_jsonl(d12b / "telemetry_log.jsonl")

    x10 = (np.array([r["unix"] for r in t10]) - t10[0]["unix"]) / 60.0
    g0_10 = np.array([r["gpu_profile"].get("gpu_0", {}).get("total_ms", np.nan) for r in t10])
    x12b = (np.array([r["unix"] for r in t12b]) - t12b[0]["unix"]) / 60.0
    g0_12b = np.array([r["gpu_profile"].get("gpu_0", {}).get("total_ms", np.nan) for r in t12b])

    def cmd_means(rows):
        out = {}
        for r in rows:
            for k, v in r.get("gpu_profile", {}).get("gpu_0", {}).get("commands_ms", {}).items():
                out.setdefault(k, []).append(v)
        return {k: float(np.mean(v)) for k, v in out.items()}

    c10 = cmd_means(t10)
    c12b = cmd_means(t12b)

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.0), layout="constrained")
    ax00 = axes[0, 0]
    ax00.plot(x10, g0_10, color="#2563eb", lw=1.0, label=f"Test 10 (copy_formed=True): mean={np.nanmean(g0_10):.1f} ms")
    ax00.plot(x12b, g0_12b, color="#16a34a", lw=1.1, label=f"Test 12B (copy_formed=False): mean={np.nanmean(g0_12b):.1f} ms")
    ax00.axhline(51.2, color="#dc2626", ls="--", lw=1.3, label="51.2 ms Real-Time Frame Budget")
    ax00.axhline(sjson["gpu_0_active_work_ms"]["mean"], color="#0284c7", ls=":", lw=1.2, label=f"Test 10 Active GPU Work: {sjson['gpu_0_active_work_ms']['mean']:.1f} ms")
    ax00.set_title("GPU Stage Latency: Full Complex Host Copy vs Power-Only")
    ax00.set_xlabel("Elapsed Time (min)")
    ax00.set_ylabel("GPU 0 Stage Total (ms)")
    ax00.grid(True, alpha=0.25)
    ax00.legend(fontsize=7.8)

    ax01 = axes[0, 1]
    cmd_keys = [
        ("copy_in:input: voltage_0", "Copy In (voltages)"),
        ("kernel:cudaAntennaMaskCommand", "Antenna Mask"),
        ("kernel:cudaRfiMitigationCommand", "RFI Mitigation"),
        ("kernel:cudaDirectBeamTrackerCommand", "10-Beam Tracker"),
        ("copy_out:output: formed_beams_0", "Copy Out (formed_beams)"),
        ("copy_out:output: beam_power_0", "Copy Out (beam_power)"),
    ]
    yk = np.arange(len(cmd_keys))
    h = 0.35
    v10 = [c10.get(k, 0.0) for k, _ in cmd_keys]
    v12b = [c12b.get(k, 0.0) for k, _ in cmd_keys]
    ax01.barh(yk + h / 2, v10, height=h, color="#2563eb", label="Test 10 (blank=T, copy_formed=T)")
    ax01.barh(yk - h / 2, v12b, height=h, color="#16a34a", label="Test 12B (monitor=T, copy_formed=F)")
    for idx in range(len(cmd_keys)):
        ax01.text(v10[idx] + 0.12, yk[idx] + h / 2, f"{v10[idx]:.2f}", va="center", fontsize=7.5, color="#1e3a8a")
        ax01.text(v12b[idx] + 0.12, yk[idx] - h / 2, f"{v12b[idx]:.2f}", va="center", fontsize=7.5, color="#14532d")
    ax01.set_yticks(yk)
    ax01.set_yticklabels([lbl for _, lbl in cmd_keys], fontsize=8.5)
    ax01.set_xlim(0, 9.2)
    ax01.set_title("Per-Command GPU Breakdown: Eliminating 7.5 ms DMA Copy")
    ax01.set_xlabel("Mean Execution Time (ms)")
    ax01.grid(True, axis="x", alpha=0.25)
    ax01.legend(fontsize=7.8, loc="upper right")

    ax10 = axes[1, 0]
    bufs = sjson["buffer_max_occupancy_pct"]
    b_names = sorted(bufs.keys())
    yb = np.arange(len(b_names))
    b_vals = [bufs[n] for n in b_names]
    ax10.barh(yb, b_vals, color="#0284c7", height=0.55)
    ax10.set_yticks(yb)
    ax10.set_yticklabels(b_names, fontsize=8.5)
    ax10.set_xlim(0, 100)
    ax10.axvline(5.0, color="#16a34a", ls="--", lw=1.0, label="5% (1 / 20 frames)")
    ax10.set_title("Ring-Buffer High-Water Occupancy Over 60 Min (8.63B Packets, 0 Lost)")
    ax10.set_xlabel("Max Occupancy (%)")
    ax10.grid(True, axis="x", alpha=0.25)
    ax10.legend(fontsize=8.0)

    ax11 = axes[1, 1]
    temp10 = [r.get("nvidia_smi", {}).get("temp_c", np.nan) for r in t10]
    pwr10 = [r.get("nvidia_smi", {}).get("power_w", np.nan) for r in t10]
    pwr12b = [r.get("nvidia_smi", {}).get("power_w", np.nan) for r in t12b]
    ax11.plot(x10, pwr10, color="#9333ea", lw=1.0, label=f"Test 10 Power: {np.nanmean(pwr10):.1f} W (58.9% util)")
    ax11.plot(x12b, pwr12b, color="#16a34a", lw=1.0, label=f"Test 12B Power: {np.nanmean(pwr12b):.1f} W (32.6% util)")
    ax11.set_xlabel("Elapsed Time (min)")
    ax11.set_ylabel("RTX 5090 Power (W)")
    ax11.set_ylim(140, 210)
    ax11.grid(True, alpha=0.25)
    ax11_t = ax11.twinx()
    ax11_t.plot(x10, temp10, color="#dc2626", lw=1.1, alpha=0.8, label=f"Test 10 Temp: {np.nanmean(temp10):.1f}°C")
    ax11_t.set_ylabel("GPU Temperature (°C)", color="#dc2626")
    ax11_t.set_ylim(35, 65)
    lines1, labs1 = ax11.get_legend_handles_labels()
    lines2, labs2 = ax11_t.get_legend_handles_labels()
    ax11.legend(lines1 + lines2, labs1 + labs2, fontsize=7.8, loc="upper right")
    ax11.set_title("RTX 5090 Power & Thermal Stability (-32 W in Power-Only Mode)")

    fig.suptitle("Test 10 & Power-Only Optimization: 60-Min 10-Beam Production Stress Audit", fontsize=13.5, fontweight="bold")
    atomic_savefig(fig, out_path, dpi=150)
    plt.close(fig)


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    p06 = PLOTS_DIR / "06_test11_48h_phase_delay_drift.png"
    p07 = PLOTS_DIR / "07_test12a_geo_satellites_survey.png"
    p08 = PLOTS_DIR / "08_test12b_airport_santiago_horizon.png"
    p09 = PLOTS_DIR / "09_test10_stress_and_dma_savings.png"
    plot_test11_drift(p06)
    print(f"Saved {p06}")
    plot_test12a_geo(p07)
    print(f"Saved {p07}")
    plot_test12b_horizon(p08)
    print(f"Saved {p08}")
    plot_test10_stress(p09)
    print(f"Saved {p09}")


if __name__ == "__main__":
    main()
