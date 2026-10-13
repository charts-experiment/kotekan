#!/usr/bin/env python3
"""
CHARTS Correlator Multi-Baseline Waterfall & Dynamic Spectrum Visualizer
========================================================================
Generates publication-quality high-resolution PNG waterfall dashboards from
Kotekan cudaCorrelatorAstron binary dumps (corr_*.bin):
  1. Full-Array Mean Auto-Power Waterfall P_auto(t, f)
     Time vs Frequency showing bandpass profile, transient sweeps (FRBs, pulsars),
     and narrowband/persistent RFI lines across the entire 1-minute window.
  2. Short Baseline Fringe Waterfall (e.g. Ant 0 - Ant 1, 0.6 m East-West)
     Real visibility Re(V_01(t, f)) showing interferometric fringes and mutual coupling.
  3. Diagonal Baseline Fringe Waterfall (e.g. Ant 0 - Ant 9, 0.85 m diagonal)
     Re(V_09(t, f)) displaying 2D spatial fringe modulation.
  4. Long Baseline Fringe Waterfall (e.g. Ant 0 - Ant 63, 5.94 m aperture diagonal)
     Re(V_0,63(t, f)) capturing maximum array spatial resolution.
  5. Off-Diagonal Coherence Dynamic Spectrum
     Ratio of coherent cross-power to auto-power across (t, f), highlighting coherent signals.
  6. Active Channel & Quiet Channel 64x64 Correlation Matrix Snapshots
     Magnitude |V_ij| and Phase arg(V_ij) at key moments (e.g. peak event vs baseline).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

# Setup paths
_test_charts_dir = Path(__file__).resolve().parent
_kotekan_root = _test_charts_dir.parent
if str(_test_charts_dir) not in sys.path:
    sys.path.insert(0, str(_test_charts_dir))

from constants import (
    CHARTS_CHANNEL_WIDTH_MHZ,
    DEFAULT_FREQUENCY_START_MHZ,
    DEFAULT_SPACING_M,
)
from inspect_correlator_dump import load_astron_correlator_dump


def baseline_index(receiver_y: int, receiver_x: int) -> int:
    """Computes lower-triangular baseline index for (receiver_y, receiver_x)."""
    return receiver_y * (receiver_y + 1) // 2 + receiver_x


def extract_baseline_visibilities(
    c_data: np.ndarray,
    ant_i: int,
    ant_j: int,
    polarizations: int = 2,
) -> np.ndarray:
    """
    Extracts visibility V_ij(f) of shape (num_channels,) from packed correlator data
    c_data of shape (num_channels, num_baselines, pol_y, pol_x).
    """
    rec_a, pol_a = divmod(ant_i, polarizations)
    rec_b, pol_b = divmod(ant_j, polarizations)
    if rec_b <= rec_a:
        b_idx = baseline_index(rec_a, rec_b)
        return c_data[:, b_idx, pol_a, pol_b]
    else:
        b_idx = baseline_index(rec_b, rec_a)
        return np.conj(c_data[:, b_idx, pol_b, pol_a])


def extract_all_auto_powers(
    c_data: np.ndarray,
    num_elements: int = 64,
    polarizations: int = 2,
) -> np.ndarray:
    """
    Extracts auto-power for all antennas.
    Returns array of shape (num_channels, num_elements) real float64.
    """
    num_channels = c_data.shape[0]
    autos = np.zeros((num_channels, num_elements), dtype=np.float64)
    for ant in range(num_elements):
        rec, pol = divmod(ant, polarizations)
        b_idx = baseline_index(rec, rec)
        autos[:, ant] = np.real(c_data[:, b_idx, pol, pol])
    return autos


def load_correlator_raw_packed(
    bin_path: Path,
    num_elements: int = 64,
    num_channels: int = 672,
    polarizations: int = 2,
) -> np.ndarray:
    """
    Reads correlator binary dump and returns c_data complex array
    of shape (num_channels, num_baselines, polarizations, polarizations).
    """
    raw = np.fromfile(bin_path, dtype=np.uint8)
    if raw.size < 4:
        raise ValueError(f"File {bin_path} is smaller than 4-byte header.")
    metadata_size = int(np.frombuffer(raw[:4].tobytes(), dtype="<u4", count=1)[0])
    payload_offset = 4 + metadata_size

    receiver_count = num_elements // polarizations
    num_baselines = receiver_count * (receiver_count + 1) // 2
    expected_ints = num_channels * num_baselines * polarizations * polarizations * 2
    expected_bytes = expected_ints * 4

    payload_size = raw.size - payload_offset
    if payload_size < expected_bytes:
        raise ValueError(
            f"File {bin_path} payload size {payload_size} < expected {expected_bytes}"
        )

    raw_ints = np.frombuffer(
        raw, dtype="<i4", count=expected_ints, offset=payload_offset
    )
    packed = raw_ints.reshape(
        num_channels, num_baselines, polarizations, polarizations, 2
    )
    c_data = packed[..., 0].astype(np.float64) + 1j * packed[..., 1].astype(np.float64)
    return c_data


def generate_correlator_waterfalls(
    corr_dir: Path,
    output_png: Path,
    num_elements: int = 64,
    num_channels: int = 672,
    duration_s: float = 60.0,
    max_time_samples: int = 200,
    window_label: str = "15:00 UTC",
):
    """
    Generates a 6-panel high-resolution waterfall and matrix dashboard.
    """
    bin_files = sorted(corr_dir.glob("*.bin"))
    if not bin_files:
        raise FileNotFoundError(f"No correlator .bin files found in {corr_dir}")

    total_files = len(bin_files)
    target_samples = min(total_files, max_time_samples)
    indices = np.linspace(0, total_files - 1, target_samples).astype(int)
    sampled_files = [bin_files[i] for i in indices]

    time_s = np.linspace(0.0, duration_s, target_samples)
    freq_mhz = (
        DEFAULT_FREQUENCY_START_MHZ + np.arange(num_channels) * CHARTS_CHANNEL_WIDTH_MHZ
    )

    print(
        f"Extracting correlator waterfalls from {target_samples} sampled frames "
        f"across {total_files} dumps in {corr_dir.name} ..."
    )

    # Baselines to track:
    # 1. Ant 0 - Ant 1 (0.6m East-West)
    # 2. Ant 0 - Ant 8 (0.6m North-South) or Ant 0 - Ant 9 (0.85m diagonal)
    # 3. Ant 0 - Ant 63 (5.94m diagonal across entire array)
    bl_short = (0, 1)
    bl_diag = (0, 9)
    bl_long = (0, min(63, num_elements - 1))

    # Preallocate cubes: (num_times, num_channels)
    auto_pwr_mean = np.zeros((target_samples, num_channels), dtype=np.float64)
    vis_short = np.zeros((target_samples, num_channels), dtype=np.complex128)
    vis_diag = np.zeros((target_samples, num_channels), dtype=np.complex128)
    vis_long = np.zeros((target_samples, num_channels), dtype=np.complex128)
    coherence_cube = np.zeros((target_samples, num_channels), dtype=np.float64)

    for t_idx, fpath in enumerate(sampled_files):
        c_data = load_correlator_raw_packed(fpath, num_elements, num_channels)
        autos = extract_all_auto_powers(
            c_data, num_elements
        )  # (num_channels, num_elements)
        mean_auto = np.mean(autos, axis=1)  # (num_channels,)
        auto_pwr_mean[t_idx] = mean_auto

        vis_short[t_idx] = extract_baseline_visibilities(
            c_data, bl_short[0], bl_short[1]
        )
        vis_diag[t_idx] = extract_baseline_visibilities(c_data, bl_diag[0], bl_diag[1])
        vis_long[t_idx] = extract_baseline_visibilities(c_data, bl_long[0], bl_long[1])

        # Coherence estimate: ratio of selected cross-power to auto-power
        cross_mag = (
            np.abs(vis_short[t_idx]) + np.abs(vis_diag[t_idx]) + np.abs(vis_long[t_idx])
        )
        coherence_cube[t_idx] = cross_mag / (3.0 * np.maximum(1e-6, mean_auto))

    # Identify peak transient frame and baseline frame for 64x64 matrix snapshots
    total_power_per_frame = np.sum(auto_pwr_mean, axis=1)
    peak_t_idx = int(np.argmax(total_power_per_frame))
    base_t_idx = 0 if peak_t_idx != 0 else (target_samples - 1)

    peak_file = sampled_files[peak_t_idx]
    base_file = sampled_files[base_t_idx]

    # Find active channel (channel with highest variation or peak power)
    var_per_channel = np.var(auto_pwr_mean, axis=0)
    active_ch = int(np.argmax(var_per_channel))
    # Quiet channel: 25th percentile of variance
    quiet_ch = int(np.argsort(var_per_channel)[len(var_per_channel) // 4])

    print(
        f"Loading full 64x64 matrices for peak frame {peak_t_idx} (t={time_s[peak_t_idx]:.2f}s) and baseline frame {base_t_idx} ..."
    )
    peak_matrix = load_astron_correlator_dump(peak_file, num_elements, num_channels)
    base_matrix = load_astron_correlator_dump(base_file, num_elements, num_channels)

    # -------------------------------------------------------------------------
    # Plotting: Publication-Quality Dark Multi-Panel Dashboard
    # -------------------------------------------------------------------------
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(22, 14), facecolor="#0a0a1a")

    gs = GridSpec(
        nrows=3,
        ncols=3,
        width_ratios=[1.2, 1.2, 0.9],
        height_ratios=[1.0, 1.0, 1.0],
        wspace=0.22,
        hspace=0.32,
        left=0.05,
        right=0.96,
        top=0.93,
        bottom=0.06,
    )

    extent = [time_s[0], time_s[-1], freq_mhz[0], freq_mhz[-1]]

    # 1. Array Mean Auto-Power Waterfall
    ax1 = fig.add_subplot(gs[0, 0])
    auto_db = 10.0 * np.log10(np.maximum(1e-3, auto_pwr_mean.T))  # (freq, time)
    vmin_auto = float(np.percentile(auto_db, 2))
    vmax_auto = float(np.percentile(auto_db, 99.5))
    im1 = ax1.imshow(
        auto_db,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="inferno",
        vmin=vmin_auto,
        vmax=vmax_auto,
    )
    ax1.set_title(
        "1. Array Mean Auto-Power P_auto(t, f) [dB]\n(All 64 Antennas Averaged)",
        fontsize=11,
        fontweight="bold",
        color="#FFD700",
    )
    ax1.set_ylabel("Frequency [MHz]", fontsize=10, color="#C9D1D9")
    ax1.axvline(
        time_s[peak_t_idx],
        color="#00FFCC",
        linestyle="--",
        alpha=0.75,
        label=f"Peak: t={time_s[peak_t_idx]:.1f}s",
    )
    ax1.legend(loc="upper right", fontsize=8, facecolor="#161B22", edgecolor="#30363D")
    cb1 = plt.colorbar(im1, ax=ax1, pad=0.02, shrink=0.9)
    cb1.set_label("dB", color="#C9D1D9", fontsize=9)

    # 2. Short Baseline Fringe Waterfall (Ant 0 - Ant 1, 0.6m East-West)
    ax2 = fig.add_subplot(gs[0, 1])
    re_short = np.real(vis_short).T
    sigma_s = float(np.std(re_short))
    mean_s = float(np.mean(re_short))
    im2 = ax2.imshow(
        re_short,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="twilight",
        vmin=mean_s - 2.5 * sigma_s,
        vmax=mean_s + 2.5 * sigma_s,
    )
    ax2.set_title(
        f"2. Short Baseline Fringe Waterfall: Ant {bl_short[0]} - Ant {bl_short[1]}\n(0.6 m East-West, Re(V_01))",
        fontsize=11,
        fontweight="bold",
        color="#00FFCC",
    )
    cb2 = plt.colorbar(im2, ax=ax2, pad=0.02, shrink=0.9)
    cb2.set_label("Linear", color="#C9D1D9", fontsize=9)

    # 3. Diagonal Baseline Fringe Waterfall (Ant 0 - Ant 9, 0.85m diagonal)
    ax3 = fig.add_subplot(gs[1, 0])
    re_diag = np.real(vis_diag).T
    sigma_d = float(np.std(re_diag))
    mean_d = float(np.mean(re_diag))
    im3 = ax3.imshow(
        re_diag,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="twilight",
        vmin=mean_d - 2.5 * sigma_d,
        vmax=mean_d + 2.5 * sigma_d,
    )
    ax3.set_title(
        f"3. Diagonal Baseline Fringe Waterfall: Ant {bl_diag[0]} - Ant {bl_diag[1]}\n(0.85 m Diagonal, Re(V_09))",
        fontsize=11,
        fontweight="bold",
        color="#38ef7d",
    )
    ax3.set_ylabel("Frequency [MHz]", fontsize=10, color="#C9D1D9")
    cb3 = plt.colorbar(im3, ax=ax3, pad=0.02, shrink=0.9)
    cb3.set_label("Linear", color="#C9D1D9", fontsize=9)

    # 4. Long Baseline Fringe Waterfall (Ant 0 - Ant 63, 5.94m full diagonal)
    ax4 = fig.add_subplot(gs[1, 1])
    re_long = np.real(vis_long).T
    sigma_l = float(np.std(re_long))
    mean_l = float(np.mean(re_long))
    im4 = ax4.imshow(
        re_long,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="twilight",
        vmin=mean_l - 2.5 * sigma_l,
        vmax=mean_l + 2.5 * sigma_l,
    )
    ax4.set_title(
        f"4. Long Baseline Fringe Waterfall: Ant {bl_long[0]} - Ant {bl_long[1]}\n(5.94 m Full Array Aperture, Re(V_0,63))",
        fontsize=11,
        fontweight="bold",
        color="#40C4FF",
    )
    cb4 = plt.colorbar(im4, ax=ax4, pad=0.02, shrink=0.9)
    cb4.set_label("Linear", color="#C9D1D9", fontsize=9)

    # 5. Coherence Ratio Dynamic Spectrum
    ax5 = fig.add_subplot(gs[2, :2])
    coh_data = coherence_cube.T  # (freq, time)
    vmax_coh = float(min(1.0, np.percentile(coh_data, 99.0)))
    im5 = ax5.imshow(
        coh_data,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="plasma",
        vmin=0.0,
        vmax=vmax_coh,
    )
    ax5.set_title(
        "5. Interferometric Coherence Dynamic Spectrum |V_cross| / P_auto\n(Shows coherent transients & RFI vs uncorrelated thermal noise floor)",
        fontsize=11,
        fontweight="bold",
        color="#FF5252",
    )
    ax5.set_xlabel("Window Elapsed Time [seconds]", fontsize=10, color="#C9D1D9")
    ax5.set_ylabel("Frequency [MHz]", fontsize=10, color="#C9D1D9")
    cb5 = plt.colorbar(im5, ax=ax5, pad=0.015, shrink=0.9)
    cb5.set_label("Coherence Ratio", color="#C9D1D9", fontsize=9)

    # 6. Active Channel 64x64 Matrix Snapshot (Peak Transient Frame)
    ax6 = fig.add_subplot(gs[0, 2])
    peak_act_mag = np.abs(peak_matrix[active_ch])
    im6 = ax6.imshow(
        peak_act_mag,
        origin="lower",
        cmap="inferno",
        aspect="equal",
        vmin=0.0,
        vmax=float(np.percentile(peak_act_mag, 99.5)),
    )
    ax6.set_title(
        f"6a. Peak Frame Matrix |V_ij|\nActive Ch {active_ch} ({freq_mhz[active_ch]:.1f} MHz, t={time_s[peak_t_idx]:.1f}s)",
        fontsize=9.5,
        fontweight="bold",
        color="#FFD700",
    )
    ax6.set_xlabel("Antenna j", fontsize=8, color="#C9D1D9")
    ax6.set_ylabel("Antenna i", fontsize=8, color="#C9D1D9")
    plt.colorbar(im6, ax=ax6, pad=0.03, shrink=0.8)

    # 7. Active Channel 64x64 Phase arg(V_ij)
    ax7 = fig.add_subplot(gs[1, 2])
    peak_act_ph = np.angle(peak_matrix[active_ch])
    im7 = ax7.imshow(
        peak_act_ph,
        origin="lower",
        cmap="twilight",
        aspect="equal",
        vmin=-np.pi,
        vmax=np.pi,
    )
    ax7.set_title(
        f"6b. Peak Frame Phase arg(V_ij)\nActive Ch {active_ch} ({freq_mhz[active_ch]:.1f} MHz)",
        fontsize=9.5,
        fontweight="bold",
        color="#FFD700",
    )
    ax7.set_xlabel("Antenna j", fontsize=8, color="#C9D1D9")
    ax7.set_ylabel("Antenna i", fontsize=8, color="#C9D1D9")
    plt.colorbar(im7, ax=ax7, pad=0.03, shrink=0.8)

    # 8. Quiet Channel 64x64 Matrix (Baseline Frame)
    ax8 = fig.add_subplot(gs[2, 2])
    base_qui_mag = np.abs(base_matrix[quiet_ch])
    im8 = ax8.imshow(
        base_qui_mag,
        origin="lower",
        cmap="inferno",
        aspect="equal",
        vmin=0.0,
        vmax=float(np.percentile(base_qui_mag, 99.5)),
    )
    ax8.set_title(
        f"7. Quiet Channel Matrix |V_ij|\nQuiet Ch {quiet_ch} ({freq_mhz[quiet_ch]:.1f} MHz, Noise Baseline)",
        fontsize=9.5,
        fontweight="bold",
        color="#40C4FF",
    )
    ax8.set_xlabel("Antenna j", fontsize=8, color="#C9D1D9")
    ax8.set_ylabel("Antenna i", fontsize=8, color="#C9D1D9")
    plt.colorbar(im8, ax=ax8, pad=0.03, shrink=0.8)

    # Master Title
    fig.suptitle(
        f"CHARTS 64-Antenna Correlator Waterfalls & Visibility Diagnostics — Window [{window_label}]\n"
        f"Band: {freq_mhz[0]:.1f}–{freq_mhz[-1]:.1f} MHz ({num_channels} channels) | Duration: {duration_s:.1f} s | Site: Observatorio Carén",
        fontsize=13,
        fontweight="bold",
        color="white",
        y=0.98,
    )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=180, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print(f"[SUCCESS] Saved correlator waterfall dashboard: {output_png}")
    return output_png


def main():
    parser = argparse.ArgumentParser(
        description="CHARTS Correlator Multi-Baseline Waterfall Generator"
    )
    parser.add_argument(
        "--corr-dir",
        type=str,
        required=True,
        help="Path to correlator output directory",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to output PNG file"
    )
    parser.add_argument(
        "--num-elements", type=int, default=64, help="Number of antennas (default: 64)"
    )
    parser.add_argument(
        "--num-channels",
        type=int,
        default=672,
        help="Number of channels (default: 672)",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=60.0,
        help="Window duration in seconds (default: 60.0)",
    )
    parser.add_argument(
        "--max-time-samples",
        type=int,
        default=200,
        help="Time samples across window (default: 200)",
    )
    parser.add_argument(
        "--window-label",
        type=str,
        default="Window",
        help="Window label (e.g. '15:00 UTC')",
    )

    args = parser.parse_args()

    generate_correlator_waterfalls(
        corr_dir=Path(args.corr_dir),
        output_png=Path(args.output),
        num_elements=args.num_elements,
        num_channels=args.num_channels,
        duration_s=args.duration_s,
        max_time_samples=args.max_time_samples,
        window_label=args.window_label,
    )


if __name__ == "__main__":
    main()
