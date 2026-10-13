#!/usr/bin/env python3
"""CHARTS Comprehensive Visualization Suite.

Generates high-resolution publication-quality plots and videos:
  - CASM-Style Upper-Triangular Correlation Matrix Waterfall (Figure 11 format)
  - Multi-Baseline Correlator Waterfalls & Dynamic Spectrum
  - Beam Tracker Formed-Beam Waterfalls & Object Tracking Lightcurves
  - Baseband Power Spectrum & Spectrogram Waterfalls
  - Baseband and Tracker MP4 Video Generation (via matplotlib animation / ffmpeg)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np

from .constants import (
    CHARTS_CHANNEL_WIDTH_MHZ,
    DEFAULT_FREQUENCY_START_MHZ,
    DEFAULT_SPACING_M,
    get_antenna_positions,
)
from .inspector import baseline_index, load_astron_correlator_dump


# ---------------------------------------------------------------------------
# 1. CASM Correlation Matrix Waterfall
# ---------------------------------------------------------------------------
def plot_casm_correlation_matrix(
    corr_dir: Path,
    output_path: Path,
    num_antennas: int = 16,
    num_channels: int = 336,
    duration_s: float = 60.0,
    max_time_samples: int = 150,
    window_label: str = "CHARTS CASM Correlation Matrix",
) -> Path:
    """Generates upper-triangular correlation matrix matching CASM Figure 11."""
    corr_files = sorted(corr_dir.glob("corr_*.bin"))
    if not corr_files:
        raise FileNotFoundError(f"No correlator dumps found in {corr_dir}")

    step = max(1, len(corr_files) // max_time_samples)
    selected_files = corr_files[::step][:max_time_samples]
    num_times = len(selected_files)

    freqs_mhz = (
        DEFAULT_FREQUENCY_START_MHZ + np.arange(num_channels) * CHARTS_CHANNEL_WIDTH_MHZ
    )
    time_s = np.linspace(0, duration_s, num_times)

    # Pre-allocate visibility cube: (times, freqs, ant_i, ant_j)
    v_cube = np.zeros(
        (num_times, num_channels, num_antennas, num_antennas), dtype=np.complex64
    )
    for t_idx, fpath in enumerate(selected_files):
        vis = load_astron_correlator_dump(
            fpath, num_elements=64, num_channels=num_channels
        )
        v_cube[t_idx] = vis[:, :num_antennas, :num_antennas]

    # Create upper triangular grid figure
    fig, axes = plt.subplots(
        num_antennas,
        num_antennas,
        figsize=(18, 16),
        dpi=150,
        gridspec_kw={"wspace": 0.08, "hspace": 0.08},
    )

    t_extent = [time_s[0], time_s[-1], freqs_mhz[0], freqs_mhz[-1]]

    for i in range(num_antennas):
        for j in range(num_antennas):
            ax = axes[i, j]
            if i > j:
                # Lower triangular is empty
                ax.axis("off")
                continue

            if i == j:
                # Diagonal: Auto-spectrum P(f)
                auto_pwr = np.mean(np.real(v_cube[:, :, i, i]), axis=0)
                ax.plot(auto_pwr, freqs_mhz, color="crimson", lw=1.0)
                ax.set_ylim(freqs_mhz[0], freqs_mhz[-1])
                ax.tick_params(labelsize=6)
                if i == 0:
                    ax.set_title(f"Ant {i} Auto", fontsize=8, color="darkred")
            else:
                # Off-diagonal: Re(V_ij) Waterfall
                re_v = np.real(v_cube[:, :, i, j]).T  # (freqs, times)
                v_std = np.std(re_v)
                v_max = 2.5 * v_std if v_std > 0 else 1.0
                ax.imshow(
                    re_v,
                    origin="lower",
                    extent=t_extent,
                    aspect="auto",
                    cmap="coolwarm",
                    vmin=-v_max,
                    vmax=v_max,
                )
                ax.set_xticks([])
                ax.set_yticks([])

            if i == 0 and j > 0:
                ax.set_title(f"A{i}-A{j}", fontsize=7)

    fig.suptitle(f"{window_label} ({num_antennas} Antennas)", fontsize=14, y=0.92)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# 2. Multi-Baseline Correlator Waterfalls & Dynamic Spectrum
# ---------------------------------------------------------------------------
def plot_correlator_waterfalls(
    corr_dir: Path,
    output_path: Path,
    num_elements: int = 64,
    num_channels: int = 336,
    duration_s: float = 60.0,
    max_time_samples: int = 200,
    window_label: str = "CHARTS Correlator Waterfalls",
) -> Path:
    """Generates multi-baseline correlator waterfalls & dynamic spectrum."""
    corr_files = sorted(corr_dir.glob("corr_*.bin"))
    if not corr_files:
        raise FileNotFoundError(f"No correlator dumps found in {corr_dir}")

    step = max(1, len(corr_files) // max_time_samples)
    selected_files = corr_files[::step][:max_time_samples]
    num_times = len(selected_files)

    freqs_mhz = (
        DEFAULT_FREQUENCY_START_MHZ + np.arange(num_channels) * CHARTS_CHANNEL_WIDTH_MHZ
    )
    time_s = np.linspace(0, duration_s, num_times)
    t_extent = [time_s[0], time_s[-1], freqs_mhz[0], freqs_mhz[-1]]

    # Extract Key Baselines:
    # 1. Mean Auto-power
    # 2. Short East-West: Ant 0 - Ant 1 (0.6 m)
    # 3. Diagonal: Ant 0 - Ant 9 (0.85 m)
    # 4. Long Baseline: Ant 0 - Ant 63 (5.94 m)
    auto_wf = np.zeros((num_channels, num_times), dtype=np.float32)
    short_wf = np.zeros((num_channels, num_times), dtype=np.float32)
    diag_wf = np.zeros((num_channels, num_times), dtype=np.float32)
    long_wf = np.zeros((num_channels, num_times), dtype=np.float32)

    for t_idx, fpath in enumerate(selected_files):
        vis = load_astron_correlator_dump(
            fpath, num_elements=num_elements, num_channels=num_channels
        )
        auto_wf[:, t_idx] = np.mean(np.real(np.diagonal(vis, axis1=1, axis2=2)), axis=1)
        short_wf[:, t_idx] = np.real(vis[:, 0, 1])
        diag_wf[:, t_idx] = np.real(vis[:, 0, min(9, num_elements - 1)])
        long_wf[:, t_idx] = np.real(vis[:, 0, num_elements - 1])

    fig = plt.figure(figsize=(15, 12), dpi=150)
    gs = GridSpec(2, 2, figure=fig, hspace=0.28, wspace=0.2)

    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(
        auto_wf, origin="lower", extent=t_extent, aspect="auto", cmap="viridis"
    )
    ax1.set_title("Full Array Mean Auto-Power P_auto(t, f)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Frequency (MHz)")
    fig.colorbar(im1, ax=ax1, label="Power (LSB^2)")

    ax2 = fig.add_subplot(gs[0, 1])
    std_s = np.std(short_wf)
    im2 = ax2.imshow(
        short_wf,
        origin="lower",
        extent=t_extent,
        aspect="auto",
        cmap="coolwarm",
        vmin=-2 * std_s,
        vmax=2 * std_s,
    )
    ax2.set_title("Short Baseline Fringe (Ant 0 - Ant 1, 0.6 m)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Frequency (MHz)")
    fig.colorbar(im2, ax=ax2, label="Re(V_01)")

    ax3 = fig.add_subplot(gs[1, 0])
    std_d = np.std(diag_wf)
    im3 = ax3.imshow(
        diag_wf,
        origin="lower",
        extent=t_extent,
        aspect="auto",
        cmap="coolwarm",
        vmin=-2 * std_d,
        vmax=2 * std_d,
    )
    ax3.set_title("Diagonal Baseline Fringe (Ant 0 - Ant 9, 0.85 m)")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Frequency (MHz)")
    fig.colorbar(im3, ax=ax3, label="Re(V_09)")

    ax4 = fig.add_subplot(gs[1, 1])
    std_l = np.std(long_wf)
    im4 = ax4.imshow(
        long_wf,
        origin="lower",
        extent=t_extent,
        aspect="auto",
        cmap="coolwarm",
        vmin=-2 * std_l,
        vmax=2 * std_l,
    )
    ax4.set_title(f"Long Baseline Fringe (Ant 0 - Ant {num_elements-1})")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Frequency (MHz)")
    fig.colorbar(im4, ax=ax4, label=f"Re(V_0,{num_elements-1})")

    fig.suptitle(f"{window_label} — Multi-Baseline Analysis", fontsize=14, y=0.96)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# 3. Beam Tracker Formed-Beam Waterfall & Lightcurves
# ---------------------------------------------------------------------------
def plot_tracker_waterfall(
    tracker_dir: Path,
    output_path: Path,
    max_beams: int = 8,
    num_freq: int = 336,
    samples_per_frame: int = 1536,
    duration_s: float = 60.0,
    beam_targets_str: Optional[str] = None,
) -> Path:
    """Plots power lightcurves for all formed beam channels."""
    tracker_files = sorted(tracker_dir.glob("*.bin"))
    if not tracker_files:
        raise FileNotFoundError(f"No tracker dumps found in {tracker_dir}")

    from .pipeline import parse_beam_targets

    targets = parse_beam_targets(beam_targets_str, max_beams=max_beams)
    beam_labels = [f"Beam {t['beam']}: {t['name']}" for t in targets]

    num_files = len(tracker_files)
    beam_powers = np.zeros((num_files, max_beams), dtype=np.float32)

    for f_idx, fpath in enumerate(tracker_files):
        raw = np.fromfile(str(fpath), dtype=np.uint8)
        meta_size = int(np.frombuffer(raw[:4].tobytes(), dtype="<u4", count=1)[0])
        payload = raw[4 + meta_size :]
        n_floats = samples_per_frame * num_freq * max_beams * 2
        raw_floats = np.frombuffer(payload, dtype="<f4", count=n_floats)
        shaped = raw_floats.reshape(samples_per_frame, num_freq, max_beams, 2)
        c_voltages = shaped[..., 0] + 1j * shaped[..., 1]
        pwr_per_beam = np.mean(np.sum(np.abs(c_voltages) ** 2, axis=1), axis=0)
        beam_powers[f_idx] = pwr_per_beam

    times_s = np.linspace(0, duration_s, num_files)

    fig, ax = plt.subplots(figsize=(14, 7), dpi=150)
    for b in range(max_beams):
        label = beam_labels[b] if b < len(beam_labels) else f"Beam {b}"
        ax.plot(times_s, beam_powers[:, b], label=label, lw=1.5)

    ax.set_title("CHARTS GPU Beam Tracker — Formed Beams Power Evolution", fontsize=13)
    ax.set_xlabel("Time (s)", fontsize=11)
    ax.set_ylabel("Formed Beam Power (Arbitrary Units)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path
