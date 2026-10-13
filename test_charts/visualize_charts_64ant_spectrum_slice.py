#!/usr/bin/env python3
"""
CHARTS 64-Antenna Instantaneous Frequency Spectrum Visualizer (Time Slice)
==========================================================================
Extracts and plots the instantaneous power spectrum P(f) = |V(t0, f)|^2 at a
specified time slice (e.g. t = 0.5 ms) across frequency channels (300 MHz to 500 MHz):
  - Y-axis: Power in dB: 10 * log10(P + 1) (or linear scale via --linear).
  - X-axis: Frequency in MHz (300 MHz to 500 MHz).

Generates two complementary visualizations:
  1. 8x8 Physical Grid Dashboard:
     Subplots arranged according to physical antenna positions on the ground
     (Row 7 North at top to Row 0 South at bottom; Col 0 West to Col 7 East),
     with uniform dB scaling for direct feed-to-feed comparison and highlighted
     saturated feeds (7, 23, 42, 55).
  2. Master Overlay & Diagnostics Dashboard:
     All 64 antenna spectral traces overlaid on a single high-resolution plot,
     contrasting operational feeds against saturated feeds, with array mean and
     median profiles.

Usage Examples:
  # 1. Plot at time = 0.5 ms for a specific HDF5 dump:
  python test_charts/visualize_charts_64ant_spectrum_slice.py \\
      --input ./dumps_charts_64ant/baseband/vela_with_noise_64ant_5ms.h5 \\
      --time 0.5

  # 2. Or using the automatic launcher on Trillium:
  bash test_charts/run_visualize_spectrum_slice.sh \\
      --input ./dumps_charts_64ant/baseband/sun_with_noise_saturated_64ant_5ms.h5 \\
      --time 0.5

  # 3. Batch mode for all dumps:
  bash test_charts/run_visualize_spectrum_slice.sh --batch --time 0.5
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as e:
    sys.stderr.write(
        f"\n[ERROR] Missing required Python scientific stack: {e}\n\n"
        "On Trillium / Compute Canada, please load the scientific environment:\n\n"
        "    module load python/3.11 scipy-stack\n\n"
        "Or use the automated launcher:\n\n"
        "    bash test_charts/run_visualize_spectrum_slice.sh [options]\n\n"
    )
    sys.exit(1)

# Ensure test_charts path is available
_test_charts_dir = Path(__file__).resolve().parent
_kotekan_root = _test_charts_dir.parent
if str(_test_charts_dir) not in sys.path:
    sys.path.insert(0, str(_test_charts_dir))

try:
    import h5py

    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

from constants import (
    CHARTS_CHANNEL_WIDTH_MHZ,
    CHARTS_LATITUDE_DEG,
    CHARTS_LONGITUDE_DEG,
    DEFAULT_FREQUENCY_START_MHZ,
    DEFAULT_SPACING_M,
    FPGA_TIME_RESOLUTION_US,
    LOCAL_FREQUENCY_CHANNELS,
)

# Precomputed LUT for 4-bit two's complement integer decoding: [-8..+7]
INT4_LUT = np.array(
    [0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1],
    dtype=np.float32,
)

DEFAULT_SATURATED_ANTENNAS = [7, 23, 42, 55]


def unpack_int4x2_power(packed_bytes: np.ndarray) -> np.ndarray:
    """Decodes packed 4-bit complex integers into power |V|^2 = real^2 + imag^2."""
    r = INT4_LUT[packed_bytes & 0x0F]
    i = INT4_LUT[(packed_bytes >> 4) & 0x0F]
    return r * r + i * i


def load_baseband_frame(
    file_path: Path,
    frame_idx: int = 0,
    samples_per_frame: int = 1536,
) -> Tuple[np.ndarray, Dict[str, any]]:
    """Loads 1 frame of baseband data from either .h5 or .bin file."""
    if file_path.suffix.lower() in [".h5", ".hdf5"]:
        if not HAS_H5PY:
            raise ImportError(
                "h5py is required to read .h5 files. Install via: pip install h5py"
            )
        with h5py.File(file_path, "r") as f:
            meta = {k: f.attrs[k] for k in f.attrs.keys()}
            if "baseband" in f:
                dset = f["baseband"]
                n_ant, n_freq, total_time = dset.shape
                start_t = frame_idx * samples_per_frame
                end_t = min(start_t + samples_per_frame, total_time)
                frame_packed = dset[:, :, start_t:end_t]
            elif "fengine_compat/voltage" in f:
                dset = f["fengine_compat/voltage"]
                n_ant, _, n_freq, total_time = dset.shape
                start_t = frame_idx * samples_per_frame
                end_t = min(start_t + samples_per_frame, total_time)
                frame_packed = dset[:, 0, :, start_t:end_t]
            else:
                raise KeyError(
                    f"No valid baseband dataset found in {file_path}. Keys: {list(f.keys())}"
                )

        meta.setdefault("num_antennas", n_ant)
        meta.setdefault("num_freq", n_freq)
        meta.setdefault("samples_per_frame", samples_per_frame)
        meta.setdefault("freq_start_MHz", DEFAULT_FREQUENCY_START_MHZ)
        meta.setdefault("delta_freq_MHz", CHARTS_CHANNEL_WIDTH_MHZ)
        meta.setdefault("delta_time_us", FPGA_TIME_RESOLUTION_US)
        meta.setdefault("scenario", file_path.stem)
        meta.setdefault("target_name", meta.get("scenario", file_path.stem))
        return frame_packed, meta

    # RAW Binary format
    payload_bytes_per_frame = samples_per_frame * LOCAL_FREQUENCY_CHANNELS * 64
    file_size = os.path.getsize(file_path)

    with open(file_path, "rb") as f:
        meta_size_bytes = f.read(4)
        if len(meta_size_bytes) < 4:
            raise ValueError(f"File {file_path} is too small to contain a header.")
        meta_size = int(np.frombuffer(meta_size_bytes, dtype="<u4", count=1)[0])
        frame_stride = 4 + meta_size + payload_bytes_per_frame

        if file_size < frame_stride:
            if file_size >= payload_bytes_per_frame:
                frame_stride = payload_bytes_per_frame
                f.seek(frame_idx * frame_stride)
            else:
                raise ValueError(f"File {file_path} too small for one frame.")
        else:
            seek_pos = frame_idx * frame_stride + 4 + meta_size
            f.seek(seek_pos)

        raw_data = np.fromfile(f, dtype=np.uint8, count=payload_bytes_per_frame)

    shaped = raw_data.reshape(samples_per_frame, LOCAL_FREQUENCY_CHANNELS, 64)
    frame_packed = np.transpose(shaped, (2, 1, 0))

    meta = {
        "num_antennas": 64,
        "num_freq": LOCAL_FREQUENCY_CHANNELS,
        "samples_per_frame": samples_per_frame,
        "freq_start_MHz": DEFAULT_FREQUENCY_START_MHZ,
        "delta_freq_MHz": CHARTS_CHANNEL_WIDTH_MHZ,
        "delta_time_us": FPGA_TIME_RESOLUTION_US,
        "scenario": file_path.stem,
        "target_name": file_path.stem,
    }
    return frame_packed, meta


def extract_spectrum_at_time(
    frame_packed: np.ndarray,
    meta: Dict[str, any],
    target_time_ms: float = 0.5,
    window_samples: int = 1,
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    """
    Extracts the instantaneous frequency spectrum slice P(f) at target_time_ms.

    Returns:
        power_spectrum: np.ndarray shape (64, num_freq)
        freqs_mhz: np.ndarray shape (num_freq,)
        actual_time_ms: float actual timestamp of sampled slice
        sample_idx: int sample index in frame
    """
    n_ant, n_freq, n_time = frame_packed.shape
    dt_us = float(meta.get("delta_time_us", FPGA_TIME_RESOLUTION_US))
    dt_ms = dt_us / 1000.0
    f_start_mhz = float(meta.get("freq_start_MHz", DEFAULT_FREQUENCY_START_MHZ))
    df_mhz = float(meta.get("delta_freq_MHz", CHARTS_CHANNEL_WIDTH_MHZ))

    freqs_mhz = f_start_mhz + np.arange(n_freq) * df_mhz

    # Determine sample index for target_time_ms
    sample_idx = int(round(target_time_ms / dt_ms))
    sample_idx = max(0, min(sample_idx, n_time - 1))
    actual_time_ms = sample_idx * dt_ms

    if window_samples <= 1:
        slice_packed = frame_packed[:, :, sample_idx]
        power_spectrum = unpack_int4x2_power(slice_packed)
    else:
        s_start = max(0, sample_idx - window_samples // 2)
        s_end = min(n_time, s_start + window_samples)
        window_packed = frame_packed[:, :, s_start:s_end]
        power_spectrum = unpack_int4x2_power(window_packed).mean(axis=-1)

    return power_spectrum, freqs_mhz, actual_time_ms, sample_idx


# =============================================================================
# Visualization 1: 8x8 Physical Grid Dashboard (dB vs Freq)
# =============================================================================


def plot_8x8_spectrum_grid(
    power_spectrum: np.ndarray,
    freqs_mhz: np.ndarray,
    meta: Dict[str, any],
    actual_time_ms: float,
    sample_idx: int,
    output_png: Path,
    use_db: bool = True,
    freq_min: float = 300.0,
    freq_max: float = 500.0,
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
) -> Path:
    """
    Plots an 8x8 physical matrix of frequency spectra (dB vs MHz) matching
    the CHARTS array orientation on the ground at Observatorio Caren.
    """
    n_ant, n_freq = power_spectrum.shape
    saturated_antennas = set(meta.get("saturated_antennas", []))
    if not saturated_antennas:
        mean_p = power_spectrum.mean(axis=1)
        saturated_antennas = set(np.where(mean_p > 45.0)[0].tolist())

    if use_db:
        disp_spectrum = 10.0 * np.log10(power_spectrum + 1.0)
        unit_label = "Power [dB]"
    else:
        disp_spectrum = power_spectrum
        unit_label = "Power [linear]"

    normal_ant_mask = [a for a in range(64) if a not in saturated_antennas]
    if not normal_ant_mask:
        normal_ant_mask = list(range(64))

    normal_data = disp_spectrum[normal_ant_mask]
    if ymin is None:
        ymin = max(0.0, float(np.percentile(normal_data, 1.0)) - 2.0)
    if ymax is None:
        ymax = max(float(np.percentile(disp_spectrum, 99.8)) + 3.0, ymin + 15.0)

    # Set x-limits honoring the user's requested 300 to 500 MHz band
    data_fmin = float(freqs_mhz[0])
    data_fmax = float(freqs_mhz[-1])
    disp_fmin = min(freq_min, data_fmin)
    disp_fmax = max(freq_max, data_fmax)

    plt.style.use("dark_background")
    fig, axes = plt.subplots(
        8,
        8,
        figsize=(24, 20),
        sharex=True,
        sharey=True,
        gridspec_kw={
            "wspace": 0.08,
            "hspace": 0.08,
            "left": 0.06,
            "right": 0.95,
            "top": 0.91,
            "bottom": 0.06,
        },
    )

    for a in range(64):
        col = a & 7  # 0 to 7 (West to East)
        row = a >> 3  # 0 to 7 (South to North)

        plot_row = 7 - row
        plot_col = col
        ax = axes[plot_row, plot_col]

        y_vals = disp_spectrum[a]
        is_sat = a in saturated_antennas

        if is_sat:
            line_color = "#ef4444"
            fill_color = "#7f1d1d"
            badge_text = f"A{a:02d} [SAT]"
            badge_bg = "#7f1d1d"
            badge_color = "#fef08a"
            for spine in ax.spines.values():
                spine.set_edgecolor("#ef4444")
                spine.set_linewidth(2.2)
        else:
            line_color = "#38bdf8"
            fill_color = "#0369a1"
            badge_text = f"A{a:02d} [r{row},c{col}]"
            badge_bg = "#0f172a"
            badge_color = "#38bdf8"
            for spine in ax.spines.values():
                spine.set_edgecolor("#334155")
                spine.set_linewidth(1.0)

        # Plot spectrum curve and area
        ax.plot(freqs_mhz, y_vals, color=line_color, linewidth=1.1, zorder=3)
        ax.fill_between(freqs_mhz, ymin, y_vals, color=fill_color, alpha=0.25, zorder=2)

        ax.set_xlim(disp_fmin, disp_fmax)
        ax.set_ylim(ymin, ymax)
        ax.grid(True, linestyle="--", color="#1e293b", alpha=0.6, zorder=1)

        # Antenna badge
        ax.text(
            0.05,
            0.90,
            badge_text,
            transform=ax.transAxes,
            fontsize=8.5,
            fontweight="bold",
            color=badge_color,
            verticalalignment="top",
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor=badge_bg,
                edgecolor=badge_color,
                alpha=0.85,
                linewidth=0.8,
            ),
        )

        ax.tick_params(axis="both", which="both", labelsize=7.5, colors="#94a3b8")
        if plot_row != 7:
            ax.set_xticklabels([])
        if plot_col != 0:
            ax.set_yticklabels([])

    # Margins and Labels
    for c in range(8):
        axes[7, c].set_xlabel("Freq [MHz]", fontsize=9.5, color="#e2e8f0", labelpad=3)
    for r in range(8):
        axes[r, 0].set_ylabel(
            f"{unit_label}", fontsize=9.5, color="#e2e8f0", labelpad=3
        )

    # Cardinal orientation labels
    fig.text(
        0.50,
        0.932,
        "[^] NORTH (+Y, Row 7)",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="#38bdf8",
    )
    fig.text(
        0.50,
        0.024,
        "[v] SOUTH (-Y, Row 0)",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="#38bdf8",
    )
    fig.text(
        0.015,
        0.485,
        "[<] WEST (-X, Col 0)",
        ha="center",
        va="center",
        rotation=90,
        fontsize=11,
        fontweight="bold",
        color="#38bdf8",
    )
    fig.text(
        0.965,
        0.485,
        "EAST (+X, Col 7) [>]",
        ha="center",
        va="center",
        rotation=-90,
        fontsize=11,
        fontweight="bold",
        color="#38bdf8",
    )

    target_name = meta.get("target_name", meta.get("scenario", "Simulation"))
    scenario = meta.get("scenario", "CHARTS-64")

    fig.suptitle(
        f"CHARTS 64-Antenna Frequency Spectrum Slice (8x8 Layout) - {target_name}\n"
        f"Time Slice: t = {actual_time_ms:.3f} ms (sample #{sample_idx}) | "
        f"Frequency: {disp_fmin:.1f} to {disp_fmax:.1f} MHz ({n_freq} channels) | "
        f"Scenario: {scenario} | Site: Caren ({CHARTS_LATITUDE_DEG:.4f} deg, {CHARTS_LONGITUDE_DEG:.4f} deg)",
        fontsize=13,
        fontweight="bold",
        color="#ffffff",
        y=0.975,
    )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=180, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)

    print(f"[OK] Saved 8x8 antenna spectrum slice: {output_png}")
    return output_png


# =============================================================================
# Visualization 2: Master Overlay & Diagnostics Dashboard
# =============================================================================


def plot_spectrum_overlay(
    power_spectrum: np.ndarray,
    freqs_mhz: np.ndarray,
    meta: Dict[str, any],
    actual_time_ms: float,
    sample_idx: int,
    output_png: Path,
    use_db: bool = True,
    freq_min: float = 300.0,
    freq_max: float = 500.0,
) -> Path:
    """
    Renders an overlay of all 64 antenna spectra at t = actual_time_ms on a single plot,
    highlighting operational vs saturated feeds, with array mean and median curves.
    """
    n_ant, n_freq = power_spectrum.shape
    saturated_antennas = set(meta.get("saturated_antennas", []))
    if not saturated_antennas:
        mean_p = power_spectrum.mean(axis=1)
        saturated_antennas = set(np.where(mean_p > 45.0)[0].tolist())

    if use_db:
        disp_spectrum = 10.0 * np.log10(power_spectrum + 1.0)
        unit_label = "Power [dB]"
    else:
        disp_spectrum = power_spectrum
        unit_label = "Power [linear]"

    plt.style.use("dark_background")
    fig, (ax_main, ax_stat) = plt.subplots(
        1,
        2,
        figsize=(18, 9),
        gridspec_kw={
            "width_ratios": [3.2, 1],
            "wspace": 0.18,
            "left": 0.07,
            "right": 0.95,
            "top": 0.90,
            "bottom": 0.10,
        },
    )

    data_fmin = float(freqs_mhz[0])
    data_fmax = float(freqs_mhz[-1])
    disp_fmin = min(freq_min, data_fmin)
    disp_fmax = max(freq_max, data_fmax)

    # Plot operational feeds
    normal_labeled = False
    for a in range(64):
        if a not in saturated_antennas:
            lbl = "Operational Feeds (60)" if not normal_labeled else None
            ax_main.plot(
                freqs_mhz,
                disp_spectrum[a],
                color="#38bdf8",
                alpha=0.30,
                linewidth=0.9,
                label=lbl,
                zorder=2,
            )
            normal_labeled = True

    # Plot saturated feeds in high contrast
    sat_colors = ["#ef4444", "#f97316", "#eab308", "#ec4899", "#a855f7"]
    for idx, bad_ant in enumerate(sorted(list(saturated_antennas))):
        color = sat_colors[idx % len(sat_colors)]
        ax_main.plot(
            freqs_mhz,
            disp_spectrum[bad_ant],
            color=color,
            linewidth=2.2,
            alpha=0.95,
            label=f"Saturated Ant #{bad_ant:02d}",
            zorder=4,
        )

    # Array Statistics: Median and Mean
    normal_ant_mask = [a for a in range(64) if a not in saturated_antennas]
    if normal_ant_mask:
        mean_spectrum = disp_spectrum[normal_ant_mask].mean(axis=0)
        median_spectrum = np.median(disp_spectrum[normal_ant_mask], axis=0)

        ax_main.plot(
            freqs_mhz,
            mean_spectrum,
            color="#ffffff",
            linewidth=2.5,
            linestyle="-",
            label="Array Mean (Normal Feeds)",
            zorder=5,
        )
        ax_main.plot(
            freqs_mhz,
            median_spectrum,
            color="#22c55e",
            linewidth=2.0,
            linestyle="--",
            label="Array Median (Normal Feeds)",
            zorder=5,
        )

    ax_main.set_xlim(disp_fmin, disp_fmax)
    ax_main.set_xlabel(
        "Frequency [MHz]", fontsize=12, fontweight="bold", color="#f8fafc", labelpad=8
    )
    ax_main.set_ylabel(
        f"Spectrum Intensity ({unit_label})",
        fontsize=12,
        fontweight="bold",
        color="#f8fafc",
        labelpad=8,
    )
    ax_main.grid(True, linestyle="--", color="#334155", alpha=0.6, zorder=1)
    ax_main.tick_params(colors="#94a3b8", labelsize=10)
    ax_main.legend(
        loc="upper right",
        framealpha=0.85,
        facecolor="#0f172a",
        edgecolor="#475569",
        fontsize=9.5,
    )

    # Side Panel: Physical Array Mini-map & Statistics
    ax_stat.axis("off")
    pos_x = np.array([a & 7 for a in range(64)]) * DEFAULT_SPACING_M
    pos_y = np.array([a >> 3 for a in range(64)]) * DEFAULT_SPACING_M

    # Mini array layout inset
    ax_inset = fig.add_axes([0.76, 0.52, 0.18, 0.35])
    ax_inset.set_facecolor("#0f172a")
    for a in range(64):
        if a in saturated_antennas:
            ax_inset.scatter(
                pos_x[a], pos_y[a], c="#ef4444", s=60, marker="X", zorder=4
            )
        else:
            ax_inset.scatter(pos_x[a], pos_y[a], c="#38bdf8", s=30, zorder=3)
    ax_inset.set_title("Array Geometry (8x8)", fontsize=9.5, color="#f8fafc", pad=4)
    ax_inset.set_xlabel("East-West [m]", fontsize=8, color="#94a3b8")
    ax_inset.set_ylabel("North-South [m]", fontsize=8, color="#94a3b8")
    ax_inset.tick_params(colors="#94a3b8", labelsize=7)
    ax_inset.grid(True, linestyle=":", color="#334155", alpha=0.5)

    # Statistical text block
    norm_mean = disp_spectrum[normal_ant_mask].mean() if normal_ant_mask else 0.0
    norm_std = disp_spectrum[normal_ant_mask].std() if normal_ant_mask else 0.0
    sat_mean = (
        disp_spectrum[list(saturated_antennas)].mean() if saturated_antennas else 0.0
    )

    stat_box_text = (
        f"--- SPECTRUM SLICE INFO ---\n"
        f"Target: {meta.get('target_name', 'Simulation')}\n"
        f"Scenario: {meta.get('scenario', 'N/A')}\n"
        f"Target Time: t = {actual_time_ms:.3f} ms\n"
        f"Sample Index: #{sample_idx} / {meta.get('samples_per_frame', 1536)}\n"
        f"Channels: {n_freq} (df = {meta.get('delta_freq_MHz', 0.3):.2f} MHz)\n\n"
        f"--- POWER STATISTICS ({unit_label}) ---\n"
        f"Normal Feeds Mean: {norm_mean:.2f} +/- {norm_std:.2f}\n"
        f"Saturated Feeds: {len(saturated_antennas)} feeds\n"
        f"Saturated Mean: {sat_mean:.2f}\n"
        f"Theoretical Rail: 19.96 dB\n"
        f"Site: Observatorio Caren"
    )

    ax_stat.text(
        0.05,
        0.40,
        stat_box_text,
        transform=ax_stat.transAxes,
        fontsize=9.5,
        family="monospace",
        color="#e2e8f0",
        verticalalignment="top",
        bbox=dict(
            boxstyle="round,pad=0.6",
            facecolor="#1e293b",
            edgecolor="#475569",
            linewidth=1.2,
        ),
    )

    fig.suptitle(
        f"CHARTS 64-Antenna Frequency Spectrum Slice (Overlay) - {meta.get('target_name', 'Simulation')}\n"
        f"t = {actual_time_ms:.3f} ms | Band: {disp_fmin:.1f} - {disp_fmax:.1f} MHz | {unit_label} vs Frequency",
        fontsize=13,
        fontweight="bold",
        color="#ffffff",
        y=0.97,
    )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=180, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)

    print(f"[OK] Saved spectrum overlay dashboard: {output_png}")
    return output_png


# =============================================================================
# CLI Main Routine
# =============================================================================


def find_candidate_file(baseband_dir: Path, scenario: str) -> Optional[Path]:
    """Finds matching .h5 or .bin file for a scenario in baseband_dir."""
    candidates = [
        baseband_dir / f"{scenario}_64ant_5ms.h5",
        baseband_dir / f"{scenario}_64ant_5ms.bin",
        baseband_dir / f"{scenario}.h5",
        baseband_dir / f"{scenario}.bin",
    ]
    for c in candidates:
        if c.exists():
            return c
    h5_matches = list(baseband_dir.glob(f"*{scenario}*.h5"))
    if h5_matches:
        return h5_matches[0]
    bin_matches = list(baseband_dir.glob(f"*{scenario}*.bin"))
    if bin_matches:
        return bin_matches[0]
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Visualize frequency spectrum shape P(f) (dB vs MHz, 300 to 500 MHz) at specified time slice.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=None,
        help="Path to .h5 or .bin baseband file.",
    )
    parser.add_argument(
        "--baseband-dir",
        type=Path,
        default=Path("./dumps_charts_64ant/baseband"),
        help="Directory containing baseband dumps (default: ./dumps_charts_64ant/baseband).",
    )
    parser.add_argument(
        "--scenario",
        "-s",
        type=str,
        default="vela_with_noise",
        help="Scenario name to find if --input is omitted (default: vela_with_noise).",
    )
    parser.add_argument(
        "--time",
        "-t",
        type=float,
        default=0.5,
        help="Target time slice in milliseconds (default: 0.5 ms).",
    )
    parser.add_argument(
        "--window-samples",
        type=int,
        default=1,
        help="Number of time samples to average around target time (default: 1 for instantaneous).",
    )
    parser.add_argument(
        "--freq-min",
        type=float,
        default=300.0,
        help="Minimum display frequency in MHz (default: 300.0).",
    )
    parser.add_argument(
        "--freq-max",
        type=float,
        default=500.0,
        help="Maximum display frequency in MHz (default: 500.0).",
    )
    parser.add_argument(
        "--linear",
        action="store_true",
        help="Plot linear power instead of logarithmic dB scale.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Explicit output PNG path (if omitted, generates both 8x8 grid and overlay).",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=Path("./dumps_charts_64ant/plots"),
        help="Directory to save generated plots (default: ./dumps_charts_64ant/plots).",
    )
    parser.add_argument(
        "--frame",
        "-f",
        type=int,
        default=0,
        help="Frame index to load (default: 0).",
    )
    parser.add_argument(
        "--overlay-only",
        action="store_true",
        help="Only generate the 64-antenna overlay plot.",
    )
    parser.add_argument(
        "--grid-only",
        action="store_true",
        help="Only generate the 8x8 physical grid plot.",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Batch process all baseband dumps found in --baseband-dir.",
    )

    args = parser.parse_args()

    # Handle Batch Mode
    if args.batch:
        if not args.baseband_dir.exists():
            print(f"[ERROR] Baseband directory does not exist: {args.baseband_dir}")
            sys.exit(1)

        files = sorted(
            list(args.baseband_dir.glob("*.h5")) + list(args.baseband_dir.glob("*.bin"))
        )
        if not files:
            print(f"[WARN] No .h5 or .bin files found in {args.baseband_dir}")
            sys.exit(0)

        print(f"=== Found {len(files)} baseband dumps in {args.baseband_dir} ===")
        args.plots_dir.mkdir(parents=True, exist_ok=True)

        for filepath in files:
            stem = filepath.stem.replace("_64ant_5ms", "")
            t_str = f"t{args.time:.1f}ms"
            grid_out = args.plots_dir / f"spectrum_8x8_{stem}_{t_str}.png"
            over_out = args.plots_dir / f"spectrum_overlay_{stem}_{t_str}.png"

            print(f"\nProcessing {filepath.name} at t = {args.time:.2f} ms ...")
            try:
                frame_packed, meta = load_baseband_frame(filepath, frame_idx=args.frame)
                power_spectrum, freqs_mhz, actual_t_ms, sample_idx = (
                    extract_spectrum_at_time(
                        frame_packed,
                        meta,
                        target_time_ms=args.time,
                        window_samples=args.window_samples,
                    )
                )
                if not args.overlay_only:
                    plot_8x8_spectrum_grid(
                        power_spectrum,
                        freqs_mhz,
                        meta,
                        actual_t_ms,
                        sample_idx,
                        grid_out,
                        use_db=not args.linear,
                        freq_min=args.freq_min,
                        freq_max=args.freq_max,
                    )
                if not args.grid_only:
                    plot_spectrum_overlay(
                        power_spectrum,
                        freqs_mhz,
                        meta,
                        actual_t_ms,
                        sample_idx,
                        over_out,
                        use_db=not args.linear,
                        freq_min=args.freq_min,
                        freq_max=args.freq_max,
                    )
            except Exception as e:
                print(f"[ERROR] Failed to process {filepath}: {e}")

        print("\n[OK] Batch spectrum slicing complete.")
        return

    # Single File / Scenario Mode
    target_file = args.input
    if target_file is None:
        target_file = find_candidate_file(args.baseband_dir, args.scenario)
        if target_file is None:
            print(
                f"[ERROR] Could not find baseband file for scenario '{args.scenario}' in {args.baseband_dir}"
            )
            print("Please specify an explicit path using --input <path.h5 | path.bin>")
            sys.exit(1)

    if not target_file.exists():
        print(f"[ERROR] Input file does not exist: {target_file}")
        sys.exit(1)

    print(f"Loading baseband frame from: {target_file}")
    t0 = time.perf_counter()
    frame_packed, meta = load_baseband_frame(target_file, frame_idx=args.frame)
    power_spectrum, freqs_mhz, actual_t_ms, sample_idx = extract_spectrum_at_time(
        frame_packed, meta, target_time_ms=args.time, window_samples=args.window_samples
    )
    extract_ms = (time.perf_counter() - t0) * 1000.0
    print(
        f"Extracted spectrum slice at t = {actual_t_ms:.3f} ms (sample #{sample_idx}) in {extract_ms:.1f} ms."
    )

    stem = target_file.stem.replace("_64ant_5ms", "")
    t_str = f"t{args.time:.1f}ms"

    if args.output is not None:
        grid_out = args.output
        over_out = (
            args.output.parent / f"{args.output.stem}_overlay{args.output.suffix}"
        )
    else:
        args.plots_dir.mkdir(parents=True, exist_ok=True)
        grid_out = args.plots_dir / f"spectrum_8x8_{stem}_{t_str}.png"
        over_out = args.plots_dir / f"spectrum_overlay_{stem}_{t_str}.png"

    if not args.overlay_only:
        plot_8x8_spectrum_grid(
            power_spectrum,
            freqs_mhz,
            meta,
            actual_t_ms,
            sample_idx,
            grid_out,
            use_db=not args.linear,
            freq_min=args.freq_min,
            freq_max=args.freq_max,
        )

    if not args.grid_only:
        plot_spectrum_overlay(
            power_spectrum,
            freqs_mhz,
            meta,
            actual_t_ms,
            sample_idx,
            over_out,
            use_db=not args.linear,
            freq_min=args.freq_min,
            freq_max=args.freq_max,
        )

    print(f"[OK] Completed spectrum slice visualizations at t = {actual_t_ms:.3f} ms.")


if __name__ == "__main__":
    main()
