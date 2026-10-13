#!/usr/bin/env python3
"""
CHARTS 64-Antenna Baseband Spectrogram & Spectrum Shape Visualization Tool
=========================================================================
Generates an 8x8 physical subplot matrix of:
  1. Dynamic Spectra (Spectrograms: Frequency vs Time) over a specified time window
     (default: 0.5 ms = 150 time samples @ dt = 10/3 us).
  2. Bandpass Frequency Spectra (Spectrum Shape: Power P(f) vs Frequency) averaged
     over the 0.5 ms window to reveal channel responses and noise floors across the array.

Features:
  - Standalone script: reads directly from existing HDF5 (.h5) or raw binary (.bin)
    baseband dumps without re-running Kotekan or simulation pipelines.
  - Physical 8x8 layout: maps antenna subplots according to their physical positions
    on the ground at Observatorio Carén (Row 7 North at top, Row 0 South at bottom;
    Col 0 West at left, Col 7 East at right).
  - Time window control (--duration-ms 0.5): zooms into a 0.5 ms slice to reveal
    the clear spectral shape and dynamic time structure without over-compression.
  - Fast vectorized int4x2 decoding: transforms 4-bit complex voltages to power
    |V(t, f)|^2 using an optimized LUT in under 300 ms for all 64 antennas.
  - Array-wide calibrated scales: consistent dB scales across all 64 feeds enable
    direct visual comparison of antenna gains, fringe patterns, and saturated feeds.
  - Feed health diagnostics: highlights saturated/rail-clipped feeds (red borders,
    [SAT] badge) with distinct colors.
  - Single-antenna zoom mode: optionally generates an in-depth 3-panel diagnostic plot
    (2D spectrogram + frequency spectrum + time lightcurve) for any specified antenna.
  - Batch mode: automatically processes all baseband dumps in a directory.

Usage Examples:
  # 1. Visualize a specific HDF5 dump (both 2D spectrogram & 1D spectrum shape @ 0.5 ms):
  python test_charts/visualize_charts_64ant_spectrogram.py \\
      --input ./dumps_charts_64ant/baseband/vela_with_noise_64ant_5ms.h5 \\
      --duration-ms 0.5

  # 2. Visualize by scenario name from default directory:
  python test_charts/visualize_charts_64ant_spectrogram.py \\
      --scenario sun_with_noise_saturated \\
      --duration-ms 0.5

  # 3. Zoom into single antenna (e.g. saturated antenna 7):
  python test_charts/visualize_charts_64ant_spectrogram.py \\
      --input ./dumps_charts_64ant/baseband/sun_with_noise_saturated_64ant_5ms.h5 \\
      --antenna 7 \\
      --duration-ms 0.5

  # 4. Batch process all baseband files (generates both 2D spectrogram & 1D spectrum shape):
  python test_charts/visualize_charts_64ant_spectrogram.py --batch --duration-ms 0.5
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
        "    bash test_charts/run_visualize_spectrograms.sh [options]\n\n"
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


# =============================================================================
# Fast Baseband Decoders (HDF5 and RAW Binary)
# =============================================================================


def unpack_int4x2_power(packed_bytes: np.ndarray) -> np.ndarray:
    """
    Decodes packed 4-bit complex integers into power |V|^2 = real^2 + imag^2.

    Args:
        packed_bytes: np.ndarray of dtype uint8 with shape (..., ).
                      Lower 4 bits = signed real [-8..+7],
                      Upper 4 bits = signed imag [-8..+7].
    Returns:
        power: np.ndarray of dtype float32 with same shape as packed_bytes.
    """
    r = INT4_LUT[packed_bytes & 0x0F]
    i = INT4_LUT[(packed_bytes >> 4) & 0x0F]
    return r * r + i * i


def load_frame_from_hdf5(
    h5_path: Path,
    frame_idx: int = 0,
    samples_per_frame: int = 1536,
) -> Tuple[np.ndarray, Dict[str, any]]:
    """
    Extracts 1 frame of baseband data for all 64 antennas from an HDF5 dump.

    Returns:
        frame_packed: np.ndarray shape (64, num_freq, samples_per_frame) uint8
        meta: dict of metadata attributes
    """
    if not HAS_H5PY:
        raise ImportError(
            "h5py is required to read .h5 files. Install via: pip install h5py"
        )

    with h5py.File(h5_path, "r") as f:
        meta = {k: f.attrs[k] for k in f.attrs.keys()}

        # Determine dataset name
        if "baseband" in f:
            dset = f["baseband"]
            # Shape is (num_ant, num_freq, total_time)
            n_ant, n_freq, total_time = dset.shape
            start_t = frame_idx * samples_per_frame
            end_t = min(start_t + samples_per_frame, total_time)
            if start_t >= total_time:
                raise ValueError(
                    f"frame_idx {frame_idx} out of range (total time samples: {total_time}, "
                    f"samples_per_frame: {samples_per_frame})"
                )
            frame_packed = dset[:, :, start_t:end_t]
        elif "fengine_compat/voltage" in f:
            dset = f["fengine_compat/voltage"]
            # Shape is (num_ant, 1, num_freq, total_time)
            n_ant, _, n_freq, total_time = dset.shape
            start_t = frame_idx * samples_per_frame
            end_t = min(start_t + samples_per_frame, total_time)
            frame_packed = dset[:, 0, :, start_t:end_t]
        else:
            raise KeyError(
                f"No valid baseband dataset found in {h5_path}. Keys: {list(f.keys())}"
            )

    # Fallback metadata defaults if missing
    meta.setdefault("num_antennas", n_ant)
    meta.setdefault("num_freq", n_freq)
    meta.setdefault("samples_per_frame", samples_per_frame)
    meta.setdefault("freq_start_MHz", DEFAULT_FREQUENCY_START_MHZ)
    meta.setdefault("delta_freq_MHz", CHARTS_CHANNEL_WIDTH_MHZ)
    meta.setdefault("delta_time_us", FPGA_TIME_RESOLUTION_US)
    meta.setdefault("scenario", h5_path.stem)
    meta.setdefault("target_name", meta.get("scenario", h5_path.stem))

    return frame_packed, meta


def load_frame_from_raw_bin(
    bin_path: Path,
    frame_idx: int = 0,
    num_antennas: int = 64,
    num_freq: int = LOCAL_FREQUENCY_CHANNELS,
    samples_per_frame: int = 1536,
) -> Tuple[np.ndarray, Dict[str, any]]:
    """
    Extracts 1 frame of baseband data from a Kotekan rawFileRead / rawFileWrite binary file.

    Returns:
        frame_packed: np.ndarray shape (64, num_freq, samples_per_frame) uint8
        meta: dict of metadata
    """
    payload_bytes_per_frame = samples_per_frame * num_freq * num_antennas
    file_size = os.path.getsize(bin_path)

    with open(bin_path, "rb") as f:
        # Check for Kotekan standard raw file header: [uint32 metadata_size]
        meta_size_bytes = f.read(4)
        if len(meta_size_bytes) < 4:
            raise ValueError(f"File {bin_path} is too small to contain a header.")

        meta_size = int(np.frombuffer(meta_size_bytes, dtype="<u4", count=1)[0])
        frame_stride = 4 + meta_size + payload_bytes_per_frame

        if file_size < frame_stride:
            # Maybe pure payload without headers
            if file_size >= payload_bytes_per_frame:
                frame_stride = payload_bytes_per_frame
                f.seek(frame_idx * frame_stride)
            else:
                raise ValueError(
                    f"Binary file {bin_path} size ({file_size} bytes) is smaller than one frame "
                    f"({payload_bytes_per_frame} bytes)."
                )
        else:
            # Kotekan frame format with header
            seek_pos = frame_idx * frame_stride + 4 + meta_size
            if seek_pos + payload_bytes_per_frame > file_size:
                raise ValueError(
                    f"frame_idx {frame_idx} out of range in binary file {bin_path} (file size {file_size})."
                )
            f.seek(seek_pos)

        raw_data = np.fromfile(f, dtype=np.uint8, count=payload_bytes_per_frame)

    if raw_data.size != payload_bytes_per_frame:
        raise ValueError(
            f"Read {raw_data.size} bytes, expected {payload_bytes_per_frame} bytes for frame {frame_idx}."
        )

    shaped = raw_data.reshape(samples_per_frame, num_freq, num_antennas)
    frame_packed = np.transpose(shaped, (2, 1, 0))

    meta = {
        "num_antennas": num_antennas,
        "num_freq": num_freq,
        "samples_per_frame": samples_per_frame,
        "freq_start_MHz": DEFAULT_FREQUENCY_START_MHZ,
        "delta_freq_MHz": CHARTS_CHANNEL_WIDTH_MHZ,
        "delta_time_us": FPGA_TIME_RESOLUTION_US,
        "scenario": bin_path.stem,
        "target_name": bin_path.stem,
    }

    return frame_packed, meta


def load_baseband_frame(
    file_path: Path,
    frame_idx: int = 0,
    samples_per_frame: int = 1536,
) -> Tuple[np.ndarray, Dict[str, any]]:
    """Loads 1 frame of baseband data from either .h5 or .bin file."""
    if file_path.suffix.lower() in [".h5", ".hdf5"]:
        return load_frame_from_hdf5(
            file_path, frame_idx=frame_idx, samples_per_frame=samples_per_frame
        )
    elif file_path.suffix.lower() in [".bin", ".raw"]:
        return load_frame_from_raw_bin(
            file_path, frame_idx=frame_idx, samples_per_frame=samples_per_frame
        )
    else:
        if HAS_H5PY:
            try:
                return load_frame_from_hdf5(
                    file_path, frame_idx=frame_idx, samples_per_frame=samples_per_frame
                )
            except Exception:
                pass
        return load_frame_from_raw_bin(
            file_path, frame_idx=frame_idx, samples_per_frame=samples_per_frame
        )


# =============================================================================
# Time Window Slicing Helper
# =============================================================================


def slice_time_window(
    power_cube: np.ndarray,
    meta: Dict[str, any],
    frame_idx: int = 0,
    duration_ms: Optional[float] = 0.5,
    start_time_ms: float = 0.0,
) -> Tuple[np.ndarray, float, float, int, int]:
    """
    Slices power_cube (64, num_freq, num_time) to the specified time window.

    Returns:
        sliced_cube: np.ndarray shape (64, num_freq, window_samples)
        actual_t_start_ms: start time in ms
        actual_t_end_ms: end time in ms
        start_sample: starting index
        end_sample: ending index
    """
    n_ant, n_freq, n_time = power_cube.shape
    dt_us = float(meta.get("delta_time_us", FPGA_TIME_RESOLUTION_US))
    frame_base_ms = frame_idx * (n_time * dt_us / 1000.0)

    start_sample = max(0, int(round((start_time_ms * 1000.0) / dt_us)))
    if duration_ms is not None and duration_ms > 0:
        n_samples = int(round((duration_ms * 1000.0) / dt_us))
        end_sample = min(start_sample + n_samples, n_time)
    else:
        end_sample = n_time

    if start_sample >= n_time:
        start_sample = 0
        end_sample = n_time

    sliced_cube = power_cube[:, :, start_sample:end_sample]
    actual_t_start_ms = frame_base_ms + (start_sample * dt_us / 1000.0)
    actual_t_end_ms = frame_base_ms + (end_sample * dt_us / 1000.0)

    return sliced_cube, actual_t_start_ms, actual_t_end_ms, start_sample, end_sample


# =============================================================================
# 1. 8x8 Dynamic Spectrogram Visualizer (2D: Time vs Frequency)
# =============================================================================


def plot_8x8_antenna_spectrograms(
    power_cube: np.ndarray,
    meta: Dict[str, any],
    output_png: Path,
    frame_idx: int = 0,
    duration_ms: Optional[float] = 0.5,
    start_time_ms: float = 0.0,
    use_db: bool = True,
    cmap_name: str = "inferno",
    downsample_factor: int = 1,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> Path:
    """
    Renders an 8x8 matrix of dynamic spectra (spectrograms: time vs frequency)
    corresponding to the physical layout of the CHARTS 64-antenna array.
    """
    sliced_cube, t_start_ms, t_end_ms, start_sample, end_sample = slice_time_window(
        power_cube,
        meta,
        frame_idx=frame_idx,
        duration_ms=duration_ms,
        start_time_ms=start_time_ms,
    )
    n_ant, n_freq, n_time = sliced_cube.shape
    dt_us = float(meta.get("delta_time_us", FPGA_TIME_RESOLUTION_US))
    df_mhz = float(meta.get("delta_freq_MHz", CHARTS_CHANNEL_WIDTH_MHZ))
    f_start_mhz = float(meta.get("freq_start_MHz", DEFAULT_FREQUENCY_START_MHZ))
    f_end_mhz = f_start_mhz + n_freq * df_mhz

    # Optional downsample in time
    if downsample_factor > 1 and n_time >= downsample_factor:
        truncate_len = (n_time // downsample_factor) * downsample_factor
        sliced_cube = (
            sliced_cube[:, :, :truncate_len]
            .reshape(
                n_ant, n_freq, truncate_len // downsample_factor, downsample_factor
            )
            .mean(axis=-1)
        )
        n_time = sliced_cube.shape[-1]

    # Convert to dB scale if requested
    if use_db:
        disp_cube = 10.0 * np.log10(sliced_cube + 1.0)
        unit_label = "Power [dB]"
    else:
        disp_cube = sliced_cube
        unit_label = "Power [linear]"

    # Saturated feeds identification
    saturated_antennas = set(meta.get("saturated_antennas", []))
    if not saturated_antennas:
        mean_p = sliced_cube.mean(axis=(1, 2))
        sat_indices = np.where(mean_p > 45.0)[0]
        saturated_antennas = set(sat_indices.tolist())

    normal_ant_mask = [a for a in range(64) if a not in saturated_antennas]
    if len(normal_ant_mask) == 0:
        normal_ant_mask = list(range(64))

    normal_data = disp_cube[normal_ant_mask]
    if vmin is None:
        vmin = float(np.percentile(normal_data, 2.0))
    if vmax is None:
        vmax = float(np.percentile(disp_cube, 99.8))
        if vmax <= vmin:
            vmax = vmin + 10.0

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
            "left": 0.05,
            "right": 0.90,
            "top": 0.92,
            "bottom": 0.05,
        },
    )

    extent = [t_start_ms, t_end_ms, f_start_mhz, f_end_mhz]

    img_last = None
    for a in range(64):
        col = a & 7  # 0 to 7 (West to East)
        row = a >> 3  # 0 to 7 (South to North)

        plot_row = 7 - row  # Row 7 is North at top
        plot_col = col  # Col 0 is West at left

        ax = axes[plot_row, plot_col]
        spec_data = disp_cube[a]

        img = ax.imshow(
            spec_data,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap=cmap_name,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        img_last = img

        is_sat = a in saturated_antennas

        if is_sat:
            for spine in ax.spines.values():
                spine.set_edgecolor("#ef4444")
                spine.set_linewidth(2.8)
            badge_text = f"A{a:02d} [SAT]"
            badge_bg = "#7f1d1d"
            badge_color = "#fef08a"
        else:
            for spine in ax.spines.values():
                spine.set_edgecolor("#334155")
                spine.set_linewidth(1.0)
            badge_text = f"A{a:02d} [r{row},c{col}]"
            badge_bg = "#0f172a"
            badge_color = "#38bdf8"

        ax.text(
            0.04,
            0.92,
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

    # Axis labels on outer margins
    for c in range(8):
        axes[7, c].set_xlabel("Time [ms]", fontsize=9, color="#e2e8f0", labelpad=3)
    for r in range(8):
        axes[r, 0].set_ylabel("Freq [MHz]", fontsize=9, color="#e2e8f0", labelpad=3)

    # Master Colorbar on the right
    cbar_ax = fig.add_axes([0.915, 0.10, 0.015, 0.78])
    cbar = fig.colorbar(img_last, cax=cbar_ax)
    cbar.set_label(
        f"Spectrogram Intensity ({unit_label})",
        fontsize=11,
        color="#f8fafc",
        labelpad=10,
    )
    cbar.ax.tick_params(labelsize=9, colors="#94a3b8")

    # Cardinal direction orientation markers
    fig.text(
        0.475,
        0.942,
        "[^] NORTH (+Y, Row 7)",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="#38bdf8",
    )
    fig.text(
        0.475,
        0.022,
        "[v] SOUTH (-Y, Row 0)",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="#38bdf8",
    )
    fig.text(
        0.012,
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
        0.895,
        0.485,
        "EAST (+X, Col 7) [>]",
        ha="center",
        va="center",
        rotation=-90,
        fontsize=11,
        fontweight="bold",
        color="#38bdf8",
    )

    # Master Title and Header Info
    target_name = meta.get("target_name", meta.get("scenario", "Simulation"))
    scenario = meta.get("scenario", "CHARTS-64")
    total_samples = end_sample - start_sample
    duration_val = (total_samples * dt_us) / 1000.0

    fig.suptitle(
        f"CHARTS 64-Antenna Baseband Spectrogram Matrix (8x8 Layout) - {target_name}\n"
        f"Scenario: {scenario} | Window: {t_start_ms:.2f} to {t_end_ms:.2f} ms ({duration_val:.2f} ms, "
        f"{total_samples} samples @ {dt_us:.2f} us) | Band: {f_start_mhz:.1f} - {f_end_mhz:.1f} MHz ({n_freq} ch) | "
        f"Site: Caren ({CHARTS_LATITUDE_DEG:.4f} deg, {CHARTS_LONGITUDE_DEG:.4f} deg)",
        fontsize=13,
        fontweight="bold",
        color="#ffffff",
        y=0.98,
    )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=180, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)

    print(f"[OK] Saved 8x8 antenna spectrogram dashboard: {output_png}")
    return output_png


# =============================================================================
# 2. 8x8 Antenna Frequency Spectrum Shape Visualizer (1D: Power vs Freq)
# =============================================================================


def plot_8x8_antenna_spectra(
    power_cube: np.ndarray,
    meta: Dict[str, any],
    output_png: Path,
    frame_idx: int = 0,
    duration_ms: Optional[float] = 0.5,
    start_time_ms: float = 0.0,
    use_db: bool = True,
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
) -> Path:
    """
    Renders an 8x8 matrix of 1D frequency spectrum shapes P(f) averaged over
    the specified time window (default 0.5 ms = 150 time samples) for all 64 antennas.
    Reveals bandpass shape, channel gain profiles, and noise levels across the array.
    """
    sliced_cube, t_start_ms, t_end_ms, start_sample, end_sample = slice_time_window(
        power_cube,
        meta,
        frame_idx=frame_idx,
        duration_ms=duration_ms,
        start_time_ms=start_time_ms,
    )
    n_ant, n_freq, n_time = sliced_cube.shape
    dt_us = float(meta.get("delta_time_us", FPGA_TIME_RESOLUTION_US))
    df_mhz = float(meta.get("delta_freq_MHz", CHARTS_CHANNEL_WIDTH_MHZ))
    f_start_mhz = float(meta.get("freq_start_MHz", DEFAULT_FREQUENCY_START_MHZ))
    f_end_mhz = f_start_mhz + n_freq * df_mhz
    freqs_mhz = np.linspace(f_start_mhz, f_end_mhz, n_freq)

    # Average power over time window to get clear spectral shape
    mean_power = sliced_cube.mean(axis=-1)  # shape (64, n_freq)

    if use_db:
        spec_data = 10.0 * np.log10(mean_power + 1.0)
        unit_label = "Power [dB]"
    else:
        spec_data = mean_power
        unit_label = "Power [linear]"

    saturated_antennas = set(meta.get("saturated_antennas", []))
    if not saturated_antennas:
        sat_indices = np.where(mean_power.mean(axis=1) > 45.0)[0]
        saturated_antennas = set(sat_indices.tolist())

    normal_mask = [a for a in range(64) if a not in saturated_antennas]
    if not normal_mask:
        normal_mask = list(range(64))

    if ymin is None:
        ymin = max(0.0, float(np.percentile(spec_data[normal_mask], 1.0)) - 1.5)
    if ymax is None:
        ymax = float(np.percentile(spec_data, 99.8)) + 2.0
        if ymax <= ymin:
            ymax = ymin + 10.0

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
            "left": 0.05,
            "right": 0.95,
            "top": 0.92,
            "bottom": 0.05,
        },
    )

    for a in range(64):
        col = a & 7
        row = a >> 3
        plot_row = 7 - row
        plot_col = col

        ax = axes[plot_row, plot_col]
        is_sat = a in saturated_antennas

        line_color = "#ef4444" if is_sat else "#38bdf8"
        line_width = 1.4 if is_sat else 1.1

        ax.plot(freqs_mhz, spec_data[a], color=line_color, linewidth=line_width)
        ax.set_ylim(ymin, ymax)
        ax.grid(True, linestyle="--", color="#334155", alpha=0.5)

        if is_sat:
            for spine in ax.spines.values():
                spine.set_edgecolor("#ef4444")
                spine.set_linewidth(2.2)
            badge_text = f"A{a:02d} [SAT]"
            badge_bg = "#7f1d1d"
            badge_color = "#fef08a"
        else:
            for spine in ax.spines.values():
                spine.set_edgecolor("#334155")
                spine.set_linewidth(1.0)
            badge_text = f"A{a:02d} [r{row},c{col}]"
            badge_bg = "#0f172a"
            badge_color = "#38bdf8"

        ax.text(
            0.04,
            0.92,
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

    for c in range(8):
        axes[7, c].set_xlabel("Freq [MHz]", fontsize=9, color="#e2e8f0", labelpad=3)
    for r in range(8):
        axes[r, 0].set_ylabel(
            f"Spectrum ({unit_label})", fontsize=9, color="#e2e8f0", labelpad=3
        )

    # Cardinal direction orientation markers
    fig.text(
        0.50,
        0.942,
        "[^] NORTH (+Y, Row 7)",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="#38bdf8",
    )
    fig.text(
        0.50,
        0.022,
        "[v] SOUTH (-Y, Row 0)",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="#38bdf8",
    )
    fig.text(
        0.012,
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
        0.975,
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
    total_samples = end_sample - start_sample
    duration_val = (total_samples * dt_us) / 1000.0

    fig.suptitle(
        f"CHARTS 64-Antenna Frequency Spectrum Shape P(f) (8x8 Layout) - {target_name}\n"
        f"Scenario: {scenario} | Time Window: {t_start_ms:.2f} to {t_end_ms:.2f} ms ({duration_val:.2f} ms, "
        f"{total_samples} samples @ {dt_us:.2f} us) | Band: {f_start_mhz:.1f} - {f_end_mhz:.1f} MHz ({n_freq} ch) | "
        f"Site: Caren ({CHARTS_LATITUDE_DEG:.4f} deg, {CHARTS_LONGITUDE_DEG:.4f} deg)",
        fontsize=13,
        fontweight="bold",
        color="#ffffff",
        y=0.98,
    )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=180, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)

    print(f"[OK] Saved 8x8 antenna spectrum shape dashboard: {output_png}")
    return output_png


# =============================================================================
# 3. Detailed Single-Antenna Zoom Mode
# =============================================================================


def plot_single_antenna_detail(
    power_cube: np.ndarray,
    antenna_idx: int,
    meta: Dict[str, any],
    output_png: Path,
    frame_idx: int = 0,
    duration_ms: Optional[float] = 0.5,
    start_time_ms: float = 0.0,
    cmap_name: str = "inferno",
) -> Path:
    """
    Renders an in-depth 3-panel diagnostic plot for a specific antenna:
      Panel 1: Full 2D Dynamic Spectrum (Spectrogram: Time vs Freq).
      Panel 2: Time-averaged Frequency Bandpass Spectrum P(f).
      Panel 3: Frequency-averaged Time Lightcurve P(t).
    """
    sliced_cube, t_start_ms, t_end_ms, start_sample, end_sample = slice_time_window(
        power_cube,
        meta,
        frame_idx=frame_idx,
        duration_ms=duration_ms,
        start_time_ms=start_time_ms,
    )
    n_ant, n_freq, n_time = sliced_cube.shape
    dt_us = float(meta.get("delta_time_us", FPGA_TIME_RESOLUTION_US))
    df_mhz = float(meta.get("delta_freq_MHz", CHARTS_CHANNEL_WIDTH_MHZ))
    f_start_mhz = float(meta.get("freq_start_MHz", DEFAULT_FREQUENCY_START_MHZ))
    f_end_mhz = f_start_mhz + n_freq * df_mhz

    col = antenna_idx & 7
    row = antenna_idx >> 3

    ant_power = sliced_cube[antenna_idx]  # shape (n_freq, n_time)
    ant_db = 10.0 * np.log10(ant_power + 1.0)

    time_ms = np.linspace(t_start_ms, t_end_ms, n_time)
    freqs_mhz = np.linspace(f_start_mhz, f_end_mhz, n_freq)

    # 1D projections
    spectrum_db = ant_db.mean(axis=1)  # Average over time -> spectrum shape
    lightcurve_db = ant_db.mean(axis=0)  # Average over freq -> light curve

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(
        2, 2, width_ratios=[3, 1], height_ratios=[1, 3], hspace=0.15, wspace=0.15
    )

    ax_lightcurve = fig.add_subplot(gs[0, 0])
    ax_spec2d = fig.add_subplot(gs[1, 0], sharex=ax_lightcurve)
    ax_bandpass = fig.add_subplot(gs[1, 1], sharey=ax_spec2d)
    ax_stat = fig.add_subplot(gs[0, 1])
    ax_stat.axis("off")

    # Panel 1: 2D Spectrogram
    extent = [t_start_ms, t_end_ms, f_start_mhz, f_end_mhz]
    vmin, vmax = np.percentile(ant_db, 1.0), np.percentile(ant_db, 99.5)
    img = ax_spec2d.imshow(
        ant_db,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap=cmap_name,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    ax_spec2d.set_xlabel("Time [ms]", fontsize=11, color="#f8fafc")
    ax_spec2d.set_ylabel("Frequency [MHz]", fontsize=11, color="#f8fafc")
    ax_spec2d.tick_params(colors="#94a3b8", labelsize=9)

    # Panel 2: Lightcurve (top)
    ax_lightcurve.plot(time_ms, lightcurve_db, color="#38bdf8", linewidth=1.2)
    ax_lightcurve.set_ylabel("Power [dB]", fontsize=10, color="#f8fafc")
    ax_lightcurve.grid(True, linestyle="--", color="#334155", alpha=0.5)
    ax_lightcurve.tick_params(labelbottom=False, colors="#94a3b8", labelsize=9)
    ax_lightcurve.set_title(
        f"Frequency-Averaged Lightcurve P(t)", fontsize=10, color="#38bdf8"
    )

    # Panel 3: Bandpass Spectrum (right)
    ax_bandpass.plot(spectrum_db, freqs_mhz, color="#a855f7", linewidth=1.2)
    ax_bandpass.set_xlabel("Power [dB]", fontsize=10, color="#f8fafc")
    ax_bandpass.grid(True, linestyle="--", color="#334155", alpha=0.5)
    ax_bandpass.tick_params(labelleft=False, colors="#94a3b8", labelsize=9)
    ax_bandpass.set_title(
        f"Time-Averaged Spectrum Shape P(f)", fontsize=10, color="#a855f7"
    )

    # Panel 4: Metadata and Statistics
    is_sat = antenna_idx in meta.get("saturated_antennas", DEFAULT_SATURATED_ANTENNAS)
    status_str = "SATURATED / ADC CLIPPED" if is_sat else "OPERATIONAL"
    status_color = "#ef4444" if is_sat else "#22c55e"

    stats_text = (
        f"Antenna ID: #{antenna_idx:02d}\n"
        f"Grid Pos: Row {row} (Y), Col {col} (X)\n"
        f"Status: {status_str}\n"
        f"Mean Power: {ant_power.mean():.2f} ({ant_db.mean():.2f} dB)\n"
        f"Max Power: {ant_power.max():.2f} ({ant_db.max():.2f} dB)\n"
        f"Min Power: {ant_power.min():.2f} ({ant_db.min():.2f} dB)\n"
        f"Std Dev: {ant_power.std():.2f}\n"
        f"Scenario: {meta.get('scenario', 'N/A')}\n"
        f"Time Window: {t_start_ms:.2f} - {t_end_ms:.2f} ms ({t_end_ms - t_start_ms:.2f} ms)\n"
        f"Frame: #{frame_idx}"
    )
    ax_stat.text(
        0.05,
        0.90,
        stats_text,
        transform=ax_stat.transAxes,
        fontsize=10,
        family="monospace",
        color="#e2e8f0",
        verticalalignment="top",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="#1e293b",
            edgecolor=status_color,
            linewidth=1.5,
        ),
    )

    fig.suptitle(
        f"CHARTS Antenna #{antenna_idx:02d} Spectrogram & Dynamic Profile - {meta.get('target_name', 'Simulation')}",
        fontsize=13,
        fontweight="bold",
        color="#ffffff",
        y=0.98,
    )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=180, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)

    print(f"[OK] Saved single-antenna detail plot: {output_png}")
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
        description="Visualize 8x8 physical spectrogram and spectrum shape matrix for CHARTS 64-antenna baseband dumps.",
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
        help="Scenario name to find in --baseband-dir if --input is omitted (default: vela_with_noise).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output PNG file path (default: auto-named in --plots-dir).",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=Path("./dumps_charts_64ant/plots"),
        help="Directory to store generated plots (default: ./dumps_charts_64ant/plots).",
    )
    parser.add_argument(
        "--duration-ms",
        "-d",
        type=float,
        default=0.5,
        help="Duration of the time window to analyze in ms (default: 0.5 ms). Use 0 for full frame.",
    )
    parser.add_argument(
        "--start-time-ms",
        type=float,
        default=0.0,
        help="Start time offset within frame in ms (default: 0.0 ms).",
    )
    parser.add_argument(
        "--plot-type",
        type=str,
        default="all",
        choices=["all", "spectrogram", "spectrum"],
        help="Plot type to generate: 'spectrogram' (2D time vs freq), 'spectrum' (1D P(f) bandpass shape), or 'all' (both). Default: all.",
    )
    parser.add_argument(
        "--frame",
        "-f",
        type=int,
        default=0,
        help="Frame index to visualize (default: 0).",
    )
    parser.add_argument(
        "--samples-per-frame",
        type=int,
        default=1536,
        help="Number of time samples per frame (default: 1536 = 5.12 ms).",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=1,
        help="Time binning downsample factor (default: 1 = no downsampling).",
    )
    parser.add_argument(
        "--linear",
        action="store_true",
        help="Plot linear power instead of logarithmic dB scale.",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="inferno",
        help="Matplotlib colormap (e.g. inferno, viridis, magma, plasma, cividis).",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=None,
        help="Optional minimum value for colorbar scale.",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Optional maximum value for colorbar scale.",
    )
    parser.add_argument(
        "--antenna",
        "-a",
        type=int,
        default=None,
        help="Antenna index (0-63) to generate an additional single-antenna zoom diagnostic plot.",
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

        files_to_process = sorted(
            list(args.baseband_dir.glob("*.h5")) + list(args.baseband_dir.glob("*.bin"))
        )
        if not files_to_process:
            print(f"[WARN] No .h5 or .bin files found in {args.baseband_dir}")
            sys.exit(0)

        print(
            f"=== Found {len(files_to_process)} baseband dumps in {args.baseband_dir} ==="
        )
        print(
            f"=== Time window: {args.duration_ms:.2f} ms (offset: {args.start_time_ms:.2f} ms), plot type: {args.plot_type} ==="
        )
        args.plots_dir.mkdir(parents=True, exist_ok=True)

        for filepath in files_to_process:
            stem = filepath.stem.replace("_64ant_5ms", "")
            print(f"\nProcessing {filepath.name} ...")
            try:
                t0 = time.perf_counter()
                frame_packed, meta = load_baseband_frame(
                    filepath,
                    frame_idx=args.frame,
                    samples_per_frame=args.samples_per_frame,
                )
                power_cube = unpack_int4x2_power(frame_packed)

                if args.plot_type in ["all", "spectrogram"]:
                    out_spec = args.plots_dir / f"spectrogram_8x8_{stem}.png"
                    plot_8x8_antenna_spectrograms(
                        power_cube=power_cube,
                        meta=meta,
                        output_png=out_spec,
                        frame_idx=args.frame,
                        duration_ms=args.duration_ms,
                        start_time_ms=args.start_time_ms,
                        use_db=not args.linear,
                        cmap_name=args.cmap,
                        downsample_factor=args.downsample,
                        vmin=args.vmin,
                        vmax=args.vmax,
                    )

                if args.plot_type in ["all", "spectrum"]:
                    out_shape = args.plots_dir / f"spectrum_shape_8x8_{stem}.png"
                    plot_8x8_antenna_spectra(
                        power_cube=power_cube,
                        meta=meta,
                        output_png=out_shape,
                        frame_idx=args.frame,
                        duration_ms=args.duration_ms,
                        start_time_ms=args.start_time_ms,
                        use_db=not args.linear,
                    )

                dt = (time.perf_counter() - t0) * 1000.0
                print(f" Finished in {dt:.1f} ms.")
            except Exception as e:
                print(f"[ERROR] Failed to process {filepath}: {e}")

        print("\n[OK] Batch processing complete.")
        return

    # Handle Single File / Scenario Mode
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

    args.plots_dir.mkdir(parents=True, exist_ok=True)
    stem = target_file.stem.replace("_64ant_5ms", "")

    print(f"Loading frame #{args.frame} from: {target_file}")
    t0 = time.perf_counter()
    frame_packed, meta = load_baseband_frame(
        target_file, frame_idx=args.frame, samples_per_frame=args.samples_per_frame
    )
    decode_t0 = time.perf_counter()
    power_cube = unpack_int4x2_power(frame_packed)
    decode_ms = (time.perf_counter() - decode_t0) * 1000.0

    print(
        f"Decoded {power_cube.shape[0]} antennas x {power_cube.shape[1]} channels x "
        f"{power_cube.shape[2]} samples in {decode_ms:.1f} ms."
    )

    # 1. Generate 2D dynamic spectrogram if requested
    if args.plot_type in ["all", "spectrogram"]:
        if args.output and args.plot_type == "spectrogram":
            out_spec = args.output
        else:
            out_spec = args.plots_dir / f"spectrogram_8x8_{stem}.png"

        plot_8x8_antenna_spectrograms(
            power_cube=power_cube,
            meta=meta,
            output_png=out_spec,
            frame_idx=args.frame,
            duration_ms=args.duration_ms,
            start_time_ms=args.start_time_ms,
            use_db=not args.linear,
            cmap_name=args.cmap,
            downsample_factor=args.downsample,
            vmin=args.vmin,
            vmax=args.vmax,
        )

    # 2. Generate 1D spectrum shape if requested
    if args.plot_type in ["all", "spectrum"]:
        if args.output and args.plot_type == "spectrum":
            out_shape = args.output
        else:
            out_shape = args.plots_dir / f"spectrum_shape_8x8_{stem}.png"

        plot_8x8_antenna_spectra(
            power_cube=power_cube,
            meta=meta,
            output_png=out_shape,
            frame_idx=args.frame,
            duration_ms=args.duration_ms,
            start_time_ms=args.start_time_ms,
            use_db=not args.linear,
        )

    # 3. Single-antenna zoom mode if requested
    if args.antenna is not None:
        if not (0 <= args.antenna < power_cube.shape[0]):
            print(
                f"[WARN] Antenna index {args.antenna} is out of bounds (0-63). Skipping zoom plot."
            )
        else:
            zoom_png = args.plots_dir / f"spectrogram_ant{args.antenna:02d}_{stem}.png"
            plot_single_antenna_detail(
                power_cube=power_cube,
                antenna_idx=args.antenna,
                meta=meta,
                output_png=zoom_png,
                frame_idx=args.frame,
                duration_ms=args.duration_ms,
                start_time_ms=args.start_time_ms,
                cmap_name=args.cmap,
            )

    total_s = time.perf_counter() - t0
    print(f"[OK] Done in {total_s:.2f} s.")


if __name__ == "__main__":
    main()
