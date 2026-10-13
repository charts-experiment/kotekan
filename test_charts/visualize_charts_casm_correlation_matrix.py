#!/usr/bin/env python3
"""
CHARTS CASM-Style Correlation Matrix Waterfall Visualizer
=========================================================
Generates an upper-triangular correlation matrix visualization matching
CASM / CHIME Figure 11:
  - Diagonals (i == j): 1D unflattened auto-spectrum P(f) vs frequency
    showing the antenna bandpass profile and power levels.
  - Off-Diagonals (i < j): 2D waterfall image of the visibility real-component
    Re(V_ij(t, f)) as a function of frequency (vertical axis) and time (horizontal axis).
  - Fringe Visualization: Striking interferometric fringe stripes cos(2*pi*b*s/lambda)
    sweeping across frequency and time, with cross-talk patterns visible on short baselines.
  - Baseline Headers: Column headers showing baseline separation vectors (dx, dy) in meters.

Data Inputs Supported:
  1. Correlator Dumps: Reads a directory of Kotekan cudaCorrelatorAstron binary files (corr_*.bin)
     and reconstructs the time-frequency visibility cube V_ij(t, f).
  2. Baseband Dumps: Reads an HDF5 (.h5) or raw binary (.bin) baseband file and computes
     short-time cross-correlations across time chunks.
  3. Analytical / Transit Simulation: Computes the exact physical celestial transit
     visibilities for the CHARTS 64-antenna array (Carén Observatory) with geometric delays,
     sidereal fringe rates, thermal noise, and short-baseline mutual coupling cross-talk.

Usage Examples:
  # 1. 16-Antenna CASM Matrix (Matching Figure 11 in paper) from correlator dumps:
  python test_charts/visualize_charts_casm_correlation_matrix.py \\
      --corr-dir ./dumps_charts_64ant/correlator \\
      --num-antennas 16 \\
      --output ./dumps_charts_64ant/plots/casm_correlation_matrix_16ant.png

  # 2. 8-Antenna CASM Matrix from baseband HDF5 dump:
  python test_charts/visualize_charts_casm_correlation_matrix.py \\
      --baseband ./dumps_charts_64ant/baseband/sun_with_noise_saturated_64ant_5ms.h5 \\
      --num-antennas 8

  # 3. Analytical Solar Transit Simulation with cross-talk:
  python test_charts/visualize_charts_casm_correlation_matrix.py \\
      --scenario sun_with_noise \\
      --num-antennas 16 \\
      --cross-talk

  # 4. Using Trillium launcher:
  bash test_charts/run_visualize_casm.sh --scenario sun_with_noise --num-antennas 16
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
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as e:
    sys.stderr.write(
        f"\n[ERROR] Missing required Python scientific stack: {e}\n\n"
        "On Trillium / Compute Canada, please load the scientific environment:\n\n"
        "    module load python/3.11 scipy-stack\n\n"
        "Or use the automated launcher:\n\n"
        "    bash test_charts/run_visualize_casm.sh [options]\n\n"
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
    C_LIGHT,
    CHARTS_CHANNEL_WIDTH_MHZ,
    CHARTS_LATITUDE_DEG,
    CHARTS_LONGITUDE_DEG,
    DEFAULT_FREQUENCY_START_MHZ,
    DEFAULT_SPACING_M,
    FPGA_TIME_RESOLUTION_US,
    LOCAL_FREQUENCY_CHANNELS,
)
from generate_charts_64ant_dumps import (
    ASTRONOMICAL_CATALOG,
    DEFAULT_SATURATED_ANTENNAS,
    compute_transit_direction_cosines,
    get_charts_64_antenna_positions,
)
from inspect_correlator_dump import load_astron_correlator_dump

# 4-bit LUT for baseband decoding
INT4_LUT = np.array(
    [0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1],
    dtype=np.float64,
)


# =============================================================================
# Antenna Subset Selection for CASM Matrix
# =============================================================================


def select_representative_antennas(
    total_antennas: int = 64,
    num_selected: int = 16,
    saturated_antennas: List[int] = DEFAULT_SATURATED_ANTENNAS,
) -> List[int]:
    """
    Selects a representative subset of antennas from the 8x8 grid that spans:
      - Short baselines (0.6m, 1.2m, 1.8m)
      - East-West baselines (dy = 0)
      - North-South baselines (dx = 0)
      - Diagonal baselines (dx > 0, dy > 0)
      - Saturated / flagged feeds to showcase non-linear cross-talk and decorrelation
    """
    if num_selected >= total_antennas:
        return list(range(total_antennas))

    if num_selected == 8:
        # 8 antennas: 4 along Row 0 (col 0, 1, 2, 3) + 2 along Col 0 (row 1, 2) + diagonal + saturated
        candidates = [0, 1, 2, 3, 8, 16, 24, 7]
        return sorted(list(set(candidates)))[:8]

    if num_selected == 16:
        # 16 antennas: 2x8 subarray or L-shape + cross-array
        # Antennas 0..7 (Row 0, West-East line of 8 feeds)
        # Antennas 8, 16, 24, 32, 40, 48, 56 (Col 0, South-North line)
        # Antenna 63 (opposite corner)
        candidates = [0, 1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 32, 40, 48, 56, 63]
        return sorted(list(set(candidates)))[:16]

    # General step-based selection
    step = max(1, total_antennas // num_selected)
    selected = [i * step for i in range(num_selected)]
    return selected


# =============================================================================
# Data Loaders: Correlator Dumps, Baseband Dumps, and Physical Simulation
# =============================================================================


def load_visibilities_from_correlator_dir(
    corr_dir: Path,
    num_elements: int = 64,
    num_channels: int = 672,
    max_frames: Optional[int] = None,
    max_time_samples: int = 150,
    duration_s: float = 60.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads sequential correlator dumps from corr_dir, sampling evenly across time.
    Returns:
        vis_cube: shape (num_times, num_channels, num_elements, num_elements) complex128
        time_s: shape (num_times,)
        freq_mhz: shape (num_channels,)
    """
    bin_files = sorted(corr_dir.glob("*.bin"))
    if not bin_files:
        raise FileNotFoundError(f"No correlator .bin files found in {corr_dir}")

    total_files = len(bin_files)
    limit_samples = max_frames if max_frames else max_time_samples
    target_count = min(total_files, limit_samples)
    indices = np.linspace(0, total_files - 1, target_count).astype(int)
    sampled_files = [bin_files[i] for i in indices]

    num_times = len(sampled_files)
    print(
        f"Loading {num_times} sampled correlator dumps (out of {total_files} total) from {corr_dir} ..."
    )

    vis_cube = np.zeros(
        (num_times, num_channels, num_elements, num_elements), dtype=np.complex128
    )
    for t_idx, fpath in enumerate(sampled_files):
        vis = load_astron_correlator_dump(
            fpath, num_elements=num_elements, num_channels=num_channels
        )
        vis_cube[t_idx] = vis

    time_s = np.linspace(0.0, duration_s, num_times)
    freq_mhz = (
        DEFAULT_FREQUENCY_START_MHZ + np.arange(num_channels) * CHARTS_CHANNEL_WIDTH_MHZ
    )
    return vis_cube, time_s, freq_mhz


def load_visibilities_from_baseband(
    baseband_path: Path,
    num_time_chunks: int = 64,
    samples_per_frame: int = 1536,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads baseband voltages and computes short-time cross-correlations V_ij(t, f).
    Returns:
        vis_cube: shape (num_time_chunks, num_channels, 64, 64) complex128
        time_s: shape (num_time_chunks,)
        freq_mhz: shape (num_channels,)
    """
    print(f"Computing short-time cross-correlations from baseband: {baseband_path} ...")

    if baseband_path.suffix.lower() in [".h5", ".hdf5"]:
        if not HAS_H5PY:
            raise ImportError("h5py required to read .h5 files.")
        with h5py.File(baseband_path, "r") as f:
            if "baseband" in f:
                dset = f["baseband"]  # (n_ant, n_freq, total_time)
                n_ant, n_freq, total_time = dset.shape
                raw_data = dset[:]
            elif "fengine_compat/voltage" in f:
                dset = f["fengine_compat/voltage"]
                n_ant, _, n_freq, total_time = dset.shape
                raw_data = dset[:, 0, :, :]
            else:
                raise KeyError("No valid baseband dataset found in HDF5.")
            dt_us = float(f.attrs.get("delta_time_us", FPGA_TIME_RESOLUTION_US))
            f_start = float(f.attrs.get("freq_start_MHz", DEFAULT_FREQUENCY_START_MHZ))
            df = float(f.attrs.get("delta_freq_MHz", CHARTS_CHANNEL_WIDTH_MHZ))
    else:
        # RAW binary
        file_size = os.path.getsize(baseband_path)
        payload_frame = samples_per_frame * LOCAL_FREQUENCY_CHANNELS * 64
        num_frames = file_size // payload_frame
        n_ant = 64
        n_freq = LOCAL_FREQUENCY_CHANNELS
        total_time = num_frames * samples_per_frame
        raw_arr = np.fromfile(baseband_path, dtype=np.uint8)
        # Reshape to (total_time, n_freq, n_ant) then transpose
        shaped = raw_arr[: total_time * n_freq * n_ant].reshape(
            total_time, n_freq, n_ant
        )
        raw_data = np.transpose(shaped, (2, 1, 0))
        dt_us = FPGA_TIME_RESOLUTION_US
        f_start = DEFAULT_FREQUENCY_START_MHZ
        df = CHARTS_CHANNEL_WIDTH_MHZ

    # Unpack int4x2 to complex voltages
    r = INT4_LUT[raw_data & 0x0F]
    i = INT4_LUT[(raw_data >> 4) & 0x0F]
    voltages = r + 1j * i  # shape (n_ant, n_freq, total_time)

    # Chunk along time axis to compute correlation waterfalls
    chunk_len = max(1, total_time // num_time_chunks)
    actual_chunks = total_time // chunk_len

    vis_cube = np.zeros((actual_chunks, n_freq, n_ant, n_ant), dtype=np.complex128)
    for c_idx in range(actual_chunks):
        t_slice = slice(c_idx * chunk_len, (c_idx + 1) * chunk_len)
        v_chunk = voltages[:, :, t_slice]  # (n_ant, n_freq, chunk_len)
        # Cross product: V_ij = sum_t (v_i * conj(v_j)) / chunk_len
        # Einsum: 'ift,jft->fij'
        vis_cube[c_idx] = np.einsum("ift,jft->fij", v_chunk, np.conj(v_chunk)) / float(
            chunk_len
        )

    total_duration_s = total_time * dt_us * 1e-6
    time_s = np.linspace(0.0, total_duration_s, actual_chunks)
    freq_mhz = f_start + np.arange(n_freq) * df
    return vis_cube, time_s, freq_mhz


def simulate_transit_visibilities(
    scenario: str = "sun_with_noise",
    num_elements: int = 64,
    num_channels: int = 336,
    num_times: int = 120,
    duration_s: float = 120.0,
    add_crosstalk: bool = True,
    spacing_m: float = DEFAULT_SPACING_M,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulates physical celestial transit visibilities V_ij(t, f) for CHARTS array:
      - Sidereal drift of source across the meridian: l(t), m(t).
      - Geometric phase delays: phi_ij(t, f) = 2*pi*f * (b_ij . s(t)) / c.
      - Mutual coupling cross-talk on short baselines (<= 1.2m).
      - Antenna thermal noise and non-linear ADC clipping on saturated feeds.
    """
    target_key = (
        "sun" if "sun" in scenario else ("vela" if "vela" in scenario else "sgr_a")
    )
    target = ASTRONOMICAL_CATALOG.get(target_key, ASTRONOMICAL_CATALOG["sun"])

    pos_x, pos_y = get_charts_64_antenna_positions(num_elements, spacing_m)
    l0, m0 = compute_transit_direction_cosines(target["dec_deg"], CHARTS_LATITUDE_DEG)
    amp = float(target["nominal_amp"])

    time_s = np.linspace(0.0, duration_s, num_times)
    freq_mhz = (
        DEFAULT_FREQUENCY_START_MHZ + np.arange(num_channels) * CHARTS_CHANNEL_WIDTH_MHZ
    )
    freq_hz = freq_mhz * 1e6
    two_pi = 2.0 * np.pi
    c_inv = 1.0 / C_LIGHT

    # Sidereal rate: Earth rotates at 360 deg / 86164 s = 7.292115e-5 rad/s
    omega_earth = 7.292115e-5
    # Transit drift: l(t) shifts primarily East-West with time
    dl_dt = np.cos(np.radians(target["dec_deg"])) * omega_earth

    vis_cube = np.zeros(
        (num_times, num_channels, num_elements, num_elements), dtype=np.complex128
    )
    rng = np.random.default_rng(42)

    # Base noise sigmas per antenna
    use_noise = "no_noise" not in scenario
    if use_noise:
        sigmas = rng.uniform(0.45, 0.70, size=num_elements)
    else:
        sigmas = np.zeros(num_elements)

    saturate = "saturat" in scenario
    sat_feeds = DEFAULT_SATURATED_ANTENNAS if saturate else []

    # Mutual coupling cross-talk matrix C_ij(f) for short baselines (<= 1.8m)
    crosstalk_matrix = np.zeros(
        (num_channels, num_elements, num_elements), dtype=np.complex128
    )
    if add_crosstalk:
        for i in range(num_elements):
            for j in range(i + 1, num_elements):
                dx = pos_x[j] - pos_x[i]
                dy = pos_y[j] - pos_y[i]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist <= 1.8:
                    # Mutual coupling decreases with distance, produces stationary frequency ripples
                    coupling_amp = 0.25 * amp / (dist + 0.1)
                    phase_xtalk = two_pi * freq_hz * (dist * c_inv)
                    c_val = coupling_amp * np.exp(-1j * phase_xtalk)
                    crosstalk_matrix[:, i, j] = c_val
                    crosstalk_matrix[:, j, i] = np.conj(c_val)

    # Compute visibilities across time
    for t_idx, t_val in enumerate(time_s):
        # Time-dependent direction cosines
        t_offset = t_val - duration_s * 0.5
        cur_l = l0 + dl_dt * t_offset
        cur_m = m0

        # Geometric delays: tau_i = (l*x + m*y) / c
        delays = (cur_l * pos_x + cur_m * pos_y) * c_inv  # shape (num_elements,)

        # Phase per antenna and frequency: (num_channels, num_elements)
        phases = two_pi * freq_hz[:, None] * delays[None, :]
        s_vec = amp * np.exp(-1j * phases)  # shape (num_channels, num_elements)

        # Outer product: V_sig(f, i, j) = s_i(f) * conj(s_j(f))
        v_frame = s_vec[:, :, None] * np.conj(s_vec[:, None, :])

        # Add cross-talk
        if add_crosstalk:
            v_frame += crosstalk_matrix

        # Add thermal noise on auto-correlations and cross-correlations
        if use_noise:
            # Auto-power: noise variance
            for a in range(num_elements):
                v_frame[:, a, a] += (sigmas[a] ** 2) * 20.0
            # Cross-noise fluctuations
            noise_cross = (
                (
                    rng.normal(0, 0.04, size=(num_channels, num_elements, num_elements))
                    + 1j
                    * rng.normal(
                        0, 0.04, size=(num_channels, num_elements, num_elements)
                    )
                )
                * np.outer(sigmas, sigmas)[None, :, :]
                * 5.0
            )
            v_frame += 0.5 * (noise_cross + np.conj(np.swapaxes(noise_cross, 1, 2)))

        # Saturation effects on flagged feeds
        if saturate:
            for bad_ant in sat_feeds:
                v_frame[:, bad_ant, :] *= 3.0
                v_frame[:, :, bad_ant] *= 3.0
                v_frame[:, bad_ant, bad_ant] *= 4.0

        vis_cube[t_idx] = v_frame

    return vis_cube, time_s, freq_mhz


# =============================================================================
# CASM-Style Triangular Matrix Visualizer
# =============================================================================


def plot_casm_correlation_matrix(
    vis_cube: np.ndarray,
    time_s: np.ndarray,
    freq_mhz: np.ndarray,
    selected_antennas: List[int],
    output_png: Path,
    clip_sigma: float = 2.5,
    clip_percentile: Optional[float] = None,
    dark_mode: bool = False,
    title: Optional[str] = None,
    scenario_name: str = "CHARTS-64",
    spacing_m: float = DEFAULT_SPACING_M,
) -> Path:
    """
    Renders an upper-triangular correlation matrix matching CASM Figure 11:
      - Diagonals: 1D auto-spectrum plot P(f) vs frequency.
      - Off-diagonals: 2D waterfall image of clipped Re(V_ij(t, f)).
      - Column headers: baseline displacement vector (dx, dy) in meters.
    """
    N = len(selected_antennas)
    pos_x, pos_y = get_charts_64_antenna_positions(64, spacing_m)

    # Style configuration: white paper style (matching Figure 11) or dark mode
    if dark_mode:
        bg_color = "#0a0a1a"
        panel_bg = "#111122"
        text_color = "#ffffff"
        sub_text_color = "#94a3b8"
        line_color = "#38bdf8"
        grid_color = "#334155"
        cmap_waterfall = "gray"
    else:
        bg_color = "#ffffff"
        panel_bg = "#ffffff"
        text_color = "#000000"
        sub_text_color = "#333333"
        line_color = "#000000"
        grid_color = "#cccccc"
        cmap_waterfall = "Greys"

    # Compute figure dimensions based on N
    # For N=16: 22x22 inches; for N=8: 16x16 inches
    fig_size = max(14, int(N * 1.35))
    fig = plt.figure(figsize=(fig_size, fig_size), facecolor=bg_color)

    # Create subplots with tight spacing matching CASM Figure 11
    gs = fig.add_gridspec(
        nrows=N,
        ncols=N,
        wspace=0.04,
        hspace=0.04,
        left=0.06,
        right=0.96,
        top=0.92,
        bottom=0.07,
    )

    t_min, t_max = time_s[0], time_s[-1]
    f_min, f_max = freq_mhz[0], freq_mhz[-1]
    extent = [t_min, t_max, f_min, f_max]

    # Precompute time-averaged auto-spectra for diagonal plots
    # Shape of vis_cube: (num_times, num_channels, 64, 64)
    auto_spectra = {}
    for idx_a, ant_a in enumerate(selected_antennas):
        # Auto-correlation over all times: V_aa(t, f)
        auto_pwr = np.real(vis_cube[:, :, ant_a, ant_a]).mean(axis=0)  # (num_channels,)
        auto_spectra[ant_a] = auto_pwr

    # Render each panel in the upper-triangular matrix
    for row_idx in range(N):
        ant_i = selected_antennas[row_idx]
        for col_idx in range(N):
            ant_j = selected_antennas[col_idx]

            # 1. Lower triangle: leave completely blank
            if col_idx < row_idx:
                continue

            ax = fig.add_subplot(gs[row_idx, col_idx])
            ax.set_facecolor(panel_bg)

            # 2. Diagonal (i == j): 1D Auto-Spectrum
            if col_idx == row_idx:
                auto_p = auto_spectra[ant_i]
                # Log scale power or positive power
                pos_power = np.maximum(1e-3, auto_p)

                ax.plot(freq_mhz, pos_power, color=line_color, linewidth=0.9)
                ax.set_yscale("log")
                ax.set_xlim(f_min, f_max)

                # Format tick labels
                ax.tick_params(colors=sub_text_color, labelsize=6.0, pad=1, length=2.5)
                ax.grid(True, linestyle=":", color=grid_color, alpha=0.6)

                # Diagonal border
                for sp in ax.spines.values():
                    sp.set_color(text_color)
                    sp.set_linewidth(0.8)

                # Show ticks on auto-spectrum
                if row_idx == 0:
                    ax.tick_params(labeltop=False, labelbottom=False)
                if col_idx == 0:
                    ax.set_ylabel("Power", fontsize=7.5, color=text_color, labelpad=2)

            # 3. Off-Diagonal (i < j): 2D Waterfall of Re(V_ij)
            else:
                # Visibility time-frequency slice: V_ij(t, f)
                # vis_cube shape: (num_times, num_channels, 64, 64)
                v_ij_tf = vis_cube[:, :, ant_i, ant_j]  # (num_times, num_channels)
                re_v = np.real(
                    v_ij_tf
                ).T  # Transpose to (num_channels, num_times) -> (Freq, Time)

                # Normalize by auto-power to make fringes uniform across channels
                auto_norm = np.sqrt(
                    np.outer(auto_spectra[ant_i], np.ones(len(time_s)))
                    * np.outer(auto_spectra[ant_j], np.ones(len(time_s)))
                )
                norm_re_v = re_v / np.maximum(1e-6, auto_norm)

                # Clip real component to highlight fringe stripes
                if clip_percentile is not None:
                    vmin = float(np.percentile(norm_re_v, clip_percentile))
                    vmax = float(np.percentile(norm_re_v, 100.0 - clip_percentile))
                else:
                    sigma = float(np.std(norm_re_v))
                    mean_val = float(np.mean(norm_re_v))
                    vmin = mean_val - clip_sigma * sigma
                    vmax = mean_val + clip_sigma * sigma

                if vmax <= vmin:
                    vmax = vmin + 1.0

                ax.imshow(
                    norm_re_v,
                    origin="lower",
                    aspect="auto",
                    extent=extent,
                    cmap=cmap_waterfall,
                    vmin=vmin,
                    vmax=vmax,
                    interpolation="nearest",
                )

                # Off-diagonal borders
                for sp in ax.spines.values():
                    sp.set_color(text_color)
                    sp.set_linewidth(0.6)

                # Hide internal tick labels to preserve clean paper aesthetic
                ax.set_xticks([])
                ax.set_yticks([])

            # Column Headers (Top Row): Baseline displacement vectors (dx, dy)
            if row_idx == 0:
                dx = pos_x[ant_j] - pos_x[selected_antennas[0]]
                dy = pos_y[ant_j] - pos_y[selected_antennas[0]]
                header_text = f"({dx:.1f}m, {dy:.1f}m)"
                ax.set_title(
                    header_text,
                    fontsize=7.0,
                    color=text_color,
                    pad=4,
                    fontweight="bold",
                )

    # Master Title & CASM Caption at bottom
    if title is None:
        title = (
            f"CHARTS 64-Antenna Correlation Matrix Waterfall ({N}x{N} Baselines) — {scenario_name}\n"
            f"Band: {f_min:.1f} - {f_max:.1f} MHz | Duration: {t_max:.1f} s | Site: Observatorio Carén"
        )
    fig.suptitle(title, fontsize=12, fontweight="bold", color=text_color, y=0.97)

    # Explanatory Caption matching CASM Figure 11
    caption_text = (
        "Figure: The correlation matrix during transit. Each off-diagonal panel corresponds to visibility data as a function of\n"
        "frequency (vertical axis) and time (horizontal axis). We plot a clipped real-component of V_ij showing geometric fringe patterns.\n"
        "The impact of cross-talk can be seen in short (<= 1.8 m) baselines. Diagonals show each antenna's unflattened auto-spectrum."
    )
    fig.text(
        0.50,
        0.025,
        caption_text,
        ha="center",
        va="center",
        fontsize=9.0,
        style="italic",
        color=sub_text_color,
    )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=180, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)

    print(f"[OK] Saved CASM correlation matrix waterfall plot: {output_png}")
    return output_png


# =============================================================================
# CLI Main Routine
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Generate CASM-style correlation matrix waterfall plot for antenna pairs (like Figure 11).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--corr-dir",
        type=Path,
        default=None,
        help="Directory containing Kotekan correlator binary dumps (corr_*.bin).",
    )
    parser.add_argument(
        "--baseband",
        type=Path,
        default=None,
        help="Path to HDF5 (.h5) or raw binary (.bin) baseband file.",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="sun_with_noise",
        help="Celestial scenario for simulation or fallback (default: sun_with_noise).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output PNG file path (default: ./dumps_charts_64ant/plots/casm_correlation_matrix_<scenario>.png).",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=Path("./dumps_charts_64ant/plots"),
        help="Directory to save generated plots (default: ./dumps_charts_64ant/plots).",
    )
    parser.add_argument(
        "--num-antennas",
        "-n",
        type=int,
        default=16,
        help="Number of antennas to include in matrix (default: 16 matching CASM Figure 11; use 8 for fast compact view).",
    )
    parser.add_argument(
        "--antennas",
        type=str,
        default=None,
        help="Explicit comma-separated list of antenna IDs to include (e.g. 0,1,2,3,4,5,6,7).",
    )
    parser.add_argument(
        "--all-antennas",
        action="store_true",
        help="Plot all 64 antennas (64x64 matrix with 2,112 subplots).",
    )
    parser.add_argument(
        "--clip-sigma",
        type=float,
        default=2.5,
        help="Clipping threshold for real visibility fringes in units of standard deviation (default: 2.5).",
    )
    parser.add_argument(
        "--cross-talk",
        action="store_true",
        default=True,
        help="Enable mutual coupling cross-talk on short baselines (<= 1.8m) in simulation mode.",
    )
    parser.add_argument(
        "--dark-mode",
        action="store_true",
        help="Use dark theme instead of white publication paper style.",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=60.0,
        help="Simulation window duration in seconds (default: 60.0 = 1 minute).",
    )
    parser.add_argument(
        "--num-channels",
        "--num-freq",
        dest="num_channels",
        type=int,
        default=672,
        help="Number of frequency channels (default: 672).",
    )
    parser.add_argument(
        "--max-time-samples",
        type=int,
        default=150,
        help="Maximum time samples to load evenly across window (default: 150).",
    )

    args = parser.parse_args()

    # Determine antenna subset
    if args.all_antennas:
        selected_antennas = list(range(64))
    elif args.antennas:
        selected_antennas = [
            int(x.strip()) for x in args.antennas.split(",") if x.strip().isdigit()
        ]
    else:
        selected_antennas = select_representative_antennas(
            total_antennas=64, num_selected=args.num_antennas
        )

    print(
        f"Selected {len(selected_antennas)} antennas for CASM matrix: {selected_antennas}"
    )

    # Determine data source: Correlator Dumps > Baseband Dump > Simulated Transit
    vis_cube = None
    time_s = None
    freq_mhz = None
    source_desc = ""

    if args.corr_dir and args.corr_dir.exists():
        try:
            vis_cube, time_s, freq_mhz = load_visibilities_from_correlator_dir(
                corr_dir=args.corr_dir,
                num_elements=64,
                num_channels=args.num_channels,
                max_time_samples=args.max_time_samples,
                duration_s=args.duration_s,
            )
            source_desc = f"Correlator Dumps ({args.corr_dir.name})"
        except Exception as e:
            print(f"[WARN] Failed to load correlator dumps from {args.corr_dir}: {e}")

    if vis_cube is None and args.baseband and args.baseband.exists():
        try:
            vis_cube, time_s, freq_mhz = load_visibilities_from_baseband(args.baseband)
            source_desc = f"Baseband Dump ({args.baseband.name})"
        except Exception as e:
            print(
                f"[WARN] Failed to compute correlations from baseband {args.baseband}: {e}"
            )

    # Fallback to high-fidelity analytical transit simulation with cross-talk
    if vis_cube is None:
        print(
            f"Generating physical transit simulation with cross-talk for: {args.scenario} ..."
        )
        vis_cube, time_s, freq_mhz = simulate_transit_visibilities(
            scenario=args.scenario,
            num_elements=64,
            num_channels=336,
            num_times=120,
            duration_s=args.duration_s,
            add_crosstalk=args.cross_talk,
        )
        source_desc = f"Physical Transit Model ({args.scenario}, cross-talk enabled)"

    # Output path
    if args.output is not None:
        output_png = args.output
    else:
        args.plots_dir.mkdir(parents=True, exist_ok=True)
        stem = args.scenario
        if args.baseband:
            stem = args.baseband.stem.replace("_64ant_5ms", "")
        output_png = (
            args.plots_dir
            / f"casm_correlation_matrix_{stem}_{len(selected_antennas)}ant.png"
        )

    plot_casm_correlation_matrix(
        vis_cube=vis_cube,
        time_s=time_s,
        freq_mhz=freq_mhz,
        selected_antennas=selected_antennas,
        output_png=output_png,
        clip_sigma=args.clip_sigma,
        dark_mode=args.dark_mode,
        scenario_name=source_desc,
    )


if __name__ == "__main__":
    main()
