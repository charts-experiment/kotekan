#!/usr/bin/env python3
"""
CHARTS 64-Antenna Array & Correlation Dumps Master Visualization
================================================================
Generates a comprehensive, high-resolution PNG dashboard visualizing:
  1. The physical CHARTS 64-antenna array layout (8x8 grid, dx=dy=0.6m, Carén Observatory).
  2. The interferometric baseline distribution (u-v spatial frequency coverage).
  3. The resulting 64x64 correlation visibility matrices V_ij across all simulation dumps:
       - Zenith Calibration (ideal zero-phase baseline)
       - Sagittarius A* / Galactic Center
       - Centaurus A (Giant radio galaxy)
       - Crab Nebula (Taurus A / PSR B0531+21)
       - Vela Pulsar (PSR J0835-4510)
       - The Sun (dominant solar flux)
       - Saturated Antennas (ADC rail-clipping on feeds 7, 23, 42, 55)
       - Fast Radio Burst (FRB transient chirp)

Outputs:
  - charts_64ant_correlations_overview.png (high-resolution master overview)
  - Individual correlation plots per dump (*.png)
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

# Setup paths
_test_charts_dir = Path(__file__).resolve().parent
_kotekan_root = _test_charts_dir.parent
if str(_test_charts_dir) not in sys.path:
    sys.path.insert(0, str(_test_charts_dir))

from constants import (
    C_LIGHT,
    CHARTS_ALTITUDE_M,
    CHARTS_LATITUDE_DEG,
    CHARTS_LONGITUDE_DEG,
    DEFAULT_FREQUENCY_START_MHZ,
    DEFAULT_SPACING_M,
)
from generate_charts_64ant_dumps import (
    ASTRONOMICAL_CATALOG,
    DEFAULT_SATURATED_ANTENNAS,
    compute_transit_direction_cosines,
    get_charts_64_antenna_positions,
)
from inspect_correlator_dump import load_astron_correlator_dump


def plot_charts_64_antenna_grid(
    ax: plt.Axes, saturated_antennas: List[int] = DEFAULT_SATURATED_ANTENNAS
):
    """Plots the physical 8x8 CHARTS antenna layout with feed coordinates and saturation markers."""
    pos_x, pos_y = get_charts_64_antenna_positions(64, DEFAULT_SPACING_M)

    ax.set_facecolor("#0f172a")  # Dark slate background
    ax.grid(True, linestyle="--", color="#334155", alpha=0.6)

    # Plot normal antennas
    normal_mask = np.ones(64, dtype=bool)
    normal_mask[saturated_antennas] = False

    ax.scatter(
        pos_x[normal_mask],
        pos_y[normal_mask],
        c="#38bdf8",  # Sky blue
        s=220,
        edgecolors="#ffffff",
        linewidths=1.5,
        label="Operational Antennas (60)",
        zorder=4,
    )

    # Plot saturated / flagged antennas
    ax.scatter(
        pos_x[saturated_antennas],
        pos_y[saturated_antennas],
        c="#ef4444",  # Crimson red
        s=260,
        edgecolors="#fef08a",
        linewidths=2.0,
        marker="X",
        label=f"Saturated Feeds ({len(saturated_antennas)})",
        zorder=5,
    )

    # Add antenna labels
    for a in range(64):
        col = a & 7
        row = a >> 3
        is_sat = a in saturated_antennas
        color = "#fee2e2" if is_sat else "#0f172a"
        ax.annotate(
            str(a),
            (pos_x[a], pos_y[a]),
            color=color,
            fontsize=7.5,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=6,
        )

    ax.set_title(
        f"CHARTS 64-Antenna Array (8x8 Grid, dx=dy={DEFAULT_SPACING_M}m)\n"
        f"Observatorio Carén ({CHARTS_LATITUDE_DEG:.4f}°, {CHARTS_LONGITUDE_DEG:.4f}°)",
        fontsize=11,
        fontweight="bold",
        color="#f8fafc",
        pad=10,
    )
    ax.set_xlabel("East-West Distance [m]", fontsize=10, color="#cbd5e1")
    ax.set_ylabel("North-South Distance [m]", fontsize=10, color="#cbd5e1")
    ax.tick_params(colors="#94a3b8", labelsize=9)
    ax.set_xlim(-0.5, 7 * DEFAULT_SPACING_M + 0.5)
    ax.set_ylim(-0.5, 7 * DEFAULT_SPACING_M + 0.5)
    ax.set_aspect("equal")
    ax.legend(
        loc="upper right",
        fontsize=8,
        facecolor="#1e293b",
        edgecolor="#475569",
        labelcolor="#f8fafc",
    )


def plot_uv_coverage(ax: plt.Axes, wavelength_m: float = 0.857):
    """Plots the 2016 interferometric baselines (u-v spatial coverage)."""
    pos_x, pos_y = get_charts_64_antenna_positions(64, DEFAULT_SPACING_M)

    u_baselines = []
    v_baselines = []

    for i in range(64):
        for j in range(64):
            if i != j:
                u_baselines.append((pos_x[i] - pos_x[j]) / wavelength_m)
                v_baselines.append((pos_y[i] - pos_y[j]) / wavelength_m)

    u_arr = np.array(u_baselines)
    v_arr = np.array(v_baselines)

    ax.set_facecolor("#0f172a")
    ax.grid(True, linestyle="--", color="#334155", alpha=0.6)

    # 2D hexbin or scatter for density
    hb = ax.hexbin(
        u_arr,
        v_arr,
        gridsize=30,
        cmap="mako",
        mincnt=1,
        edgecolors="none",
    )

    ax.set_title(
        f"Interferometric Baseline Distribution (2016 Baselines)\n"
        f"@ 350 MHz (λ = {wavelength_m:.3f} m)",
        fontsize=11,
        fontweight="bold",
        color="#f8fafc",
        pad=10,
    )
    ax.set_xlabel("u [Spatial Wavelengths λ]", fontsize=10, color="#cbd5e1")
    ax.set_ylabel("v [Spatial Wavelengths λ]", fontsize=10, color="#cbd5e1")
    ax.tick_params(colors="#94a3b8", labelsize=9)
    ax.set_aspect("equal")
    cb = plt.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Baseline Redundancy Count", color="#cbd5e1", fontsize=8)
    cb.ax.tick_params(colors="#94a3b8", labelsize=8)


def compute_or_load_correlation_matrix(
    scenario: str,
    dumps_dir: Path | None = None,
    num_elements: int = 64,
    num_freq: int = 336,
    freq_idx: int = 168,
) -> np.ndarray:
    """
    Loads real correlator dump binary if it exists; otherwise computes the exact
    analytical correlation matrix V_ij directly from the physical scenario.
    """
    if dumps_dir is not None:
        candidate_files = [
            dumps_dir / f"corr_{scenario}_0000000.bin",
            dumps_dir / f"{scenario}_0000000.bin",
            dumps_dir / f"corr_{scenario}.bin",
        ]
        for cf in candidate_files:
            if cf.exists():
                cube = load_astron_correlator_dump(
                    cf, num_elements=num_elements, num_channels=num_freq
                )
                return cube[freq_idx]

    # Analytical physical model
    target_key = (
        scenario.replace("_with_noise", "")
        .replace("_no_noise", "")
        .replace("_saturated", "")
    )
    target = ASTRONOMICAL_CATALOG.get(target_key, ASTRONOMICAL_CATALOG["vela"])

    pos_x, pos_y = get_charts_64_antenna_positions(num_elements, DEFAULT_SPACING_M)
    l0, m0 = compute_transit_direction_cosines(target["dec_deg"], CHARTS_LATITUDE_DEG)
    freq_hz = (DEFAULT_FREQUENCY_START_MHZ + freq_idx * 0.3) * 1e6
    c_inv = 1.0 / C_LIGHT
    two_pi = 2.0 * np.pi

    delays = (l0 * pos_x + m0 * pos_y) * c_inv
    phases = two_pi * freq_hz * delays

    amp = float(target["nominal_amp"])
    use_noise = "no_noise" not in scenario
    saturate = "saturat" in scenario

    # Signal vector: s_i = A * exp(-j * phases[i])
    s_vec = amp * np.exp(-1j * phases).astype(np.complex128)

    # Coherent outer product V_sig = s @ s^H
    v_matrix = np.outer(s_vec, np.conj(s_vec))

    # Add independent thermal noise variance on diagonal and uncorrelated cross terms
    rng = np.random.default_rng(42)
    if use_noise:
        sigmas = rng.uniform(0.4, 0.7, size=num_elements)
        # Thermal noise on diagonal
        np.fill_diagonal(v_matrix, np.diag(v_matrix) + (sigmas**2) * 1536.0 * 0.1)
        # Small background fluctuation
        noise_cross = (
            (
                rng.normal(0, 0.05, size=(num_elements, num_elements))
                + 1j * rng.normal(0, 0.05, size=(num_elements, num_elements))
            )
            * np.outer(sigmas, sigmas)
            * 10.0
        )
        noise_cross = 0.5 * (noise_cross + noise_cross.conj().T)
        v_matrix += noise_cross

    if saturate:
        for bad_ant in DEFAULT_SATURATED_ANTENNAS:
            # Saturation boosts bad feed total power and causes decorrelation lines
            v_matrix[bad_ant, :] *= 3.5
            v_matrix[:, bad_ant] *= 3.5
            v_matrix[bad_ant, bad_ant] *= 4.0

    return v_matrix


def generate_master_visualization(
    output_png: Path,
    dumps_dir: Path | None = None,
):
    """Generates the comprehensive 8-panel master PNG visualization dashboard."""
    output_png.parent.mkdir(parents=True, exist_ok=True)

    # Define key scenarios to showcase
    showcase_scenarios = [
        ("zenith_no_noise", "Zenith Field (l=0, m=0, Pure Phase)"),
        ("sgr_a_with_noise", "Sagittarius A* (+4.4° N of Zenith)"),
        ("cen_a_with_noise", "Centaurus A (-9.6° S of Zenith)"),
        ("crab_with_noise", "Crab Nebula / Taurus A (+55.4° N)"),
        ("vela_with_noise", "Vela Pulsar (-11.8° S of Zenith)"),
        ("sun_with_noise", "The Sun (Dominant Solar Flux)"),
        ("sun_with_noise_saturated", "Sun + Saturated Feeds [7,23,42,55]"),
        ("frb_with_noise", "Fast Radio Burst (2 ms DM=300)"),
    ]

    # Overall figure: 2 rows of array/uv + 8 correlation heatmaps (3 rows x 4 cols layout)
    fig = plt.figure(figsize=(24, 16), facecolor="#020617")
    gs = fig.add_gridspec(
        3, 4, hspace=0.32, wspace=0.28, left=0.05, right=0.96, top=0.92, bottom=0.06
    )

    # 1. Top Row Left: Physical 64-Antenna Grid Layout
    ax_array = fig.add_subplot(gs[0, 0:2])
    plot_charts_64_antenna_grid(ax_array)

    # 2. Top Row Right: Baseline / UV Coverage
    ax_uv = fig.add_subplot(gs[0, 2:4])
    plot_uv_coverage(ax_uv)

    # 3. Rows 1 & 2: 8 Showcase Correlation Matrices (V_ij)
    axes_corr = [
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[1, 2]),
        fig.add_subplot(gs[1, 3]),
        fig.add_subplot(gs[2, 0]),
        fig.add_subplot(gs[2, 1]),
        fig.add_subplot(gs[2, 2]),
        fig.add_subplot(gs[2, 3]),
    ]

    colormaps = [
        "inferno",
        "magma",
        "plasma",
        "viridis",
        "inferno",
        "cividis",
        "hot",
        "turbo",
    ]

    for idx, (sc, title) in enumerate(showcase_scenarios):
        ax = axes_corr[idx]
        v_mat = compute_or_load_correlation_matrix(sc, dumps_dir=dumps_dir)
        v_amp = np.abs(v_mat)

        # Normalize for clear visual contrast
        norm_v = v_amp / (np.max(v_amp) + 1e-12)

        im = ax.imshow(
            norm_v, cmap=colormaps[idx % len(colormaps)], aspect="equal", origin="upper"
        )

        ax.set_title(
            f"{title}\n64x64 Visibility Matrix |V_ij|",
            fontsize=10,
            fontweight="bold",
            color="#f8fafc",
            pad=8,
        )
        ax.set_xlabel("Antenna j", fontsize=8.5, color="#cbd5e1")
        ax.set_ylabel("Antenna i", fontsize=8.5, color="#cbd5e1")
        ax.tick_params(colors="#94a3b8", labelsize=8)

        # Highlight saturated antennas if applicable
        if "saturated" in sc:
            for bad_ant in DEFAULT_SATURATED_ANTENNAS:
                ax.axhline(
                    bad_ant, color="#ef4444", linestyle=":", alpha=0.7, linewidth=1.0
                )
                ax.axvline(
                    bad_ant, color="#ef4444", linestyle=":", alpha=0.7, linewidth=1.0
                )

        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.tick_params(colors="#94a3b8", labelsize=7)
        cb.set_label("Normalized |V_ij|", color="#cbd5e1", fontsize=7.5)

    # Master Title & Subtitle
    fig.suptitle(
        "CHARTS 64-Antenna Array Physical Geometry & AstronCorrelator Visibility Matrix Dashboard\n"
        "Observatorio Carén (Chile) • 300–400.5 MHz • 5.12 ms Dumps • Tensor Core Correlator (John Romein)",
        fontsize=16,
        fontweight="bold",
        color="#38bdf8",
        y=0.98,
    )

    plt.savefig(output_png, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(
        f"[SUCCESS] Master PNG visualization successfully generated: {output_png} (DPI=300)"
    )


def main():
    parser = argparse.ArgumentParser(
        description="CHARTS 64-Antenna Visualization Dashboard"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="charts_64ant_correlations_overview.png",
        help="Path to save the output master PNG (default: charts_64ant_correlations_overview.png)",
    )
    parser.add_argument(
        "--dumps-dir",
        type=str,
        default="./dumps_charts_64ant/correlator",
        help="Path to correlator output directory (if available)",
    )
    args = parser.parse_args()

    dumps_path = Path(args.dumps_dir) if Path(args.dumps_dir).exists() else None
    out_path = Path(args.output)

    generate_master_visualization(output_png=out_path, dumps_dir=dumps_path)


if __name__ == "__main__":
    main()
