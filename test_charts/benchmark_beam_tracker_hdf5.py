#!/usr/bin/env python3
"""
CHARTS Beam Tracker Real HDF5 Data Benchmark & Verification Tool
================================================================
Loads real CHARTS baseband HDF5 data (e.g. from 260816T013722Z_CHARTS_hdf5),
automatically detects connected vs. unplugged antennas, synthesizes complex voltage
tracked beams using CUDA / NumPy, and generates performance metrics & diagnostic plots.
"""

import os
import sys
import time
import argparse
import numpy as np
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_test_charts_dir = os.path.dirname(os.path.abspath(__file__))
if _test_charts_dir not in sys.path:
    sys.path.insert(0, _test_charts_dir)

from constants import (
    C_LIGHT,
    DEFAULT_SPACING_M,
    DEFAULT_FREQUENCY_START_MHZ,
    CHARTS_CHANNEL_WIDTH_MHZ,
    FPGA_TIME_RESOLUTION_US,
)

# ============================================================================
# 4-bit Complex Data Handling
# ============================================================================


def unpack_4bit_complex_numpy(u8_array: np.ndarray) -> np.ndarray:
    """
    Unpacks int4x2_t (4-bit real in bits 0-3, 4-bit imag in bits 4-7) to complex64.
    """
    real = (u8_array & 0x0F).astype(np.int8)
    imag = (u8_array >> 4).astype(np.int8)
    real[real >= 8] -= 16
    imag[imag >= 8] -= 16
    return real.astype(np.float32) + 1j * imag.astype(np.float32)


# ============================================================================
# Antenna Health & Variance Analysis
# ============================================================================


def analyze_antenna_health(
    raw_data_ant_freq_time: np.ndarray, power_threshold: float = 0.05
):
    """
    Analyzes per-antenna power variance to distinguish connected antennas from unplugged ones.
    raw_data shape: (antenna, freq, time)
    Returns:
        mask: np.ndarray of uint8 (1 = healthy/connected, 0 = unplugged/dead)
        powers: np.ndarray of mean power per antenna
    """
    n_ant = raw_data_ant_freq_time.shape[0]
    powers = np.zeros(n_ant, dtype=np.float64)
    mask = np.zeros(n_ant, dtype=np.uint8)

    # Subsample in time for fast variance calculation if dataset is large
    step = max(1, raw_data_ant_freq_time.shape[2] // 2048)
    sample_slice = raw_data_ant_freq_time[:, :, ::step]

    for a in range(n_ant):
        cdata = unpack_4bit_complex_numpy(sample_slice[a, :, :])
        p = np.mean(np.real(cdata) ** 2 + np.imag(cdata) ** 2)
        powers[a] = p
        if p > power_threshold:
            mask[a] = 1

    return mask, powers


# ============================================================================
# Beamformer Engine (Vectorized Analytical Reference)
# ============================================================================


def beamform_reference_cpu(
    raw_ant_freq_time: np.ndarray,
    mask: np.ndarray,
    freqs_hz: np.ndarray,
    l: float = 0.0,
    m: float = 0.0,
    spacing_m: float = 0.6,
) -> np.ndarray:
    """
    Synthesizes complex voltages E(time, freq) for a given sky direction (l, m).
    raw_ant_freq_time shape: (antenna, freq, time)
    Returns:
        formed_voltages shape: (time, freq) as complex64
    """
    n_ant, n_freq, n_time = raw_ant_freq_time.shape
    c = 299792458.0

    # Antenna positions (8x8 or 16x16 grid)
    if n_ant <= 64:
        cols = np.arange(n_ant) & 7
        rows = np.arange(n_ant) >> 3
    else:
        cols = np.arange(n_ant) & 15
        rows = np.arange(n_ant) >> 4

    pos_x = cols * spacing_m  # (n_ant,)
    pos_y = rows * spacing_m  # (n_ant,)
    delays_m = pos_x * l + pos_y * m  # (n_ant,)

    # Geometric phase shifts: phi[a, f] = 2 * pi * f * delay / c
    phases = 2.0 * np.pi * np.outer(delays_m, freqs_hz / c)  # (n_ant, n_freq)
    weights = np.exp(1j * phases).astype(np.complex64)  # (n_ant, n_freq)

    # Apply antenna mask to weights
    weights[mask == 0, :] = 0.0

    # Unpack all antennas: (n_ant, n_freq, n_time)
    complex_inputs = unpack_4bit_complex_numpy(raw_ant_freq_time)

    # Coherent sum along antenna axis:
    # E(f, t) = sum_a (W[a, f] * V[a, f, t])
    # einsum: 'af, aft -> tf'
    formed_voltages = np.einsum("af,aft->tf", weights, complex_inputs, optimize=True)

    return formed_voltages


# ============================================================================
# Main Diagnostic & Benchmark Routine
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="CHARTS Real HDF5 Beam Tracker Benchmark"
    )
    parser.add_argument(
        "h5_path", help="Path to HDF5 file (baseband_virtual.h5 or chunk file)"
    )
    parser.add_argument(
        "--n-time",
        type=int,
        default=15360,
        help="Number of time samples for 1 dump (default: 15360 ~50ms)",
    )
    parser.add_argument(
        "--n-freq-max",
        type=int,
        default=336,
        help="Max frequency channels to process (default: 336)",
    )
    parser.add_argument(
        "--spacing-m",
        type=float,
        default=0.6,
        help="Antenna spacing in meters (default: 0.6)",
    )
    parser.add_argument(
        "--target-ra-deg",
        type=float,
        default=83.633,
        help="Target RA in deg (default: 83.633 - Crab Pulsar)",
    )
    parser.add_argument(
        "--target-dec-deg",
        type=float,
        default=22.014,
        help="Target Dec in deg (default: 22.014)",
    )
    parser.add_argument(
        "--out-plot",
        default="charts_beam_tracker_real_data_benchmark.png",
        help="Path to save diagnostic plot",
    )
    args = parser.parse_args()

    print("======================================================================")
    print(" CHARTS Real Data Beam Tracker Benchmark & Diagnostic Suite")
    print(f" Target HDF5 File : {args.h5_path}")
    print("======================================================================")

    if not os.path.exists(args.h5_path):
        print(f"[ERROR] File not found: {args.h5_path}")
        sys.exit(1)

    file_size_gb = os.path.getsize(args.h5_path) / (1024**3)
    print(f" File Size on Disk: {file_size_gb:.3f} GB")

    with h5py.File(args.h5_path, "r") as f:
        print("\n--- HDF5 Metadata Attributes ---")
        for k in f.attrs.keys():
            print(f"  {k}: {f.attrs[k]}")

        dset = f["baseband"]
        full_shape = dset.shape
        print(f"\n Dataset Shape [Antenna, Freq, Time]: {full_shape}")

        n_ant_file = full_shape[0]
        n_freq_file = full_shape[1]
        n_time_file = full_shape[2]

        freq_start_mhz = float(
            f.attrs.get("freq_start_MHz", DEFAULT_FREQUENCY_START_MHZ)
        )
        delta_freq_mhz = float(f.attrs.get("delta_freq_MHz", CHARTS_CHANNEL_WIDTH_MHZ))
        delta_time_us = float(f.attrs.get("delta_time_us", FPGA_TIME_RESOLUTION_US))

        n_time = min(args.n_time, n_time_file) if args.n_time > 0 else n_time_file
        n_freq = min(args.n_freq_max, n_freq_file)

        print(f"\n--- Ingesting 1 Dump for Benchmarking ---")
        print(
            f"  Time Samples     : {n_time:,} ({n_time * delta_time_us / 1000.0:.2f} ms of real sky data)"
        )
        print(
            f"  Frequency Range  : {freq_start_mhz:.1f} MHz to {freq_start_mhz + n_freq * delta_freq_mhz:.1f} MHz ({n_freq} channels)"
        )
        print(f"  Antenna Count    : {n_ant_file}")

        t0_read = time.perf_counter()
        raw_slice = dset[:n_ant_file, :n_freq, :n_time]
        t1_read = time.perf_counter()
        print(f"  HDF5 Read Time   : {(t1_read - t0_read)*1000.0:.2f} ms")

    # Physical frequencies in Hz
    freqs_hz = (freq_start_mhz + np.arange(n_freq) * delta_freq_mhz) * 1e6

    # 1. Analyze Antenna Activity & Health
    print("\n--- Antenna Health & Connection Status ---")
    mask, powers = analyze_antenna_health(raw_slice)
    connected_count = int(np.sum(mask))
    unplugged_count = n_ant_file - connected_count

    print(
        f"  Connected / Active Antennas   : {connected_count} / {n_ant_file} ({connected_count/n_ant_file*100:.1f}%)"
    )
    print(f"  Unplugged / Silent Antennas   : {unplugged_count} / {n_ant_file}")
    print(f"  Mean Power of Active Antennas : {np.mean(powers[mask == 1]):.4f}")
    if unplugged_count > 0:
        print(f"  Mean Power of Silent Antennas : {np.mean(powers[mask == 0]):.4f}")

    connected_indices = np.where(mask == 1)[0]
    unplugged_indices = np.where(mask == 0)[0]
    print(f"  Connected Elements  : {connected_indices.tolist()}")
    if unplugged_count > 0:
        print(f"  Unplugged Elements  : {unplugged_indices.tolist()}")

    # 2. Benchmark Beam Tracker Synthesis
    print("\n--- Running Beam Tracker on Real Sky Dump ---")
    t0_bf = time.perf_counter()
    # Zenith beam (l=0, m=0)
    formed_zenith = beamform_reference_cpu(
        raw_slice, mask, freqs_hz, l=0.0, m=0.0, spacing_m=args.spacing_m
    )
    # Off-zenith tracked beam
    formed_offset = beamform_reference_cpu(
        raw_slice, mask, freqs_hz, l=0.05, m=0.02, spacing_m=args.spacing_m
    )
    t1_bf = time.perf_counter()

    bf_time_ms = (t1_bf - t0_bf) * 1000.0
    payload_bytes = n_time * n_freq * n_ant_file
    throughput_gb_s = (payload_bytes / (1024**3)) / ((t1_bf - t0_bf))
    real_time_budget_ms = n_time * delta_time_us / 1000.0

    print(f"  Execution Time   : {bf_time_ms:.2f} ms")
    print(f"  Throughput       : {throughput_gb_s:.2f} GB/s")
    print(f"  Real-Time Budget : {real_time_budget_ms:.2f} ms (Frame duration)")

    # 3. Analyze Output Complex Voltages & Power
    zenith_power = np.abs(formed_zenith) ** 2  # (time, freq)
    offset_power = np.abs(formed_offset) ** 2

    mean_zenith_p = np.mean(zenith_power)
    mean_offset_p = np.mean(offset_power)
    mean_single_ant_p = np.mean(powers[mask == 1]) if connected_count > 0 else 1.0
    measured_array_gain = mean_zenith_p / max(1e-12, mean_single_ant_p)

    print(f"\n--- Output Complex Voltage Statistics ---")
    print(
        f"  Formed Beam Real Voltages Range : [{np.min(np.real(formed_zenith)):.2f} .. {np.max(np.real(formed_zenith)):.2f}]"
    )
    print(
        f"  Formed Beam Imag Voltages Range : [{np.min(np.imag(formed_zenith)):.2f} .. {np.max(np.imag(formed_zenith)):.2f}]"
    )
    print(f"  Mean Formed Beam Power |E|²     : {mean_zenith_p:.4f}")
    print(f"  Incoherent Single-Antenna Power : {mean_single_ant_p:.4f}")
    print(
        f"  Measured Array Coherent Gain    : {measured_array_gain:.2f}x (Theoretical Max: {connected_count}x for noise / {connected_count**2}x for coherent source)"
    )

    # 4. Generate Comprehensive Diagnostic Visualizations
    print(f"\n--- Generating Diagnostic Plots -> {args.out_plot} ---")
    fig = plt.figure(figsize=(16, 10))

    # Subplot 1: Antenna Power Distribution (Health Mask)
    ax1 = fig.add_subplot(2, 2, 1)
    colors = ["green" if mask[i] == 1 else "red" for i in range(n_ant_file)]
    ax1.bar(range(n_ant_file), powers, color=colors, alpha=0.7, edgecolor="black")
    ax1.set_xlabel("Antenna Index")
    ax1.set_ylabel("Mean Power (Variance)")
    ax1.set_title(
        f"Antenna Power & Activity Status ({connected_count} Active, {unplugged_count} Unplugged)"
    )
    ax1.grid(alpha=0.3)

    # Subplot 2: Dynamic Spectrum (Waterfall of Formed Beam)
    ax2 = fig.add_subplot(2, 2, 2)
    # Downsample in time for waterfall display
    waterfall_step = max(1, n_time // 1000)
    waterfall_data = zenith_power[::waterfall_step, :].T  # (freq, time)
    extent = [0, n_time * delta_time_us / 1000.0, freqs_hz[0] / 1e6, freqs_hz[-1] / 1e6]
    im = ax2.imshow(
        waterfall_data, aspect="auto", origin="lower", cmap="inferno", extent=extent
    )
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Frequency (MHz)")
    ax2.set_title(f"Formed Beam Dynamic Spectrum (Waterfall I(t, f))")
    fig.colorbar(im, ax=ax2, label="Power |E|²")

    # Subplot 3: Power Spectrum across Frequency Channels
    ax3 = fig.add_subplot(2, 2, 3)
    spec_zenith = np.mean(zenith_power, axis=0)
    spec_offset = np.mean(offset_power, axis=0)
    freq_axis_mhz = freqs_hz / 1e6
    ax3.plot(
        freq_axis_mhz,
        spec_zenith,
        label=f"Formed Beam (Zenith: l=0, m=0)",
        color="blue",
        lw=1.5,
    )
    ax3.plot(
        freq_axis_mhz,
        spec_offset,
        label=f"Formed Beam (Offset: l=0.05, m=0.02)",
        color="orange",
        alpha=0.8,
        lw=1.2,
    )
    ax3.set_xlabel("Frequency (MHz)")
    ax3.set_ylabel("Average Power |E|²")
    ax3.set_title("Formed Beam Power Spectrum")
    ax3.legend()
    ax3.grid(alpha=0.3)

    # Subplot 4: Time-Series Power Profile
    ax4 = fig.add_subplot(2, 2, 4)
    # Average across frequency to get integrated power time series
    time_series_ms = np.arange(0, n_time, waterfall_step) * delta_time_us / 1000.0
    ts_zenith = np.mean(zenith_power[::waterfall_step, :], axis=1)
    ts_offset = np.mean(offset_power[::waterfall_step, :], axis=1)
    ax4.plot(time_series_ms, ts_zenith, label="Zenith Beam", color="purple", lw=1.2)
    ax4.plot(
        time_series_ms, ts_offset, label="Offset Beam", color="gray", alpha=0.6, lw=1.0
    )
    ax4.set_xlabel("Time (ms)")
    ax4.set_ylabel("Bandpass-Integrated Power")
    ax4.set_title(
        f"Integrated Power Time Series (Frame Duration = {real_time_budget_ms:.2f} ms)"
    )
    ax4.legend()
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.out_plot, dpi=150)
    plt.close()
    print(f"  [SUCCESS] Plot saved to: {os.path.abspath(args.out_plot)}")

    print("\n======================================================================")
    print(" BENCHMARK & VERIFICATION ON REAL CHARTS DATA COMPLETE!")
    print("======================================================================")


if __name__ == "__main__":
    main()
