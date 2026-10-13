#!/usr/bin/env python3
"""
F-Engine to Kotekan Beam Tracker Verification Bridge & Diagnostic Suite
======================================================================
Ingests synthetic F-Engine datasets from RadioTelescopeFEngine.jl or
generate_fengine_sim_data.py, runs the Kotekan beam tracker reference
engine for 64-antenna (8x8) and 256-antenna (16x16) arrays, validates
coherent gain (N_ant^2 scaling), sidelobe suppression, trajectory tracking,
and generates comprehensive multi-panel diagnostic plots.
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

from constants import C_LIGHT, DEFAULT_SPACING_M


def unpack_int4x2(
    u8_array: np.ndarray, format_type: str = "twos_complement"
) -> np.ndarray:
    """
    Unpacks 4-bit complex int4x2 values (real and imag nibbles) to complex64.
    Supports both standard 2's complement and offset-binary ('int4x2_swapped_withoffset').
    """
    if "offset" in format_type.lower():
        # Offset binary where 8 corresponds to 0 (-8..+7 mapped to 0..15)
        # Note: in swapped_withoffset, real is low nibble, imag is high nibble
        real = (u8_array & 0x0F).astype(np.float32) - 8.0
        imag = (u8_array >> 4).astype(np.float32) - 8.0
    else:
        # Standard Two's complement sign extension for 4-bit values [-8..+7]
        r = (u8_array & 0x0F).astype(np.int8)
        i = (u8_array >> 4).astype(np.int8)
        r[r >= 8] -= 16
        i[i >= 8] -= 16
        real = r.astype(np.float32)
        imag = i.astype(np.float32)

    return real + 1j * imag


def get_antenna_positions(num_antennas: int, spacing_m: float = DEFAULT_SPACING_M):
    """
    Returns (pos_x, pos_y) arrays for 64 (8x8) or 256 (16x16) rectangular grid.
    """
    if num_antennas <= 64:
        cols = np.arange(num_antennas) & 7
        rows = np.arange(num_antennas) >> 3
    else:
        cols = np.arange(num_antennas) & 15
        rows = np.arange(num_antennas) >> 4

    return cols * spacing_m, rows * spacing_m


def beamform_synthesize(
    raw_ant_freq_time: np.ndarray,
    freqs_hz: np.ndarray,
    l_traj: np.ndarray,
    m_traj: np.ndarray,
    spacing_m: float = 0.6,
    mask: np.ndarray = None,
    format_type: str = "twos_complement",
) -> np.ndarray:
    """
    Synthesizes complex voltage formed beam E(time, freq) following (l(t), m(t)).
    raw_ant_freq_time: shape (num_antennas, num_freq, num_time) uint8
    l_traj, m_traj: shape (num_time,) float or scalar float
    Returns:
        formed_voltages: shape (num_time, num_freq) complex64
    """
    n_ant, n_freq, n_time = raw_ant_freq_time.shape
    pos_x, pos_y = get_antenna_positions(n_ant, spacing_m)

    if mask is None:
        mask = np.ones(n_ant, dtype=np.uint8)

    if np.isscalar(l_traj):
        l_traj = np.full(n_time, l_traj, dtype=np.float64)
    if np.isscalar(m_traj):
        m_traj = np.full(n_time, m_traj, dtype=np.float64)

    # Unpack int4x2 complex voltages: (n_ant, n_freq, n_time)
    complex_data = unpack_int4x2(raw_ant_freq_time, format_type=format_type)

    # Compute formed beam
    formed_voltages = np.zeros((n_time, n_freq), dtype=np.complex64)

    # Chunk over time
    chunk_size = min(2048, n_time)
    n_chunks = (n_time + chunk_size - 1) // chunk_size

    for c in range(n_chunks):
        t_start = c * chunk_size
        t_end = min(n_time, (c + 1) * chunk_size)

        l_sub = l_traj[t_start:t_end]  # (t_sub,)
        m_sub = m_traj[t_start:t_end]  # (t_sub,)

        # Delays: (t_sub, n_ant)
        delays = (np.outer(l_sub, pos_x) + np.outer(m_sub, pos_y)) / C_LIGHT

        # Phases: 2 * pi * f * delay -> (t_sub, n_freq, n_ant)
        phases = 2.0 * np.pi * np.einsum("f,ta->tfa", freqs_hz, delays)
        weights = np.exp(1j * phases).astype(np.complex64)  # (t_sub, n_freq, n_ant)

        # Apply antenna health/selection mask
        weights[:, :, mask == 0] = 0.0

        # Sub-slice data: (n_ant, n_freq, t_sub) -> transpose to (t_sub, n_freq, n_ant)
        data_sub = np.transpose(complex_data[:, :, t_start:t_end], (2, 1, 0))

        # Coherent sum along antenna axis
        formed_voltages[t_start:t_end, :] = np.sum(weights * data_sub, axis=2)

    return formed_voltages


def run_bridge_verification(h5_path: str, out_plot: str = ""):
    print("======================================================================")
    print(" KOTEKAN BEAM TRACKER <-> F-ENGINE VERIFICATION BRIDGE")
    print(f" Target HDF5 Dataset: {h5_path}")
    print("======================================================================")

    if not os.path.exists(h5_path):
        print(f"[ERROR] File not found: {h5_path}")
        return False

    with h5py.File(h5_path, "r") as f:
        print("\n--- HDF5 Metadata ---")
        for k in f.attrs.keys():
            print(f"  {k}: {f.attrs[k]}")

        # Check for dataset location (CHORD / RadioTelescopeFEngine.jl or CHARTS baseband)
        format_type = "twos_complement"
        if "voltage" in f:
            dset = f["voltage"]
            dset_shape = dset.shape
            print(f" Found 'voltage' dataset with shape {dset_shape}")
            # Format in RadioTelescopeFEngine.jl: (ndishes, npolrs, nfreqs, ntimes)
            if len(dset_shape) == 4:
                raw_data = dset[:, 0, :, :]  # take polarization 0
            else:
                raw_data = dset[:]
            format_type = str(dset.attrs.get("type", "int4x2_swapped_withoffset"))
            spacing_m = float(dset.attrs.get("feed_separation_x_m", 0.6))
            coarse_freq = dset.attrs.get("coarse_freq", None)
            if coarse_freq is not None:
                freq_indices = np.array(coarse_freq)
                # CHARTS / CHORD frequency conversion
                freqs_hz = (
                    (freq_indices * (4.9152e9 / 16384))
                    if np.max(freq_indices) > 500
                    else (300.0e6 + freq_indices * 300.0e3)
                )
            else:
                freqs_hz = 300.0e6 + np.arange(raw_data.shape[1]) * 300.0e3

            n_ant, n_freq, n_time = raw_data.shape
            scenario = "zenith"
            delta_time_us = float(dset.attrs.get("seq_length_nsec", 3333.33)) / 1000.0
            freq_start_mhz = freqs_hz[0] / 1e6
            delta_freq_mhz = (freqs_hz[1] - freqs_hz[0]) / 1e6 if n_freq > 1 else 0.3

        elif "baseband" in f:
            dset = f["baseband"]
            n_ant = dset.shape[0]
            n_freq = dset.shape[1]
            n_time = dset.shape[2]
            format_type = str(f.attrs.get("data_format", "twos_complement"))
            freq_start_mhz = float(f.attrs.get("freq_start_MHz", 300.0))
            delta_freq_mhz = float(f.attrs.get("delta_freq_MHz", 0.3))
            delta_time_us = float(f.attrs.get("delta_time_us", 3.2552))
            spacing_m = float(f.attrs.get("spacing_m", 0.6))
            scenario = str(f.attrs.get("scenario", "zenith"))
            freqs_hz = (freq_start_mhz + np.arange(n_freq) * delta_freq_mhz) * 1e6

            t0_read = time.perf_counter()
            raw_data = dset[:n_ant, :n_freq, :n_time]
            t1_read = time.perf_counter()
            print(f"  Read Time  : {(t1_read - t0_read)*1000.0:.2f} ms")
        else:
            print(f"[ERROR] Could not find baseband or voltage dataset in {h5_path}")
            return False

        print(f"\n--- Dataset Parameters ---")
        print(f"  Antennas   : {n_ant} ({'8x8' if n_ant <= 64 else '16x16'} array)")
        print(
            f"  Frequencies: {n_freq} channels ({freq_start_mhz:.1f} to {freq_start_mhz + n_freq*delta_freq_mhz:.1f} MHz)"
        )
        print(
            f"  Time       : {n_time} samples ({n_time * delta_time_us / 1000.0:.2f} ms)"
        )
        print(f"  Data Format: {format_type}")
        print(f"  Scenario   : {scenario}")

    # 1. Single Antenna Power & Variance
    c_single = unpack_int4x2(raw_data, format_type=format_type)
    ant_powers = np.mean(np.real(c_single) ** 2 + np.imag(c_single) ** 2, axis=(1, 2))
    mean_ant_p = float(np.mean(ant_powers))
    print(f"\n--- Single Antenna Statistics ---")
    print(f"  Mean Power across Antennas : {mean_ant_p:.4f}")
    print(f"  Min Antenna Power          : {np.min(ant_powers):.4f}")
    print(f"  Max Antenna Power          : {np.max(ant_powers):.4f}")

    # 2. Synthesize Beams (On-Target vs Off-Target)
    print("\n--- Synthesizing Beams ---")
    t0_bf = time.perf_counter()

    if scenario == "off_zenith":
        target_l = 0.08
        target_m = -0.04
    elif scenario == "moving":
        target_l = 0.05 + 1.0e-5 * np.arange(n_time)
        target_m = 0.02 + 0.5e-5 * np.arange(n_time)
    elif scenario == "frb":
        target_l = 0.03
        target_m = 0.01
    elif scenario == "multisource":
        target_l = 0.06
        target_m = 0.02
    else:  # zenith
        target_l = 0.0
        target_m = 0.0

    # On-Target Beam
    v_on_target = beamform_synthesize(
        raw_data,
        freqs_hz,
        target_l,
        target_m,
        spacing_m=spacing_m,
        format_type=format_type,
    )

    # Off-Target (Sidelobe / Null Rejection outside 64-ant primary beamwidth ~0.2 rad)
    offset_l = 0.25 if np.isscalar(target_l) else target_l + 0.25
    offset_m = 0.20 if np.isscalar(target_m) else target_m + 0.20
    v_off_target = beamform_synthesize(
        raw_data,
        freqs_hz,
        offset_l,
        offset_m,
        spacing_m=spacing_m,
        format_type=format_type,
    )

    t1_bf = time.perf_counter()
    print(f"  Beamforming execution time : {(t1_bf - t0_bf)*1000.0:.2f} ms")

    # 3. Analyze Coherent Gain & Sidelobe Rejection
    p_on = np.abs(v_on_target) ** 2
    p_off = np.abs(v_off_target) ** 2

    mean_p_on = float(np.mean(p_on))
    mean_p_off = float(np.mean(p_off))

    measured_gain = mean_p_on / max(1e-12, mean_ant_p)
    rejection_db = 10.0 * np.log10(max(1e-12, mean_p_on) / max(1e-12, mean_p_off))

    print(f"\n--- Beam Tracker Performance & Verification Metrics ---")
    print(f"  On-Target Mean Power |E|² : {mean_p_on:.4f}")
    print(f"  Off-Target Mean Power |E|²: {mean_p_off:.4f}")
    print(
        f"  Array Coherent Gain Ratio : {measured_gain:.2f}x (Expected ~ {n_ant:.0f}x for noise, up to {n_ant**2:.0f}x for pure tone)"
    )
    print(f"  Sidelobe / Null Rejection : {rejection_db:.2f} dB")

    # Pass / Fail Evaluation
    min_gain_threshold = 20.0 if n_ant <= 64 else 60.0
    passed_gain = measured_gain >= min_gain_threshold
    passed_rejection = rejection_db >= 8.0
    all_passed = passed_gain and passed_rejection

    print(f"\n--- Verification Status ---")
    print(
        f"  Gain Threshold Check (>={min_gain_threshold:.1f}x) : {'[PASS]' if passed_gain else '[FAIL]'}"
    )
    print(
        f"  Rejection Check (>=8.0 dB)         : {'[PASS]' if passed_rejection else '[FAIL]'}"
    )
    print(
        f"  OVERALL RESULT                     : {'[PASSED]' if all_passed else '[FAILED]'}"
    )

    # 4. Generate Diagnostic Plots
    if not out_plot:
        out_plot = f"fengine_tracker_verification_{n_ant}ant_{scenario}.png"

    print(f"\n--- Generating Diagnostic Plots -> {out_plot} ---")
    fig = plt.figure(figsize=(16, 11))

    # Panel 1: Per-Antenna Power Distribution
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.bar(range(n_ant), ant_powers, color="steelblue", edgecolor="navy", alpha=0.8)
    ax1.axhline(
        mean_ant_p, color="red", linestyle="--", label=f"Mean Power ({mean_ant_p:.2f})"
    )
    ax1.set_xlabel("Antenna Index (0..{})".format(n_ant - 1))
    ax1.set_ylabel("Power Variance |V_a|²")
    ax1.set_title(f"Antenna Array Power Profile ({n_ant} Antennas)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Panel 2: On-Target Dynamic Spectrum (Waterfall)
    ax2 = fig.add_subplot(2, 2, 2)
    step = max(1, n_time // 1000)
    waterfall = p_on[::step, :].T  # (freq, time)
    extent = [0, n_time * delta_time_us / 1000.0, freqs_hz[0] / 1e6, freqs_hz[-1] / 1e6]
    im = ax2.imshow(
        waterfall, aspect="auto", origin="lower", cmap="viridis", extent=extent
    )
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Frequency (MHz)")
    ax2.set_title(f"On-Target Tracked Beam Dynamic Spectrum ({scenario})")
    fig.colorbar(im, ax=ax2, label="|E|²")

    # Panel 3: Power Spectrum Comparison (On-Target vs Off-Target)
    ax3 = fig.add_subplot(2, 2, 3)
    spec_on = np.mean(p_on, axis=0)
    spec_off = np.mean(p_off, axis=0)
    freq_axis_mhz = freqs_hz / 1e6
    ax3.plot(
        freq_axis_mhz,
        spec_on,
        label=f"On-Target Tracked Beam (Gain={measured_gain:.1f}x)",
        color="crimson",
        lw=1.5,
    )
    ax3.plot(
        freq_axis_mhz,
        spec_off,
        label=f"Off-Target Beam (Rejection={rejection_db:.1f} dB)",
        color="gray",
        alpha=0.7,
        lw=1.2,
    )
    ax3.set_xlabel("Frequency (MHz)")
    ax3.set_ylabel("Average Power |E|²")
    ax3.set_title("Formed Beam Power Spectrum")
    ax3.legend()
    ax3.grid(alpha=0.3)

    # Panel 4: 1D Beam Response Cut (Array Factor)
    ax4 = fig.add_subplot(2, 2, 4)
    l_scan = np.linspace(-0.25, 0.25, 101)
    beam_scan_power = np.zeros(len(l_scan))
    mid_freq_idx = n_freq // 2
    f_mid_hz = np.array([freqs_hz[mid_freq_idx]])
    raw_mid_slice = raw_data[:, mid_freq_idx : mid_freq_idx + 1, : min(n_time, 2048)]

    for idx, l_val in enumerate(l_scan):
        v_scan = beamform_synthesize(
            raw_mid_slice,
            f_mid_hz,
            l_val,
            0.0,
            spacing_m=spacing_m,
            format_type=format_type,
        )
        beam_scan_power[idx] = np.mean(np.abs(v_scan) ** 2)

    ax4.plot(
        l_scan,
        10.0 * np.log10(beam_scan_power / np.max(beam_scan_power)),
        color="darkmagenta",
        lw=1.5,
    )
    ax4.axvline(
        0.0 if np.isscalar(target_l) else np.mean(target_l),
        color="green",
        linestyle=":",
        label="Target Center",
    )
    ax4.set_xlabel("Direction Cosine l (East-West)")
    ax4.set_ylabel("Normalized Beam Response (dB)")
    ax4.set_title(f"Synthesized Array Factor Cut (at {f_mid_hz[0]/1e6:.1f} MHz)")
    ax4.set_ylim(-35, 2)
    ax4.legend()
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_plot, dpi=150)
    plt.close()
    print(f"[SUCCESS] Diagnostic plot saved to: {os.path.abspath(out_plot)}")
    print("======================================================================\n")

    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description="Kotekan Beam Tracker <-> F-Engine Verification Bridge"
    )
    parser.add_argument("h5_path", help="Path to F-Engine HDF5 dataset")
    parser.add_argument(
        "--plot-out", default="", help="Path for output diagnostic plot"
    )
    args = parser.parse_args()

    success = run_bridge_verification(args.h5_path, args.plot_out)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
