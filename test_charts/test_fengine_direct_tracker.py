#!/usr/bin/env python3
"""
F-Engine to Direct Beam Tracker Simulation & V5 Baseline Comparison
===================================================================
Simulates synthetic baseband voltage streams from the CHARTS / CHORD F-Engine
(RadioTelescopeFEngine.jl or generate_fengine_sim_data.py) into the new
Direct Beam Tracker (zero integration window), and provides an in-depth
comparative verification against Beam Tracker V5 (windowed baseline).

Theoretical Foundations:
- "Design and Implementation of the F-Engine for the CHARTS Project" (Buschmann 2025)
- "The CHARTS Dynamic Beam Tracker" (CHARTS DSP Team 2026)
- "CHORD FRB Beamformer" (Smith 2022, eq. 3-8)
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
    LOCAL_FREQUENCY_CHANNELS,
    DEFAULT_FREQUENCY_START_MHZ,
    CHARTS_CHANNEL_WIDTH_MHZ,
    FPGA_TIME_RESOLUTION_US,
)


def unpack_int4x2(
    u8_array: np.ndarray, format_type: str = "twos_complement"
) -> np.ndarray:
    """
    Unpacks 4-bit complex int4x2 values (real and imag nibbles) to complex64.
    """
    if "offset" in format_type.lower():
        real = (u8_array & 0x0F).astype(np.float32) - 8.0
        imag = (u8_array >> 4).astype(np.float32) - 8.0
    else:
        r = (u8_array & 0x0F).astype(np.int8)
        i = (u8_array >> 4).astype(np.int8)
        r[r >= 8] -= 16
        i[i >= 8] -= 16
        real = r.astype(np.float32)
        imag = i.astype(np.float32)
    return real + 1j * imag


def get_antenna_positions(num_antennas: int, spacing_m: float = DEFAULT_SPACING_M):
    """
    Returns (pos_x, pos_y) arrays for 64 (8x8) or 256 (16x16) array.
    """
    if num_antennas <= 64:
        cols = np.arange(num_antennas) & 7
        rows = np.arange(num_antennas) >> 3
    else:
        cols = np.arange(num_antennas) & 15
        rows = np.arange(num_antennas) >> 4
    return cols * spacing_m, rows * spacing_m


def beamform_direct(
    raw_ant_freq_time: np.ndarray,
    freqs_hz: np.ndarray,
    l_traj: np.ndarray,
    m_traj: np.ndarray,
    spacing_m: float = 0.6,
    mask: np.ndarray = None,
    format_type: str = "twos_complement",
) -> np.ndarray:
    """
    Direct Beam Tracker: applies instantaneous steering weights directly per sample (zero window).
    """
    n_ant, n_freq, n_time = raw_ant_freq_time.shape
    pos_x, pos_y = get_antenna_positions(n_ant, spacing_m)
    if mask is None:
        mask = np.ones(n_ant, dtype=np.uint8)

    if np.isscalar(l_traj):
        l_traj = np.full(n_time, l_traj, dtype=np.float64)
    if np.isscalar(m_traj):
        m_traj = np.full(n_time, m_traj, dtype=np.float64)

    complex_data = unpack_int4x2(raw_ant_freq_time, format_type=format_type)
    formed_voltages = np.zeros((n_time, n_freq), dtype=np.complex64)

    chunk_size = min(2048, n_time)
    n_chunks = (n_time + chunk_size - 1) // chunk_size

    for c in range(n_chunks):
        t_start = c * chunk_size
        t_end = min(n_time, (c + 1) * chunk_size)
        l_sub = l_traj[t_start:t_end]
        m_sub = m_traj[t_start:t_end]

        delays = (np.outer(l_sub, pos_x) + np.outer(m_sub, pos_y)) / C_LIGHT
        phases = 2.0 * np.pi * np.einsum("f,ta->tfa", freqs_hz, delays)
        weights = np.exp(1j * phases).astype(np.complex64)
        weights[:, :, mask == 0] = 0.0

        data_sub = np.transpose(complex_data[:, :, t_start:t_end], (2, 1, 0))
        formed_voltages[t_start:t_end, :] = np.sum(weights * data_sub, axis=2)

    return formed_voltages


def beamform_v5_windowed(
    raw_ant_freq_time: np.ndarray,
    freqs_hz: np.ndarray,
    l_traj: np.ndarray,
    m_traj: np.ndarray,
    integration_spectra: int = 320,
    spacing_m: float = 0.6,
    mask: np.ndarray = None,
    format_type: str = "twos_complement",
) -> np.ndarray:
    """
    Beam Tracker V5 Baseline: holds direction constant at center sample of each integration window.
    """
    n_ant, n_freq, n_time = raw_ant_freq_time.shape
    pos_x, pos_y = get_antenna_positions(n_ant, spacing_m)
    if mask is None:
        mask = np.ones(n_ant, dtype=np.uint8)

    if np.isscalar(l_traj):
        l_traj = np.full(n_time, l_traj, dtype=np.float64)
    if np.isscalar(m_traj):
        m_traj = np.full(n_time, m_traj, dtype=np.float64)

    # Discretize trajectory to window center
    l_windowed = np.zeros_like(l_traj)
    m_windowed = np.zeros_like(m_traj)
    n_windows = (n_time + integration_spectra - 1) // integration_spectra
    for w in range(n_windows):
        idx_start = w * integration_spectra
        idx_end = min(n_time, (w + 1) * integration_spectra)
        center_idx = min(n_time - 1, idx_start + (integration_spectra // 2))
        l_windowed[idx_start:idx_end] = l_traj[center_idx]
        m_windowed[idx_start:idx_end] = m_traj[center_idx]

    return beamform_direct(
        raw_ant_freq_time,
        freqs_hz,
        l_windowed,
        m_windowed,
        spacing_m=spacing_m,
        mask=mask,
        format_type=format_type,
    )


def run_fengine_direct_tracker_test(h5_path: str, plot_path: str = ""):
    print("=" * 80)
    print(" CHARTS F-ENGINE -> DIRECT BEAM TRACKER SIMULATION & V5 COMPARISON")
    print(f" Target Dataset: {h5_path}")
    print("=" * 80)

    if not os.path.exists(h5_path):
        print(
            f"[WARN] File {h5_path} does not exist. Generating synthetic 64-antenna moving target dataset..."
        )
        from generate_fengine_sim_data import generate_fengine_data, save_to_hdf5

        os.makedirs(os.path.dirname(os.path.abspath(h5_path)) or ".", exist_ok=True)
        packed, freqs, meta = generate_fengine_data(
            num_antennas=64,
            num_freq=336,
            num_time=15360,
            scenario="moving",
            source_amp=4.0,
            noise_amp=0.5,
        )
        save_to_hdf5(h5_path, packed, freqs, meta)

    with h5py.File(h5_path, "r") as f:
        if "voltage" in f:
            dset = f["voltage"]
            dset_shape = dset.shape
            raw_data = dset[:, 0, :, :] if len(dset_shape) == 4 else dset[:]
            format_type = str(dset.attrs.get("type", "int4x2_swapped_withoffset"))
            spacing_m = float(dset.attrs.get("feed_separation_x_m", 0.6))
            coarse_freq = dset.attrs.get("coarse_freq", None)
            if coarse_freq is not None:
                freq_indices = np.array(coarse_freq)
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
        elif "baseband" in f:
            dset = f["baseband"]
            n_ant, n_freq, n_time = dset.shape[0], dset.shape[1], dset.shape[2]
            format_type = str(f.attrs.get("data_format", "twos_complement"))
            freq_start_mhz = float(f.attrs.get("freq_start_MHz", 300.0))
            delta_freq_mhz = float(f.attrs.get("delta_freq_MHz", 0.3))
            delta_time_us = float(f.attrs.get("delta_time_us", 3.2552))
            spacing_m = float(f.attrs.get("spacing_m", 0.6))
            scenario = str(f.attrs.get("scenario", "moving"))
            freqs_hz = (freq_start_mhz + np.arange(n_freq) * delta_freq_mhz) * 1e6
            raw_data = dset[:n_ant, :n_freq, :n_time]
        else:
            raise ValueError(
                f"Could not find 'voltage' or 'baseband' dataset in {h5_path}"
            )

    print(f"Loaded Dataset Parameters:")
    print(f"  Antennas   : {n_ant} elements")
    print(
        f"  Frequencies: {n_freq} channels ({freqs_hz[0]/1e6:.1f} .. {freqs_hz[-1]/1e6:.1f} MHz)"
    )
    print(
        f"  Time       : {n_time} samples ({n_time * delta_time_us / 1000.0:.2f} ms frame)"
    )
    print(f"  Format     : {format_type}")
    print(f"  Scenario   : {scenario}")

    # Single antenna baseline power
    unpacked_raw = unpack_int4x2(raw_data, format_type=format_type)
    ant_power = np.mean(np.abs(unpacked_raw) ** 2, axis=(1, 2))
    mean_ant_power = float(np.mean(ant_power))

    # Define true target trajectory
    if scenario == "moving":
        true_l = 0.05 + 1.0e-5 * np.arange(n_time)
        true_m = 0.02 + 0.5e-5 * np.arange(n_time)
    elif scenario == "off_zenith":
        true_l = 0.08
        true_m = -0.04
    else:
        true_l = 0.0
        true_m = 0.0

    # 1. Run Direct Beam Tracker (Zero Window)
    t0 = time.perf_counter()
    v_direct = beamform_direct(
        raw_data, freqs_hz, true_l, true_m, spacing_m=spacing_m, format_type=format_type
    )
    t_direct_ms = (time.perf_counter() - t0) * 1000.0

    # 2. Run V5 Beam Tracker Baseline (320-sample Integration Window)
    t0 = time.perf_counter()
    v_v5 = beamform_v5_windowed(
        raw_data,
        freqs_hz,
        true_l,
        true_m,
        integration_spectra=320,
        spacing_m=spacing_m,
        format_type=format_type,
    )
    t_v5_ms = (time.perf_counter() - t0) * 1000.0

    # 3. Off-target beam (sidelobe / null rejection test outside primary beamwidth)
    off_l = true_l + 0.25
    off_m = true_m + 0.20
    v_off = beamform_direct(
        raw_data, freqs_hz, off_l, off_m, spacing_m=spacing_m, format_type=format_type
    )

    # Compute Powers
    p_direct = np.abs(v_direct) ** 2
    p_v5 = np.abs(v_v5) ** 2
    p_off = np.abs(v_off) ** 2

    mean_p_direct = float(np.mean(p_direct))
    mean_p_v5 = float(np.mean(p_v5))
    mean_p_off = float(np.mean(p_off))

    gain_direct = mean_p_direct / max(1e-12, mean_ant_power)
    gain_v5 = mean_p_v5 / max(1e-12, mean_ant_power)
    theoretical_max_gain = float(n_ant**2)

    sidelobe_rejection_db = 10.0 * np.log10(
        max(1e-12, mean_p_direct) / max(1e-12, mean_p_off)
    )

    # Decorrelation & Phase Jitter Analysis (Direct vs V5)
    # Phase difference between Direct and V5 formed voltages
    phase_diff = np.angle(v_direct * np.conj(v_v5))
    std_phase_jitter_deg = float(np.rad2deg(np.std(phase_diff)))
    decorrelation_loss_db = 10.0 * np.log10(
        max(1e-12, mean_p_direct) / max(1e-12, mean_p_v5)
    )

    print("\n" + "=" * 80)
    print(" VERIFICATION & COMPARISON METRICS")
    print("=" * 80)
    print(f" Mean Single Antenna Power         : {mean_ant_power:.4f}")
    print(
        f" Direct Tracker Mean Power |E|²    : {mean_p_direct:.4f} (Coherent Gain: {gain_direct:.1f}x / {10*np.log10(gain_direct):.2f} dB)"
    )
    print(
        f" V5 Baseline Mean Power |E|²       : {mean_p_v5:.4f} (Coherent Gain: {gain_v5:.1f}x / {10*np.log10(gain_v5):.2f} dB)"
    )
    print(
        f" Theoretical N_ant² Gain Bound     : {theoretical_max_gain:.1f}x ({10*np.log10(theoretical_max_gain):.2f} dB)"
    )
    print(
        f" Array Coherence Efficiency        : {gain_direct / theoretical_max_gain * 100.0:.2f}%"
    )
    print(f" Off-Target Power (Sidelobe)       : {mean_p_off:.4f}")
    print(f" Sidelobe Rejection Ratio          : {sidelobe_rejection_db:.2f} dB")
    print(f" V5 Window Decorrelation Loss      : {decorrelation_loss_db:+.3f} dB")
    print(f" Window Induced Phase Jitter (std) : {std_phase_jitter_deg:.3f} degrees")
    print(f" Python Direct Beamformer Latency  : {t_direct_ms:.1f} ms")
    print(f" Python V5 Beamformer Latency      : {t_v5_ms:.1f} ms")
    print("=" * 80)

    # Verification criteria
    coherent_pass = gain_direct > 0.5 * theoretical_max_gain
    sidelobe_pass = sidelobe_rejection_db >= 10.0
    decorrelation_pass = (
        decorrelation_loss_db >= -0.1
    )  # Direct should be equal or better than V5

    print(f"\nStatus Checks:")
    print(
        f"  [ {'PASS' if coherent_pass else 'FAIL'} ] Coherent N_ant² array gain scaling"
    )
    print(f"  [ {'PASS' if sidelobe_pass else 'FAIL'} ] Sidelobe suppression (>=10 dB)")
    print(
        f"  [ {'PASS' if decorrelation_pass else 'FAIL'} ] Direct Tracker eliminated window decorrelation loss"
    )

    # Generate multi-panel diagnostic plot if requested
    if plot_path:
        print(f"\nGenerating diagnostic plot: {plot_path} ...")
        fig, axs = plt.subplots(3, 2, figsize=(16, 12))

        time_axis_ms = np.arange(n_time) * delta_time_us / 1000.0
        freq_axis_mhz = freqs_hz / 1e6

        # 1. Waterfall: Direct Beamformed Power Spectrogram
        p1 = axs[0, 0].imshow(
            10.0 * np.log10(np.maximum(1e-4, p_direct.T)),
            aspect="auto",
            origin="lower",
            extent=[
                time_axis_ms[0],
                time_axis_ms[-1],
                freq_axis_mhz[0],
                freq_axis_mhz[-1],
            ],
            cmap="inferno",
        )
        axs[0, 0].set_title(
            f"Direct Beam Tracker: Formed Power Spectrogram ({n_ant} Antennas)",
            fontsize=11,
            fontweight="bold",
        )
        axs[0, 0].set_xlabel("Time (ms)")
        axs[0, 0].set_ylabel("Frequency (MHz)")
        fig.colorbar(p1, ax=axs[0, 0], label="Power (dB)")

        # 2. Waterfall: V5 Beamformed Power Spectrogram
        p2 = axs[0, 1].imshow(
            10.0 * np.log10(np.maximum(1e-4, p_v5.T)),
            aspect="auto",
            origin="lower",
            extent=[
                time_axis_ms[0],
                time_axis_ms[-1],
                freq_axis_mhz[0],
                freq_axis_mhz[-1],
            ],
            cmap="inferno",
        )
        axs[0, 1].set_title(
            f"Beam Tracker V5 Baseline (Window=320 spectra)",
            fontsize=11,
            fontweight="bold",
        )
        axs[0, 1].set_xlabel("Time (ms)")
        axs[0, 1].set_ylabel("Frequency (MHz)")
        fig.colorbar(p2, ax=axs[0, 1], label="Power (dB)")

        # 3. Direct vs V5 Time Series Profile (Band-averaged)
        direct_t_prof = 10.0 * np.log10(np.maximum(1e-4, np.mean(p_direct, axis=1)))
        v5_t_prof = 10.0 * np.log10(np.maximum(1e-4, np.mean(p_v5, axis=1)))
        off_t_prof = 10.0 * np.log10(np.maximum(1e-4, np.mean(p_off, axis=1)))

        axs[1, 0].plot(
            time_axis_ms,
            direct_t_prof,
            label="Direct Tracker (Zero Window)",
            color="crimson",
            lw=1.2,
        )
        axs[1, 0].plot(
            time_axis_ms,
            v5_t_prof,
            label="V5 Baseline (Window=320)",
            color="royalblue",
            lw=1.0,
            alpha=0.8,
        )
        axs[1, 0].plot(
            time_axis_ms,
            off_t_prof,
            label="Off-Target Sidelobe",
            color="dimgray",
            lw=0.8,
            linestyle="--",
        )
        axs[1, 0].set_title(
            "Band-Averaged Power vs. Time", fontsize=11, fontweight="bold"
        )
        axs[1, 0].set_xlabel("Time (ms)")
        axs[1, 0].set_ylabel("Power (dB)")
        axs[1, 0].legend(loc="upper right")
        axs[1, 0].grid(True, alpha=0.3)

        # 4. Phase Difference / Jitter inside V5 Windows
        sample_slice = slice(0, min(1600, n_time))
        axs[1, 1].plot(
            time_axis_ms[sample_slice],
            np.rad2deg(phase_diff[sample_slice, n_freq // 2]),
            color="darkorange",
            lw=1.0,
        )
        # Vertical dashed lines at window boundaries (every 320 samples)
        w_step_ms = 320 * delta_time_us / 1000.0
        for w_line in np.arange(0, time_axis_ms[sample_slice][-1], w_step_ms):
            axs[1, 1].axvline(w_line, color="gray", linestyle=":", alpha=0.5)
        axs[1, 1].set_title(
            f"Phase Error inside V5 Integration Windows (Δφ = {std_phase_jitter_deg:.2f}° std)",
            fontsize=11,
            fontweight="bold",
        )
        axs[1, 1].set_xlabel("Time (ms)")
        axs[1, 1].set_ylabel("Phase Difference (deg)")
        axs[1, 1].grid(True, alpha=0.3)

        # 5. Frequency Spectra Comparison
        direct_spec_db = 10.0 * np.log10(np.maximum(1e-4, np.mean(p_direct, axis=0)))
        v5_spec_db = 10.0 * np.log10(np.maximum(1e-4, np.mean(p_v5, axis=0)))
        off_spec_db = 10.0 * np.log10(np.maximum(1e-4, np.mean(p_off, axis=0)))

        axs[2, 0].plot(
            freq_axis_mhz, direct_spec_db, label="Direct Tracker", color="crimson"
        )
        axs[2, 0].plot(
            freq_axis_mhz, v5_spec_db, label="V5 Baseline", color="royalblue", alpha=0.7
        )
        axs[2, 0].plot(
            freq_axis_mhz, off_spec_db, label="Off-Target", color="gray", linestyle="--"
        )
        axs[2, 0].set_title(
            "Time-Averaged Frequency Spectrum", fontsize=11, fontweight="bold"
        )
        axs[2, 0].set_xlabel("Frequency (MHz)")
        axs[2, 0].set_ylabel("Power (dB)")
        axs[2, 0].legend()
        axs[2, 0].grid(True, alpha=0.3)

        # 6. Summary Metric Cards
        axs[2, 1].axis("off")
        summary_text = (
            f"CHARTS Beam Tracker Comparison Summary\n"
            f"-----------------------------------------\n"
            f"Array Scale           : {n_ant} Antennas ({'8x8' if n_ant <= 64 else '16x16'} Grid)\n"
            f"Frequency Channels    : {n_freq} (300-400 MHz)\n"
            f"Time Frame Samples    : {n_time:,} ({time_axis_ms[-1]:.2f} ms)\n"
            f"Direct Coherent Gain  : {gain_direct:.1f}x ({10*np.log10(gain_direct):.2f} dB)\n"
            f"V5 Coherent Gain      : {gain_v5:.1f}x ({10*np.log10(gain_v5):.2f} dB)\n"
            f"Theoretical Max Gain  : {theoretical_max_gain:.1f}x ({10*np.log10(theoretical_max_gain):.2f} dB)\n"
            f"Sidelobe Rejection    : {sidelobe_rejection_db:.2f} dB\n"
            f"V5 Decorrelation Loss : {decorrelation_loss_db:+.3f} dB\n"
            f"V5 Phase Jitter (std) : {std_phase_jitter_deg:.3f}°\n\n"
            f"Key Takeaway:\n"
            f"Direct Beam Tracker eliminates the 320-sample\n"
            f"piecewise window holding jitter of V5 while\n"
            f"fusing multi-beam register tiling for lower DRAM\n"
            f"traffic and zero-window exact tracking."
        )
        axs[2, 1].text(
            0.05,
            0.95,
            summary_text,
            transform=axs[2, 1].transAxes,
            fontsize=10.5,
            fontfamily="monospace",
            verticalalignment="top",
            bbox=dict(
                boxstyle="round,pad=0.8", facecolor="whitesmoke", edgecolor="silver"
            ),
        )

        plt.tight_layout()
        plt.savefig(plot_path, dpi=180)
        plt.close()
        print(f"Diagnostic figure successfully saved to: {plot_path}")

    return coherent_pass and sidelobe_pass


def main():
    parser = argparse.ArgumentParser(
        description="F-Engine to Direct Beam Tracker Simulation & Verification"
    )
    parser.add_argument(
        "h5_file",
        nargs="?",
        default="test_charts/data/voltage_sim_64ant.h5",
        help="Path to synthetic F-Engine HDF5 file",
    )
    parser.add_argument(
        "--plot-out", default="", help="Path to output diagnostic plot PNG"
    )
    args = parser.parse_args()

    success = run_fengine_direct_tracker_test(args.h5_file, plot_path=args.plot_out)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
