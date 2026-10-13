#!/usr/bin/env python3
"""
F-Engine Simulation Data Generator for Kotekan Beam Tracker Testing
===================================================================
Faithfully reproduces the radio telescope F-Engine physics from
RadioTelescopeFEngine.jl (antenna geometric delays, PFB channelization,
4-bit complex quantization int4x2_t) for CHARTS 64-antenna (8x8) and
256-antenna (16x16) arrays.

Outputs standard HDF5 baseband files and raw binary stream dumps.
"""

import os
import sys
import time
import argparse
import numpy as np
import h5py

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


def get_antenna_positions(num_antennas: int, spacing_m: float = DEFAULT_SPACING_M):
    """
    Computes (x, y) physical coordinates for 64 (8x8) or 256 (16x16) array.
    """
    if num_antennas <= 64:
        cols = np.arange(num_antennas) & 7
        rows = np.arange(num_antennas) >> 3
    else:
        cols = np.arange(num_antennas) & 15
        rows = np.arange(num_antennas) >> 4

    pos_x = cols * spacing_m
    pos_y = rows * spacing_m
    return pos_x, pos_y


def generate_fengine_data(
    num_antennas: int = 64,
    num_freq: int = LOCAL_FREQUENCY_CHANNELS,
    num_time: int = 15360,
    freq_start_mhz: float = DEFAULT_FREQUENCY_START_MHZ,
    delta_freq_mhz: float = CHARTS_CHANNEL_WIDTH_MHZ,
    delta_time_us: float = FPGA_TIME_RESOLUTION_US,
    spacing_m: float = DEFAULT_SPACING_M,
    scenario: str = "zenith",
    noise_amp: float = 0.5,
    source_amp: float = 3.0,
    target_l: float = 0.0,
    target_m: float = 0.0,
    drift_dl: float = 0.0,
    drift_dm: float = 0.0,
    seed: int = 42,
):
    """
    Generates complex baseband voltage data packed in 4-bit complex format (int4x2_t).

    Returns:
        packed_data: np.ndarray shape (num_time, num_freq, num_antennas) uint8
        freqs_hz: np.ndarray shape (num_freq,) float64
        meta: dict of scenario metadata
    """
    np.random.seed(seed)
    pos_x, pos_y = get_antenna_positions(num_antennas, spacing_m)
    freqs_hz = (freq_start_mhz + np.arange(num_freq) * delta_freq_mhz) * 1e6

    print(f"Generating F-Engine simulation:")
    print(
        f"  Antennas   : {num_antennas} ({'8x8' if num_antennas <= 64 else '16x16'} grid, spacing={spacing_m}m)"
    )
    print(
        f"  Frequencies: {num_freq} channels ({freq_start_mhz:.1f} to {freq_start_mhz + num_freq*delta_freq_mhz:.1f} MHz)"
    )
    print(
        f"  Time       : {num_time} samples ({num_time * delta_time_us / 1000.0:.2f} ms)"
    )
    print(f"  Scenario   : {scenario}")

    # Define sources based on scenario
    sources = []
    if scenario == "zenith":
        sources.append({"l0": 0.0, "m0": 0.0, "dl": 0.0, "dm": 0.0, "amp": source_amp})
    elif scenario == "off_zenith":
        l_val = target_l if target_l != 0.0 else 0.08
        m_val = target_m if target_m != 0.0 else -0.04
        sources.append(
            {"l0": l_val, "m0": m_val, "dl": 0.0, "dm": 0.0, "amp": source_amp}
        )
    elif scenario == "moving":
        l_val = target_l if target_l != 0.0 else 0.05
        m_val = target_m if target_m != 0.0 else 0.02
        dl_val = drift_dl if drift_dl != 0.0 else 1.0e-5
        dm_val = drift_dm if drift_dm != 0.0 else 0.5e-5
        sources.append(
            {"l0": l_val, "m0": m_val, "dl": dl_val, "dm": dm_val, "amp": source_amp}
        )
    elif scenario == "multisource":
        sources.append(
            {"l0": 0.06, "m0": 0.02, "dl": 0.0, "dm": 0.0, "amp": source_amp}
        )
        sources.append(
            {"l0": -0.07, "m0": -0.05, "dl": 0.0, "dm": 0.0, "amp": source_amp * 0.8}
        )
    elif scenario == "frb":
        # Dispersed FRB chirp
        sources.append(
            {
                "l0": 0.03,
                "m0": 0.01,
                "dl": 0.0,
                "dm": 0.0,
                "amp": source_amp,
                "is_frb": True,
            }
        )
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    dt_s = delta_time_us * 1e-6

    # Packed output array (time, freq, ant)
    packed_buffer = np.zeros((num_time, num_freq, num_antennas), dtype=np.uint8)

    # Process in time chunks for memory efficiency
    chunk_size = min(1024, num_time)
    n_chunks = (num_time + chunk_size - 1) // chunk_size

    t0_start = time.perf_counter()

    for c in range(n_chunks):
        t_start = c * chunk_size
        t_end = min(num_time, (c + 1) * chunk_size)
        t_len = t_end - t_start
        t_chunk = np.arange(t_start, t_end, dtype=np.float64)

        # Real and Imag components: (t_len, num_freq, num_antennas) in float32
        if noise_amp > 0.0:
            # Generate float32 noise directly without float64 intermediate
            v_real = (
                np.random.randn(t_len, num_freq, num_antennas) * noise_amp
            ).astype(np.float32)
            v_imag = (
                np.random.randn(t_len, num_freq, num_antennas) * noise_amp
            ).astype(np.float32)
        else:
            v_real = np.zeros((t_len, num_freq, num_antennas), dtype=np.float32)
            v_imag = np.zeros((t_len, num_freq, num_antennas), dtype=np.float32)

        # Pre-cast constants to float32
        freqs_hz_f32 = freqs_hz.astype(np.float32)
        pos_x_f32 = pos_x.astype(np.float32)
        pos_y_f32 = pos_y.astype(np.float32)
        c_inv = np.float32(1.0 / C_LIGHT)
        two_pi_f32 = np.float32(2.0 * np.pi)

        # Add Sources
        for src in sources:
            amp = np.float32(src["amp"])
            # Time-dependent direction cosines in float32
            l_t = (src["l0"] + src["dl"] * t_chunk).astype(np.float32)  # (t_len,)
            m_t = (src["m0"] + src["dm"] * t_chunk).astype(np.float32)  # (t_len,)

            # Geometric delays in float32: (t_len, num_antennas)
            delays = (np.outer(l_t, pos_x_f32) + np.outer(m_t, pos_y_f32)) * c_inv

            # Intrinsic source phase: base signal across time and frequency
            if src.get("is_frb", False):
                # Quadratic dispersion delay
                f_ref = freqs_hz[-1]
                dm_delays_s = (
                    4.1488e-3
                    * 100.0
                    * (1.0 / (freqs_hz / 1e9) ** 2 - 1.0 / (f_ref / 1e9) ** 2)
                )
                t_physical_s = (t_chunk * dt_s).astype(np.float32)
                pulse_center = np.float32((num_time * dt_s) * 0.4)
                t_diff = t_physical_s[:, None] - (
                    pulse_center + dm_delays_s[None, :].astype(np.float32)
                )
                pulse_width_s = np.float32(50.0 * dt_s)
                envelope = np.exp(-0.5 * (t_diff / pulse_width_s) ** 2).astype(
                    np.float32
                )

                phase_base = two_pi_f32 * (
                    np.outer((t_chunk * 0.01).astype(np.float32), freqs_hz_f32 / 1e8)
                )
                geom_phase = two_pi_f32 * (
                    delays[:, None, :] * freqs_hz_f32[None, :, None]
                )
                total_phase = phase_base[:, :, None] - geom_phase
                v_real += amp * envelope[:, :, None] * np.cos(total_phase)
                v_imag += amp * envelope[:, :, None] * np.sin(total_phase)
            else:
                phase_base = two_pi_f32 * (
                    np.outer(
                        (t_chunk * 0.005).astype(np.float32), (freqs_hz_f32 * 1e-8)
                    )
                )
                # Direct broadcasting (t_len, 1, num_ant) * (1, num_freq, 1) in float32
                geom_phase = two_pi_f32 * (
                    delays[:, None, :] * freqs_hz_f32[None, :, None]
                )
                total_phase = phase_base[:, :, None] - geom_phase
                v_real += amp * np.cos(total_phase)
                v_imag += amp * np.sin(total_phase)

        # 4-bit Quantization [-7..+7] (int4x2_t: real in low 4 bits, imag in high 4 bits)
        r_quant = np.clip(np.round(v_real), -7, 7).astype(np.int8)
        i_quant = np.clip(np.round(v_imag), -7, 7).astype(np.int8)

        r_nibble = (r_quant & 0x0F).astype(np.uint8)
        i_nibble = ((i_quant & 0x0F) << 4).astype(np.uint8)

        packed_buffer[t_start:t_end, :, :] = r_nibble | i_nibble

    t1_end = time.perf_counter()
    print(f"  Simulation completed in {(t1_end - t0_start)*1000.0:.2f} ms")

    meta = {
        "num_antennas": num_antennas,
        "num_freq": num_freq,
        "num_time": num_time,
        "freq_start_mhz": freq_start_mhz,
        "delta_freq_mhz": delta_freq_mhz,
        "delta_time_us": delta_time_us,
        "spacing_m": spacing_m,
        "scenario": scenario,
        "sources": sources,
    }

    return packed_buffer, freqs_hz, meta


def save_to_hdf5(
    filepath: str, packed_data: np.ndarray, freqs_hz: np.ndarray, meta: dict
):
    """
    Saves packed voltage data to HDF5 file matching CHORD/CHARTS metadata specifications.
    Layout in HDF5: [Antenna, Freq, Time] (shape: (num_ant, num_freq, num_time))
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    n_time, n_freq, n_ant = packed_data.shape

    # Transpose from [Time, Freq, Antenna] to [Antenna, Freq, Time] for standard HDF5 baseband dataset
    h5_array = np.transpose(packed_data, (2, 1, 0))

    if h5py is None:
        raise ImportError(
            "h5py is required to save HDF5 format. Install via: pip install h5py"
        )

    with h5py.File(filepath, "w") as f:
        # File attributes
        f.attrs["instrument"] = f"{n_ant}-antenna CHARTS"
        f.attrs["num_antennas"] = n_ant
        f.attrs["num_freq"] = n_freq
        f.attrs["num_time"] = n_time
        f.attrs["freq_start_MHz"] = float(meta["freq_start_mhz"])
        f.attrs["delta_freq_MHz"] = float(meta["delta_freq_mhz"])
        f.attrs["delta_time_us"] = float(meta["delta_time_us"])
        f.attrs["spacing_m"] = float(meta["spacing_m"])
        f.attrs["scenario"] = str(meta["scenario"])
        f.attrs["data_format"] = "complex_4bit_packed_int4x2"

        # Primary dataset
        dset = f.create_dataset(
            "baseband",
            data=h5_array,
            dtype=np.uint8,
            chunks=(n_ant, 1, min(n_time, 2048)),
        )
        dset.attrs["axes"] = ["antenna", "frequency", "time"]
        dset.attrs["units"] = "packed_int4x2_t"

        # Create RadioTelescopeFEngine compatibility group/dataset
        f_group = f.create_group("fengine_compat")
        v_dset = f_group.create_dataset(
            "voltage",
            data=np.expand_dims(h5_array, axis=1),  # (ant, pol=1, freq, time)
            dtype=np.uint8,
        )
        v_dset.attrs["dim_names"] = ["D", "P", "F", "T"]
        v_dset.attrs["feed_separation_x_m"] = float(meta["spacing_m"])
        v_dset.attrs["feed_separation_y_m"] = float(meta["spacing_m"])
        v_dset.attrs["coarse_freq"] = list(range(n_freq))

    print(
        f"[SUCCESS] Saved HDF5 dataset to: {os.path.abspath(filepath)} ({os.path.getsize(filepath)/(1024**2):.2f} MB)"
    )


def main():
    parser = argparse.ArgumentParser(
        description="CHARTS F-Engine Simulation Data Generator"
    )
    parser.add_argument(
        "--antennas",
        type=int,
        choices=[64, 256],
        default=64,
        help="Number of antennas (64 or 256)",
    )
    parser.add_argument(
        "--n-time",
        type=int,
        default=15360,
        help="Number of time samples (default: 15360)",
    )
    parser.add_argument(
        "--n-freq",
        type=int,
        default=336,
        help="Number of frequency channels (default: 336)",
    )
    parser.add_argument(
        "--scenario",
        choices=["zenith", "off_zenith", "moving", "multisource", "frb"],
        default="zenith",
        help="Simulation scenario",
    )
    parser.add_argument(
        "--target-l", type=float, default=0.0, help="Target direction cosine l"
    )
    parser.add_argument(
        "--target-m", type=float, default=0.0, help="Target direction cosine m"
    )
    parser.add_argument(
        "--drift-dl", type=float, default=0.0, help="Direction rate dl/sample"
    )
    parser.add_argument(
        "--drift-dm", type=float, default=0.0, help="Direction rate dm/sample"
    )
    parser.add_argument(
        "--noise-amp", type=float, default=0.5, help="Noise standard deviation"
    )
    parser.add_argument(
        "--source-amp", type=float, default=3.0, help="Point source amplitude"
    )
    parser.add_argument("--output", default="", help="Output HDF5 path")
    args = parser.parse_args()

    if not args.output:
        os.makedirs("test_charts/data", exist_ok=True)
        args.output = (
            f"test_charts/data/fengine_sim_{args.antennas}ant_{args.scenario}.h5"
        )

    packed_data, freqs_hz, meta = generate_fengine_data(
        num_antennas=args.antennas,
        num_freq=args.n_freq,
        num_time=args.n_time,
        scenario=args.scenario,
        noise_amp=args.noise_amp,
        source_amp=args.source_amp,
        target_l=args.target_l,
        target_m=args.target_m,
        drift_dl=args.drift_dl,
        drift_dm=args.drift_dm,
    )

    save_to_hdf5(args.output, packed_data, freqs_hz, meta)


if __name__ == "__main__":
    main()
