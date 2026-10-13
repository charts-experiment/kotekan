#!/usr/bin/env python3
"""
CHARTS 64-Antenna F-Engine Baseband Dump Generator for Trillium Simulation
=========================================================================
Simulates physical complex baseband voltage streams for the CHARTS 64-antenna
array (8x8 uniform rectangular grid, dx=dy=0.6m, Carén Observatory, Chile).

Comprehensive Astronomical & Physical Target Catalog:
  1. Calibradores Primarios y Fuentes Galácticas:
     - 'sgr_a'     : Sagittarius A* / Centro Galáctico (transita a +4.4° N del cenit)
     - 'cen_a'     : Centaurus A / NGC 5128 (~1000 Jy, transita a -9.6° S del cenit)
     - 'crab'      : Taurus A / Nebulosa del Cangrejo (PSR B0531+21, culminación el 34.6° N)
     - 'pictor_a'  : Pictor A (~400 Jy, transita a -12.4° S del cenit)
     - 'puppis_a'  : Puppis A (SNR extendido cerca de Vela, -9.6° S del cenit)
     - 'vela'      : Pulsar de Vela (PSR J0835-4510, -11.8° S del cenit)
     - 'sun'       : El Sol (tránsito meridiano, flujo dominante)

  2. Pulsares del Cielo Austral (Dispersión y Timing):
     - 'psr_j0437' : PSR J0437-4715 (MSP más brillante, DM=2.64 pc/cm^3)
     - 'psr_j1644' : PSR J1644-4559 / B1641-45 (~1 Jy, alta dispersión DM=478 pc/cm^3)
     - 'psr_j0737' : PSR J0737-3039A/B (Doble Pulsar, transita a +2.76° N del cenit)

  3. Fuentes Transitorias y Artificiales:
     - 'zenith'    : Cenit Local Puro / Campo Blanco (l=0, m=0, fase geométrica tau_i = 0)
     - 'frb'       : Fast Radio Burst sintético dispersado (pulso de 2 ms, DM=300 pc/cm^3)
     - 'rfi_leo'   : Satélite LEO / RFI rápida barriendo lóbulos del arreglo

Noise & Saturation Variants:
  - Any target can be generated as '[target]_no_noise' or '[target]_with_noise'.
  - Saturated variants (e.g. 'sun_with_noise_saturated', 'rfi_leo_saturated') force
    specific antennas into ADC rail-clipping ([-7, +7]).

Chunking / Time Windows:
  - 5.12 ms per frame (1536 samples @ dt = 10/3 us).
  - 1536 is divisible by 32, required by John Romein's Tensor Core correlator.

Outputs:
  - Kotekan rawFileRead binary file (*.bin) with [uint32 metadata_size=0][payload: time, freq, ant].
  - CHORD/CHARTS standard HDF5 baseband file (*.h5) with complete astronomical & site metadata.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np

# Ensure test_charts path is available
_test_charts_dir = Path(__file__).resolve().parent
_kotekan_root = _test_charts_dir.parent
if str(_test_charts_dir) not in sys.path:
    sys.path.insert(0, str(_test_charts_dir))

from constants import (
    C_LIGHT,
    CHARTS_ALTITUDE_M,
    CHARTS_CHANNEL_WIDTH_MHZ,
    CHARTS_LATITUDE_DEG,
    CHARTS_LONGITUDE_DEG,
    DEFAULT_FREQUENCY_START_MHZ,
    DEFAULT_SPACING_M,
    FPGA_TIME_RESOLUTION_US,
    K_DM,
    LOCAL_FREQUENCY_CHANNELS,
)

# ---------------------------------------------------------------------------
# Comprehensive Astronomical Catalog for Observatorio Carén (Lat = -33.4211°)
# ---------------------------------------------------------------------------
ASTRONOMICAL_CATALOG: Dict[str, Dict[str, Any]] = {
    # 1. Primary Calibrators & Galactic Sources
    "vela": {
        "name": "Vela Pulsar (PSR J0835-4510)",
        "ra_deg": 128.836,
        "dec_deg": -45.176,
        "category": "calibrator",
        "nominal_amp": 3.0,
        "description": "Brightest southern radio pulsar, transit 11.76 deg S of zenith",
    },
    "sun": {
        "name": "The Sun (Solar Transit)",
        "ra_deg": 0.0,
        "dec_deg": 0.0,
        "category": "solar",
        "nominal_amp": 5.5,
        "description": "Dominant celestial radio source, transit 33.42 deg N of zenith",
    },
    "sgr_a": {
        "name": "Sagittarius A* / Galactic Center",
        "ra_deg": 266.417,
        "dec_deg": -29.008,
        "category": "calibrator",
        "nominal_amp": 4.5,
        "description": "Supermassive black hole & Galactic Center, transits 4.41 deg N of zenith",
    },
    "cen_a": {
        "name": "Centaurus A (NGC 5128)",
        "ra_deg": 201.365,
        "dec_deg": -43.019,
        "category": "calibrator",
        "nominal_amp": 4.0,
        "description": "Giant radio galaxy (~1000 Jy), transits 9.60 deg S of zenith",
    },
    "crab": {
        "name": "Taurus A / Crab Nebula (PSR B0531+21)",
        "ra_deg": 83.633,
        "dec_deg": +22.014,
        "category": "calibrator",
        "nominal_amp": 3.5,
        "description": "Supernova remnant & pulsar calibrator (culmination el 34.6 deg North)",
    },
    "pictor_a": {
        "name": "Pictor A",
        "ra_deg": 79.958,
        "dec_deg": -45.779,
        "category": "calibrator",
        "nominal_amp": 3.2,
        "description": "Southern giant radio galaxy (~400 Jy), transits 12.36 deg S of zenith",
    },
    "puppis_a": {
        "name": "Puppis A",
        "ra_deg": 125.617,
        "dec_deg": -42.983,
        "category": "calibrator",
        "nominal_amp": 3.2,
        "description": "Bright extended SNR, transits 9.56 deg S of zenith",
    },
    # 2. Southern Pulsars (Dispersion & Timing)
    "psr_j0437": {
        "name": "PSR J0437-4715",
        "ra_deg": 69.316,
        "dec_deg": -47.252,
        "category": "pulsar",
        "nominal_amp": 2.8,
        "dm": 2.64,
        "description": "Brightest millisecond pulsar (MSP), ultra-low DM=2.64 pc/cm^3",
    },
    "psr_j1644": {
        "name": "PSR J1644-4559 (PSR B1641-45)",
        "ra_deg": 251.205,
        "dec_deg": -45.987,
        "category": "pulsar",
        "nominal_amp": 2.5,
        "dm": 478.0,
        "description": "Bright southern galactic plane pulsar, high DM=478.0 pc/cm^3",
    },
    "psr_j0737": {
        "name": "PSR J0737-3039A/B (Double Pulsar)",
        "ra_deg": 114.463,
        "dec_deg": -30.661,
        "category": "pulsar",
        "nominal_amp": 2.2,
        "dm": 48.9,
        "description": "Relativistic double pulsar, transits 2.76 deg N of zenith",
    },
    # 3. Transient & Artificial Sources
    "zenith": {
        "name": "Caren Zenith Transit Field",
        "ra_deg": 24.346,
        "dec_deg": CHARTS_LATITUDE_DEG,
        "category": "zenith",
        "nominal_amp": 3.0,
        "description": "Pure on-axis field (l=0, m=0, tau_i=0 for all antennas)",
    },
    "frb": {
        "name": "Synthetic Fast Radio Burst (FRB)",
        "ra_deg": 150.0,
        "dec_deg": -35.0,
        "category": "transient",
        "nominal_amp": 6.0,
        "dm": 300.0,
        "is_transient": True,
        "description": "Dispersed 2 ms pulse chirp (DM=300 pc/cm^3)",
    },
    "rfi_leo": {
        "name": "RFI / Low Earth Orbit (LEO) Satellite",
        "ra_deg": 0.0,
        "dec_deg": CHARTS_LATITUDE_DEG,
        "category": "artificial",
        "nominal_amp": 15.0,
        "drift_dl": 5.0e-5,
        "drift_dm": 2.5e-5,
        "description": "Fast-moving, high-amplitude satellite RFI sweeping across beams",
    },
}

# Default saturated antennas for saturation scenarios
DEFAULT_SATURATED_ANTENNAS = [7, 23, 42, 55]


def compute_transit_direction_cosines(
    dec_deg: float, lat_deg: float = CHARTS_LATITUDE_DEG
) -> Tuple[float, float]:
    """
    Computes topocentric direction cosines (l, m) at meridian transit for Carén Observatory.
    At transit:
      l = 0 (aligned with local meridian)
      m = sin(Dec - Lat)
    """
    delta_rad = math.radians(dec_deg - lat_deg)
    l = 0.0
    m = math.sin(delta_rad)
    return float(l), float(m)


def get_charts_64_antenna_positions(
    num_antennas: int = 64, spacing_m: float = DEFAULT_SPACING_M
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes physical (x, y) coordinates for 64-antenna (8x8) CHARTS array."""
    cols = np.arange(num_antennas) & 7
    rows = np.arange(num_antennas) >> 3
    pos_x = (cols * spacing_m).astype(np.float64)
    pos_y = (rows * spacing_m).astype(np.float64)
    return pos_x, pos_y


def parse_scenario_name(scenario: str) -> Tuple[str, bool, bool]:
    """
    Parses a scenario name into (target_key, use_noise, saturate).
    Examples:
      'sgr_a_no_noise'             -> ('sgr_a', False, False)
      'sgr_a_with_noise'           -> ('sgr_a', True, False)
      'sun_with_noise_saturated'   -> ('sun', True, True)
      'rfi_leo_saturated'          -> ('rfi_leo', True, True)
      'frb'                        -> ('frb', True, False)
    """
    s = scenario.lower()
    saturate = "saturat" in s
    if "_no_noise" in s:
        use_noise = False
        target_key = (
            s.replace("_no_noise", "")
            .replace("_saturated", "")
            .replace("saturated_", "")
        )
    elif "_with_noise" in s:
        use_noise = True
        target_key = (
            s.replace("_with_noise", "")
            .replace("_saturated", "")
            .replace("saturated_", "")
        )
    elif "saturated" in s:
        use_noise = True
        target_key = s.replace("_saturated", "").replace("saturated_", "")
    else:
        # Default single target name: use noise by default
        use_noise = True
        target_key = s

    if target_key not in ASTRONOMICAL_CATALOG:
        # Check alias
        alias_map = {
            "sgr_a*": "sgr_a",
            "sgra": "sgr_a",
            "cen_a": "cen_a",
            "cena": "cen_a",
            "crab_nebula": "crab",
            "taurus_a": "crab",
            "double_pulsar": "psr_j0737",
        }
        target_key = alias_map.get(target_key, target_key)

    if target_key not in ASTRONOMICAL_CATALOG:
        raise ValueError(
            f"Unknown target '{target_key}' parsed from scenario '{scenario}'. "
            f"Available targets: {list(ASTRONOMICAL_CATALOG.keys())}"
        )

    return target_key, use_noise, saturate


def generate_scenario_baseband(
    scenario: str,
    num_antennas: int = 64,
    num_freq: int = LOCAL_FREQUENCY_CHANNELS,  # 336
    samples_per_frame: int = 1536,  # 5.12 ms (1536 * 3.333 us)
    num_frames: int = 1,
    freq_start_mhz: float = DEFAULT_FREQUENCY_START_MHZ,
    delta_freq_mhz: float = CHARTS_CHANNEL_WIDTH_MHZ,
    delta_time_us: float = FPGA_TIME_RESOLUTION_US,
    spacing_m: float = DEFAULT_SPACING_M,
    saturated_antennas: List[int] | None = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Generates complex baseband voltage frames packed as 4-bit complex (int4x2_t).
    """
    target_key, use_noise, saturate = parse_scenario_name(scenario)
    target = ASTRONOMICAL_CATALOG[target_key]

    rng = np.random.default_rng(seed)
    pos_x, pos_y = get_charts_64_antenna_positions(num_antennas, spacing_m)
    freqs_hz = (freq_start_mhz + np.arange(num_freq) * delta_freq_mhz) * 1e6
    c_inv = np.float32(1.0 / C_LIGHT)
    two_pi_f32 = np.float32(2.0 * np.pi)
    dt_s = delta_time_us * 1e-6

    if saturated_antennas is None:
        saturated_antennas = DEFAULT_SATURATED_ANTENNAS if saturate else []

    # Per-antenna independent thermal noise variance (T_sys variations across receivers)
    if use_noise:
        antenna_noise_sigmas = rng.uniform(0.40, 0.70, size=num_antennas).astype(
            np.float32
        )
    else:
        antenna_noise_sigmas = np.zeros(num_antennas, dtype=np.float32)

    # Base direction cosines
    l0, m0 = compute_transit_direction_cosines(target["dec_deg"], CHARTS_LATITUDE_DEG)
    drift_dl = float(target.get("drift_dl", 0.0))
    drift_dm = float(target.get("drift_dm", 0.0))
    source_amp = float(target["nominal_amp"])

    # Dispersion Measure (DM) modeling
    dm = float(target.get("dm", 0.0))
    is_frb = target.get("is_transient", False)

    packed_frames = np.zeros(
        (num_frames, samples_per_frame, num_freq, num_antennas), dtype=np.uint8
    )

    t0_start = time.perf_counter()

    for frame_idx in range(num_frames):
        global_t_offset = frame_idx * samples_per_frame
        t_indices = np.arange(
            global_t_offset, global_t_offset + samples_per_frame, dtype=np.float32
        )

        # 1. Background thermal noise: independent per antenna
        if use_noise:
            noise_r = rng.normal(
                0.0, 1.0, size=(samples_per_frame, num_freq, num_antennas)
            ).astype(np.float32)
            noise_i = rng.normal(
                0.0, 1.0, size=(samples_per_frame, num_freq, num_antennas)
            ).astype(np.float32)
            v_real = noise_r * antenna_noise_sigmas[None, None, :]
            v_imag = noise_i * antenna_noise_sigmas[None, None, :]
        else:
            v_real = np.zeros(
                (samples_per_frame, num_freq, num_antennas), dtype=np.float32
            )
            v_imag = np.zeros(
                (samples_per_frame, num_freq, num_antennas), dtype=np.float32
            )

        # 2. Celestial source wavefront
        # Dynamic direction cosines (for moving satellites/RFI, else static)
        l_t = (l0 + drift_dl * t_indices).astype(np.float32)
        m_t = (m0 + drift_dm * t_indices).astype(np.float32)

        # Antenna geometric path delays: (samples_per_frame, num_antennas) in seconds
        delays = (
            np.outer(l_t, pos_x.astype(np.float32))
            + np.outer(m_t, pos_y.astype(np.float32))
        ) * c_inv

        # Intrinsic phase across time and frequency
        base_phases = two_pi_f32 * np.outer(
            t_indices * np.float32(0.005),
            freqs_hz.astype(np.float32) * np.float32(1e-8),
        )

        # Geometric phase shifts: (samples_per_frame, num_freq, num_antennas)
        geom_phases = two_pi_f32 * (
            delays[:, None, :] * freqs_hz.astype(np.float32)[None, :, None]
        )
        total_phases = base_phases[:, :, None] - geom_phases

        # Dispersion delay and pulse envelope for FRB / Pulsar
        if is_frb:
            # Quadratic dispersion delay: Delta t = 4.1488e-3 * DM * (f^-2 - f_ref^-2)
            f_ref = freqs_hz[-1]
            dm_delays_s = (
                (K_DM * 1e-6)
                * dm
                * (1.0 / (freqs_hz / 1e9) ** 2 - 1.0 / (f_ref / 1e9) ** 2)
            )
            t_physical_s = (t_indices * dt_s).astype(np.float32)
            pulse_center = np.float32((samples_per_frame * dt_s) * 0.4)
            pulse_width_s = np.float32(0.001)  # 1 ms width
            t_diff = t_physical_s[:, None] - (
                pulse_center + dm_delays_s[None, :].astype(np.float32)
            )
            envelope = np.exp(-0.5 * (t_diff / pulse_width_s) ** 2).astype(np.float32)

            v_real += (
                np.float32(source_amp) * envelope[:, :, None] * np.cos(total_phases)
            )
            v_imag += (
                np.float32(source_amp) * envelope[:, :, None] * np.sin(total_phases)
            )
        else:
            v_real += np.float32(source_amp) * np.cos(total_phases)
            v_imag += np.float32(source_amp) * np.sin(total_phases)

        # 3. Saturation injection: force selected antennas into extreme ADC clipping
        if saturate and len(saturated_antennas) > 0:
            for bad_ant in saturated_antennas:
                v_real[:, :, bad_ant] *= 12.0
                v_imag[:, :, bad_ant] *= 12.0

        # 4. Quantize to 4-bit complex integers [-7..+7] (int4x2_t format)
        r_quant = np.clip(np.round(v_real), -7, 7).astype(np.int8)
        i_quant = np.clip(np.round(v_imag), -7, 7).astype(np.int8)

        r_nibble = (r_quant & 0x0F).astype(np.uint8)
        i_nibble = ((i_quant & 0x0F) << 4).astype(np.uint8)

        packed_frames[frame_idx] = r_nibble | i_nibble

    elapsed_ms = (time.perf_counter() - t0_start) * 1000.0

    meta = {
        "scenario": scenario,
        "target_key": target_key,
        "target_name": target["name"],
        "category": target.get("category", "astronomical"),
        "ra_deg": float(target["ra_deg"]),
        "dec_deg": float(target["dec_deg"]),
        "transit_l": float(l0),
        "transit_m": float(m0),
        "drift_dl": drift_dl,
        "drift_dm": drift_dm,
        "source_amp": float(source_amp),
        "dm": dm,
        "use_noise": use_noise,
        "antenna_noise_sigmas": antenna_noise_sigmas.tolist(),
        "saturate": saturate,
        "saturated_antennas": saturated_antennas,
        "num_antennas": num_antennas,
        "num_freq": num_freq,
        "samples_per_frame": samples_per_frame,
        "num_frames": num_frames,
        "frame_duration_ms": samples_per_frame * delta_time_us / 1000.0,
        "freq_start_mhz": freq_start_mhz,
        "delta_freq_mhz": delta_freq_mhz,
        "delta_time_us": delta_time_us,
        "spacing_m": spacing_m,
        "site_lat_deg": CHARTS_LATITUDE_DEG,
        "site_lon_deg": CHARTS_LONGITUDE_DEG,
        "site_alt_m": CHARTS_ALTITUDE_M,
        "generation_time_ms": elapsed_ms,
    }

    return packed_frames, freqs_hz, meta


def save_baseband_to_hdf5(
    filepath: Path, packed_frames: np.ndarray, freqs_hz: np.ndarray, meta: dict
):
    """Saves packed voltage data to HDF5 matching CHORD/CHARTS metadata specifications."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    num_frames, n_time, n_freq, n_ant = packed_frames.shape
    total_time = num_frames * n_time

    flat_time = packed_frames.reshape(total_time, n_freq, n_ant)
    h5_array = np.transpose(flat_time, (2, 1, 0))

    with h5py.File(filepath, "w") as f:
        f.attrs["instrument"] = "CHARTS-64"
        f.attrs["telescope_name"] = "Observatorio Caren CHARTS"
        f.attrs["num_antennas"] = n_ant
        f.attrs["num_freq"] = n_freq
        f.attrs["num_time"] = total_time
        f.attrs["samples_per_frame"] = n_time
        f.attrs["num_frames"] = num_frames
        f.attrs["frame_duration_ms"] = float(meta["frame_duration_ms"])
        f.attrs["freq_start_MHz"] = float(meta["freq_start_mhz"])
        f.attrs["delta_freq_MHz"] = float(meta["delta_freq_mhz"])
        f.attrs["delta_time_us"] = float(meta["delta_time_us"])
        f.attrs["feed_separation_x_m"] = float(meta["spacing_m"])
        f.attrs["feed_separation_y_m"] = float(meta["spacing_m"])
        f.attrs["itrs_lat_deg"] = float(meta["site_lat_deg"])
        f.attrs["itrs_lon_deg"] = float(meta["site_lon_deg"])
        f.attrs["itrs_alt_m"] = float(meta["site_alt_m"])
        f.attrs["scenario"] = str(meta["scenario"])
        f.attrs["target_name"] = str(meta["target_name"])
        f.attrs["category"] = str(meta["category"])
        f.attrs["transit_l"] = float(meta["transit_l"])
        f.attrs["transit_m"] = float(meta["transit_m"])
        f.attrs["dm"] = float(meta["dm"])
        f.attrs["use_noise"] = bool(meta["use_noise"])
        f.attrs["saturate"] = bool(meta["saturate"])
        f.attrs["saturated_antennas"] = list(meta["saturated_antennas"])
        f.attrs["data_format"] = "complex_4bit_packed_int4x2"

        dset = f.create_dataset(
            "baseband",
            data=h5_array,
            dtype=np.uint8,
            chunks=(n_ant, 1, min(total_time, 1536)),
        )
        dset.attrs["axes"] = ["antenna", "frequency", "time"]
        dset.attrs["units"] = "packed_int4x2_t"

        f_group = f.create_group("fengine_compat")
        v_dset = f_group.create_dataset(
            "voltage",
            data=np.expand_dims(h5_array, axis=1),
            dtype=np.uint8,
        )
        v_dset.attrs["dim_names"] = ["D", "P", "F", "T"]
        v_dset.attrs["coarse_freq"] = list(range(n_freq))


def save_baseband_to_raw_bin(filepath: Path, packed_frames: np.ndarray):
    """Saves packed voltage frames to Kotekan rawFileRead compatible binary format."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    num_frames = packed_frames.shape[0]

    with open(filepath, "wb") as f:
        for frame_idx in range(num_frames):
            frame_data = packed_frames[frame_idx]
            np.uint32(0).tofile(f)
            frame_data.tofile(f)


def get_standard_scenarios_list() -> List[str]:
    """Returns standard comprehensive list of scenarios for production run."""
    targets = list(ASTRONOMICAL_CATALOG.keys())
    scenarios = []

    # Standard targets with noise and without noise
    for t in targets:
        scenarios.append(f"{t}_no_noise")
        scenarios.append(f"{t}_with_noise")

    # Saturation cases
    scenarios.append("sun_with_noise_saturated")
    scenarios.append("rfi_leo_saturated")

    return scenarios


def run_all_scenarios(
    output_dir: Path,
    scenario_list: List[str] | None = None,
    num_frames: int = 1,
    samples_per_frame: int = 1536,
    num_antennas: int = 64,
    num_freq: int = 336,
):
    """Generates all requested scenarios and saves both HDF5 and raw .bin files."""
    if scenario_list is None:
        scenario_list = get_standard_scenarios_list()

    print("=" * 78)
    print(" CHARTS 64-Antenna Baseband Dump Generator")
    print(f" Output Directory   : {output_dir}")
    print(
        f" Antennas           : {num_antennas} (8x8 grid, dx=dy={DEFAULT_SPACING_M}m)"
    )
    print(f" Frequency Channels : {num_freq} (300.0 - 400.5 MHz)")
    print(
        f" Samples Per Frame  : {samples_per_frame} ({samples_per_frame * FPGA_TIME_RESOLUTION_US / 1000.0:.2f} ms)"
    )
    print(f" Frames Per Scenario: {num_frames}")
    print(f" Total Scenarios    : {len(scenario_list)}")
    print("=" * 78)

    baseband_dir = output_dir / "baseband"
    baseband_dir.mkdir(parents=True, exist_ok=True)

    summary = []

    for sc in scenario_list:
        print(f"\n>>> Simulating scenario: {sc} ...")
        packed_frames, freqs_hz, meta = generate_scenario_baseband(
            scenario=sc,
            num_antennas=num_antennas,
            num_freq=num_freq,
            samples_per_frame=samples_per_frame,
            num_frames=num_frames,
        )

        h5_path = baseband_dir / f"{sc}_64ant_5ms.h5"
        bin_path = baseband_dir / f"{sc}_64ant_5ms_0000000.bin"

        print(f"    Saving HDF5 dump: {h5_path.name}")
        save_baseband_to_hdf5(h5_path, packed_frames, freqs_hz, meta)

        print(f"    Saving Kotekan raw binary dump: {bin_path.name}")
        save_baseband_to_raw_bin(bin_path, packed_frames)

        h5_size_mb = h5_path.stat().st_size / (1024 * 1024)
        bin_size_mb = bin_path.stat().st_size / (1024 * 1024)

        summary.append(
            {
                "scenario": sc,
                "target": meta["target_name"],
                "category": meta["category"],
                "noise": "Yes (random per-ant)" if meta["use_noise"] else "No",
                "saturated": (
                    f"Ants {meta['saturated_antennas']}" if meta["saturate"] else "None"
                ),
                "h5_file": str(h5_path),
                "h5_size_mb": f"{h5_size_mb:.2f} MB",
                "bin_file": str(bin_path),
                "bin_size_mb": f"{bin_size_mb:.2f} MB",
                "sim_time_ms": f"{meta['generation_time_ms']:.1f} ms",
            }
        )

    print("\n" + "=" * 78)
    print(" ALL REQUESTED SCENARIOS GENERATED SUCCESSFULLY!")
    print("=" * 78)
    for s in summary:
        print(f"  [{s['scenario']}] ({s['category']}) -> {s['target']}")
        print(f"    Noise: {s['noise']} | Saturation: {s['saturated']}")
        print(f"    HDF5: {s['h5_file']} ({s['h5_size_mb']})")
        print(f"    BIN : {s['bin_file']} ({s['bin_size_mb']})")


def main():
    parser = argparse.ArgumentParser(
        description="CHARTS 64-Antenna Baseband Dump Generator"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="all",
        help="Specific scenario name or 'all' or comma-separated list",
    )
    parser.add_argument(
        "--category",
        choices=[
            "all",
            "calibrator",
            "solar",
            "pulsar",
            "transient",
            "artificial",
            "zenith",
        ],
        default="all",
        help="Filter scenarios by astronomical category",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/project/def-vanderli/ferpb",
        help="Base output directory (default: /project/def-vanderli/ferpb)",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=1,
        help="Number of 5.12 ms frames (default: 1)",
    )
    parser.add_argument(
        "--samples-per-frame",
        type=int,
        default=1536,
        help="Samples per frame (default: 1536 = 5.12 ms)",
    )
    parser.add_argument(
        "--antennas", type=int, default=64, help="Number of antennas (default: 64)"
    )
    parser.add_argument(
        "--num-freq", type=int, default=336, help="Frequency channels (default: 336)"
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)

    if args.scenario == "all":
        sc_list = get_standard_scenarios_list()
        if args.category != "all":
            # Filter by category
            filtered = []
            for sc in sc_list:
                t_key, _, _ = parse_scenario_name(sc)
                if ASTRONOMICAL_CATALOG[t_key].get("category") == args.category:
                    filtered.append(sc)
            sc_list = filtered

        run_all_scenarios(
            output_dir=out_dir,
            scenario_list=sc_list,
            num_frames=args.num_frames,
            samples_per_frame=args.samples_per_frame,
            num_antennas=args.antennas,
            num_freq=args.num_freq,
        )
    elif "," in args.scenario:
        sc_list = [s.strip() for s in args.scenario.split(",")]
        run_all_scenarios(
            output_dir=out_dir,
            scenario_list=sc_list,
            num_frames=args.num_frames,
            samples_per_frame=args.samples_per_frame,
            num_antennas=args.antennas,
            num_freq=args.num_freq,
        )
    else:
        out_baseband = out_dir / "baseband"
        packed_frames, freqs_hz, meta = generate_scenario_baseband(
            scenario=args.scenario,
            num_antennas=args.antennas,
            num_freq=args.num_freq,
            samples_per_frame=args.samples_per_frame,
            num_frames=args.num_frames,
        )
        h5_path = out_baseband / f"{args.scenario}_64ant_5ms.h5"
        bin_path = out_baseband / f"{args.scenario}_64ant_5ms_0000000.bin"
        save_baseband_to_hdf5(h5_path, packed_frames, freqs_hz, meta)
        save_baseband_to_raw_bin(bin_path, packed_frames)
        print(
            f"[SUCCESS] Saved {h5_path} ({h5_path.stat().st_size / (1024*1024):.2f} MB)"
        )
        print(
            f"[SUCCESS] Saved {bin_path} ({bin_path.stat().st_size / (1024*1024):.2f} MB)"
        )


if __name__ == "__main__":
    main()
