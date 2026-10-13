#!/usr/bin/env python3
"""CHARTS Physical, Instrumental, and Observatory Constants.

Centralizes physical constants, array dimensions, RF front-end specifications,
and site coordinates (Observatorio Carén, Chile) for Kotekan CHARTS simulations.
Supports optional dynamic loading from external `charts_constants` package if present.
"""

from __future__ import annotations

import importlib
import logging
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger("kotekan.charts.sim.constants")

# ---------------------------------------------------------------------------
# Attempt external package resolution
# ---------------------------------------------------------------------------
_external_module = None
_source_description = "embedded_fallback"

_sim_dir = Path(__file__).resolve().parent
_test_charts_dir = _sim_dir.parent
_kotekan_root = _test_charts_dir.parent
_charts_root = _kotekan_root.parent

try:
    _external_module = importlib.import_module("charts_constants")
    _source_description = "installed_charts_constants"
except Exception:
    _adjacent_paths = [
        _charts_root / "charts-constants",
        _kotekan_root / "charts-constants",
    ]
    for _path in _adjacent_paths:
        if _path.exists() and (_path / "charts_constants").is_dir():
            if str(_path) not in sys.path:
                sys.path.insert(0, str(_path))
            try:
                _external_module = importlib.import_module("charts_constants")
                _source_description = f"path_charts_constants ({_path})"
                break
            except Exception:
                pass


def _extract_val(val: Any) -> Any:
    """Extract raw float/int/array from Astropy Quantity or return as-is."""
    if hasattr(val, "value"):
        return val.value
    return val


def _has_external(attr_name: str) -> bool:
    return _external_module is not None and hasattr(_external_module, attr_name)


def _get_external(attr_name: str, default: Any) -> Any:
    if _has_external(attr_name):
        raw = getattr(_external_module, attr_name)
        return _extract_val(raw)
    return default


# ---------------------------------------------------------------------------
# Metadata Flags
# ---------------------------------------------------------------------------
USING_CHARTS_CONSTANTS_PACKAGE: bool = _external_module is not None
USING_FALLBACK_CONSTANTS: bool = not USING_CHARTS_CONSTANTS_PACKAGE
CONSTANTS_SOURCE: str = _source_description


# ---------------------------------------------------------------------------
# 1. Dispersion Measure & Physical Constants
# ---------------------------------------------------------------------------
K_DM: float = float(_get_external("K_DM", 4148.741601))
C_LIGHT: float = float(_get_external("C_LIGHT", 299_792_458.0))
SPEED_OF_LIGHT: float = C_LIGHT
SPEED_OF_LIGHT_M_PER_S: float = C_LIGHT
K_BOLTZMANN: float = 1.380649e-23


# ---------------------------------------------------------------------------
# 2. Instrumental & Sampling Constants
# ---------------------------------------------------------------------------
ADC_SAMPLING_FREQ_HZ: float = float(_get_external("ADC_SAMPLING_FREQ", 2457.6e6))
ADC_SAMPLING_FREQ_MHZ: float = float(_get_external("ADC_SAMPLING_FREQ_MHZ", 2457.6))
DEFAULT_SAMPLE_RATE: float = ADC_SAMPLING_FREQ_MHZ
FPGA_FREQ0_MHZ: float = ADC_SAMPLING_FREQ_MHZ

FPGA_NUM_SAMP_FFT: int = int(_get_external("FPGA_NUM_SAMP_FFT", 8192))
DEFAULT_NFFT: int = FPGA_NUM_SAMP_FFT
NFFT: int = FPGA_NUM_SAMP_FFT

CHARTS_CHANNEL_WIDTH_HZ: float = float(
    _get_external("CHARTS_CHANNEL_WIDTH_HZ", 300_000.0)
)
CHARTS_CHANNEL_WIDTH_MHZ: float = float(_get_external("CHARTS_CHANNEL_WIDTH_MHZ", 0.3))
CHANNEL_WIDTH_MHZ: float = CHARTS_CHANNEL_WIDTH_MHZ

FPGA_TIME_RESOLUTION_US: float = float(
    _get_external("FPGA_TIME_RESOLUTION_US", 10.0 / 3.0)
)
TIME_PER_SPECTRUM_US: float = FPGA_TIME_RESOLUTION_US
TIME_PER_SPECTRUM_S: float = FPGA_TIME_RESOLUTION_US * 1e-6
SAMPLE_RATE_HZ: float = CHARTS_CHANNEL_WIDTH_HZ

# Digitizer operating point: per-component (Re/Im) channel-noise sigma in LSB at
# the reference temperature T_ref. Calibrated against the F-engine reference
# chain (sim/fengine.py, Julia parity): the 15-level (±7) round-clamp quantizer
# has its minimum SNR loss (~0.06 dB) at sigma ≈ 2.9 LSB; 2.0 LSB is chosen to
# keep headroom for bright sources (loss ~0.09 dB, well inside the < 0.5 dB
# equivalence bound of AGENTS.md §3).
DIGITIZER_NOMINAL_SIGMA_LSB: float = 2.0


# ---------------------------------------------------------------------------
# 3. Frequency Grid & Sharding
# ---------------------------------------------------------------------------
CHARTS_N_FREQ: int = int(_get_external("CHARTS_N_FREQ", 672))
LOCAL_FREQUENCY_CHANNELS: int = int(_get_external("LOCAL_FREQUENCY_CHANNELS", 336))
FREQUENCY_SHARD_COUNT: int = int(_get_external("FREQUENCY_SHARD_COUNT", 2))

DEFAULT_FREQUENCY_START_MHZ: float = float(
    _get_external("DEFAULT_FREQUENCY_START_MHZ", 300.0)
)
DEFAULT_FREQUENCY_START_HZ: float = float(
    _get_external("DEFAULT_FREQUENCY_START_HZ", 300_000_000.0)
)


# ---------------------------------------------------------------------------
# 4. Array Geometry & Dimensions
# ---------------------------------------------------------------------------
DEFAULT_SPACING_M: float = float(_get_external("DEFAULT_SPACING_M", 0.6))
CHARTS_N_ANTENNAS: int = int(_get_external("CHARTS_N_ANTENNAS", 256))
DEFAULT_ANTENNAS: int = 64


# ---------------------------------------------------------------------------
# 5. Site Coordinates (Observatorio Carén, Chile)
# ---------------------------------------------------------------------------
CHARTS_LATITUDE_DEG: float = float(_get_external("CHARTS_LATITUDE_DEG", -33.4211146))
CHARTS_LONGITUDE_DEG: float = float(_get_external("CHARTS_LONGITUDE_DEG", -70.8634710))
CHARTS_ALTITUDE_M: float = float(_get_external("CHARTS_ALTITUDE_M", 458.0))

# Calán Observatory (Auxiliary)
CALAN_LATITUDE_DEG: float = float(_get_external("CALAN_LATITUDE_DEG", -33.397222))
CALAN_LONGITUDE_DEG: float = float(_get_external("CALAN_LONGITUDE_DEG", -70.536111))
CALAN_ALTITUDE_M: float = float(_get_external("CALAN_ALTITUDE_M", 867.0))


# ---------------------------------------------------------------------------
# 6. Networking & CPT Packet Format
# ---------------------------------------------------------------------------
CPT_SAMPLE_RATE: float = float(_get_external("CPT_SAMPLE_RATE", 4915.2))
CPT_SPECTRA_PER_PACKET: int = int(_get_external("CPT_SPECTRA_PER_PACKET", 4))


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def get_antenna_positions(
    num_antennas: int = 64, spacing_m: float = DEFAULT_SPACING_M
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes physical (x, y) coordinates for uniform rectangular grid."""
    if num_antennas <= 64:
        cols = np.arange(num_antennas) & 7
        rows = np.arange(num_antennas) >> 3
    else:
        cols = np.arange(num_antennas) & 15
        rows = np.arange(num_antennas) >> 4
    pos_x = (cols * spacing_m).astype(np.float32)
    pos_y = (rows * spacing_m).astype(np.float32)
    return pos_x, pos_y


def get_frequency_channels_mhz(
    num_channels: int = LOCAL_FREQUENCY_CHANNELS,
    f_start_mhz: float = DEFAULT_FREQUENCY_START_MHZ,
    df_mhz: float = CHARTS_CHANNEL_WIDTH_MHZ,
) -> np.ndarray:
    """Returns array of channel center frequencies in MHz."""
    return f_start_mhz + np.arange(num_channels, dtype=np.float64) * df_mhz
