#!/usr/bin/env python3
"""CHARTS Pipeline Data Inspector & Verification Suite.

Validates and inspects output data products from Kotekan:
  - Baseband 4-bit voltage frames (signal levels, clipping rates, ADC rails)
  - Tensor Core Correlator dumps (reconstructs 64x64 Hermitian visibilities,
    validates Hermitian symmetry ||V - V^H|| / ||V|| ~ 0.0, dynamic range)
  - Beam tracker float32 voltage dumps (formed beam power lightcurves, SNRs)
"""

from __future__ import annotations

import array
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def baseline_index(receiver_y: int, receiver_x: int) -> int:
    """Computes lower-triangular baseline index for (receiver_y, receiver_x)."""
    return receiver_y * (receiver_y + 1) // 2 + receiver_x


def load_astron_correlator_dump(
    bin_path: Path,
    num_elements: int = 64,
    num_channels: int = 336,
    polarizations: int = 2,
) -> np.ndarray:
    """Reads rawFileWrite binary file containing cudaCorrelatorAstron output.
    Returns:
        vis_cube: np.ndarray shape (num_channels, num_elements, num_elements) complex128
    """
    raw = np.fromfile(str(bin_path), dtype=np.uint8)
    if raw.size < 4:
        raise ValueError(f"File {bin_path} is smaller than 4-byte metadata header.")

    metadata_size = int(np.frombuffer(raw[:4].tobytes(), dtype="<u4", count=1)[0])
    payload_offset = 4 + metadata_size

    receiver_count = num_elements // polarizations
    num_baselines = receiver_count * (receiver_count + 1) // 2
    expected_ints = num_channels * num_baselines * polarizations * polarizations * 2
    expected_bytes = expected_ints * 4

    payload_size = raw.size - payload_offset
    if payload_size < expected_bytes:
        raise ValueError(
            f"File {bin_path} payload size {payload_size} bytes is less than expected {expected_bytes} bytes."
        )

    raw_ints = np.frombuffer(raw, dtype="<i4", count=expected_ints, offset=payload_offset)
    packed = raw_ints.reshape(num_channels, num_baselines, polarizations, polarizations, 2)
    c_data = packed[..., 0].astype(np.float64) + 1j * packed[..., 1].astype(np.float64)

    # Reconstruct full (channels, num_elements, num_elements) Hermitian matrix
    vis_cube = np.zeros((num_channels, num_elements, num_elements), dtype=np.complex128)
    for ry in range(receiver_count):
        for rx in range(ry + 1):
            b_idx = baseline_index(ry, rx)
            for py in range(polarizations):
                for px in range(polarizations):
                    ey = ry * polarizations + py
                    ex = rx * polarizations + px
                    val = c_data[:, b_idx, py, px]
                    vis_cube[:, ey, ex] = val
                    if ey != ex:
                        vis_cube[:, ex, ey] = np.conj(val)

    return vis_cube


def inspect_correlator_matrix(
    vis_cube: np.ndarray,
    freq_idx: Optional[int] = None,
) -> Dict[str, Any]:
    """Computes mathematical & scientific diagnostics on reconstructed visibility matrix."""
    num_channels, num_elements, _ = vis_cube.shape
    ch = freq_idx if freq_idx is not None else num_channels // 2
    v = vis_cube[ch]

    # 1. Hermitian symmetry error
    v_herm = np.conj(v.T)
    norm_diff = np.linalg.norm(v - v_herm, "fro")
    norm_v = np.linalg.norm(v, "fro")
    herm_err = float(norm_diff / norm_v) if norm_v > 0 else 0.0

    # 2. Diagonal (Autocorrelation) positivity & power
    diag = np.real(np.diag(v))
    min_diag = float(np.min(diag))
    max_diag = float(np.max(diag))
    mean_diag = float(np.mean(diag))

    # 3. Off-diagonal (Cross-correlation) power
    mask = ~np.eye(num_elements, dtype=bool)
    off_diag = np.abs(v[mask])
    mean_cross = float(np.mean(off_diag)) if off_diag.size > 0 else 0.0
    max_cross = float(np.max(off_diag)) if off_diag.size > 0 else 0.0
    snr_cross = float(max_cross / (mean_cross + 1e-9)) if mean_cross > 0 else 0.0

    return {
        "channel_inspected": ch,
        "num_elements": num_elements,
        "num_channels": num_channels,
        "hermitian_error": herm_err,
        "hermitian_valid": herm_err < 1e-6,
        "min_autocorr": min_diag,
        "max_autocorr": max_diag,
        "mean_autocorr": mean_diag,
        "mean_cross": mean_cross,
        "max_cross": max_cross,
        "cross_snr": snr_cross,
    }


def inspect_tracker_dump(
    bin_path: Path,
    num_freq: int = 336,
    max_beams: int = 8,
    samples_per_data_set: int = 1536,
) -> Dict[str, Any]:
    """Inspects float32 beam tracker voltages [time][freq][beam] and computes formed beam powers."""
    raw = np.fromfile(str(bin_path), dtype=np.uint8)
    if raw.size < 4:
        raise ValueError(f"File {bin_path} is smaller than 4-byte metadata header.")

    metadata_size = int(np.frombuffer(raw[:4].tobytes(), dtype="<u4", count=1)[0])
    payload_offset = 4 + metadata_size

    # 2 floats per complex float sample
    expected_floats = samples_per_data_set * num_freq * max_beams * 2
    expected_bytes = expected_floats * 4

    payload_size = raw.size - payload_offset
    if payload_size < expected_bytes:
        raise ValueError(f"Payload size {payload_size} is less than expected {expected_bytes}")

    raw_floats = np.frombuffer(raw, dtype="<f4", count=expected_floats, offset=payload_offset)
    # Shape: (samples, freq, beams, 2)
    shaped = raw_floats.reshape(samples_per_data_set, num_freq, max_beams, 2)
    c_voltages = shaped[..., 0].astype(np.float64) + 1j * shaped[..., 1].astype(np.float64)

    # Power per beam: average over time and sum over frequency
    beam_powers = np.mean(np.sum(np.abs(c_voltages) ** 2, axis=1), axis=0)

    return {
        "file_name": bin_path.name,
        "num_freq": num_freq,
        "max_beams": max_beams,
        "samples_per_frame": samples_per_data_set,
        "beam_powers": [float(p) for p in beam_powers],
        "total_power": float(np.sum(beam_powers)),
        "peak_beam": int(np.argmax(beam_powers)),
    }


def inspect_beamformer_dump(
    bin_path: Path,
    num_freq: int = 336,
    num_beams: int = 64,
    samples_per_data_set: int = 15360,
    integration_spectra: int = 10,
    input_mode: str = "fine",
    output_format: str = "int8",
) -> Dict[str, Any]:
    """Inspects float32 or int8 integrated beam intensity dump [T_integrated][freq][beam] from cudaChartsBeamformerCommand."""
    raw = np.fromfile(str(bin_path), dtype=np.uint8)
    if raw.size < 4:
        raise ValueError(f"File {bin_path} is smaller than 4-byte metadata header.")

    metadata_size = int(np.frombuffer(raw[:4].tobytes(), dtype="<u4", count=1)[0])
    payload_offset = 4 + metadata_size

    if input_mode == "fine":
        n_time_bf = samples_per_data_set // 32
        n_freq_out = 336 * 32
        integration_spectra = 10
    else:
        n_time_bf = samples_per_data_set
        n_freq_out = num_freq

    n_time_out = (n_time_bf + integration_spectra - 1) // integration_spectra
    expected_samples = n_time_out * n_freq_out * num_beams
    bytes_per_sample = 1 if output_format == "int8" else 4
    expected_bytes = expected_samples * bytes_per_sample

    payload_size = raw.size - payload_offset
    if payload_size != expected_bytes:
        raise ValueError(
            f"Payload size {payload_size} bytes does not match expected {expected_bytes} bytes"
        )

    dtype = "i1" if output_format == "int8" else "<f4"
    intensities = np.frombuffer(
        raw, dtype=dtype, count=expected_samples, offset=payload_offset
    ).reshape(n_time_out, n_freq_out, num_beams)

    beam_powers = np.mean(intensities, axis=(0, 1))
    return {
        "file_name": bin_path.name,
        "input_mode": input_mode,
        "output_format": output_format,
        "shape": (n_time_out, n_freq_out, num_beams),
        "all_finite": bool(np.all(np.isfinite(intensities))),
        "min_intensity": float(np.min(intensities)),
        "max_intensity": float(np.max(intensities)),
        "beam_powers": [float(p) for p in beam_powers],
        "peak_beam": int(np.argmax(beam_powers)),
        "intensities": intensities,
    }


def compute_reference_beamformer_intensity(
    packed_voltage: np.ndarray,
    shard_id: int = 0,
    num_beams: int = 64,
    integration_spectra: int = 320,
    spacing_m: float = 0.6,
    beam_l_step: float = 0.01,
    freq_zero_hz: float = 300e6,
    freq_step_hz: float = 300e3,
) -> np.ndarray:
    """Computes analytical/CPU reference coarse integrated beam intensity [T_int][336][B] in float32."""
    from .writer import unpack_int4x2

    n_time, n_freq, n_ant = packed_voltage.shape
    voltages = unpack_int4x2(packed_voltage)  # complex64 [T, F, E]

    rows = 4 if n_ant == 32 else 8
    cols = 8
    pos_x = np.array([float(c) * spacing_m for _ in range(rows) for c in range(cols)], dtype=np.float64)

    center = num_beams // 2
    l_vals = np.array([(b - center) * beam_l_step for b in range(num_beams)], dtype=np.float64)

    coarse_origin = shard_id * 336
    freqs_hz = freq_zero_hz + (coarse_origin + np.arange(n_freq, dtype=np.float64)) * freq_step_hz
    wave_numbers = (2.0 * math.pi * freqs_hz) / 299792458.0

    # phase[b, f, a] = wave_number[f] * (pos_x[a] * l_vals[b])
    phases = wave_numbers[None, :, None] * (l_vals[:, None, None] * pos_x[None, None, :])
    weights = np.cos(phases).astype(np.float32) + 1j * np.sin(phases).astype(np.float32)  # [B, F, E]

    # Sum over antennas: sum_real = w_r * v_r - w_i * v_i, sum_imag = w_r * v_i + w_i * v_r
    # Notice this is standard complex multiplication: weights[b, f, e] * voltages[t, f, e]
    formed = np.einsum("tfe,bfe->tfb", voltages, weights, optimize=True)
    power = (formed.real ** 2 + formed.imag ** 2).astype(np.float32)

    n_windows = (n_time + integration_spectra - 1) // integration_spectra
    integrated = np.zeros((n_windows, n_freq, num_beams), dtype=np.float32)
    for w in range(n_windows):
        t0 = w * integration_spectra
        t1 = min(n_time, t0 + integration_spectra)
        integrated[w] = np.sum(power[t0:t1], axis=0, dtype=np.float32)
    return integrated
