#!/usr/bin/env python3
"""Build combined CHARTS beamformer weights offline: gain correction x geometric steering.

The output is one canonical array per shard and mode,

    W[b, nu, e] = conj(C[nu, e] * exp(-2*pi*i * nu * (n_b . p_e) / c))
                = exp(+2*pi*i * nu * (n_b . p_e) / c) / conj(g[nu, e]),

applied by the beamformer as V[b, nu] = sum_e W[b, nu, e] * x[nu, e] (no conjugate in
the kernel).

Component conventions (the reason for the outer conjugate):

* Correlator/charts-utils voltage x' = high nibble + i*low nibble. cudaShuffleAstron
  swaps nibbles before the TCC kernel, whose native real part is the low nibble, and
  charts-utils baseband decoding also takes real from the high nibble. TCC stores
  corr[Y >= X] = x'_Y * conj(x'_X); charts-utils reads it as V_ab = x'_a * conj(x'_b).
  Its gains g (x' = g*s, C = 1/g) and fringestop model (tau_a = -(n . p_a)/c, phase
  -2*pi*nu*(tau_a - tau_b)) describe x', so W' = C * exp(-2*pi*i*nu*(n . p)/c) is
  coherent for x'.
* Beamformer voltage x = low nibble + i*high nibble (cudaChartsUpchannelizer.hpp),
  i.e. x = i * conj(x'). Sum_e W x is coherent iff W = conj(W'); the factor i is a
  common phase and does not affect intensity.

The resulting steering sign equals CHORD calcBBPhase's, but that is not the reason:
CHARTS conventions above govern.

Element orders (both always named explicitly, never implied):

* ``charts_logical``: CHARTS antenna a = 0..63. RFSoC 1 -> a = 0..31, RFSoC 0 -> 32..63.
  charts-utils gains and antenna positions use this order.
* ``kotekan_raw``: element axis of Kotekan voltage buffers, raw = 63 - a.
  The beamformer input uses this order, so the written weights use it too.

Coarse gains are copied to all 32 fine bins, as CHIME/CHORD ``processFeedGains`` does.
No normalisation is applied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import warnings

import numpy as np

SCRIPT_VERSION = 3  # 3: conjugate for the correlator vs beamformer nibble convention
DESCRIPTOR_FORMAT = "charts_beamformer_weights"

N_ELEMENTS = 64
N_COARSE = 672
N_SHARDS = 2
SHARD_COARSE = N_COARSE // N_SHARDS  # 336
UPCHAN = 32
FIRST_COARSE_CENTER_HZ = 300e6
COARSE_WIDTH_HZ = 300e3
SPEED_OF_LIGHT_M_PER_S = 299792458.0
ANTENNA_SPACING_M = 0.6
ANTENNAS_PER_ROW = 8

CHARTS_LOGICAL = "charts_logical"
KOTEKAN_RAW = "kotekan_raw"

MODES = ("coarse", "fine")
DEFAULT_DIRECTIONS = (Path(__file__).resolve().parents[1] / "config"
                      / "charts_hex64_beam_directions.json")


# ----------------------------------------------------------------------------- element order

def charts_logical_to_kotekan_raw_indices(n_elements: int = N_ELEMENTS) -> np.ndarray:
    """Kotekan raw element index of each charts_logical antenna: raw = N - 1 - a."""
    return np.arange(n_elements - 1, -1, -1)


def kotekan_raw_to_charts_logical_indices(n_elements: int = N_ELEMENTS) -> np.ndarray:
    """charts_logical antenna of each Kotekan raw element: a = N - 1 - raw."""
    return np.arange(n_elements - 1, -1, -1)


def charts_logical_to_kotekan_raw(array: np.ndarray, axis: int = -1) -> np.ndarray:
    """Reorder an element axis from charts_logical to kotekan_raw.

    out[..., raw] = in[..., a] with raw = N - 1 - a.
    """
    n = array.shape[axis]
    return np.take(array, kotekan_raw_to_charts_logical_indices(n), axis=axis)


# ----------------------------------------------------------------------------- frequencies

def shard_frequencies_hz(mode: str, shard: int,
                         first_coarse_center_hz: float = FIRST_COARSE_CENTER_HZ,
                         coarse_width_hz: float = COARSE_WIDTH_HZ) -> np.ndarray:
    """Centre frequencies of a shard, in Kotekan's local channel order.

    coarse: nu = nu0 + (336*s + f) * dnu,                f = 0..335
    fine:   nu = nu0 + (336*s + f + (u - 15.5)/32) * dnu, index f*32 + u
    """
    _check_mode_shard(mode, shard)
    coarse = SHARD_COARSE * shard + np.arange(SHARD_COARSE, dtype=np.float64)
    if mode == "coarse":
        return first_coarse_center_hz + coarse * coarse_width_hz
    offsets = (np.arange(UPCHAN, dtype=np.float64) - (UPCHAN - 1) / 2.0) / UPCHAN
    return first_coarse_center_hz + (coarse[:, None] + offsets[None, :]).reshape(-1) * coarse_width_hz


def _check_mode_shard(mode: str, shard: int) -> None:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if shard not in range(N_SHARDS):
        raise ValueError(f"shard must be 0 or 1, got {shard!r}")


# ----------------------------------------------------------------------------- geometry

def default_positions_charts_logical(n_elements: int = N_ELEMENTS,
                                     spacing_m: float = ANTENNA_SPACING_M) -> np.ndarray:
    """Default 8x8 ENU grid in charts_logical order, as charts-utils.

    Antenna 0 is the south-east corner; increasing a moves west along a row, then north:
    east = -spacing * (a % 8), north = spacing * (a // 8), up = 0.
    """
    a = np.arange(n_elements)
    positions = np.zeros((n_elements, 3), dtype=np.float64)
    positions[:, 0] = -spacing_m * (a % ANTENNAS_PER_ROW)
    positions[:, 1] = spacing_m * (a // ANTENNAS_PER_ROW)
    return positions


def load_positions_charts_logical(path: Path) -> np.ndarray:
    """Load an (64, 3) ENU metre array in charts_logical order from .npy."""
    positions = np.load(path, allow_pickle=False)
    if positions.shape != (N_ELEMENTS, 3) or not np.all(np.isfinite(positions)):
        raise ValueError(f"positions in {path} must be finite with shape ({N_ELEMENTS}, 3)")
    return positions.astype(np.float64)


def load_directions(path: Path) -> np.ndarray:
    """Load beam directions as (B, 3) unit ENU vectors from a charts_beam_directions JSON."""
    with open(path) as f:
        spec = json.load(f)
    if spec.get("format") != "charts_beam_directions":
        raise ValueError(f"{path} is not a charts_beam_directions file")
    nxy = np.asarray(spec["directions"], dtype=np.float64)
    if nxy.ndim != 2 or nxy.shape[1] != 2 or not np.all(np.isfinite(nxy)):
        raise ValueError(f"{path}: directions must be finite [B][n_x, n_y]")
    if not 1 <= nxy.shape[0] <= 128:
        raise ValueError(f"{path}: beam count must be 1..128, got {nxy.shape[0]}")
    transverse = np.sum(nxy**2, axis=1)
    if np.any(transverse > 1.0):
        raise ValueError(f"{path}: direction below the horizon (n_x^2 + n_y^2 > 1)")
    return np.column_stack([nxy, np.sqrt(1.0 - transverse)])


def geometric_steering(frequencies_hz: np.ndarray, positions_m: np.ndarray,
                       directions: np.ndarray) -> np.ndarray:
    """G[b, nu, e] = exp(-2*pi*i * nu * (n_b . p_e) / c), element axis as given.

    Coherent steering for correlator-convention voltages x' (charts-utils model); see
    ``to_beamformer_component_convention`` for the beamformer's convention.
    """
    path_s = directions @ positions_m.T / SPEED_OF_LIGHT_M_PER_S  # n.p/c, [B][E]
    phase = 2.0 * np.pi * frequencies_hz[None, :, None] * path_s[:, None, :]
    return np.exp(-1j * phase)


# ----------------------------------------------------------------------------- gains

def gain_correction(gains: np.ndarray, convention: str = "response") -> tuple[np.ndarray, np.ndarray]:
    """Return (C[672][64] complex128, flagged[672][64] bool), charts_logical order.

    gains is charts-utils output, (672, 64) or (672, 64, n_time). A time axis is reduced
    with nanmean, as charts-utils ``--save-avg-gains``. With convention "response", g is
    the instrument response (x = g*s) and C = 1/g. With "correction", C = g as supplied.
    NaN or zero entries are flagged and get C = 0, disabling that element and channel.
    """
    gains = np.asarray(gains)
    if not np.iscomplexobj(gains):
        raise ValueError("gains must be complex")
    if gains.ndim == 3:
        with warnings.catch_warnings():
            # Fully flagged channels average to NaN, which stays flagged.
            warnings.simplefilter("ignore", RuntimeWarning)
            gains = np.nanmean(gains, axis=-1)
    if gains.shape != (N_COARSE, N_ELEMENTS):
        raise ValueError(f"gains must have shape ({N_COARSE}, {N_ELEMENTS}[, n_time]), "
                         f"got {gains.shape}")
    if np.any(np.isinf(gains)):
        raise ValueError("gains contain infinite values")
    flagged = np.isnan(gains) | (gains == 0)
    correction = np.zeros(gains.shape, dtype=np.complex128)
    good = ~flagged
    if convention == "response":
        correction[good] = 1.0 / gains[good]
    elif convention == "correction":
        correction[good] = gains[good]
    else:
        raise ValueError(f"unknown gain convention {convention!r}")
    return correction, flagged


def shard_gain_correction(correction: np.ndarray, mode: str, shard: int) -> np.ndarray:
    """Slice the shard's 336 coarse channels; fine mode copies each to its 32 bins."""
    _check_mode_shard(mode, shard)
    local = correction[SHARD_COARSE * shard:SHARD_COARSE * (shard + 1)]
    return np.repeat(local, UPCHAN, axis=0) if mode == "fine" else local


def synthetic_gains(seed: int, n_time: int = 1, flag_antennas: tuple[int, ...] = (),
                    flag_channels: tuple[int, ...] = ()) -> np.ndarray:
    """Charts-utils-shaped fake gains (672, 64, n_time), charts_logical, reference antenna 0.

    Smooth per-antenna amplitude ripple, a per-antenna delay and a small time jitter;
    flagged antennas/channels are NaN, as charts-utils ``solve_gain`` returns them.
    """
    rng = np.random.default_rng(seed)
    nu = FIRST_COARSE_CENTER_HZ + np.arange(N_COARSE) * COARSE_WIDTH_HZ
    amp = rng.uniform(0.5, 2.0, N_ELEMENTS) * (
        1.0 + 0.1 * np.sin(2 * np.pi * nu[:, None] / 40e6 + rng.uniform(0, 2 * np.pi, N_ELEMENTS)))
    delay_s = rng.uniform(-5e-9, 5e-9, N_ELEMENTS)
    phase0 = rng.uniform(-np.pi, np.pi, N_ELEMENTS)
    g = amp * np.exp(1j * (2 * np.pi * nu[:, None] * delay_s + phase0))
    g = g / (g[:, :1] / np.abs(g[:, :1]))  # charts-utils references phase to antenna 0
    jitter = 1.0 + 0.01 * (rng.standard_normal((N_COARSE, N_ELEMENTS, n_time))
                           + 1j * rng.standard_normal((N_COARSE, N_ELEMENTS, n_time)))
    jitter[:, 0, :] = np.abs(jitter[:, 0, :])
    cube = g[:, :, None] * jitter
    cube[:, list(flag_antennas), :] = np.nan
    cube[list(flag_channels), :, :] = np.nan
    return cube


# ----------------------------------------------------------------------------- combine

def to_beamformer_component_convention(weights_correlator: np.ndarray) -> np.ndarray:
    """Convert weights coherent for x' (real = high nibble) to x (real = low nibble).

    x = i * conj(x'), so sum W x = i * conj(sum conj(W) x'): W = conj(W').
    """
    return np.conj(weights_correlator)


def build_weights(mode: str, shard: int, correction: np.ndarray, positions_m: np.ndarray,
                  directions: np.ndarray) -> np.ndarray:
    """Combined complex64 W[B][F_local][64] in kotekan_raw order, beamformer convention.

    correction and positions_m are charts_logical. Weights are combined in the
    correlator/charts-utils convention, converted to the beamformer's component
    convention, and the element axis is reordered once, at the end.
    """
    freqs = shard_frequencies_hz(mode, shard)
    local = shard_gain_correction(correction, mode, shard)  # [F][E]
    weights = np.empty((directions.shape[0], freqs.size, N_ELEMENTS), dtype=np.complex64)
    for b in range(directions.shape[0]):
        steering = geometric_steering(freqs, positions_m, directions[b:b + 1])[0]
        w = to_beamformer_component_convention(local * steering)
        weights[b] = charts_logical_to_kotekan_raw(w, axis=-1)
    return weights


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def descriptor(mode: str, shard: int, weights_path: Path, n_beams: int, flagged: np.ndarray,
               inputs: dict) -> dict:
    freqs = shard_frequencies_hz(mode, shard)
    local_flags = flagged[SHARD_COARSE * shard:SHARD_COARSE * (shard + 1)]
    return {
        "format": DESCRIPTOR_FORMAT,
        "version": SCRIPT_VERSION,
        "mode": mode,
        "shard": shard,
        "dtype": "complex64",
        "layout": "C-order [beam][frequency][element]",
        "shape": [n_beams, int(freqs.size), N_ELEMENTS],
        "element_order": KOTEKAN_RAW,
        "element_order_note": "raw element r holds charts_logical antenna a = 63 - r",
        "upchannelization_factor": UPCHAN if mode == "fine" else 1,
        "frequency_hz": {"first": float(freqs[0]), "last": float(freqs[-1]),
                         "rule": ("nu0 + (336*shard + f) * dnu" if mode == "coarse" else
                                  "nu0 + (336*shard + f + (u - 15.5)/32) * dnu, index f*32 + u"),
                         "nu0_hz": FIRST_COARSE_CENTER_HZ, "dnu_hz": COARSE_WIDTH_HZ},
        "application": ("V[b,nu] = sum_e W[b,nu,e] * x[nu,e], x = low nibble + i*high nibble; "
                        "apply once, no extra steering"),
        "weight": "W = conj(C * exp(-2*pi*i*nu*(n.p)/c)) = exp(+2*pi*i*nu*(n.p)/c) / conj(g)",
        "component_convention": ("gains/steering derived for correlator x' = high + i*low "
                                 "(cudaShuffleAstron nibble swap, charts-utils); "
                                 "beamformer x = i*conj(x'), hence the outer conjugate"),
        "steering": ("positions ENU m, directions ENU unit vectors; charts-utils "
                     "tau=-(n.p)/c, V_ab=x'_a*conj(x'_b)"),
        "gain": {"convention": inputs["gain_convention"],
                 "applied": ("conj(C), C = 1/g" if inputs["gain_convention"] == "response"
                             else "conj(C), C = g"),
                 "calibrated": inputs["gains"] is not None,
                 "fine_mapping": "coarse gain copied to all 32 fine bins" if mode == "fine" else None,
                 "flagged_logical_channel_antenna_count": int(local_flags.sum())},
        "normalization": "none",
        "inputs": inputs,
        "sha256": sha256_file(weights_path),
    }


# ----------------------------------------------------------------------------- CLI

def _build(args: argparse.Namespace) -> int:
    directions = load_directions(args.directions)
    positions = (load_positions_charts_logical(args.positions) if args.positions
                 else default_positions_charts_logical())
    if args.gains:
        gains = np.load(args.gains, allow_pickle=False)
        correction, flagged = gain_correction(gains, args.gain_convention)
    else:
        correction = np.ones((N_COARSE, N_ELEMENTS), dtype=np.complex128)
        flagged = np.zeros((N_COARSE, N_ELEMENTS), dtype=bool)
        print("No --gains: unit gains, geometric steering only (uncalibrated).", file=sys.stderr)

    inputs = {
        "gains": str(args.gains) if args.gains else None,
        "gains_sha256": sha256_file(args.gains) if args.gains else None,
        "gains_element_order": CHARTS_LOGICAL,
        "gain_convention": args.gain_convention,
        "calibration_note": args.calibration_note,
        "directions": str(args.directions),
        "directions_sha256": sha256_file(args.directions),
        "positions": str(args.positions) if args.positions else "default 8x8 grid, 0.6 m",
        "positions_element_order": CHARTS_LOGICAL,
        "script": "charts/scripts/build_beamformer_weights.py",
        "script_version": SCRIPT_VERSION,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shards = range(N_SHARDS) if args.shard is None else [args.shard]
    for shard in shards:
        weights = build_weights(args.mode, shard, correction, positions, directions)
        stem = args.output_dir / f"weights_{args.mode}_shard{shard}"
        np.save(stem.with_suffix(".npy"), weights)
        if args.emit_legacy_bin:
            # Raw canonical complex float32 [B][F][E]: the current weights_file reader input.
            weights.tofile(stem.with_suffix(".bin"))
        desc = descriptor(args.mode, shard, stem.with_suffix(".npy"), directions.shape[0],
                          flagged, inputs)
        if args.emit_legacy_bin:
            desc["legacy_bin_sha256"] = sha256_file(stem.with_suffix(".bin"))
        with open(stem.with_suffix(".json"), "w") as f:
            json.dump(desc, f, indent=1)
        print(f"wrote {stem}.npy shape {weights.shape}")
    return 0


def _synthetic(args: argparse.Namespace) -> int:
    cube = synthetic_gains(args.seed, args.n_time, tuple(args.flag_antennas),
                           tuple(args.flag_channels))
    if args.save_avg_gains:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            gains = np.nanmean(cube, axis=-1)
    else:
        gains = cube
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, gains.astype(np.complex128))
    print(f"wrote synthetic charts-utils gains {args.output} shape {gains.shape}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="combine gains and geometry into per-shard weights")
    b.add_argument("--mode", choices=MODES, required=True)
    b.add_argument("--shard", type=int, choices=range(N_SHARDS), default=None,
                   help="one shard (default: both)")
    b.add_argument("--gains", type=Path, default=None,
                   help="charts-utils gains .npy, (672, 64[, n_time]), charts_logical order")
    b.add_argument("--gain-convention", choices=("response", "correction"), default="response",
                   help="response: g is x = g*s (charts-utils), apply 1/g")
    b.add_argument("--calibration-note", default=None,
                   help="calibrator, observation time and validity, recorded verbatim")
    b.add_argument("--directions", type=Path, default=DEFAULT_DIRECTIONS)
    b.add_argument("--positions", type=Path, default=None,
                   help=".npy (64, 3) ENU metres, charts_logical order (default 8x8 grid)")
    b.add_argument("--output-dir", type=Path, required=True)
    b.add_argument("--emit-legacy-bin", action="store_true",
                   help="also write raw .bin for the current weights_file reader")
    b.set_defaults(func=_build)

    s = sub.add_parser("synthetic-gains", help="write fake charts-utils-shaped gains")
    s.add_argument("--output", type=Path, required=True)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--n-time", type=int, default=4)
    s.add_argument("--save-avg-gains", action="store_true", help="save (672, 64) time average")
    s.add_argument("--flag-antennas", type=int, nargs="*", default=[])
    s.add_argument("--flag-channels", type=int, nargs="*", default=[])
    s.set_defaults(func=_synthetic)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
