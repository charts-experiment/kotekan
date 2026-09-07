#!/usr/bin/env python3
"""Plot CHARTS GPU correlation frames written by rawFileWrite.

The correlation payload is the cudaCorrelatorAstron layout:
    [local_frequency, lower-triangle receiver baseline, pol_y, pol_x, real_imag]
where every pair of input elements is recovered through its receiver/polarization
indices.  The selected files are averaged as complex visibilities before plotting.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_CHANNELS = 672
DEFAULT_ELEMENTS = 32
DEFAULT_POLARIZATIONS = 2
DEFAULT_FREQ_START_MHZ = 300.0
DEFAULT_CHANNEL_WIDTH_MHZ = 201.6 / DEFAULT_CHANNELS


def parse_elements(selection: str, element_count: int) -> list[int]:
    """Parse ``all``, comma lists, and inclusive integer ranges."""
    if selection.lower() == "all":
        return list(range(element_count))

    values: list[int] = []
    for part in selection.split(","):
        token = part.strip()
        if not token:
            raise ValueError("empty item in --antennas")
        if "-" in token:
            start_text, stop_text = token.split("-", 1)
            start, stop = int(start_text), int(stop_text)
            if stop < start:
                raise ValueError(f"invalid descending range: {token}")
            values.extend(range(start, stop + 1))
        else:
            values.append(int(token))

    if any(value < 0 or value >= element_count for value in values):
        raise ValueError(f"antenna IDs must be in [0, {element_count - 1}]")
    return list(dict.fromkeys(values))


def baseline_index(receiver_y: int, receiver_x: int) -> int:
    """Index of ``(receiver_y, receiver_x)`` in the lower-triangle layout."""
    return receiver_y * (receiver_y + 1) // 2 + receiver_x


def read_correlation_frame(
    path: Path, channels: int, elements: int, polarizations: int
) -> np.ndarray:
    """Read one rawFileWrite correlation payload as complex128.

    rawFileWrite prepends a uint32 metadata size and serialized metadata.  The
    metadata is intentionally skipped: the payload shape is fully defined by
    this command's arguments and the correlator configuration.
    """
    receiver_count, remainder = divmod(elements, polarizations)
    if remainder:
        raise ValueError("--elements must be divisible by --polarizations")
    baselines = receiver_count * (receiver_count + 1) // 2
    expected_bytes = channels * baselines * polarizations * polarizations * 2 * 4

    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size < 4:
        raise ValueError(f"{path} is shorter than a rawFileWrite metadata prefix")
    metadata_size = int(np.frombuffer(raw[:4].tobytes(), dtype="<u4", count=1)[0])
    payload_offset = 4 + metadata_size
    payload_size = raw.size - payload_offset
    if payload_size != expected_bytes:
        raise ValueError(
            f"{path}: payload is {payload_size} bytes, expected {expected_bytes}. "
            "Check --channels, --elements, and --polarizations."
        )

    payload = np.frombuffer(raw, dtype="<i4", count=expected_bytes // 4, offset=payload_offset)
    packed = payload.reshape(channels, baselines, polarizations, polarizations, 2)
    return packed[..., 0].astype(np.float64) + 1j * packed[..., 1].astype(np.float64)


def extract_visibility(correlation: np.ndarray, element_a: int, element_b: int, polarizations: int) -> np.ndarray:
    """Return V[a,b] across frequency, applying Hermitian conjugation as needed."""
    receiver_a, pol_a = divmod(element_a, polarizations)
    receiver_b, pol_b = divmod(element_b, polarizations)
    if receiver_b <= receiver_a:
        return correlation[:, baseline_index(receiver_a, receiver_b), pol_a, pol_b]
    return np.conj(correlation[:, baseline_index(receiver_b, receiver_a), pol_b, pol_a])


def plot_element_pairs(
    correlation: np.ndarray,
    reference: int,
    selected: list[int],
    polarizations: int,
    frequencies_mhz: np.ndarray,
    output_path: Path,
) -> None:
    """Make one multi-panel spectrum plot for one selected input element."""
    partners = [element for element in selected if element != reference]
    columns = min(4, len(partners))
    rows = math.ceil(len(partners) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(4.6 * columns, 2.9 * rows), squeeze=False)

    for axis, partner in zip(axes.flat, partners):
        amplitude = np.abs(extract_visibility(correlation, reference, partner, polarizations))
        axis.plot(frequencies_mhz, amplitude, linewidth=0.7)
        axis.set_title(f"|V[{reference}, {partner}]|", fontsize=10)
        axis.set_xlabel("Frequency [MHz]", fontsize=8)
        axis.set_ylabel("Correlation amplitude", fontsize=8)
        axis.grid(alpha=0.35)

    for axis in axes.flat[len(partners) :]:
        axis.set_visible(False)

    figure.suptitle(
        f"GPU correlations for input {reference} with selected inputs", fontsize=15
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_global_matrix(
    correlation: np.ndarray,
    selected: list[int],
    polarizations: int,
    output_path: Path,
) -> np.ndarray:
    """Plot a compact all-pairs summary using mean magnitude over frequency."""
    matrix = np.empty((len(selected), len(selected)), dtype=np.float64)
    for row, element_a in enumerate(selected):
        for column, element_b in enumerate(selected):
            visibility = extract_visibility(correlation, element_a, element_b, polarizations)
            matrix[row, column] = np.mean(np.abs(visibility), dtype=np.float64)

    finite_positive = matrix[np.isfinite(matrix) & (matrix > 0)]
    reference = np.max(finite_positive) if finite_positive.size else 1.0
    matrix_db = 20.0 * np.log10(np.maximum(matrix, np.finfo(np.float64).tiny) / reference)

    figure_size = max(7.0, 0.42 * len(selected))
    figure, axis = plt.subplots(figsize=(figure_size, figure_size))
    image = axis.imshow(matrix_db, origin="lower", aspect="equal", cmap="viridis", vmin=-60.0, vmax=0.0)
    axis.set_xticks(range(len(selected)), selected, rotation=90)
    axis.set_yticks(range(len(selected)), selected)
    axis.set_xlabel("Input element b")
    axis.set_ylabel("Input element a")
    axis.set_title("Mean |V[a,b]| over frequency (dB relative to selected maximum)")
    colorbar = figure.colorbar(image, ax=axis, shrink=0.85)
    colorbar.set_label("dB rel. max")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)
    return matrix


def select_files(data_dir: Path, pattern: str, count: int, position: str) -> list[Path]:
    files = sorted(data_dir.glob(pattern))
    if not files:
        raise ValueError(f"no correlation files matched {data_dir / pattern}")
    if count == 0 or count >= len(files):
        return files
    return files[:count] if position == "first" else files[-count:]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing raw correlation files")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for PNG and NPZ output")
    parser.add_argument("--pattern", default="correlation_0_*.bin", help="Glob for rawFileWrite correlation files")
    parser.add_argument("--file-count", type=int, default=5, help="Frames to average; 0 means all matching files")
    parser.add_argument("--file-position", choices=("first", "last"), default="last")
    parser.add_argument("--antennas", default="all", help="Input elements: all, 0,2,4, or 0-7,16-23")
    parser.add_argument("--channels", type=int, default=DEFAULT_CHANNELS)
    parser.add_argument("--elements", type=int, default=DEFAULT_ELEMENTS)
    parser.add_argument("--polarizations", type=int, default=DEFAULT_POLARIZATIONS)
    parser.add_argument("--frequency-start-mhz", type=float, default=DEFAULT_FREQ_START_MHZ)
    parser.add_argument("--channel-width-mhz", type=float, default=DEFAULT_CHANNEL_WIDTH_MHZ)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.file_count < 0:
        raise SystemExit("--file-count must be non-negative")
    if args.channels <= 0 or args.elements <= 1 or args.polarizations <= 0:
        raise SystemExit("--channels, --elements, and --polarizations must be positive")
    try:
        selected = parse_elements(args.antennas, args.elements)
        if len(selected) < 2:
            raise ValueError("select at least two input elements")
        files = select_files(args.data_dir, args.pattern, args.file_count, args.file_position)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    average = None
    for path in files:
        frame = read_correlation_frame(path, args.channels, args.elements, args.polarizations)
        average = frame if average is None else average + frame
        print(f"Read {path}")
    assert average is not None
    average /= len(files)

    frequencies_mhz = args.frequency_start_mhz + args.channel_width_mhz * np.arange(args.channels)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for element in selected:
        output_path = args.output_dir / f"correlations_input_{element:02d}.png"
        plot_element_pairs(average, element, selected, args.polarizations, frequencies_mhz, output_path)
        print(f"Saved {output_path}")

    matrix_path = args.output_dir / "correlations_selected_matrix.png"
    mean_magnitude = plot_global_matrix(average, selected, args.polarizations, matrix_path)
    npz_path = args.output_dir / "correlations_selected.npz"
    np.savez(
        npz_path,
        selected_elements=np.asarray(selected, dtype=np.int16),
        frequency_mhz=frequencies_mhz,
        mean_magnitude=mean_magnitude,
        averaged_files=np.asarray([str(path) for path in files]),
    )
    print(f"Saved {matrix_path}")
    print(f"Saved {npz_path}")


if __name__ == "__main__":
    main()
