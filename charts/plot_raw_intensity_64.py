#!/usr/bin/env python3
"""Visualize RFSoC antenna intensity from rawFileWrite frames.

Examples:
    # Last five files from NIC/handler 0, all physical antennas.
    python3 test_charts/plot_raw_intensity_64.py --handler 0

    # First two matching frames from both handlers, selected physical antennas.
    python3 test_charts/plot_raw_intensity_64.py \
        --handler both --file-count 2 --file-position first \
        --antennas 0-7,31,63 --time-bin-spectra 64

The voltage payload is byte-sized int4x2, so payload endianness does not apply.
Metadata endianness and nibble/IQ encoding are configurable independently.
"""

from __future__ import annotations

import argparse
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_SPECTRA_PER_FRAME = 15360
DEFAULT_CHANNELS = 336
DEFAULT_ELEMENTS = 64
DEFAULT_SPECTRA_PER_SECOND = 300000.0
DEFAULT_FREQ0_MHZ = 300.0
DEFAULT_CHANNEL_WIDTH_MHZ = 0.3


@dataclass(frozen=True)
class FrameGroup:
    """A time-aligned interval backed by one file from each selected handler."""

    paths: tuple[Path, ...]
    fpga_start: int
    handler_offsets: tuple[int, ...]
    spectra: int


def parse_antenna_selection(value: str, num_elements: int) -> list[int]:
    """Parse physical antenna selections such as 'all' or '0-7,31,63'."""
    if value.strip().lower() == "all":
        return list(range(num_elements))

    selected: list[int] = []
    seen: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            first_text, last_text = item.split("-", 1)
            first = int(first_text)
            last = int(last_text)
            if first > last:
                raise ValueError(f"invalid descending antenna range: {item}")
            values = range(first, last + 1)
        else:
            values = (int(item),)

        for antenna in values:
            if not 0 <= antenna < num_elements:
                raise ValueError(
                    f"physical antenna {antenna} is outside [0, {num_elements - 1}]"
                )
            if antenna not in seen:
                selected.append(antenna)
                seen.add(antenna)

    if not selected:
        raise ValueError("no antennas were selected")
    return selected


def physical_antenna_to_element(antenna: int, num_elements: int) -> int:
    """Map physical antenna 0..63 to the descending output E index."""
    return num_elements - 1 - antenna


def endian_prefix(name: str) -> str:
    return {"native": "=", "little": "<", "big": ">"}[name]


def read_raw_metadata(path: Path, metadata_endian: str) -> tuple[int, int, int]:
    """Return FPGA sequence number, metadata size and payload offset."""
    prefix = endian_prefix(metadata_endian)
    with path.open("rb") as stream:
        size_bytes = stream.read(4)
        if len(size_bytes) != 4:
            raise ValueError(f"{path}: missing metadata-size word")
        metadata_size = struct.unpack(f"{prefix}I", size_bytes)[0]
        metadata = stream.read(metadata_size)
        if len(metadata) != metadata_size:
            raise ValueError(f"{path}: truncated metadata")

    # chartsMetadata starts with int32 frame_counter, four padding bytes, then int64 fpga_seq.
    fpga_seq = struct.unpack_from(f"{prefix}q", metadata, 8)[0] if metadata_size >= 16 else -1
    return fpga_seq, metadata_size, 4 + metadata_size


def read_raw_frame(
    path: Path,
    spectra_per_frame: int,
    channels: int,
    elements: int,
    metadata_endian: str,
) -> tuple[np.memmap, int, int]:
    """Open one rawFileWrite frame and return its memmap and metadata summary."""
    fpga_seq, metadata_size, data_offset = read_raw_metadata(path, metadata_endian)
    expected_data_size = spectra_per_frame * channels * elements
    actual_data_size = path.stat().st_size - data_offset
    if actual_data_size != expected_data_size:
        raise ValueError(
            f"{path}: expected {expected_data_size} data bytes after metadata, "
            f"found {actual_data_size}. Check dimensions, endianness, or stale files."
        )

    frame = np.memmap(
        path,
        dtype=np.uint8,
        mode="r",
        offset=data_offset,
        shape=(spectra_per_frame, channels, elements),
        order="C",
    )
    return frame, fpga_seq, metadata_size


def decode_components(
    values: np.ndarray,
    encoding: str,
    iq_order: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode packed int4x2 bytes into float32 real and imaginary components."""
    low = (values & 0x0F).astype(np.int8)
    high = ((values >> 4) & 0x0F).astype(np.int8)

    if encoding == "twos-complement":
        low = ((low << 4) >> 4).astype(np.float32)
        high = ((high << 4) >> 4).astype(np.float32)
    else:
        low = (low.astype(np.float32) - 8.0).astype(np.float32)
        high = (high.astype(np.float32) - 8.0).astype(np.float32)

    if iq_order == "ri":
        return low, high
    return high, low


def candidate_files(args: argparse.Namespace, handler: int) -> list[Path]:
    """Return sorted raw files for one handler before time selection."""
    if args.pattern:
        if args.handler == "both" and "{handler}" not in args.pattern:
            raise SystemExit(
                "With --handler both, --pattern must contain the {handler} placeholder"
            )
        pattern = args.pattern.format(handler=handler)
    else:
        pattern = f"*network_{handler}_*.bin"

    files = sorted(args.data_dir.glob(pattern))
    if not files:
        raise SystemExit(f"No handler {handler} files matched {args.data_dir / pattern}")
    return files


def select_file_groups(
    args: argparse.Namespace,
) -> tuple[tuple[int, ...], list[FrameGroup]]:
    """Select one handler or align both handlers by overlapping FPGA intervals."""
    handlers = (0, 1) if args.handler == "both" else (int(args.handler),)
    files_by_handler = {handler: candidate_files(args, handler) for handler in handlers}

    if len(handlers) == 1:
        files = files_by_handler[handlers[0]]
        if args.file_count > 0:
            files = (
                files[: args.file_count]
                if args.file_position == "first"
                else files[-args.file_count :]
            )
        groups: list[FrameGroup] = []
        for path in files:
            fpga_seq, _, _ = read_raw_metadata(path, args.metadata_endian)
            if fpga_seq < 0:
                raise SystemExit(f"Cannot index {path}: metadata has no FPGA sequence number")
            groups.append(
                FrameGroup(
                    paths=(path,),
                    fpga_start=fpga_seq,
                    handler_offsets=(0,),
                    spectra=args.spectra_per_frame,
                )
            )
        return handlers, groups

    entries_by_handler: dict[int, list[tuple[int, Path]]] = {}
    for handler in handlers:
        entries: list[tuple[int, Path]] = []
        seen_sequences: dict[int, Path] = {}
        for path in files_by_handler[handler]:
            fpga_seq, _, _ = read_raw_metadata(path, args.metadata_endian)
            if fpga_seq < 0:
                raise SystemExit(f"Cannot align {path}: metadata has no FPGA sequence number")
            if fpga_seq in seen_sequences:
                raise SystemExit(
                    f"Duplicate FPGA sequence {fpga_seq} for handler {handler}: "
                    f"{seen_sequences[fpga_seq]} and {path}"
                )
            seen_sequences[fpga_seq] = path
            entries.append((fpga_seq, path))
        entries_by_handler[handler] = sorted(entries)

    entries_0 = entries_by_handler[handlers[0]]
    entries_1 = entries_by_handler[handlers[1]]
    groups = []
    index_0 = 0
    index_1 = 0
    while index_0 < len(entries_0) and index_1 < len(entries_1):
        start_0, path_0 = entries_0[index_0]
        start_1, path_1 = entries_1[index_1]
        stop_0 = start_0 + args.spectra_per_frame
        stop_1 = start_1 + args.spectra_per_frame

        overlap_start = max(start_0, start_1)
        overlap_stop = min(stop_0, stop_1)
        if overlap_start < overlap_stop:
            groups.append(
                FrameGroup(
                    paths=(path_0, path_1),
                    fpga_start=overlap_start,
                    handler_offsets=(overlap_start - start_0, overlap_start - start_1),
                    spectra=overlap_stop - overlap_start,
                )
            )

        if stop_0 <= stop_1:
            index_0 += 1
        if stop_1 <= stop_0:
            index_1 += 1

    if not groups:
        range_0 = (entries_0[0][0], entries_0[-1][0] + args.spectra_per_frame)
        range_1 = (entries_1[0][0], entries_1[-1][0] + args.spectra_per_frame)
        raise SystemExit(
            "Handlers 0 and 1 have no overlapping FPGA intervals. "
            f"handler 0 range={range_0}, handler 1 range={range_1}"
        )

    used_paths = {path for group in groups for path in group.paths}
    for handler in handlers:
        unused = sum(path not in used_paths for _, path in entries_by_handler[handler])
        if unused:
            print(
                f"WARNING: ignoring {unused} handler {handler} file(s) "
                "that do not overlap the other handler"
            )

    if args.file_count > 0:
        groups = (
            groups[: args.file_count]
            if args.file_position == "first"
            else groups[-args.file_count :]
        )
    return handlers, groups

def scale_intensity(values: np.ndarray, scale: str, db_floor: float) -> np.ndarray:
    if scale == "linear":
        return values
    with np.errstate(divide="ignore", invalid="ignore"):
        scaled = 10.0 * np.log10(values)
    return np.maximum(np.nan_to_num(scaled, nan=db_floor, neginf=db_floor), db_floor)


def antenna_ticks(axis, antennas: list[int]) -> None:
    max_ticks = 16
    step = max(1, math.ceil(len(antennas) / max_ticks))
    positions = np.arange(0, len(antennas), step)
    axis.set_yticks(positions)
    axis.set_yticklabels([str(antennas[position]) for position in positions])


def plot_frequency_maps(
    mean_intensity: np.ndarray,
    max_intensity: np.ndarray,
    antennas: list[int],
    frequency_mhz: np.ndarray,
    scale: str,
    db_floor: float,
    color_min: float | None,
    color_max: float | None,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    mean_display = scale_intensity(mean_intensity, scale, db_floor)
    max_display = scale_intensity(max_intensity, scale, db_floor)
    label = "Intensity [I^2 + Q^2]" if scale == "linear" else "Intensity [dB]"

    figure, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True, constrained_layout=True)
    for axis, values, title in zip(
        axes,
        (mean_display, max_display),
        ("Mean intensity over selected spectra", "Maximum intensity over selected spectra"),
    ):
        image = axis.imshow(
            values,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            extent=(
                frequency_mhz[0] - DEFAULT_CHANNEL_WIDTH_MHZ / 2,
                frequency_mhz[-1] + DEFAULT_CHANNEL_WIDTH_MHZ / 2,
                -0.5,
                len(antennas) - 0.5,
            ),
            vmin=color_min,
            vmax=color_max,
        )
        axis.set_ylabel("Physical antenna")
        axis.set_title(title)
        antenna_ticks(axis, antennas)
        figure.colorbar(image, ax=axis, label=label, pad=0.01)
    axes[-1].set_xlabel("Frequency [MHz]")
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_time_maps(
    time_mean: np.ndarray,
    time_max: np.ndarray,
    time_seconds: np.ndarray,
    antennas: list[int],
    scale: str,
    db_floor: float,
    color_min: float | None,
    color_max: float | None,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    mean_display = scale_intensity(time_mean.T, scale, db_floor)
    max_display = scale_intensity(time_max.T, scale, db_floor)
    label = "Intensity [I^2 + Q^2]" if scale == "linear" else "Intensity [dB]"
    time_start = float(time_seconds[0])
    time_stop = float(time_seconds[-1])
    if time_stop <= time_start:
        time_stop = time_start + 1.0

    figure, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True, constrained_layout=True)
    for axis, values, title in zip(
        axes,
        (mean_display, max_display),
        ("Mean intensity per time bin and antenna", "Maximum intensity per time bin and antenna"),
    ):
        image = axis.imshow(
            values,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            extent=(time_start, time_stop, -0.5, len(antennas) - 0.5),
            vmin=color_min,
            vmax=color_max,
        )
        axis.set_ylabel("Physical antenna")
        axis.set_title(title)
        antenna_ticks(axis, antennas)
        figure.colorbar(image, ax=axis, label=label, pad=0.01)
    axes[-1].set_xlabel("Time since selected data start [s]")
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_spectra_grid(
    intensity: np.ndarray,
    antennas: list[int],
    frequency_mhz: np.ndarray,
    statistic: str,
    scale: str,
    db_floor: float,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    display = scale_intensity(intensity, scale, db_floor)
    columns = min(8, len(antennas))
    rows = math.ceil(len(antennas) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(2.5 * columns, 1.8 * rows),
        sharex=True,
        squeeze=False,
        constrained_layout=True,
    )
    for index, axis in enumerate(axes.flat):
        if index >= len(antennas):
            axis.set_visible(False)
            continue
        axis.plot(frequency_mhz, display[index], linewidth=0.65)
        axis.set_title(f"Antenna {antennas[index]}", fontsize=9)
        axis.grid(alpha=0.2)
        axis.set_ylim(-10, 25)
    figure.suptitle(f"{statistic.capitalize()} intensity spectrum")
    figure.supxlabel("Frequency [MHz]")
    figure.supylabel("Intensity" if scale == "linear" else "Intensity [dB]")
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot mean and peak intensity for all or selected physical RFSoC antennas."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("/data/32_test"))
    parser.add_argument("--handler", choices=("0", "1", "both"), default="0")
    parser.add_argument(
        "--pattern",
        help=(
            "Override the raw file glob pattern; with --handler both it must "
            "contain {handler}, for example '*network_{handler}_*.bin'"
        ),
    )
    parser.add_argument(
        "--file-count",
        type=int,
        default=5,
        help="Number of files to use; 0 means all files (default: 5)",
    )
    parser.add_argument(
        "--file-position",
        choices=("first", "last"),
        default="last",
        help="Select files from the beginning or end of the sorted list",
    )
    parser.add_argument(
        "--antennas",
        default="all",
        help="Physical antennas: all, a comma list, or ranges such as 0-7,31,63",
    )
    parser.add_argument("--channel-start", type=int, default=0, help="First local channel")
    parser.add_argument("--channel-stop", type=int, help="Exclusive last local channel")
    parser.add_argument(
        "--start-spectrum",
        type=int,
        default=0,
        help="First spectrum in the concatenated selected files",
    )
    parser.add_argument(
        "--stop-spectrum",
        type=int,
        help="Exclusive last spectrum in the concatenated selected files",
    )
    parser.add_argument("--spectra-per-frame", type=int, default=DEFAULT_SPECTRA_PER_FRAME)
    parser.add_argument("--channels", type=int, default=DEFAULT_CHANNELS)
    parser.add_argument("--elements", type=int, default=DEFAULT_ELEMENTS)
    parser.add_argument(
        "--spectra-per-second", type=float, default=DEFAULT_SPECTRA_PER_SECOND
    )
    parser.add_argument(
        "--time-bin-spectra",
        type=int,
        default=256,
        help="Spectra averaged into one time bin (256 is about 0.853 ms)",
    )
    parser.add_argument(
        "--metadata-endian",
        choices=("native", "little", "big"),
        default="native",
        help="Endianness of rawFileWrite metadata; voltage payload is uint8",
    )
    parser.add_argument(
        "--encoding",
        choices=("twos-complement", "offset-binary"),
        default="twos-complement",
        help="Encoding of each 4-bit voltage component",
    )
    parser.add_argument(
        "--iq-order",
        choices=("ri", "ir"),
        default="ri",
        help="ri: low nibble real; ir: low nibble imaginary",
    )
    parser.add_argument(
        "--exclude-zero-spectra",
        action="store_true",
        help="Exclude all-zero spectra from means (useful for zeroSamples diagnosis)",
    )
    parser.add_argument("--scale", choices=("linear", "db"), default="db")
    parser.add_argument("--db-floor", type=float, default=-60.0)
    parser.add_argument("--color-min", type=float, help="Fixed heatmap lower color limit")
    parser.add_argument("--color-max", type=float, help="Fixed heatmap upper color limit")
    parser.add_argument("--spectra-grid", action="store_true", help="Also create antenna panels")
    parser.add_argument(
        "--grid-stat", choices=("mean", "max"), default="mean", help="Statistic for panels"
    )
    parser.add_argument("--output-prefix", type=Path)
    return parser.parse_args()


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.file_count < 0:
        parser.error("--file-count must be non-negative")
    if args.time_bin_spectra <= 0:
        parser.error("--time-bin-spectra must be positive")
    if args.spectra_per_second <= 0:
        parser.error("--spectra-per-second must be positive")
    if args.start_spectrum < 0:
        parser.error("--start-spectrum must be non-negative")
    if args.channel_stop is None:
        args.channel_stop = args.channels
    if not 0 <= args.channel_start < args.channel_stop <= args.channels:
        parser.error("channel range must satisfy 0 <= start < stop <= channels")
    if args.color_min is not None and args.color_max is not None:
        if args.color_min >= args.color_max:
            parser.error("--color-min must be lower than --color-max")


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    del parser
    args = parse_args()

    # argparse's parser.error is only needed for consistent user-facing validation.
    validation_parser = argparse.ArgumentParser(prog=Path(__file__).name, add_help=False)
    validate_args(args, validation_parser)

    try:
        antennas = parse_antenna_selection(args.antennas, args.elements)
    except ValueError as error:
        raise SystemExit(f"Invalid --antennas: {error}") from error

    handlers, file_groups = select_file_groups(args)
    possible_spectra = sum(group.spectra for group in file_groups)
    stop_spectrum = args.stop_spectrum if args.stop_spectrum is not None else possible_spectra
    if not args.start_spectrum < stop_spectrum <= possible_spectra:
        raise SystemExit(
            "Spectrum range must satisfy 0 <= start < stop <= "
            f"{possible_spectra} for the selected files"
        )

    elements = np.asarray(
        [physical_antenna_to_element(antenna, args.elements) for antenna in antennas],
        dtype=np.intp,
    )
    local_channels = np.arange(args.channel_start, args.channel_stop, dtype=np.int64)
    global_channels = np.concatenate(
        [handler * args.channels + local_channels for handler in handlers]
    )
    frequency_mhz = (
        DEFAULT_FREQ0_MHZ + global_channels * DEFAULT_CHANNEL_WIDTH_MHZ
    )
    output_local_channels = np.tile(local_channels, len(handlers))
    selected_channels = len(global_channels)

    frequency_sum = np.zeros((len(antennas), selected_channels), dtype=np.float64)
    frequency_max = np.full((len(antennas), selected_channels), -np.inf, dtype=np.float32)
    included_spectra = 0
    zero_spectra = 0
    inspected_spectra = 0
    time_centers: list[float] = []
    time_mean_rows: list[np.ndarray] = []
    time_max_rows: list[np.ndarray] = []
    time_zero_fraction: list[float] = []
    fpga_sequences: list[int] = []
    group_spectra: list[int] = []
    time_origin_fpga: float | None = None

    print(
        f"handlers={handlers}, aligned intervals={len(file_groups)}, antennas={antennas}, "
        f"local channels per handler=[{args.channel_start}, {args.channel_stop}), "
        f"global channels={global_channels[0]}..{global_channels[-1]}"
    )
    print("Selected aligned intervals:")
    for group in file_groups:
        file_names = " | ".join(path.name for path in group.paths)
        print(
            f"  fpga=[{group.fpga_start}, {group.fpga_start + group.spectra}), "
            f"offsets={group.handler_offsets}: {file_names}"
        )

    concatenated_start = 0
    for group in file_groups:
        paths = group.paths
        frames: list[np.memmap] = []
        file_sequences: list[int] = []
        metadata_sizes: list[int] = []
        for path in paths:
            frame, fpga_seq, metadata_size = read_raw_frame(
                path,
                args.spectra_per_frame,
                args.channels,
                args.elements,
                args.metadata_endian,
            )
            frames.append(frame)
            file_sequences.append(fpga_seq)
            metadata_sizes.append(metadata_size)

        expected_offsets = tuple(group.fpga_start - sequence for sequence in file_sequences)
        if expected_offsets != group.handler_offsets:
            raise SystemExit(
                "Internal alignment error: metadata changed while reading files; "
                f"expected offsets {group.handler_offsets}, found {expected_offsets}"
            )
        fpga_sequences.append(group.fpga_start)
        group_spectra.append(group.spectra)

        file_global_start = concatenated_start
        concatenated_start += group.spectra
        local_start = max(0, args.start_spectrum - file_global_start)
        local_stop = min(group.spectra, stop_spectrum - file_global_start)
        if local_start >= local_stop:
            del frames
            continue

        for start in range(local_start, local_stop, args.time_bin_spectra):
            stop = min(start + args.time_bin_spectra, local_stop)
            raw_parts: list[np.ndarray] = []
            zero_by_handler: list[np.ndarray] = []
            for frame, handler_offset in zip(frames, group.handler_offsets):
                frame_start = handler_offset + start
                frame_stop = handler_offset + stop
                raw_part = np.asarray(
                    frame[
                        frame_start:frame_stop,
                        args.channel_start : args.channel_stop,
                        :,
                    ]
                )
                raw_parts.append(raw_part)
                zero_by_handler.append(np.all(raw_part == 0, axis=(1, 2)))

            raw_all_elements = (
                np.concatenate(raw_parts, axis=1) if len(raw_parts) > 1 else raw_parts[0]
            )
            # A combined spectrum is incomplete if either handler is entirely zero.
            any_handler_zero = np.any(np.stack(zero_by_handler, axis=0), axis=0)
            raw_selected = np.take(raw_all_elements, elements, axis=2)
            real, imag = decode_components(raw_selected, args.encoding, args.iq_order)
            power = real * real + imag * imag

            inspected_spectra += power.shape[0]
            zero_spectra += int(np.count_nonzero(any_handler_zero))
            valid = (
                ~any_handler_zero
                if args.exclude_zero_spectra
                else np.ones_like(any_handler_zero, dtype=bool)
            )
            valid_count = int(np.count_nonzero(valid))

            if valid_count > 0:
                valid_power = power[valid]
                frequency_sum += np.sum(valid_power, axis=0, dtype=np.float64).T
                frequency_max = np.maximum(
                    frequency_max, np.max(valid_power, axis=0).T
                )
                included_spectra += valid_count
                time_mean = np.mean(valid_power, axis=(0, 1), dtype=np.float64)
                time_max = np.max(valid_power, axis=(0, 1))
            else:
                time_mean = np.full(len(antennas), np.nan, dtype=np.float64)
                time_max = np.full(len(antennas), np.nan, dtype=np.float32)

            fpga_center = group.fpga_start + 0.5 * (start + stop)
            if time_origin_fpga is None:
                time_origin_fpga = float(group.fpga_start + start)
            time_centers.append((fpga_center - time_origin_fpga) / args.spectra_per_second)
            time_mean_rows.append(time_mean)
            time_max_rows.append(time_max)
            time_zero_fraction.append(float(np.mean(any_handler_zero)))

        del frames
        details = ", ".join(
            f"handler {handler}: {path.name} (file_seq={sequence}, "
            f"offset={offset}, metadata={metadata_size})"
            for handler, path, sequence, offset, metadata_size in zip(
                handlers,
                paths,
                file_sequences,
                group.handler_offsets,
                metadata_sizes,
            )
        )
        print(
            f"Processed aligned fpga=[{group.fpga_start}, "
            f"{group.fpga_start + group.spectra}), {details}, "
            f"overlap spectra=[{local_start}, {local_stop})"
        )

    if included_spectra == 0:
        raise SystemExit("No non-excluded spectra remain for the intensity average")

    mean_intensity = frequency_sum / included_spectra
    frequency_max[~np.isfinite(frequency_max)] = np.nan
    time_mean = np.asarray(time_mean_rows, dtype=np.float64)
    time_max = np.asarray(time_max_rows, dtype=np.float32)
    time_seconds = np.asarray(time_centers, dtype=np.float64)
    zero_fraction = zero_spectra / inspected_spectra if inspected_spectra else math.nan

    if len(fpga_sequences) > 1:
        starts = np.asarray(fpga_sequences, dtype=np.int64)
        lengths = np.asarray(group_spectra, dtype=np.int64)
        gaps = starts[1:] - (starts[:-1] + lengths[:-1])
        if np.any(gaps != 0):
            print(f"WARNING: gaps between aligned FPGA intervals: {gaps[gaps != 0]}")

    output_prefix = args.output_prefix or Path(f"raw_intensity_network_{args.handler}")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    frequency_path = output_prefix.with_name(f"{output_prefix.name}_frequency.png")
    time_path = output_prefix.with_name(f"{output_prefix.name}_time.png")
    npz_path = output_prefix.with_suffix(".npz")

    try:
        plot_frequency_maps(
            mean_intensity,
            frequency_max,
            antennas,
            frequency_mhz,
            args.scale,
            args.db_floor,
            args.color_min,
            args.color_max,
            frequency_path,
        )
        plot_time_maps(
            time_mean,
            time_max,
            time_seconds,
            antennas,
            args.scale,
            args.db_floor,
            args.color_min,
            args.color_max,
            time_path,
        )
        if args.spectra_grid:
            grid_path = output_prefix.with_name(f"{output_prefix.name}_grid.png")
            grid_data = mean_intensity if args.grid_stat == "mean" else frequency_max
            plot_spectra_grid(
                grid_data,
                antennas,
                frequency_mhz,
                args.grid_stat,
                args.scale,
                args.db_floor,
                grid_path,
            )
            print(f"Saved {grid_path}")
    except ImportError as error:
        raise SystemExit("matplotlib is required to generate intensity plots") from error

    np.savez(
        npz_path,
        antenna=np.asarray(antennas, dtype=np.int16),
        element=elements,
        handler=np.asarray(handlers, dtype=np.int8),
        local_channel=output_local_channels,
        global_channel=global_channels,
        frequency_mhz=frequency_mhz,
        mean_intensity=mean_intensity,
        max_intensity=frequency_max,
        time_seconds=time_seconds,
        time_mean_intensity=time_mean,
        time_max_intensity=time_max,
        time_zero_fraction=np.asarray(time_zero_fraction, dtype=np.float64),
        fpga_seq=np.asarray(fpga_sequences, dtype=np.int64),
        aligned_spectra=np.asarray(group_spectra, dtype=np.int64),
        handler_offset=np.asarray(
            [group.handler_offsets for group in file_groups], dtype=np.int64
        ),
        files=np.asarray(
            [[str(path) for path in group.paths] for group in file_groups]
        ),
        inspected_spectra=inspected_spectra,
        included_spectra=included_spectra,
        zero_spectra=zero_spectra,
        encoding=args.encoding,
        iq_order=args.iq_order,
        metadata_endian=args.metadata_endian,
    )

    print(
        f"All-zero spectra in selected data: {zero_spectra}/{inspected_spectra} "
        f"({100.0 * zero_fraction:.3f}%)"
    )
    print(f"Spectra included in mean: {included_spectra}")
    print(f"Saved {frequency_path}")
    print(f"Saved {time_path}")
    print(f"Saved {npz_path}")


if __name__ == "__main__":
    main()
