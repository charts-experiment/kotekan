#!/usr/bin/env python3
"""Convert CHARTS BasebandWriter raw files into one virtual HDF5 dataset.

The output VDS maps one contiguous frequency block from each input raw file
into /baseband with axes (antenna, frequency, time).  Its source files live in
<output-stem>.sources and must be kept alongside the VDS.
"""

import argparse
import datetime as dt
import os
from pathlib import Path
import re
import struct
import sys

import h5py
import numpy as np


METADATA_SIZE = 96
METADATA_STRUCT = struct.Struct("<QQQQQdddqqQii")
RAW_FILE_RE = re.compile(r"^baseband_(?P<event>\d+)_(?P<freq>\d+)(?:\.data)?$")
FORMAT_VERSION = "charts-baseband-raw2h-v1"


# Fixed backing-file layout, matching the old converter's ~500 MB target.
# 64 antennas x 84 frequencies x 93,000 uint8 samples = 499,968,000 bytes.
FREQ_CHUNK = 672 // 8
MAX_SAMPLES_CHUNK = 93_000
FRAME_METADATA_DTYPE = np.dtype(
    [
        ("event_id", "<u8"),
        ("freq_id", "<u8"),
        ("event_start_fpga", "<u8"),
        ("event_end_fpga", "<u8"),
        ("time0_fpga", "<u8"),
        ("time0_ctime", "<f8"),
        ("time0_ctime_offset", "<f8"),
        ("first_packet_recv_time", "<f8"),
        ("frame_fpga_seq", "<i8"),
        ("valid_to", "<i8"),
        ("fpga0_ns", "<u8"),
        ("num_elements", "<i4"),
        ("reserved", "<i4"),
        ("time_index", "<i8"),
    ]
)


def fail(message):
    raise RuntimeError(message)


def parse_metadata(raw):
    if len(raw) != METADATA_SIZE:
        fail(f"expected {METADATA_SIZE} metadata bytes, got {len(raw)}")
    return METADATA_STRUCT.unpack(raw)


def metadata_record(values, time_index):
    return tuple(values) + (time_index,)


def discover_raw_files(raw_dir, expected_event_id):
    files = []
    for path in sorted(Path(raw_dir).iterdir()):
        match = RAW_FILE_RE.match(path.name)
        if path.is_file() and match:
            event_id = int(match.group("event"))
            if expected_event_id is None or event_id == expected_event_id:
                files.append((path, event_id, int(match.group("freq"))))
    if not files:
        fail(f"no baseband_<event>_<freq> raw files found in {raw_dir}")
    return files


def inspect_file(path, filename_event_id, filename_freq_id, args):
    data_bytes = args.spectra_per_frame * args.local_num_freq * args.num_elements
    record_bytes = METADATA_SIZE + data_bytes
    size = path.stat().st_size
    if size == 0 or size % record_bytes:
        fail(
            f"{path}: size {size} is not a whole number of {record_bytes}-byte records; "
            "check --spectra-per-frame, --local-num-freq, and --num-elements"
        )

    frames = []
    with path.open("rb") as raw_file:
        for frame_index in range(size // record_bytes):
            values = parse_metadata(raw_file.read(METADATA_SIZE))
            event_id, freq_id = values[0], values[1]
            valid_to = values[9]
            if event_id != filename_event_id or freq_id != filename_freq_id:
                fail(
                    f"{path}, frame {frame_index}: filename IDs "
                    f"{filename_event_id}/{filename_freq_id} disagree with metadata {event_id}/{freq_id}"
                )
            if values[11] != args.num_elements:
                fail(
                    f"{path}, frame {frame_index}: metadata has {values[11]} elements, "
                    f"expected {args.num_elements}"
                )
            if not 0 < valid_to <= args.spectra_per_frame:
                fail(
                    f"{path}, frame {frame_index}: invalid valid_to={valid_to}; "
                    "this converter requires raw files produced by the updated chartsBasebandReadout"
                )
            frames.append(values)
            raw_file.seek(data_bytes, os.SEEK_CUR)

    if not frames:
        fail(f"{path}: contains no frames")
    return {"path": path, "event_id": filename_event_id, "freq_id": filename_freq_id,
            "frames": frames, "data_bytes": data_bytes, "record_bytes": record_bytes}


def validate_streams(streams, args):
    event_ids = {stream["event_id"] for stream in streams}
    if len(event_ids) != 1:
        fail(f"input files contain multiple event IDs: {sorted(event_ids)}")
    freq_ids = [stream["freq_id"] for stream in streams]
    if len(freq_ids) != len(set(freq_ids)):
        fail("multiple raw files have the same freq_id")
    for stream in streams:
        f0 = stream["freq_id"]
        if f0 + args.local_num_freq > args.total_num_freq:
            fail(f"{stream['path']}: frequency range {f0}:{f0 + args.local_num_freq} exceeds "
                 f"--total-num-freq={args.total_num_freq}")
    for left, right in zip(sorted(streams, key=lambda item: item["freq_id"]),
                           sorted(streams, key=lambda item: item["freq_id"])[1:]):
        if left["freq_id"] + args.local_num_freq > right["freq_id"]:
            fail("input frequency blocks overlap")

    # rfsocHandlerShuffle records a packet timestamp in every frame, so time0_fpga
    # legitimately differs between frames and NICs. Preserve it in frame_metadata
    # and use the earliest frame only as the event-level time reference.
    earliest_frame = min((frame for stream in streams for frame in stream["frames"]),
                         key=lambda frame: frame[8])
    first_fpga = earliest_frame[8]
    last_fpga = max(frame[8] + frame[9] for stream in streams for frame in stream["frames"])
    if (last_fpga - first_fpga) <= 0:
        fail("input frames have an empty time range")
    return next(iter(event_ids)), earliest_frame[4], first_fpga, last_fpga


def source_descriptor(source_dir, freq_start, freq_end, chunk_index, total_samples):
    time_start = chunk_index * MAX_SAMPLES_CHUNK
    return {
        "path": source_dir / f"bb_f{freq_start:04d}_{freq_end:04d}_chunk{chunk_index:05d}.h5",
        "freq_start": freq_start,
        "freq_end": freq_end,
        "time_start": time_start,
        "time_end": min(time_start + MAX_SAMPLES_CHUNK, total_samples),
    }


def create_source_dataset(descriptor, args):
    descriptor["path"].parent.mkdir(parents=True, exist_ok=True)
    freq_len = descriptor["freq_end"] - descriptor["freq_start"]
    time_len = descriptor["time_end"] - descriptor["time_start"]
    h5_file = h5py.File(descriptor["path"], "w")
    dataset = h5_file.create_dataset(
        "baseband", (args.num_elements, freq_len, time_len), np.uint8,
        chunks=(args.num_elements, freq_len, min(time_len, 1024)), fillvalue=0,
    )
    dataset.attrs["axes"] = ["antenna", "frequency", "time"]
    dataset.attrs["freq_start_idx"] = descriptor["freq_start"]
    dataset.attrs["freq_end_idx"] = descriptor["freq_end"]
    dataset.attrs["time_start_idx"] = descriptor["time_start"]
    return h5_file, dataset


def write_stream_sources(stream, source_dir, first_fpga, total_samples, args):
    open_sources = {}
    descriptors = {}
    stream_f0 = stream["freq_id"]
    stream_f1 = stream_f0 + args.local_num_freq
    with stream["path"].open("rb") as raw_file:
        for index, values in enumerate(stream["frames"]):
            raw_file.seek(METADATA_SIZE, os.SEEK_CUR)
            valid_to = values[9]
            time_index = values[8] - first_fpga
            if time_index < 0 or time_index + valid_to > total_samples:
                fail(f"{stream['path']}, frame {index}: frame is outside calculated time range")
            data = np.frombuffer(raw_file.read(stream["data_bytes"]), dtype=np.uint8)
            data = data.reshape(args.spectra_per_frame, args.local_num_freq, args.num_elements)
            input_time = 0
            while input_time < valid_to:
                global_time = time_index + input_time
                chunk_index = global_time // MAX_SAMPLES_CHUNK
                chunk_end = min((chunk_index + 1) * MAX_SAMPLES_CHUNK, total_samples)
                copy_len = min(valid_to - input_time, chunk_end - global_time)
                for freq_start in range(stream_f0, stream_f1, FREQ_CHUNK):
                    freq_end = min(freq_start + FREQ_CHUNK, stream_f1)
                    key = (freq_start, chunk_index)
                    if key not in open_sources:
                        descriptor = source_descriptor(
                            source_dir, freq_start, freq_end, chunk_index, total_samples
                        )
                        descriptors[key] = descriptor
                        open_sources[key] = create_source_dataset(descriptor, args)
                    _, dataset = open_sources[key]
                    local_f0 = freq_start - stream_f0
                    local_f1 = freq_end - stream_f0
                    local_t0 = global_time - descriptors[key]["time_start"]
                    dataset[:, :, local_t0:local_t0 + copy_len] = np.transpose(
                        data[input_time:input_time + copy_len, local_f0:local_f1, :], (2, 1, 0)
                    )
                input_time += copy_len
    for h5_file, _ in open_sources.values():
        h5_file.close()
    return list(descriptors.values())


def write_vds(output_path, source_descriptors, streams, event_id, time0_fpga, first_fpga,
              total_samples, args):
    layout = h5py.VirtualLayout(
        shape=(args.num_elements, args.total_num_freq, total_samples), dtype=np.uint8
    )
    records = []
    for descriptor in source_descriptors:
        relative_source = os.path.relpath(descriptor["path"], output_path.parent)
        source = h5py.VirtualSource(
            relative_source, "baseband", shape=(
                args.num_elements,
                descriptor["freq_end"] - descriptor["freq_start"],
                descriptor["time_end"] - descriptor["time_start"],
            )
        )
        layout[:, descriptor["freq_start"]:descriptor["freq_end"],
               descriptor["time_start"]:descriptor["time_end"]] = source
    for stream in streams:
        records.extend(metadata_record(values, values[8] - first_fpga) for values in stream["frames"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as h5_file:
        h5_file.attrs["format"] = FORMAT_VERSION
        h5_file.attrs["event_id"] = event_id
        h5_file.attrs["num_elements"] = args.num_elements
        h5_file.attrs["total_num_freq"] = args.total_num_freq
        h5_file.attrs["local_num_freq"] = args.local_num_freq
        h5_file.attrs["spectra_per_frame"] = args.spectra_per_frame
        h5_file.attrs["reference_time0_fpga"] = time0_fpga
        h5_file.attrs["time0_fpga_per_frame"] = True
        h5_file.attrs["first_frame_fpga_seq"] = first_fpga
        h5_file.attrs["delta_time_us"] = args.delta_time_us
        h5_file.attrs["freq0_mhz"] = args.freq0_mhz
        h5_file.attrs["delta_freq_mhz"] = args.delta_freq_mhz
        h5_file.attrs["freq_chunk"] = FREQ_CHUNK
        h5_file.attrs["max_samples_chunk"] = MAX_SAMPLES_CHUNK
        h5_file.attrs["source_directory"] = os.path.relpath(
            source_descriptors[0]["path"].parent, output_path.parent
        )
        # rfsocHandlerShuffle stores time0_fpga as the absolute packet time in
        # microseconds; frame_fpga_seq is only an ordering/alignment index.
        start_time_us = time0_fpga
        h5_file.attrs["start_time_utc_us"] = start_time_us
        h5_file.attrs["start_time_utc"] = dt.datetime.fromtimestamp(
            start_time_us / 1_000_000, tz=dt.timezone.utc
        ).isoformat()

        dataset = h5_file.create_virtual_dataset("baseband", layout, fillvalue=0)
        dataset.attrs["axes"] = ["antenna", "frequency", "time"]
        dataset.attrs["sample_dtype"] = "uint8 packed RFSoC voltage"
        dataset.attrs["description"] = "CHARTS baseband virtual dataset"
        metadata = np.array(records, dtype=FRAME_METADATA_DTYPE)
        h5_file.create_dataset("frame_metadata", data=metadata)
        h5_file.create_dataset("stream_freq_ids", data=np.array([s["freq_id"] for s in streams], dtype="u8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_dir", type=Path, help="Directory containing BasebandWriter raw files")
    parser.add_argument("--output", type=Path,
                        help="Output VDS .h5 path (overrides the dated default directory)")
    parser.add_argument("--outdir-base", "-o", type=Path, default=Path("/hdd"),
                        help="Base directory for the dated output directory (default: /hdd)")
    parser.add_argument("--spectra-per-frame", type=int, default=15360)
    parser.add_argument("--num-elements", type=int, default=64)
    parser.add_argument("--local-num-freq", type=int, default=336)
    parser.add_argument("--total-num-freq", type=int, default=672)
    parser.add_argument("--freq0-mhz", type=float, default=300.0)
    parser.add_argument("--delta-freq-mhz", type=float, default=0.3)
    parser.add_argument("--delta-time-us", type=float, default=10.0 / 3.0)
    parser.add_argument("--event-id", type=int, help="Require this event ID")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    if not args.raw_dir.is_dir():
        parser.error(f"not a directory: {args.raw_dir}")
    if min(args.spectra_per_frame, args.num_elements, args.local_num_freq, args.total_num_freq) <= 0:
        parser.error("frame and dimension values must be positive")

    try:
        raw_files = discover_raw_files(args.raw_dir, args.event_id)
        streams = [inspect_file(*raw_file, args) for raw_file in raw_files]
        event_id, time0_fpga, first_fpga, last_fpga = validate_streams(streams, args)
        total_samples = last_fpga - first_fpga
        start_time_us = time0_fpga

        if args.output is None:
            timestamp = dt.datetime.fromtimestamp(
                start_time_us / 1_000_000, tz=dt.timezone.utc
            ).strftime("%y%m%dT%H%M%SZ")
            args.output = args.outdir_base / f"{timestamp}_CHARTS_hdf5" / "baseband_virtual.h5"
        if args.output.exists() and args.output.is_dir():
            parser.error("--output must be an .h5 file; use --outdir-base for a dated output directory")
        if args.output.exists() and not args.overwrite and not args.verify_only:
            parser.error(f"output exists: {args.output}; use --overwrite to replace it")

        for stream in streams:
            print(f"  {stream['path'].name}: freq {stream['freq_id']}:{stream['freq_id'] + args.local_num_freq}, "
                  f"{len(stream['frames'])} frames")
        if args.verify_only:
            print(f"Output VDS: {args.output}")
            return

        source_dir = args.output.with_suffix("").with_name(args.output.stem + ".sources")
        if source_dir.exists() and not args.overwrite:
            fail(f"source directory exists: {source_dir}; use --overwrite to replace it")
        if source_dir.exists():
            for source in source_dir.glob("*.h5"):
                source.unlink()
        source_descriptors = []
        for stream in streams:
            source_descriptors.extend(
                write_stream_sources(stream, source_dir, first_fpga, total_samples, args)
            )
        write_vds(args.output, source_descriptors, streams, event_id, time0_fpga, first_fpga,
                  total_samples, args)
        print(f"Created VDS: {args.output}")
        print(f"Keep backing source files: {source_dir}")
    except RuntimeError as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
