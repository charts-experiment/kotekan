"""Parity and serialization checks for ``chartsMetadata`` and its Python readers.

Verifies without requiring a Linux/DPDK build that:
1. ``chartsMetadata::serialize()`` populates ``first_packet_recv_time`` in
   ``chartsMetadataFormat`` so raw-file / IPC round-trips preserve timestamps.
2. ``chartsMetadata::to_json()`` emits both ``time_downsampling_fpga`` and
   ``first_packet_recv_time``.
3. The binary tail layout of ``chartsMetadataFormat`` matches the hardcoded
   offsets unpacked by ``charts/analyze_tonight_grus.py``.
"""

from __future__ import annotations

from pathlib import Path
import re
import struct

KOTEKAN_ROOT = Path(__file__).resolve().parents[2]
META_CPP = KOTEKAN_ROOT / "lib" / "metadata" / "chartsMetadata.cpp"
META_HPP = KOTEKAN_ROOT / "lib" / "metadata" / "chartsMetadata.hpp"
GRUS_SCRIPT = KOTEKAN_ROOT / "charts" / "analyze_tonight_grus.py"


def _extract_block(text: str, start_marker: str, end_marker: str) -> str:
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    return text[start:end]


def test_serialize_populates_first_packet_recv_time():
    src = META_CPP.read_text(encoding="utf-8")
    serialize_body = _extract_block(
        src,
        "size_t chartsMetadata::serialize(",
        "size_t chartsMetadata::set_from_bytes(",
    )
    assert "fmt->first_packet_recv_time" in serialize_body
    assert "has_first_packet_recv_time()" in serialize_body
    assert "get_first_packet_recv_time()" in serialize_body


def test_to_json_includes_downsampling_and_recv_time():
    src = META_CPP.read_text(encoding="utf-8")
    to_json_body = src[src.index("nlohmann::json chartsMetadata::to_json()") :]
    assert 'j["time_downsampling_fpga"]' in to_json_body
    assert 'j["first_packet_recv_time"]' in to_json_body


def test_charts_metadata_tail_layout_matches_python_readers():
    """Guard the byte offsets unpacked by analyze_tonight_grus.py (64-bit LP64 Linux)."""
    hpp = META_HPP.read_text(encoding="utf-8")
    cpp = META_CPP.read_text(encoding="utf-8")
    constants_hpp = (KOTEKAN_ROOT / "lib" / "core" / "chartsConstants.hpp").read_text(
        encoding="utf-8"
    )

    max_dim = int(re.search(r"const int CHARTS_META_MAX_DIM\s*=\s*(\d+);", hpp).group(1))
    max_dimname = int(
        re.search(r"const int CHARTS_META_MAX_DIMNAME\s*=\s*(\d+);", hpp).group(1)
    )
    max_freq = int(
        re.search(r"charts_full_band_channels\s*=\s*(\d+);", constants_hpp).group(1)
    )

    # Compute x86_64 LP64 struct offsets for chartsMetadataFormat up to the tail fields:
    # int32 frame_counter (0), 4B pad, int64 fpga_seq_num (8), int32 time_downsampling_fpga (16),
    # char name[MAX_DIMNAME] (20), int32 type, int32 dims, int32 dim[MAX_DIM],
    # char dim_name[MAX_DIM][MAX_DIMNAME], int64 stride[MAX_DIM], int64 offset,
    # int32 nfreq, int32 coarse_freq[MAX_FREQ], int32 lost_timesamples,
    # 4B pad, timeval(int64, int64) first_packet_recv_time, uint64 time0_fpga.
    def align(offset: int, alignment: int) -> int:
        return (offset + alignment - 1) // alignment * alignment

    off = 0
    off += 4  # frame_counter
    off = align(off, 8) + 8  # fpga_seq_num
    off += 4  # time_downsampling_fpga
    off += max_dimname  # name
    off = align(off, 4) + 4  # type
    off += 4  # dims
    off += 4 * max_dim  # dim
    off += max_dim * max_dimname  # dim_name
    off = align(off, 8) + 8 * max_dim  # stride
    off += 8  # offset
    off += 4  # nfreq
    off += 4 * max_freq  # coarse_freq
    lost_offset = off
    off += 4  # lost_timesamples
    recv_time_offset = align(off, 8)
    off = recv_time_offset + struct.calcsize("<qq")  # timeval on 64-bit Linux
    time0_offset = align(off, 8)
    total_size = align(time0_offset + 8, 8)

    assert "struct chartsMetadataFormat" in cpp
    grus_src = GRUS_SCRIPT.read_text(encoding="utf-8")
    assert f"len(meta_buf) >= {total_size}" in grus_src
    assert f"struct.unpack_from(\"<i\", meta_buf, {lost_offset})" in grus_src
    assert f"struct.unpack_from(\"<qq\", meta_buf, {recv_time_offset})" in grus_src
    assert f"struct.unpack_from(\"<Q\", meta_buf, {time0_offset})" in grus_src
