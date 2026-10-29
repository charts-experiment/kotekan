"""Readers for recorded CHARTS data: triggered baseband and correlator frames.

* Triggered baseband (`BasebandWriter` / `BasebandFileRaw`): fixed-size records of the
  96-byte BasebandMetadataFormat followed by `[spectra][num_local_freq][num_elements]`
  raw int4x2 bytes. chartsTriggeredBasebandReadout copies element `63 - a` of the input
  frame for each selected charts_logical antenna `a`, so the last axis is in the
  selection order and bytes are untouched (x = low nibble + i*high nibble).
* Correlator frames (rawFileWrite of chartsAccumulate output): uint32 metadata size,
  chartsMetadata, then int32 `[F][528][2][2][re, im]` in the ASTRON receiver layout.
  `corr_matrix` maps it to charts_logical V'_ab = x'_a * conj(x'_b) (x' = high nibble
  + i*low nibble), following charts-utils `gpu_corr_to_visibilities`, but keeps exact
  int64 values.
"""

from __future__ import annotations

import ctypes as ct
from dataclasses import dataclass
from pathlib import Path
import struct

import numpy as np

from .packet_loss import NativeChartsMetadata

N_ELEMENTS = 64
N_POL = 2
N_RECV = N_ELEMENTS // N_POL
N_BASELINES = N_RECV * (N_RECV + 1) // 2

BASEBAND_META = np.dtype([
    ("event_id", "<u8"), ("freq_id", "<u8"), ("event_start_fpga", "<u8"),
    ("event_end_fpga", "<u8"), ("time0_fpga", "<u8"), ("time0_ctime", "<f8"),
    ("time0_ctime_offset", "<f8"), ("first_packet_recv_time", "<f8"),
    ("frame_fpga_seq", "<i8"), ("valid_to", "<i8"), ("fpga0_ns", "<u8"),
    ("num_elements", "<i4"), ("reserved", "<i4"),
])
assert BASEBAND_META.itemsize == 96


@dataclass(frozen=True)
class BasebandFile:
    path: Path
    spectra_per_record: int
    num_freq: int
    num_elements: int
    meta: np.ndarray  # BASEBAND_META per record

    @property
    def record_bytes(self) -> int:
        return BASEBAND_META.itemsize + self.spectra_per_record * self.num_freq * self.num_elements

    def record(self, i: int) -> np.ndarray:
        """Valid spectra of record i as uint8 [valid_to][num_freq][num_elements]."""
        n = int(self.meta["valid_to"][i])
        return np.fromfile(self.path, dtype=np.uint8,
                           count=n * self.num_freq * self.num_elements,
                           offset=i * self.record_bytes + BASEBAND_META.itemsize
                           ).reshape(n, self.num_freq, self.num_elements)


def open_baseband(path: Path, spectra_per_record: int, num_freq: int = 336) -> BasebandFile:
    path = Path(path)
    with open(path, "rb") as f:
        first = np.frombuffer(f.read(BASEBAND_META.itemsize), BASEBAND_META)[0]
    num_elements = int(first["num_elements"])
    record = BASEBAND_META.itemsize + spectra_per_record * num_freq * num_elements
    size = path.stat().st_size
    if size % record:
        raise ValueError(f"{path}: {size} bytes is not a multiple of the {record}-byte record")
    count = size // record
    meta = np.empty(count, BASEBAND_META)
    with open(path, "rb") as f:
        for i in range(count):
            f.seek(i * record)
            meta[i] = np.frombuffer(f.read(BASEBAND_META.itemsize), BASEBAND_META)[0]
    if np.any(meta["num_elements"] != num_elements):
        raise ValueError(f"{path}: num_elements changes between records")
    return BasebandFile(path, spectra_per_record, num_freq, num_elements, meta)


def read_charts_metadata(path: Path) -> NativeChartsMetadata:
    with open(path, "rb") as f:
        (size,) = struct.unpack("=I", f.read(4))
        if size != ct.sizeof(NativeChartsMetadata):
            raise ValueError(f"{path}: metadata size {size} != {ct.sizeof(NativeChartsMetadata)}")
        return NativeChartsMetadata.from_buffer_copy(f.read(size))


def read_corr_raw(path: Path, num_freq: int = 336) -> np.ndarray:
    """int64 complex parts [F][528][2][2][2] (exact)."""
    offset = 4 + ct.sizeof(NativeChartsMetadata)
    n = num_freq * N_BASELINES * N_POL * N_POL * 2
    raw = np.fromfile(path, dtype="<i4", count=n, offset=offset)
    if raw.size != n:
        raise ValueError(f"{path}: truncated correlation payload")
    return raw.reshape(num_freq, N_BASELINES, N_POL, N_POL, 2).astype(np.int64)


def _corr_index_tables():
    """For charts_logical (a, b): baseline index, pol_a, pol_b and conjugation flag."""
    a, b = np.meshgrid(np.arange(N_ELEMENTS), np.arange(N_ELEMENTS), indexing="ij")
    ra, rb = N_ELEMENTS - 1 - a, N_ELEMENTS - 1 - b
    recv_a, pol_a = ra // N_POL, ra % N_POL
    recv_b, pol_b = rb // N_POL, rb % N_POL
    direct = recv_a >= recv_b
    hi = np.where(direct, recv_a, recv_b)
    lo = np.where(direct, recv_b, recv_a)
    bidx = hi * (hi + 1) // 2 + lo
    p1 = np.where(direct, pol_a, pol_b)
    p2 = np.where(direct, pol_b, pol_a)
    return bidx, p1, p2, ~direct


_TABLES = _corr_index_tables()


def corr_matrix(raw: np.ndarray, antennas=None) -> np.ndarray:
    """Exact complex128 V'[F][A][A] in charts_logical order for the given antennas
    (default all 64). V'_ab = x'_a * conj(x'_b); integer-valued real and imaginary parts.

    Entries with a < b are read as charts-utils does; a > b is their conjugate. The ASTRON
    layout also stores the swapped polarization order inside one receiver, which differs
    from that conjugate only through the correlator's -8 nibble imaginary corruption (as
    does the imaginary part of the diagonal); `corr_auto_imag` reports that diagnostic."""
    sel = np.arange(N_ELEMENTS) if antennas is None else np.asarray(antennas)
    if np.any(np.diff(sel) <= 0):
        raise ValueError("antennas must be strictly increasing")
    bidx, p1, p2, conj = (t[np.ix_(sel, sel)] for t in _TABLES)
    v = raw[:, bidx, p1, p2]  # [F][A][A][2]
    re, im = v[..., 0], np.where(conj[None], -v[..., 1], v[..., 1])
    m = re.astype(np.float64) + 1j * im.astype(np.float64)
    upper = np.triu(np.ones(len(sel), bool), 1)
    m = np.where(upper[None], m, np.conj(np.swapaxes(m, 1, 2)))
    diag = np.arange(len(sel))
    m[:, diag, diag] = m[:, diag, diag].real
    return m


def corr_auto_imag(raw: np.ndarray, antennas=None) -> np.ndarray:
    """Imaginary part of the stored autocorrelations [F][A]; nonzero only through -8 nibbles."""
    sel = np.arange(N_ELEMENTS) if antennas is None else np.asarray(antennas)
    bidx, p1, p2, _ = (t[sel, sel] for t in _TABLES)
    return raw[:, bidx, p1, p2, 1]


# int4x2 decoding in the beamformer convention (low nibble real).
_NIB = ((np.arange(16) ^ 8) - 8).astype(np.int64)
BYTE_RE = _NIB[np.arange(256) & 0xF]
BYTE_IM = _NIB[np.arange(256) >> 4]
BYTE_POWER = BYTE_RE ** 2 + BYTE_IM ** 2


def convert_baseband(bb: BasebandFile, antennas, shard: int, out_dir: Path, name: str):
    """Write the full records of a baseband file as native 64-element rawFileRead records
    `{name}_{n:07d}.bin` ([T][336][64], antenna a in kotekan_raw slot 63 - a, raw bytes,
    other inputs 0x00, FPGA_SEQ_NUM from the baseband metadata).

    Returns (seqs [N], minus8 [N][336][A]): per record and channel, the samples of each
    antenna with a -8 nibble in either component."""
    from .packet_loss import write_frame

    if bb.num_elements != len(antennas):
        raise ValueError("antenna selection does not match the baseband records")
    full = np.nonzero(bb.meta["valid_to"] == bb.spectra_per_record)[0]
    frame = np.zeros((bb.spectra_per_record, bb.num_freq, N_ELEMENTS), np.uint8)
    seqs = np.empty(len(full), np.int64)
    minus8 = np.zeros((len(full), bb.num_freq, len(antennas)), np.int64)
    out_dir.mkdir(parents=True, exist_ok=True)
    for n, i in enumerate(full):
        rec = bb.record(int(i))
        for j, a in enumerate(antennas):
            frame[:, :, N_ELEMENTS - 1 - a] = rec[:, :, j]
        minus8[n] = (((rec & 0xF) == 8) | ((rec >> 4) == 8)).sum(axis=0)
        seqs[n] = int(bb.meta["frame_fpga_seq"][i])
        write_frame(out_dir / f"{name}_{n:07d}.bin", frame, int(seqs[n]), shard)
    return seqs, minus8
