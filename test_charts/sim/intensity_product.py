"""Reader/decoder for the CHARTS integrated intensity product (charts-intensity-product-v1..v3).

The descriptor is the JSON sidecar written by cudaChartsBeamformerCommand when
`product_descriptor_file` is configured. Frame records are native rawFileWrite records:
raw chartsMetadata (input provenance, immutable) followed by the payload. Intensity, exposure
and int8 parameter records of one frame are paired by (shard, FPGA_SEQ_NUM, T_raw).

Beam axis: since 2026-11-24 the int8 product may hold only selected 16-beam groups
(`output_beam_groups`); `beams.written` maps output beam j to weight-file beam written[j].
Label or select beams with beam_indices()/beam_positions(), never by position alone.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .packet_loss import read_product

SCHEMA = "charts-intensity-product-v3"
# v1 descriptors (before 2026-11-02) have no history.upchan_gain; their fine gain is unity.
# v2: int4 fine path (beamform-64). v3: fp16 fine path, no fine_quantization block; fp16
# device weights (weights.device), beamformer.{kernel, arithmetic} and
# history.upchan_gain.normalization are required.
SUPPORTED_SCHEMAS = ("charts-intensity-product-v1", "charts-intensity-product-v2", SCHEMA)
BEAMFORMER_KERNELS = ("tensor_core", "scalar_reference")
INVALID_CODE = -128
BEAM_GROUP_SIZE = 16


class ProductError(ValueError):
    pass


def load_descriptor(path: Path) -> dict:
    d = json.loads(Path(path).read_text())
    if d.get("schema") not in SUPPORTED_SCHEMAS:
        raise ProductError(f"unsupported schema {d.get('schema')!r}, expected one of "
                           f"{SUPPORTED_SCHEMAS}")
    for key in ("shard", "T_raw", "K", "U", "intensity", "frequency", "association"):
        if key not in d:
            raise ProductError(f"descriptor lacks {key!r}")
    if d["schema"] == SCHEMA:
        history = d.get("history") or {}
        if "fine_quantization" in history:
            raise ProductError("v3 descriptor must not carry history.fine_quantization")
        if history.get("fine_format") != "fp16":
            raise ProductError("v3 descriptor must declare history.fine_format = 'fp16'")
        if "normalization" not in (history.get("upchan_gain") or {}):
            raise ProductError("v3 descriptor must declare history.upchan_gain.normalization")
        device = (d.get("weights") or {}).get("device") or {}
        if device.get("precision") != "fp16" or not device.get("effective_sha256"):
            raise ProductError("v3 descriptor must describe the fp16 device weights "
                               "(weights.device.precision, effective_sha256)")
        if (d.get("beamformer") or {}).get("kernel") not in BEAMFORMER_KERNELS:
            raise ProductError(f"v3 descriptor must name beamformer.kernel, one of "
                               f"{BEAMFORMER_KERNELS}")
    if "beams" in d:
        _check_beams(d)
    elif d["schema"] == SCHEMA and d["intensity"]["shape"][2] != 128:
        # v3 without the block predates beam selection and always wrote B = 128.
        raise ProductError(f"v3 descriptor without a beams block has intensity.shape[2] = "
                           f"{d['intensity']['shape'][2]}, expected 128")
    return d


def _check_beams(d: dict) -> None:
    """The beams block (v3, added 2026-11-24): whole groups of 16 of the computed beams."""
    b = d["beams"]
    if not isinstance(b, dict):
        raise ProductError("beams must be an object")
    for key in ("computed", "group_size", "groups", "written"):
        if key not in b:
            raise ProductError(f"beams lacks {key!r}")
    size, computed = b["group_size"], b["computed"]
    if size != BEAM_GROUP_SIZE or computed <= 0 or computed % size:
        raise ProductError(f"beams: group_size {size} / computed {computed} invalid")
    groups = list(b["groups"])
    if not groups or any(not isinstance(g, int) or isinstance(g, bool)
                         or g < 0 or g >= computed // size
                         for g in groups) or any(a >= c for a, c in zip(groups, groups[1:])):
        raise ProductError(f"beams.groups {groups} must be strictly increasing in "
                           f"0..{computed // size - 1}")
    expected = [g * size + j for g in groups for j in range(size)]
    if list(b["written"]) != expected:
        raise ProductError("beams.written is inconsistent with beams.groups")
    if d["intensity"]["shape"][2] != len(expected):
        raise ProductError(f"intensity.shape[2] = {d['intensity']['shape'][2]} but "
                           f"{len(expected)} beams are written")
    params = (d.get("int8") or {}).get("parameters")
    if params and params.get("beam_tiles") != len(groups):
        raise ProductError(f"int8.parameters.beam_tiles = {params.get('beam_tiles')} but "
                           f"{len(groups)} beam groups are written")


def beam_indices(d: dict) -> np.ndarray:
    """Weight-file beam index of every output beam (output beam j is weight beam result[j]).
    Descriptors without a beams block were written with all beams, in order."""
    if "beams" in d:
        return np.asarray(d["beams"]["written"], dtype=np.int64)
    return np.arange(d["intensity"]["shape"][2], dtype=np.int64)


def beam_positions(d: dict, beams) -> np.ndarray:
    """Output position of each requested weight-file beam; ProductError if one was not
    written."""
    where = {int(b): j for j, b in enumerate(beam_indices(d))}
    missing = [int(b) for b in beams if int(b) not in where]
    if missing:
        raise ProductError(f"beams {missing} were not written (beams.groups = "
                           f"{d.get('beams', {}).get('groups')})")
    return np.asarray([where[int(b)] for b in beams], dtype=np.int64)


def frequencies_hz(d: dict) -> np.ndarray:
    f = d["frequency"]
    local = np.arange(336)
    if d["U"] == 1:
        offs = local.astype(float)
    else:
        u = np.arange(d["U"])
        offs = (local[:, None] + (u[None, :] - (d["U"] - 1) / 2) / d["U"]).reshape(-1)
    return f["nu0_hz"] + (336 * d["shard"] + offs) * f["dnu_hz"]


def upchan_gain(d: dict) -> np.ndarray | None:
    """Applied fine-channel gain G as float32 [336*32] (index f*32 + u), or None in coarse
    mode. Intensity scales as G^2; divide by it to refer power to the ungained upchannelizer."""
    if d["U"] == 1:
        return None
    g = d.get("history", {}).get("upchan_gain")
    if g is None:
        # Only v1 (before the gain existed) implies unity; v2 must state it.
        if d.get("schema") != "charts-intensity-product-v1":
            raise ProductError("fine-mode descriptor lacks history.upchan_gain")
        return np.ones(336 * d["U"], np.float32)
    values = np.asarray(g["values"], np.float32)
    if values.shape != (d["U"],):
        raise ProductError(f"history.upchan_gain.values has shape {values.shape}")
    return np.tile(values, 336)


def record_key(meta) -> tuple:
    """(shard or None, FPGA_SEQ_NUM) from the raw metadata. The raw metadata has no shard
    field; it is inferred from coarse_freq[0] = 336*shard when the producer filled it."""
    shard = int(meta.coarse_freq[0]) // 336 if meta.nfreq > 0 else None
    return shard, int(meta.fpga_seq_num)


def read_intensity(d: dict, path: Path):
    """Returns ((shard, fpga_seq_num), payload) with the descriptor's dtype and shape.
    The payload length check against the descriptor shape also fixes T_raw and K."""
    meta, data = read_product(path, d["intensity"]["dtype"], tuple(d["intensity"]["shape"]))
    return record_key(meta), data


def read_quant_params(d: dict, path: Path):
    p = d["int8"]["parameters"] if d.get("int8") else None
    if p is None:
        raise ProductError("int8 parameters were not exported; codes are not reconstructable")
    meta, data = read_product(path, "<f4", (p["count"], 2))
    return record_key(meta), data


def chunk_index(d: dict) -> np.ndarray:
    """[T][F][B] -> parameter index, from the descriptor's chunk and index rule."""
    ct, cf, cb = d["int8"]["chunk"]
    t, f, b = d["intensity"]["shape"]
    p = d["int8"]["parameters"]
    ti = (np.arange(t) // ct)[:, None, None]
    fi = (np.arange(f) // cf)[None, :, None]
    bi = (np.arange(b) // cb)[None, None, :]
    return (ti * p["frequency_tiles"] + fi) * p["beam_tiles"] + bi


def decode_int8(d: dict, codes: tuple[int, np.ndarray], params: tuple[int, np.ndarray],
                *, shard: int | None = None) -> np.ndarray:
    """Reconstruct float64 intensity I = offset + q*scale; invalid chunks become NaN.

    codes/params are ((shard, fpga_seq_num), array) from read_intensity/read_quant_params.
    Records of different frames or shards, or a descriptor of another shard, are rejected;
    records are never paired by index. Decode in float64.
    """
    if d["intensity"]["dtype"] != "int8":
        raise ProductError("intensity is not int8")
    if shard is not None and shard != d["shard"]:
        raise ProductError(f"descriptor is for shard {d['shard']}, not {shard}")
    (key_q, q), (key_p, table) = codes, params
    if key_q != key_p:
        raise ProductError(f"codes (shard, FPGA_SEQ_NUM) {key_q} != parameters {key_p}")
    if key_q[0] is not None and key_q[0] != d["shard"]:
        raise ProductError(f"records are from shard {key_q[0]}, descriptor is shard {d['shard']}")
    idx = chunk_index(d)
    offset = table[:, 0].astype(np.float64)[idx]
    scale = table[:, 1].astype(np.float64)[idx]
    out = offset + q.astype(np.float64) * scale
    invalid_chunks = np.unique(idx[q == INVALID_CODE])
    out[np.isin(idx, invalid_chunks)] = np.nan
    return out
