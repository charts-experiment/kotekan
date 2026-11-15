"""Reader/decoder for the CHARTS integrated intensity product (charts-intensity-product-v1..v3).

The descriptor is the JSON sidecar written by cudaChartsBeamformerCommand when
`product_descriptor_file` is configured. Frame records are native rawFileWrite records:
raw chartsMetadata (input provenance, immutable) followed by the payload. Intensity, exposure
and int8 parameter records of one frame are paired by (shard, FPGA_SEQ_NUM, T_raw).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .packet_loss import read_product

SCHEMA = "charts-intensity-product-v3"
# v1 descriptors (before 2026-11-02) have no history.upchan_gain; their fine gain is unity.
# v2: int4 fine path (beamform-64). v3: fp16 fine path, no fine_quantization block.
SUPPORTED_SCHEMAS = ("charts-intensity-product-v1", "charts-intensity-product-v2", SCHEMA)
INVALID_CODE = -128


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
    return d


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
