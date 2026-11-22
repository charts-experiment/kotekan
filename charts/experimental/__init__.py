"""Shared constants, loaders, and forward array-factor utilities for CHARTS experimental modules."""

from __future__ import annotations

import glob
import json
import os
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml
from charts_constants import CHARTS_ANTENNA_SPACING_M, C_LIGHT, DELTA_TIME, FREQ, K_DM


def _as_float(val: Any) -> float:
    """Convert numeric constants or Astropy Quantities into plain Python floats."""
    return float(getattr(val, "value", val))


FREQS_MHZ: np.ndarray = np.asarray(FREQ, dtype=np.float64)
N_CHAN_SB: int = 336
N_CHAN_TOTAL: int = int(FREQS_MHZ.size)
C_LIGHT_M_S: float = _as_float(C_LIGHT)
FPGA_DT_S: float = _as_float(DELTA_TIME)
K_DM_PC_CM3_MHZ2_S: float = _as_float(K_DM)
NOMINAL_SPACING_M: float = _as_float(CHARTS_ANTENNA_SPACING_M)


def load_antenna_positions_from_yaml(
    yaml_path: str | Path,
) -> Tuple[np.ndarray, List[int], Dict[str, Any]]:
    """Load active antenna ENU coordinates and tracker metadata from a Kotekan YAML config."""
    with open(yaml_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    stack: List[Any] = [cfg]
    while stack:
        node = stack.pop(0)
        if isinstance(node, dict):
            if "custom_antenna_positions" in node and "active_antennas" in node:
                all_pos = np.asarray(node["custom_antenna_positions"], dtype=np.float64)
                active_ids = [int(idx) for idx in node["active_antennas"]]
                pos_active = all_pos[active_ids].copy()
                meta = {
                    "antenna_order": node.get("antenna_order", "ascending"),
                    "conjugate_gains": bool(node.get("conjugate_gains", True)),
                    "num_elements": int(node.get("num_elements", all_pos.shape[0])),
                    "spacing_m": float(node.get("spacing_m", NOMINAL_SPACING_M)),
                }
                return pos_active, active_ids, meta
            for val in node.values():
                if isinstance(val, (dict, list)):
                    stack.append(val)
        elif isinstance(node, list):
            for item in node:
                if isinstance(item, (dict, list)):
                    stack.append(item)

    raise KeyError(
        f"Could not find custom_antenna_positions + active_antennas in {yaml_path}"
    )


def load_beam_power_subband(
    run_dir: str | Path,
    sb_idx: int,
    n_beams: int,
    n_freq: int = N_CHAN_SB,
    max_frames: Optional[int] = None,
    start_frame: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read rawFileWrite beam_power_freq{sb_idx}_*.bin into (T, n_beams, n_freq), seq, time0_us."""
    run_path = Path(run_dir)
    files = sorted(glob.glob(str(run_path / f"beam_power_freq{sb_idx}_*.bin")))
    if not files:
        raise FileNotFoundError(f"No beam_power_freq{sb_idx}_*.bin found in {run_path}")

    payload_floats = n_beams * n_freq
    payload_bytes = payload_floats * 4
    frames_list: List[np.ndarray] = []
    seq_list: List[int] = []
    t0_list: List[int] = []
    seen_frames = 0

    for fpath in files:
        if max_frames is not None and len(seq_list) >= max_frames:
            break
        fsize = os.path.getsize(fpath)
        if fsize < 8:
            continue
        with open(fpath, "rb") as handle:
            hdr_sz_buf = handle.read(4)
            if len(hdr_sz_buf) < 4:
                continue
            (meta_size,) = struct.unpack("<I", hdr_sz_buf)
            if meta_size < 16 or meta_size > 1_000_000:
                raise ValueError(f"Unexpected meta_size={meta_size} in {fpath}")
            rec_bytes = 4 + meta_size + payload_bytes

            if fsize % rec_bytes == 0:
                n_rec = fsize // rec_bytes
                if seen_frames + n_rec <= start_frame:
                    seen_frames += n_rec
                    continue
                local_start = max(0, start_frame - seen_frames)
                local_count = n_rec - local_start
                if max_frames is not None:
                    local_count = min(local_count, max_frames - len(seq_list))
                seen_frames += n_rec
                if local_count <= 0:
                    continue
                handle.seek(local_start * rec_bytes)
                raw = np.fromfile(handle, dtype=np.uint8, count=local_count * rec_bytes)
                n_actual = raw.size // rec_bytes
                if n_actual == 0:
                    continue
                recs = raw[: n_actual * rec_bytes].reshape(n_actual, rec_bytes)
                msizes = recs[:, :4].view("<u4")[:, 0]
                if np.all(msizes == meta_size):
                    seqs = recs[:, 12:20].view("<i8")[:, 0].copy()
                    t0s = recs[:, 4 + meta_size - 8 : 4 + meta_size].view("<u8")[:, 0].copy()
                    arr = (
                        recs[:, 4 + meta_size :]
                        .view("<f4")
                        .reshape(n_actual, n_beams, n_freq)
                        .copy()
                    )
                    frames_list.append(arr)
                    seq_list.extend(seqs.tolist())
                    t0_list.extend(t0s.tolist())
                    continue

            handle.seek(0)
            while True:
                if max_frames is not None and len(seq_list) >= max_frames:
                    break
                sz_buf = handle.read(4)
                if len(sz_buf) < 4:
                    break
                (msz,) = struct.unpack("<I", sz_buf)
                if msz < 16 or msz > 1_000_000:
                    break
                meta = handle.read(msz)
                pay = handle.read(payload_bytes)
                if len(meta) < msz or len(pay) < payload_bytes:
                    break
                if seen_frames < start_frame:
                    seen_frames += 1
                    continue
                seen_frames += 1
                seq_num = struct.unpack_from("<q", meta, 8)[0]
                t0_us = struct.unpack_from("<Q", meta, msz - 8)[0]
                arr = np.frombuffer(pay, dtype="<f4").reshape(n_beams, n_freq).copy()
                frames_list.append(arr[None, :, :])
                seq_list.append(seq_num)
                t0_list.append(t0_us)

    if not frames_list:
        raise ValueError(f"No readable frames in {run_path} for subband {sb_idx}")

    return (
        np.concatenate(frames_list, axis=0),
        np.asarray(seq_list, dtype=np.int64),
        np.asarray(t0_list, dtype=np.uint64),
    )


def load_dual_subband_beam_power(
    run_dir: str | Path,
    n_beams: int,
    max_frames: Optional[int] = None,
    start_frame: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load and seq-align subband 0 & 1 beam power waterfalls into (T, n_beams, 672)."""
    d0, seq0, t0_0 = load_beam_power_subband(
        run_dir, 0, n_beams, max_frames=max_frames, start_frame=start_frame
    )
    d1, seq1, t0_1 = load_beam_power_subband(
        run_dir, 1, n_beams, max_frames=max_frames, start_frame=start_frame
    )

    common_seq, idx0, idx1 = np.intersect1d(seq0, seq1, return_indices=True)
    if common_seq.size > 0:
        wf = np.concatenate([d0[idx0], d1[idx1]], axis=2)
        seq_out = common_seq
        t_out = t0_0[idx0]
        aligned_by_seq = True
    else:
        n_t = min(seq0.size, seq1.size)
        wf = np.concatenate([d0[:n_t], d1[:n_t]], axis=2)
        seq_out = seq0[:n_t]
        t_out = t0_0[:n_t]
        aligned_by_seq = False

    dseq0 = np.diff(seq0)
    med_dseq = int(np.median(dseq0)) if dseq0.size > 0 else 0
    dt_s = med_dseq * FPGA_DT_S if med_dseq > 0 else 0.0
    if dt_s <= 0.0 and t_out.size > 1:
        dt_s = float(np.median(np.diff(t_out.astype(np.int64)))) * 1e-6

    info = {
        "aligned_by_seq": aligned_by_seq,
        "frames_out": int(wf.shape[0]),
        "med_dseq": med_dseq,
        "dt_s": dt_s,
    }
    return wf, seq_out, t_out, info


def load_pointing_log(run_dir: str | Path) -> List[Dict[str, Any]]:
    """Load sorted beam_pointing_log.jsonl entries if present."""
    log_path = Path(run_dir) / "beam_pointing_log.jsonl"
    if not log_path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with open(log_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda r: float(r["unix"]))
    return rows


def load_rfi_snapshots(run_dir: str | Path) -> List[Dict[str, Any]]:
    """Load rfi_channel_snapshots.jsonl entries if present."""
    snap_path = Path(run_dir) / "rfi_channel_snapshots.jsonl"
    if not snap_path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with open(snap_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def array_factor_power(
    positions_m: np.ndarray,
    freqs_mhz: Sequence[float] | np.ndarray,
    l_dir: float | np.ndarray,
    m_dir: float | np.ndarray,
    l0: float | np.ndarray,
    m0: float | np.ndarray,
    antenna_delays_s: Optional[np.ndarray] = None,
    conjugate_sign: float = 1.0,
) -> np.ndarray:
    """Evaluate coherent phased-array normalized power response |AF|^2 <= 1.

    Supports scalar or broadcasted (l_dir, m_dir) and (l0, m0) arrays with trailing
    frequency axis (..., F). Uses 3D direction cosines (l, m, n = sqrt(max(0, 1 - l^2 - m^2))).
    """
    pos = np.asarray(positions_m, dtype=np.float64)
    freqs_hz = np.asarray(freqs_mhz, dtype=np.float64) * 1.0e6
    l_d = np.asarray(l_dir, dtype=np.float64)
    m_d = np.asarray(m_dir, dtype=np.float64)
    l_0 = np.asarray(l0, dtype=np.float64)
    m_0 = np.asarray(m0, dtype=np.float64)

    n_d = np.sqrt(np.maximum(0.0, 1.0 - l_d * l_d - m_d * m_d))
    n_0 = np.sqrt(np.maximum(0.0, 1.0 - l_0 * l_0 - m_0 * m_0))

    du_l = l_d - conjugate_sign * l_0
    du_m = m_d - conjugate_sign * m_0
    du_n = n_d - conjugate_sign * n_0

    # Dot with antenna coordinates -> shape (..., N_ant)
    tau_geom = (
        du_l[..., None] * pos[:, 0]
        + du_m[..., None] * pos[:, 1]
        + du_n[..., None] * pos[:, 2]
    ) / C_LIGHT_M_S

    if antenna_delays_s is not None:
        tau_tot = tau_geom + np.asarray(antenna_delays_s, dtype=np.float64)
    else:
        tau_tot = tau_geom

    phase = (2.0 * np.pi) * tau_tot[..., None] * freqs_hz
    af_complex = np.mean(np.exp(1j * phase), axis=-2)
    return (af_complex.real * af_complex.real + af_complex.imag * af_complex.imag).astype(
        np.float64
    )
