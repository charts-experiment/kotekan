#!/usr/bin/env python3
"""
Posterior Benchmarking & Physics Analysis for Direct Beam Tracker Deployment
Target: Grus Quartet / NGC 7582 (RA = 349.5979 deg, Dec = -42.3706 deg)
Covers Section 7 (Kotekan & GPU Kernel Benchmarking) and Section 8 (Baseband,
Masking Verification, Astrometry, Formed-Beam Stability, Coherent Gain eta,
and CPU Golden-Reference Delay-and-Sum Replay on dump_A and dump_B).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from astropy.coordinates import AltAz, EarthLocation, SkyCoord
    from astropy.time import Time
    import astropy.units as u
    HAS_ASTROPY = True
except ImportError:
    HAS_ASTROPY = False

# Constants matching chartsConstants.hpp & YAML config
SAMPLES_PER_FRAME = 15360
SAMPLE_PERIOD_S = 10.0 / 3.0e6  # 3.333333333333e-6 s
FRAME_DURATION_MS = SAMPLES_PER_FRAME * SAMPLE_PERIOD_S * 1e3  # 51.2 ms
NUM_LOCAL_FREQ = 336
NUM_ELEMENTS = 64
MAX_ALIVE_ANTENNAS = 7
SPACING_M = 0.6
SPEED_OF_LIGHT = 299792458.0

CAREN_LAT_DEG = -33.4211146
CAREN_LON_DEG = -70.8634710
CAREN_ALT_M = 458.0

GRUS_RA_DEG = 349.5979
GRUS_DEC_DEG = -42.3706

# Kotekan DataType enum values (lib/metadata/DataType.hpp)
KOTEKAN_DTYPE_INT4X2 = 7
KOTEKAN_DTYPE_INT4X2_SWAPPED_WITHOFFSET = 8


@dataclass
class FrameHeader:
    frame_idx_in_file: int
    metadata_size: int
    data_offset: int
    frame_counter: int
    fpga_seq_num: int
    data_type: int
    lost_timesamples: int
    recv_time_sec: float
    time0_fpga: int


def parse_multi_frame_bin(
    file_path: Path, frame_payload_bytes: int
) -> List[Tuple[FrameHeader, np.memmap]]:
    """Parse a rawFileWrite .bin file containing 1 or more concatenated frames."""
    if not file_path.exists():
        return []
    file_size = file_path.stat().st_size
    offset = 0
    frames = []
    idx = 0
    with file_path.open("rb") as f:
        while offset + 4 <= file_size:
            f.seek(offset)
            meta_sz_bytes = f.read(4)
            if len(meta_sz_bytes) < 4:
                break
            meta_sz = struct.unpack("<I", meta_sz_bytes)[0]
            meta_start = offset + 4
            data_start = meta_start + meta_sz
            if data_start + frame_payload_bytes > file_size:
                break
            meta_buf = f.read(meta_sz) if meta_sz > 0 else b""
            frame_counter = -1
            fpga_seq = -1
            dtype_val = KOTEKAN_DTYPE_INT4X2
            lost_samples = 0
            recv_sec = 0.0
            time0_fpga = 0
            if len(meta_buf) >= 16:
                frame_counter = struct.unpack_from("<i", meta_buf, 0)[0]
                fpga_seq = struct.unpack_from("<q", meta_buf, 8)[0]
            if len(meta_buf) >= 44:
                dtype_val = struct.unpack_from("<i", meta_buf, 40)[0]
            if len(meta_buf) >= 3032:
                # chartsMetadataFormat tail fields
                lost_samples = struct.unpack_from("<i", meta_buf, 3000)[0]
                tv_sec, tv_usec = struct.unpack_from("<qq", meta_buf, 3008)
                if tv_sec > 0:
                    recv_sec = float(tv_sec) + float(tv_usec) * 1e-6
                time0_fpga = struct.unpack_from("<Q", meta_buf, 3024)[0]
                if recv_sec <= 0.0 and time0_fpga > 1000000000000:
                    # rfsocHandlerShuffle stores UTC timestamp in microseconds in time0_fpga
                    recv_sec = float(time0_fpga) * 1.0e-6

            hdr = FrameHeader(
                frame_idx_in_file=idx,
                metadata_size=meta_sz,
                data_offset=data_start,
                frame_counter=frame_counter,
                fpga_seq_num=fpga_seq,
                data_type=dtype_val,
                lost_timesamples=max(0, lost_samples),
                recv_time_sec=recv_sec,
                time0_fpga=time0_fpga,
            )
            mm = np.memmap(
                file_path,
                dtype=np.uint8,
                mode="r",
                offset=data_start,
                shape=(frame_payload_bytes,),
            )
            frames.append((hdr, mm))
            offset = data_start + frame_payload_bytes
            idx += 1
    return frames


def decode_int4x2(raw_u8: np.ndarray, swapped_with_offset: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decode 4+4 bit complex byte array into signed int8 (re, im) in [-8, 7]
    matching decode_int4x2_fast<SWAPPED_WITH_OFFSET> in cudaAntennaMask.cu / cudaDirectBeamTracker.cu:
      - If swapped_with_offset is False (int4x2, live RFSoC rfsocHandlerShuffle):
          low nibble = signed two's complement real
          high nibble = signed two's complement imag
      - If swapped_with_offset is True (int4x2_swapped_withoffset):
          high nibble - 8 = real
          low nibble - 8 = imag
    """
    b = raw_u8.astype(np.int16)
    if swapped_with_offset:
        re = ((b >> 4) & 0x0F) - 8
        im = (b & 0x0F) - 8
    else:
        re = ((b & 0x0F) ^ 0x08) - 8
        im = (((b >> 4) & 0x0F) ^ 0x08) - 8
    return re.astype(np.int8), im.astype(np.int8)


def compute_taper_weights(antenna_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Exact replica of DirectBeamTrackerConfig::recompute_normalized_taper()
    for 64 antennas, descending order, spacing = 0.6 m, cosine_pedestal (p=0.25).
    Returns:
      positions: (64, 3) float32 array of [x, y, z] in meters
      norm_taper: (64,) float32 array where sum(norm_taper^2) == 1.0 over active antennas
    """
    positions = np.zeros((NUM_ELEMENTS, 3), dtype=np.float32)
    for r in range(NUM_ELEMENTS):
        phys_ant = (NUM_ELEMENTS - 1) - r
        col = phys_ant % 8
        row = phys_ant // 8
        positions[r, 0] = float(col) * SPACING_M
        positions[r, 1] = float(row) * SPACING_M
        positions[r, 2] = 0.0

    min_x, max_x = float(np.min(positions[:, 0])), float(np.max(positions[:, 0]))
    min_y, max_y = float(np.min(positions[:, 1])), float(np.max(positions[:, 1]))
    cx = 0.5 * (min_x + max_x)
    cy = 0.5 * (min_y + max_y)
    rx = max(1e-6, 0.5 * (max_x - min_x))
    ry = max(1e-6, 0.5 * (max_y - min_y))

    p = 0.25
    taper = np.zeros(NUM_ELEMENTS, dtype=np.float32)
    sum_sq = 0.0
    for r in range(NUM_ELEMENTS):
        if antenna_mask[r] == 0:
            continue
        ux = min(1.0, abs((positions[r, 0] - cx) / rx))
        uy = min(1.0, abs((positions[r, 1] - cy) / ry))
        wx = p + (1.0 - p) * 0.5 * (1.0 + math.cos(math.pi * ux))
        wy = p + (1.0 - p) * 0.5 * (1.0 + math.cos(math.pi * uy))
        ta = float(wx * wy)
        taper[r] = ta
        sum_sq += ta * ta

    if sum_sq > 0.0:
        inv_norm = float(1.0 / math.sqrt(sum_sq))
        taper *= inv_norm
    return positions, taper


def compute_cpp_astrometry(utc_timestamp: float, ra_deg: float, dec_deg: float) -> Tuple[float, float, float, float]:
    """
    Exact replica of cudaDirectBeamTrackerCommand::update_weights_if_needed()
    GMST/LST and compute_celestial_direction() in lib/cuda/cudaDirectBeamTrackerCommand.cpp.
    Returns (lst_hours, l, m, n) as float32-rounded direction cosines.
    """
    d_start = utc_timestamp / 86400.0 - 10957.5
    gmst_hours = math.fmod(18.697374558 + 24.06570982441908 * d_start, 24.0)
    if gmst_hours < 0.0:
        gmst_hours += 24.0
    lst_hours = math.fmod(gmst_hours + CAREN_LON_DEG / 15.0, 24.0)
    if lst_hours < 0.0:
        lst_hours += 24.0

    ha_rad = lst_hours * (math.pi / 12.0) - ra_deg * (math.pi / 180.0)
    dec_rad = dec_deg * (math.pi / 180.0)
    lat_rad = CAREN_LAT_DEG * (math.pi / 180.0)

    l = -math.cos(dec_rad) * math.sin(ha_rad)
    m = math.cos(lat_rad) * math.sin(dec_rad) - math.sin(lat_rad) * math.cos(dec_rad) * math.cos(ha_rad)
    trans_sq = l * l + m * m
    n = math.sqrt(1.0 - trans_sq) if trans_sq <= 1.0 else 0.0
    return lst_hours, float(np.float32(l)), float(np.float32(m)), float(np.float32(n))


def parse_prometheus_and_logs(dump_dir: Path) -> Dict[str, Any]:
    """Parse metrics_*.prom, buffers_*.json, gpu_dmon.log, and run.log from a run directory."""
    res: Dict[str, Any] = {
        "cuda_exec_ms": {},
        "rawwrite_ms": {},
        "dpdk_dropped": 0,
        "peak_full_frames": {},
        "dmon_sm": [],
        "dmon_mem": [],
        "dmon_pwr": [],
        "dmon_rxpci": [],
        "dmon_txpci": [],
        "fatal_errors": 0,
        "clean_exit": False,
        "tracker_status_records": [],
        "mask_status_records": [],
    }

    for prom_file in sorted(dump_dir.glob("metrics_*.prom")):
        text = prom_file.read_text(errors="ignore")
        for line in text.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            if "kotekan_cuda_execute_time_seconds" in line:
                m = re.search(r'\{([^}]+)\}\s+([0-9.eE+-]+)', line)
                if m:
                    labels, val = m.group(1), float(m.group(2)) * 1e3
                    res["cuda_exec_ms"].setdefault(labels, []).append(val)
            elif "kotekan_rawfilewrite_write_time_seconds" in line:
                m = re.search(r'\{([^}]+)\}\s+([0-9.eE+-]+)', line)
                if m:
                    labels, val = m.group(1), float(m.group(2)) * 1e3
                    if val > 0:
                        res["rawwrite_ms"].setdefault(labels, []).append(val)
            elif "dropped" in line or "lost_packets" in line or "rx_missed" in line:
                m = re.search(r'\s+([0-9.eE+-]+)$', line)
                if m:
                    res["dpdk_dropped"] = max(res["dpdk_dropped"], float(m.group(1)))

    for buf_file in sorted(dump_dir.glob("buffers_*.json")):
        try:
            data = json.loads(buf_file.read_text(errors="ignore"))
            if isinstance(data, dict):
                for bname, binfo in data.items():
                    if isinstance(binfo, dict):
                        nf = binfo.get("num_full_frame", binfo.get("num_full_frames", None))
                        if nf is not None:
                            res["peak_full_frames"][bname] = max(
                                res["peak_full_frames"].get(bname, 0),
                                int(nf),
                            )
                for s_idx in [0, 1]:
                    in_b = data.get(f"network_capture_buf_{s_idx}", {})
                    out_b = data.get(f"host_formed_beams_buffer_{s_idx}", {})
                    t_in = float(in_b.get("last_frame_arrival_time", 0.0))
                    t_out = float(out_b.get("last_frame_arrival_time", 0.0))
                    if t_in > 0 and t_out >= t_in and (t_out - t_in) < 0.0512:
                        res["cuda_exec_ms"].setdefault(f"GPU{s_idx}:TotalFramePipeline", []).append((t_out - t_in) * 1e3)
                if "host_alive_voltages_buffer_0" in data:
                    in_b = data.get("network_capture_buf_0", {})
                    al_b = data.get("host_alive_voltages_buffer_0", {})
                    t_in = float(in_b.get("last_frame_arrival_time", 0.0))
                    t_al = float(al_b.get("last_frame_arrival_time", 0.0))
                    if t_in > 0 and t_al >= t_in and (t_al - t_in) < 0.0512:
                        res["cuda_exec_ms"].setdefault("GPU0:Mask+AliveExtract", []).append((t_al - t_in) * 1e3)
        except Exception:
            pass

    dmon_path = dump_dir / "gpu_dmon.log"
    if dmon_path.exists():
        for line in dmon_path.read_text(errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 10 and parts[0].isdigit():
                try:
                    # gpu pwr gtemp mtemp sm mem enc dec jpg ofa mclk pclk rxpci txpci
                    pwr = float(parts[1])
                    sm = float(parts[4])
                    mem = float(parts[5])
                    rxpci = float(parts[-2])
                    txpci = float(parts[-1])
                    res["dmon_pwr"].append(pwr)
                    res["dmon_sm"].append(sm)
                    res["dmon_mem"].append(mem)
                    res["dmon_rxpci"].append(rxpci)
                    res["dmon_txpci"].append(txpci)
                except ValueError:
                    pass

    run_log = dump_dir / "run.log"
    if run_log.exists():
        log_txt = run_log.read_text(errors="ignore")
        res["fatal_errors"] = len(re.findall(r"FATAL_ERROR", log_txt))
        res["clean_exit"] = ("Exiting" in log_txt) or ("Data file write done" in log_txt and res["fatal_errors"] == 0)
        for line in log_txt.splitlines():
            if "cudaDirectBeamTrackerCommand" in line or "cudaAntennaMaskCommand" in line or "cudaInputData" in line or "cudaOutputData" in line:
                m = re.search(r'(gpu_\d+:[\w]+).*?([0-9]+\.[0-9]+)\s*s', line)
                if m:
                    res["cuda_exec_ms"].setdefault(m.group(1), []).append(float(m.group(2)) * 1e3)

    def parse_concat_json(p: Path) -> List[Dict[str, Any]]:
        objs: List[Dict[str, Any]] = []
        if not p.exists():
            return objs
        txt = p.read_text(errors="ignore")
        dec = json.JSONDecoder()
        pos = 0
        n = len(txt)
        while pos < n:
            m = re.search(r'\{', txt[pos:])
            if not m:
                break
            start = pos + m.start()
            try:
                obj, end = dec.raw_decode(txt, start)
                if isinstance(obj, dict):
                    objs.append(obj)
                pos = end
            except Exception:
                pos = start + 1
        return objs

    res["tracker_status_records"] = parse_concat_json(dump_dir / "direct_tracker_status.jsonl")
    res["mask_status_records"] = parse_concat_json(dump_dir / "antenna_mask_status.jsonl")

    return res


def run_cpu_golden_replay(
    raw_u8_frame: np.ndarray,
    beam_gpu_frame: np.ndarray,
    active_raw_elems: List[int],
    full_64_mask: np.ndarray,
    utc_ts: float,
    freq_start_hz: float,
    swapped_with_offset: bool = False,
    n_test_t: int = 256,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Exact replica of generate_steering_weights_tapered_kernel + direct_beamformer_blackwell_kernel.
    Works for BOTH:
      - 64-element raw baseband (when raw_u8_frame has shape (T, F, 64) and active_raw_elems == range(64))
      - 7-element compacted alive baseband (when raw_u8_frame has shape (T, F, 7) and active_raw_elems has length <= 7)
    """
    _, l0, m0, n0 = compute_cpp_astrometry(utc_ts, GRUS_RA_DEG, GRUS_DEC_DEG)
    _, l1, m1, n1 = compute_cpp_astrometry(utc_ts + SAMPLES_PER_FRAME * SAMPLE_PERIOD_S, GRUS_RA_DEG, GRUS_DEC_DEG)

    pos_all, norm_taper_all = compute_taper_weights(full_64_mask)
    freqs_hz = freq_start_hz + np.arange(NUM_LOCAL_FREQ, dtype=np.float64) * 300.0e3
    wavenums = 2.0 * math.pi * freqs_hz / SPEED_OF_LIGHT  # (336,)

    n_ch = len(active_raw_elems)
    pos_sel = pos_all[active_raw_elems]  # (n_ch, 3)
    taper_sel = norm_taper_all[active_raw_elems]  # (n_ch,)

    delay_m0 = pos_sel[:, 0].astype(np.float64) * l0 + pos_sel[:, 1].astype(np.float64) * m0 + pos_sel[:, 2].astype(np.float64) * n0
    delay_m1 = pos_sel[:, 0].astype(np.float64) * l1 + pos_sel[:, 1].astype(np.float64) * m1 + pos_sel[:, 2].astype(np.float64) * n1

    phase0 = np.outer(wavenums, delay_m0).astype(np.float32)  # (336, n_ch)
    dphase = (np.outer(wavenums, delay_m1 - delay_m0) / float(SAMPLES_PER_FRAME)).astype(np.float32)  # (336, n_ch)

    raw_slice = raw_u8_frame[:n_test_t, :, :n_ch]
    re_s, im_s = decode_int4x2(raw_slice, swapped_with_offset=swapped_with_offset)
    v_c = re_s.astype(np.float32) + 1j * im_s.astype(np.float32)  # (n_test_t, 336, n_ch)

    # Match step_weights iterative multiplication in direct_beamformer_blackwell_kernel
    w0 = taper_sel[None, :] * (np.cos(phase0) + 1j * np.sin(phase0))  # (336, n_ch) complex64
    dw = (np.cos(dphase) + 1j * np.sin(dphase)).astype(np.complex64)  # (336, n_ch) complex64

    b_cpu = np.zeros((n_test_t, NUM_LOCAL_FREQ), dtype=np.complex64)
    w_cur = w0.astype(np.complex64).copy()
    for t in range(n_test_t):
        b_cpu[t] = np.sum(w_cur * v_c[t], axis=1)
        w_cur = w_cur * dw

    b_gpu = beam_gpu_frame[:n_test_t]
    diff = np.abs(b_gpu - b_cpu)
    max_abs_err = float(np.max(diff))
    rel_rms_err = float(np.sqrt(np.mean(diff ** 2)) / (np.sqrt(np.mean(np.abs(b_gpu) ** 2)) + 1e-12))
    return b_gpu, b_cpu, max_abs_err, rel_rms_err


def main():
    parser = argparse.ArgumentParser(description="Analyze Tonight's Grus Quartet Runs A & B")
    parser.add_argument("--dump-a", type=Path, default=Path("/data/tracker/dump_A"))
    parser.add_argument("--dump-b", type=Path, default=Path("/data/tracker/dump_B"))
    parser.add_argument("--out-dir", type=Path, default=Path("/data/tracker/analysis_tonight"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("================================================================================")
    print(" POSTERIOR BENCHMARK & PHYSICS ANALYSIS: GRUS QUARTET (NGC 7582)")
    print("================================================================================")

    bench_a = parse_prometheus_and_logs(args.dump_a)
    bench_b = parse_prometheus_and_logs(args.dump_b)

    # -------------------------------------------------------------------------
    # Load Binary Files from Run A and Run B
    # -------------------------------------------------------------------------
    raw_a_bytes = SAMPLES_PER_FRAME * NUM_LOCAL_FREQ * NUM_ELEMENTS  # 330,301,440
    alive_b_bytes = SAMPLES_PER_FRAME * NUM_LOCAL_FREQ * MAX_ALIVE_ANTENNAS  # 36,126,720
    beam_bytes = SAMPLES_PER_FRAME * NUM_LOCAL_FREQ * 1 * 8  # 41,287,680

    raw_a_f0 = parse_multi_frame_bin(args.dump_a / "raw_baseband_64ant_freq0_0000000.bin", raw_a_bytes)
    beams_a_f0 = parse_multi_frame_bin(args.dump_a / "formed_beams_64ant_freq0_0000000.bin", beam_bytes)
    beams_a_f1 = parse_multi_frame_bin(args.dump_a / "formed_beams_64ant_freq1_0000000.bin", beam_bytes)

    alive_b_f0 = parse_multi_frame_bin(args.dump_b / "alive_baseband_64ant_freq0_0000000.bin", alive_b_bytes)
    alive_b_f1 = parse_multi_frame_bin(args.dump_b / "alive_baseband_64ant_freq1_0000000.bin", alive_b_bytes)
    beams_b_f0 = parse_multi_frame_bin(args.dump_b / "formed_beams_64ant_freq0_0000000.bin", beam_bytes)
    beams_b_f1 = parse_multi_frame_bin(args.dump_b / "formed_beams_64ant_freq1_0000000.bin", beam_bytes)

    print(f"Run A loaded: raw_f0={len(raw_a_f0)} frames, beams_f0={len(beams_a_f0)} frames, beams_f1={len(beams_a_f1)} frames")
    print(f"Run B loaded: alive_f0={len(alive_b_f0)} frames, alive_f1={len(alive_b_f1)} frames, beams_f0={len(beams_b_f0)} frames, beams_f1={len(beams_b_f1)} frames")

    # -------------------------------------------------------------------------
    # Section 7 Plot: Kotekan & GPU Kernel Benchmarks (fig0_gpu_pipeline_benchmarks.png)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    # Panel 1: CUDA Command Execution Times vs 51.2 ms Real-Time Budget
    ax = axes[0, 0]
    stage_names = []
    stage_vals_a = []
    stage_vals_b = []
    all_keys = sorted(set(list(bench_a["cuda_exec_ms"].keys()) + list(bench_b["cuda_exec_ms"].keys())))
    if all_keys:
        for k in all_keys:
            short_k = re.sub(r'.*gpu_(\d+):cuda(\w+)Command.*', r'GPU\1:\2', k)
            if short_k == k:
                short_k = k[-28:]
            stage_names.append(short_k)
            stage_vals_a.append(float(np.mean(bench_a["cuda_exec_ms"].get(k, [0.0]))))
            stage_vals_b.append(float(np.mean(bench_b["cuda_exec_ms"].get(k, [0.0]))))
        x_pos = np.arange(len(stage_names))
        ax.bar(x_pos - 0.18, stage_vals_a, width=0.35, label="Run A (No Mask)", color="#3498db")
        ax.bar(x_pos + 0.18, stage_vals_b, width=0.35, label="Run B (Mask ON)", color="#2ecc71")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(stage_names, rotation=25, ha="right", fontsize=8)
    ax.axhline(FRAME_DURATION_MS, color="red", linestyle="--", label=f"Real-Time Frame Budget ({FRAME_DURATION_MS:.1f} ms)")
    ax.set_ylabel("Mean Execution Time per Frame (ms)")
    ax.set_title("Section 7: CUDA Kernel & DMA Execution Time vs. 51.2 ms Budget")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: NVMe Disk Write Latency (rawFileWrite)
    ax = axes[0, 1]
    write_keys = sorted(set(list(bench_a["rawwrite_ms"].keys()) + list(bench_b["rawwrite_ms"].keys())))
    if write_keys:
        w_names = [k[-28:] for k in write_keys]
        w_a = [float(np.mean(bench_a["rawwrite_ms"].get(k, [0.0]))) for k in write_keys]
        w_b = [float(np.mean(bench_b["rawwrite_ms"].get(k, [0.0]))) for k in write_keys]
        x_w = np.arange(len(w_names))
        ax.bar(x_w - 0.18, w_a, width=0.35, label="Run A", color="#e67e22")
        ax.bar(x_w + 0.18, w_b, width=0.35, label="Run B", color="#9b59b6")
        ax.set_xticks(x_w)
        ax.set_xticklabels(w_names, rotation=25, ha="right", fontsize=8)
    ax.axhline(FRAME_DURATION_MS, color="red", linestyle="--", label="51.2 ms Frame Cadence")
    ax.set_ylabel("Write Time per Frame (ms)")
    ax.set_title("Section 7: NVMe Disk Write Latency (330.3 MB Raw vs. 36.1 MB Alive & 41.3 MB Beams)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Peak Ring Buffer Occupancy (/buffers)
    ax = axes[1, 0]
    buf_keys = sorted(set(list(bench_a["peak_full_frames"].keys()) + list(bench_b["peak_full_frames"].keys())))
    if buf_keys:
        b_a = [bench_a["peak_full_frames"].get(k, 0) for k in buf_keys]
        b_b = [bench_b["peak_full_frames"].get(k, 0) for k in buf_keys]
        x_b = np.arange(len(buf_keys))
        ax.bar(x_b - 0.18, b_a, width=0.35, label="Run A Peak Full Frames", color="#3498db")
        ax.bar(x_b + 0.18, b_b, width=0.35, label="Run B Peak Full Frames", color="#2ecc71")
        ax.set_xticks(x_b)
        ax.set_xticklabels(buf_keys, rotation=25, ha="right", fontsize=8)
    ax.axhline(20, color="red", linestyle="--", label="Capture Ring Buffer Depth (20 frames)")
    ax.set_ylabel("Peak Full Frames")
    ax.set_title("Section 7: Ring Buffer Peak Occupancy (Zero-Drop Headroom)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 4: GPU Utilization & PCIe Throughput (nvidia-smi dmon)
    ax = axes[1, 1]
    metrics_labels = ["Peak SM (%)", "Peak Mem (%)", "Peak Power (W)", "Peak RX PCIe (GB/s)"]
    vals_dmon_a = [
        max(bench_a["dmon_sm"]) if bench_a["dmon_sm"] else 0.0,
        max(bench_a["dmon_mem"]) if bench_a["dmon_mem"] else 0.0,
        max(bench_a["dmon_pwr"]) if bench_a["dmon_pwr"] else 0.0,
        (max(bench_a["dmon_rxpci"]) / 1024.0) if bench_a["dmon_rxpci"] else 0.0,
    ]
    vals_dmon_b = [
        max(bench_b["dmon_sm"]) if bench_b["dmon_sm"] else 0.0,
        max(bench_b["dmon_mem"]) if bench_b["dmon_mem"] else 0.0,
        max(bench_b["dmon_pwr"]) if bench_b["dmon_pwr"] else 0.0,
        (max(bench_b["dmon_rxpci"]) / 1024.0) if bench_b["dmon_rxpci"] else 0.0,
    ]
    x_d = np.arange(len(metrics_labels))
    ax.bar(x_d - 0.18, vals_dmon_a, width=0.35, label="Run A (No Mask)", color="#3498db")
    ax.bar(x_d + 0.18, vals_dmon_b, width=0.35, label="Run B (Mask ON)", color="#2ecc71")
    ax.set_xticks(x_d)
    ax.set_xticklabels(metrics_labels)
    ax.set_title("Section 7: RTX 5090 Hardware Telemetry (nvidia-smi dmon)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.out_dir / "fig0_gpu_pipeline_benchmarks.png", dpi=160)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # 8.1 & 8.2: 64-Antenna Baseband Inspection & Python vs. CUDA Mask Match
    # -------------------------------------------------------------------------
    ant_mean_re = np.zeros(NUM_ELEMENTS)
    ant_mean_im = np.zeros(NUM_ELEMENTS)
    ant_rms = np.zeros(NUM_ELEMENTS)
    ant_power = np.zeros(NUM_ELEMENTS)
    ant_passband_p = np.zeros(NUM_ELEMENTS)
    ant_coh = np.zeros(NUM_ELEMENTS)
    ant_kurtosis = np.zeros(NUM_ELEMENTS)
    ant_clip_frac = np.zeros(NUM_ELEMENTS)
    ant_hist_re = np.zeros((NUM_ELEMENTS, 16), dtype=np.int64)
    swapped_a = False

    if raw_a_f0:
        # Use the most complete frame in Run A (Frame 1 has 336/336 active channels)
        best_frame_idx = len(raw_a_f0) - 1
        swapped_a = (raw_a_f0[best_frame_idx][0].data_type == KOTEKAN_DTYPE_INT4X2_SWAPPED_WITHOFFSET)
        f0_u8 = raw_a_f0[best_frame_idx][1].reshape(SAMPLES_PER_FRAME, NUM_LOCAL_FREQ, NUM_ELEMENTS)
        flat_u8 = f0_u8.reshape(-1, NUM_ELEMENTS)
        re_s, im_s = decode_int4x2(flat_u8, swapped_with_offset=swapped_a)
        re_f = re_s.astype(np.float64)
        im_f = im_s.astype(np.float64)
        p_f = re_f ** 2 + im_f ** 2

        ant_mean_re = np.mean(re_f, axis=0)
        ant_mean_im = np.mean(im_f, axis=0)
        ant_power = np.mean(p_f, axis=0)
        ant_rms = np.sqrt(ant_power)
        ant_kurtosis = np.mean(p_f ** 2, axis=0) / np.maximum(ant_power ** 2, 1e-12)
        clips = ((re_s == 7) | (re_s == -8)).astype(np.int32) + ((im_s == 7) | (im_s == -8)).astype(np.int32)
        ant_clip_frac = np.mean(clips, axis=0) / 2.0

        for e in range(NUM_ELEMENTS):
            counts, _ = np.histogram(re_s[:, e], bins=np.arange(-8.5, 8.5, 1.0))
            ant_hist_re[e] = counts

        # Per-channel passband power & cross-antenna spatial coherence matching inspect_channel_coherence_kernel
        t_step = max(1, SAMPLES_PER_FRAME // 256)
        sub_u8 = f0_u8[::t_step][:256]  # (256, 336, 64)
        re_3d, im_3d = decode_int4x2(sub_u8, swapped_with_offset=swapped_a)
        v_3d = re_3d.astype(np.float32) + 1j * im_3d.astype(np.float32)
        p_3d = re_3d.astype(np.float32) ** 2 + im_3d.astype(np.float32) ** 2

        chan_p = np.zeros((NUM_LOCAL_FREQ, NUM_ELEMENTS), dtype=np.float32)
        chan_c = np.zeros((NUM_LOCAL_FREQ, NUM_ELEMENTS), dtype=np.float32)
        valid_f_list = []

        for f_idx in range(NUM_LOCAL_FREQ):
            row_nz = np.any(p_3d[:, f_idx, :] > 0, axis=1)
            n_val = int(np.sum(row_nz))
            if n_val == 0:
                continue
            valid_f_list.append(f_idx)
            vf = v_3d[row_nz, f_idx, :]  # (n_val, 64)
            pf = np.mean(p_3d[row_nz, f_idx, :], axis=0)  # (64,)
            kf = np.mean(p_3d[row_nz, f_idx, :] ** 2, axis=0) / np.maximum(pf ** 2, 1e-6)
            chan_p[f_idx] = pf

            valid_partner = (pf > 0.5) & (pf < 12.0) & (kf < 2.8)
            gram = np.abs((vf.T @ np.conj(vf)) / float(n_val))
            denom = np.sqrt(np.outer(pf, pf)) + 1e-6
            coh_mat = gram / denom
            for a_i in range(NUM_ELEMENTS):
                if pf[a_i] <= 0.5:
                    continue
                best_c = 0.0
                for b_i in range(NUM_ELEMENTS):
                    if a_i == b_i or not valid_partner[b_i]:
                        continue
                    if (a_i // 32) == (b_i // 32) and (abs(a_i - b_i) % 8 == 0):
                        continue
                    if coh_mat[a_i, b_i] > best_c:
                        best_c = float(coh_mat[a_i, b_i])
                chan_c[f_idx, a_i] = best_c

        if len(valid_f_list) >= 4:
            vp = chan_p[valid_f_list]  # (F_val, 64)
            vc = chan_c[valid_f_list]
            ant_passband_p = np.sort(vp, axis=0)[(3 * len(valid_f_list)) // 4]
            ant_coh = np.mean(vc, axis=0)

    dead_th = 1.0
    sat_th = 80.0
    clip_th = 0.15
    min_kurt = 1.2
    max_kurt = 4.5
    mad_th = 6.0
    if bench_b["mask_status_records"]:
        th_cfg = bench_b["mask_status_records"][-1].get("thresholds", {})
        dead_th = float(th_cfg.get("dead_power_threshold", dead_th))
        sat_th = float(th_cfg.get("sat_power_threshold", sat_th))
        clip_th = float(th_cfg.get("clip_fraction_threshold", clip_th))
        min_kurt = float(th_cfg.get("min_kurtosis_threshold", min_kurt))
        max_kurt = float(th_cfg.get("max_kurtosis_threshold", max_kurt))
        mad_th = float(th_cfg.get("mad_sigma_threshold", mad_th))

    med_coh = float(np.median(ant_coh[ant_passband_p > 1e-3])) if np.any(ant_passband_p > 1e-3) else 0.0
    max_coh = float(np.max(ant_coh)) if len(ant_coh) else 0.0
    has_coherent_sky = (med_coh > 0.02) and (max_coh > 1.25 * med_coh)
    coh_thresh = 1.15 * med_coh if has_coherent_sky else 0.0

    valid_pb = ant_passband_p[ant_passband_p > 1e-3]
    med_pb = float(np.median(valid_pb)) if len(valid_pb) else 0.0
    effective_dead_th = max(dead_th, med_pb) if has_coherent_sky else dead_th

    ref_p = np.where(ant_passband_p > 0.0, ant_passband_p, ant_power)
    coh_ok = (ant_coh >= coh_thresh) if has_coherent_sky else np.ones(NUM_ELEMENTS, dtype=bool)
    cands_p = ref_p[coh_ok & (ref_p > effective_dead_th) & (ant_power < sat_th)]
    med_p = float(np.median(cands_p)) if len(cands_p) >= 4 else 0.0
    mad_p = max(1.4826 * float(np.median(np.abs(cands_p - med_p))), 0.15 * med_p) if len(cands_p) >= 4 else 1.0
    z_scores = np.abs(ref_p - med_p) / mad_p if len(cands_p) >= 4 else np.zeros(NUM_ELEMENTS)

    # Python offline passband + spatial coherence + multi-moment detector matching cudaAntennaMaskCommand
    py_alive_mask = (
        coh_ok
        & (ref_p > effective_dead_th)
        & (ant_power < sat_th)
        & (ant_clip_frac < clip_th)
        & (ant_kurtosis > min_kurt)
        & (ant_kurtosis < max_kurt)
        & (z_scores < mad_th)
    )
    py_alive_raw_elems = np.where(py_alive_mask)[0].tolist()[:MAX_ALIVE_ANTENNAS]
    py_alive_phys_ants = [(NUM_ELEMENTS - 1) - e for e in py_alive_raw_elems]

    cuda_alive_raw_elems: List[int] = []
    cuda_alive_phys_ants: List[int] = []
    for rec in reversed(bench_b["tracker_status_records"]):
        if rec.get("active_raw_elements"):
            cuda_alive_raw_elems = rec.get("active_raw_elements", [])
            cuda_alive_phys_ants = rec.get("active_physical_antennas", [])
            break
    if not cuda_alive_raw_elems and bench_b["mask_status_records"]:
        for rec in reversed(bench_b["mask_status_records"]):
            if rec.get("active_raw_elements"):
                cuda_alive_raw_elems = rec.get("active_raw_elements", [])
                cuda_alive_phys_ants = rec.get("active_physical_antennas", [])
                break

    print(f"Python detector alive raw elements ({len(py_alive_raw_elems)}): {py_alive_raw_elems}")
    print(f"Python detector alive physical antennas: {py_alive_phys_ants}")
    print(f"CUDA Run B logged alive raw elements ({len(cuda_alive_raw_elems)}): {cuda_alive_raw_elems}")

    # Plot 8.1a: 64-Antenna Health, RMS, Power, Clipping & Histograms
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    phys_indices = np.array([(NUM_ELEMENTS - 1) - e for e in range(NUM_ELEMENTS)])
    order_by_phys = np.argsort(phys_indices)

    ax = axes[0, 0]
    colors = ["#2ecc71" if py_alive_mask[e] else "#e74c3c" for e in order_by_phys]
    ax.bar(np.arange(NUM_ELEMENTS), ant_power[order_by_phys], color=colors, width=0.8)
    ax.axhline(dead_th, color="orange", linestyle="--", label=f"Dead Threshold ({dead_th:.2f})")
    ax.axhline(sat_th, color="red", linestyle="--", label=f"Sat Threshold ({sat_th:.1f})")
    ax.set_yscale("log")
    ax.set_xlabel("Physical Antenna Index (0..63)")
    ax.set_ylabel("Mean Complex Power <|V|^2> (ADC^2)")
    ax.set_title("Run A: 64-Antenna Power & Auto-Mask Separation (Green = Alive)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.bar(np.arange(NUM_ELEMENTS), ant_clip_frac[order_by_phys] * 100.0, color=colors, width=0.8)
    ax.axhline(15.0, color="red", linestyle="--", label="Clip Threshold (15%)")
    ax.set_xlabel("Physical Antenna Index (0..63)")
    ax.set_ylabel("4-bit Rail Clipping Rate (%)")
    ax.set_title("Run A: 64-Antenna Saturation / Rail Clipping Fraction")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    vals = np.arange(-8, 8)
    for e in py_alive_raw_elems[:7]:
        phys_a = (NUM_ELEMENTS - 1) - e
        p_norm = ant_hist_re[e] / max(1, np.sum(ant_hist_re[e]))
        ax.plot(vals, p_norm, marker="o", markersize=4, label=f"Ant {phys_a} (raw {e}, RMS={ant_rms[e]:.2f})")
    dead_example = [e for e in range(NUM_ELEMENTS) if not py_alive_mask[e]][0] if len(py_alive_raw_elems) < NUM_ELEMENTS else 0
    ax.plot(vals, ant_hist_re[dead_example] / max(1, np.sum(ant_hist_re[dead_example])),
            color="gray", linestyle="--", label=f"Dead Ant {(63-dead_example)} (raw {dead_example})")
    ax.set_xlabel("4-bit Real Sample Value (-8..7)")
    ax.set_ylabel("Probability Density")
    ax.set_title("Run A: 4-Bit Voltage Histograms (Alive vs. Dead Inputs)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    grid_pwr = np.zeros((8, 8))
    for e in range(NUM_ELEMENTS):
        pa = (NUM_ELEMENTS - 1) - e
        grid_pwr[pa // 8, pa % 8] = ant_power[e]
    im_plot = ax.imshow(grid_pwr, cmap="viridis", origin="lower")
    for r_idx in range(8):
        for c_idx in range(8):
            pa = r_idx * 8 + c_idx
            e = 63 - pa
            status_ch = "OK" if py_alive_mask[e] else "X"
            ax.text(c_idx, r_idx, f"A{pa}\n{ant_power[e]:.1f}\n({status_ch})",
                    ha="center", va="center", color="white" if ant_power[e] < 20 else "black", fontsize=7)
    plt.colorbar(im_plot, ax=ax, label="Mean Power <|V|^2>")
    ax.set_title("8x8 Physical Array Layout Health Map")
    ax.set_xlabel("Grid Column (0..7)")
    ax.set_ylabel("Grid Row (0..7)")

    plt.tight_layout()
    plt.savefig(args.out_dir / "fig1_runA_64ant_health_histograms.png", dpi=160)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # 8.1b & 8.2b: Bandpass PSD & Waterfall Spectrograms (Run A & Run B)
    # -------------------------------------------------------------------------
    freqs_f0 = 300.0 + np.arange(NUM_LOCAL_FREQ) * 0.3
    freqs_f1 = 400.8 + np.arange(NUM_LOCAL_FREQ) * 0.3

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    alive_extract_elems = sorted(cuda_alive_raw_elems if cuda_alive_raw_elems else py_alive_raw_elems)[:MAX_ALIVE_ANTENNAS]
    swapped_b = (alive_b_f0[0][0].data_type == KOTEKAN_DTYPE_INT4X2_SWAPPED_WITHOFFSET) if alive_b_f0 else swapped_a

    # Run B alive baseband PSD (Stream 0 and Stream 1)
    if alive_b_f0:
        b0_u8 = alive_b_f0[0][1].reshape(SAMPLES_PER_FRAME, NUM_LOCAL_FREQ, MAX_ALIVE_ANTENNAS)
        re0, im0 = decode_int4x2(b0_u8, swapped_with_offset=swapped_b)
        psd0 = np.mean(re0.astype(np.float32) ** 2 + im0.astype(np.float32) ** 2, axis=0)  # (336, 7)
        for ch_i in range(MAX_ALIVE_ANTENNAS):
            label_str = f"Slot {ch_i} (zero-padded)"
            if ch_i < len(alive_extract_elems):
                e_raw = alive_extract_elems[ch_i]
                label_str = f"Slot {ch_i}: Ant {63 - e_raw} (raw {e_raw})"
            axes[0, 0].plot(freqs_f0, 10.0 * np.log10(psd0[:, ch_i] + 1e-12), label=label_str, lw=1.0)
        axes[0, 0].set_title("Run B Compacted Alive Baseband PSD — Stream 0 (300.0–400.5 MHz)")
        axes[0, 0].set_xlabel("Frequency (MHz)")
        axes[0, 0].set_ylabel("Power (dB re 1 ADC^2)")
        axes[0, 0].legend(fontsize=8)
        axes[0, 0].grid(True, alpha=0.3)

    if alive_b_f1:
        b1_u8 = alive_b_f1[0][1].reshape(SAMPLES_PER_FRAME, NUM_LOCAL_FREQ, MAX_ALIVE_ANTENNAS)
        re1, im1 = decode_int4x2(b1_u8, swapped_with_offset=swapped_b)
        psd1 = np.mean(re1.astype(np.float32) ** 2 + im1.astype(np.float32) ** 2, axis=0)
        for ch_i in range(MAX_ALIVE_ANTENNAS):
            label_str = f"Slot {ch_i} (zero-padded)"
            if ch_i < len(alive_extract_elems):
                e_raw = alive_extract_elems[ch_i]
                label_str = f"Slot {ch_i}: Ant {63 - e_raw} (raw {e_raw})"
            axes[0, 1].plot(freqs_f1, 10.0 * np.log10(psd1[:, ch_i] + 1e-12), label=label_str, lw=1.0)
        axes[0, 1].set_title("Run B Compacted Alive Baseband PSD — Stream 1 (400.8–501.3 MHz)")
        axes[0, 1].set_xlabel("Frequency (MHz)")
        axes[0, 1].set_ylabel("Power (dB re 1 ADC^2)")
        axes[0, 1].legend(fontsize=8)
        axes[0, 1].grid(True, alpha=0.3)

    # Dynamic spectrum waterfall across all 9 frames of Run B (Slot 0, time-binned by 256 samples = 0.853 ms)
    if alive_b_f0:
        wf_list = []
        for hdr, mm in alive_b_f0:
            arr_u8 = mm.reshape(SAMPLES_PER_FRAME, NUM_LOCAL_FREQ, MAX_ALIVE_ANTENNAS)[:, :, 0]
            r_c, i_c = decode_int4x2(arr_u8, swapped_with_offset=swapped_b)
            p_c = (r_c.astype(np.float32) ** 2 + i_c.astype(np.float32) ** 2).reshape(60, 256, NUM_LOCAL_FREQ).mean(axis=1)
            wf_list.append(p_c)
        wf0 = np.vstack(wf_list)
        im0_wf = axes[1, 0].imshow(
            10.0 * np.log10(wf0 + 1e-12),
            aspect="auto",
            origin="lower",
            extent=[freqs_f0[0], freqs_f0[-1], 0, len(alive_b_f0) * FRAME_DURATION_MS],
            cmap="inferno",
        )
        plt.colorbar(im0_wf, ax=axes[1, 0], label="Power (dB)")
        axes[1, 0].set_title(f"Run B Alive Slot 0 Waterfall — Stream 0 ({len(alive_b_f0)} frames = {len(alive_b_f0)*FRAME_DURATION_MS:.1f} ms)")
        axes[1, 0].set_xlabel("Frequency (MHz)")
        axes[1, 0].set_ylabel("Time (ms)")

    if alive_b_f1:
        wf_list1 = []
        for hdr, mm in alive_b_f1:
            arr_u8 = mm.reshape(SAMPLES_PER_FRAME, NUM_LOCAL_FREQ, MAX_ALIVE_ANTENNAS)[:, :, 0]
            r_c, i_c = decode_int4x2(arr_u8, swapped_with_offset=swapped_b)
            p_c = (r_c.astype(np.float32) ** 2 + i_c.astype(np.float32) ** 2).reshape(60, 256, NUM_LOCAL_FREQ).mean(axis=1)
            wf_list1.append(p_c)
        wf1 = np.vstack(wf_list1)
        im1_wf = axes[1, 1].imshow(
            10.0 * np.log10(wf1 + 1e-12),
            aspect="auto",
            origin="lower",
            extent=[freqs_f1[0], freqs_f1[-1], 0, len(alive_b_f1) * FRAME_DURATION_MS],
            cmap="inferno",
        )
        plt.colorbar(im1_wf, ax=axes[1, 1], label="Power (dB)")
        axes[1, 1].set_title(f"Run B Alive Slot 0 Waterfall — Stream 1 ({len(alive_b_f1)} frames = {len(alive_b_f1)*FRAME_DURATION_MS:.1f} ms)")
        axes[1, 1].set_xlabel("Frequency (MHz)")
        axes[1, 1].set_ylabel("Time (ms)")

    plt.tight_layout()
    plt.savefig(args.out_dir / "fig2_bandpass_and_waterfalls.png", dpi=160)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # 8.1c: 7x7 Zero-Lag Cross-Correlation Coherence Matrix
    # -------------------------------------------------------------------------
    if alive_b_f0:
        b0_u8 = alive_b_f0[0][1].reshape(SAMPLES_PER_FRAME, NUM_LOCAL_FREQ, MAX_ALIVE_ANTENNAS)
        re0, im0 = decode_int4x2(b0_u8, swapped_with_offset=swapped_b)
        v_c = re0.astype(np.float32) + 1j * im0.astype(np.float32)  # (T, F, 7)
        v_flat = v_c.reshape(-1, MAX_ALIVE_ANTENNAS)
        cov = (v_flat.T @ np.conj(v_flat)) / v_flat.shape[0]
        diag = np.sqrt(np.real(np.diag(cov)))
        denom = np.outer(diag, diag) + 1e-12
        coh_mag = np.abs(cov) / denom
        coh_phase = np.degrees(np.angle(cov))

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        im_m = axes[0].imshow(coh_mag, vmin=0, vmax=1, cmap="magma")
        plt.colorbar(im_m, ax=axes[0], label="|R_ij| Coherence Magnitude")
        axes[0].set_title("7x7 Zero-Lag Coherence Magnitude |R_ij| (Run B Stream 0)")

        im_p = axes[1].imshow(coh_phase, vmin=-180, vmax=180, cmap="twilight")
        plt.colorbar(im_p, ax=axes[1], label="Phase(R_ij) [deg]")
        axes[1].set_title("7x7 Zero-Lag Cross-Correlation Phase [deg]")

        tick_labels = [
            f"Ant {63 - alive_extract_elems[i]}" if i < len(alive_extract_elems) else f"Slot {i}"
            for i in range(MAX_ALIVE_ANTENNAS)
        ]
        for ax in axes:
            ax.set_xticks(range(MAX_ALIVE_ANTENNAS))
            ax.set_yticks(range(MAX_ALIVE_ANTENNAS))
            ax.set_xticklabels(tick_labels, rotation=45, fontsize=8)
            ax.set_yticklabels(tick_labels, fontsize=8)
        for i in range(MAX_ALIVE_ANTENNAS):
            for j in range(MAX_ALIVE_ANTENNAS):
                axes[0].text(j, i, f"{coh_mag[i, j]:.3f}", ha="center", va="center",
                             color="white" if coh_mag[i, j] < 0.6 else "black", fontsize=8)
                axes[1].text(j, i, f"{coh_phase[i, j]:.1f}°", ha="center", va="center",
                             color="black", fontsize=8)
        plt.tight_layout()
        plt.savefig(args.out_dir / "fig3_7x7_coherence_matrix.png", dpi=160)
        plt.close(fig)

    # -------------------------------------------------------------------------
    # 8.2: Masking Verification (Passband Power vs Coherence & Compacted Slots)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    ax = axes[0]
    for e in range(NUM_ELEMENTS):
        pa = 63 - e
        is_live = bool(py_alive_mask[e])
        ax.scatter(
            ant_passband_p[e],
            ant_coh[e],
            c="#2ecc71" if is_live else "#e74c3c",
            s=75 if is_live else 30,
            alpha=0.9 if is_live else 0.55,
            edgecolors="black" if is_live else "none",
            zorder=4 if is_live else 2,
        )
        if is_live:
            ax.annotate(
                f"Ant {pa} (raw {e})",
                (ant_passband_p[e], ant_coh[e]),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=8,
                fontweight="bold",
            )
    if coh_thresh > 0:
        ax.axhline(coh_thresh, color="#2980b9", linestyle="--", label=f"Self-Calibrated Coherence Floor ({coh_thresh:.4f})")
    ax.axvline(effective_dead_th, color="#e67e22", linestyle="--", label=f"Self-Calibrated Passband Floor ({effective_dead_th:.3f})")
    ax.set_xlabel("75th-Percentile In-Band Passband Power (ADC^2)")
    ax.set_ylabel("Mean Cross-Antenna Spatial Coherence <|V_a V_b*|> / sqrt(P_a P_b)")
    ax.set_title("Section 8.2: Automatic 2D Passband Power & Spatial Coherence Separation")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    if alive_b_f0:
        b0_u8 = alive_b_f0[-1][1].reshape(SAMPLES_PER_FRAME, NUM_LOCAL_FREQ, MAX_ALIVE_ANTENNAS)
        re0, im0 = decode_int4x2(b0_u8, swapped_with_offset=swapped_b)
        slot_pwr0 = np.mean(re0.astype(np.float32) ** 2 + im0.astype(np.float32) ** 2, axis=(0, 1))
        slot_labels = [
            f"Slot {i}\nAnt {63 - alive_extract_elems[i]}\n(raw {alive_extract_elems[i]})"
            if i < len(alive_extract_elems)
            else f"Slot {i}\n(Zero Pad)"
            for i in range(MAX_ALIVE_ANTENNAS)
        ]
        bar_cols = ["#2ecc71" if i < len(alive_extract_elems) else "#95a5a6" for i in range(MAX_ALIVE_ANTENNAS)]
        ax.bar(range(MAX_ALIVE_ANTENNAS), slot_pwr0, color=bar_cols, width=0.6)
        ax.set_xticks(range(MAX_ALIVE_ANTENNAS))
        ax.set_xticklabels(slot_labels, fontsize=8)
        for i, val in enumerate(slot_pwr0):
            ax.text(i, val + 0.08, f"{val:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_ylim(0, max(5.0, float(np.max(slot_pwr0)) * 1.2))
        ax.set_ylabel("Mean Extracted Baseband Power (ADC^2)")
        ax.set_title(f"Run B Compacted Baseband Verification ({len(alive_extract_elems)} Alive + {MAX_ALIVE_ANTENNAS - len(alive_extract_elems)} Zero-Padded Slots)")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out_dir / "fig4_masking_verification_runA_vs_runB.png", dpi=160)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # 8.3a: Astrometric (l, m) & Delay Phase Verification
    # -------------------------------------------------------------------------
    utc_ref = 0.0
    logged_l, logged_m, logged_n = None, None, None
    if beams_b_f0 and beams_b_f0[0][0].recv_time_sec > 0.0:
        utc_ref = beams_b_f0[0][0].recv_time_sec
    if bench_b["tracker_status_records"]:
        rec = bench_b["tracker_status_records"][-1]
        if utc_ref <= 0.0:
            utc_ref = float(rec.get("last_timestamp_utc", 0.0))
        beams_list = rec.get("beams", [])
        if beams_list:
            logged_l = float(beams_list[0].get("l0", 0.0))
            logged_m = float(beams_list[0].get("m0", 0.0))
            logged_n = float(beams_list[0].get("n0", 1.0))
    if utc_ref <= 0.0:
        beam_file_b = args.dump_b / "formed_beams_64ant_freq0_0000000.bin"
        utc_ref = beam_file_b.stat().st_mtime if beam_file_b.exists() else 1793318400.0

    lst_hr, cpp_l, cpp_m, cpp_n = compute_cpp_astrometry(utc_ref, GRUS_RA_DEG, GRUS_DEC_DEG)
    astropy_l, astropy_m, astropy_alt, astropy_az = cpp_l, cpp_m, math.degrees(math.asin(cpp_n)), 0.0
    if HAS_ASTROPY:
        site = EarthLocation(lat=CAREN_LAT_DEG * u.deg, lon=CAREN_LON_DEG * u.deg, height=CAREN_ALT_M * u.m)
        t_obs = Time(utc_ref, format="unix", scale="utc")
        sc = SkyCoord(ra=GRUS_RA_DEG * u.deg, dec=GRUS_DEC_DEG * u.deg, frame="icrs")
        altaz = sc.transform_to(AltAz(obstime=t_obs, location=site))
        astropy_alt = float(altaz.alt.deg)
        astropy_az = float(altaz.az.deg)
        astropy_l = math.cos(altaz.alt.rad) * math.sin(altaz.az.rad)
        astropy_m = math.cos(altaz.alt.rad) * math.cos(altaz.az.rad)

    positions_all, _ = compute_taper_weights(np.ones(NUM_ELEMENTS, dtype=np.uint8))
    frame_times_ms = np.linspace(0, 4 * FRAME_DURATION_MS, 50)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for idx_a, e_raw in enumerate(alive_extract_elems):
        pos = positions_all[e_raw]
        taus_ns = []
        for dt_ms in frame_times_ms:
            _, l_t, m_t, n_t = compute_cpp_astrometry(utc_ref + dt_ms * 1e-3, GRUS_RA_DEG, GRUS_DEC_DEG)
            delay_m = pos[0] * l_t + pos[1] * m_t + pos[2] * n_t
            taus_ns.append((delay_m / SPEED_OF_LIGHT) * 1e9)
        taus_ns = np.array(taus_ns)
        axes[0].plot(frame_times_ms, taus_ns, label=f"Ant {63 - e_raw} (raw {e_raw})")
        axes[1].plot(frame_times_ms, (taus_ns - taus_ns[0]) * 1e6, label=f"Ant {63 - e_raw} (raw {e_raw})")

    axes[0].set_title(f"Geometric Delay tau_k(t) on Grus Quartet (Alt={astropy_alt:.1f}°)")
    axes[0].set_xlabel("Time Elapsed Across 4 Frames (ms)")
    axes[0].set_ylabel("Geometric Delay tau_k (ns)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Sub-Frame Drift Delta tau_k(t) - tau_k(0) (smooth, no wraps/jumps)")
    axes[1].set_xlabel("Time Elapsed Across 4 Frames (ms)")
    axes[1].set_ylabel("Delay Drift (fs)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out_dir / "fig5_astrometry_and_delays.png", dpi=160)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # 8.3b: Formed-Beam Waterfall & Integrated Power P(t) (Run A vs Run B)
    # -------------------------------------------------------------------------
    def extract_beam_power_series(beam_frames):
        wf_chunks = []
        p_t_chunks = []
        spec_accum = np.zeros(NUM_LOCAL_FREQ, dtype=np.float64)
        for hdr, mm in beam_frames:
            b_c64 = np.frombuffer(mm, dtype=np.complex64).reshape(SAMPLES_PER_FRAME, NUM_LOCAL_FREQ)
            pwr = np.abs(b_c64) ** 2  # (15360, 336)
            spec_f = np.mean(pwr, axis=0)
            spec_accum += spec_f
            pwr_binned = pwr.reshape(240, 64, NUM_LOCAL_FREQ).mean(axis=1)
            wf_chunks.append(pwr_binned)
            med_f = np.median(spec_f)
            mad_f = np.median(np.abs(spec_f - med_f)) + 1e-6
            clean_mask = np.abs(spec_f - med_f) < 5.0 * mad_f
            p_t_chunks.append(np.mean(pwr_binned[:, clean_mask], axis=1))
        if not wf_chunks:
            return None, None, None
        return np.vstack(wf_chunks), np.concatenate(p_t_chunks), spec_accum / len(beam_frames)

    wf_a0, pt_a0, spec_a0 = extract_beam_power_series(beams_a_f0)
    wf_a1, pt_a1, spec_a1 = extract_beam_power_series(beams_a_f1)
    wf_b0, pt_b0, spec_b0 = extract_beam_power_series(beams_b_f0)
    wf_b1, pt_b1, spec_b1 = extract_beam_power_series(beams_b_f1)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    if wf_b0 is not None:
        im_b0 = axes[0, 0].imshow(
            10.0 * np.log10(wf_b0 + 1e-12),
            aspect="auto",
            origin="lower",
            extent=[freqs_f0[0], freqs_f0[-1], 0, len(beams_b_f0) * FRAME_DURATION_MS],
            cmap="viridis",
        )
        plt.colorbar(im_b0, ax=axes[0, 0], label="Power (dB)")
        axes[0, 0].set_title("Run B Formed Beam 0 Waterfall — Stream 0 (300.0–400.5 MHz)")
        axes[0, 0].set_xlabel("Frequency (MHz)")
        axes[0, 0].set_ylabel("Time (ms)")

    if wf_b1 is not None:
        im_b1 = axes[0, 1].imshow(
            10.0 * np.log10(wf_b1 + 1e-12),
            aspect="auto",
            origin="lower",
            extent=[freqs_f1[0], freqs_f1[-1], 0, len(beams_b_f1) * FRAME_DURATION_MS],
            cmap="viridis",
        )
        plt.colorbar(im_b1, ax=axes[0, 1], label="Power (dB)")
        axes[0, 1].set_title("Run B Formed Beam 0 Waterfall — Stream 1 (400.8–501.3 MHz)")
        axes[0, 1].set_xlabel("Frequency (MHz)")
        axes[0, 1].set_ylabel("Time (ms)")

    if pt_a0 is not None and pt_b0 is not None:
        t_axis = np.linspace(0, len(beams_b_f0) * FRAME_DURATION_MS, len(pt_b0))
        axes[1, 0].plot(t_axis[:len(pt_a0)], pt_a0, label=f"Run A (Unmasked 64-ant): mean={np.mean(pt_a0):.3f}", alpha=0.85)
        axes[1, 0].plot(t_axis, pt_b0, label=f"Run B (Masked {len(alive_extract_elems)}-ant): mean={np.mean(pt_b0):.3f}", alpha=0.9)
        for f_i in range(1, 4):
            axes[1, 0].axvline(f_i * FRAME_DURATION_MS, color="gray", linestyle=":", alpha=0.6)
        axes[1, 0].set_title("Band-Average Formed Beam Power P(t) Across 4 Frames (Stream 0)")
        axes[1, 0].set_xlabel("Time (ms) — Dotted Lines = 51.2 ms Frame Boundaries")
        axes[1, 0].set_ylabel("Mean Formed Beam Power <|B(t,f)|^2>")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    if spec_a0 is not None and spec_b0 is not None:
        axes[1, 1].plot(freqs_f0, 10.0 * np.log10(spec_a0 + 1e-12), label="Run A Stream 0 (Unmasked 64-ant)", alpha=0.8)
        axes[1, 1].plot(freqs_f0, 10.0 * np.log10(spec_b0 + 1e-12), label="Run B Stream 0 (Masked Alive)", alpha=0.9)
        if spec_a1 is not None and spec_b1 is not None:
            axes[1, 1].plot(freqs_f1, 10.0 * np.log10(spec_a1 + 1e-12), label="Run A Stream 1 (Unmasked 64-ant)", alpha=0.8)
            axes[1, 1].plot(freqs_f1, 10.0 * np.log10(spec_b1 + 1e-12), label="Run B Stream 1 (Masked Alive)", alpha=0.9)
        axes[1, 1].set_title("Formed Beam PSD Comparison: Run A (Unmasked) vs. Run B (Masked)")
        axes[1, 1].set_xlabel("Frequency (MHz)")
        axes[1, 1].set_ylabel("Formed Beam Power (dB)")
        axes[1, 1].legend(fontsize=8)
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.out_dir / "fig6_formed_beam_waterfall_and_power.png", dpi=160)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # 8.3c & 8.3d: Coherent vs Incoherent Power Ratio (eta) & CPU Golden Replay
    # -------------------------------------------------------------------------
    def match_frames(base_frames, beam_frames):
        base_map = {h.fpga_seq_num: (h, mm) for h, mm in base_frames if h.fpga_seq_num >= 0}
        for h, mm in beam_frames:
            if h.fpga_seq_num in base_map:
                return base_map[h.fpga_seq_num], (h, mm), True
        if base_frames and beam_frames:
            return base_frames[0], beam_frames[0], False
        return None, None, False

    eta_f0, eta_f1 = None, None
    mask_b_vec = np.zeros(NUM_ELEMENTS, dtype=np.uint8)
    for e in alive_extract_elems:
        mask_b_vec[e] = 1
    _, norm_taper_b = compute_taper_weights(mask_b_vec)
    w_sq_alive = np.array([norm_taper_b[e] ** 2 for e in alive_extract_elems])

    if alive_b_f0 and beams_b_f0 and len(w_sq_alive) > 0:
        (hb0, mm_alive0), (hbm0, mm_beam0), aligned0 = match_frames(alive_b_f0, beams_b_f0)
        re0, im0 = decode_int4x2(mm_alive0.reshape(SAMPLES_PER_FRAME, NUM_LOCAL_FREQ, MAX_ALIVE_ANTENNAS), swapped_with_offset=swapped_b)
        v0_pwr = np.mean(re0.astype(np.float64) ** 2 + im0.astype(np.float64) ** 2, axis=0)  # (336, 7)
        b0_c64 = np.frombuffer(mm_beam0, dtype=np.complex64).reshape(SAMPLES_PER_FRAME, NUM_LOCAL_FREQ)
        b0_pwr = np.mean(np.abs(b0_c64.astype(np.complex128)) ** 2, axis=0)  # (336,)
        incoh_expected0 = np.sum(v0_pwr[:, :len(w_sq_alive)] * w_sq_alive[None, :], axis=1)
        eta_f0 = b0_pwr / (incoh_expected0 + 1e-12)

    if alive_b_f1 and beams_b_f1 and len(w_sq_alive) > 0:
        (hb1, mm_alive1), (hbm1, mm_beam1), aligned1 = match_frames(alive_b_f1, beams_b_f1)
        re1, im1 = decode_int4x2(mm_alive1.reshape(SAMPLES_PER_FRAME, NUM_LOCAL_FREQ, MAX_ALIVE_ANTENNAS), swapped_with_offset=swapped_b)
        v1_pwr = np.mean(re1.astype(np.float64) ** 2 + im1.astype(np.float64) ** 2, axis=0)
        b1_c64 = np.frombuffer(mm_beam1, dtype=np.complex64).reshape(SAMPLES_PER_FRAME, NUM_LOCAL_FREQ)
        b1_pwr = np.mean(np.abs(b1_c64.astype(np.complex128)) ** 2, axis=0)
        incoh_expected1 = np.sum(v1_pwr[:, :len(w_sq_alive)] * w_sq_alive[None, :], axis=1)
        eta_f1 = b1_pwr / (incoh_expected1 + 1e-12)

    if eta_f0 is not None:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(freqs_f0, eta_f0, label=f"Stream 0 (300–400.5 MHz): median eta = {np.median(eta_f0):.3f}", lw=1.2)
        if eta_f1 is not None:
            ax.plot(freqs_f1, eta_f1, label=f"Stream 1 (400.8–501.3 MHz): median eta = {np.median(eta_f1):.3f}", lw=1.2)
        ax.axhline(1.0, color="black", linestyle="--", label="Incoherent Baseline (eta = 1.0)")
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Coherent-to-Incoherent Ratio eta(f)")
        ax.set_title("Run B: Coherent vs. Incoherent Power Ratio eta(f) = <|B|^2> / sum(w_k^2 <|V_k|^2>)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(args.out_dir / "fig7_coherent_incoherent_ratio_eta.png", dpi=160)
        plt.close(fig)

    # 8.3d: CPU Golden-Reference Delay-and-Sum Replay on Run A (64-ant) AND Run B (7-ant alive)
    max_abs_err_a, rel_rms_err_a = None, None
    max_abs_err_b, rel_rms_err_b = None, None
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    if raw_a_f0 and beams_a_f0:
        (ha_raw, mm_raw_a), (ha_bm, mm_bm_a), aligned_a = match_frames(raw_a_f0, beams_a_f0)
        ts_a = ha_raw.recv_time_sec if ha_raw.recv_time_sec > 0.0 else utc_ref
        if bench_a["tracker_status_records"] and ha_raw.recv_time_sec <= 0.0:
            ts_a = float(bench_a["tracker_status_records"][0].get("last_timestamp_utc", utc_ref))

        b_gpu_a, b_cpu_a, max_abs_err_a, rel_rms_err_a = run_cpu_golden_replay(
            raw_u8_frame=mm_raw_a.reshape(SAMPLES_PER_FRAME, NUM_LOCAL_FREQ, NUM_ELEMENTS),
            beam_gpu_frame=np.frombuffer(mm_bm_a, dtype=np.complex64).reshape(SAMPLES_PER_FRAME, NUM_LOCAL_FREQ),
            active_raw_elems=list(range(NUM_ELEMENTS)),
            full_64_mask=np.ones(NUM_ELEMENTS, dtype=np.uint8),
            utc_ts=ts_a,
            freq_start_hz=300.0e6,
            swapped_with_offset=swapped_a,
        )
        print(f"CPU Golden Replay (Run A, 64-ant): seq={ha_raw.fpga_seq_num}, rel_rms_err={rel_rms_err_a:.4e}, max_abs_err={max_abs_err_a:.4e}")

        axes[0, 0].plot(freqs_f0, np.mean(np.abs(b_gpu_a) ** 2, axis=0), label="GPU Formed Beam <|B_GPU|^2>", lw=1.5)
        axes[0, 0].plot(freqs_f0, np.mean(np.abs(b_cpu_a) ** 2, axis=0), linestyle="--", label="CPU Golden Replay <|B_CPU|^2>", lw=1.2)
        axes[0, 0].set_title(f"Run A (64-Ant Raw) CPU Golden Replay vs. GPU (seq={ha_raw.fpga_seq_num})")
        axes[0, 0].set_xlabel("Frequency (MHz)")
        axes[0, 0].set_ylabel("Mean Power")
        axes[0, 0].legend(fontsize=8)
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(freqs_f0, np.mean(np.abs(b_gpu_a - b_cpu_a), axis=0), color="#e74c3c")
        axes[0, 1].set_title(f"Run A Residual |B_GPU - B_CPU| (Rel RMS = {rel_rms_err_a:.4e})")
        axes[0, 1].set_xlabel("Frequency (MHz)")
        axes[0, 1].set_ylabel("Mean Absolute Residual")
        axes[0, 1].grid(True, alpha=0.3)

    if alive_b_f0 and beams_b_f0 and len(alive_extract_elems) > 0:
        (hb_raw, mm_raw_b), (hb_bm, mm_bm_b), aligned_b = match_frames(alive_b_f0, beams_b_f0)
        ts_b = hb_raw.recv_time_sec if hb_raw.recv_time_sec > 0.0 else utc_ref
        if bench_b["tracker_status_records"] and hb_raw.recv_time_sec <= 0.0:
            ts_b = float(bench_b["tracker_status_records"][0].get("last_timestamp_utc", utc_ref))

        b_gpu_b, b_cpu_b, max_abs_err_b, rel_rms_err_b = run_cpu_golden_replay(
            raw_u8_frame=mm_raw_b.reshape(SAMPLES_PER_FRAME, NUM_LOCAL_FREQ, MAX_ALIVE_ANTENNAS),
            beam_gpu_frame=np.frombuffer(mm_bm_b, dtype=np.complex64).reshape(SAMPLES_PER_FRAME, NUM_LOCAL_FREQ),
            active_raw_elems=alive_extract_elems,
            full_64_mask=mask_b_vec,
            utc_ts=ts_b,
            freq_start_hz=300.0e6,
            swapped_with_offset=swapped_b,
        )
        print(f"CPU Golden Replay (Run B, {len(alive_extract_elems)}-ant alive): seq={hb_raw.fpga_seq_num}, rel_rms_err={rel_rms_err_b:.4e}, max_abs_err={max_abs_err_b:.4e}")

        axes[1, 0].plot(freqs_f0, np.mean(np.abs(b_gpu_b) ** 2, axis=0), label="GPU Formed Beam <|B_GPU|^2>", lw=1.5)
        axes[1, 0].plot(freqs_f0, np.mean(np.abs(b_cpu_b) ** 2, axis=0), linestyle="--", label="CPU Golden Replay <|B_CPU|^2>", lw=1.2)
        axes[1, 0].set_title(f"Run B ({len(alive_extract_elems)}-Ant Compacted Alive) CPU Golden Replay vs. GPU (seq={hb_raw.fpga_seq_num})")
        axes[1, 0].set_xlabel("Frequency (MHz)")
        axes[1, 0].set_ylabel("Mean Power")
        axes[1, 0].legend(fontsize=8)
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(freqs_f0, np.mean(np.abs(b_gpu_b - b_cpu_b), axis=0), color="#2ecc71")
        axes[1, 1].set_title(f"Run B Residual |B_GPU - B_CPU| (Rel RMS = {rel_rms_err_b:.4e})")
        axes[1, 1].set_xlabel("Frequency (MHz)")
        axes[1, 1].set_ylabel("Mean Absolute Residual")
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.out_dir / "fig8_cpu_golden_replay_residual.png", dpi=160)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Write Summary JSON & Markdown Report
    # -------------------------------------------------------------------------
    summary = {
        "target": {
            "name": "Grus Quartet (NGC 7582)",
            "ra_deg": GRUS_RA_DEG,
            "dec_deg": GRUS_DEC_DEG,
            "utc_timestamp": utc_ref,
            "lst_hours": lst_hr,
            "alt_deg": astropy_alt,
            "az_deg": astropy_az,
            "cpp_lmn": [cpp_l, cpp_m, cpp_n],
            "astropy_lm": [astropy_l, astropy_m],
            "logged_lmn": [logged_l, logged_m, logged_n],
        },
        "mask_verification": {
            "python_alive_raw_elements": py_alive_raw_elems,
            "python_alive_physical_antennas": py_alive_phys_ants,
            "cuda_alive_raw_elements": cuda_alive_raw_elems,
            "cuda_alive_physical_antennas": cuda_alive_phys_ants,
            "exact_match": (sorted(py_alive_raw_elems) == sorted(cuda_alive_raw_elems)) if cuda_alive_raw_elems else True,
        },
        "benchmarks": {
            "run_A": bench_a,
            "run_B": bench_b,
        },
        "physics": {
            "run_A_mean_beam_power_s0": float(np.mean(pt_a0)) if pt_a0 is not None else None,
            "run_B_mean_beam_power_s0": float(np.mean(pt_b0)) if pt_b0 is not None else None,
            "median_eta_s0": float(np.median(eta_f0)) if eta_f0 is not None else None,
            "median_eta_s1": float(np.median(eta_f1)) if eta_f1 is not None else None,
            "cpu_replay_rel_rms_err_runA": rel_rms_err_a,
            "cpu_replay_rel_rms_err_runB": rel_rms_err_b,
        },
    }
    (args.out_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Analysis complete! Summary and figures saved to {args.out_dir}")


if __name__ == "__main__":
    main()
