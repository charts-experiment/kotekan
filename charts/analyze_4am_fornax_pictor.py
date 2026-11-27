#!/usr/bin/env python3
"""
Posterior Benchmark, System Telemetry & Multi-Beam Physics Analysis
for the Extended 4-Beam Run until 04:00 AM CLST:
  - Beam 0: Fornax A (NGC 1316) West Lobe (RA=50.3583°, Dec=-37.2667°)
  - Beam 1: Fornax A (NGC 1316) East Lobe (RA=50.9750°, Dec=-37.1683°)
  - Beam 2: Pictor A (PKS 0518-45) West Lobe (RA=79.6000°, Dec=-45.7667°)
  - Beam 3: Pictor A (PKS 0518-45) East Lobe (RA=80.3000°, Dec=-45.7900°)
"""

import argparse
import json
import math
import os
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
except Exception:
    HAS_ASTROPY = False

CAREN_LAT_DEG = -33.4211146
CAREN_LON_DEG = -70.8634710
CAREN_ALT_M = 458.0

NUM_ELEMENTS = 64
NUM_LOCAL_FREQ = 336
SAMPLES_PER_FRAME = 15360
SAMPLE_PERIOD_S = 3.333333333333e-6
FRAME_DURATION_MS = SAMPLES_PER_FRAME * SAMPLE_PERIOD_S * 1e3  # 51.2 ms
MAX_BEAMS = 4
SPACING_M = 0.6
SPEED_OF_LIGHT = 299792458.0

BEAM_TARGETS = [
    {"id": 0, "name": "Fornax A West", "ra_deg": 50.3583, "dec_deg": -37.2667, "color": "#e74c3c"},
    {"id": 1, "name": "Fornax A East", "ra_deg": 50.9750, "dec_deg": -37.1683, "color": "#e67e22"},
    {"id": 2, "name": "Pictor A West", "ra_deg": 79.6000, "dec_deg": -45.7667, "color": "#2980b9"},
    {"id": 3, "name": "Pictor A East", "ra_deg": 80.3000, "dec_deg": -45.7900, "color": "#8e44ad"},
]


@dataclass
class FrameHeader:
    metadata_size: int
    fpga_seq_num: int
    time0_fpga: int
    recv_time_sec: float
    coarse_freq_idx: int
    rfsoc_id: int
    data_type: int
    data_offset: int
    file_mtime: float


def parse_single_frame_bin(path: Path, expected_frame_bytes: int) -> Optional[Tuple[FrameHeader, np.memmap]]:
    if not path.exists():
        return None
    file_size = path.stat().st_size
    if file_size < 4 + expected_frame_bytes:
        return None
    with open(path, "rb") as f:
        ms_bytes = f.read(4)
        (meta_size,) = struct.unpack("<I", ms_bytes)
        fpga_seq_num = -1
        time0_fpga = 0
        recv_sec = 0.0
        coarse_freq = 0
        rfsoc_id = 0
        dtype_val = 0
        if meta_size >= 48:
            mbuf = f.read(meta_size)
            fpga_seq_num, time0_fpga, tv_sec, tv_usec, coarse_freq, rfsoc_id, dtype_val = struct.unpack(
                "<qqqqHBI", mbuf[:39]
            )
            if tv_sec > 0:
                recv_sec = float(tv_sec) + float(tv_usec) * 1e-6
            elif time0_fpga > 1_000_000_000_000:
                recv_sec = float(time0_fpga) * 1e-6
    data_offset = 4 + meta_size
    if data_offset + expected_frame_bytes > file_size:
        return None
    mm = np.memmap(path, dtype=np.uint8, mode="r", offset=data_offset, shape=(expected_frame_bytes,))
    hdr = FrameHeader(
        metadata_size=meta_size,
        fpga_seq_num=fpga_seq_num,
        time0_fpga=time0_fpga,
        recv_time_sec=recv_sec,
        coarse_freq_idx=coarse_freq,
        rfsoc_id=rfsoc_id,
        data_type=dtype_val,
        data_offset=data_offset,
        file_mtime=path.stat().st_mtime,
    )
    return hdr, mm


def compute_cpp_astrometry(utc_timestamp: float, ra_deg: float, dec_deg: float) -> Tuple[float, float, float, float, float]:
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
    alt_deg = math.degrees(math.asin(max(-1.0, min(1.0, n))))
    return lst_hours, float(np.float32(l)), float(np.float32(m)), float(np.float32(n)), alt_deg


def parse_concat_json(p: Path) -> List[Dict[str, Any]]:
    objs: List[Dict[str, Any]] = []
    if not p.exists():
        return objs
    txt = p.read_text(errors="ignore")
    dec = json.JSONDecoder()
    pos = 0
    n = len(txt)
    while pos < n:
        m = re.search(r"\{", txt[pos:])
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


def main():
    parser = argparse.ArgumentParser(description="Analyze 4-Beam Extended Run (Fornax A W/E & Pictor A W/E)")
    parser.add_argument("--dump-dir", type=Path, default=Path("/data/tracker/dump_4am"))
    parser.add_argument("--out-dir", type=Path, default=Path("/data/tracker/analysis_4am"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    telem_dir = args.dump_dir / "telemetry"

    print("================================================================================")
    print(" POSTERIOR ANALYSIS: 4-BEAM EXTENDED RUN (FORNAX A W/E & PICTOR A W/E)")
    print("================================================================================")

    frame_bytes = SAMPLES_PER_FRAME * NUM_LOCAL_FREQ * MAX_BEAMS * 8  # 165,150,720 bytes

    f0_files = sorted(args.dump_dir.glob("formed_beams_4beams_freq0_*.bin"))
    f1_files = sorted(args.dump_dir.glob("formed_beams_4beams_freq1_*.bin"))

    frames_f0 = []
    for p in f0_files:
        res = parse_single_frame_bin(p, frame_bytes)
        if res is not None:
            frames_f0.append((p.name, res[0], res[1]))

    frames_f1 = []
    for p in f1_files:
        res = parse_single_frame_bin(p, frame_bytes)
        if res is not None:
            frames_f1.append((p.name, res[0], res[1]))

    total_bytes_written = sum(p.stat().st_size for p in f0_files + f1_files if p.exists())
    print(f"Loaded Stream 0: {len(frames_f0)} spaced files, Stream 1: {len(frames_f1)} spaced files")
    print(f"Total disk volume written: {total_bytes_written:,} bytes ({total_bytes_written / 1e9:.3f} GB)")

    # Parse telemetry logs
    sys_records = parse_concat_json(telem_dir / "system_cpu_ram_gpu.jsonl")
    tracker_records = parse_concat_json(telem_dir / "direct_tracker_status.jsonl")
    mask_records = parse_concat_json(telem_dir / "antenna_mask_status.jsonl")
    profile0_records = parse_concat_json(telem_dir / "gpu_profile_0.jsonl")
    profile1_records = parse_concat_json(telem_dir / "gpu_profile_1.jsonl")
    trackers_cur_records = parse_concat_json(telem_dir / "trackers_current.jsonl")

    # Also parse Prometheus metrics & buffers snapshots
    rawwrite_s0_ms = []
    rawwrite_s1_ms = []
    dpdk_rx_gbps_p0 = []
    dpdk_rx_gbps_p1 = []
    peak_buf_occ: Dict[str, int] = {}

    if telem_dir.exists():
        for prom_file in sorted(telem_dir.glob("metrics_*.prom")):
            txt = prom_file.read_text(errors="ignore")
            for line in txt.splitlines():
                if line.startswith("#") or not line.strip():
                    continue
                if 'kotekan_rawfilewrite_write_time_seconds{stage_name="/raw_file_write_beams_0"}' in line:
                    parts = line.split()
                    if len(parts) >= 2 and float(parts[1]) > 0:
                        rawwrite_s0_ms.append(float(parts[1]) * 1e3)
                elif 'kotekan_rawfilewrite_write_time_seconds{stage_name="/raw_file_write_beams_1"}' in line:
                    parts = line.split()
                    if len(parts) >= 2 and float(parts[1]) > 0:
                        rawwrite_s1_ms.append(float(parts[1]) * 1e3)

        for buf_file in sorted(telem_dir.glob("buffers_*.json")):
            try:
                bdata = json.loads(buf_file.read_text(errors="ignore"))
                if isinstance(bdata, dict):
                    for k, v in bdata.items():
                        if isinstance(v, dict):
                            nf = v.get("num_full_frame", v.get("num_full_frames", 0))
                            peak_buf_occ[k] = max(peak_buf_occ.get(k, 0), int(nf))
            except Exception:
                pass

    # Extract CUDA kernel timings from gpu_profile_0 / gpu_profile_1 or trackers_current
    cuda_stage_times_ms: Dict[str, List[float]] = {}
    for label_prefix, prof_list in [("GPU0", profile0_records), ("GPU1", profile1_records)]:
        for rec in prof_list:
            for k, v in rec.items():
                if "time" not in k:
                    continue
                if isinstance(v, (int, float)) and v > 0:
                    cuda_stage_times_ms.setdefault(f"{label_prefix}:{k}", []).append(float(v) * 1e3)
                elif isinstance(v, dict):
                    val = v.get("average", v.get("last", v.get("value", 0.0)))
                    if isinstance(val, (int, float)) and val > 0:
                        cuda_stage_times_ms.setdefault(f"{label_prefix}:{k}", []).append(float(val) * 1e3)

    for rec in trackers_cur_records:
        for stage_k, stage_dict in rec.items():
            if isinstance(stage_dict, dict):
                for metric_k, metric_obj in stage_dict.items():
                    if isinstance(metric_obj, dict) and "execute_time" in metric_k:
                        val = metric_obj.get("avg", metric_obj.get("current", 0.0))
                        if isinstance(val, (int, float)) and val > 0:
                            short_s = stage_k.replace("/gpu/", "")
                            cuda_stage_times_ms.setdefault(f"{short_s}:{metric_k}", []).append(float(val) * 1e3)
                    elif isinstance(metric_obj, dict) and "direct_tracker_exec_time" in metric_k:
                        val = metric_obj.get("avg", metric_obj.get("current", 0.0))
                        if isinstance(val, (int, float)) and val > 0:
                            short_s = stage_k.replace("/gpu/", "")
                            cuda_stage_times_ms.setdefault(f"{short_s}:direct_tracker_ms", []).append(float(val))

    # -------------------------------------------------------------------------
    # Figure 0: 4-Hour GPU, CPU, RAM, NVMe & CUDA Kernel Telemetry
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Panel 1: CUDA Kernel & Stage Execution Time vs. 51.2 ms Frame Budget
    ax = axes[0, 0]
    if cuda_stage_times_ms:
        names = sorted(cuda_stage_times_ms.keys())
        means = [float(np.mean(cuda_stage_times_ms[k])) for k in names]
        maxs = [float(np.max(cuda_stage_times_ms[k])) for k in names]
        x_p = np.arange(len(names))
        ax.bar(x_p - 0.18, means, width=0.35, label="Mean Time (ms)", color="#2ecc71")
        ax.bar(x_p + 0.18, maxs, width=0.35, label="Max Time (ms)", color="#3498db")
        ax.set_xticks(x_p)
        ax.set_xticklabels([n[-30:] for n in names], rotation=25, ha="right", fontsize=8)
    ax.axhline(FRAME_DURATION_MS, color="red", linestyle="--", label=f"51.2 ms Frame Budget")
    ax.set_ylabel("Execution Time (ms)")
    ax.set_title("4-Beam CUDA Hardware Event Timings vs. 51.2 ms Real-Time Budget")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: RTX 5090 GPU Telemetry Over Time
    ax = axes[0, 1]
    if sys_records:
        t0_sys = float(sys_records[0].get("utc_ts", 0.0))
        t_hrs = [(float(r.get("utc_ts", t0_sys)) - t0_sys) / 3600.0 for r in sys_records]
        gpu_sm = [float(r.get("gpu_sm_pct", 0.0)) for r in sys_records]
        gpu_mem = [float(r.get("gpu_mem_bw_pct", 0.0)) for r in sys_records]
        gpu_pwr = [float(r.get("gpu_power_w", 0.0)) for r in sys_records]
        gpu_temp = [float(r.get("gpu_temp_c", 0.0)) for r in sys_records]
        ax.plot(t_hrs, gpu_sm, label="GPU SM Util (%)", lw=1.5, color="#2ecc71")
        ax.plot(t_hrs, gpu_mem, label="GPU Mem BW (%)", lw=1.5, color="#3498db")
        ax.plot(t_hrs, gpu_temp, label="GPU Temp (°C)", lw=1.5, color="#e67e22")
        ax.plot(t_hrs, [p / 5.0 for p in gpu_pwr], label="GPU Power / 5 (W/5)", lw=1.5, color="#e74c3c")
    ax.set_xlabel("Elapsed Observation Time (hours)")
    ax.set_ylabel("Percentage / °C / (W/5)")
    ax.set_title("RTX 5090 GPU Utilization, Temperature & Power Over Extended Run")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Host CPU & RAM Utilization Over Time
    ax = axes[1, 0]
    if sys_records:
        t0_sys = float(sys_records[0].get("utc_ts", 0.0))
        t_hrs = [(float(r.get("utc_ts", t0_sys)) - t0_sys) / 3600.0 for r in sys_records]
        cpu_sys = [float(r.get("system_cpu_pct", 0.0)) for r in sys_records]
        kot_rss_gb = [float(r.get("kotekan_rss_gb", 0.0)) for r in sys_records]
        ram_used_gb = [float(r.get("system_ram_used_gb", 0.0)) for r in sys_records]
        ax.plot(t_hrs, cpu_sys, label="Host CPU Util (%)", lw=1.5, color="#9b59b6")
        ax.plot(t_hrs, kot_rss_gb, label="Kotekan Process RSS (GB)", lw=1.5, color="#2980b9")
        ax.plot(t_hrs, ram_used_gb, label="System RAM Used (GB)", lw=1.5, color="#16a085")
    ax.set_xlabel("Elapsed Observation Time (hours)")
    ax.set_ylabel("CPU (%) / Memory (GB)")
    ax.set_title("Host CPU, System RAM & Kotekan RSS Over Extended Run")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 4: Spaced Disk Writes Cumulative Volume & Write Latency
    ax = axes[1, 1]
    if frames_f0:
        t0_f = frames_f0[0][1].file_mtime
        dump_hrs = [(hdr.file_mtime - t0_f) / 3600.0 for _, hdr, _ in frames_f0]
        cum_gb = [(i + 1) * (2.0 * frame_bytes) / 1e9 for i in range(len(frames_f0))]
        ax.step(dump_hrs, cum_gb, where="post", marker="o", lw=2.0, color="#2c3e50", label="Cumulative Disk Written (GB)")
        for i, (dh, cg) in enumerate(zip(dump_hrs, cum_gb)):
            ax.annotate(f"#{i}\n{cg:.2f}GB", (dh, cg), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=7)
    ax.axhline(3.0, color="red", linestyle="--", label="Target Budget (~3.0 GB)")
    ax.set_xlabel("Elapsed Observation Time (hours)")
    ax.set_ylabel("Cumulative Formed Beams Written (GB)")
    ax.set_title(f"Spaced Disk Recording Schedule ({len(frames_f0)} Epochs = {total_bytes_written/1e9:.3f} GB)")
    ax.set_ylim(0, 3.6)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.out_dir / "fig0_4am_system_and_kernel_telemetry.png", dpi=160)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Figure 1: Automatic 5-Live-Antenna Masking Stability Over 4 Hours
    # -------------------------------------------------------------------------
    valid_mask_recs = [
        r for r in mask_records
        if r.get("antennas") and max(float(a.get("passband_power", 0.0)) for a in r["antennas"]) > 0.1
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    live_raw_ids = [23, 26, 31, 58, 63]
    live_labels = {
        63: "C0 (pos 0, raw 63)",
        58: "C1 (pos 5, raw 58)",
        31: "C6 (pos 32, raw 31)",
        26: "C7 (pos 37, raw 26)",
        23: "CHORD (pos 40, raw 23)",
    }

    if valid_mask_recs:
        x_idx = np.arange(len(valid_mask_recs))
        for r_id in live_raw_ids:
            pbs = [float(rec["antennas"][r_id].get("passband_power", 0.0)) for rec in valid_mask_recs]
            cohs = [float(rec["antennas"][r_id].get("coherence", 0.0)) for rec in valid_mask_recs]
            axes[0, 0].plot(x_idx, pbs, marker=".", markersize=4, label=live_labels[r_id])
            axes[0, 1].plot(x_idx, cohs, marker=".", markersize=4, label=live_labels[r_id])

        # Also plot median of the 59 masked inputs
        dead_pbs = [
            float(np.median([rec["antennas"][a].get("passband_power", 0.0) for a in range(64) if a not in live_raw_ids]))
            for rec in valid_mask_recs
        ]
        dead_cohs = [
            float(np.median([rec["antennas"][a].get("coherence", 0.0) for a in range(64) if a not in live_raw_ids]))
            for rec in valid_mask_recs
        ]
        axes[0, 0].plot(x_idx, dead_pbs, color="gray", linestyle="--", label="59 Unconnected Inputs (Median)")
        axes[0, 1].plot(x_idx, dead_cohs, color="gray", linestyle="--", label="59 Unconnected Inputs (Median)")
        axes[0, 1].plot(x_idx, [1.15 * c for c in dead_cohs], color="red", linestyle=":", label="1.15x Median Coherence Threshold")

        axes[0, 0].set_xlabel("Telemetry Snapshot Index")
        axes[0, 0].set_ylabel("75th-Pct In-Band Passband Power (ADC^2)")
        axes[0, 0].set_title("Passband Power Stability of 5 Auto-Detected Live Antennas")
        axes[0, 0].legend(fontsize=8)
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].set_xlabel("Telemetry Snapshot Index")
        axes[0, 1].set_ylabel("Cross-Antenna Spatial Coherence")
        axes[0, 1].set_title("Spatial Coherence Separation Over Extended Run")
        axes[0, 1].legend(fontsize=8)
        axes[0, 1].grid(True, alpha=0.3)

        # Panel 3: Last snapshot 2D scatter
        last_rec = valid_mask_recs[-1]
        for a in range(64):
            m = last_rec["antennas"][a]
            is_act = bool(m.get("active", False))
            pb_v = float(m.get("passband_power", 0.0))
            ch_v = float(m.get("coherence", 0.0))
            axes[1, 0].scatter(
                pb_v, ch_v,
                c="#2ecc71" if is_act else "#e74c3c",
                s=70 if is_act else 25,
                edgecolors="black" if is_act else "none",
                alpha=0.9 if is_act else 0.5,
            )
            if is_act:
                axes[1, 0].annotate(f"Ant {63-a} (raw {a})", (pb_v, ch_v), xytext=(5, 4), textcoords="offset points", fontsize=8, fontweight="bold")
        axes[1, 0].set_xlabel("In-Band Passband Power (ADC^2)")
        axes[1, 0].set_ylabel("Mean Spatial Coherence")
        axes[1, 0].set_title(f"Final Snapshot 2D Separation (Active: {last_rec.get('active_physical_antennas')})")
        axes[1, 0].grid(True, alpha=0.3)

        # Panel 4: 8x8 Physical Array Map
        grid_p = np.zeros((8, 8))
        for a in range(64):
            pa = 63 - a
            grid_p[pa // 8, pa % 8] = float(last_rec["antennas"][a].get("passband_power", 0.0))
        im_g = axes[1, 1].imshow(grid_p, cmap="viridis", origin="lower")
        for r_i in range(8):
            for c_i in range(8):
                pa = r_i * 8 + c_i
                a = 63 - pa
                act = bool(last_rec["antennas"][a].get("active", False))
                axes[1, 1].text(c_i, r_i, f"A{pa}\n{grid_p[r_i, c_i]:.1f}\n({'OK' if act else 'X'})",
                                ha="center", va="center", color="white" if not act else "yellow", fontsize=7,
                                fontweight="bold" if act else "normal")
        plt.colorbar(im_g, ax=axes[1, 1], label="Passband Power (ADC^2)")
        axes[1, 1].set_title("8x8 Physical Array Layout (5 Live Feeds Automatically Locked)")
        axes[1, 1].set_xlabel("Grid Column (0..7)")
        axes[1, 1].set_ylabel("Grid Row (0..7)")

    plt.tight_layout()
    plt.savefig(args.out_dir / "fig1_4am_antenna_mask_stability.png", dpi=160)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Figure 2: 4-Beam Astrometric Trajectories & Transit Altitude Curve
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    if frames_f0:
        t_start = frames_f0[0][1].file_mtime
        t_end = frames_f0[-1][1].file_mtime
        if t_end <= t_start:
            t_end = t_start + 3.8 * 3600.0
    else:
        t_start = 1793330340.0
        t_end = t_start + 3.8 * 3600.0

    t_grid = np.linspace(t_start, max(t_end, t_start + 3.78 * 3600.0), 200)
    hrs_grid = (t_grid - t_start) / 3600.0

    for b_cfg in BEAM_TARGETS:
        alts = []
        ls = []
        ms = []
        for ts in t_grid:
            _, l_v, m_v, n_v, alt_v = compute_cpp_astrometry(ts, b_cfg["ra_deg"], b_cfg["dec_deg"])
            alts.append(alt_v)
            ls.append(l_v)
            ms.append(m_v)
        axes[0].plot(hrs_grid, alts, label=f"Beam {b_cfg['id']}: {b_cfg['name']} ({b_cfg['ra_deg']:.2f}°, {b_cfg['dec_deg']:.2f}°)", color=b_cfg["color"], lw=1.8)
        axes[1].plot(ls, ms, label=f"Beam {b_cfg['id']}: {b_cfg['name']}", color=b_cfg["color"], lw=1.8)
        axes[1].scatter([ls[0], ls[-1]], [ms[0], ms[-1]], color=b_cfg["color"], s=40, zorder=5)

    # Overlay actual dump timestamps
    if frames_f0:
        for i, (_, hdr, _) in enumerate(frames_f0):
            ts_d = hdr.file_mtime
            hr_d = (ts_d - t_start) / 3600.0
            axes[0].axvline(hr_d, color="gray", linestyle=":", alpha=0.6)
            axes[0].text(hr_d, 51.0, f"E{i}", rotation=90, va="bottom", ha="right", fontsize=7, color="#555555")
            for b_cfg in BEAM_TARGETS:
                _, l_d, m_d, _, _ = compute_cpp_astrometry(ts_d, b_cfg["ra_deg"], b_cfg["dec_deg"])
                axes[1].scatter([l_d], [m_d], color=b_cfg["color"], marker="x", s=40)

    axes[0].set_xlabel("Elapsed Time Since Start (hours) — Dotted Lines = Spaced File Writes")
    axes[0].set_ylabel("Elevation / Altitude (deg)")
    axes[0].set_title("4-Beam Elevation Trajectories (Fornax A Transit at 86.2° & Pictor A Rising to 76°)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Direction Cosine l (East)")
    axes[1].set_ylabel("Direction Cosine m (North)")
    axes[1].set_title("Topocentric Sky Trajectory (l, m) — Distinct Non-Overlapping W/E Lobe Offsets")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.out_dir / "fig2_4am_astrometry_and_transit_trajectories.png", dpi=160)
    plt.close(fig)

    # -------------------------------------------------------------------------
    # Figure 3 & Figure 4: 4-Beam Fullband Spectra (300-501.3 MHz) & Lightcurves
    # -------------------------------------------------------------------------
    freqs_s0 = 300.0 + np.arange(NUM_LOCAL_FREQ) * 0.3
    freqs_s1 = 400.8 + np.arange(NUM_LOCAL_FREQ) * 0.3

    # Compute per-epoch, per-beam spectrum and band-integrated power
    n_epochs = min(len(frames_f0), len(frames_f1)) if frames_f1 else len(frames_f0)
    beam_power_s0 = np.zeros((max(1, len(frames_f0)), MAX_BEAMS))
    beam_power_s1 = np.zeros((max(1, len(frames_f1)), MAX_BEAMS))
    epoch_times_hr = []

    mean_psd_s0 = np.zeros((MAX_BEAMS, NUM_LOCAL_FREQ))
    mean_psd_s1 = np.zeros((MAX_BEAMS, NUM_LOCAL_FREQ))

    for i, (fname, hdr, mm) in enumerate(frames_f0):
        ts_d = hdr.file_mtime
        epoch_times_hr.append((ts_d - t_start) / 3600.0)
        # Layout: [time=15360][freq=336][beam=4] of complex64
        b_c64 = np.frombuffer(mm, dtype=np.complex64).reshape(SAMPLES_PER_FRAME, NUM_LOCAL_FREQ, MAX_BEAMS)
        pwr = np.mean(np.abs(b_c64) ** 2, axis=0)  # (336, 4)
        for b in range(MAX_BEAMS):
            spec_b = pwr[:, b]
            mean_psd_s0[b] += spec_b / len(frames_f0)
            med_s = np.median(spec_b)
            mad_s = np.median(np.abs(spec_b - med_s)) + 1e-6
            clean = np.abs(spec_b - med_s) < 5.0 * mad_s
            beam_power_s0[i, b] = float(np.mean(spec_b[clean]))

    for i, (fname, hdr, mm) in enumerate(frames_f1):
        b_c64 = np.frombuffer(mm, dtype=np.complex64).reshape(SAMPLES_PER_FRAME, NUM_LOCAL_FREQ, MAX_BEAMS)
        pwr = np.mean(np.abs(b_c64) ** 2, axis=0)  # (336, 4)
        for b in range(MAX_BEAMS):
            spec_b = pwr[:, b]
            mean_psd_s1[b] += spec_b / len(frames_f1)
            med_s = np.median(spec_b)
            mad_s = np.median(np.abs(spec_b - med_s)) + 1e-6
            clean = np.abs(spec_b - med_s) < 5.0 * mad_s
            beam_power_s1[i, b] = float(np.mean(spec_b[clean]))

    # Figure 3: Fullband PSD & Multi-Epoch Waterfall for the 4 Beams
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for b_cfg in BEAM_TARGETS:
        b = b_cfg["id"]
        if frames_f0:
            axes[0, 0].plot(freqs_s0, 10.0 * np.log10(mean_psd_s0[b] + 1e-12), label=f"Beam {b}: {b_cfg['name']}", color=b_cfg["color"], lw=1.2)
        if frames_f1:
            axes[0, 1].plot(freqs_s1, 10.0 * np.log10(mean_psd_s1[b] + 1e-12), label=f"Beam {b}: {b_cfg['name']}", color=b_cfg["color"], lw=1.2)

    axes[0, 0].set_title("4-Beam Mean Formed-Beam PSD — Stream 0 (300.0–400.5 MHz, In-Band)")
    axes[0, 0].set_xlabel("Frequency (MHz)")
    axes[0, 0].set_ylabel("Power (dB)")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_title("4-Beam Mean Formed-Beam PSD — Stream 1 (400.8–501.3 MHz)")
    axes[0, 1].set_xlabel("Frequency (MHz)")
    axes[0, 1].set_ylabel("Power (dB)")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    # Bottom panels: Waterfall of Beam 0 (Fornax A West) and Beam 2 (Pictor A West) in Stream 0
    if frames_f0:
        wf_b0 = []
        wf_b2 = []
        for _, _, mm in frames_f0:
            b_c64 = np.frombuffer(mm, dtype=np.complex64).reshape(SAMPLES_PER_FRAME, NUM_LOCAL_FREQ, MAX_BEAMS)
            p_b0 = (np.abs(b_c64[:, :, 0]) ** 2).reshape(60, 256, NUM_LOCAL_FREQ).mean(axis=1)
            p_b2 = (np.abs(b_c64[:, :, 2]) ** 2).reshape(60, 256, NUM_LOCAL_FREQ).mean(axis=1)
            wf_b0.append(p_b0)
            wf_b2.append(p_b2)
        im_0 = axes[1, 0].imshow(10.0 * np.log10(np.vstack(wf_b0) + 1e-12), aspect="auto", origin="lower",
                                 extent=[freqs_s0[0], freqs_s0[-1], 0, len(frames_f0)], cmap="viridis")
        plt.colorbar(im_0, ax=axes[1, 0], label="Power (dB)")
        axes[1, 0].set_title(f"Beam 0 (Fornax A West) Stacked Spaced Waterfall ({len(frames_f0)} Epochs, Stream 0)")
        axes[1, 0].set_xlabel("Frequency (MHz)")
        axes[1, 0].set_ylabel("Spaced Dump Epoch Index")

        im_2 = axes[1, 1].imshow(10.0 * np.log10(np.vstack(wf_b2) + 1e-12), aspect="auto", origin="lower",
                                 extent=[freqs_s0[0], freqs_s0[-1], 0, len(frames_f0)], cmap="plasma")
        plt.colorbar(im_2, ax=axes[1, 1], label="Power (dB)")
        axes[1, 1].set_title(f"Beam 2 (Pictor A West) Stacked Spaced Waterfall ({len(frames_f0)} Epochs, Stream 0)")
        axes[1, 1].set_xlabel("Frequency (MHz)")
        axes[1, 1].set_ylabel("Spaced Dump Epoch Index")

    plt.tight_layout()
    plt.savefig(args.out_dir / "fig3_4am_4beam_spectra_and_waterfalls.png", dpi=160)
    plt.close(fig)

    # Figure 4: Transit Lightcurves (S0 & S1), West-East Lobe Differential & 4x4 Coherence
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    if frames_f0:
        x_t = epoch_times_hr if len(epoch_times_hr) == len(frames_f0) else list(range(len(frames_f0)))
        for b_cfg in BEAM_TARGETS:
            b = b_cfg["id"]
            axes[0, 0].plot(x_t, beam_power_s0[:, b], marker="o", lw=1.8, label=f"Beam {b}: {b_cfg['name']}", color=b_cfg["color"])
            if frames_f1 and len(frames_f1) == len(frames_f0):
                axes[0, 1].plot(x_t, beam_power_s1[:, b], marker="s", lw=1.8, label=f"Beam {b}: {b_cfg['name']}", color=b_cfg["color"])

        # Panel 3: West minus East Lobe Differential across epochs
        diff_fornax_s0 = beam_power_s0[:, 0] - beam_power_s0[:, 1]
        diff_pictor_s0 = beam_power_s0[:, 2] - beam_power_s0[:, 3]
        axes[1, 0].plot(x_t, diff_fornax_s0, marker="o", lw=1.8, color="#e74c3c", label="Fornax A (West − East, Stream 0)")
        axes[1, 0].plot(x_t, diff_pictor_s0, marker="o", lw=1.8, color="#2980b9", label="Pictor A (West − East, Stream 0)")
        if frames_f1 and len(frames_f1) == len(frames_f0):
            diff_fornax_s1 = beam_power_s1[:, 0] - beam_power_s1[:, 1]
            diff_pictor_s1 = beam_power_s1[:, 2] - beam_power_s1[:, 3]
            axes[1, 0].plot(x_t, diff_fornax_s1, marker="s", linestyle="--", lw=1.5, color="#e67e22", label="Fornax A (West − East, Stream 1)")
            axes[1, 0].plot(x_t, diff_pictor_s1, marker="s", linestyle="--", lw=1.5, color="#8e44ad", label="Pictor A (West − East, Stream 1)")
        axes[1, 0].axhline(0.0, color="gray", linestyle=":")
        axes[1, 0].set_xlabel("Elapsed Observation Time (hours)")
        axes[1, 0].set_ylabel("Lobe Power Differential P(West) − P(East)")
        axes[1, 0].set_title("West vs. East Radio Lobe Drift Differential Across Meridian Transit")
        axes[1, 0].legend(fontsize=8)
        axes[1, 0].grid(True, alpha=0.3)

        # Panel 4: Cross-beam correlation / non-overlap check on latest frame
        _, _, mm_last = frames_f0[-1]
        b_last = np.frombuffer(mm_last, dtype=np.complex64).reshape(-1, MAX_BEAMS)
        gram = np.abs((b_last.T @ np.conj(b_last)) / b_last.shape[0])
        diag = np.sqrt(np.real(np.diag(gram)))
        coh_4x4 = gram / (np.outer(diag, diag) + 1e-12)
        im_c = axes[1, 1].imshow(coh_4x4, vmin=0, vmax=1, cmap="magma")
        plt.colorbar(im_c, ax=axes[1, 1], label="|R_bb'| Inter-Beam Coherence")
        b_names = [f"B{b['id']}: {b['name']}" for b in BEAM_TARGETS]
        axes[1, 1].set_xticks(range(MAX_BEAMS))
        axes[1, 1].set_yticks(range(MAX_BEAMS))
        axes[1, 1].set_xticklabels(b_names, rotation=25, ha="right", fontsize=8)
        axes[1, 1].set_yticklabels(b_names, fontsize=8)
        for i in range(MAX_BEAMS):
            for j in range(MAX_BEAMS):
                axes[1, 1].text(j, i, f"{coh_4x4[i, j]:.3f}", ha="center", va="center",
                                color="white" if coh_4x4[i, j] < 0.6 else "black", fontsize=9, fontweight="bold")
        axes[1, 1].set_title("4x4 Inter-Beam Cross-Coherence Matrix (Confirming Non-Identical Beams)")

    axes[0, 0].set_xlabel("Elapsed Observation Time (hours)")
    axes[0, 0].set_ylabel("Clean Band-Average Formed Beam Power <|B_b|^2>")
    axes[0, 0].set_title("4-Beam Band-Integrated Power Evolution — Stream 0 (300.0–400.5 MHz)")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_xlabel("Elapsed Observation Time (hours)")
    axes[0, 1].set_ylabel("Clean Band-Average Formed Beam Power <|B_b|^2>")
    axes[0, 1].set_title("4-Beam Band-Integrated Power Evolution — Stream 1 (400.8–501.3 MHz)")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.out_dir / "fig4_4am_transit_lightcurves_and_west_east_offset.png", dpi=160)
    plt.close(fig)

    # Save JSON summary
    summary = {
        "num_spaced_epochs_stream0": len(frames_f0),
        "num_spaced_epochs_stream1": len(frames_f1),
        "total_bytes_written": total_bytes_written,
        "total_gb_written": total_bytes_written / 1e9,
        "beams": BEAM_TARGETS,
        "cuda_stage_times_mean_ms": {k: float(np.mean(v)) for k, v in cuda_stage_times_ms.items() if v},
        "cuda_stage_times_max_ms": {k: float(np.max(v)) for k, v in cuda_stage_times_ms.items() if v},
        "nvme_write_mean_ms_s0": float(np.mean(rawwrite_s0_ms)) if rawwrite_s0_ms else None,
        "nvme_write_mean_ms_s1": float(np.mean(rawwrite_s1_ms)) if rawwrite_s1_ms else None,
        "peak_buffer_occupancy": peak_buf_occ,
        "beam_power_stream0_per_epoch": beam_power_s0.tolist() if frames_f0 else [],
        "beam_power_stream1_per_epoch": beam_power_s1.tolist() if frames_f1 else [],
    }
    (args.out_dir / "analysis_4am_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"4 AM Posterior Analysis complete! Figures and JSON written to {args.out_dir}")


if __name__ == "__main__":
    main()
