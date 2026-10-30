#!/usr/bin/env python3
"""
Spaced Telemetry Collector for the 4-Beam Extended Run until 04:00 AM CLST.
Logs GPU, CPU, RAM, DPDK, Ring Buffers, Antenna Mask, Direct Beam Tracker,
and Hardware CUDA Kernel Profiling metrics aligned with the spaced rawFileWrite dumps.
"""

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def fetch_url(url: str, timeout: float = 2.0) -> str:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""


def read_proc_stat() -> tuple[int, int]:
    try:
        for line in Path("/proc/stat").read_text().splitlines():
            if line.startswith("cpu "):
                parts = [int(x) for x in line.split()[1:]]
                idle = parts[3] + (parts[4] if len(parts) > 4 else 0)
                total = sum(parts)
                return idle, total
    except Exception:
        pass
    return 0, 1


def collect_host_and_gpu_snapshot(prev_cpu: tuple[int, int]) -> tuple[dict, tuple[int, int]]:
    now_utc = time.time()
    idle1, total1 = read_proc_stat()
    d_idle = max(0, idle1 - prev_cpu[0])
    d_tot = max(1, total1 - prev_cpu[1])
    sys_cpu_pct = 100.0 * (1.0 - float(d_idle) / float(d_tot))

    load1, load5, load15 = 0.0, 0.0, 0.0
    try:
        la = Path("/proc/loadavg").read_text().split()
        load1, load5, load15 = float(la[0]), float(la[1]), float(la[2])
    except Exception:
        pass

    mem_total_kb = 0
    mem_avail_kb = 0
    hp_total = 0
    hp_free = 0
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                mem_total_kb = int(line.split()[1])
            elif line.startswith("MemAvailable:"):
                mem_avail_kb = int(line.split()[1])
            elif line.startswith("HugePages_Total:"):
                hp_total = int(line.split()[1])
            elif line.startswith("HugePages_Free:"):
                hp_free = int(line.split()[1])
    except Exception:
        pass

    ram_used_gb = max(0.0, (mem_total_kb - mem_avail_kb) / (1024.0 * 1024.0))
    ram_total_gb = mem_total_kb / (1024.0 * 1024.0)

    # Kotekan process RSS & CPU%
    kot_cpu_pct = 0.0
    kot_rss_gb = 0.0
    kot_vsz_gb = 0.0
    try:
        out = subprocess.check_output(
            ["ps", "-C", "kotekan", "-o", "%cpu=,rss=,vsz="],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        for line in out.splitlines():
            p = line.split()
            if len(p) >= 3:
                kot_cpu_pct += float(p[0])
                kot_rss_gb += float(p[1]) / (1024.0 * 1024.0)
                kot_vsz_gb += float(p[2]) / (1024.0 * 1024.0)
    except Exception:
        pass

    # GPU query via nvidia-smi
    gpu_sm_pct = 0.0
    gpu_mem_bw_pct = 0.0
    gpu_vram_used_mb = 0.0
    gpu_vram_total_mb = 0.0
    gpu_power_w = 0.0
    gpu_temp_c = 0.0
    gpu_sm_clock_mhz = 0.0
    gpu_mem_clock_mhz = 0.0
    try:
        smi = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,clocks.current.sm,clocks.current.memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        first = smi.splitlines()[0].split(",")
        if len(first) >= 8:
            gpu_sm_pct = float(first[0].strip())
            gpu_mem_bw_pct = float(first[1].strip())
            gpu_vram_used_mb = float(first[2].strip())
            gpu_vram_total_mb = float(first[3].strip())
            gpu_power_w = float(first[4].strip())
            gpu_temp_c = float(first[5].strip())
            gpu_sm_clock_mhz = float(first[6].strip())
            gpu_mem_clock_mhz = float(first[7].strip())
    except Exception:
        pass

    snap = {
        "utc_ts": now_utc,
        "iso_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now_utc)),
        "system_cpu_pct": round(sys_cpu_pct, 2),
        "loadavg_1_5_15": [load1, load5, load15],
        "system_ram_used_gb": round(ram_used_gb, 3),
        "system_ram_total_gb": round(ram_total_gb, 3),
        "hugepages_used": hp_total - hp_free,
        "hugepages_total": hp_total,
        "kotekan_cpu_pct": round(kot_cpu_pct, 2),
        "kotekan_rss_gb": round(kot_rss_gb, 3),
        "kotekan_vsz_gb": round(kot_vsz_gb, 3),
        "gpu_sm_pct": gpu_sm_pct,
        "gpu_mem_bw_pct": gpu_mem_bw_pct,
        "gpu_vram_used_mb": gpu_vram_used_mb,
        "gpu_vram_total_mb": gpu_vram_total_mb,
        "gpu_power_w": gpu_power_w,
        "gpu_temp_c": gpu_temp_c,
        "gpu_sm_clock_mhz": gpu_sm_clock_mhz,
        "gpu_mem_clock_mhz": gpu_mem_clock_mhz,
    }
    return snap, (idle1, total1)


def poll_kotekan_endpoints(telem_dir: Path, tag: str, kotekan_url: str = "http://127.0.0.1:12048") -> bool:
    prom = fetch_url(f"{kotekan_url}/metrics")
    if not prom:
        return False

    (telem_dir / f"metrics_{tag}.prom").write_text(prom)

    bufs = fetch_url(f"{kotekan_url}/buffers")
    if bufs:
        (telem_dir / f"buffers_{tag}.json").write_text(bufs)

    for url_path, out_file in [
        ("/direct_tracker/status", "direct_tracker_status.jsonl"),
        ("/antenna_mask/status", "antenna_mask_status.jsonl"),
        ("/gpu_profile/gpu/gpu_0", "gpu_profile_0.jsonl"),
        ("/gpu_profile/gpu/gpu_1", "gpu_profile_1.jsonl"),
        ("/trackers_current", "trackers_current.jsonl"),
    ]:
        body = fetch_url(f"{kotekan_url}{url_path}")
        if body:
            with open(telem_dir / out_file, "a") as f:
                f.write(body.strip() + "\n")

    try:
        ctrl_out = subprocess.check_output(
            ["python3", "/home/fpalma/kotekan/test_charts/kotekan_tracker_control.py", "status"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5.0,
        )
        with open(telem_dir / "tracker_status.log", "a") as f:
            f.write(f"\n--- [{tag}] {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())} ---\n")
            f.write(ctrl_out)
    except Exception:
        pass
    return True


def main():
    dump_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/data/tracker/dump_4am")
    telem_dir = dump_dir / "telemetry"
    telem_dir.mkdir(parents=True, exist_ok=True)

    prev_cpu = read_proc_stat()
    time.sleep(0.5)

    seen_files: set[str] = set()
    start_ts = time.time()
    last_periodic_ts = 0.0
    snap_idx = 0

    while True:
        now = time.time()
        elapsed = now - start_ts

        # Check current .bin files in dump_dir
        cur_bins = {p.name for p in dump_dir.glob("formed_beams_4beams_freq*.bin") if p.stat().st_size >= 165_150_720}
        new_bins = cur_bins - seen_files

        # Trigger snapshot during initial warmup (5s, 12s, 25s), on every new spaced .bin dump, or every 120s
        is_warmup_poll = (elapsed < 35.0 and (now - last_periodic_ts) >= 5.0)
        is_spaced_dump = len(new_bins) > 0
        is_periodic_poll = (now - last_periodic_ts) >= 120.0

        if is_warmup_poll or is_spaced_dump or is_periodic_poll:
            tag = f"epoch_{len(cur_bins)//2:02d}_snap_{snap_idx:04d}"
            snap, prev_cpu = collect_host_and_gpu_snapshot(prev_cpu)
            snap["tag"] = tag
            snap["completed_bin_files"] = len(cur_bins)
            snap["trigger"] = "spaced_dump" if is_spaced_dump else ("warmup" if is_warmup_poll else "periodic")
            with open(telem_dir / "system_cpu_ram_gpu.jsonl", "a") as f:
                f.write(json.dumps(snap) + "\n")

            poll_kotekan_endpoints(telem_dir, tag)
            last_periodic_ts = now
            snap_idx += 1

            if is_spaced_dump:
                seen_files |= new_bins
                try:
                    subprocess.run(
                        [
                            "/home/fpalma/kotekan/.venv_analysis/bin/python3",
                            "/home/fpalma/kotekan/charts/analyze_4am_fornax_pictor.py",
                            "--dump-dir",
                            str(dump_dir),
                            "--out-dir",
                            "/data/tracker/analysis_4am",
                        ],
                        timeout=60.0,
                        check=False,
                    )
                except Exception:
                    pass

        # Stop if kotekan has exited (after initial 20s startup window)
        if elapsed > 20.0:
            try:
                subprocess.check_call(["pgrep", "-x", "kotekan"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except subprocess.CalledProcessError:
                try:
                    subprocess.run(
                        [
                            "/home/fpalma/kotekan/.venv_analysis/bin/python3",
                            "/home/fpalma/kotekan/charts/analyze_4am_fornax_pictor.py",
                            "--dump-dir",
                            str(dump_dir),
                            "--out-dir",
                            "/data/tracker/analysis_4am",
                        ],
                        timeout=60.0,
                        check=False,
                    )
                except Exception:
                    pass
                break

        time.sleep(2.0)


if __name__ == "__main__":
    main()
