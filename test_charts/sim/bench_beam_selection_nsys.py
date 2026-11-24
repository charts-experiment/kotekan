"""WP14 test 7: GPU cost of the int8 output vs the number of written beam groups (nsys replay).

Both shards in one Kotekan, unpaced rawFileRead at T_raw = 15360, int8 output with exported
parameters (as in the live configuration), legacy .bin weights (timing does not depend on the
values). One case per output_beam_groups setting, each under `nsys profile -t cuda`. Reports
per-kernel medians, every device-to-host copy grouped by size (intensity vs parameters), and
the arrival rate of the output files. Outputs go to /dev/shm and are deleted after each case.

Usage (repo root, PYTHONPATH=., GPU idle, no kotekan running):
  ~/miniconda3/envs/kotekan_test/bin/python -m test_charts.sim.bench_beam_selection_nsys \
      --work /dev/shm/wp14_bench --frames 24 \
      --case tag=old,bin=build-tc/kotekan/kotekan.f7d1968c5 \
      --case tag=g8,bin=build-tc/kotekan/kotekan \
      --case tag=g4,bin=build-tc/kotekan/kotekan,groups=0:1:2:3 \
      --case tag=g1,bin=build-tc/kotekan/kotekan,groups=0
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

import numpy as np
import yaml

from test_charts.tests.test_beam_selection_gpu import merge, shard_config

T_RAW, F, E, FF, K = 15360, 336, 64, 336 * 32, 10
N_OUT = (T_RAW // 32 + K - 1) // K
NSYS = "/usr/local/cuda-12.8/bin/nsys"
SHORT = {"tensor_core": "beamformer_tc", "upchannelizer": "upchannelizer",
         "quantize": "quantize_int8", "history_update": "history_update"}


def short_name(name: str) -> str:
    return next((v for k, v in SHORT.items() if k in name), name[:60])


def weights(work: Path) -> Path:
    path = work / "weights.bin"
    if not path.exists():
        rng = np.random.default_rng(2026_11_24)
        with open(path, "wb") as fh:
            for _ in range(128):
                w = (rng.standard_normal((FF, E), dtype=np.float32)
                     + 1j * rng.standard_normal((FF, E), dtype=np.float32)) / 8
                w.astype(np.complex64).tofile(fh)
    return path


def nsys_csv(rep: Path, report: str) -> list[dict]:
    r = subprocess.run([NSYS, "stats", "--report", report, "--format", "csv",
                        "--force-export=true", str(rep)], capture_output=True, text=True,
                       check=True)
    lines = r.stdout.splitlines()
    start = next((i for i, l in enumerate(lines) if "(ns)" in l), None)
    return list(csv.DictReader(io.StringIO("\n".join(lines[start:])))) if start is not None else []


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run_case(work: Path, case: dict, frames, wfile: Path, env: dict) -> dict:
    tag = case["tag"]
    cdir = work / tag
    shutil.rmtree(cdir, ignore_errors=True)
    cdir.mkdir(parents=True)
    groups = case["groups"]
    cfg = merge([shard_config(cdir, s, frames[s], t_raw=T_RAW, weights=wfile, depth=3,
                              groups=groups) for s in (0, 1)])
    (cdir / "run.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    rep = cdir / "nsys"
    t0 = time.time()
    r = subprocess.run([NSYS, "profile", "-t", "cuda", "--sample=none", "--cpuctxsw=none",
                        "--force-overwrite=true", "-o", str(rep), case["bin"], "-b",
                        f"127.0.0.1:{free_port()}", "-c", str(cdir / "run.yaml")],
                       env=env, capture_output=True, text=True, timeout=1800)
    wall = time.time() - t0
    (cdir / "kotekan.log").write_text(r.stdout + r.stderr)
    out = cdir / "out"
    res = {"case": case, "rc": r.returncode, "wall_s": wall, "kernels": {}, "d2h": {},
           "frames_out": {}, "arrival_ms": {}}
    for s in (0, 1):
        files = sorted(out.glob(f"intensity.shard{s}_*.bin"))
        res["frames_out"][s] = len(files)
        mt = np.array([p.stat().st_mtime for p in files])
        if mt.size > 3:
            res["arrival_ms"][s] = float(np.median(np.diff(mt[2:]))) * 1e3
        res.setdefault("intensity_file_bytes", files[0].stat().st_size if files else None)
    nrep = rep.with_suffix(".nsys-rep")
    for row in nsys_csv(nrep, "cuda_gpu_kern_sum"):
        res["kernels"][short_name(row["Name"])] = {
            "n": int(row["Instances"]), "med_ms": float(row["Med (ns)"]) / 1e6,
            "min_ms": float(row["Min (ns)"]) / 1e6, "max_ms": float(row["Max (ns)"]) / 1e6}
    sizes: dict[int, list[float]] = {}
    for row in nsys_csv(nrep, "cuda_gpu_trace"):
        if "Device-to-Host" not in row.get("Name", ""):
            continue
        nbytes = int(round(float(row["Bytes (MB)"]) * 1e6))
        sizes.setdefault(nbytes, []).append(float(row["Duration (ns)"]) / 1e6)
    for nbytes, d in sorted(sizes.items(), reverse=True):
        res["d2h"][str(nbytes)] = {"n": len(d), "med_ms": float(np.median(d)),
                                   "max_ms": float(np.max(d)),
                                   "GBps_med": nbytes / (np.median(d) * 1e-3) / 1e9}
    res["kernels_per_frame_med_ms"] = sum(v["med_ms"] for v in res["kernels"].values())
    shutil.rmtree(out, ignore_errors=True)               # keep only the report and the log
    return res


def parse_case(text: str) -> dict:
    kv = dict(item.split("=", 1) for item in text.split(","))
    groups = [int(g) for g in kv["groups"].split(":")] if "groups" in kv else None
    return {"tag": kv["tag"], "bin": str(Path(kv["bin"]).resolve()), "groups": groups}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--case", action="append", required=True, type=parse_case)
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--results", type=Path)
    a = ap.parse_args()
    a.work.mkdir(parents=True, exist_ok=True)
    wfile = weights(a.work)
    rng = np.random.default_rng(2026_11_20)
    frames = {s: [rng.integers(0, 256, (T_RAW, F, E), dtype=np.uint8) for _ in range(a.frames)]
              for s in (0, 1)}
    env = dict(os.environ, PATH=f"{Path(sys.executable).parent}:/usr/bin:{os.environ['PATH']}")
    results = {"frames": a.frames, "T_raw": T_RAW, "cases": {}}
    for case in a.case:
        res = run_case(a.work, case, frames, wfile, env)
        results["cases"][case["tag"]] = res
        g = case["groups"]
        print(f"== {case['tag']}: groups {g if g is not None else 'all (no key)'} rc {res['rc']} "
              f"frames {res['frames_out']} wall {res['wall_s']:.1f} s "
              f"arrival {res['arrival_ms']} ms")
        for name, v in res["kernels"].items():
            print(f"  {v['n']:>4} x med {v['med_ms']:7.3f}  min {v['min_ms']:7.3f}  "
                  f"max {v['max_ms']:7.3f} ms  {name}")
        for nbytes, v in res["d2h"].items():
            print(f"  {v['n']:>4} x med {v['med_ms']:7.3f}  max {v['max_ms']:7.3f} ms  "
                  f"D2H {int(nbytes):>10d} B  {v['GBps_med']:.1f} GB/s")
        print(f"  kernels per frame (sum of medians) {res['kernels_per_frame_med_ms']:.3f} ms")
    path = a.results or a.work / "bench_results.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"results -> {path}")
    sys.exit(1 if any(r["rc"] != 0 for r in results["cases"].values()) else 0)


if __name__ == "__main__":
    main()
