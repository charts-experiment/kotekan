"""Exposure counts byte-identical to the beamform-64 reference binary (FP16 plan WP4).

Run with ~/charts-utils/.venv/bin/python -m test_charts.tests.test_exposure_ab_gpu \
    --reference-bin PATH [--results-dir PATH] [--kotekan-bin PATH]   (PYTHONPATH = repo root)

The exposure product (valid supplied spectra per window and coarse channel, uint8
[windows][336]) depends only on the handler masks, the PFB support and the history/reset
logic, none of which the fp16 fine path changed. Both binaries replay the same two-shard
configuration (create_packet_loss_yaml of this branch), the same pre-zeroed voltages and the
same masks; every exposure record (raw metadata + payload) must be byte-identical. The
reference is a build of branch beamform-64 (int4 fine path); without --reference-bin the
test is skipped. Mask patterns: isolated missing spectra, bursts across window and frame
boundaries, a fully missing frame, first/last samples, seeded random loss, and a sequence
jump on shard 0 (history reset). Synthetic verification only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import yaml

from test_charts.sim.packet_loss import create_packet_loss_yaml, write_frame

BINARY = Path(__file__).resolve().parents[2] / "build-tc/kotekan/kotekan"
REFERENCE = None
RESULTS = None


def masks_for(nt: int, frames: int, shard: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed + shard)
    out = []
    for frame in range(frames):
        flags = np.zeros(nt, np.uint8)
        kind = frame % 6
        if kind == 1:
            flags[shard * 33] = 1                                  # isolated spectrum
        elif kind == 2:
            flags[100 + shard * 32:141 + shard * 32] = 1           # burst across windows
        elif kind == 3:
            flags[[0, nt - 1]] = 1                                 # frame boundaries
            flags[nt - 50:] = 1                                    # burst into the next frame
        elif kind == 4:
            flags[rng.random(nt) < 0.02] = 1                       # random loss
        elif kind == 5 and frame == 5:
            flags[:] = 1                                           # whole frame missing
        out.append(flags)
    return out


def run(binary: Path, work: Path, nt: int, frames: int, seed: int) -> dict:
    shutil.rmtree(work, ignore_errors=True)
    inp, out = work / "in", work / "out"
    inp.mkdir(parents=True)
    cfg_path = create_packet_loss_yaml(work / "run.yaml", inp, out, frames, raw_n_time=nt)
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["log_level"] = "warn"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    rng = np.random.default_rng(seed)
    for shard in (0, 1):
        for frame, flags in enumerate(masks_for(nt, frames, shard, seed)):
            jumps = int(shard == 0 and frame >= 7)                 # sequence jump: reset
            seq = 10000 + nt * (frame + jumps)
            raw = rng.integers(0, 256, (nt, 336, 64), dtype=np.uint8)
            raw[flags == 1] = 0
            write_frame(inp / f"voltage.shard{shard}_{frame:07d}.bin", raw, seq, shard)
            write_frame(inp / f"missing.shard{shard}_{frame:07d}.bin", flags, seq, shard,
                        mask=True, timestamp_step=1)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = dict(os.environ, PATH="/usr/bin:" + os.environ.get("PATH", ""))
    log = work / "kotekan.txt"
    with log.open("w") as fh:
        rc = subprocess.run([str(binary.resolve()), "-b", f"127.0.0.1:{port}", "-c",
                             str(cfg_path)], stdout=fh, stderr=subprocess.STDOUT, env=env,
                            timeout=300).returncode
    files = {p.name: p.read_bytes() for p in sorted(out.glob("exposure.shard*_*.bin"))}
    return {"rc": rc, "files": files, "log": log.read_text(errors="replace")}


class ExposureAB(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if REFERENCE is None:
            raise unittest.SkipTest("no --reference-bin (beamform-64 build) given")
        for b in (BINARY, REFERENCE):
            if not b.is_file():
                raise unittest.SkipTest(f"no kotekan binary at {b}")
        cls.tmp = tempfile.TemporaryDirectory(dir="/dev/shm" if Path("/dev/shm").is_dir() else None)
        cls.root = Path(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def check(self, name, nt, frames, seed):
        got = run(BINARY, self.root / f"{name}_fp16", nt, frames, seed)
        ref = run(REFERENCE, self.root / f"{name}_b64", nt, frames, seed)
        names = sorted(set(got["files"]) | set(ref["files"]))
        diff = [n for n in names if got["files"].get(n) != ref["files"].get(n)]
        payload_bytes = -(-(nt // 32) // 10) * 336                     # uint8 [windows][336]
        distinct = len({ref["files"][n][-payload_bytes:] for n in ref["files"]})
        m = {"T_raw": nt, "frames": frames, "rc": [got["rc"], ref["rc"]],
             "records": [len(got["files"]), len(ref["files"])],
             "mismatched_records": diff,
             "distinct_payloads": distinct,
             "sha256": hashlib.sha256(b"".join(ref["files"][n] for n in names
                                               if n in ref["files"])).hexdigest(),
             "reference_binary": str(REFERENCE)}
        if RESULTS:
            RESULTS.mkdir(parents=True, exist_ok=True)
            (RESULTS / f"ab_{name}.json").write_text(json.dumps(m, indent=2) + "\n")
        self.assertEqual(m["rc"], [0, 0], (got["log"][-2000:], ref["log"][-2000:]))
        self.assertEqual(m["records"], [2 * frames, 2 * frames], m)
        self.assertEqual(diff, [], m)
        self.assertGreaterEqual(distinct, 4, m)                   # masks change the counts

    def test_short_frames_with_reset(self):
        self.check("t672", 672, 12, seed=672)

    def test_production_frames(self):
        self.check("t15360", 15360, 6, seed=15360)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-bin", type=Path)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--kotekan-bin", type=Path)
    args, rest = parser.parse_known_args()
    if args.reference_bin:
        REFERENCE = args.reference_bin.resolve()
    if args.results_dir:
        RESULTS = args.results_dir.resolve()
    if args.kotekan_bin:
        BINARY = args.kotekan_bin.resolve()
    unittest.main(argv=[sys.argv[0], *rest])
