"""Step 3: two-shard GPU ring wrap and observed consumer backpressure.

No new Kotekan stage; an external FIFO reader stalls existing rawFileWrite.
File replay remains explicitly post-zeroSamples, as in steps 1–2.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import time
import unittest

from test_charts.sim.backpressure import FifoBackpressure
from test_charts.tests import test_packet_loss_gpu as replay


class FifoHelperTests(unittest.TestCase):
    def test_pipe_reader_preserves_records_and_obeys_release(self):
        import os
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            probe = FifoBackpressure(p, p / "received", "intensity", 0, 3, 32)
            errors = []
            def write():
                try:
                    for frame in range(3):
                        with probe.path(p, frame).open("wb") as pipe:
                            pipe.write(bytes([frame]) * 32)
                except Exception as error:
                    errors.append(error)
            probe.start()
            writer = threading.Thread(target=write, daemon=True)
            writer.start()
            try:
                time.sleep(0.03)
                self.assertEqual(probe.records, [])
                probe.release.set()
                writer.join(5)
                probe.finish()
                self.assertFalse(writer.is_alive())
                self.assertFalse(errors)
                for frame in range(3):
                    self.assertEqual(probe.path(p / "received", frame).read_bytes(), bytes([frame]) * 32)
            finally:
                probe.close()


class LifecycleGPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        replay.PacketLossGPU.setUpClass()

    _run_replay = replay.PacketLossGPU._run_replay

    def test_coarse_single_gpu_slot_intensity_backpressure(self):
        self._run_replay(mode="coarse", k=320, lifecycle={
            "frames": 48, "depths": [1, 2, 3, 1], "compute_stream": 2,
            "hold": ["intensity", 0]})

    def test_fine_rotating_slots_exposure_backpressure(self):
        self._run_replay(constant=True, lifecycle={
            "frames": 48, "depths": [3, 4, 5, 3], "compute_stream": 3,
            "hold": ["exposure", 1]})

    def test_fine_deeper_rings_intensity_backpressure(self):
        self._run_replay(constant=True, lifecycle={
            "frames": 56, "depths": [4, 6, 7, 5], "compute_stream": 2,
            "hold": ["intensity", 0]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--kotekan-bin", type=Path, default=replay.BINARY)
    args = parser.parse_args()
    replay.RESULTS, replay.BINARY = args.results_dir.resolve(), args.kotekan_bin.resolve()
    replay.RESULTS.mkdir(parents=True, exist_ok=True)
    suite = unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__, fromlist=["*"]))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    summary = {"run": result.testsRun, "passed": result.testsRun - len(result.failures)
               - len(result.errors) - len(result.skipped), "failures": len(result.failures),
               "errors": len(result.errors), "skipped": len(result.skipped),
               "failure_details": [(str(test), error) for test, error in result.failures + result.errors],
               "skips": [(str(test), why) for test, why in result.skipped],
               "binary": str(replay.BINARY), "binary_sha256": replay.sha256(replay.BINARY),
               "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=replay.ROOT, text=True).strip(),
               "scope": "post-zero two-shard lifecycle/backpressure; not a cadence benchmark"}
    for name, command in {"gpu": ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                          "cuda": ["/usr/local/cuda/bin/nvcc", "--version"]}.items():
        r = subprocess.run(command, capture_output=True, text=True)
        summary[name] = {"command": command, "exit_code": r.returncode, "output": r.stdout + r.stderr}
    (replay.RESULTS / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    raise SystemExit(0 if result.wasSuccessful() else 1)
