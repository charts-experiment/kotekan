"""Stage A Kotekan GPU replay with analytical fixtures, not a CPU voltage oracle.

Run with /usr/bin/python3 -m test_charts.tests.test_packet_loss_gpu --results-dir PATH.
The GPU tests execute the built Kotekan pipeline with pre-zeroed file fixtures.
The unchanged zeroSamples handoff has a separately ordered host integration test.
"""

from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
import unittest

import numpy as np
import yaml

from test_charts.sim.packet_loss import (
    NativeChartsMetadata, create_packet_loss_yaml, read_product, write_frame,
)

ROOT = Path(__file__).resolve().parents[2]
RESULTS = None
BINARY = ROOT / "build/kotekan/kotekan"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exposure_expectation(missing, previous_missing, *, mode, k, reset):
    """Analytical bad-sample interval union on the fine grid.

    A missing raw tick x affects m from ceil((x−31)/32) through floor((x+96)/32).
    This deliberately uses interval endpoints, independently of the GPU's tap loop.
    """
    if mode == "coarse":
        complete = missing == 0
    else:
        complete = np.ones(len(missing) // 32, dtype=bool)
        if reset:
            complete[:3] = False
            previous_missing = np.zeros(96, dtype=np.uint8)
        losses = np.concatenate((np.flatnonzero(previous_missing) - 96,
                                 np.flatnonzero(missing)))
        for x in losses:
            lo = -(-(int(x) - 31) // 32)
            hi = (int(x) + 96) // 32
            complete[max(0, lo):min(len(complete), hi + 1)] = False
    counts = np.array([np.count_nonzero(complete[t:t + k])
                       for t in range(0, len(complete), k)])
    return np.repeat(counts[:, None], 336, axis=1)


def constant_fine_power(raw_n_time, *, reset, amplitude=7, previous_amplitude=7):
    """Closed-form response of a constant source: three startup cases and steady state.

    Evaluate the documented 128-tap Fourier sum once per case. No arbitrary input
    processing, CPU FFT, packed-voltage decoder or CPU beamformer is used.
    All elements are identical; coherent beam 32 has power 64²·|packed Y|².
    """
    s = np.arange(128)
    q = (2 * s - 127) / 258
    h = np.cos(np.pi * q) ** 2 * np.sinc(4 * q) / 32
    phase = np.exp(-2j * np.pi * (np.arange(32)[:, None] - 15.5) * s / 32)
    if reset:
        previous_amplitude = 0
    # Two constant plateaus, one on either side of the frame boundary. This also
    # covers a fully missing (zeroed) frame and its following contaminated history.
    responses = np.array([amplitude * (phase[:, start:] @ h[start:])
                          + previous_amplitude * (phase[:, :start] @ h[:start])
                          for start in (96, 64, 32, 0)])
    components = np.concatenate((responses.real.ravel(), responses.imag.ravel()))
    half_margin = float(np.min(np.abs(components - (np.floor(components) + 0.5))))
    if half_margin < 0.02:
        raise AssertionError("constant fixture is too close to a packing boundary")
    real = np.clip(np.rint(responses.real), -7, 7)
    imag = np.clip(np.rint(responses.imag), -7, 7)
    power = 4096 * (real * real + imag * imag)
    indices = np.full(raw_n_time // 32, 3)
    indices[:3] = np.arange(3)
    spectrum_power = power[indices]
    integrated = np.array([spectrum_power[t:t + 10].sum(axis=0)
                           for t in range(0, len(indices), 10)])
    return np.tile(integrated[:, None, :], (1, 336, 1)).reshape(len(integrated), 10752), half_margin


class ReplayHelpers(unittest.TestCase):
    def test_descriptor_and_shared_mask_wiring(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            create_packet_loss_yaml(p / "run.yaml", p / "in", p / "out", 3,
                                    raw_n_time=672)
            cfg = yaml.safe_load((p / "run.yaml").read_text())
            descriptor = json.loads((p / "out/exposure.v1.json").read_text())
            self.assertEqual(descriptor["shape"], [3, 336])
            self.assertEqual(descriptor["intensity_shape"], [3, 10752, 64])
            self.assertEqual(descriptor["tail_supplied_spectra"], 1)
            for shard in (0, 1):
                self.assertNotIn(f"zero_{shard}", cfg)
                self.assertEqual(cfg[f"read_missing_{shard}"]["buf"],
                                 cfg[f"gpu_{shard}"]["in_buffers"]["missing"])
                commands = cfg[f"gpu_{shard}"]["commands"]
                copies = [c for c in commands if c["name"] == "cudaOutputData"]
                self.assertEqual(len(copies), 2)
                self.assertTrue(all(c["cuda_stream"] == 1 for c in copies))
                self.assertEqual(cfg[f"voltage_{shard}"]["num_frames"], 4)
                self.assertEqual(cfg[f"missing_{shard}"]["num_frames"], 5)
            for kwargs in ({"mask_depth": 2}, {"raw_n_time": 64}, {"k": 320}):
                with self.assertRaises(ValueError):
                    create_packet_loss_yaml(p / "bad.yaml", p, p, 1, **kwargs)

    def test_native_record_round_trip_and_wrong_shape_rejection(self):
        self.assertEqual(ct.sizeof(NativeChartsMetadata), 3032)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frame.bin"
            data = np.zeros(96, np.uint8)
            write_frame(path, data, 12345, 1)
            meta, product = read_product(path, "uint8", (96,))
            self.assertEqual(meta.fpga_seq_num, 12345)
            self.assertEqual(meta.timestamp_step, 1)
            self.assertEqual(list(meta.coarse_freq)[:336], list(range(336, 672)))
            np.testing.assert_array_equal(data, product)
            with self.assertRaises(ValueError):
                read_product(path, "uint8", (95,))

    def test_support_endpoints_and_count_width(self):
        missing = np.zeros(672, np.uint8)
        missing[0] = 1
        counts = exposure_expectation(missing, np.zeros(96), mode="fine", k=10, reset=False)
        np.testing.assert_array_equal(counts[:, 0], [6, 10, 1])
        missing[:] = 0
        previous = np.zeros(96)
        previous[-1] = 1
        counts = exposure_expectation(missing, previous, mode="fine", k=10, reset=False)
        np.testing.assert_array_equal(counts[:, 0], [7, 10, 1])
        counts = exposure_expectation(missing, previous, mode="coarse", k=320, reset=False)
        np.testing.assert_array_equal(counts[:, 0], [320, 320, 32])


class PacketLossGPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not BINARY.is_file():
            raise unittest.SkipTest(f"Kotekan binary missing: {BINARY}")
        probe = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
        if probe.returncode:
            raise unittest.SkipTest("GPU unavailable in this execution environment: " + probe.stderr)

    def _run_replay(self, *, mode="fine", k=10, constant=False, rejection=None,
                    lifecycle=None):
        case = self._testMethodName
        artifact = RESULTS / case if RESULTS else None
        if artifact:
            artifact.mkdir(parents=True, exist_ok=True)
        report = {"case": case, "mode": mode, "K": k, "T_raw": 672, "E": 64, "B": 64,
                  "status": "running", "frames": [], "input_sha256": {}}
        try:
            with tempfile.TemporaryDirectory(prefix="charts-stage-a-") as tmp:
                p = Path(tmp)
                inp, out = p / "in", p / "out"
                nt = 672
                frames = lifecycle["frames"] if lifecycle else (1 if rejection else (9 if constant else 12))
                depths = lifecycle["depths"] if lifecycle else (3, 4, 5, 4)
                cfg_path = create_packet_loss_yaml(p / "run.yaml", inp, out, frames,
                                                   mode=mode, k=k, raw_n_time=nt,
                                                   gpu_depth=depths[0], raw_depth=depths[1],
                                                   mask_depth=depths[2], output_depth=depths[3])
                cfg = yaml.safe_load(cfg_path.read_text())
                cfg["log_level"] = "warn"
                if lifecycle:
                    stream = lifecycle["compute_stream"]
                    for shard in (0, 1):
                        cfg[f"gpu_{shard}"]["num_cuda_streams"] = max(3, stream + 1)
                        for c in cfg[f"gpu_{shard}"]["commands"]:
                            if c["name"] == "cudaChartsBeamformerCommand":
                                c["cuda_stream"] = stream
                    report["lifecycle"] = lifecycle
                    report["wraps_per_shard"] = {"gpu": frames // depths[0],
                        "raw": frames // depths[1], "mask": frames // depths[2],
                        "output": frames // depths[3]}
                cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
                masks, sequences, amplitudes = {}, {}, {}
                for shard in (0, 1):
                    for frame in range(frames):
                        flags = np.zeros(nt, np.uint8)
                        if constant and (frame % 11 == 1 if lifecycle else frame == 1):
                            flags[:] = 1
                        if not constant:
                            if frame == 1:
                                flags[shard * 33] = 1
                            elif frame == 2:
                                flags[100 + shard * 32:141 + shard * 32] = 1
                            elif frame == 3 and shard == 0:
                                flags[-1] = 1
                            elif frame == 5:
                                flags[:] = 1
                            elif frame == 9:
                                flags[[0, nt - 1]] = 1
                        if lifecycle:
                            jumps = sum(frame >= f for f in range(7 + 8 * shard, frames, 19))
                        else:
                            jumps = int(shard == 0 and frame >= 7)
                        seq = 10000 + nt * (frame + jumps)
                        amplitude = ((2, 4, 7)[(frame + shard) % 3] if lifecycle and constant
                                     else (7 if constant else (1 + (frame + 3 * shard) % 7)))
                        raw = np.full((nt, 336, 64), amplitude if mode == "coarse" or constant
                                      else 0, np.uint8)
                        # File replay starts at the post-zeroSamples boundary. Independent
                        # readers cannot supply the live handler's paired close ordering.
                        raw[flags == 1] = 0
                        raw_path = inp / f"voltage.shard{shard}_{frame:07d}.bin"
                        mask_path = inp / f"missing.shard{shard}_{frame:07d}.bin"
                        write_frame(raw_path, raw, seq, shard)
                        if rejection == "nonbinary" and shard == 0:
                            flags[10] = 2
                        mask_seq = seq + (1 if rejection == "sequence" and shard == 0 else 0)
                        step = 2 if rejection == "timestamp" and shard == 0 else 1
                        write_frame(mask_path, flags, mask_seq, shard, mask=True, timestamp_step=step)
                        if rejection == "metadata" and shard == 0:
                            mask_path.write_bytes(b"\0\0\0\0" + flags.tobytes())
                        for path in (raw_path, mask_path):
                            report["input_sha256"][path.name] = sha256(path)
                        masks[shard, frame], sequences[shard, frame] = flags, seq
                        amplitudes[shard, frame] = amplitude
                with socket.socket() as sock:
                    sock.bind(("127.0.0.1", 0))
                    port = sock.getsockname()[1]
                command = [str(BINARY.resolve()), "-b", f"127.0.0.1:{port}", "-c", str(cfg_path)]
                report["command"] = command
                env = os.environ.copy()
                # Kotekan's YAML loader invokes python3. Use the same installed
                # interpreter/dependencies as this test rather than /usr/local/bin.
                env["PATH"] = "/usr/bin:" + env.get("PATH", "")
                env["LD_LIBRARY_PATH"] = str(BINARY.resolve().parents[1] / "external/n2k") + ":" + env.get("LD_LIBRARY_PATH", "")
                log_path = p / "kotekan.txt"
                slow = None
                if lifecycle:
                    from test_charts.sim.backpressure import FifoBackpressure
                    product, shard = lifecycle["hold"]
                    record_sizes = {name: cfg[f"{name}_{shard}"]["frame_size"] + 4
                                     + ct.sizeof(NativeChartsMetadata)
                                     for name in ("intensity", "exposure")}
                    slow = FifoBackpressure(out, p / "received", product, shard, frames,
                                             record_sizes[product])
                    slow.start()
                try:
                    with log_path.open("w") as log:
                        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
                        try:
                            if slow:
                                report["backpressure"] = slow.observe_stall(process, port, cfg, record_sizes)
                                slow.release.set()
                            process.wait(timeout=180 if lifecycle else 120)
                            if slow:
                                slow.finish()
                                report["backpressure"]["received_records"] = slow.records
                            result = process
                        finally:
                            if slow:
                                slow.release.set()
                            if process.poll() is None:
                                process.kill()
                                process.wait(timeout=10)
                finally:
                    if slow:
                        slow.close()
                    if artifact:
                        shutil.copyfile(cfg_path, artifact / "run.yaml")
                        shutil.copyfile(log_path, artifact / "kotekan.txt")
                        shutil.copyfile(out / "exposure.v1.json", artifact / "exposure.v1.json")
                def product_path(name, shard, frame):
                    directory = slow.received_dir if slow and slow.product == name and slow.shard == shard else out
                    return directory / f"{name}.shard{shard}_{frame:07d}.bin"
                report["exit_code"] = result.returncode
                if artifact:
                    shutil.copyfile(cfg_path, artifact / "run.yaml")
                    shutil.copyfile(log_path, artifact / "kotekan.txt")
                    shutil.copyfile(out / "exposure.v1.json", artifact / "exposure.v1.json")
                if rejection:
                    self.assertNotEqual(result.returncode, 0, log_path.read_text()[-4000:])
                    self.assertIn("CHARTS packet-loss input rejected", log_path.read_text())
                    self.assertFalse(list(out.glob("intensity.shard0_*.bin")))
                    report["status"] = "passed"
                    return
                self.assertEqual(result.returncode, 0, log_path.read_text()[-5000:])
                n_time = nt // 32 if mode == "fine" else nt
                n_integrated = (n_time + k - 1) // k
                n_freq = 10752 if mode == "fine" else 336
                dtype = "uint8" if mode == "fine" else "uint16"
                for shard in (0, 1):
                    for frame in range(frames):
                        reset = frame == 0 or sequences[shard, frame] != sequences[shard, frame - 1] + nt
                        previous = np.zeros(96) if reset else masks[shard, frame - 1][-96:]
                        expected = exposure_expectation(masks[shard, frame], previous,
                                                        mode=mode, k=k, reset=reset)
                        em, exposure = read_product(product_path("exposure", shard, frame),
                                                    dtype, (n_integrated, 336))
                        im, power = read_product(product_path("intensity", shard, frame),
                                                "float32", (n_integrated, n_freq, 64))
                        if constant:
                            expected_power, margin = constant_fine_power(
                                nt, reset=reset,
                                amplitude=0 if np.all(masks[shard, frame]) else amplitudes[shard, frame],
                                previous_amplitude=0 if reset or np.all(masks[shard, frame - 1])
                                else amplitudes[shard, frame - 1])
                            actual = power[:, :, 32]
                            report["packing_half_integer_margin"] = margin
                        elif mode == "coarse":
                            expected_power = expected * 4096.0 * amplitudes[shard, frame] ** 2
                            actual = power[:, :, 32]
                        else:
                            expected_power = np.zeros_like(power)
                            actual = power
                        allowance = 1e-3 + 1e-5 * np.abs(expected_power)
                        error = np.abs(actual - expected_power)
                        report["frames"].append({"shard": shard, "frame": frame,
                            "sequence": em.fpga_seq_num, "reset": reset,
                            "missing_spectra": int(np.count_nonzero(masks[shard, frame])),
                            "exposure_first_channel": exposure[:, 0].tolist(),
                            "exposure_mismatches": int(np.count_nonzero(exposure != expected)),
                            "max_power_abs_error": float(error.max()),
                            "power_tolerance_violations": int(np.count_nonzero(error > allowance)),
                            "power_peak": float(actual.max())})
                        bad_identity = (em.fpga_seq_num != sequences[shard, frame]
                                        or im.fpga_seq_num != em.fpga_seq_num
                                        or im.timestamp_step != 1)
                        if artifact and (bad_identity or not np.array_equal(exposure, expected)
                                         or not np.all(error <= allowance)):
                            for name in ("exposure", "intensity"):
                                source = product_path(name, shard, frame)
                                shutil.copyfile(source, artifact / ("failed_" + source.name))
                            np.savez_compressed(artifact / "failed_expectations.npz",
                                                exposure=expected, power=expected_power)
                        self.assertFalse(bad_identity, f"wrong frame association shard={shard} frame={frame}")
                        self.assertEqual(list(im.coarse_freq)[:336], list(range(336 * shard, 336 * (shard + 1))))
                        np.testing.assert_array_equal(exposure, expected)
                        self.assertTrue(np.isfinite(power).all())
                        self.assertTrue(np.all(error <= allowance),
                                        f"power mismatch shard={shard} frame={frame} max={error.max()}")
                report["status"] = "passed"
        finally:
            if report["status"] == "running":
                report["status"] = "failed"
            if artifact:
                (artifact / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")

    def test_coarse_k320_constant_power_and_loss(self):
        self._run_replay(mode="coarse", k=320)

    def test_coarse_k10_constant_power_and_loss(self):
        self._run_replay(mode="coarse", k=10)

    def test_fine_zero_power_loss_support_and_resets(self):
        self._run_replay()

    def test_fine_constant_startup_and_steady_power(self):
        self._run_replay(constant=True)

    def test_reject_nonbinary_mask(self):
        self._run_replay(rejection="nonbinary")

    def test_reject_mispaired_sequence(self):
        self._run_replay(rejection="sequence")

    def test_reject_wrong_timestamp_step(self):
        self._run_replay(rejection="timestamp")

    def test_reject_missing_mask_metadata(self):
        self._run_replay(rejection="metadata")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--kotekan-bin", type=Path, default=BINARY)
    args = parser.parse_args()
    RESULTS, BINARY = args.results_dir.resolve(), args.kotekan_bin.resolve()
    RESULTS.mkdir(parents=True, exist_ok=True)
    suite = unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__, fromlist=["*"]))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    summary = {"run": result.testsRun, "passed": result.testsRun - len(result.failures)
               - len(result.errors) - len(result.skipped), "failures": len(result.failures),
               "errors": len(result.errors), "skipped": len(result.skipped),
               "failure_details": [(str(test), error) for test, error in result.failures + result.errors],
               "skips": [(str(test), why) for test, why in result.skipped],
               "binary": str(BINARY), "binary_sha256": sha256(BINARY),
               "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    for name, command in {"gpu": ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                          "cuda": ["/usr/local/cuda/bin/nvcc", "--version"]}.items():
        r = subprocess.run(command, capture_output=True, text=True)
        summary[name] = {"command": command, "exit_code": r.returncode, "output": r.stdout + r.stderr}
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    raise SystemExit(0 if result.wasSuccessful() else 1)
