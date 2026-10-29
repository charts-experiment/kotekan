"""Native `.npy` combined weights in cudaChartsBeamformerCommand (coarse, float32).

Run with /usr/bin/python3 -m test_charts.tests.test_weights_npy_gpu [--results-dir PATH].

Acceptance:
* The `.npy` + JSON descriptor from build_beamformer_weights.py gives output bit-identical
  to the legacy raw `.bin` of the same coefficients (so the header bytes are never read as
  coefficients and no steering/gain is applied twice), for both shards, and the intensity
  is the analytical array factor already checked in test_weights_replay_gpu.
* Every malformed or mismatched input is rejected at startup, naming the cause, with no
  intensity written: missing descriptor, wrong mode/shard/frequency/element order/shape,
  wrong dtype, Fortran order, trailing or missing bytes, non-finite values, bad magic, and a
  legacy `.bin` whose size is wrong (including a `.npy` renamed to `.bin`).
Synthetic verification only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np

from test_charts.sim.packet_loss import read_product, write_frame
from test_charts.sim.pipeline import create_beamformer_yaml, execute_kotekan
from test_charts.tests import test_weights_replay_gpu as rp

bw = rp.bw
RESULTS = None


class NpyWeightsGpu(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not rp.BINARY.is_file():
            raise unittest.SkipTest(f"no kotekan binary at {rp.BINARY}")
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.directions = bw.load_directions(rp.ROOT / "charts/config/charts_hex64_beam_directions.json")
        cls.positions = bw.default_positions_charts_logical()
        cls.gains = rp.make_gains()
        cls.correction, flagged = bw.gain_correction(cls.gains)
        cls.flagged = flagged.all(axis=1)
        transverse = np.hypot(cls.directions[:, 0], cls.directions[:, 1])
        cls.n_src = cls.directions[int(np.argmin(np.abs(transverse - 0.45)))]
        cls.cases = 0

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    # ------------------------------------------------------------------ helpers
    def build_files(self, shard, name):
        """Write builder-style .npy + descriptor and the matching legacy .bin."""
        d = self.root / name
        d.mkdir(parents=True, exist_ok=True)
        weights = bw.build_weights("coarse", shard, self.correction, self.positions,
                                   self.directions)
        npy = d / "w.npy"
        np.save(npy, weights)
        inputs = {"gain_convention": "response", "gains": "synthetic", "calibration_note": name}
        desc = bw.descriptor("coarse", shard, npy, self.directions.shape[0], self.flagged, inputs)
        (d / "w.json").write_text(json.dumps(desc))
        np.ascontiguousarray(weights, dtype=np.complex64).tofile(d / "w.bin")
        return d, weights, desc

    def replay(self, name, shard, weights_file, descriptor=None):
        work = self.root / f"run_{name}"
        inp, out = work / "in", work / "out"
        inp.mkdir(parents=True, exist_ok=True)
        out.mkdir(parents=True, exist_ok=True)
        volt, _ = rp.pack_voltage(self.gains, shard, self.n_src, self.positions)
        write_frame(inp / f"voltage.shard{shard}_0000000.bin", volt, 0, shard)
        extra = {}
        if descriptor is not None:
            extra["weights_descriptor"] = descriptor
        yaml_path = create_beamformer_yaml(
            work / "run.yaml", inp, f"voltage.shard{shard}", out, f"intensity.shard{shard}", 1,
            samples_per_data_set=rp.T_RAW, integration_spectra=rp.K, input_mode="coarse",
            output_format="float32", shard_id=shard, weights_file=weights_file)
        if descriptor is not None:
            text = yaml_path.read_text().replace(
                f'weights_file: "{Path(weights_file).as_posix()}"',
                f'weights_file: "{Path(weights_file).as_posix()}"\n'
                f'      weights_descriptor: "{Path(descriptor).as_posix()}"')
            yaml_path.write_text(text)
        rc = execute_kotekan(yaml_path, rp.BINARY, work / "kotekan.log", timeout_s=120)
        files = sorted(out.glob("intensity.shard*_*.bin"))
        log = (work / "kotekan.log").read_text(errors="replace")
        power = None
        if files:
            _, power = read_product(files[0], "float32", (rp.N_OUT, 336, 64))
        return rc, power, log

    def assert_rejected(self, name, shard, weights_file, message, descriptor=None):
        rc, power, log = self.replay(name, shard, weights_file, descriptor)
        self.assertIsNone(power, f"{name}: output written despite invalid weights")
        self.assertIn(message, log, f"{name}: expected '{message}' in log (rc={rc})")

    def mutate_descriptor(self, d, name, **changes):
        desc = json.loads((d / "w.json").read_text())
        for key, value in changes.items():
            desc[key] = value
        path = d / f"{name}.json"
        path.write_text(json.dumps(desc))
        return path

    # ------------------------------------------------------------------ acceptance
    def check_matches_bin(self, shard):
        d, _, desc = self.build_files(shard, f"nominal{shard}")
        rc_n, npy_power, log = self.replay(f"npy{shard}", shard, d / "w.npy")
        rc_b, bin_power, _ = self.replay(f"bin{shard}", shard, d / "w.bin")
        self.assertIsNotNone(npy_power, log[-2000:])
        self.assertIsNotNone(bin_power)
        self.assertTrue(np.array_equal(npy_power, bin_power))
        self.assertGreater(float(npy_power.max()), 0.0)
        self.assertIn("CHARTS weights loaded from .npy", log)
        self.assertIn(desc["sha256"], log)
        if shard == 0:  # flagged channel (all coefficients zero) is exactly zero
            self.assertEqual(float(npy_power[:, rp.FLAG_CHAN].max()), 0.0)
        if RESULTS:
            (RESULTS / f"npy_nominal_shard{shard}.json").write_text(json.dumps(
                {"identical_to_bin": True, "max_power": float(npy_power.max()),
                 "sha256": desc["sha256"]}, indent=2) + "\n")

    def test_npy_equals_legacy_bin_shard0(self):
        self.check_matches_bin(0)

    def test_npy_equals_legacy_bin_shard1(self):
        self.check_matches_bin(1)

    def test_reject_missing_descriptor(self):
        d, *_ = self.build_files(0, "nodesc")
        (d / "w.json").unlink()
        self.assert_rejected("nodesc", 0, d / "w.npy", "cannot open weights_descriptor")

    def test_reject_descriptor_mode(self):
        d, *_ = self.build_files(0, "mode")
        p = self.mutate_descriptor(d, "bad", mode="fine")
        self.assert_rejected("mode", 0, d / "w.npy", "mode 'fine' != configured 'coarse'", p)

    def test_reject_descriptor_shard(self):
        d, *_ = self.build_files(0, "shard")
        p = self.mutate_descriptor(d, "bad", shard=1)
        self.assert_rejected("shard", 0, d / "w.npy", "shard does not match", p)

    def test_reject_shard1_file_for_shard0(self):
        d, *_ = self.build_files(1, "wrongshard")
        self.assert_rejected("wrongshard", 0, d / "w.npy", "shard does not match")

    def test_reject_descriptor_element_order(self):
        d, *_ = self.build_files(0, "order")
        p = self.mutate_descriptor(d, "bad", element_order="charts_logical")
        self.assert_rejected("order", 0, d / "w.npy", "element_order must be kotekan_raw", p)

    def test_reject_descriptor_normalization(self):
        d, *_ = self.build_files(0, "norm")
        p = self.mutate_descriptor(d, "bad", normalization="unit")
        self.assert_rejected("norm", 0, d / "w.npy", "normalization must be 'none'", p)

    def test_reject_descriptor_shape(self):
        d, *_ = self.build_files(0, "dshape")
        p = self.mutate_descriptor(d, "bad", shape=[64, 335, 64])
        self.assert_rejected("dshape", 0, d / "w.npy", "shape does not match", p)

    def test_reject_descriptor_frequency(self):
        d, *_ = self.build_files(0, "freq")
        desc = json.loads((d / "w.json").read_text())
        desc["frequency_hz"]["first"] += 300e3
        (d / "bad.json").write_text(json.dumps(desc))
        self.assert_rejected("freq", 0, d / "w.npy", "first/last frequency disagrees", d / "bad.json")

    def test_reject_descriptor_format_and_version(self):
        d, *_ = self.build_files(0, "fmtver")
        p = self.mutate_descriptor(d, "fmt", format="other")
        self.assert_rejected("fmt", 0, d / "w.npy", "format must be", p)
        p = self.mutate_descriptor(d, "v2", version=2)  # pre-conjugate builder
        self.assert_rejected("v2", 0, d / "w.npy", "version must be >= 3", p)

    def test_reject_descriptor_step(self):
        # Shard 0 first channel is nu0 for any step; the step itself must be checked.
        d, *_ = self.build_files(0, "step")
        desc = json.loads((d / "w.json").read_text())
        desc["frequency_hz"]["dnu_hz"] = 250e3
        (d / "bad.json").write_text(json.dumps(desc))
        self.assert_rejected("step", 0, d / "w.npy", "nu0_hz/dnu_hz disagree", d / "bad.json")
        desc = json.loads((d / "w.json").read_text())
        desc["frequency_hz"]["last"] += 300e3
        (d / "last.json").write_text(json.dumps(desc))
        self.assert_rejected("last", 0, d / "w.npy", "first/last frequency disagrees",
                             d / "last.json")

    def test_reject_noninteger_shard(self):
        d, *_ = self.build_files(0, "fshard")
        p = self.mutate_descriptor(d, "bad", shard=0.4)
        self.assert_rejected("fshard", 0, d / "w.npy", "'shard' must be a non-negative integer", p)

    def test_reject_complex128(self):
        d, w, _ = self.build_files(0, "c16")
        np.save(d / "w.npy", w.astype(np.complex128))
        self.assert_rejected("c16", 0, d / "w.npy", "dtype must be little-endian complex64")

    def test_reject_fortran_order(self):
        d, w, _ = self.build_files(0, "fortran")
        np.save(d / "w.npy", np.asfortranarray(w))
        self.assert_rejected("fortran", 0, d / "w.npy", "Fortran order is not supported")

    def test_reject_wrong_npy_shape(self):
        d, w, _ = self.build_files(0, "nshape")
        np.save(d / "w.npy", w[:63])
        self.assert_rejected("nshape", 0, d / "w.npy", "does not match expected")

    def test_reject_trailing_bytes(self):
        d, *_ = self.build_files(0, "trail")
        with open(d / "w.npy", "ab") as f:
            f.write(b"\0" * 8)
        self.assert_rejected("trail", 0, d / "w.npy", "payload is")

    def test_reject_truncated(self):
        d, *_ = self.build_files(0, "trunc")
        data = (d / "w.npy").read_bytes()
        (d / "w.npy").write_bytes(data[:-8])
        self.assert_rejected("trunc", 0, d / "w.npy", "payload is")

    def test_reject_nonfinite(self):
        for label, value in (("nan", np.nan), ("inf", np.inf)):
            d, w, _ = self.build_files(0, f"nf_{label}")
            w = w.copy()
            w[3, 100, 7] = value
            np.save(d / "w.npy", w)
            self.assert_rejected(f"nf_{label}", 0, d / "w.npy", "non-finite coefficient")

    def test_reject_bad_magic(self):
        d, *_ = self.build_files(0, "magic")
        (d / "w.npy").write_bytes(b"not a npy file" * 100)
        self.assert_rejected("magic", 0, d / "w.npy", "magic")

    def test_reject_object_array(self):
        d, *_ = self.build_files(0, "obj")
        np.save(d / "w.npy", np.empty((64, 336, 64), dtype=object), allow_pickle=True)
        self.assert_rejected("obj", 0, d / "w.npy", "dtype must be little-endian complex64")

    def test_reject_sha256_mismatch(self):
        # A valid, finite .npy that differs from the one the descriptor describes.
        d, w, _ = self.build_files(0, "sha")
        w = w.copy()
        w[5, 200, 9] *= 2
        np.save(d / "w.npy", w)
        self.assert_rejected("sha", 0, d / "w.npy", "sha256 mismatch")

    def test_reject_missing_sha256(self):
        d, *_ = self.build_files(0, "nosha")
        desc = json.loads((d / "w.json").read_text())
        del desc["sha256"]
        (d / "bad.json").write_text(json.dumps(desc))
        self.assert_rejected("nosha", 0, d / "w.npy", "missing or malformed 'sha256'",
                             d / "bad.json")

    def test_legacy_bin_requires_exact_size(self):
        d, *_ = self.build_files(0, "legacy")
        with open(d / "w.bin", "ab") as f:
            f.write(b"\0" * 8)
        self.assert_rejected("legacy_trail", 0, d / "w.bin", "weights_file byte size mismatch")
        shutil.copy(d / "w.npy", d / "renamed.bin")  # header bytes would shift the data
        self.assert_rejected("legacy_renamed", 0, d / "renamed.bin",
                             "weights_file byte size mismatch")


class FineNpyWeightsGpu(rp.FineWeightsReplayGpu):
    """Fine U=32, K=10, float32: `.npy` (352 MB per shard) versus legacy `.bin`.

    Same tone fixtures as test_weights_replay_gpu; the `.npy` run must be bit-identical to
    the `.bin` run and match the analytical array factor in steady state.
    """

    # inherit fixtures and metrics only; the replay test methods are not re-run here
    test_fine_shard0_matches_array_factor = None
    test_fine_shard1_matches_array_factor = None
    test_fine_negative_control_missing_conjugate = None
    test_fine_negative_control_opposite_sign = None

    def run_fine_file(self, name, shard, weights_file):
        work = self.root / f"npy_{name}"
        inp, out = work / "in", work / "out"
        inp.mkdir(parents=True, exist_ok=True)
        out.mkdir(parents=True, exist_ok=True)
        h0 = rp.fine_gain()
        volt, amp, u = rp.pack_fine_voltage(self.gains, shard, self.n_src, self.positions, h0)
        write_frame(inp / f"voltage.shard{shard}_0000000.bin", volt, 0, shard)
        yaml_path = create_beamformer_yaml(
            work / "run.yaml", inp, f"voltage.shard{shard}", out, f"intensity.shard{shard}", 1,
            samples_per_data_set=rp.FINE_T_RAW, integration_spectra=rp.FINE_K,
            input_mode="fine", output_format="float32", shard_id=shard,
            weights_file=weights_file)
        rc = execute_kotekan(yaml_path, rp.BINARY, work / "kotekan.log", timeout_s=600)
        log = (work / "kotekan.log").read_text(errors="replace")
        files = sorted(out.glob("intensity.shard*_*.bin"))
        power = None
        if files:
            _, power = read_product(files[0], "float32", (rp.FINE_N_OUT, 10752, 64))
        return rc, power, log, amp, u, h0

    def check_fine_npy(self, shard):
        d = self.root / f"fine_files{shard}"
        d.mkdir(parents=True, exist_ok=True)
        weights = self.fine_weights(shard)
        np.save(d / "w.npy", weights)
        inputs = {"gain_convention": "response", "gains": "synthetic",
                  "calibration_note": f"fine{shard}"}
        flagged = bw.gain_correction(self.gains)[1]
        desc = bw.descriptor("fine", shard, d / "w.npy", self.directions.shape[0], flagged,
                             inputs)
        (d / "w.json").write_text(json.dumps(desc))
        np.ascontiguousarray(weights, dtype=np.complex64).tofile(d / "w.bin")
        del weights

        rc_n, npy_power, log, amp, u, h0 = self.run_fine_file(f"fine{shard}", shard, d / "w.npy")
        self.assertIsNotNone(npy_power, log[-2000:])
        _, bin_power, *_ = self.run_fine_file(f"fine_bin{shard}", shard, d / "w.bin")
        self.assertIsNotNone(bin_power)
        self.assertTrue(np.array_equal(npy_power, bin_power))
        self.assertIn("CHARTS weights loaded from .npy", log)
        self.assertIn(desc["sha256"], log)

        m = self.fine_metrics(shard, npy_power[1].astype(np.float64), amp, u, h0)
        self.assertGreater(m["src_ratio_min"], 0.90, m)
        self.assertLess(m["src_ratio_max"], 1.10, m)
        self.assertEqual(m["src_is_beam_argmax_fraction"], 1.0, m)
        self.assertEqual(m["tone_in_expected_fine_bin_fraction"], 1.0, m)
        self.assertLess(m["off_peak_max_err_over_peak"], 0.10, m)
        if shard == 0:
            self.assertEqual(m["flagged_channel_power_max"], 0.0, m)
        if RESULTS:
            (RESULTS / f"npy_fine_shard{shard}.json").write_text(json.dumps(
                {"identical_to_bin": True, "sha256": desc["sha256"], **m}, indent=2) + "\n")
        for p in d.iterdir():  # 352 MB each
            p.unlink()

    def test_fine_npy_equals_bin_and_array_factor_shard0(self):
        self.check_fine_npy(0)

    def test_fine_npy_equals_bin_and_array_factor_shard1(self):
        self.check_fine_npy(1)

    def test_fine_rejects_coarse_file(self):
        d = self.root / "fine_coarse_file"
        d.mkdir(parents=True, exist_ok=True)
        w = bw.build_weights("coarse", 0, self.correction, self.positions, self.directions)
        np.save(d / "w.npy", w)
        desc = bw.descriptor("coarse", 0, d / "w.npy", self.directions.shape[0],
                             np.zeros((672, 64), bool), {"gain_convention": "response",
                                                         "gains": "synthetic",
                                                         "calibration_note": "coarse"})
        (d / "w.json").write_text(json.dumps(desc))
        _, power, log, *_ = self.run_fine_file("coarse_in_fine", 0, d / "w.npy")
        self.assertIsNone(power)
        self.assertIn("mode 'coarse' != configured 'fine'", log)


def main():
    global RESULTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--kotekan-bin", type=Path)
    args, rest = parser.parse_known_args()
    if args.results_dir:
        RESULTS = args.results_dir
        RESULTS.mkdir(parents=True, exist_ok=True)
    if args.kotekan_bin:
        rp.BINARY = args.kotekan_bin
    unittest.main(argv=[sys.argv[0], *rest], verbosity=2)


if __name__ == "__main__":
    main()
