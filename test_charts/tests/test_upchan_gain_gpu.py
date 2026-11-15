"""Fine upchannelizer gain G(u) (`upchan_gain`): stage 1, wiring with unity default.

Run with ~/miniconda3/envs/kotekan_test/bin/python -m test_charts.tests.test_upchan_gain_gpu
    [--results-dir PATH] [--kotekan-bin PATH]

fp16 fine path (FP16_TC_ARCHITECTURE.md). Acceptance:
* Unity default. Fine float32 and int8 outputs (both shards) with no `upchan_gain` key and
  with an explicit 32 x 1.0 gain are byte-identical. (No comparison with a pre-change binary:
  the int4 path of beamform-64 differs from the fp16 path by design.)
* Index wiring. With G = 1 except G[u0] = 3, only fine channels f*32 + u0 change, in every
  local coarse channel and on both shards, by a power ratio of 9 (linear path; fp16 rounding
  of 3X vs 3 x fp16(X) only perturbs it at the 1e-3 level). A second u0 checks the order.
* Descriptor v3: history.upchan_gain carries the values, unity flag, source, reference,
  purpose, placement and the SHA-256 of the applied float32 [336*32] array;
  ip.upchan_gain() returns that array; history.fine_format is fp16, no fine_quantization.
* Invalid configurations are rejected with their message: wrong length, <= 0, > 3000 (fp16
  range), source inconsistent with the values, unknown source, the removed quantize_clamp.
  NaN is rejected by the kotekan config parser (JSON has no NaN) before the command's own
  finite check.
Synthetic verification of the plumbing only; the numerical effect of G is stage 2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from test_charts.sim import intensity_product as ip
from test_charts.tests import test_stage_c_numerics_gpu as sc
from test_charts.tests import test_stage_d_product_gpu as sd

RESULTS = None
B, E, F, U, FF = 128, 64, 336, 32, 336 * 32
T_RAW = 640


def record(name, payload):
    if RESULTS:
        (RESULTS / f"{name}.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")


def gain_sha256(values) -> str:
    return hashlib.sha256(np.tile(np.asarray(values, np.float32), F).tobytes()).hexdigest()


class UpchanGainGpu(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not sd.BINARY.is_file():
            raise unittest.SkipTest(f"no kotekan binary at {sd.BINARY}")
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        w = sc.probe_weights(seed=41)
        cls.wfile = sc.write_weights(cls.root / "fine_w.bin",
                                     np.broadcast_to(w[:, None, :], (B, FF, E)))
        cls.frames = [sc.fine_fixture(T_RAW, seed=700, n0=i * T_RAW, frame_seed=7000 + i)[0]
                      for i in range(2)]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_fine(self, label, *, shard, fmt, extra=None):
        run = sd.run_product(self.root / label, self.frames, mode="fine", t_raw=T_RAW,
                             k=10, shard=shard, weights_file=self.wfile, fmt=fmt, extra=extra)
        self.assertEqual(len(run["intensity"]), len(self.frames), run["log"][-3000:])
        return run

    @staticmethod
    def files_bytes(run, key="intensity"):
        return [p.read_bytes() for p in run[key]]

    # ------------------------------------------------------------------ unity
    def test_unity_is_bit_identical(self):
        report = {}
        for shard in (0, 1):
            for fmt in ("float32", "int8"):
                base = self.run_fine(f"u_default_{shard}_{fmt}", shard=shard, fmt=fmt)
                ones = self.run_fine(f"u_ones_{shard}_{fmt}", shard=shard, fmt=fmt,
                                     extra={"upchan_gain": [1.0] * U})
                entry = {
                    "explicit_unity_identical": self.files_bytes(base) == self.files_bytes(ones),
                    "qparams_identical": self.files_bytes(base, "qparams")
                    == self.files_bytes(ones, "qparams"),
                }
                entry["files"] = len(base["intensity"])
                report[f"shard{shard}_{fmt}"] = entry
        record("g_unity_bit_identical", report)
        for label, entry in report.items():
            if isinstance(entry, dict):
                for key, ok in entry.items():
                    if key != "files":
                        self.assertTrue(ok, (label, key, report))

    # ------------------------------------------------------------------ index wiring
    def test_single_bin_gain_changes_only_that_bin(self):
        report = {}
        for shard in (0, 1):
            base = self.run_fine(f"w_base_{shard}", shard=shard, fmt="float32")
            d0 = ip.load_descriptor(base["descriptor"])
            i0 = np.stack([ip.read_intensity(d0, p)[1] for p in base["intensity"]])
            for u0 in (5, 26):
                g = [1.0] * U
                g[u0] = 3.0
                run = self.run_fine(f"w_u{u0}_{shard}", shard=shard, fmt="float32",
                                    extra={"upchan_gain": g, "upchan_gain_source": "measured",
                                           "upchan_gain_reference": f"wiring test u0={u0}"})
                d = ip.load_descriptor(run["descriptor"])
                i1 = np.stack([ip.read_intensity(d, p)[1] for p in run["intensity"]])
                changed = np.any(i1 != i0, axis=(0, 1, 3))                    # [FF]
                u_of = np.arange(FF) % U
                f_of = np.arange(FF) // U
                report[f"shard{shard}_u{u0}"] = {
                    "changed_outside_u0": int(np.sum(changed & (u_of != u0))),
                    "changed_u_values": sorted(set(u_of[changed].tolist())),
                    "coarse_channels_changed_at_u0": int(len(set(f_of[changed & (u_of == u0)]))),
                    "untouched_bins_bit_identical": bool(
                        np.array_equal(i1[:, :, u_of != u0], i0[:, :, u_of != u0])),
                    "power_ratio_at_u0_median": float(np.median(
                        i1[:, :, u_of == u0][i0[:, :, u_of == u0] > 1]
                        / i0[:, :, u_of == u0][i0[:, :, u_of == u0] > 1])),
                }
        record("g_single_bin_wiring", report)
        for label, r in report.items():
            self.assertEqual(r["changed_outside_u0"], 0, (label, r))
            self.assertTrue(r["untouched_bins_bit_identical"], (label, r))
            # The fixture's channels with a tone or noise in u0 must respond, linearly.
            self.assertGreater(r["coarse_channels_changed_at_u0"], 0, (label, r))
            self.assertLess(abs(r["power_ratio_at_u0_median"] - 9.0), 0.01, (label, r))

    # ------------------------------------------------------------------ descriptor
    def test_descriptor_records_gain(self):
        unity = self.run_fine("d_unity", shard=0, fmt="float32")
        d = ip.load_descriptor(unity["descriptor"])
        self.assertEqual(d["schema"], "charts-intensity-product-v3")
        self.assertEqual(d["history"]["fine_format"], "fp16")
        self.assertNotIn("fine_quantization", d["history"])
        self.assertIn("(2s - 127)/256", d["history"]["pfb"])
        g = d["history"]["upchan_gain"]
        self.assertIn("scalloping", g["purpose"])
        self.assertIn("before the fp16 conversion", g["placement"])
        self.assertTrue(g["unity"])
        self.assertEqual(g["source"], "unity")
        self.assertEqual(g["values"], [1.0] * U)
        self.assertEqual(g["sha256"], gain_sha256([1.0] * U))
        np.testing.assert_array_equal(ip.upchan_gain(d), np.ones(FF, np.float32))

        # A profile with non-dyadic values (1/H_amp of the firmware PFB, rounded).
        h_amp = np.array([0.7048, 0.7175, 0.7410, 0.7722, 0.8076, 0.8439, 0.8784, 0.9092,
                          0.9352, 0.9561, 0.9720, 0.9834, 0.9912, 0.9960, 0.9988, 1.0000])
        values = (1.0 / np.concatenate([h_amp, h_amp[::-1]])).astype(np.float32)
        run = self.run_fine("d_profile", shard=1, fmt="float32",
                            extra={"upchan_gain": [float(v) for v in values],
                                   "upchan_gain_source": "analytic",
                                   "upchan_gain_reference": "f-engine pfb_fir 4-tap hamming"})
        d = ip.load_descriptor(run["descriptor"])
        g = d["history"]["upchan_gain"]
        record("g_descriptor", {"unity": ip.load_descriptor(unity["descriptor"])["history"],
                                "profile": d["history"]})
        self.assertFalse(g["unity"])
        self.assertEqual(g["source"], "analytic")
        self.assertEqual(g["reference"], "f-engine pfb_fir 4-tap hamming")
        np.testing.assert_array_equal(np.asarray(g["values"], np.float32), values)
        self.assertEqual(g["sha256"], gain_sha256(values))
        np.testing.assert_array_equal(ip.upchan_gain(d), np.tile(values, F))
        self.assertIn("upchan_gain source=analytic", run["log"])

    # ------------------------------------------------------------------ rejections
    def test_rejected_configurations(self):
        bad_len = [1.0] * (U - 1)
        nan = [1.0] * U
        nan[3] = float("nan")
        zero = [1.0] * U
        zero[0] = 0.0
        neg = [1.0] * U
        neg[31] = -2.0
        two = [2.0] * U
        too_large = [1.0] * U
        too_large[7] = 3001.0
        limit = "upchan_gain values must be finite, > 0 and <= 3000"
        cases = {
            "length": ({"upchan_gain": bad_len}, "fine", "upchan_gain must have 32 values"),
            # JSON has no NaN, so the kotekan config parser rejects it before the command; the
            # command's finite check is a second line of defence.
            "nan_config_parser": ({"upchan_gain": nan, "upchan_gain_source": "measured"},
                                  "fine", "json.exception.parse_error"),
            "zero": ({"upchan_gain": zero, "upchan_gain_source": "measured"}, "fine", limit),
            "negative": ({"upchan_gain": neg, "upchan_gain_source": "measured"}, "fine", limit),
            "above_fp16_limit": ({"upchan_gain": too_large, "upchan_gain_source": "measured"},
                                 "fine", limit),
            "quantize_clamp_removed": ({"quantize_clamp": 7.0}, "fine",
                                       "quantize_clamp was removed"),
            "non_unity_without_source": ({"upchan_gain": two}, "fine",
                                         "upchan_gain_source must be"),
            "non_unity_labelled_unity": ({"upchan_gain": two, "upchan_gain_source": "unity"},
                                         "fine", "upchan_gain_source 'unity' is required"),
            "unity_labelled_analytic": ({"upchan_gain_source": "analytic"}, "fine",
                                        "upchan_gain_source 'unity' is required"),
            "unknown_source": ({"upchan_gain": two, "upchan_gain_source": "guess"}, "fine",
                               "upchan_gain_source must be"),
        }
        report = {}
        for label, (extra, mode, message) in cases.items():
            run = sd.run_product(self.root / f"rej_{label}", self.frames[:1], mode=mode,
                                 t_raw=T_RAW, k=10, shard=0, weights_file=self.wfile,
                                 fmt="float32", extra=extra)
            report[label] = {"rejected": len(run["intensity"]) == 0,
                             "message_found": message in run["log"]}
        record("g_rejected_configurations", report)
        for label, r in report.items():
            self.assertTrue(r["rejected"] and r["message_found"], (label, report))


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
        sd.BINARY = args.kotekan_bin
    unittest.main(argv=[sys.argv[0], *rest], verbosity=2)


if __name__ == "__main__":
    main()
