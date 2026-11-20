"""fp16 tensor-core beamformer against the scalar fp32 reference on identical X (FP16 plan WP3).

Run with ~/charts-utils/.venv/bin/python -m test_charts.tests.test_tensor_core_gpu
    [--results-dir PATH] [--kotekan-bin PATH]      (PYTHONPATH = repository root)

Both kernels read the same fp16 fine voltages X (the upchannelizer is deterministic, so two
runs on the same input produce the same X) and the same fp16 weights with power-of-two scales;
`beamformer_kernel` selects the kernel (tensor_core is the default). The tensor core multiplies
fp16 values exactly and accumulates in fp32 in a hardware-defined order, so:

* every output of both kernels is within the stage C float64 bound (expected_power with the
  effective weights and the fp16 ambiguity of X), 0 violations;
* |P_tc - P_scalar| <= 2 * A, A = 4 eps32 (E + K) sum_t (sum_e |W||X|)^2 the float32
  accumulation part of that bound (each kernel is within A of the exact power of the GPU X;
  |X| is bounded by |x16| + eps, the fp16 rounding ambiguity of the reference);
* bit-identical where the arithmetic leaves no freedom: beams whose weights are all zero
  (s = 0) and one-hot beams (one exact product, the rest exact zeros);
* the descriptor names the kernel that ran (beamformer.kernel) and the log reports its
  resources; an unknown kernel name is rejected at startup.

Cases cover a partial tail (T = 25), fewer spectra than K (T = 9), a frame whose windows cross
the kernel's 4-window staging round (T = 47: 5 windows, tail 7), weights distinct per (b, f, e)
with scales over ~20 octaves, and the production frame T_raw = 15360 (48 windows).
Synthetic verification only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from test_charts.sim import intensity_product as ip
from test_charts.tests import test_stage_c_numerics_gpu as sc
from test_charts.tests import test_stage_d_product_gpu as sd
from test_charts.tests import test_weights_fp16_gpu as wf

RESULTS = None
B, E, F, U, FF, K = sc.B, sc.E, sc.F, sc.U, sc.FF, 10
EPS32 = sc.EPS32
KERNELS = ("tensor_core", "scalar_reference")


def record(name, payload):
    if RESULTS:
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / f"{name}.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")


def accumulation_allowance(x16, eps, w):
    """A [windows][FF][B] = 4 eps32 (E + K) sum_t (sum_e |W_eff| (|x16| + eps))^2, computed
    directly (not as expected_power's bound minus its 1e-3 floor, which cancels to 0 for
    small weights)."""
    aw = np.abs(sc.fp16_device_weights(w)[3])                        # [B][FF][E] or [B][E]
    awt = aw.T if aw.ndim == 2 else np.ascontiguousarray(aw.transpose(1, 2, 0))
    upper = np.abs(x16).astype(np.float64) + eps
    t = upper.shape[0]
    out = np.zeros((-(-t // K), FF, B))
    for wdx in range(out.shape[0]):
        xs = upper[wdx * K:min(t, (wdx + 1) * K)].transpose(1, 0, 2)  # [FF][t][E]
        out[wdx] = np.sum((xs @ awt) ** 2, axis=1)
    return 4 * EPS32 * (E + K) * out


class TensorCoreGpu(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not sd.BINARY.is_file():
            raise unittest.SkipTest(f"no kotekan binary at {sd.BINARY}")
        cls.tmp = tempfile.TemporaryDirectory(dir="/dev/shm" if Path("/dev/shm").is_dir() else None)
        cls.root = Path(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_kernel(self, name, raw, wfile, t_raw, shard, kernel):
        r = sd.run_product(self.root / f"{name}_{kernel}", [raw], t_raw=t_raw, shard=shard,
                           weights_file=wfile, fmt="float32", export_params=False,
                           extra={"beamformer_kernel": kernel})
        self.assertEqual(len(r["intensity"]), 1, r["log"][-3000:])
        d = ip.load_descriptor(r["descriptor"])
        self.assertEqual(d["beamformer"]["kernel"], kernel)
        self.assertIn(f"beamformer kernel {kernel}:", r["log"])
        return d, ip.read_intensity(d, r["intensity"][0])[1], r["log"]

    def check_pair(self, name, raw, w, t_raw, shard, onehot_beams=()):
        wfile = sc.write_weights(self.root / f"{name}_w.bin", w)
        got, logs = {}, {}
        for kernel in KERNELS:
            _, got[kernel], logs[kernel] = self.run_kernel(name, raw, wfile, t_raw, shard, kernel)
        x16, eps = sc.fine_fp16(raw, None)
        want, bound, ambig = sc.expected_power(x16, eps, w, K)
        tc = got["tensor_core"].astype(np.float64)
        ref = got["scalar_reference"].astype(np.float64)
        allowance = accumulation_allowance(x16, eps, w)
        diff = np.abs(tc - ref)
        scale = sc.fp16_device_weights(w)[2]                         # [B][FF] or [B]
        zero_bf = (scale == 0).T if scale.ndim == 2 else np.broadcast_to(scale == 0, (FF, B))
        resources = next((ln.split("beamformer kernel tensor_core:")[1].strip()
                          for ln in logs["tensor_core"].splitlines()
                          if "beamformer kernel tensor_core:" in ln), "")
        m = {"t_raw": t_raw, "shard": shard, "windows": int(tc.shape[0]),
             "tail_spectra": int(t_raw // U - (tc.shape[0] - 1) * K),
             "tensor_core_vs_float64": sc.compare(got["tensor_core"], want, bound, ambig),
             "scalar_vs_float64": sc.compare(got["scalar_reference"], want, bound, ambig),
             "kernel_diff_violations": int(np.sum(diff > 2 * allowance)),
             "kernel_diff_max_over_2A": float(np.max(diff / np.maximum(2 * allowance, 1e-300))),
             "kernel_diff_max_rel": float(np.max(diff / np.maximum(np.abs(ref), 1e-30))),
             "kernel_diff_median_rel_nonzero": float(np.median(
                 diff[ref > 0] / ref[ref > 0])) if np.any(ref > 0) else 0.0,
             "bit_identical_fraction": float(np.mean(tc == ref)),
             "zero_bf_count": int(zero_bf.sum()),
             "zero_bf_max": float(max(np.abs(tc[:, zero_bf]).max(initial=0.0),
                                      np.abs(ref[:, zero_bf]).max(initial=0.0))),
             "tensor_core_resources": resources}
        if onehot_beams:
            oh = list(onehot_beams)
            m["onehot_identical"] = bool(np.array_equal(tc[:, :, oh], ref[:, :, oh]))
            m["onehot_max_abs_diff"] = float(diff[:, :, oh].max())
            m["onehot_tensor_core"] = sc.onehot_check(got["tensor_core"][:, :, oh], want[:, :, oh],
                                                      ambig[:, :, oh], n_beams=len(oh))
        record(f"tc_{name}", m)
        for key in ("tensor_core_vs_float64", "scalar_vs_float64"):
            self.assertEqual(m[key]["violations"], 0, m)
        self.assertEqual(m["kernel_diff_violations"], 0, m)
        self.assertEqual(m["zero_bf_max"], 0.0, m)
        self.assertTrue(resources, m)
        if onehot_beams:
            self.assertTrue(m["onehot_identical"], m)
            self.assertEqual(m["onehot_tensor_core"]["onehot_violations"], 0, m)
        return m

    # codec fixture: all 256 byte patterns, zero / constant / extreme channels; codec weights
    # have one-hot beams 0..7 at element boundaries, element 9 disabled, beam 127 non-unit.
    def test_codec_partial_tail_t25(self):
        raw = sc.codec_fixture(800, seed=800)
        self.check_pair("codec_t800", raw, sc.codec_weights(seed=7), 800, 0,
                        onehot_beams=range(8))

    def test_codec_t_less_than_k_shard1(self):
        raw = sc.codec_fixture(288, seed=289)
        self.check_pair("codec_t288_s1", raw, sc.codec_weights(seed=8), 288, 1,
                        onehot_beams=range(8))

    def test_distinct_weights_across_staging_round(self):
        # 47 spectra = 5 windows (tail 7): windows 0..3 in the first staging round, 4 alone.
        raw, _ = sc.fine_fixture(1504, seed=1504)
        self.check_pair("distinct_t1504", raw, wf.distinct_weights(seed=31), 1504, 0)

    def test_production_frame_shard1(self):
        rng = np.random.default_rng(15360)
        re = rng.integers(-7, 8, (15360, F, E))
        im = rng.integers(-7, 8, (15360, F, E))
        raw = sc.encode(re, im)
        m = self.check_pair("noise_t15360_s1", raw, wf.distinct_weights(seed=32), 15360, 1)
        self.assertEqual(m["windows"], 48)

    def test_reject_unknown_kernel(self):
        wfile = sc.write_weights(self.root / "rej_w.bin", sc.codec_weights(seed=9))
        r = sd.run_product(self.root / "reject", [np.zeros((288, F, E), np.uint8)], t_raw=288,
                           shard=0, weights_file=wfile, fmt="float32", export_params=False,
                           extra={"beamformer_kernel": "cublas"})
        record("tc_reject_unknown", {"rc": r["rc"], "outputs": len(r["intensity"])})
        self.assertNotEqual(r["rc"], 0)
        self.assertFalse(r["intensity"])
        self.assertIn("beamformer_kernel must be 'tensor_core' or 'scalar_reference'", r["log"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--kotekan-bin", type=Path)
    args, rest = parser.parse_known_args()
    if args.results_dir:
        RESULTS = args.results_dir.resolve()
    if args.kotekan_bin:
        sd.BINARY = sc.BINARY = args.kotekan_bin.resolve()
    unittest.main(argv=[sys.argv[0], *rest])
