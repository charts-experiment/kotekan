"""fp16 device weights with a power-of-two scale per (beam, fine channel) (FP16 plan WP2).

Run with ~/charts-utils/.venv/bin/python -m test_charts.tests.test_weights_fp16_gpu
    [--results-dir PATH] [--kotekan-bin PATH]      (PYTHONPATH = repository root)

The command converts the canonical complex64 [B][F][E] weights once at load:
s[b,f] = 2^k, the smallest k with 2^(2k) >= max_e |W|^2, W16 = fp16_rn(W / s), s = 0 for an
all-zero (b, f); the scalar beamformer applies P = sum (s Yr)^2 + (s Yi)^2. The reference is
the documented rule (test_stage_c_numerics_gpu.fp16_device_weights), never a port of the kernel.

Acceptance:
* Bit-exact conversion: the descriptor's weights.device.effective_sha256 (fp16 planes then
  float32 scales) equals the hash of the rule evaluated in Python, and its error statistics
  and zero-(b, f) count match.
* Weights distinct per (b, f, e), with scales spanning ~20 octaves, zero elements, zero (b, f)
  and a zero 16-beam tile: float32 output within the stage C bound computed with the effective
  weights, 0 violations, on both shards. Wrong fine channel (weights rolled by one in f) and
  wrong beam (adjacent beams swapped) must fail; zero (b, f) are exactly 0, also through int8.
* Nulls: identical int4 input on every element (identical fp16 fine voltages) and non-dyadic
  weights with sum_e W = 0: null power / power of the same |W| phased up <= -60 dB.
* A weight magnitude below the float32 range of the scale is rejected at startup.
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

RESULTS = None
B, E, F, U, FF = sc.B, sc.E, sc.F, sc.U, sc.FF
T_RAW = 640                     # 20 fine spectra: two K = 10 windows
ZERO_TILE = slice(112, 128)     # one whole int8 16-beam tile of zero weights


def record(name, payload):
    if RESULTS:
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / f"{name}.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")


def distinct_weights(seed: int) -> np.ndarray:
    """[B][FF][E] complex64, different for every (b, f, e); per-(b, f) magnitude 2^U(-10, 10)
    so the scales span ~20 octaves; element 9 off in even beams; 2 % of (b, f) all zero;
    beams 112..127 all zero."""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((B, FF, E)) + 1j * rng.standard_normal((B, FF, E))
    w *= np.exp2(rng.uniform(-10, 10, (B, FF)))[:, :, None]
    w[::2, :, 9] = 0
    w[rng.random((B, FF)) < 0.02] = 0
    w[ZERO_TILE] = 0
    return w.astype(np.complex64)


def null_weights(seed: int) -> np.ndarray:
    """Beams 0..63: non-dyadic weights with sum_e W = 0 (mean removed in float64);
    beams 64..127: |W| of beam b - 64, all in phase (the matched reference)."""
    rng = np.random.default_rng(seed)
    n = rng.uniform(0.3, 1.7, (64, FF, E)) * np.exp(2j * np.pi * rng.random((64, FF, E)))
    n *= np.exp2(rng.uniform(-6, 6, (64, FF)))[:, :, None]
    n -= n.mean(axis=2, keepdims=True)
    w = np.empty((B, FF, E), np.complex128)
    w[:64] = n
    w[64:] = np.abs(n)
    return w.astype(np.complex64)


def coherent_input(seed: int) -> np.ndarray:
    """[T][336][64] packed int4, identical on every element, random in (t, f), no -8."""
    rng = np.random.default_rng(seed)
    re = rng.integers(-7, 8, (T_RAW, F))
    im = rng.integers(-7, 8, (T_RAW, F))
    return np.repeat(sc.encode(re, im)[:, :, None], E, axis=2)


class Fp16WeightsGpu(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not sd.BINARY.is_file():
            raise unittest.SkipTest(f"no kotekan binary at {sd.BINARY}")
        cls.tmp = tempfile.TemporaryDirectory(dir="/dev/shm" if Path("/dev/shm").is_dir() else None)
        cls.root = Path(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_float32(self, name, raw, w, shard):
        wfile = sc.write_weights(self.root / f"{name}_w.bin", w)
        r = sd.run_product(self.root / name, [raw], t_raw=T_RAW, shard=shard, weights_file=wfile,
                           fmt="float32", export_params=False)
        self.assertEqual(len(r["intensity"]), 1, r["log"][-3000:])
        d = ip.load_descriptor(r["descriptor"])
        return d, ip.read_intensity(d, r["intensity"][0])[1], wfile, r["log"]

    def check_device_block(self, d, w, log):
        dev = d["weights"]["device"]
        _, _, scale, eff = sc.fp16_device_weights(w)
        nz = scale > 0
        wd = w.astype(np.complex128)
        rel = np.abs(eff - wd) / np.sqrt(np.max(np.abs(wd) ** 2, axis=-1))[..., None].clip(1e-300)
        rel = rel[nz]
        m = {"descriptor_sha256": dev["effective_sha256"],
             "python_sha256": sc.effective_weights_sha256(w),
             "zero_beam_channels": [dev["zero_beam_channels"], int(np.sum(~nz))],
             "max_relative_error": [dev["max_relative_error"], float(rel.max())],
             "rms_relative_error": [dev["rms_relative_error"], float(np.sqrt(np.mean(rel ** 2)))],
             "subnormal": dev["fp16_subnormal_components"],
             "flushed": dev["fp16_flushed_components"],
             "logged": "fp16 weights: max rel error" in log}
        self.assertEqual(dev["precision"], "fp16")
        self.assertEqual(m["descriptor_sha256"], m["python_sha256"], m)
        self.assertEqual(*m["zero_beam_channels"], m)
        np.testing.assert_allclose(*m["max_relative_error"], rtol=1e-9)
        np.testing.assert_allclose(*m["rms_relative_error"], rtol=1e-6)
        self.assertLessEqual(m["max_relative_error"][0], 2.0 ** -10, m)   # <= one fp16 ulp at 1
        self.assertTrue(m["logged"], m)
        return m, scale

    # ------------------------------------------------------------------ distinct (b, f, e)
    def check_distinct(self, shard):
        raw, _ = sc.fine_fixture(T_RAW, seed=900 + shard)
        w = distinct_weights(seed=90 + shard)
        d, got, wfile, log = self.run_float32(f"distinct{shard}", raw, w, shard)
        m, scale = self.check_device_block(d, w, log)
        x16, eps = sc.fine_fp16(raw, None)
        cmp = sc.compare(got, *sc.expected_power(x16, eps, w, 10))
        zero_bf = (scale == 0).T                                       # [FF][B]
        m.update({"shard": shard, "scale_log2_range": [float(np.log2(scale[scale > 0]).min()),
                                                       float(np.log2(scale[scale > 0]).max())],
                  "compare": cmp,
                  "zero_bf_outputs_max": float(np.abs(got[:, zero_bf]).max()),
                  "zero_bf_count": int(zero_bf.sum())})
        # Negative controls: the right weights at the wrong fine channel or beam.
        rolled = np.roll(w, 1, axis=1)
        swapped = w.copy()
        swapped[0:64:2], swapped[1:64:2] = w[1:64:2], w[0:64:2]
        m["wrong_channel_violations"] = sc.compare(got, *sc.expected_power(x16, eps, rolled, 10))["violations"]
        m["wrong_beam_violations"] = sc.compare(got, *sc.expected_power(x16, eps, swapped, 10))["violations"]
        record(f"w2_distinct_shard{shard}", m)
        self.assertEqual(cmp["violations"], 0, m)
        self.assertEqual(m["zero_bf_outputs_max"], 0.0, m)
        self.assertGreater(m["zero_bf_count"], 1000, m)
        # 2^U(-10, 10) times max_e |w| (~2..4 for 64 complex Gaussians): about 20 octaves.
        self.assertGreaterEqual(m["scale_log2_range"][1] - m["scale_log2_range"][0], 18, m)
        self.assertGreater(m["wrong_channel_violations"], 10000, m)
        self.assertGreater(m["wrong_beam_violations"], 10000, m)
        return raw, w, wfile, got

    def test_distinct_weights_shard0(self):
        self.check_distinct(0)

    def test_distinct_weights_shard1_and_int8_zero_tile(self):
        raw, w, wfile, got = self.check_distinct(1)
        r = sd.run_product(self.root / "distinct1_int8", [raw], t_raw=T_RAW, shard=1,
                           weights_file=wfile, fmt="int8")
        self.assertEqual(len(r["intensity"]), 1, r["log"][-3000:])
        d = ip.load_descriptor(r["descriptor"])
        codes = ip.read_intensity(d, r["intensity"][0])
        params = ip.read_quant_params(d, r["qparams"][0])
        decoded = ip.decode_int8(d, codes, params, shard=1)
        m = {"zero_tile_max_decoded": float(np.abs(decoded[:, :, ZERO_TILE]).max()),
             "zero_tile_codes_nonzero": int(np.count_nonzero(codes[1][:, :, ZERO_TILE])),
             "float32_zero_tile_max": float(np.abs(got[:, :, ZERO_TILE]).max())}
        record("w2_int8_zero_tile", m)
        self.assertEqual(m["float32_zero_tile_max"], 0.0, m)
        self.assertEqual(m["zero_tile_max_decoded"], 0.0, m)
        self.assertEqual(m["zero_tile_codes_nonzero"], 0, m)

    # ------------------------------------------------------------------ nulls
    def test_nulls_at_least_60_dB(self):
        raw = coherent_input(seed=77)
        w = null_weights(seed=78)
        d, got, _, log = self.run_float32("nulls", raw, w, 0)
        m, _ = self.check_device_block(d, w, log)
        x16, eps = sc.fine_fp16(raw, None)
        cmp = sc.compare(got, *sc.expected_power(x16, eps, w, 10))
        p_null = got[:, :, :64].astype(np.float64)
        p_ref = got[:, :, 64:].astype(np.float64)
        live = p_ref > 0
        depth_db = 10 * np.log10(np.maximum(p_null[live], 1e-300) / p_ref[live])
        # What the float64 weights would give (identical voltages: |sum_e W|^2 / (sum_e |W|)^2).
        wd = w.astype(np.complex128)
        ideal = np.abs(wd[:64].sum(-1)) ** 2 / np.abs(wd[:64]).sum(-1) ** 2
        _, _, _, eff = sc.fp16_device_weights(w)
        model = np.abs(eff[:64].sum(-1)) ** 2 / np.abs(eff[64:].sum(-1)) ** 2
        m.update({"compare": cmp, "outputs": int(live.sum()),
                  "null_depth_dB_max": float(depth_db.max()),
                  "null_depth_dB_median": float(np.median(depth_db)),
                  "float32_weights_depth_dB_max": float(10 * np.log10(ideal.max())),
                  "fp16_weights_model_depth_dB_max": float(10 * np.log10(model.max()))})
        record("w2_nulls", m)
        self.assertEqual(cmp["violations"], 0, m)
        self.assertGreater(m["outputs"], 0.9 * p_ref.size, m)
        self.assertLessEqual(m["null_depth_dB_max"], -60.0, m)

    # ------------------------------------------------------------------ rejection
    def test_reject_scale_below_float32_range(self):
        w = np.zeros((B, FF, E), np.complex64)
        w[0, :, 0] = 1
        w[1, 5, 3] = np.float32(1e-39)            # needs s = 2^-129: below the float32 range
        wfile = sc.write_weights(self.root / "tiny_w.bin", w)
        r = sd.run_product(self.root / "tiny", [np.zeros((T_RAW, F, E), np.uint8)], t_raw=T_RAW,
                           shard=0, weights_file=wfile, fmt="float32", export_params=False)
        record("w2_reject_tiny", {"rc": r["rc"], "outputs": len(r["intensity"])})
        self.assertNotEqual(r["rc"], 0)
        self.assertFalse(r["intensity"])
        self.assertIn("weight magnitude outside the float32 normal range", r["log"])


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
