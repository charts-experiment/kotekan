"""Versioned product descriptor and reconstructable int8 output.

Run with /usr/bin/python3 -m test_charts.tests.test_stage_d_product_gpu [--results-dir PATH].

Acceptance:
* cudaChartsBeamformerCommand writes charts-intensity-product-v1 (product_descriptor_file);
  its shape/dtype/U/K/frequency map/timing/weights identity/exposure/int8 sections match the
  configuration and the frame files, and test_charts.sim.intensity_product round-trips them.
* int8 offset/scale per 1x16x16 chunk is exported per frame (gpu_mem_quant_params ->
  existing cudaOutputData -> out_buf_quant_params) with the same raw metadata as the codes.
  The reference is the same GPU float32 product (output_format: float32, same input):
  |I_hat - I| <= scale/2 + float32 allowance; finite codes -127..127; chunk extremes at the
  endpoints; constant chunks have scale 0 and reconstruct exactly; a chunk containing a
  non-finite value is all invalid (-128 marks it); the parameter order and frame pairing are
  checked with negative controls; records are paired by FPGA_SEQ_NUM through ring wrap.
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
import yaml

from test_charts.sim import intensity_product as ip
from test_charts.sim.packet_loss import create_packet_loss_yaml, write_frame
from test_charts.sim.pipeline import create_beamformer_yaml, execute_kotekan
from test_charts.tests import test_stage_c_numerics_gpu as sc
from test_charts.tests import test_weights_replay_gpu as rp

BINARY = sc.ROOT / "build/kotekan/kotekan"
RESULTS = None
B, E, F, FF = 64, 64, 336, 336 * 32
EPS32 = 2.0 ** -24


def record(name, payload):
    if RESULTS:
        (RESULTS / f"{name}.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")


def run_product(work: Path, frames, *, mode, t_raw, k, shard, weights_file, fmt,
                depth=3, export_params=True, extra=None):
    """rawFileRead -> cudaChartsBeamformerCommand -> intensity (+ int8 params) files."""
    shutil.rmtree(work, ignore_errors=True)
    inp, out = work / "in", work / "out"
    inp.mkdir(parents=True)
    out.mkdir(parents=True)
    for i, frame in enumerate(frames):
        write_frame(inp / f"voltage.shard{shard}_{i:07d}.bin", frame, i * t_raw, shard)
    path = create_beamformer_yaml(
        work / "base.yaml", inp, f"voltage.shard{shard}", out, f"intensity.shard{shard}",
        len(frames), samples_per_data_set=t_raw, integration_spectra=k, input_mode=mode,
        output_format=fmt, shard_id=shard, buffer_depth=depth, weights_file=weights_file)
    cfg = yaml.safe_load(path.read_text())
    gpu = cfg["gpu_beamformer"]
    command = next(c for c in gpu["commands"] if c["name"] == "cudaChartsBeamformerCommand")
    descriptor = out / f"intensity.shard{shard}.product.json"
    command["product_descriptor_file"] = descriptor.as_posix()
    command.update(extra or {})
    if fmt == "int8" and export_params:
        n_spec = t_raw // 32 if mode == "fine" else t_raw
        n_out = -(-n_spec // k)
        n_freq = FF if mode == "fine" else F
        count = n_out * (n_freq // 16) * (B // 16)
        cfg["qparams_buf"] = dict(cfg["intensity_buf"], frame_size=count * 8)
        gpu["out_buffers"]["qparams"] = "qparams_buf"
        command.update(gpu_mem_quant_params=f"charts.qparams.shard{shard}",
                       out_buf_quant_params="qparams")
        gpu["commands"].append({"name": "cudaOutputData", "in_buf": "raw",
                                "gpu_mem": f"charts.qparams.shard{shard}", "out_buf": "qparams"})
        cfg["write_qparams"] = dict(cfg["write_intensity"], in_buf="qparams_buf",
                                    file_name=f"qparams.shard{shard}")
    for c in gpu["commands"]:  # as in the intensity + exposure configuration
        c["cuda_stream"] = {"cudaInputData": 0, "cudaChartsBeamformerCommand": 2,
                            "cudaOutputData": 1}.get(c["name"], c.get("cuda_stream"))
        if c["cuda_stream"] is None:
            del c["cuda_stream"]
    (work / "run.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    rc = execute_kotekan(work / "run.yaml", BINARY, work / "kotekan.log", timeout_s=600)
    return {"rc": rc, "log": (work / "kotekan.log").read_text(errors="replace"),
            "descriptor": descriptor,
            "intensity": sorted(out.glob(f"intensity.shard{shard}_*.bin")),
            "qparams": sorted(out.glob(f"qparams.shard{shard}_*.bin"))}


def reconstruction_metrics(d, i_float, codes, table):
    """Bound check of I_hat against the GPU float32 product of the same input.

    Allowance: the kernel forms (I - offset)/scale in float32, |I - offset| <= 127.5*scale, so
    the subtraction and division contribute about 2 * 128 * 2^-24 * scale = 1.5e-5 * scale
    (taken as 1e-4 * scale, 6x margin), plus 2^-24 * (|I| + |offset|) for the subtraction of
    large pedestals (taken 4x). Observed: max error 1.00005 * scale/2.
    """
    idx = ip.chunk_index(d)
    offset = table[:, 0].astype(np.float64)[idx]
    scale = table[:, 1].astype(np.float64)[idx]
    i_ref = i_float.astype(np.float64)
    i_hat = offset + codes.astype(np.float64) * scale
    allowance = 1e-4 * scale + 4 * EPS32 * (np.abs(i_ref) + np.abs(offset))
    err = np.abs(i_hat - i_ref)
    bound = scale / 2 + allowance
    n_chunks = table.shape[0]
    flat_idx, flat_q = idx.reshape(-1), codes.reshape(-1)
    q_min = np.full(n_chunks, 999)
    q_max = np.full(n_chunks, -999)
    np.minimum.at(q_min, flat_idx, flat_q)
    np.maximum.at(q_max, flat_idx, flat_q)
    varying = table[:, 1] > 0
    in_low = i_ref <= offset + (-127 + 12.75) * scale            # lowest 10% of the code range
    return {
        "outputs": int(err.size), "chunks": int(n_chunks),
        "violations": int(np.sum(err > bound)),
        "max_err_over_half_scale": float(np.max(np.where(scale > 0, err / np.maximum(scale, 1e-300) * 2, 0))),
        "constant_chunks": int(np.sum(~varying)),
        "constant_chunks_exact": bool(np.all(err[scale == 0] == 0)),
        "finite_code_range": [int(codes.min()), int(codes.max())],
        "invalid_codes": int(np.sum(codes == ip.INVALID_CODE)),
        "chunk_extremes_at_endpoints": bool(np.all(q_min[varying] == -127) and np.all(q_max[varying] == 127)),
        "normalized_rms": float(np.sqrt(np.mean((i_hat - i_ref) ** 2)) / np.sqrt(np.mean(i_ref ** 2))),
        "normalized_rms_definition": "rms(I_hat - I) / rms(I) over all outputs of the frame",
        "low_power_bias_over_scale": float(np.mean(((i_hat - i_ref) / np.where(scale > 0, scale, 1))[in_low & (scale > 0)])),
        "low_power_bias_definition": "mean((I_hat - I)/scale) over values in the lowest 10% of their chunk's code range",
    }


class ProductGpu(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not BINARY.is_file():
            raise unittest.SkipTest(f"no kotekan binary at {BINARY}")
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def assert_ok(self, run, n):
        self.assertEqual(len(run["intensity"]), n, run["log"][-3000:])

    # ------------------------------------------------------------------ fine round trip
    def test_fine_int8_round_trip_through_ring_wrap(self):
        t_raw, n_frames, depth = 640, 5, 2
        w = sc.probe_weights(seed=31)
        wfile = sc.write_weights(self.root / "fine_w.bin", np.broadcast_to(w[:, None, :], (B, FF, E)))
        frames = [sc.fine_fixture(t_raw, seed=500, n0=i * t_raw, frame_seed=5000 + i)[0]
                  for i in range(n_frames)]
        common = dict(mode="fine", t_raw=t_raw, k=10, shard=1, weights_file=wfile, depth=depth)
        ref = run_product(self.root / "fine_f32", frames, fmt="float32", **common)
        q8 = run_product(self.root / "fine_i8", frames, fmt="int8", **common)
        self.assert_ok(ref, n_frames)
        self.assert_ok(q8, n_frames)
        self.assertEqual(len(q8["qparams"]), n_frames)
        d = ip.load_descriptor(q8["descriptor"])
        d_ref = ip.load_descriptor(ref["descriptor"])
        self.check_descriptor(d, mode="fine", t_raw=t_raw, k=10, shard=1, fmt="int8")
        self.check_descriptor(d_ref, mode="fine", t_raw=t_raw, k=10, shard=1, fmt="float32")
        self.assertTrue(d["int8"]["reconstructable"])
        self.assertEqual(d["weights"]["source"], "legacy_bin")

        report = {"frames": n_frames, "depth": depth, "per_frame": []}
        decoded = []
        for i in range(n_frames):
            seq_r, i_float = ip.read_intensity(d_ref, ref["intensity"][i])
            codes = ip.read_intensity(d, q8["intensity"][i])
            params = ip.read_quant_params(d, q8["qparams"][i])
            self.assertEqual(codes[0], params[0])
            self.assertEqual(codes[0], seq_r)
            self.assertEqual(codes[0], (1, i * t_raw))
            m = reconstruction_metrics(d, i_float, codes[1], params[1])
            i_hat = ip.decode_int8(d, codes, params, shard=1)
            # The published decoder itself must satisfy the same bound.
            idx = ip.chunk_index(d)
            half = params[1][idx, 1].astype(np.float64) / 2
            m["decoder_violations"] = int(np.sum(~(np.abs(i_hat - i_float) <= half
                                                   + 1e-4 * 2 * half + 4 * EPS32 * (np.abs(i_float) + np.abs(params[1][idx, 0])))))
            decoded.append((codes, params, i_float))
            report["per_frame"].append(m)
            self.assertEqual(m["violations"], 0, m)
            self.assertEqual(m["decoder_violations"], 0, m)
            self.assertEqual(m["invalid_codes"], 0, m)
            self.assertTrue(m["chunk_extremes_at_endpoints"], m)
            self.assertGreater(m["constant_chunks"], 0, m)     # late-start channels, window 0
            self.assertTrue(m["constant_chunks_exact"], m)
            self.assertGreaterEqual(m["finite_code_range"][0], -127)

        # Negative controls: wrong parameter order and wrong frame pairing must be detected.
        codes, params, i_float = decoded[2]
        p = d["int8"]["parameters"]
        t_tiles = d["intensity"]["shape"][0]
        swapped = params[1].reshape(t_tiles, p["frequency_tiles"], p["beam_tiles"], 2)
        swapped = swapped[:, :, ::-1].reshape(-1, 2)
        report["swapped_beam_tile_violations"] = reconstruction_metrics(
            d, i_float, codes[1], swapped)["violations"]
        report["frame_i_plus_1_params_violations"] = reconstruction_metrics(
            d, i_float, codes[1], decoded[3][1][1])["violations"]
        with self.assertRaises(ip.ProductError):
            ip.decode_int8(d, codes, decoded[3][1])
        with self.assertRaises(ip.ProductError):
            ip.decode_int8(d, codes, params, shard=0)
        with self.assertRaises(ip.ProductError):   # same sequence number, other shard
            ip.decode_int8(d, codes, ((0, params[0][1]), params[1]))
        record("d_fine_int8_round_trip", report)
        self.assertGreater(report["swapped_beam_tile_violations"], 1000)
        self.assertGreater(report["frame_i_plus_1_params_violations"], 1000)

    def check_descriptor(self, d, *, mode, t_raw, k, shard, fmt):
        u = 32 if mode == "fine" else 1
        n_spec = t_raw // u
        n_freq = F * u
        self.assertEqual((d["input_mode"], d["T_raw"], d["K"], d["U"], d["shard"]),
                         (mode, t_raw, k, u, shard))
        self.assertEqual(d["intensity"]["dtype"], fmt)
        self.assertEqual(d["intensity"]["shape"], [-(-n_spec // k), n_freq, B])
        self.assertEqual(d["tail_supplied_spectra"], n_spec - (-(-n_spec // k) - 1) * k)
        freqs = ip.frequencies_hz(d)
        self.assertEqual(freqs.size, n_freq)
        self.assertAlmostEqual(d["frequency"]["first_hz"], freqs[0], delta=1e-3)
        self.assertAlmostEqual(d["frequency"]["last_hz"], freqs[-1], delta=1e-3)
        self.assertEqual(d["association"]["key"], ["shard", "FPGA_SEQ_NUM", "T_raw"])
        self.assertEqual(d["timing"]["raw_tick_seconds"], [10, 3000000])
        self.assertTrue(d["kotekan_git_commit"])
        if fmt == "float32":
            self.assertIsNone(d["int8"])

    # ------------------------------------------------------------------ coarse cases
    def coarse_weights(self):
        w = np.zeros((B, F, E), np.complex64)
        w[:16, :, 0] = 1                                   # beams 0..15 identical
        rng = np.random.default_rng(41)
        w[16:] = rng.normal(size=(B - 16, F, E)) + 1j * rng.normal(size=(B - 16, F, E))
        return w

    def coarse_input(self, t_raw):
        rng = np.random.default_rng(42)
        raw = rng.integers(0, 256, (t_raw, F, E), dtype=np.uint8)
        raw[:, :16, :] = 0x23                              # (3, 2) in channels 0..15
        return raw

    def test_coarse_constant_chunk_and_nonfinite_chunk(self):
        t_raw, k = 650, 320
        w = self.coarse_weights()
        w_inf = w.copy()
        w_inf[20, 40, :] = 1e20                            # |V|^2 overflows to inf
        raw = self.coarse_input(t_raw)
        out = {}
        for label, weights in (("finite", w), ("inf", w_inf)):
            wfile = sc.write_weights(self.root / f"coarse_{label}.bin", weights)
            for fmt in ("float32", "int8"):
                run = run_product(self.root / f"coarse_{label}_{fmt}", [raw], mode="coarse",
                                  t_raw=t_raw, k=k, shard=0, weights_file=wfile, fmt=fmt)
                self.assert_ok(run, 1)
                out[label, fmt] = run
        d = ip.load_descriptor(out["finite", "int8"]["descriptor"])
        d_f32 = ip.load_descriptor(out["finite", "float32"]["descriptor"])
        self.check_descriptor(d, mode="coarse", t_raw=t_raw, k=k, shard=0, fmt="int8")
        self.check_descriptor(d_f32, mode="coarse", t_raw=t_raw, k=k, shard=0, fmt="float32")
        report = {}
        for label in ("finite", "inf"):
            _, i_float = ip.read_intensity(d_f32, out[label, "float32"]["intensity"][0])
            codes = ip.read_intensity(d, out[label, "int8"]["intensity"][0])
            params = ip.read_quant_params(d, out[label, "int8"]["qparams"][0])
            i_hat = ip.decode_int8(d, codes, params)
            idx = ip.chunk_index(d)
            const_chunk = idx[:, 0, 0]                     # f-tile 0 (channels 0..15), beams 0..15
            m = {"constant_chunk_scale": [float(params[1][c, 1]) for c in const_chunk],
                 "constant_chunk_exact": bool(np.array_equal(i_hat[:, :16, :16],
                                                              i_float[:, :16, :16].astype(np.float64))),
                 "constant_chunk_value": float(i_float[0, 0, 0])}
            if label == "inf":
                bad = np.unique(idx[~np.isfinite(i_float)])
                m.update({
                    "nonfinite_outputs_float32": int(np.sum(~np.isfinite(i_float))),
                    "invalid_chunks": int(bad.size),
                    "invalid_codes_match_nonfinite": bool(np.array_equal(
                        codes[1] == ip.INVALID_CODE, ~np.isfinite(i_float))),
                    "invalid_chunk_params_zero": bool(np.all(params[1][bad] == 0)),
                    "finite_in_invalid_chunks_coded_0": bool(np.all(
                        codes[1][np.isin(idx, bad) & np.isfinite(i_float)] == 0)),
                    "decoder_nan_exactly_invalid_chunks": bool(np.array_equal(
                        np.isnan(i_hat), np.isin(idx, bad))),
                })
                good = ~np.isin(idx, bad)
                rm = reconstruction_metrics(d, np.where(good, i_float, 0), np.where(good, codes[1], 0), params[1])
                m["valid_chunk_violations"] = rm["violations"]
            else:
                m.update(reconstruction_metrics(d, i_float, codes[1], params[1]))
            report[label] = m
        record("d_coarse_constant_and_nonfinite", report)
        for label in ("finite", "inf"):
            self.assertTrue(all(s == 0 for s in report[label]["constant_chunk_scale"]), report)
            self.assertTrue(report[label]["constant_chunk_exact"], report)
            self.assertGreater(report[label]["constant_chunk_value"], 0)
        self.assertEqual(report["finite"]["violations"], 0, report)
        r = report["inf"]
        self.assertGreater(r["nonfinite_outputs_float32"], 0, r)
        for key in ("invalid_codes_match_nonfinite", "invalid_chunk_params_zero",
                    "finite_in_invalid_chunks_coded_0", "decoder_nan_exactly_invalid_chunks"):
            self.assertTrue(r[key], (key, r))
        self.assertEqual(r["valid_chunk_violations"], 0, r)

    def test_npy_weights_identity_in_descriptor(self):
        directions = rp.bw.load_directions(rp.ROOT / "charts/config/charts_hex64_beam_directions.json")
        correction, flagged = rp.bw.gain_correction(rp.make_gains())
        weights = rp.bw.build_weights("coarse", 0, correction, rp.bw.default_positions_charts_logical(),
                                      directions)
        npy = self.root / "w.npy"
        np.save(npy, weights)
        desc = rp.bw.descriptor("coarse", 0, npy, directions.shape[0], flagged,
                                {"gain_convention": "response", "gains": "synthetic",
                                 "calibration_note": "stage D identity"})
        (self.root / "w.json").write_text(json.dumps(desc))
        raw = self.coarse_input(320)
        run = run_product(self.root / "npy_identity", [raw], mode="coarse", t_raw=320, k=320,
                          shard=0, weights_file=npy, fmt="float32")
        self.assert_ok(run, 1)
        d = ip.load_descriptor(run["descriptor"])
        record("d_npy_identity_descriptor", d)
        self.assertEqual(d["weights"]["source"], "npy")
        self.assertEqual(d["weights"]["sha256"], desc["sha256"])
        self.assertTrue(d["weights"]["sha256_verified"])
        self.assertEqual(d["weights"]["calibration_note"], "stage D identity")
        self.assertTrue(d["weights"]["calibrated"])
        self.assertIsNone(d["exposure"])

    def test_int8_without_exported_parameters(self):
        raw = self.coarse_input(320)
        wfile = sc.write_weights(self.root / "noexp.bin", self.coarse_weights())
        run = run_product(self.root / "noexp", [raw], mode="coarse", t_raw=320, k=320, shard=0,
                          weights_file=wfile, fmt="int8", export_params=False)
        self.assert_ok(run, 1)
        d = ip.load_descriptor(run["descriptor"])
        self.assertIn("without exported offset/scale", run["log"])
        self.assertFalse(d["int8"]["reconstructable"])
        with self.assertRaises(ip.ProductError):
            ip.read_quant_params(d, run["intensity"][0])

    def test_rejected_configurations(self):
        raw = self.coarse_input(320)
        wfile = sc.write_weights(self.root / "rej.bin", self.coarse_weights())
        cases = {
            "params_in_float32": (dict(fmt="float32", extra={
                "gpu_mem_quant_params": "charts.q", "out_buf_quant_params": "intensity"}),
                "require output_format: int8"),
            "params_buffer_size": (dict(fmt="int8", extra={
                "gpu_mem_quant_params": "charts.q", "out_buf_quant_params": "intensity"},
                export_params=False), "quant params host buffer frame_size mismatch"),
            "params_without_buffer": (dict(fmt="int8", extra={
                "gpu_mem_quant_params": "charts.q"}, export_params=False),
                "gpu_mem_quant_params requires out_buf_quant_params"),
            "buffer_without_params": (dict(fmt="int8", extra={
                "out_buf_quant_params": "intensity"}, export_params=False),
                "out_buf_quant_params requires gpu_mem_quant_params"),
            "params_name_collision": (dict(fmt="int8", extra={
                "gpu_mem_quant_params": "charts.intensity.shard0",
                "out_buf_quant_params": "intensity"}, export_params=False),
                "gpu_mem_quant_params must be distinct"),
        }
        report = {}
        for label, (kwargs, message) in cases.items():
            run = run_product(self.root / f"rej_{label}", [raw], mode="coarse", t_raw=320,
                              k=320, shard=0, weights_file=wfile, **kwargs)
            report[label] = {"rejected": len(run["intensity"]) == 0, "message_found": message in run["log"]}
        record("d_rejected_configurations", report)
        for label, r in report.items():
            self.assertTrue(r["rejected"] and r["message_found"], (label, report))

    def test_descriptor_exposure_section_with_packet_loss(self):
        t_raw = 640
        work = self.root / "exposure"
        inp, out = work / "in", work / "out"
        inp.mkdir(parents=True)
        for shard in (0, 1):
            write_frame(inp / f"voltage.shard{shard}_0000000.bin",
                        np.zeros((t_raw, F, E), np.uint8), 0, shard)
            write_frame(inp / f"missing.shard{shard}_0000000.bin",
                        np.zeros(t_raw, np.uint8), 0, shard, mask=True)
        path = create_packet_loss_yaml(work / "run.yaml", inp, out, 1, mode="fine", raw_n_time=t_raw)
        cfg = yaml.safe_load(path.read_text())
        for shard in (0, 1):
            command = next(c for c in cfg[f"gpu_{shard}"]["commands"]
                           if c["name"] == "cudaChartsBeamformerCommand")
            command["product_descriptor_file"] = (out / f"intensity.shard{shard}.product.json").as_posix()
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        execute_kotekan(path, BINARY, work / "kotekan.log", timeout_s=300)
        report = {}
        for shard in (0, 1):
            d = ip.load_descriptor(out / f"intensity.shard{shard}.product.json")
            report[shard] = d["exposure"]
            self.assertEqual(d["shard"], shard)
            self.assertEqual(d["exposure"]["dtype"], "uint8")
            self.assertEqual(d["exposure"]["shape"], [2, 336])
            self.assertEqual(d["exposure"]["buffer"], "exposure")
            self.assertEqual(d["exposure"]["fine_bins_per_count"], 32)
            self.assertEqual(len(list(out.glob(f"exposure.shard{shard}_*.bin"))), 1)
        record("d_exposure_section", report)


def main():
    global RESULTS, BINARY
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--kotekan-bin", type=Path)
    args, rest = parser.parse_known_args()
    if args.results_dir:
        RESULTS = args.results_dir
        RESULTS.mkdir(parents=True, exist_ok=True)
    if args.kotekan_bin:
        BINARY = args.kotekan_bin
    unittest.main(argv=[sys.argv[0], *rest], verbosity=2)


if __name__ == "__main__":
    main()
