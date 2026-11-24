"""WP14: int8 output of selected 16-beam groups (output_beam_groups), on the GPU.

    PYTHONPATH=. ~/miniconda3/envs/kotekan_test/bin/python -m \
        test_charts.tests.test_beam_selection_gpu --kotekan-bin build-tc/kotekan/kotekan \
        --baseline-bin build-tc/kotekan/kotekan.f7d1968c5 [--results-dir PATH]

Acceptance (BEAM_SELECTION_OUTPUT_PLAN.md §4):
1. Default unchanged: without the key, int8 codes, int8 parameters and float32 intensity
   records are byte-identical to the pre-change binary on the same input and weights; the
   descriptor equals the old one plus the beams block (written = 0..127).
2. Exact selection: with output_beam_groups the codes and parameters equal the corresponding
   slices of the full product of the same binary bit for bit, decode_int8 is equal (NaN in
   the same places), through ring wrap; a constant chunk (scale 0) and a non-finite chunk
   (-128) lie inside selected groups.
3. Both shards in one Kotekan with different selections: own descriptors, records paired by
   FPGA_SEQ_NUM, equal to the slices of a two-shard full run.
4. Invalid selections and mis-sized host buffers are rejected with their message.
Synthetic verification only (U = 32, K = 10, B = 128 computed beams).
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np
import yaml

from test_charts.sim import intensity_product as ip
from test_charts.sim.packet_loss import write_frame
from test_charts.sim.pipeline import create_beamformer_yaml, execute_kotekan
from test_charts.tests import test_stage_c_numerics_gpu as sc

ROOT = sc.ROOT
BINARY = ROOT / "build-tc/kotekan/kotekan"
BASELINE = ROOT / "build-tc/kotekan/kotekan.f7d1968c5"
RESULTS = None
B, E, F, FF, GROUP = 128, 64, 336, 336 * 32, 16
IMPULSE = 0x77            # c = 7 + 7i, see test_stage_d_product_gpu (constant chunk fixture)
INF_BEAM, INF_COARSE = 20, 40   # beam group 1: |V|^2 overflows to inf in coarse channel 40


def record(name, payload):
    if RESULTS:
        (RESULTS / f"{name}.json").write_text(json.dumps(payload, indent=2, default=str) + "\n")


# ---------------------------------------------------------------------------- fixtures
def make_weights(path: Path) -> Path:
    """[B][FF][E] distinct per (b, fine channel, e) for beams 16..127; beams 0..15 identical
    one-hot on element 0 (constant chunk); beam 20 = 1e20 in coarse channel 40 (inf power)."""
    if path.exists():
        return path
    rng = np.random.default_rng(1124)
    with open(path, "wb") as fh:
        for b in range(B):
            if b < GROUP:
                w = np.zeros((FF, E), np.complex64)
                w[:, 0] = 1
            else:
                w = (rng.standard_normal((FF, E), dtype=np.float32)
                     + 1j * rng.standard_normal((FF, E), dtype=np.float32)).astype(np.complex64)
                if b == INF_BEAM:
                    w[INF_COARSE * 32:(INF_COARSE + 1) * 32, :] = 1e20
            w.tofile(fh)
    return path


def make_frames(t_raw: int, n: int, seed: int):
    """Random raw bytes with an impulse train on element 0 of coarse channel 0 at raw
    samples n = 0 mod 32 (constant fine tile 0 for the identical beams 0..15)."""
    rng = np.random.default_rng(seed)
    frames = []
    for _ in range(n):
        raw = rng.integers(0, 256, (t_raw, F, E), dtype=np.uint8)
        raw[:, 0, 0] = 0
        raw[::32, 0, 0] = IMPULSE
        frames.append(raw)
    return frames


# ---------------------------------------------------------------------------- configuration
def shard_config(work: Path, shard: int, frames, *, t_raw: int, weights: Path, fmt="int8",
                 groups=None, depth=2, export_params=True, extra=None,
                 qparam_groups=None, intensity_groups="same"):
    """Config dict of rawFileRead -> cudaChartsBeamformerCommand -> rawFileWrite for one
    shard. qparam_groups / intensity_groups override the host buffer sizes (rejection tests:
    a buffer sized for another G)."""
    inp, out = work / f"in{shard}", work / "out"
    inp.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        write_frame(inp / f"voltage.shard{shard}_{i:07d}.bin", frame, i * t_raw, shard)
    size_groups = groups if intensity_groups == "same" else intensity_groups
    path = create_beamformer_yaml(
        work / f"shard{shard}.yaml", inp, f"voltage.shard{shard}", out,
        f"intensity.shard{shard}", len(frames), samples_per_data_set=t_raw,
        output_format=fmt, shard_id=shard, buffer_depth=depth, weights_file=weights,
        output_beam_groups=size_groups)
    cfg = yaml.safe_load(path.read_text())
    gpu = cfg["gpu_beamformer"]
    command = next(c for c in gpu["commands"] if c["name"] == "cudaChartsBeamformerCommand")
    command.pop("output_beam_groups", None)
    if groups is not None:
        command["output_beam_groups"] = list(groups) if isinstance(groups, (list, tuple)) else groups
    command["product_descriptor_file"] = (out / f"intensity.shard{shard}.product.json").as_posix()
    command.update(extra or {})
    if fmt == "int8" and export_params:
        if qparam_groups is not None:
            g = qparam_groups
        else:
            g = len(groups) if groups is not None else B // GROUP
        n_out = -(-(t_raw // 32) // 10)
        cfg["qparams_buf"] = dict(cfg["intensity_buf"], frame_size=n_out * (FF // 16) * g * 8)
        gpu["out_buffers"]["qparams"] = "qparams_buf"
        command.update(gpu_mem_quant_params=f"charts.qparams.shard{shard}",
                       out_buf_quant_params="qparams")
        gpu["commands"].append({"name": "cudaOutputData", "in_buf": "raw",
                                "gpu_mem": f"charts.qparams.shard{shard}", "out_buf": "qparams"})
        cfg["write_qparams"] = dict(cfg["write_intensity"], in_buf="qparams_buf",
                                    file_name=f"qparams.shard{shard}")
    for c in gpu["commands"]:
        c["cuda_stream"] = {"cudaInputData": 0, "cudaChartsBeamformerCommand": 2,
                            "cudaOutputData": 1}.get(c["name"], c.get("cuda_stream"))
        if c["cuda_stream"] is None:
            del c["cuda_stream"]
    return cfg


def merge(configs):
    """One Kotekan config from per-shard configs (stage/buffer names get a shard suffix)."""
    if len(configs) == 1:
        return configs[0]
    merged = {}
    stages = ("voltage_buf", "intensity_buf", "qparams_buf", "read_voltage", "gpu_beamformer",
              "write_intensity", "write_qparams")
    for shard, cfg in enumerate(configs):
        ren = {name: f"{name}_s{shard}" for name in stages if name in cfg}
        for key, value in cfg.items():
            if key in ren:
                merged[ren[key]] = value
            else:
                merged.setdefault(key, value)
        merged[ren["read_voltage"]]["buf"] = ren["voltage_buf"]
        gpu = merged[ren["gpu_beamformer"]]
        gpu["in_buffers"]["raw"] = ren["voltage_buf"]
        gpu["out_buffers"]["intensity"] = ren["intensity_buf"]
        merged[ren["write_intensity"]]["in_buf"] = ren["intensity_buf"]
        if "qparams_buf" in ren:
            gpu["out_buffers"]["qparams"] = ren["qparams_buf"]
            merged[ren["write_qparams"]]["in_buf"] = ren["qparams_buf"]
    return merged


def log_errors(log: str):
    """ERROR lines except the thread affinity/name notices that every test config emits
    (cpu_affinity: [] in the generated YAML)."""
    return [l for l in log.splitlines() if "ERROR" in l
            and "Failed to set thread affinity" not in l and "Failed to set thread name" not in l]


def run(work: Path, specs, *, binary=None, timeout_s=600):
    """specs: list of (shard, frames, kwargs for shard_config). Returns per-shard outputs."""
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    cfg = merge([shard_config(work, shard, frames, **kw) for shard, frames, kw in specs])
    (work / "run.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    rc = execute_kotekan(work / "run.yaml", binary or BINARY, work / "kotekan.log",
                         timeout_s=timeout_s)
    out = work / "out"
    log = (work / "kotekan.log").read_text(errors="replace")
    # Release builds compile out the size assert of gpuDeviceInterface::get_gpu_memory*: a
    # GPU-memory size mismatch between commands only shows up as an ERROR line.
    result = {"rc": rc, "log": log, "shards": {}, "errors": log_errors(log)}
    for shard, _, _ in specs:
        result["shards"][shard] = {
            "descriptor": out / f"intensity.shard{shard}.product.json",
            "intensity": sorted(out.glob(f"intensity.shard{shard}_*.bin")),
            "qparams": sorted(out.glob(f"qparams.shard{shard}_*.bin"))}
    return result


def load_records(shard_out):
    d = ip.load_descriptor(shard_out["descriptor"])
    codes = [ip.read_intensity(d, p) for p in shard_out["intensity"]]
    params = ([ip.read_quant_params(d, p) for p in shard_out["qparams"]]
              if d["intensity"]["dtype"] == "int8" else [])
    return d, codes, params


def written(groups):
    return np.concatenate([np.arange(g * GROUP, (g + 1) * GROUP) for g in groups])


def compare_selection(full, sel, groups, shard):
    """Bitwise comparison of a selected product against the slices of the full one."""
    d_full, c_full, p_full = full
    d_sel, c_sel, p_sel = sel
    cols = written(groups)
    f_tiles = d_full["int8"]["parameters"]["frequency_tiles"]
    m = {"groups": list(groups), "frames": len(c_sel), "frames_full": len(c_full),
         "codes_bit_identical": [], "params_bit_identical": [], "decode_equal": [],
         "keys_equal": [], "invalid_chunks": 0, "constant_chunks": 0}
    for (kq, q), (kp, t), (kfq, fq), (kfp, ft) in zip(c_sel, p_sel, c_full, p_full):
        m["keys_equal"].append(kq == kp == kfq == kfp)
        m["codes_bit_identical"].append(bool(np.array_equal(q, fq[:, :, cols])))
        t4 = ft.reshape(-1, f_tiles, B // GROUP, 2)[:, :, groups, :].reshape(-1, 2)
        m["params_bit_identical"].append(bool(np.array_equal(t.view(np.uint32),
                                                             t4.view(np.uint32))))
        a = ip.decode_int8(d_sel, (kq, q), (kp, t), shard=shard)
        b = ip.decode_int8(d_full, (kfq, fq), (kfp, ft), shard=shard)[:, :, cols]
        m["decode_equal"].append(bool(np.array_equal(a, b, equal_nan=True)))
        m["invalid_chunks"] += int(np.unique(ip.chunk_index(d_sel)[q == ip.INVALID_CODE]).size)
        m["constant_chunks"] += int(np.sum((t[:, 1] == 0) & (t[:, 0] != 0)))
    m["descriptor"] = {"shape": d_sel["intensity"]["shape"],
                       "beam_tiles": d_sel["int8"]["parameters"]["beam_tiles"],
                       "count": d_sel["int8"]["parameters"]["count"],
                       "beams": d_sel["beams"]}
    return m


# ---------------------------------------------------------------------------- tests
class BeamSelectionGpu(unittest.TestCase):
    T_RAW, N_FRAMES, DEPTH = 640, 5, 2           # 20 spectra: windows 10, 10; ring wraps twice

    @classmethod
    def setUpClass(cls):
        if not BINARY.is_file():
            raise unittest.SkipTest(f"no kotekan binary at {BINARY}")
        cls.tmp = tempfile.TemporaryDirectory(dir="/dev/shm")
        cls.root = Path(cls.tmp.name)
        cls.weights = make_weights(cls.root / "weights.bin")
        cls.frames = {s: make_frames(cls.T_RAW, cls.N_FRAMES, seed=700 + s) for s in (0, 1)}
        cls.full = {}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def common(self, **kw):
        return dict(t_raw=self.T_RAW, weights=self.weights, depth=self.DEPTH, **kw)

    def full_run(self, shard):
        if shard not in self.full:
            r = run(self.root / f"full{shard}", [(shard, self.frames[shard], self.common())])
            self.assertEqual(r["errors"], [], r["errors"])
            self.assertEqual(len(r["shards"][shard]["intensity"]), self.N_FRAMES, r["log"][-3000:])
            self.full[shard] = r
        return self.full[shard]

    # 1 ---------------------------------------------------------------- default unchanged
    def test_1_default_byte_identical_to_baseline(self):
        if not BASELINE.is_file():
            self.skipTest(f"no baseline binary at {BASELINE}")
        report = {}
        for fmt in ("int8", "float32"):
            spec = [(1, self.frames[1], self.common(fmt=fmt))]
            new = run(self.root / f"t1_new_{fmt}", spec)
            old = run(self.root / f"t1_old_{fmt}", spec, binary=BASELINE)
            sn, so = new["shards"][1], old["shards"][1]
            r = {"frames": [len(sn["intensity"]), len(so["intensity"])],
                 "errors": new["errors"] + old["errors"]}
            for kind in ("intensity", "qparams"):
                r[f"{kind}_files_identical"] = [a.read_bytes() == b.read_bytes()
                                                for a, b in zip(sn[kind], so[kind])]
                r[f"{kind}_count"] = [len(sn[kind]), len(so[kind])]
            dn = json.loads(sn["descriptor"].read_text())
            do = json.loads(so["descriptor"].read_text())
            beams = dn.pop("beams", None)
            for d in (dn, do):
                d.pop("kotekan_git_commit", None)
            r["descriptor_equal_except_beams"] = dn == do
            r["beams"] = beams
            r["log_line"] = [line for line in new["log"].splitlines()
                             if "int8 output writes" in line]
            report[fmt] = r
        record("wp14_1_default_unchanged", report)
        for fmt, r in report.items():
            self.assertEqual(r["frames"], [self.N_FRAMES] * 2, (fmt, r))
            self.assertEqual(r["errors"], [], (fmt, r))
            self.assertTrue(all(r["intensity_files_identical"]), (fmt, r))
            self.assertTrue(r["descriptor_equal_except_beams"], (fmt, r))
            self.assertEqual(r["beams"]["groups"], list(range(8)), (fmt, r))
            self.assertEqual(r["beams"]["written"], list(range(B)), (fmt, r))
            self.assertEqual(r["beams"]["computed"], B, (fmt, r))
        self.assertEqual(report["int8"]["qparams_count"], [self.N_FRAMES] * 2)
        self.assertTrue(all(report["int8"]["qparams_files_identical"]), report)
        self.assertEqual(len(report["int8"]["log_line"]), 1, report)
        self.assertIn("(default, all)", report["int8"]["log_line"][0])
        self.assertEqual(report["float32"]["log_line"], [], report)

    # 2 ---------------------------------------------------------------- exact selection
    def test_2_selection_bit_identical_to_full_slices(self):
        full = load_records(self.full_run(1)["shards"][1])
        d_full = full[0]
        self.assertEqual(d_full["beams"]["written"], list(range(B)))
        report = {}
        for groups in ([0, 3, 7], [5], [0, 1, 6]):
            r = run(self.root / f"t2_{'_'.join(map(str, groups))}",
                    [(1, self.frames[1], self.common(groups=groups))])
            s = r["shards"][1]
            self.assertEqual(len(s["intensity"]), self.N_FRAMES, r["log"][-3000:])
            self.assertEqual(r["errors"], [], r["errors"])
            m = compare_selection(full, load_records(s), groups, shard=1)
            m["log_line"] = [l for l in r["log"].splitlines() if "int8 output writes" in l]
            m["intensity_bytes"] = s["intensity"][0].stat().st_size
            report[str(groups)] = m
        # Fixture checks on the full product: one constant chunk group (0), one invalid (1).
        _, c_full, p_full = full
        f_tiles = d_full["int8"]["parameters"]["frequency_tiles"]
        t = p_full[0][1].reshape(-1, f_tiles, 8, 2)
        report["full_fixture"] = {
            "constant_chunks_per_group": np.sum((t[..., 1] == 0) & (t[..., 0] != 0),
                                                axis=(0, 1)).tolist(),
            "invalid_codes_per_group": [int(np.sum(c_full[0][1][:, :, g * 16:(g + 1) * 16]
                                                   == ip.INVALID_CODE)) for g in range(8)]}
        record("wp14_2_exact_selection", report)
        fx = report["full_fixture"]
        self.assertGreater(fx["constant_chunks_per_group"][0], 0, fx)
        self.assertGreater(fx["invalid_codes_per_group"][1], 0, fx)
        for groups in ([0, 3, 7], [5], [0, 1, 6]):
            m = report[str(groups)]
            self.assertEqual(m["frames"], self.N_FRAMES, m)
            for key in ("keys_equal", "codes_bit_identical", "params_bit_identical",
                        "decode_equal"):
                self.assertTrue(all(m[key]) and len(m[key]) == self.N_FRAMES, (key, m))
            self.assertEqual(m["descriptor"]["shape"][2], 16 * len(groups))
            self.assertEqual(m["descriptor"]["beam_tiles"], len(groups))
            self.assertEqual(m["descriptor"]["beams"]["written"], written(groups).tolist())
            # Same record header (raw metadata) as the full run; payload 2 windows x 16G beams.
            full_bytes = self.full_run(1)["shards"][1]["intensity"][0].stat().st_size
            self.assertEqual(full_bytes - m["intensity_bytes"],
                             2 * FF * 16 * (8 - len(groups)), m)
            self.assertIn(f"writes {16 * len(groups)} of 128", m["log_line"][0])
        self.assertGreater(report["[0, 3, 7]"]["constant_chunks"], 0)
        self.assertGreater(report["[0, 1, 6]"]["invalid_chunks"], 0)

    # 3 ---------------------------------------------------------------- both shards
    def test_3_two_shards_different_selections(self):
        sel = {0: [0, 3, 7], 1: [1, 6]}
        full = run(self.root / "t3_full",
                   [(s, self.frames[s], self.common()) for s in (0, 1)])
        part = run(self.root / "t3_sel",
                   [(s, self.frames[s], self.common(groups=sel[s])) for s in (0, 1)])
        self.assertEqual(full["errors"] + part["errors"], [])
        report = {}
        for s in (0, 1):
            for r in (full, part):
                self.assertEqual(len(r["shards"][s]["intensity"]), self.N_FRAMES, r["log"][-3000:])
            fr, pr = load_records(full["shards"][s]), load_records(part["shards"][s])
            m = compare_selection(fr, pr, sel[s], shard=s)
            m["seq"] = [k[1] for k, _ in pr[1]]
            m["shard_keys"] = sorted({k[0] for k, _ in pr[1]})
            report[s] = m
        record("wp14_3_two_shards", report)
        for s in (0, 1):
            m = report[s]
            for key in ("keys_equal", "codes_bit_identical", "params_bit_identical",
                        "decode_equal"):
                self.assertTrue(all(m[key]) and len(m[key]) == self.N_FRAMES, (s, key, m))
            self.assertEqual(m["seq"], [i * self.T_RAW for i in range(self.N_FRAMES)], m)
            self.assertEqual(m["descriptor"]["beams"]["groups"], sel[s])
            self.assertIn(m["shard_keys"], ([s], [None]), m)

    # 3b --------------------------------------------------------------- inherited key
    def test_3b_key_inherited_from_cudaprocess_block(self):
        """output_beam_groups at the cudaProcess level applies to the command (Kotekan key
        lookup searches parent blocks), like the other keys of this command."""
        groups = [2, 5]
        work = self.root / "t3b"
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
        cfg = shard_config(work, 0, self.frames[0], **self.common(groups=groups))
        command = next(c for c in cfg["gpu_beamformer"]["commands"]
                       if c["name"] == "cudaChartsBeamformerCommand")
        cfg["gpu_beamformer"]["output_beam_groups"] = command.pop("output_beam_groups")
        (work / "run.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        execute_kotekan(work / "run.yaml", BINARY, work / "kotekan.log", timeout_s=600)
        log = (work / "kotekan.log").read_text(errors="replace")
        out = {"descriptor": work / "out/intensity.shard0.product.json",
               "intensity": sorted((work / "out").glob("intensity.shard0_*.bin")),
               "qparams": sorted((work / "out").glob("qparams.shard0_*.bin"))}
        self.assertEqual(len(out["intensity"]), self.N_FRAMES, log[-3000:])
        self.assertEqual(log_errors(log), [])
        m = compare_selection(load_records(self.full_run(0)["shards"][0]), load_records(out),
                              groups, shard=0)
        record("wp14_3b_inherited_key", m)
        for key in ("codes_bit_identical", "params_bit_identical", "decode_equal"):
            self.assertTrue(all(m[key]) and len(m[key]) == self.N_FRAMES, (key, m))
        self.assertIn("writes 32 of 128 computed beams: groups [2, 5]", log)

    # 4 ---------------------------------------------------------------- rejections
    def test_4_rejected_configurations(self):
        one = [self.frames[0][0]]
        # Invalid lists get full-size host buffers, so only the key validation can reject.
        full = dict(intensity_groups=None, qparam_groups=B // GROUP)
        cases = {
            "empty": (dict(groups=[], **full), "output_beam_groups must not be empty"),
            "duplicate": (dict(groups=[1, 1], **full), "strictly increasing"),
            "decreasing": (dict(groups=[3, 1], **full), "strictly increasing"),
            "out_of_range": (dict(groups=[8], **full), "entries must be in 0..7"),
            "negative": (dict(groups=[-1], **full), "entries must be in 0..7"),
            "too_many": (dict(groups=list(range(9)), **full), "more than 8 groups"),
            "float_entry": (dict(groups=[1.7], **full), "must be a list of integers"),
            "bool_entry": (dict(groups=[True], **full), "must be a list of integers"),
            "not_a_list": (dict(groups=3, **full), "must be a list of integers"),
            "float32": (dict(groups=[0], fmt="float32", intensity_groups=None),
                        "output_beam_groups requires output_format: int8"),
            "intensity_frame_size": (dict(groups=[0, 1], intensity_groups=None),
                                     "output host buffer frame_size mismatch: expected "
                                     f"{2 * FF * 32} bytes (32 written beams)"),
            "qparams_frame_size": (dict(groups=[0, 1], qparam_groups=8),
                                   "quant params host buffer frame_size mismatch: expected "
                                   f"{2 * (FF // 16) * 2 * 8} bytes (2 beam groups)"),
        }
        report = {}
        for label, (kw, message) in cases.items():
            kwargs = self.common(**kw)
            kwargs["depth"] = 1
            r = run(self.root / f"t4_{label}", [(0, one, kwargs)], timeout_s=300)
            report[label] = {"rejected": len(r["shards"][0]["intensity"]) == 0,
                             "message_found": message in r["log"], "rc": r["rc"],
                             "message": message}
            if not report[label]["message_found"]:
                report[label]["log_tail"] = r["log"][-1500:]
        record("wp14_4_rejected", report)
        for label, r in report.items():
            self.assertTrue(r["rejected"] and r["message_found"], (label, r))


def main():
    global RESULTS, BINARY, BASELINE
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--kotekan-bin", type=Path)
    parser.add_argument("--baseline-bin", type=Path)
    args, rest = parser.parse_known_args()
    if args.results_dir:
        RESULTS = args.results_dir
        RESULTS.mkdir(parents=True, exist_ok=True)
    if args.kotekan_bin:
        BINARY = args.kotekan_bin.resolve()
    if args.baseline_bin:
        BASELINE = args.baseline_bin.resolve()
    unittest.main(argv=[sys.argv[0], *rest], verbosity=2)


if __name__ == "__main__":
    main()
