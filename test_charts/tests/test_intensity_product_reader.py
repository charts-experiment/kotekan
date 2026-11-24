"""CPU tests of the intensity product reader for the beam selection (WP14, output_beam_groups).

The int8 product may hold only selected 16-beam groups. The v3 descriptor then carries a
`beams` block (groups, written) and intensity.shape[2] = int8.parameters.beam_tiles * 16 = 16G.
A descriptor without the block was written with all 128 beams in order.

    PYTHONPATH=. ~/miniconda3/envs/kotekan_test/bin/python -m unittest \
        test_charts.tests.test_intensity_product_reader -v

No GPU. The exact GPU selection (codes and parameters equal to slices of the full product)
is tested by test_beam_selection_gpu.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import yaml

from test_charts.sim import intensity_product as ip
from test_charts.sim.pipeline import create_beamformer_yaml

ROOT = Path(__file__).resolve().parents[2]
STORED = sorted((ROOT / "test_charts/results/wp6_live_20261124_msx00").glob(
    "*/intensity.shard*.product.json"))
B, GROUP = 128, 16


def base_descriptor() -> dict:
    """A v3 descriptor of the full product without a beams block (as written before
    2026-11-24), with only the fields the reader uses. Self-contained: the stored real
    descriptors live in test_charts/results, which is not versioned."""
    n_time, n_freq = 48, 10752
    return {
        "schema": "charts-intensity-product-v3", "shard": 0, "T_raw": 15360, "U": 32, "K": 10,
        "intensity": {"dtype": "int8", "shape": [n_time, n_freq, B]},
        "frequency": {"nu0_hz": 300e6, "dnu_hz": 300e3},
        "association": {"key": ["shard", "FPGA_SEQ_NUM", "T_raw"]},
        "history": {"fine_format": "fp16",
                    "upchan_gain": {"values": [1.0] * 32, "normalization": "none"}},
        "weights": {"device": {"precision": "fp16", "effective_sha256": "0" * 64}},
        "beamformer": {"kernel": "tensor_core"},
        "int8": {"chunk": [1, 16, 16],
                 "parameters": {"count": n_time * (n_freq // 16) * (B // 16),
                                "frequency_tiles": n_freq // 16, "beam_tiles": B // 16}},
    }


def with_selection(d: dict, groups) -> dict:
    d = copy.deepcopy(d)
    written = [g * GROUP + j for g in groups for j in range(GROUP)]
    d["intensity"]["shape"][2] = len(written)
    p = d["int8"]["parameters"]
    p["beam_tiles"] = len(groups)
    p["count"] = d["intensity"]["shape"][0] * p["frequency_tiles"] * len(groups)
    d["beams"] = {"computed": B, "group_size": GROUP, "groups": list(groups),
                  "written": written, "rule": "output beam j is weight-file beam written[j]"}
    return d


def load(d: dict) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "intensity.shard0.product.json"
        path.write_text(json.dumps(d))
        return ip.load_descriptor(path)


class StoredDescriptors(unittest.TestCase):
    def test_stored_v3_without_beams_block_loads_with_all_beams(self):
        if not STORED:
            self.skipTest("no stored v3 descriptor")
        for path in STORED:
            with self.subTest(path=path.relative_to(ROOT).as_posix()):
                d = ip.load_descriptor(path)
                self.assertNotIn("beams", d)
                np.testing.assert_array_equal(ip.beam_indices(d), np.arange(B))
                self.assertEqual(d["int8"]["parameters"]["beam_tiles"], B // GROUP)

    def test_synthetic_fixture_matches_stored_fields(self):
        """The synthetic base carries the same values as a real v3 descriptor where both
        define them (guards the fixture against drifting from the writer)."""
        if not STORED:
            self.skipTest("no stored v3 descriptor")
        real = json.loads(STORED[0].read_text())
        syn = base_descriptor()
        for path in (("schema",), ("U",), ("K",), ("T_raw",), ("intensity", "shape"),
                     ("int8", "chunk"), ("int8", "parameters", "frequency_tiles"),
                     ("int8", "parameters", "beam_tiles"), ("int8", "parameters", "count"),
                     ("history", "fine_format")):
            a, b = real, syn
            for key in path:
                a, b = a[key], b[key]
            self.assertEqual(a, b, path)

    def test_v3_without_beams_must_have_128_beams(self):
        d = base_descriptor()
        d["intensity"]["shape"][2] = 48
        with self.assertRaisesRegex(ip.ProductError, "expected 128"):
            load(d)

    def test_older_schemas_still_accepted(self):
        d = base_descriptor()
        for schema in ("charts-intensity-product-v1", "charts-intensity-product-v2"):
            with self.subTest(schema=schema):
                old = dict(d, schema=schema)
                np.testing.assert_array_equal(ip.beam_indices(load(old)), np.arange(B))


class BeamsBlock(unittest.TestCase):
    def setUp(self):
        self.d = base_descriptor()

    def test_round_trip_selection(self):
        for groups in ([0, 3, 7], [5], [0, 1, 2, 3], list(range(8))):
            with self.subTest(groups=groups):
                d = load(with_selection(self.d, groups))
                expected = np.concatenate([np.arange(g * GROUP, (g + 1) * GROUP)
                                           for g in groups])
                np.testing.assert_array_equal(ip.beam_indices(d), expected)
                self.assertEqual(d["intensity"]["shape"][2], 16 * len(groups))

    def test_beam_positions(self):
        d = load(with_selection(self.d, [0, 3, 7]))
        np.testing.assert_array_equal(ip.beam_positions(d, [0, 15, 48, 63, 112, 127]),
                                      [0, 15, 16, 31, 32, 47])
        with self.assertRaisesRegex(ip.ProductError, r"beams \[16, 100\] were not written"):
            ip.beam_positions(d, [0, 16, 100])
        np.testing.assert_array_equal(ip.beam_positions(self.d, [7, 100]), [7, 100])

    def test_inconsistent_blocks_rejected(self):
        good = with_selection(self.d, [0, 3, 7])

        def mutate(fn):
            d = copy.deepcopy(good)
            fn(d)
            return d

        cases = {
            "missing written": (mutate(lambda d: d["beams"].pop("written")), "lacks 'written'"),
            "missing groups": (mutate(lambda d: d["beams"].pop("groups")), "lacks 'groups'"),
            "empty groups": (mutate(lambda d: d["beams"].update(groups=[], written=[])),
                             "strictly increasing"),
            "duplicate": (mutate(lambda d: d["beams"].update(groups=[0, 0, 7])),
                          "strictly increasing"),
            "decreasing": (mutate(lambda d: d["beams"].update(groups=[3, 0, 7])),
                           "strictly increasing"),
            "out of range": (mutate(lambda d: d["beams"].update(groups=[0, 3, 8])),
                             "strictly increasing"),
            "negative": (mutate(lambda d: d["beams"].update(groups=[-1, 3, 7])),
                         "strictly increasing"),
            "boolean": (mutate(lambda d: d["beams"].update(groups=[False, 3, 7])),
                        "strictly increasing"),
            "group size": (mutate(lambda d: d["beams"].update(group_size=8)), "group_size"),
            "written vs groups": (mutate(lambda d: d["beams"]["written"].reverse()),
                                  "inconsistent with beams.groups"),
            "shape": (mutate(lambda d: d["intensity"]["shape"].__setitem__(2, B)),
                      r"shape\[2\] = 128 but 48 beams"),
            "beam tiles": (mutate(lambda d: d["int8"]["parameters"].update(beam_tiles=8)),
                           "beam_tiles = 8 but 3"),
            "not an object": (mutate(lambda d: d.update(beams=[0, 3, 7])), "must be an object"),
        }
        for label, (d, message) in cases.items():
            with self.subTest(label):
                with self.assertRaisesRegex(ip.ProductError, message):
                    load(d)


class DecodeSelection(unittest.TestCase):
    """decode_int8 on a selected product equals the slice of the full decode, given codes and
    parameters that are slices of the full ones (the GPU contract of output_beam_groups)."""

    def test_decode_of_selected_groups_equals_full_slice(self):
        full = copy.deepcopy(base_descriptor())
        n_time, n_freq = 2, 64                              # small: 4 frequency tiles
        full["intensity"]["shape"] = [n_time, n_freq, B]
        full["int8"]["parameters"]["frequency_tiles"] = n_freq // GROUP
        full["int8"]["parameters"]["count"] = n_time * (n_freq // GROUP) * (B // GROUP)
        rng = np.random.default_rng(1124)
        codes = rng.integers(-127, 128, size=(n_time, n_freq, B), dtype=np.int8)
        codes[1, 16:32, 48:64] = ip.INVALID_CODE            # one invalid chunk in group 3
        t_tiles, f_tiles, b_tiles = n_time, n_freq // GROUP, B // GROUP
        table = rng.normal(size=(t_tiles, f_tiles, b_tiles, 2)).astype(np.float32)
        key = (0, 15360)
        decoded = ip.decode_int8(full, (key, codes), (key, table.reshape(-1, 2)), shard=0)
        for groups in ([0, 3, 7], [5], [3]):
            with self.subTest(groups=groups):
                sel = with_selection(full, groups)
                sel["int8"]["parameters"]["count"] = t_tiles * f_tiles * len(groups)
                d = load(sel)
                cols = ip.beam_indices(d)
                q = np.ascontiguousarray(codes[:, :, cols])
                t = np.ascontiguousarray(table[:, :, groups, :]).reshape(-1, 2)
                got = ip.decode_int8(d, (key, q), (key, t), shard=0)
                np.testing.assert_array_equal(got, decoded[:, :, cols])
                self.assertEqual(int(np.isnan(got).sum()),
                                 (16 * 16) if 3 in groups else 0)


class YamlGenerator(unittest.TestCase):
    def test_output_beam_groups_sizes_intensity_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for groups, beams in ((None, 128), ([0, 2, 3], 48), ([7], 16)):
                with self.subTest(groups=groups):
                    path = create_beamformer_yaml(tmp / "bf.yaml", tmp / "in", "voltage",
                                                  tmp / "out", "intensity", 1,
                                                  output_beam_groups=groups)
                    cfg = yaml.safe_load(path.read_text())
                    self.assertEqual(cfg["intensity_buf"]["frame_size"], 48 * 10752 * beams)
                    command = next(c for c in cfg["gpu_beamformer"]["commands"]
                                   if c["name"] == "cudaChartsBeamformerCommand")
                    if groups is None:
                        self.assertNotIn("output_beam_groups", command)
                    else:
                        self.assertEqual(command["output_beam_groups"], groups)


if __name__ == "__main__":
    unittest.main()
