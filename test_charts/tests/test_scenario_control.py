"""Tier-1 tests for explicit scenario control of the baseband generator.

Verifies:
  - SimulationConfig.seed overrides the legacy seed int(1000 * utc_hour + 42);
    the default (None) keeps it, so existing windows are reproduced unchanged.
  - The same seed reproduces a window byte for byte; a different seed changes it.
  - SimulationConfig.events_json replaces the random event schedule: feeding a
    window's own <window>_events.json back reproduces the window exactly, and a
    hand-written file injects exactly the listed events.
  - Malformed event files are rejected instead of silently changing the scenario.
  - The CLI options --seed, --events-json and --source-model reach the config.
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

_test_charts_dir = Path(__file__).resolve().parent.parent
if str(_test_charts_dir) not in sys.path:
    sys.path.insert(0, str(_test_charts_dir))

from sim.generator import (  # noqa: E402
    generate_simulation_window,
    load_events_json,
    resolve_seed,
)
from sim.presets import get_preset_config  # noqa: E402
from sim.writer import read_raw_bin_frame  # noqa: E402

NUM_ANT, NUM_FREQ = 64, 336


def _event(**kw) -> dict:
    ev = {
        "event_id": "rfi_test",
        "event_type": "rfi_narrow",
        "t_start_s": 0.0,
        "duration_s": 1.0,
        "nominal_amp": 7.0,
        "l0": 0.5,
        "m0": 0.5,
        "channels": [3],
        "is_persistent": True,
    }
    ev.update(kw)
    return ev


class TestScenarioControl(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="test_scenario_control_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _cfg(self, name: str, **overrides):
        cfg = get_preset_config("quick", "day")
        cfg.duration_s = 0.02
        cfg.event_dense_s = 0.005
        cfg.scratch_dir = self.tmp / name
        cfg.window_name = "w"
        cfg.workers = 1
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg

    def _generate(self, name: str, **overrides) -> dict:
        res = generate_simulation_window(self._cfg(name, **overrides))
        frames = sorted(Path(res["target_dir"]).glob("w_*.bin"))
        res["frames"] = [p.read_bytes() for p in frames]
        res["events_list"] = json.loads(Path(res["events_json_path"]).read_text())
        return res

    def test_01_default_seed_is_legacy(self):
        cfg = self._cfg("a")
        self.assertIsNone(cfg.seed)
        self.assertEqual(resolve_seed(cfg), int(1000 * cfg.utc_hour + 42))
        cfg.seed = 7
        self.assertEqual(resolve_seed(cfg), 7)

    def test_02_seed_reproduces_and_changes_window(self):
        a = self._generate("a", seed=1)
        b = self._generate("b", seed=1)
        c = self._generate("c", seed=2)
        self.assertEqual(a["frames"], b["frames"])
        self.assertNotEqual(a["frames"][0], c["frames"][0])

    def test_03_explicit_legacy_seed_matches_default(self):
        legacy = int(1000 * 15.0 + 42)
        a = self._generate("a")
        b = self._generate("b", seed=legacy)
        self.assertEqual(a["frames"], b["frames"])

    def test_04_events_json_roundtrip_is_exact(self):
        a = self._generate("a", seed=5, num_events=2)
        b = self._generate("b", seed=5, num_events=0, events_json=a["events_json_path"])
        self.assertEqual(a["events_list"], b["events_list"])
        self.assertEqual(a["frames"], b["frames"])

    def test_05_hand_written_events(self):
        path = self.tmp / "events.json"
        path.write_text(json.dumps([_event()]))
        res = self._generate("a", seed=3, events_json=path)
        self.assertEqual(len(res["events_list"]), 1)
        self.assertEqual(res["events_list"][0]["l0"], 0.5)
        self.assertEqual(res["events_list"][0]["channels"], [3])
        frame = read_raw_bin_frame(
            Path(res["target_dir"]) / "w_0000000.bin", shape=(1536, NUM_FREQ, NUM_ANT)
        )
        power = np.mean(np.abs(frame) ** 2, axis=(0, 2))
        others = np.delete(power, [3])
        self.assertGreater(power[3], 3.0 * np.median(others))
        # No other RFI: the default persistent channels are not injected.
        for ch in (94, 133, 147):
            self.assertLess(power[ch], 1.5 * np.median(others))

    def test_06_malformed_events_rejected(self):
        cases = {
            "unknown_key": [_event(amplitude=3.0)],
            "unknown_type": [_event(event_type="rfi_wideband")],
            "missing_key": [{k: v for k, v in _event().items() if k != "duration_s"}],
            "duplicated_id": [_event(), _event(channels=[5])],
            "not_a_list": _event(),
        }
        for name, data in cases.items():
            path = self.tmp / f"{name}.json"
            path.write_text(json.dumps(data))
            with self.subTest(name), self.assertRaises(ValueError):
                load_events_json(path)

    def test_07_cli_options_reach_config(self):
        import charts_sim

        parser = argparse.ArgumentParser()
        charts_sim.add_scenario_arguments(parser)
        args = parser.parse_args(
            ["--seed", "11", "--events-json", "ev.json", "--source-model", "tone"]
        )
        cfg = charts_sim.apply_scenario_args(self._cfg("a"), args)
        self.assertEqual(cfg.seed, 11)
        self.assertEqual(cfg.events_json, Path("ev.json"))
        self.assertEqual(cfg.source_model, "tone")
        defaults = charts_sim.apply_scenario_args(self._cfg("b"), parser.parse_args([]))
        self.assertIsNone(defaults.seed)
        self.assertIsNone(defaults.events_json)
        self.assertEqual(defaults.source_model, "gaussian")
        with self.assertRaises(SystemExit):
            parser.parse_args(["--source-model", "phasor"])


if __name__ == "__main__":
    unittest.main()
