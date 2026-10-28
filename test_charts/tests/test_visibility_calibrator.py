"""End-to-end tests of ``cudaVisibilityCalibrator`` on simulated CHARTS baseband.

Each test replays the same simulated baseband window through the real Kotekan
binary (``rawFileRead -> cudaProcess -> rawFileWrite``) and compares the
``int32`` correlation frames with and without the calibrator:

  - without a table the command is a bit-exact pass-through
  - with a table the GPU equals ``round(vis * table)`` to within one count
  - ``freq_offset`` selects the matching slice of a full-band table
  - results saturate at the int32 limits instead of wrapping
  - a table of the wrong shape stops Kotekan at start-up
  - (needs charts-utils) tables built from gains reproduce ``apply_gains``

Requires the compiled Kotekan binary and an NVIDIA GPU; otherwise the suite
is skipped. See ``sim/corr_cal.py`` for ``CHARTS_TEST_LD_LIBRARY_PATH``, and set
``CHARTS_UTILS_PATH`` if charts-utils is not importable or at ``~/charts-utils``.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

_test_charts_dir = Path(__file__).resolve().parent.parent
_repo_root = _test_charts_dir.parent
if str(_test_charts_dir) not in sys.path:
    sys.path.insert(0, str(_test_charts_dir))

from sim.corr_cal import (
    create_replay_yaml,
    load_corr_frame,
    num_blocks_for,
    run_kotekan_replay,
)
from sim.generator import generate_simulation_window
from sim.presets import find_kotekan_binary, get_preset_config

NUM_ANT = 64
NUM_FREQ = 336
FULL_BAND_FREQ = 2 * NUM_FREQ
NUM_BLOCKS = num_blocks_for(NUM_ANT)
INT32_MAX_FLOAT32 = 2147483520.0  # largest float32 below 2**31, the kernel's upper clamp
INT32_MIN = -2147483648.0


def _gpu_available() -> bool:
    try:
        return subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _import_charts_utils():
    """Return the charts-utils modules used as an independent oracle, or None."""
    for path in (os.environ.get("CHARTS_UTILS_PATH"), str(Path.home() / "charts-utils")):
        if path and path not in sys.path and Path(path).is_dir():
            sys.path.insert(0, path)
    try:
        from charts_utils.cal.algorithms import apply_gains
        from charts_utils.cal.kotekan_calibration import gains_to_visibility_calibration
        from charts_utils.core.io import gpu_corr_to_visibilities
    except ImportError:
        return None
    return apply_gains, gains_to_visibility_calibration, gpu_corr_to_visibilities


def _random_table(rng: np.random.Generator, num_freq: int) -> np.ndarray:
    amp = rng.uniform(0.5, 2.0, (num_freq, NUM_BLOCKS, 2, 2))
    phase = rng.uniform(-np.pi, np.pi, (num_freq, NUM_BLOCKS, 2, 2))
    return (amp * np.exp(1j * phase)).astype(np.complex64)


@unittest.skipIf(find_kotekan_binary() is None, "kotekan binary is not built")
@unittest.skipUnless(_gpu_available(), "no NVIDIA GPU available")
class TestVisibilityCalibratorEndToEnd(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.work = Path(tempfile.mkdtemp(prefix="test_charts_viscal_"))
        cfg = get_preset_config(preset="quick", profile="day", antennas=NUM_ANT, num_freq=NUM_FREQ)
        cfg.duration_s = 0.3
        cfg.window_name = "viscal"
        cfg.scratch_dir = cls.work
        with _quiet():
            result = generate_simulation_window(cfg)
        cls.baseband_dir = Path(result["target_dir"])
        cls.num_frames = int(result["num_written"])

        rng = np.random.default_rng(2026)
        cls.table_full = _random_table(rng, FULL_BAND_FREQ)
        cls.table_path = cls.work / "table_full.npy"
        np.save(cls.table_path, cls.table_full)

        cls.runs = {}
        cls.run_replay("plain", None)
        cls.run_replay("passthrough", {})
        cls.run_replay("offset0", {"calibration_file": cls.table_path, "freq_offset": 0})
        cls.run_replay("offset336", {"calibration_file": cls.table_path, "freq_offset": NUM_FREQ})

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.work, ignore_errors=True)

    @classmethod
    def run_replay(cls, name, calibrator):
        out_dir = cls.work / name
        out_dir.mkdir()
        yaml_path = create_replay_yaml(
            cls.work / f"{name}.yaml", cls.baseband_dir, "viscal", out_dir,
            num_elements=NUM_ANT, num_local_freq=NUM_FREQ, calibrator=calibrator,
            num_frames=cls.num_frames,
        )
        log_path = cls.work / f"{name}.log"
        rc = run_kotekan_replay(yaml_path, log_path, _repo_root)
        cls.runs[name] = (rc, out_dir, log_path)
        return rc

    def frames(self, name):
        rc, out_dir, log_path = self.runs[name]
        tail = log_path.read_text()[-2000:]
        hint = ("\nHint: the installed NVRTC is newer than the GPU driver; point "
                "CHARTS_TEST_LD_LIBRARY_PATH at a matching libnvrtc." if "unsupported toolchain" in tail else "")
        self.assertEqual(rc, 0, f"kotekan failed for '{name}'; see {log_path}:\n{tail}{hint}")
        paths = sorted(out_dir.glob("corr_*.bin"))
        self.assertEqual(len(paths), self.num_frames)
        return paths

    def load(self, name):
        return [load_corr_frame(p, NUM_FREQ, NUM_BLOCKS) for p in self.frames(name)]

    def test_baseband_produces_nonzero_visibilities(self):
        self.assertTrue(any(np.any(f) for f in self.load("plain")))

    def test_passthrough_is_bit_exact(self):
        for plain, passthrough in zip(self.frames("plain"), self.frames("passthrough")):
            self.assertEqual(plain.read_bytes(), passthrough.read_bytes(), plain.name)

    def test_table_matches_numpy_product(self):
        for plain, calibrated in zip(self.load("plain"), self.load("offset0")):
            expected = plain * self.table_full[:NUM_FREQ].astype(np.complex128)
            self.assertLess(np.abs(calibrated - expected).max(), 1.0)

    def test_freq_offset_selects_table_slice(self):
        sl = slice(NUM_FREQ, FULL_BAND_FREQ)
        for plain, calibrated in zip(self.load("plain"), self.load("offset336")):
            right = plain * self.table_full[sl].astype(np.complex128)
            wrong = plain * self.table_full[:NUM_FREQ].astype(np.complex128)
            self.assertLess(np.abs(calibrated - right).max(), 1.0)
            # the test would pass for any offset if the two slices were alike
            self.assertGreater(np.abs(calibrated - wrong).max(), 100.0)

    def test_saturates_instead_of_wrapping(self):
        table = np.ones((NUM_FREQ, NUM_BLOCKS, 2, 2), dtype=np.complex64)
        table[0] = 1e9
        table[1] = -1e9
        path = self.work / "table_sat.npy"
        np.save(path, table)
        self.assertEqual(self.run_replay("saturate", {"calibration_file": path}), 0)

        plain = self.load("plain")
        for before, after in zip(plain, self.load("saturate")):
            real = after.real[:2]
            self.assertTrue(np.all(real <= INT32_MAX_FLOAT32) and np.all(real >= INT32_MIN))
            # |v| * 1e9 only exceeds 2**31 from three counts up
            pos = before.real[0] >= 3
            neg = before.real[0] <= -3
            self.assertTrue(pos.any() and neg.any())
            self.assertTrue(np.all(after.real[0][pos] == INT32_MAX_FLOAT32))
            self.assertTrue(np.all(after.real[0][neg] == INT32_MIN))
            pos1 = before.real[1] >= 3  # channel 1 has factor -1e9
            self.assertTrue(pos1.any())
            self.assertTrue(np.all(after.real[1][pos1] == INT32_MIN))
            # untouched channels stay equal to the uncalibrated output
            np.testing.assert_array_equal(after[2:], before[2:])

    def test_wrong_table_shape_stops_kotekan(self):
        path = self.work / "table_bad.npy"
        np.save(path, np.ones((FULL_BAND_FREQ, NUM_BLOCKS - 1, 2, 2), dtype=np.complex64))
        rc = self.run_replay("bad_shape", {"calibration_file": path})
        self.assertNotEqual(rc, 0)
        self.assertIn("must have shape", self.runs["bad_shape"][2].read_text())

    def test_offset_beyond_table_stops_kotekan(self):
        rc = self.run_replay("bad_offset", {"calibration_file": self.table_path,
                                            "freq_offset": FULL_BAND_FREQ - NUM_FREQ + 1})
        self.assertNotEqual(rc, 0)
        self.assertIn("exceeds", self.runs["bad_offset"][2].read_text())

    def test_gains_table_reproduces_charts_utils_apply_gains(self):
        modules = _import_charts_utils()
        if modules is None:
            self.skipTest("charts-utils is not importable (set CHARTS_UTILS_PATH)")
        apply_gains, gains_to_table, corr_to_vis = modules

        rng = np.random.default_rng(11)
        gains = (rng.uniform(0.7, 1.5, (FULL_BAND_FREQ, NUM_ANT))
                 * np.exp(1j * rng.uniform(-np.pi, np.pi, (FULL_BAND_FREQ, NUM_ANT))))
        path = self.work / "table_gains.npy"
        np.save(path, gains_to_table(gains))
        self.assertEqual(self.run_replay("gains", {"calibration_file": path, "freq_offset": 0}), 0)

        for plain, calibrated in zip(self.load("plain"), self.load("gains")):
            reference = apply_gains(corr_to_vis(plain)[..., None], gains[:NUM_FREQ])[..., 0]
            result = corr_to_vis(calibrated)
            self.assertLess(np.abs(result - reference).max() / np.abs(reference).max(), 1e-4)


class _quiet:
    """Silence the simulator's progress output while a window is generated."""

    def __enter__(self):
        self._devnull = open(os.devnull, "w")
        self._old = sys.stdout
        sys.stdout = self._devnull

    def __exit__(self, *exc):
        sys.stdout = self._old
        self._devnull.close()


if __name__ == "__main__":
    unittest.main()
