"""Tier-1 tests for the source signal model of the baseband generator.

Verifies:
  - 'gaussian' sources are mutually incoherent: with a sky source and an
    interferer in the same channel the covariance has two eigenvalues above
    the noise floor.
  - 'tone' (legacy) sources are mutually coherent and collapse into a single
    rank-1 component, which makes adaptive beamformers (MVDR) cancel the
    desired signal.
  - Both models inject the same mean power per source.
  - The noise realization is independent of the source model.

Theory:
  R = sum_k p_k a_k a_k^H + sigma_n^2 I only holds when E[s_k s_l^*] = 0 for
  k != l. Deterministic phasors with a fixed relative phase add cross terms
  and R becomes rank-1 in the source subspace.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np

_test_charts_dir = Path(__file__).resolve().parent.parent
if str(_test_charts_dir) not in sys.path:
    sys.path.insert(0, str(_test_charts_dir))

from sim.generator import SOURCE_MODELS, render_and_write_frame
from sim.writer import read_raw_bin_frame

N_SIDE, SPACING_M = 4, 0.6
NUM_ANT, NUM_FREQ, SAMPLES = N_SIDE * N_SIDE, 3, 1536
SUN_AMP, RFI_AMP, RFI_CH = 1.0, 3.0, 1


def _job(out_path: Path, source_model: str, with_sun: bool = True) -> dict:
    g = np.arange(N_SIDE) * SPACING_M
    xx, yy = np.meshgrid(g, g)
    return {
        "out_idx": 0,
        "frame_idx": 0,
        "out_file_path": str(out_path),
        "num_antennas": NUM_ANT,
        "num_freq": NUM_FREQ,
        "samples_per_frame": SAMPLES,
        "dt_s": 2.56e-6,
        "t_start_s": 0.0,
        "unix_start_s": 0.0,
        "freqs_hz": 340e6 + np.arange(NUM_FREQ) * 0.390625e6,
        "pos_x": (xx - xx.mean()).ravel(),
        "pos_y": (yy - yy.mean()).ravel(),
        "sigma_ant": np.ones(NUM_ANT),
        "bandpass": np.ones(NUM_FREQ),
        "sun_info": {"amp": SUN_AMP, "l": 0.47, "m": 0.48} if with_sun else None,
        "active_events": [
            {
                "event_type": "rfi_narrow",
                "nominal_amp": RFI_AMP,
                "l0": -0.83,
                "m0": -0.48,
                "channels": [RFI_CH],
                "t_start_s": 0.0,
            }
        ],
        "seed": 11,
        "source_model": source_model,
    }


class TestSourceModel(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="test_source_model_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _frame(self, model: str, with_sun: bool = True) -> np.ndarray:
        path = self.tmp / f"{model}_{int(with_sun)}.bin"
        render_and_write_frame(_job(path, model, with_sun))
        return read_raw_bin_frame(path, shape=(SAMPLES, NUM_FREQ, NUM_ANT))

    @staticmethod
    def _eig(v: np.ndarray) -> np.ndarray:
        R = v.T @ v.conj() / v.shape[0]
        return np.linalg.eigvalsh(R)[::-1]

    def test_01_gaussian_sources_are_incoherent(self):
        ev = self._eig(self._frame("gaussian")[:, RFI_CH, :])
        floor = np.median(ev)
        self.assertGreater(ev[1], 3.0 * floor, f"sky source lost: eigenvalues {ev[:3]}")
        self.assertLess(ev[2], 1.5 * floor)

    def test_02_tone_sources_collapse_to_rank_one(self):
        ev = self._eig(self._frame("tone")[:, RFI_CH, :])
        floor = np.median(ev)
        self.assertLess(ev[1], 1.5 * floor, f"expected rank-1 collapse: {ev[:3]}")

    def test_03_same_mean_power(self):
        p = {m: np.mean(np.abs(self._frame(m)[:, 0, :]) ** 2) for m in SOURCE_MODELS}
        self.assertAlmostEqual(p["gaussian"] / p["tone"], 1.0, delta=0.05)

    def test_04_noise_independent_of_source_model(self):
        # Channel 0 has no source when the Sun is off: identical bytes expected.
        a = self._frame("gaussian", with_sun=False)[:, 0, :]
        b = self._frame("tone", with_sun=False)[:, 0, :]
        np.testing.assert_array_equal(a, b)

    def test_05_invalid_model_rejected(self):
        with self.assertRaises(ValueError):
            render_and_write_frame(_job(self.tmp / "bad.bin", "phasor"))


if __name__ == "__main__":
    unittest.main()
