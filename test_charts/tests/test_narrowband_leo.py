"""Tier-1 tests for narrowband LEO satellite RFI.

Verifies:
  - Scheduled LEO events occupy a contiguous block of 1..LEO_MAX_CHANNELS channels.
  - Rendered LEO power is confined to those channels (out-of-band = noise only).
  - Event dicts without channels keep the legacy full-band behaviour.
  - The analytic lightcurve scales LEO power by the occupied channel fraction.
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

from sim.generator import (
    LEO_MAX_CHANNELS,
    compute_analytic_lightcurve,
    render_and_write_frame,
    schedule_random_events,
)
from sim.writer import read_raw_bin_frame

NUM_ANT, NUM_FREQ, SAMPLES = 16, 12, 1536
LEO_AMP = 2.0


def _job(out_path: Path, channels) -> dict:
    g = np.arange(4) * 0.6
    xx, yy = np.meshgrid(g, g)
    leo = {
        "event_type": "rfi_leo",
        "nominal_amp": LEO_AMP,
        "l0": 0.05,
        "m0": -0.05,
        "drift_dl": 0.02,
        "drift_dm": 0.01,
        "t_start_s": 0.0,
    }
    if channels is not None:
        leo["channels"] = channels
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
        "freqs_hz": 340e6 + np.arange(NUM_FREQ) * 0.3e6,
        "pos_x": (xx - xx.mean()).ravel(),
        "pos_y": (yy - yy.mean()).ravel(),
        "sigma_ant": np.ones(NUM_ANT),
        "bandpass": np.ones(NUM_FREQ),
        "sun_info": None,
        "active_events": [leo],
        "seed": 5,
        "source_model": "gaussian",
    }


class TestNarrowbandLEO(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="test_leo_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _channel_power(self, channels) -> np.ndarray:
        path = self.tmp / "leo.bin"
        render_and_write_frame(_job(path, channels))
        x = read_raw_bin_frame(path, shape=(SAMPLES, NUM_FREQ, NUM_ANT))
        return np.mean(np.abs(x) ** 2, axis=(0, 2))

    def test_01_schedule_contiguous_block(self):
        for seed in range(20):
            events = schedule_random_events(
                duration_s=30.0, num_events=3, allowed_types=["rfi_leo"],
                seed=seed, num_freq=NUM_FREQ,
            )
            for ev in events:
                ch = ev.channels
                self.assertTrue(1 <= len(ch) <= LEO_MAX_CHANNELS, ch)
                self.assertEqual(ch, list(range(ch[0], ch[0] + len(ch))))
                self.assertTrue(0 <= ch[0] and ch[-1] < NUM_FREQ)

    def test_02_power_confined_to_channels(self):
        p = self._channel_power([4, 5])
        noise = np.median(p)
        self.assertGreater(p[4], noise + 0.8 * LEO_AMP**2)
        self.assertGreater(p[5], noise + 0.8 * LEO_AMP**2)
        out = np.delete(p, [4, 5])
        np.testing.assert_allclose(out, noise, rtol=0.1)

    def test_03_legacy_event_without_channels_is_full_band(self):
        p = self._channel_power(None)
        self.assertTrue(np.all(p > 2.0 + 0.8 * LEO_AMP**2), p)

    def test_04_lightcurve_scales_with_channel_fraction(self):
        events = schedule_random_events(
            duration_s=10.0, num_events=1, allowed_types=["rfi_leo"], seed=3,
            num_freq=NUM_FREQ,
        )
        ev = events[0]
        freqs = 340e6 + np.arange(NUM_FREQ) * 0.3e6
        t, p = compute_analytic_lightcurve(10.0, 0.0, 0.0, [ev], freqs)
        on = (t >= ev.t_start_s) & (t <= ev.t_start_s + ev.duration_s)
        expected = ev.nominal_amp**2 * len(ev.channels) / NUM_FREQ
        np.testing.assert_allclose(p[on], expected)


if __name__ == "__main__":
    unittest.main()
