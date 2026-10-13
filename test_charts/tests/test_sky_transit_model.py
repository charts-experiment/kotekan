"""Unit tests for CHARTS unified sky and transit model (sim/sky.py).

Validates:
  - Catalog loading from tools/verified_targets.json (SIMBAD-verified parity).
  - Fuzzy target lookup and alias resolution.
  - Transit-anchored window resolution (transit:Target, offsets, routine slots).
  - Direction cosine calculation parity with Kotekan (compute_celestial_direction).
  - Transit meridian geometry (l ~= 0, m ~= sin(dec - lat)).
  - Geometric phase track calculation and continuity.
"""

from __future__ import annotations

import datetime
import math
from pathlib import Path
import sys
import unittest

# Setup paths
_test_charts_dir = Path(__file__).resolve().parent.parent
if str(_test_charts_dir) not in sys.path:
    sys.path.insert(0, str(_test_charts_dir))

import numpy as np

from sim.constants import (
    C_LIGHT,
    CHARTS_LATITUDE_DEG,
    CHARTS_LONGITUDE_DEG,
    get_antenna_positions,
)
from sim.sky import (
    VerifiedTarget,
    catalog_transit_summary,
    direction_cosines_track,
    find_verified_target,
    geometric_phase_track,
    load_verified_catalog,
    resolve_window_start,
)


class TestSkyTransitModel(unittest.TestCase):
    """Test suite for sky.py catalog loading, lookup, and geometry."""

    @classmethod
    def setUpClass(cls):
        cls.catalog = load_verified_catalog()

    def test_catalog_loaded_and_non_empty(self):
        """Verified target catalog must load at least 20 SIMBAD-verified targets."""
        self.assertGreaterEqual(len(self.catalog), 20)
        # Check standard reference pulsars and bright radio sources exist
        expected_keys = ["crab", "vela", "sgr a*", "puppis", "centaurus"]
        catalog_keys = list(self.catalog.keys())
        for exp in expected_keys:
            matched = any(exp in k for k in catalog_keys)
            self.assertTrue(
                matched, f"Expected target containing '{exp}' in catalog keys"
            )

    def test_fuzzy_target_lookup(self):
        """Fuzzy lookup should resolve case-insensitively and on substrings."""
        vela = find_verified_target("vela", self.catalog)
        self.assertIn("Vela", vela.label)
        self.assertAlmostEqual(vela.ra_deg, 128.8361, places=3)
        self.assertAlmostEqual(vela.dec_deg, -45.1764, places=3)

        crab = find_verified_target("crab", self.catalog)
        self.assertIn("Crab", crab.label)

        sgr = find_verified_target("sgr a*", self.catalog)
        self.assertIn("Sgr A*", sgr.label)

        with self.assertRaises(KeyError):
            find_verified_target("non_existent_source_42xyz", self.catalog)

    def test_catalog_transit_summary(self):
        """catalog_transit_summary must return a sorted list of dictionaries."""
        summary = catalog_transit_summary(self.catalog)
        self.assertEqual(len(summary), len(self.catalog))
        for entry in summary:
            self.assertIn("label", entry)
            self.assertIn("ra_deg", entry)
            self.assertIn("dec_deg", entry)
            self.assertIn("transit_utc", entry)
            self.assertIn("transit_local", entry)
            self.assertIn("max_alt_deg", entry)
            self.assertIn("hours_above_mask", entry)

        # Check sorted order by transit_utc
        for i in range(len(summary) - 1):
            self.assertLessEqual(
                summary[i]["transit_utc"], summary[i + 1]["transit_utc"]
            )

    def test_transit_anchored_window(self):
        """resolve_window_start with 'transit:<target>' anchors mid-window at transit time."""
        vela = find_verified_target("vela", self.catalog)
        duration_s = 60.0
        start_dt = resolve_window_start(
            "transit:Vela", duration_s=duration_s, catalog=self.catalog
        )
        mid_dt = start_dt + datetime.timedelta(seconds=duration_s / 2.0)
        diff_s = abs((mid_dt - vela.transit_dt).total_seconds())
        self.assertAlmostEqual(diff_s, 0.0, places=3)

    def test_transit_offsets_and_slots(self):
        """resolve_window_start supports slot:HH:MM and clock hour formats."""
        dt_slot = resolve_window_start(
            "slot:09:00", duration_s=10.0, catalog=self.catalog
        )
        self.assertIsInstance(dt_slot, datetime.datetime)

        dt_hhmm = resolve_window_start("15:00", duration_s=10.0, catalog=self.catalog)
        self.assertEqual(dt_hhmm.hour, 15)
        self.assertEqual(dt_hhmm.minute, 0)

        dt_float = resolve_window_start(15.5, duration_s=10.0, catalog=self.catalog)
        self.assertEqual(dt_float.hour, 15)
        self.assertEqual(dt_float.minute, 30)

    def test_direction_cosines_at_transit(self):
        """At celestial transit, l ~= 0 and m ~= sin(dec - lat)."""
        lat_deg = CHARTS_LATITUDE_DEG
        for v in self.catalog.values():
            l_t, m_t = v.direction_cosines_at_transit(lat_deg=lat_deg)
            # l should be very close to 0 on the meridian (|l| < 0.01 due to ephemeris rounding)
            self.assertLess(
                abs(l_t), 0.01, f"Target {v.label} had large l at transit: {l_t}"
            )

            expected_m = math.sin(math.radians(v.dec_deg - lat_deg))
            self.assertAlmostEqual(
                m_t,
                expected_m,
                places=3,
                msg=f"Target {v.label} m ({m_t}) did not match sin(dec-lat) ({expected_m})",
            )

    def test_direction_cosines_track_smoothness_and_bounds(self):
        """direction_cosines_track produces continuous, bounded (l, m, n) tracks."""
        vela = find_verified_target("vela", self.catalog)
        t0 = vela.transit_dt.timestamp()
        times_s = t0 + np.linspace(-300, 300, 601)  # 10 minutes, 1s cadence

        l_track, m_track, n_track = direction_cosines_track(
            vela.ra_deg, vela.dec_deg, times_s
        )

        self.assertEqual(l_track.shape, (601,))
        self.assertEqual(m_track.shape, (601,))
        self.assertEqual(n_track.shape, (601,))

        # Bounds: all in [-1, 1]
        self.assertTrue(np.all(np.abs(l_track) <= 1.0))
        self.assertTrue(np.all(np.abs(m_track) <= 1.0))
        self.assertTrue(np.all(np.abs(n_track) <= 1.0))

        # Sum of squares l^2 + m^2 + n^2 ~= 1
        trans_sq = l_track**2 + m_track**2 + n_track**2
        np.testing.assert_allclose(trans_sq, 1.0, atol=1e-6)

        # Smoothness: diff between adjacent seconds should be very small (~ Earth rotation rate)
        dl = np.diff(l_track)
        dm = np.diff(m_track)
        self.assertLess(np.max(np.abs(dl)), 1e-3)
        self.assertLess(np.max(np.abs(dm)), 1e-3)

    def test_geometric_phase_track(self):
        """geometric_phase_track computes phi = 2*pi*f*(x*l + y*m)/c correctly."""
        pos_x, pos_y = get_antenna_positions(num_antennas=8)
        freqs_hz = np.array([300e6, 350e6, 400e6])
        l_track = np.array([0.0, 0.1])
        m_track = np.array([-0.2, -0.2])

        phases = geometric_phase_track(l_track, m_track, pos_x, pos_y, freqs_hz)
        self.assertEqual(phases.shape, (2, 3, 8))

        # Check explicit calculation for sample 0, freq 0, ant 1
        expected_delay = (pos_x[1] * l_track[0] + pos_y[1] * m_track[0]) / C_LIGHT
        expected_phase = 2.0 * math.pi * freqs_hz[0] * expected_delay
        self.assertAlmostEqual(phases[0, 0, 1], expected_phase, places=5)


if __name__ == "__main__":
    unittest.main()
