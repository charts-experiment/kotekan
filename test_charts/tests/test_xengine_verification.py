"""Unit test suite for CHARTS X-Engine ground-truth verification (sim/verify.py).

Validates:
  - Pure-NumPy reference_visibilities (Hermitian symmetry, positive autocorrelations).
  - Pure-NumPy reference_beamform (coherent power gain, active antenna masking).
  - verify_visibility_phasing (baseline fringe phase matches geometric prediction).
  - verify_pointing (pointing error < 1e-4 rad, AGENTS.md §3 equivalence bound).
  - verify_quantization_snr_loss (< 0.5 dB bound).
  - verify_transit_lightcurve (formed beam peak aligns with transit time).
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

# Setup paths
_test_charts_dir = Path(__file__).resolve().parent.parent
if str(_test_charts_dir) not in sys.path:
    sys.path.insert(0, str(_test_charts_dir))

import numpy as np

from sim.constants import C_LIGHT, DIGITIZER_NOMINAL_SIGMA_LSB, get_antenna_positions
from sim.fengine import quantize_int4x2
from sim.sky import direction_cosines_track, find_verified_target
from sim.verify import (
    reference_beamform,
    reference_visibilities,
    verify_pointing,
    verify_quantization_snr_loss,
    verify_transit_lightcurve,
    verify_visibility_phasing,
)


class TestXEngineVerification(unittest.TestCase):
    """Test suite for sim/verify.py reference implementations and equivalence checks."""

    def test_reference_visibilities_hermitian_and_positive(self):
        """reference_visibilities must produce Hermitian matrices with real positive autocorrelations."""
        rng = np.random.default_rng(42)
        n_samples = 64
        n_freq = 4
        n_ant = 8

        # Random complex voltages
        v = (
            rng.standard_normal((n_samples, n_freq, n_ant))
            + 1j * rng.standard_normal((n_samples, n_freq, n_ant))
        ).astype(np.complex64)

        vis = reference_visibilities(v)
        self.assertEqual(vis.shape, (n_freq, n_ant, n_ant))

        # Check Hermitian symmetry: V_ji == conj(V_ij)
        vis_h = np.conj(np.swapaxes(vis, 1, 2))
        np.testing.assert_allclose(vis, vis_h, atol=1e-6)

        # Check diagonal (autocorrelations): real and >= 0
        for f in range(n_freq):
            diag = np.diagonal(vis[f])
            np.testing.assert_allclose(np.imag(diag), 0.0, atol=1e-6)
            self.assertTrue(np.all(np.real(diag) > 0.0))

    def test_reference_beamform_coherent_gain_and_masking(self):
        """reference_beamform must produce coherent power gain and respect antenna masking."""
        n_ant = 16
        pos_x, pos_y = get_antenna_positions(num_antennas=n_ant)
        freqs_hz = np.array([300e6, 350e6])
        t_now = np.array([1791344316.0])

        target = find_verified_target("vela")
        l, m, _ = direction_cosines_track(target.ra_deg, target.dec_deg, t_now)
        delays = (pos_x * l[0] + pos_y * m[0]) / C_LIGHT
        phases = 2.0 * math.pi * (freqs_hz[:, None] * delays[None, :])

        # Source injected as A * exp(-j * phi) with amplitude A = 1.0
        v = np.exp(-1j * phases)[None, :, :]  # (1, 2, n_ant)

        # Full array beamforming
        formed = reference_beamform(
            v, pos_x, pos_y, freqs_hz, target.ra_deg, target.dec_deg, t_now
        )
        self.assertEqual(formed.shape, (1, 2))

        # When steered in-phase with conjugate weights w = exp(+j*phi)/sqrt(N),
        # formed amplitude is sum(A * exp(-j*phi) * exp(+j*phi) / sqrt(N)) = N / sqrt(N) = sqrt(N).
        # Power |b|^2 = N.
        power = np.abs(formed) ** 2
        np.testing.assert_allclose(power, float(n_ant), rtol=1e-4)

        # Active antenna masking: only first 4 antennas active
        active_sub = [0, 1, 2, 3]
        formed_sub = reference_beamform(
            v,
            pos_x,
            pos_y,
            freqs_hz,
            target.ra_deg,
            target.dec_deg,
            t_now,
            active_antennas=active_sub,
        )
        power_sub = np.abs(formed_sub) ** 2
        np.testing.assert_allclose(power_sub, float(len(active_sub)), rtol=1e-4)

    def test_verify_visibility_phasing(self):
        """verify_visibility_phasing verifies baseline fringe phase vs. geometric delay."""
        pos_x, pos_y = get_antenna_positions(num_antennas=8)
        freqs_hz = np.array([300e6, 350e6])
        t_now = np.array([1791344316.0])

        target = find_verified_target("vela")
        l, m, _ = direction_cosines_track(target.ra_deg, target.dec_deg, t_now)
        delays = (pos_x * l[0] + pos_y * m[0]) / C_LIGHT
        phases = 2.0 * math.pi * (freqs_hz[:, None] * delays[None, :])
        v = np.exp(-1j * phases)[None, :, :]

        res = verify_visibility_phasing(
            v,
            pos_x,
            pos_y,
            freqs_hz,
            target.ra_deg,
            target.dec_deg,
            t_now,
            max_phase_err_rad=0.01,
        )
        self.assertTrue(res["passed"])
        self.assertLess(res["max_phase_error_rad"], 1e-4)
        self.assertEqual(res["num_baselines_checked"], 2 * (8 * 7 // 2))

    def test_verify_pointing_recovers_injected_source(self):
        """verify_pointing recovers source direction with angular error < 1e-4 rad (AGENTS.md §3)."""
        pos_x, pos_y = get_antenna_positions(num_antennas=64)
        freqs_hz = np.array([300e6, 350e6])
        t_now = np.array([1791344316.0])

        target = find_verified_target("vela")
        l, m, _ = direction_cosines_track(target.ra_deg, target.dec_deg, t_now)
        delays = (pos_x * l[0] + pos_y * m[0]) / C_LIGHT
        phases = 2.0 * math.pi * (freqs_hz[:, None] * delays[None, :])
        v = np.exp(-1j * phases)[None, :, :]

        res = verify_pointing(
            v,
            pos_x,
            pos_y,
            freqs_hz,
            target.ra_deg,
            target.dec_deg,
            t_now,
            angular_search_radius_deg=0.5,
            num_steps=21,
            max_offset_rad=1e-4,
        )
        self.assertTrue(res["passed"])
        self.assertLessEqual(res["offset_rad"], 1e-4)
        self.assertAlmostEqual(res["peak_ra_deg"], target.ra_deg, places=3)
        self.assertAlmostEqual(res["peak_dec_deg"], target.dec_deg, places=3)

    def test_verify_quantization_snr_loss(self):
        """verify_quantization_snr_loss verifies loss stays under 0.5 dB for digitizer operating point."""
        rng = np.random.default_rng(77)
        n = 50_000
        sigma = DIGITIZER_NOMINAL_SIGMA_LSB
        x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * sigma
        _, q = quantize_int4x2(x, scale=1.0)

        res = verify_quantization_snr_loss(x, q, max_loss_db=0.5)
        self.assertTrue(res["passed"])
        self.assertLess(res["snr_loss_db"], 0.5)
        self.assertGreater(res["snr_loss_db"], 0.0)

    def test_verify_transit_lightcurve(self):
        """verify_transit_lightcurve checks that formed beam power peaks at expected transit."""
        t_axis = np.linspace(100.0, 200.0, 101)
        expected_transit = 150.0
        # Gaussian transit profile peaking at 150.0s
        power_curve = np.exp(-(((t_axis - expected_transit) / 10.0) ** 2))

        res_pass = verify_transit_lightcurve(
            t_axis,
            power_curve,
            expected_transit_time_s=expected_transit,
            tolerance_s=1.0,
        )
        self.assertTrue(res_pass["passed"])
        self.assertEqual(res_pass["peak_time_s"], expected_transit)
        self.assertEqual(res_pass["offset_s"], 0.0)

        # Off-transit test
        res_fail = verify_transit_lightcurve(
            t_axis, power_curve, expected_transit_time_s=170.0, tolerance_s=2.0
        )
        self.assertFalse(res_fail["passed"])
        self.assertGreater(res_fail["offset_s"], 2.0)


if __name__ == "__main__":
    unittest.main()
