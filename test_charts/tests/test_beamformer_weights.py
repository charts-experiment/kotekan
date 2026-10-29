"""Offline combined-weights builder: element order, frequencies, gains and steering.

The sky-signal model is written from the charts-utils definitions (``_reference_delays``:
tau_a = -(s . p_a)/c; ``geometric_visibility_phase``: arg V_ab = -2*pi*nu*(tau_a - tau_b)
with V_ab = x'_a * conj(x'_b)) for correlator-convention voltages x' (real = high nibble).
The beamformer's voltage x (real = low nibble) is derived from the same packed bytes.
The builder's helpers are not oracles.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "charts" / "scripts" / "build_beamformer_weights.py"
_spec = importlib.util.spec_from_file_location("build_beamformer_weights", SCRIPT)
bw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bw)

C = 299792458.0


def charts_utils_delay(pos_xyz, n_x, n_y):
    """charts-utils _reference_delays: tau = -(s . p)/c, s the ENU unit source vector."""
    n_z = np.sqrt(1.0 - (n_x**2 + n_y**2))
    return -(n_x * pos_xyz[..., 0] + n_y * pos_xyz[..., 1] + n_z * pos_xyz[..., 2]) / C


def charts_utils_voltage_phase(f_hz, pos_xyz, n_x, n_y):
    """Arriving voltage phase: x_a ~ exp(-2*pi*i*nu*tau_a), so V_ab has -2*pi*nu*(tau_a-tau_b)."""
    return -2.0 * np.pi * f_hz * charts_utils_delay(pos_xyz, n_x, n_y)


def wrap(phase):
    return np.angle(np.exp(1j * phase))


def signed_nibble(v):
    v = np.asarray(v, dtype=np.int16) & 0xF
    return np.where(v >= 8, v - 16, v)


def correlator_decode(byte):
    """charts-utils unpack_4bit_complex / correlator view: real = high nibble."""
    return signed_nibble(byte >> 4) + 1j * signed_nibble(byte)


def beamformer_decode(byte):
    """cudaChartsUpchannelizer unpack_complex_int4: real = low nibble."""
    return signed_nibble(byte) + 1j * signed_nibble(byte >> 4)


def emulate_astron_corr(bytes_y, bytes_x):
    """cudaShuffleAstron nibble swap, then TCC 4-bit products with conj_perm.

    TCC A row = (lo, hi) of the Y sample; B columns = X sample (lo, hi) and
    conj_perm(X) = (lo=-hi, hi=lo). Output (real, imag) = (A.B0, A.B1).
    """
    swap = lambda b: ((b & 0x0F) << 4) | (b >> 4)
    sy, sx = swap(bytes_y), swap(bytes_x)
    ly, hy, lx, hx = (signed_nibble(sy), signed_nibble(sy >> 4),
                      signed_nibble(sx), signed_nibble(sx >> 4))
    return np.sum(ly * lx + hy * hx) + 1j * np.sum(ly * (-hx) + hy * lx)


class ComponentConvention(unittest.TestCase):
    def test_beamformer_voltage_is_i_conj_of_correlator_voltage(self):
        b = np.arange(256)
        np.testing.assert_array_equal(beamformer_decode(b), 1j * np.conj(correlator_decode(b)))

    def test_astron_correlator_matches_charts_utils_read(self):
        rng = np.random.default_rng(3)
        for _ in range(20):
            y, x = rng.integers(0, 256, (2, 64))
            # charts-utils: corr[baseline(Y >= X)] = x'_Y * conj(x'_X)
            expected = np.sum(correlator_decode(y) * np.conj(correlator_decode(x)))
            self.assertEqual(emulate_astron_corr(y, x), expected)


class ElementOrder(unittest.TestCase):
    def test_logical_to_raw_anchor_points(self):
        raw_of = bw.charts_logical_to_kotekan_raw_indices(64)
        for logical, raw in ((0, 63), (31, 32), (32, 31), (63, 0)):
            self.assertEqual(raw_of[logical], raw)

    def test_reorder_places_logical_antenna_at_raw_index(self):
        logical = np.arange(64) + 100  # value identifies the logical antenna
        raw = bw.charts_logical_to_kotekan_raw(logical)
        for r in range(64):
            self.assertEqual(raw[r], 100 + 63 - r)


class Frequencies(unittest.TestCase):
    def test_coarse_shard_edges(self):
        f0 = bw.shard_frequencies_hz("coarse", 0)
        f1 = bw.shard_frequencies_hz("coarse", 1)
        self.assertEqual(f0.size, 336)
        self.assertAlmostEqual(f0[0], 300.0e6)
        self.assertAlmostEqual(f0[-1], 400.5e6)
        self.assertAlmostEqual(f1[0], 400.8e6)
        self.assertAlmostEqual(f1[-1], 501.3e6)

    def test_fine_bins_centered_on_coarse_and_index_order(self):
        for shard in (0, 1):
            fine = bw.shard_frequencies_hz("fine", shard).reshape(336, 32)
            coarse = bw.shard_frequencies_hz("coarse", shard)
            np.testing.assert_allclose(fine.mean(axis=1), coarse, rtol=0, atol=1e-6)
            np.testing.assert_allclose(np.diff(fine, axis=1), 9375.0, rtol=0, atol=1e-6)
            # u = 19 is +3.5 fine widths above the coarse centre
            np.testing.assert_allclose(fine[:, 19] - coarse, 3.5 * 9375.0, atol=1e-6)

    def test_rejects_bad_mode_or_shard(self):
        with self.assertRaises(ValueError):
            bw.shard_frequencies_hz("medium", 0)
        with self.assertRaises(ValueError):
            bw.shard_frequencies_hz("coarse", 2)


class Geometry(unittest.TestCase):
    def test_default_layout_matches_charts_utils_convention(self):
        p = bw.default_positions_charts_logical()
        np.testing.assert_allclose(p[0], [0, 0, 0])
        np.testing.assert_allclose(p[1], [-0.6, 0, 0])
        np.testing.assert_allclose(p[8], [0, 0.6, 0])
        np.testing.assert_allclose(p[63], [-4.2, 4.2, 0])

    def test_migrated_directions(self):
        d = bw.load_directions(bw.DEFAULT_DIRECTIONS)
        self.assertEqual(d.shape, (64, 3))
        np.testing.assert_allclose(np.linalg.norm(d, axis=1), 1.0, atol=1e-12)
        np.testing.assert_allclose(d[0], [0, 0, 1])

    def test_rejects_direction_below_horizon(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "d.json"
            path.write_text(json.dumps({"format": "charts_beam_directions",
                                        "directions": [[0.9, 0.9]]}))
            with self.assertRaises(ValueError):
                bw.load_directions(path)


class Gains(unittest.TestCase):
    def test_response_inverted_and_flags_zeroed(self):
        cube = bw.synthetic_gains(1, n_time=3, flag_antennas=(5,), flag_channels=(7,))
        cube[10, 20, :] = 0
        corr, flagged = bw.gain_correction(cube)
        with np.testing.suppress_warnings() as sup:
            sup.filter(RuntimeWarning)  # fully flagged channel averages to NaN
            avg = np.nanmean(cube, axis=-1)
        good = ~flagged
        np.testing.assert_allclose((corr * avg)[good], 1.0, rtol=1e-12)
        self.assertTrue(np.all(corr[:, 5] == 0) and np.all(corr[7] == 0) and corr[10, 20] == 0)
        self.assertEqual(int(flagged.sum()), 672 + 64 - 1 + 1)

    def test_correction_convention_passthrough(self):
        g = bw.synthetic_gains(2)[..., 0]
        corr, _ = bw.gain_correction(g, "correction")
        np.testing.assert_array_equal(corr, g)

    def test_rejects_bad_gains(self):
        with self.assertRaises(ValueError):
            bw.gain_correction(np.ones((672, 64)))  # real
        with self.assertRaises(ValueError):
            bw.gain_correction(np.ones((672, 32), complex))  # wrong antennas
        bad = np.ones((672, 64), complex)
        bad[0, 0] = np.inf
        with self.assertRaises(ValueError):
            bw.gain_correction(bad)

    def test_fine_copies_coarse_gain_to_32_bins(self):
        corr, _ = bw.gain_correction(bw.synthetic_gains(3)[..., 0])
        fine = bw.shard_gain_correction(corr, "fine", 1)
        self.assertEqual(fine.shape, (10752, 64))
        np.testing.assert_array_equal(fine.reshape(336, 32, 64), np.repeat(
            corr[336:, None, :], 32, axis=1))


class CombinedWeights(unittest.TestCase):
    positions = bw.default_positions_charts_logical()
    directions = bw.load_directions(bw.DEFAULT_DIRECTIONS)

    def test_signal_model_reproduces_charts_utils_visibility_phase(self):
        nu = 350e6
        nx, ny = self.directions[12, :2]
        phase = charts_utils_voltage_phase(nu, self.positions, nx, ny)
        tau = charts_utils_delay(self.positions, nx, ny)
        x = np.exp(1j * phase)
        for a, b in ((0, 1), (5, 63), (40, 9)):
            v_ab = x[a] * np.conj(x[b])
            self.assertAlmostEqual(wrap(np.angle(v_ab) + 2 * np.pi * nu * (tau[a] - tau[b])), 0,
                                   places=9)

    def test_unit_gains_carry_arriving_correlator_phase_in_raw_order(self):
        # W = conj(exp(-i*phi)) = exp(+i*phi), phi the arriving x' phase.
        unit = np.ones((672, 64), complex)
        for shard in (0, 1):
            w = bw.build_weights("coarse", shard, unit, self.positions, self.directions)
            freqs = bw.shard_frequencies_hz("coarse", shard)
            np.testing.assert_allclose(np.abs(w), 1.0, atol=1e-6)
            for b in (0, 1, 37, 63):
                nx, ny = self.directions[b, :2]
                for r in (0, 31, 32, 63):
                    a = 63 - r
                    phi = charts_utils_voltage_phase(freqs, self.positions[a], nx, ny)
                    np.testing.assert_allclose(wrap(np.angle(w[b, :, r]) - phi), 0, atol=2e-6)

    def test_zenith_beam_is_conjugated_gain_correction(self):
        g = bw.synthetic_gains(4)[..., 0]
        corr, _ = bw.gain_correction(g)
        w = bw.build_weights("coarse", 0, corr, self.positions, self.directions[:1])
        np.testing.assert_allclose(w[0], (1 / np.conj(g[:336]))[:, ::-1], rtol=1e-6)

    def _source_response(self, mode, shard, beams, wrong_order=False, weights_hook=None):
        """Inject a unit source from beams[0] through the instrument and beamform."""
        cube = bw.synthetic_gains(5, flag_antennas=(9, 40))
        g = np.nan_to_num(cube[..., 0])
        corr, _ = bw.gain_correction(cube)
        directions = self.directions[beams]
        w = bw.build_weights(mode, shard, corr, self.positions, directions).astype(np.complex128)
        if weights_hook is not None:
            w = weights_hook(w)
        freqs = bw.shard_frequencies_hz(mode, shard)
        g_local = g[336 * shard:336 * (shard + 1)]
        if mode == "fine":
            g_local = np.repeat(g_local, 32, axis=0)
        src = directions[0]
        # charts-utils model for correlator voltages, charts_logical:
        # x'_a = g_a * exp(+i*phi_a), phi_a = 2*pi*nu*(s.p_a)/c.
        phi = charts_utils_voltage_phase(freqs[:, None], self.positions[None, :, :], src[0], src[1])
        x_corr = g_local * np.exp(1j * phi)
        x_bf = 1j * np.conj(x_corr)  # same bytes decoded by the beamformer (see above)
        x_raw = x_bf if wrong_order else x_bf[:, ::-1]  # raw r holds a = 63 - r
        v = np.einsum("bfe,fe->bf", w, x_raw)  # kernel: plain W * x, no conjugate
        n_good = 64 - 2
        return v, n_good, phi, directions, freqs

    def test_steered_beam_recovers_coherent_sum_and_array_factor(self):
        beams = [12, 0, 3, 50]
        good = np.ones(64, bool)
        good[[9, 40]] = False
        for mode, shard in (("coarse", 0), ("coarse", 1), ("fine", 0), ("fine", 1)):
            v, n_good, phi, directions, freqs = self._source_response(mode, shard, beams)
            np.testing.assert_allclose(np.abs(v[0]), n_good, rtol=1e-5)
            for k in range(1, len(beams)):
                nx, ny = directions[k, :2]
                phi_k = charts_utils_voltage_phase(freqs[:, None], self.positions[None, :, :],
                                                   nx, ny)
                af = np.exp(1j * (phi - phi_k))[:, good].sum(axis=1)
                np.testing.assert_allclose(v[k], 1j * np.conj(af), atol=1e-3 * n_good)
                self.assertTrue(np.all(np.abs(v[k]) < n_good))

    def test_missing_component_conjugate_is_detected(self):
        v, n_good, *_ = self._source_response("coarse", 0, [12, 0], weights_hook=np.conj)
        self.assertLess(np.median(np.abs(v[0])), 0.5 * n_good)

    def test_missing_element_reversal_is_detected(self):
        v, n_good, *_ = self._source_response("coarse", 0, [12, 0], wrong_order=True)
        self.assertLess(np.median(np.abs(v[0])), 0.5 * n_good)

    def test_opposite_steering_sign_is_detected(self):
        # Conjugating only the geometric part (CHORD calcBBPhase sign) must lose coherence.
        def flip_geometry(w):
            zenith = w[1:2]  # beams[1] is direction 0 (zenith): pure gain correction C
            return zenith * np.conj(w / np.where(zenith == 0, 1, zenith))
        v, n_good, *_ = self._source_response("coarse", 0, [12, 0], weights_hook=flip_geometry)
        self.assertLess(np.median(np.abs(v[0])), 0.5 * n_good)


class CommandLine(unittest.TestCase):
    def test_synthetic_gains_build_and_descriptor(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            gains = tmp / "gains.npy"
            self.assertEqual(bw.main(["synthetic-gains", "--output", str(gains), "--seed", "7",
                                      "--n-time", "3", "--flag-antennas", "2"]), 0)
            self.assertEqual(np.load(gains).shape, (672, 64, 3))
            out = tmp / "w"
            self.assertEqual(bw.main(["build", "--mode", "coarse", "--gains", str(gains),
                                      "--output-dir", str(out), "--emit-legacy-bin",
                                      "--calibration-note", "synthetic seed 7"]), 0)
            for shard in (0, 1):
                npy = out / f"weights_coarse_shard{shard}.npy"
                w = np.load(npy, allow_pickle=False)
                self.assertEqual((w.dtype, w.shape), (np.complex64, (64, 336, 64)))
                self.assertTrue(w.flags.c_contiguous)
                legacy = np.fromfile(out / f"weights_coarse_shard{shard}.bin", dtype=np.complex64)
                np.testing.assert_array_equal(legacy, w.reshape(-1))
                self.assertEqual(legacy.nbytes, 64 * 336 * 64 * 8)
                # flagged logical antenna 2 is raw element 61
                self.assertTrue(np.all(w[:, :, 61] == 0))
                desc = json.loads((out / f"weights_coarse_shard{shard}.json").read_text())
                self.assertEqual(desc["element_order"], "kotekan_raw")
                self.assertEqual(desc["inputs"]["gains_element_order"], "charts_logical")
                self.assertEqual(desc["shard"], shard)
                self.assertEqual(desc["shape"], [64, 336, 64])
                self.assertEqual(desc["sha256"], bw.sha256_file(npy))
                self.assertTrue(desc["gain"]["calibrated"])


if __name__ == "__main__":
    unittest.main()
