"""GPU replay of the beamformer with offline combined weights (coarse, K=320, float32).

Run with /usr/bin/python3 -m test_charts.tests.test_weights_replay_gpu
[--results-dir PATH] [--kotekan-bin PATH].

Synthetic voltages are built in the charts-utils (correlator) convention
x′ = A·g·exp(+2πiν n̂_s·p/c) for a source in the direction of one catalogue beam,
converted to the beamformer's x = i·conj(x′), reordered to kotekan_raw and packed to
int4 (low nibble real). Gains g are synthetic charts-utils-shaped fake gains with a
flagged antenna and a flagged channel. The weights come from
charts/scripts/build_beamformer_weights.py through the legacy raw `weights_file` reader.

Acceptance uses the analytical array factor, not a CPU beamformer: with W = e^{+iφ_b}/conj(g)
the coherent sum is i·A·Σ_a e^{i(φ_b−φ_s)}, so P_b = K·A²·|Σ_good e^{i2πν(n̂_b−n̂_s)·p/c}|².
Negative controls (no conjugate, opposite sign, no element reversal) must fail it.
Synthetic verification only; it says nothing about the sky.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from test_charts.sim.packet_loss import read_product, write_frame
from test_charts.sim.pipeline import create_beamformer_yaml, execute_kotekan

ROOT = Path(__file__).resolve().parents[2]
BINARY = ROOT / "build/kotekan/kotekan"
RESULTS = None
_spec = importlib.util.spec_from_file_location(
    "build_beamformer_weights", ROOT / "charts/scripts/build_beamformer_weights.py")
bw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bw)

C = 299792458.0
T_RAW, K, N_OUT = 640, 320, 2
FLAG_ANT, FLAG_CHAN = 9, 5


def make_gains():
    """(672, 64) response gains, charts_logical; amplitude limited so int4 keeps SNR."""
    cube = bw.synthetic_gains(3, flag_antennas=(FLAG_ANT,), flag_channels=(FLAG_CHAN,))[:, :, 0]
    amp = np.clip(np.abs(cube), 0.8, 1.25)
    return np.where(np.isnan(cube), np.nan, amp * np.exp(1j * np.angle(cube)))


def pack_voltage(gains, shard, n_source, positions):
    """Return packed [T][336][64] bytes and the per-channel amplitude A_f."""
    nu = bw.shard_frequencies_hz("coarse", shard)
    g = gains[336 * shard:336 * (shard + 1)]
    g = np.where(np.isnan(g), 0.0, g)
    if shard == 0:
        g[FLAG_CHAN, :] = 1.0  # flagged channel: voltage present, weights must null it
    amp = 7.0 / np.max(np.abs(g), axis=1)
    phase = 2 * np.pi * nu[:, None] * (positions @ n_source)[None, :] / C
    x_corr = amp[:, None] * g * np.exp(1j * phase)          # x′ (correlator convention)
    x = 1j * np.conj(x_corr)                                   # beamformer convention
    x = bw.charts_logical_to_kotekan_raw(x, axis=-1)
    re = np.clip(np.rint(x.real), -7, 7).astype(np.int64)
    im = np.clip(np.rint(x.imag), -7, 7).astype(np.int64)
    byts = (((im & 0xF) << 4) | (re & 0xF)).astype(np.uint8)
    return np.broadcast_to(byts, (T_RAW, 336, 64)).copy(), amp


def expected_power(shard, directions, n_source, positions, amp, good):
    nu = bw.shard_frequencies_hz("coarse", shard)
    delta = directions - n_source[None, :]                      # [B][3]
    phase = 2 * np.pi * nu[None, :, None] * (delta @ positions.T)[:, None, :] / C
    af = np.abs(np.sum(np.exp(1j * phase) * good[None, None, :], axis=-1)) ** 2  # [B][F]
    return (K * amp[None, :] ** 2 * af).T                       # [F][B]


class WeightsReplayGpu(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not BINARY.is_file():
            raise unittest.SkipTest(f"no kotekan binary at {BINARY}")
        cls.tmp = tempfile.TemporaryDirectory() if RESULTS is None else None
        cls.root = Path(cls.tmp.name) if cls.tmp else RESULTS / "weights_replay"
        cls.root.mkdir(parents=True, exist_ok=True)
        cls.directions = bw.load_directions(ROOT / "charts/config/charts_hex64_beam_directions.json")
        cls.positions = bw.default_positions_charts_logical()
        cls.gains = make_gains()
        cls.correction, flagged = bw.gain_correction(cls.gains)
        cls.good_ant = ~flagged.all(axis=0)                     # antennas flagged in every channel are off
        transverse = np.hypot(cls.directions[:, 0], cls.directions[:, 1])
        cls.b_src = int(np.argmin(np.abs(transverse - 0.45)))
        cls.n_src = cls.directions[cls.b_src]
        cls.cache = {}

    @classmethod
    def tearDownClass(cls):
        if cls.tmp:
            cls.tmp.cleanup()

    def run_replay(self, name, shard, weights):
        if name in self.cache:
            return self.cache[name]
        work = self.root / name
        inp, out = work / "in", work / "out"
        inp.mkdir(parents=True, exist_ok=True)
        out.mkdir(parents=True, exist_ok=True)
        wfile = work / "weights.bin"
        np.ascontiguousarray(weights, dtype=np.complex64).tofile(wfile)
        volt, amp = pack_voltage(self.gains, shard, self.n_src, self.positions)
        write_frame(inp / f"voltage.shard{shard}_0000000.bin", volt, 0, shard)
        yaml_path = create_beamformer_yaml(
            work / "run.yaml", inp, f"voltage.shard{shard}", out, f"intensity.shard{shard}", 1,
            samples_per_data_set=T_RAW, integration_spectra=K, input_mode="coarse",
            output_format="float32", shard_id=shard, weights_file=wfile)
        rc = execute_kotekan(yaml_path, BINARY, work / "kotekan.log", timeout_s=300)
        files = sorted(out.glob("intensity.shard*_*.bin"))
        if len(files) != 1:
            raise RuntimeError(f"rc={rc}; found {files}; see {work / 'kotekan.log'}")
        _, power = read_product(files[0], "float32", (N_OUT, 336, 64))
        self.cache[name] = (power.astype(np.float64), amp)
        return self.cache[name]

    def nominal_weights(self, shard):
        return bw.build_weights("coarse", shard, self.correction, self.positions, self.directions)

    def metrics(self, shard, power, amp):
        want = expected_power(shard, self.directions, self.n_src, self.positions, amp,
                              self.good_ant.astype(float))
        ok = np.ones(336, bool)
        if shard == 0:
            ok[FLAG_CHAN] = False
        got = power.mean(axis=0)                                # [F][B]; both outputs equal
        peak = want[:, self.b_src]
        return {
            "src_ratio_median": float(np.median(got[ok, self.b_src] / peak[ok])),
            "src_ratio_min": float(np.min(got[ok, self.b_src] / peak[ok])),
            "src_is_argmax_fraction": float(np.mean(np.argmax(got[ok], axis=1) == self.b_src)),
            "off_peak_max_err_over_peak": float(np.max(np.abs(got[ok] - want[ok]) / peak[ok, None])),
            "flagged_channel_power": float(got[FLAG_CHAN].max()) if shard == 0 else None,
            "temporal_consistency": float(np.max(np.abs(power[0] - power[1])) / power.max()),
        }

    def check_nominal(self, shard):
        power, amp = self.run_replay(f"nominal_shard{shard}", shard, self.nominal_weights(shard))
        m = self.metrics(shard, power, amp)
        self.record(f"nominal_shard{shard}", m)
        # Deterministic int4 rounding of the input moves the peak by up to ~4% (measured
        # 3.6%; the GPU matches Σ W·x_quantized to 5e-6 of the peak, a diagnostic only).
        self.assertGreater(m["src_ratio_min"], 0.95)
        self.assertLess(m["src_ratio_min"], 1.05)
        self.assertLess(m["src_ratio_median"], 1.05)
        self.assertEqual(m["src_is_argmax_fraction"], 1.0)
        self.assertLess(m["off_peak_max_err_over_peak"], 0.05)
        self.assertLess(m["temporal_consistency"], 1e-6)
        if shard == 0:
            self.assertEqual(m["flagged_channel_power"], 0.0)

    def record(self, name, metrics):
        if RESULTS:
            (RESULTS / f"{name}_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    def test_shard0_matches_array_factor(self):
        self.check_nominal(0)

    def test_shard1_matches_array_factor(self):
        self.check_nominal(1)

    def check_control(self, name, weights):
        power, amp = self.run_replay(name, 0, weights)
        m = self.metrics(0, power, amp)
        self.record(name, m)
        self.assertLess(m["src_ratio_median"], 0.5, m)
        self.assertLess(m["src_is_argmax_fraction"], 0.5, m)

    def test_negative_control_missing_conjugate(self):
        self.check_control("control_no_conjugate", np.conj(self.nominal_weights(0)))

    def test_negative_control_opposite_sign(self):
        w = bw.build_weights("coarse", 0, self.correction, -self.positions, self.directions)
        self.check_control("control_opposite_sign", w)

    def test_negative_control_no_element_reversal(self):
        self.check_control("control_no_reversal", self.nominal_weights(0)[:, :, ::-1])


# --------------------------------------------------------------------------- fine mode

FINE_T_RAW, FINE_K, FINE_N_OUT = 640, 10, 2


def fine_gain():
    """H0 = sum of the 128 PFB coefficients (documented sinc-Hanning, 1/32 included).

    A tone at a fine-bin centre sees exactly this scalar in its own bin, the same for
    every element, so the steady-state array-factor expectation only gains |H0|².
    """
    s = np.arange(128)
    q = (2 * s - 127) / 258
    return float(np.sum(np.cos(np.pi * q) ** 2 * np.sinc(4 * q) / 32))


def pack_fine_voltage(gains, shard, n_source, positions, h0):
    """[T][336][64] bytes: channel f carries a tone at the centre of fine bin u = f % 32."""
    u = np.arange(336) % 32
    nu = bw.shard_frequencies_hz("fine", shard).reshape(336, 32)[np.arange(336), u]
    g = gains[336 * shard:336 * (shard + 1)]
    g = np.where(np.isnan(g), 0.0, g)
    if shard == 0:
        g[FLAG_CHAN, :] = 1.0
    amp = 7.0 / (np.max(np.abs(g), axis=1) * max(1.0, abs(h0)))
    phase = 2 * np.pi * nu[:, None] * (positions @ n_source)[None, :] / C
    x_corr = amp[:, None] * g * np.exp(1j * phase)
    x = bw.charts_logical_to_kotekan_raw(1j * np.conj(x_corr), axis=-1)       # [F][64]
    n = np.arange(FINE_T_RAW)[:, None]
    # e^{2πi (u−15.5) n/32} with integer phase indexing: exact for any n.
    tone = np.exp(2j * np.pi * (((2 * u[None, :] - 31) * n) % 64) / 64)       # [T][F]
    y = x[None, :, :] * tone[:, :, None]
    re = np.clip(np.rint(y.real), -7, 7).astype(np.int64)
    im = np.clip(np.rint(y.imag), -7, 7).astype(np.int64)
    return (((im & 0xF) << 4) | (re & 0xF)).astype(np.uint8), amp, u


def fine_expected_power(shard, directions, n_source, positions, amp, u, good, h0):
    nu = bw.shard_frequencies_hz("fine", shard).reshape(336, 32)[np.arange(336), u]
    delta = directions - n_source[None, :]
    phase = 2 * np.pi * nu[None, :, None] * (delta @ positions.T)[:, None, :] / C
    af = np.abs(np.sum(np.exp(1j * phase) * good[None, None, :], axis=-1)) ** 2   # [B][F]
    return (FINE_K * (abs(h0) * amp[None, :]) ** 2 * af).T                        # [F][B]


class FineWeightsReplayGpu(WeightsReplayGpu):
    """Same fixtures through the U=32 upchannelizer, steady-state output (spectra 10-19)."""

    # inherit setUpClass only; the coarse test methods are replaced below
    test_shard0_matches_array_factor = None
    test_shard1_matches_array_factor = None
    test_negative_control_missing_conjugate = None
    test_negative_control_opposite_sign = None
    test_negative_control_no_element_reversal = None

    def run_fine(self, name, shard, weights):
        if name in self.cache:
            return self.cache[name]
        work = self.root / name
        inp, out = work / "in", work / "out"
        inp.mkdir(parents=True, exist_ok=True)
        out.mkdir(parents=True, exist_ok=True)
        wfile = work / "weights.bin"
        np.ascontiguousarray(weights, dtype=np.complex64).tofile(wfile)
        h0 = fine_gain()
        volt, amp, u = pack_fine_voltage(self.gains, shard, self.n_src, self.positions, h0)
        write_frame(inp / f"voltage.shard{shard}_0000000.bin", volt, 0, shard)
        yaml_path = create_beamformer_yaml(
            work / "run.yaml", inp, f"voltage.shard{shard}", out, f"intensity.shard{shard}", 1,
            samples_per_data_set=FINE_T_RAW, integration_spectra=FINE_K, input_mode="fine",
            output_format="float32", shard_id=shard, weights_file=wfile)
        rc = execute_kotekan(yaml_path, BINARY, work / "kotekan.log", timeout_s=600)
        files = sorted(out.glob("intensity.shard*_*.bin"))
        if len(files) != 1:
            raise RuntimeError(f"rc={rc}; found {files}; see {work / 'kotekan.log'}")
        _, power = read_product(files[0], "float32", (FINE_N_OUT, 10752, 64))
        self.cache[name] = (power[1].astype(np.float64), amp, u, h0)   # steady-state output
        return self.cache[name]

    def fine_weights(self, shard):
        return bw.build_weights("fine", shard, self.correction, self.positions, self.directions)

    def fine_metrics(self, shard, power, amp, u, h0):
        want = fine_expected_power(shard, self.directions, self.n_src, self.positions, amp, u,
                                   self.good_ant.astype(float), h0)               # [F][B]
        grid = power.reshape(336, 32, 64)
        ok = np.ones(336, bool)
        if shard == 0:
            ok[FLAG_CHAN] = False
        f = np.arange(336)
        got_tone = grid[f, u, :]                                                  # tone bin
        peak = want[:, self.b_src]
        src_bin_argmax = np.argmax(grid[:, :, self.b_src], axis=1)
        return {
            "H0": h0,
            "src_ratio_median": float(np.median(got_tone[ok, self.b_src] / peak[ok])),
            "src_ratio_min": float(np.min(got_tone[ok, self.b_src] / peak[ok])),
            "src_ratio_max": float(np.max(got_tone[ok, self.b_src] / peak[ok])),
            "src_is_beam_argmax_fraction": float(np.mean(np.argmax(got_tone[ok], axis=1) == self.b_src)),
            "tone_in_expected_fine_bin_fraction": float(np.mean(src_bin_argmax[ok] == u[ok])),
            "off_peak_max_err_over_peak": float(np.max(np.abs(got_tone[ok] - want[ok]) / peak[ok, None])),
            "flagged_channel_power_max": float(grid[FLAG_CHAN].max()) if shard == 0 else None,
        }

    def check_fine_nominal(self, shard):
        power, amp, u, h0 = self.run_fine(f"fine_nominal_shard{shard}", shard, self.fine_weights(shard))
        m = self.fine_metrics(shard, power, amp, u, h0)
        self.record(f"fine_nominal_shard{shard}", m)
        self.assertGreater(m["src_ratio_min"], 0.90, m)
        self.assertLess(m["src_ratio_max"], 1.10, m)
        self.assertEqual(m["src_is_beam_argmax_fraction"], 1.0, m)
        self.assertEqual(m["tone_in_expected_fine_bin_fraction"], 1.0, m)
        self.assertLess(m["off_peak_max_err_over_peak"], 0.10, m)
        if shard == 0:
            self.assertEqual(m["flagged_channel_power_max"], 0.0, m)

    def check_fine_control(self, name, weights):
        power, amp, u, h0 = self.run_fine(name, 0, weights)
        m = self.fine_metrics(0, power, amp, u, h0)
        self.record(name, m)
        self.assertLess(m["src_ratio_median"], 0.5, m)
        self.assertLess(m["src_is_beam_argmax_fraction"], 0.5, m)

    def test_fine_shard0_matches_array_factor(self):
        self.check_fine_nominal(0)

    def test_fine_shard1_matches_array_factor(self):
        self.check_fine_nominal(1)

    def test_fine_negative_control_missing_conjugate(self):
        self.check_fine_control("fine_control_no_conjugate", np.conj(self.fine_weights(0)))

    def test_fine_negative_control_opposite_sign(self):
        w = bw.build_weights("fine", 0, self.correction, -self.positions, self.directions)
        self.check_fine_control("fine_control_opposite_sign", w)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--kotekan-bin", type=Path, default=BINARY)
    args, rest = parser.parse_known_args()
    RESULTS = args.results_dir.resolve() if args.results_dir else None
    BINARY = args.kotekan_bin.resolve()
    if RESULTS:
        RESULTS.mkdir(parents=True, exist_ok=True)
    res = unittest.main(argv=[sys.argv[0], *rest], exit=False).result
    summary = {"tests": res.testsRun, "failures": len(res.failures),
               "errors": len(res.errors), "skipped": len(res.skipped)}
    if RESULTS:
        (RESULTS / "weights_replay_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    sys.exit(0 if not (res.failures or res.errors or res.skipped) else 1)
