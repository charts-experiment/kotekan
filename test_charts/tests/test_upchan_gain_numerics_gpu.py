"""Fine upchannelizer gain G(u) (`upchan_gain`): stage 2, numerical effect of G != 1.

Run with ~/charts-utils/.venv/bin/python -m test_charts.tests.test_upchan_gain_numerics_gpu
    [--results-dir PATH] [--kotekan-bin PATH]

fp16 fine path (FP16_TC_ARCHITECTURE.md): G multiplies the fine voltages in fp32 after the
FFT and before the fp16 conversion; there is no clamp or int4 re-quantization, so G is a
linear flux correction of the firmware PFB scalloping (CHORD upchan.jl step 7).

Acceptance:
* Exact placement. Fixture tones/noise/extremes, both shards. The output with a non-uniform
  G and with G = 1/H_amp matches the fp16 reference fine transform with G applied before the
  fp16 rounding, with 0 bound violations and one-hot power equal to the fp16 voltages to
  float32 rounding where no rounding is ambiguous. Negative controls (G ignored, G mirrored
  in u, G applied after the fp16 rounding) are rejected.
* Large G stays linear. With G = 8 = 2^3 in every bin the fp16 voltages are exactly 8 times
  those of G = 1 wherever both are normal fp16 numbers (power-of-two scaling commutes with
  rounding; subnormal sidelobe values, |X| < 2^-14, do not), so the float32 output is exactly
  64 times the G = 1 output in every window whose reference voltages scale exactly, with fine
  components far above the int4 path's +-7 clamp.
* Shaped noise. Coarse noise has the firmware PFB spectrum (CASPER pfb_fir, 8192 points,
  4 taps, hamming, fwidth 1, aliasing included) and is quantized to int4 saturating at ±7.
  - With G = 1 (σ = 1.20 per component) the centre/edge ratio is the unquantized ≈ 2.9 dB
    (the int4 path inflated it to ≈ 9.5 dB).
  - With G = 1/H_amp it is flat, within the 0.17 dB that the coarse quantization noise adds
    at the edges.
  - At each level and u, the mean window power matches the linear model K 2 (G σ_u)^2 to
    1 % (model tolerance, as in stage C) or 4 standard errors, in every bin.
* Level dependence. Coarse σ = 0.6, 1.2 and 2.4 per component. Fine power per coarse power
  (divided by G²) is linear for both G = 1 and G = 1/H_amp (no level-dependent rounding).
* History and masks. With G != 1, continuous frames, ring wrap and a sequence jump follow
  the reference with the right history (wrong-history hypotheses fail). Exposure files with
  a lossy mask are byte-identical for G = 1 and G = 1/H_amp on both shards.
Synthetic verification only; no observational claim.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np
import yaml

from test_charts.sim.packet_loss import create_packet_loss_yaml, write_frame
from test_charts.sim.pipeline import create_beamformer_yaml, execute_kotekan
from test_charts.tests import test_stage_c_numerics_gpu as sc

RESULTS = None
B, E, F, U, FF = 128, 64, 336, 32, 336 * 32
N1 = sc.N_ONEHOT

# Firmware PFB amplitude response per fine bin, normalized to u = 15, 16
# (test_charts/results/fine_scalloping_hu_20261102, analytic column). The CHORD window
# (denominator 256) changes it by < 0.0002 dB, below the 4-digit rounding here.
H_AMP_HALF = np.array([0.7048, 0.7175, 0.7410, 0.7722, 0.8076, 0.8439, 0.8784, 0.9092,
                       0.9352, 0.9561, 0.9720, 0.9834, 0.9912, 0.9960, 0.9988, 1.0000])
H_AMP = np.concatenate([H_AMP_HALF, H_AMP_HALF[::-1]])
# Scalloping correction with the band centre at unity. The normalization of the production
# profile is an open decision (FP16_TC_ARCHITECTURE.md §7); these tests do not depend on it.
G_PROFILE = (1.0 / H_AMP).astype(np.float32)


def record(name, payload):
    if RESULTS:
        (RESULTS / f"{name}.json").write_text(json.dumps(payload, indent=2, default=float) + "\n")


def gain_extra(gain, source="analytic", reference="test"):
    if gain is None:
        return {}
    return {"upchan_gain": [float(g) for g in np.asarray(gain, np.float32)],
            "upchan_gain_source": source, "upchan_gain_reference": reference}


def run_fine(work: Path, frames, *, shard, wfile, gain=None, t_raw, depth=3, sequences=None):
    """rawFileRead -> cudaChartsBeamformerCommand (fine, K=10, float32) -> intensity."""
    inp, out = work / "in", work / "out"
    shutil.rmtree(work, ignore_errors=True)
    inp.mkdir(parents=True)
    out.mkdir(parents=True)
    sequences = sequences or [i * t_raw for i in range(len(frames))]
    for i, (frame, seq) in enumerate(zip(frames, sequences)):
        write_frame(inp / f"voltage.shard{shard}_{i:07d}.bin", frame, seq, shard)
    path = create_beamformer_yaml(
        work / "base.yaml", inp, f"voltage.shard{shard}", out, f"intensity.shard{shard}",
        len(frames), samples_per_data_set=t_raw, integration_spectra=10, input_mode="fine",
        output_format="float32", shard_id=shard, buffer_depth=depth, weights_file=wfile)
    cfg = yaml.safe_load(path.read_text())
    command = next(c for c in cfg["gpu_beamformer"]["commands"]
                   if c["name"] == "cudaChartsBeamformerCommand")
    command.update(gain_extra(gain))
    (work / "run.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    rc = execute_kotekan(work / "run.yaml", sc.BINARY, work / "kotekan.log", timeout_s=600)
    return sc.read_outputs(out, shard, len(frames), rc, work, "fine", t_raw, 10)


# --------------------------------------------------------------------------- firmware PFB

def coarse_spectrum(n: int) -> np.ndarray:
    """|H|^2 summed over coarse aliases |k| <= 2 at the n offsets np.fft.fftfreq(n), centre 1.
    The response is periodic in offset with period 1, so only n/1024-spaced points are
    computed exactly and the rest interpolated (the response is smooth on 1/1024)."""
    nfw, taps = 8192, 4
    length = nfw * taps
    h = np.hamming(length) * np.sinc(np.arange(length) / nfw - taps / 2)
    grid = (np.arange(1025) - 512) / 1024
    s = np.zeros(grid.size)
    m = np.arange(length)
    for k in (-2, -1, 0, 1, 2):
        s += np.abs(np.exp(-2j * np.pi * np.outer(grid + k, m) / nfw) @ h) ** 2
    s /= s[512]
    return np.interp(np.fft.fftfreq(n), grid, s)


def shaped_noise_frame(t_raw: int, sigma: float, seed: int, spectrum: np.ndarray) -> np.ndarray:
    """Packed [T][336][64] int4 noise with the given coarse spectrum (circular over the
    frame), sigma per component before quantization, rounded and saturated at ±7 like the
    f-engine quantizer (no -8)."""
    rng = np.random.default_rng(seed)
    amp = np.sqrt(spectrum / spectrum.mean()).astype(np.float32)[:, None, None]
    raw = np.empty((t_raw, F, E), np.uint8)
    for f0 in range(0, F, 16):
        z = (rng.standard_normal((t_raw, 16, E), np.float32)
             + 1j * rng.standard_normal((t_raw, 16, E), np.float32)).astype(np.complex64)
        x = np.fft.ifft(np.fft.fft(z, axis=0) * amp, axis=0) * sigma
        raw[:, f0:f0 + 16] = sc.encode(np.clip(np.rint(x.real), -7, 7),
                                       np.clip(np.rint(x.imag), -7, 7))
    return raw


def fine_sigma_per_u(raw: np.ndarray, n_seg: int = 1024) -> tuple[np.ndarray, float]:
    """Per-component sigma of the unquantized fine voltage in each bin u, from the measured
    coarse periodogram of the input (every 7th channel, every 4th element: 48 x 16 series, all
    with the same spectrum) and the 128-tap prototype: var_u = mean_nu P(nu) |K_u(nu)|^2 / 2.
    Returns (sigma_u [32], coarse var per component)."""
    sub = raw[: (raw.shape[0] // n_seg) * n_seg, ::7, ::4]
    x = sc.decode(sub).astype(np.complex64)
    seg = x.reshape(-1, n_seg, sub.shape[1], sub.shape[2])
    p = np.zeros(n_seg)
    for i in range(seg.shape[0]):
        p += (np.abs(np.fft.fft(seg[i], axis=0)) ** 2).mean(axis=(1, 2)) / n_seg
    p /= seg.shape[0]                                    # periodogram, mean(p) = E|x|^2
    nu = np.fft.fftfreq(n_seg)
    h = sc.prototype()
    s = np.arange(128)
    var_u = []
    for u in range(U):
        k = np.exp(-2j * np.pi * np.outer(nu - (u - 15.5) / 32, s)) @ h
        var_u.append(np.mean(p * np.abs(k) ** 2) / 2)
    return np.sqrt(np.array(var_u)), float(np.mean(p) / 2)


def window_power_model(sigma_u: np.ndarray, gain: np.ndarray, k: int = 10):
    """Mean one-hot window power per u for fp16 fine voltages (linear): K * 2 * (G sigma)^2."""
    return k * 2 * (np.asarray(gain, np.float64) * np.asarray(sigma_u, np.float64)) ** 2


def measured_window_power(got: np.ndarray):
    """Steady windows (1..), one-hot beams. Mean and standard error per u over independent
    (coarse channel, element) groups."""
    s = got[1:, :, :N1].astype(np.float64).reshape(got.shape[0] - 1, F, U, N1)
    g = s.mean(axis=0)                                   # [F][U][48]
    mean = g.mean(axis=(0, 2))
    se = g.std(axis=(0, 2), ddof=1) / math.sqrt(F * N1)
    return mean, se


def db(x):
    return 10 * np.log10(x)


def centre_edge_db(p):
    edge = np.mean(p[[0, 31]])
    return float(db(np.mean(p[15:17]) / edge)) if edge > 0 else None


class UpchanGainNumericsGpu(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not sc.BINARY.is_file():
            raise unittest.SkipTest(f"no kotekan binary at {sc.BINARY}")
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.w = sc.probe_weights(seed=51)
        cls.wfile = sc.write_weights(cls.root / "probe_w.bin",
                                     np.broadcast_to(cls.w[:, None, :], (B, FF, E)))
        w1 = np.zeros((B, E), np.complex64)
        w1[np.arange(N1), np.arange(N1)] = 1
        cls.w_onehot = sc.write_weights(cls.root / "onehot_w.bin",
                                        np.broadcast_to(w1[:, None, :], (B, FF, E)))
        cls.noise = {}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    # ------------------------------------------------------------------ exact scaling
    def check_exact(self, label, shard, gain, t_raw=1280):
        raw, _ = sc.fine_fixture(t_raw, seed=900 + shard)
        unity = bool(np.all(np.asarray(gain) == 1))
        # An all-unity gain must be configured as the default (source 'unity').
        got = run_fine(self.root / f"exact_{label}_{shard}", [raw], shard=shard,
                       wfile=self.wfile, gain=None if unity else gain, t_raw=t_raw)[0]
        x16, eps = sc.fine_fp16(raw, None, gain=gain)
        want, bound, ambig = sc.expected_power(x16, eps, self.w, 10)
        m = sc.compare(got, want, bound, ambig)
        m.update(sc.onehot_check(got, want, ambig))
        m["components_above_7_reference"] = int(np.sum(np.abs(x16.real) > 7)
                                                + np.sum(np.abs(x16.imag) > 7))
        m["ambiguous_samples"] = int(np.count_nonzero(eps))
        g_arr = np.asarray(gain, np.float64)
        dyadic = bool(np.all(np.log2(g_arr) == np.round(np.log2(g_arr))))
        controls = {}
        for c_label, c_gain in (("gain_ignored", None),
                                ("gain_mirrored_u", g_arr[::-1]),
                                ("gain_after_fp16", "after")):
            if c_label == "gain_ignored" and unity:
                continue
            if c_label == "gain_mirrored_u" and np.array_equal(g_arr, g_arr[::-1]):
                continue
            if c_label == "gain_after_fp16" and dyadic:
                continue              # a power-of-two G commutes with fp16 rounding
            if isinstance(c_gain, str):
                x0, _ = sc.fine_fp16(raw, None)
                g = np.tile(g_arr, F)[None, :, None]
                w2, b2, _ = sc.expected_power(x0 * g, None, self.w, 10)
            else:
                w2, b2, _ = sc.expected_power(*sc.fine_fp16(raw, None, gain=c_gain), self.w, 10)
            controls[c_label] = sc.compare(got, w2, b2)["violations"]
        m["negative_control_violations"] = controls
        record(f"g2_exact_{label}_shard{shard}", m)
        self.assertEqual(m["violations"], 0, m)
        self.assertEqual(m["plan_tolerance_violations"], 0, m)
        self.assertGreater(m["onehot_checked"], 0, m)
        self.assertEqual(m["onehot_violations"], 0, m)
        for c_label, n in controls.items():
            self.assertGreater(n, 1000, (c_label, controls))
        return m, got

    def test_exact_nonuniform_gain_both_shards(self):
        rng = np.random.default_rng(5)
        gain = rng.uniform(0.5, 3.0, U).astype(np.float32)
        for shard in (0, 1):
            self.check_exact("random", shard, gain)

    def test_exact_firmware_profile_both_shards(self):
        for shard in (0, 1):
            self.check_exact("profile", shard, G_PROFILE)

    def test_large_gain_stays_linear(self):
        """G = 8 everywhere: no saturation, output exactly 64 x the G = 1 output."""
        m8, got8 = self.check_exact("large", 0, np.full(U, 8.0, np.float32))
        m1, got1 = self.check_exact("unity", 0, np.ones(U, np.float32))
        raw, _ = sc.fine_fixture(1280, seed=900)
        x8, e8 = sc.fine_fp16(raw, None, gain=np.full(U, 8.0))
        x1, e1 = sc.fine_fp16(raw, None)
        # Samples whose reference does not scale exactly (subnormal) or is ambiguous.
        off = (np.abs(x8.real - 8 * x1.real) + np.abs(x8.imag - 8 * x1.imag) + e8 + e1).astype(np.float32)
        _, _, inexact = sc.expected_power(x8, off, self.w, 10)
        clean = ~inexact
        ratio_exact = bool(np.array_equal(got8[clean], (got1[clean] * np.float32(64))))
        report = {"large": m8, "unity": m1, "output_exactly_64x_unity": ratio_exact,
                  "exactly_scaling_windows": int(clean.sum()), "windows": int(clean.size),
                  "max_over_unity_nonzero": float(np.max(got8[got1 > 0] / got1[got1 > 0])),
                  "min_over_unity_nonzero": float(np.min(got8[got1 > 0] / got1[got1 > 0]))}
        record("g2_large_gain_linear", report)
        self.assertGreater(m8["components_above_7_reference"], 100000, m8)
        self.assertGreater(report["exactly_scaling_windows"], 0, report)
        self.assertTrue(ratio_exact, report)

    # ------------------------------------------------------------------ shaped noise
    def noise_run(self, sigma, gain_label, gain):
        key = (sigma, gain_label)
        if key not in self.noise:
            t_raw = 15360
            if sigma not in self.noise:
                raw = shaped_noise_frame(t_raw, sigma, seed=int(sigma * 1000),
                                         spectrum=coarse_spectrum(t_raw))
                sig_u, var_c = fine_sigma_per_u(raw)
                self.noise[sigma] = (raw, sig_u, var_c)
            raw, sig_u, var_c = self.noise[sigma]
            got = run_fine(self.root / f"noise_{sigma}_{gain_label}", [raw], shard=0,
                           wfile=self.w_onehot, gain=gain, t_raw=t_raw)[0]
            mean, se = measured_window_power(got)
            g = np.ones(U) if gain is None else np.asarray(gain, np.float64)
            model = window_power_model(sig_u, g)
            self.noise[key] = {
                "sigma_in_per_component": sigma, "gain": gain_label,
                "coarse_var_per_component": var_c,
                "fine_sigma_ungained_per_u": sig_u.tolist(),
                "measured_window_power": mean.tolist(), "standard_error": se.tolist(),
                "model_window_power": model.tolist(),
                "model_rel_dev_per_u": (mean / model - 1).tolist(),
                "max_abs_model_rel_dev": float(np.max(np.abs(mean / model - 1))),
                # Linear model (no rounding regime): checked in every bin, tolerance 1 % (the
                # periodogram-based sigma_u estimate) or 4 standard errors.
                "model_tolerance_per_u": np.maximum(0.01, 4 * se / np.maximum(mean, 1e-30)).tolist(),
                "model_checked_u": list(range(U)),
                "model_ok": bool(np.all(np.abs(mean / model - 1)
                                        <= np.maximum(0.01, 4 * se / np.maximum(mean, 1e-30)))),
                "centre_edge_dB_measured": centre_edge_db(mean),
                "centre_edge_dB_model": centre_edge_db(model),
                "centre_edge_dB_unquantized_fine": centre_edge_db((g * sig_u) ** 2),
                "flatness_max_abs_dB": float(np.max(np.abs(db(mean / mean[15:17].mean())))),
                "fine_power_per_coarse_power": float(
                    np.mean(mean[8:24] / g[8:24] ** 2) / (10 * 2 * var_c)),
            }
        return self.noise[key]

    def test_shaped_noise_scalloping_and_flattening(self):
        r1 = self.noise_run(1.2, "unity", None)
        rg = self.noise_run(1.2, "inverse_H", G_PROFILE)
        report = {"unity": r1, "inverse_H": rg}
        record("g2_shaped_noise_sigma1.2", report)
        # Linear path: with G = 1 the measured scalloping is the unquantized ~2.9 dB (the int4
        # path inflated it to ~9.5 dB).
        self.assertLess(abs(r1["centre_edge_dB_measured"] - r1["centre_edge_dB_unquantized_fine"]),
                        0.1, r1)
        self.assertLess(r1["centre_edge_dB_measured"], 3.2, r1)
        self.assertGreater(r1["centre_edge_dB_measured"], 2.5, r1)
        # G = 1/H flattens it; the residual is the coarse quantization noise at the edges.
        self.assertLess(abs(rg["centre_edge_dB_measured"]), 0.4, rg)
        self.assertLess(rg["flatness_max_abs_dB"], 0.4, rg)
        for r in (r1, rg):
            self.assertTrue(r["model_ok"], r)

    def test_level_dependence(self):
        levels = (0.6, 1.2, 2.4)
        report = {"unity": {}, "inverse_H": {}}
        for sigma in levels:
            report["unity"][sigma] = self.noise_run(sigma, "unity", None)
            report["inverse_H"][sigma] = self.noise_run(sigma, "inverse_H", G_PROFILE)
        summary = {}
        for label in report:
            ratios = [report[label][s]["fine_power_per_coarse_power"] for s in levels]
            summary[label] = {"fine_per_coarse_power": ratios,
                              "spread_dB": float(db(max(ratios) / min(ratios))),
                              "centre_edge_dB": [report[label][s]["centre_edge_dB_measured"]
                                                 for s in levels],
                              "max_abs_model_rel_dev": max(
                                  report[label][s]["max_abs_model_rel_dev"] for s in levels),
                              "model_ok": all(report[label][s]["model_ok"] for s in levels),
                              "model_checked_u_count": [len(report[label][s]["model_checked_u"])
                                                        for s in levels]}
        record("g2_level_dependence", {"summary": summary, "runs": report})
        # Linear for both gains (the int4 path spread over > 20 dB with G = 1). Their fine/coarse
        # ratios divided by G^2 must agree; the residual level dependence is only the share of
        # white coarse quantization noise in the input spectrum.
        for label in summary:
            self.assertLess(summary[label]["spread_dB"], 1.5, summary)
            self.assertTrue(summary[label]["model_ok"], summary)
        self.assertLess(abs(summary["unity"]["spread_dB"] - summary["inverse_H"]["spread_dB"]),
                        0.1, summary)

    # ------------------------------------------------------------------ history
    def test_history_ring_and_jump_with_gain(self):
        t_raw, n_frames, depth, jump_at = 640, 7, 2, 4
        frames = [sc.fine_fixture(t_raw, seed=61, n0=i * t_raw, frame_seed=6100 + i)[0]
                  for i in range(n_frames)]
        seqs = [i * t_raw + (t_raw if i >= jump_at else 0) for i in range(n_frames)]
        got = run_fine(self.root / "history", frames, shard=1, wfile=self.wfile,
                       gain=G_PROFILE, t_raw=t_raw, depth=depth, sequences=seqs)
        resets = {0, jump_at}

        def violations(i, history):
            x16, eps = sc.fine_fp16(frames[i], history, gain=G_PROFILE)
            return sc.compare(got[i], *sc.expected_power(x16, eps, self.w, 10))["violations"]

        right, wrong = [], {"no_history": [], "continuous_at_jump": []}
        for i in range(n_frames):
            right.append(violations(i, None if i in resets else frames[i - 1][-96:]))
            if i in resets and i > 0:
                wrong["continuous_at_jump"].append(violations(i, frames[i - 1][-96:]))
            elif i not in resets:
                wrong["no_history"].append(violations(i, None))
        report = {"frames": n_frames, "depth": depth, "sequences": seqs, "jump_at": jump_at,
                  "violations_per_frame": right, "wrong_hypothesis_violations": wrong}
        record("g2_history_ring_jump", report)
        self.assertEqual(sum(right), 0, report)
        for label, counts in wrong.items():
            self.assertTrue(counts and all(c > 0 for c in counts), (label, report))

    # ------------------------------------------------------------------ masks / exposure
    def test_exposure_independent_of_gain(self):
        t_raw, n_frames = 640, 2
        rng = np.random.default_rng(77)
        outputs = {}
        for label, gain in (("unity", None), ("inverse_H", G_PROFILE)):
            work = self.root / f"mask_{label}"
            shutil.rmtree(work, ignore_errors=True)
            inp, out = work / "in", work / "out"
            inp.mkdir(parents=True)
            for shard in (0, 1):
                mrng = np.random.default_rng(1000 + shard)
                for i in range(n_frames):
                    raw = sc.fine_fixture(t_raw, seed=80 + shard, n0=i * t_raw,
                                          frame_seed=8000 + 10 * shard + i)[0]
                    mask = (mrng.random(t_raw) < 0.03).astype(np.uint8)
                    mask[100:140] = 1                      # a contiguous loss
                    raw = raw.copy()
                    raw[mask == 1] = 0                     # replay fixtures are pre-zeroed
                    write_frame(inp / f"voltage.shard{shard}_{i:07d}.bin", raw, i * t_raw, shard)
                    write_frame(inp / f"missing.shard{shard}_{i:07d}.bin", mask, i * t_raw,
                                shard, mask=True)
            path = create_packet_loss_yaml(work / "run.yaml", inp, out, n_frames, mode="fine",
                                           raw_n_time=t_raw)
            cfg = yaml.safe_load(path.read_text())
            for shard in (0, 1):
                command = next(c for c in cfg[f"gpu_{shard}"]["commands"]
                               if c["name"] == "cudaChartsBeamformerCommand")
                command.update(gain_extra(gain))
            path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            rc = execute_kotekan(path, sc.BINARY, work / "kotekan.log", timeout_s=300)
            self.assertEqual(rc, 0, (label, (work / "kotekan.log").read_text()[-3000:]))
            outputs[label] = {shard: {
                "exposure": [p.read_bytes() for p in sorted(out.glob(f"exposure.shard{shard}_*.bin"))],
                "intensity": [p.read_bytes() for p in sorted(out.glob(f"intensity.shard{shard}_*.bin"))]}
                for shard in (0, 1)}
        report = {}
        for shard in (0, 1):
            a, b = outputs["unity"][shard], outputs["inverse_H"][shard]
            report[shard] = {"exposure_files": len(a["exposure"]),
                             "exposure_identical": a["exposure"] == b["exposure"],
                             "intensity_files": len(b["intensity"]),
                             "intensity_differs": a["intensity"] != b["intensity"]}
        record("g2_exposure_independent_of_gain", report)
        for shard, r in report.items():
            self.assertEqual(r["exposure_files"], n_frames, report)
            self.assertEqual(r["intensity_files"], n_frames, report)
            self.assertTrue(r["exposure_identical"], report)
            self.assertTrue(r["intensity_differs"], report)


def main():
    global RESULTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--kotekan-bin", type=Path)
    args, rest = parser.parse_known_args()
    if args.results_dir:
        RESULTS = args.results_dir
        RESULTS.mkdir(parents=True, exist_ok=True)
    if args.kotekan_bin:
        sc.BINARY = args.kotekan_bin
    unittest.main(argv=[sys.argv[0], *rest], verbosity=2)


if __name__ == "__main__":
    main()
