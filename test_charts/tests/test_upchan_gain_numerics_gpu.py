"""Fine upchannelizer gain G(u) (`upchan_gain`): stage 2, numerical effect of G != 1.

Run with ~/charts-utils/.venv/bin/python -m test_charts.tests.test_upchan_gain_numerics_gpu
    [--results-dir PATH] [--kotekan-bin PATH]

Acceptance (FINE_PATH_SCALLOPING_QUANTIZATION.md §4 steps 2, 4-7):
* Exact scaling before the rounding. Fixture tones/noise/extremes, both shards. The output
  with a non-uniform G and with G = 5.65/H_amp matches the reference fine transform with
  G applied before the clamp and round, with 0 bound violations and exact one-hot power
  where no rounding is ambiguous. Negative controls (G ignored, G mirrored in u, G applied
  after the rounding) are rejected.
* Shaped noise. Coarse noise has the firmware PFB spectrum (CASPER pfb_fir, 8192 points,
  4 taps, hamming, fwidth 1, aliasing included) and is quantized to int4 saturating at ±7.
  - With G = 1 at the f-engine level (σ = 1.20 per component), the centre/edge ratio is
    inflated to ≈ 9.5 dB.
  - With G = 5.65/H_amp it is flat, within the 0.17 dB that the coarse quantization noise
    adds at the edges.
  - At each level and u, the mean window power matches the Gaussian rounded-square model to
    1 % (model tolerance, as in stage C), or to 4 standard errors where the power is small.
* Level dependence. Coarse σ = 0.6, 1.2 and 2.4 per component. Fine power per coarse power
  (divided by G²) varies over several decades with G = 1 and stays close to linear with g₀.
* Clipping. A large gain (G = 8) clips the fixture's fine outputs at ±7 exactly as the
  reference does.
* History and masks. With G != 1, continuous frames, ring wrap and a sequence jump follow
  the reference with the right history (wrong-history hypotheses fail). Exposure files with
  a lossy mask are byte-identical for G = 1 and G = 5.65/H_amp on both shards.
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
# (test_charts/results/fine_scalloping_hu_20261102, analytic column).
H_AMP_HALF = np.array([0.7048, 0.7175, 0.7410, 0.7722, 0.8076, 0.8439, 0.8784, 0.9092,
                       0.9352, 0.9561, 0.9720, 0.9834, 0.9912, 0.9960, 0.9988, 1.0000])
H_AMP = np.concatenate([H_AMP_HALF, H_AMP_HALF[::-1]])
G0 = 5.65
G_PROFILE = (G0 / H_AMP).astype(np.float32)


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
    """Mean one-hot window power per u: K * 2 * E[round(clamp(G sigma z))^2]."""
    return np.array([k * 2 * sc.quantized_square_moments(float(g * s), n_max=2)[0]
                     for s, g in zip(sigma_u, gain)])


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
        got = run_fine(self.root / f"exact_{label}_{shard}", [raw], shard=shard,
                       wfile=self.wfile, gain=gain, t_raw=t_raw)[0]
        q, amb = sc.fine_quantized(raw, None, gain=gain)
        want, bound, ambig = sc.expected_power(q, amb, self.w, 10)
        m = sc.compare(got, want, bound, ambig)
        exact_mask = ~ambig[..., :N1]
        m["onehot_unambiguous_exact"] = bool(np.array_equal(
            got[..., :N1][exact_mask], want[..., :N1][exact_mask].astype(np.float32)))
        m["clipped_components_reference"] = int(np.sum(np.abs(q.real) == 7)
                                                + np.sum(np.abs(q.imag) == 7))
        m["ambiguous_components"] = int(amb.sum())
        controls = {}
        for c_label, c_gain in (("gain_ignored", None),
                                ("gain_mirrored_u", np.asarray(gain)[::-1]),
                                ("gain_after_rounding", "after")):
            if isinstance(c_gain, str):
                q0, _ = sc.fine_quantized(raw, None)
                g = np.tile(np.asarray(gain, np.float64), F)[None, :, None]
                w2, b2, _ = sc.expected_power(q0 * g, None, self.w, 10)
            else:
                w2, b2, _ = sc.expected_power(*sc.fine_quantized(raw, None, gain=c_gain),
                                              self.w, 10)
            controls[c_label] = sc.compare(got, w2, b2)["violations"]
        m["negative_control_violations"] = controls
        record(f"g2_exact_{label}_shard{shard}", m)
        self.assertEqual(m["violations"], 0, m)
        self.assertEqual(m["plan_tolerance_violations"], 0, m)
        self.assertTrue(m["onehot_unambiguous_exact"], m)
        for c_label, n in controls.items():
            if c_label == "gain_mirrored_u" and np.array_equal(gain, np.asarray(gain)[::-1]):
                continue
            self.assertGreater(n, 1000, (c_label, controls))
        return m

    def test_exact_nonuniform_gain_both_shards(self):
        rng = np.random.default_rng(5)
        gain = rng.uniform(0.5, 3.0, U).astype(np.float32)
        for shard in (0, 1):
            self.check_exact("random", shard, gain)

    def test_exact_firmware_profile_both_shards(self):
        for shard in (0, 1):
            self.check_exact("profile", shard, G_PROFILE)

    def test_exact_clipping_large_gain(self):
        gain = np.full(U, 8.0, np.float32)
        gain[0] = 7.5                         # not symmetric, so the mirror control applies
        m = self.check_exact("clip", 0, gain)
        self.assertGreater(m["clipped_components_reference"], 100000, m)

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
                # Model tolerance 1 %, or 4 standard errors where the power is small. The
                # Gaussian model is checked only where G*sigma >= 0.3: below it a nonzero code
                # is a > 1.7 sigma tail event of a 128-tap sum of int4 samples, whose tail is
                # not Gaussian (reported, not asserted).
                "model_tolerance_per_u": np.maximum(0.01, 4 * se / np.maximum(mean, 1e-30)).tolist(),
                "model_checked_u": [int(u) for u in np.flatnonzero(g * sig_u >= 0.3)],
                "model_ok": bool(np.all((np.abs(mean / model - 1)
                                         <= np.maximum(0.01, 4 * se / np.maximum(mean, 1e-30)))
                                        [g * sig_u >= 0.3])),
                "centre_edge_dB_measured": centre_edge_db(mean),
                "centre_edge_dB_model": centre_edge_db(model),
                "centre_edge_dB_unquantized_fine": centre_edge_db((g * sig_u) ** 2),
                "flatness_max_abs_dB": float(np.max(np.abs(db(mean / mean[15:17].mean())))),
                "fine_power_per_coarse_power": float(
                    np.mean(mean[8:24] / g[8:24] ** 2) / (10 * 2 * var_c)),
                "predicted_zero_fraction_centre": float(
                    sc.rounded_gaussian_pmf(float(g[15] * sig_u[15]))[1][8] ** 2),
            }
        return self.noise[key]

    def test_shaped_noise_scalloping_and_flattening(self):
        r1 = self.noise_run(1.2, "unity", None)
        rg = self.noise_run(1.2, "g0_over_H", G_PROFILE)
        report = {"unity": r1, "g0_over_H": rg}
        record("g2_shaped_noise_sigma1.2", report)
        # Diagnosis: the int4 rounding inflates the ~2.9 dB input scalloping to ~9-10 dB.
        self.assertGreater(r1["centre_edge_dB_measured"], 8.0, r1)
        self.assertLess(r1["centre_edge_dB_unquantized_fine"], 3.2, r1)
        # G = g0/H flattens it; the residual is the coarse quantization noise at the edges.
        self.assertLess(abs(rg["centre_edge_dB_measured"]), 0.4, rg)
        self.assertLess(rg["flatness_max_abs_dB"], 0.4, rg)
        # G = 1 is entirely in the tail regime (G*sigma <= 0.22): only the centre/edge ratio is
        # compared with the model. With g0 every bin is checked.
        self.assertEqual(r1["model_checked_u"], [], r1)
        self.assertLess(abs(r1["centre_edge_dB_measured"] - r1["centre_edge_dB_model"]), 0.3, r1)
        self.assertEqual(len(rg["model_checked_u"]), U, rg)
        self.assertTrue(rg["model_ok"], rg)

    def test_level_dependence(self):
        levels = (0.6, 1.2, 2.4)
        report = {"unity": {}, "g0_over_H": {}}
        for sigma in levels:
            report["unity"][sigma] = self.noise_run(sigma, "unity", None)
            report["g0_over_H"][sigma] = self.noise_run(sigma, "g0_over_H", G_PROFILE)
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
        self.assertGreater(summary["unity"]["spread_dB"], 20.0, summary)
        self.assertLess(summary["g0_over_H"]["spread_dB"], 1.5, summary)
        for label in summary:
            self.assertTrue(summary[label]["model_ok"], summary)
        self.assertEqual(summary["g0_over_H"]["model_checked_u_count"], [U, U, U], summary)

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
            q, amb = sc.fine_quantized(frames[i], history, gain=G_PROFILE)
            return sc.compare(got[i], *sc.expected_power(q, amb, self.w, 10))["violations"]

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
        for label, gain in (("unity", None), ("g0_over_H", G_PROFILE)):
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
            a, b = outputs["unity"][shard], outputs["g0_over_H"][shard]
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
