"""Recorded baseband through the unchanged GPU beamformer vs the correlator.

    /usr/bin/python3 -m test_charts.tests.test_real_baseband_replay_gpu \
        --run-dir /data/corr_1020 --event 44 --results-dir test_charts/results/real_replay_<date>
        [--work-dir /dev/shm/charts_real_phase1] [--kotekan-bin build/kotekan/kotekan]

Skips without the binary or the data.

A triggered baseband capture (chartsTriggeredBasebandReadout + BasebandWriter, 1–5
charts_logical antennas, both shards) is placed into native 64-element records (antenna a
in kotekan_raw slot 63 − a, raw bytes copied, other inputs 0x00, FPGA_SEQ_NUM from the
baseband metadata) and replayed in coarse K=320 float32 with probe weights loaded through
the `.npy` + descriptor path:

* beam i (i < A): W = 1 on antenna i only, I_a = Σ |x_a|²;
* beam A + 4p + k for pair p = (a, b): W_a = 1, W_b = e^{ikπ/2}, so
  I_ab(φ) = I_a + I_b + 2·Re(e^{iφ}·C_ab) with C_ab = Σ x_b·conj(x_a);
* all other beams zero.

Every weight is ±1 or ±i and every window sum is an integer below 2²⁴, so the GPU result is
exact: zero beams are 0 and I(0)+I(180) = I(90)+I(270) = 2(I_a+I_b). With x′ = i·conj(x),
the correlator's V′_ab = x′_a·conj(x′_b) = conj(x_a)·x_b = C_ab, and V′_aa = I_a.

Two comparison modes, chosen from the data:

* exact — correlator integrations lying entirely inside the capture: per integration and
  channel, I_a = V′_aa, Re C_ab = Re V′_ab exactly, and Im C_ab = Im V′_ab exactly in every
  channel without a −8 nibble on a or b (the correlator's −8 imaginary error,
  test_correlator_convention_gpu). Lost samples are zeroed upstream of both, so they do
  not break equality.
* statistical — otherwise, against the integration sharing most samples: normalized auto
  spectra and per-channel cross-term phase, with conjugate and channel-reversal controls
  that must fail. Requires `lost_timesamples` = 0 in that integration.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys
import unittest

import numpy as np

from test_charts.sim.packet_loss import read_product
from test_charts.sim.pipeline import create_beamformer_yaml, execute_kotekan
from test_charts.sim.real_data import (convert_baseband, corr_matrix, open_baseband,
                                      read_charts_metadata, read_corr_raw)

ROOT = Path(__file__).resolve().parents[2]
BINARY = ROOT / "build/kotekan/kotekan"
RUN_DIR = Path("/data/corr_1020")
EVENT = 44
RESULTS = None
WORK = Path("/dev/shm/charts_real_phase1")
_spec = importlib.util.spec_from_file_location(
    "build_beamformer_weights", ROOT / "charts/scripts/build_beamformer_weights.py")
bw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bw)

T_RAW, K = 15360, 320
N_WIN = T_RAW // K
PHASES = np.array([1, 1j, -1, -1j])          # phi = 0, 90, 180, 270 deg
# Statistical acceptance, fixed before the first run (2026-10-29).
COH_MIN = 0.05            # channels used for phase: correlator coherence >= this
PHASE_MEDIAN_MAX = 0.10   # rad, median |arg(rho_bf conj(rho_corr))|
CONTROL_MEDIAN_MIN = 0.30 # rad, controls must exceed this
AUTO_RATIO_RANGE = (0.8, 1.25)  # median per-channel ratio of normalized autos


def probe_weights(antennas):
    pairs = [(i, j) for i in range(len(antennas)) for j in range(i + 1, len(antennas))]
    n_used = len(antennas) + 4 * len(pairs)
    if n_used > 64:
        raise ValueError(f"{len(antennas)} antennas need {n_used} probe beams; at most 5 fit")
    w = np.zeros((64, 336, 64), np.complex64)
    for i, a in enumerate(antennas):
        w[i, :, 63 - a] = 1
    for p, (i, j) in enumerate(pairs):
        for k, ph in enumerate(PHASES):
            b = len(antennas) + 4 * p + k
            w[b, :, 63 - antennas[i]] = 1
            w[b, :, 63 - antennas[j]] = ph
    return w, pairs, n_used


def write_weights(directory: Path, shard: int, weights: np.ndarray) -> tuple[Path, Path]:
    npy = directory / f"probe_weights_shard{shard}.npy"
    np.save(npy, weights)
    inputs = {"gain_convention": "response", "gains": None,
              "calibration_note": "probe weights: one-hot autos and 4-phase pairs, "
                                  "unused inputs 0; not a sky beam"}
    desc = bw.descriptor("coarse", shard, npy, weights.shape[0],
                         np.zeros((672, 64), bool), inputs)
    js = npy.with_suffix(".json")
    js.write_text(json.dumps(desc, indent=1))
    return npy, js


def run_integration_spectra(run_dir: Path) -> int:
    configs = sorted(run_dir.glob("kotekan_config_*.json"))
    if not configs:
        raise RuntimeError(f"no kotekan_config_*.json in {run_dir}")
    cfg = json.loads(configs[-1].read_text())
    return int(cfg["samples_per_data_set"]) * int(cfg["integration_frames"])


def correlator_index(run_dir: Path, shard: int) -> tuple[list[Path], np.ndarray, np.ndarray]:
    """Correlator files of one shard (any hostname prefix), with seq and lost samples."""
    rx = re.compile(rf"^(?:.+_)?corr_freq_{shard}_(\d+)\.bin$")
    files = sorted((p for p in run_dir.iterdir() if rx.match(p.name)),
                   key=lambda p: int(rx.match(p.name).group(1)))
    meta = [read_charts_metadata(p) for p in files]
    return (files, np.array([m.fpga_seq_num for m in meta], np.int64),
            np.array([m.lost_timesamples for m in meta], np.int64))


def replay_shard(run_dir: Path, event: int, shard: int, antennas, weights) -> dict:
    """Convert, run Kotekan; per-frame float64 beam sums [N][336][64] and −8 counts."""
    work = WORK / f"shard{shard}"
    shutil.rmtree(work, ignore_errors=True)
    inp, out = work / "in", work / "out"
    inp.mkdir(parents=True)
    out.mkdir(parents=True)
    bb = open_baseband(run_dir / f"baseband_raw_{event}" / f"baseband_{event}_{336 * shard}.data",
                       T_RAW)
    seqs, minus8 = convert_baseband(bb, antennas, shard, inp, f"voltage.shard{shard}")
    npy, js = write_weights(work, shard, weights)
    yaml_path = create_beamformer_yaml(
        work / "run.yaml", inp, f"voltage.shard{shard}", out, f"intensity.shard{shard}",
        len(seqs), samples_per_data_set=T_RAW, integration_spectra=K, input_mode="coarse",
        output_format="float32", shard_id=shard, weights_file=npy)
    text = yaml_path.read_text().replace(
        f'weights_file: "{npy.as_posix()}"',
        f'weights_file: "{npy.as_posix()}"\n      weights_descriptor: "{js.as_posix()}"')
    yaml_path.write_text(text)
    rc = execute_kotekan(yaml_path, BINARY, work / "kotekan.log", timeout_s=1800)
    log = (work / "kotekan.log").read_text(errors="replace")
    files = sorted(out.glob("intensity.shard*_*.bin"))
    if len(files) != len(seqs):
        raise RuntimeError(f"rc={rc}: {len(files)} of {len(seqs)} outputs; {work}/kotekan.log")
    per_frame = np.zeros((len(files), 336, 64))
    out_seqs = []
    for n, p in enumerate(files):
        meta, power = read_product(p, "float32", (N_WIN, 336, 64))
        out_seqs.append(int(meta.fpga_seq_num))
        per_frame[n] = power.astype(np.float64).sum(axis=0)
    result = {"per_frame": per_frame, "minus8": minus8, "seqs": seqs,
              "frames": len(seqs), "spectra": len(seqs) * T_RAW,
              "seq_first": int(seqs[0]), "seq_end": int(seqs[-1]) + T_RAW,
              "contiguous": bool(np.all(np.diff(seqs) == T_RAW)),
              "output_seqs_match": out_seqs == seqs.tolist(), "rc": rc,
              "npy_loaded": "CHARTS weights loaded from .npy" in log}
    if RESULTS:
        shutil.copy(work / "kotekan.log", RESULTS / f"kotekan_shard{shard}.log")
        shutil.copy(js, RESULTS / js.name)
    shutil.rmtree(work)
    return result


def phase_residual(rho_bf, rho_c, use):
    return np.abs(np.angle(rho_bf[use] * np.conj(rho_c[use])))


def cross_from_beams(power, n_ant, p):
    """C_ab from the four phase beams of pair p; power [..., 64]."""
    b = n_ant + 4 * p
    i0, i90, i180, i270 = (power[..., b + k] for k in range(4))
    return (i0 - i180) / 4 + 1j * (i270 - i90) / 4


class RealBasebandReplayGpu(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not BINARY.is_file():
            raise unittest.SkipTest(f"no kotekan binary at {BINARY}")
        sel_path = RUN_DIR / f"baseband_raw_{EVENT}/antenna_selection.json"
        if not sel_path.is_file():
            raise unittest.SkipTest(f"no recorded baseband at {sel_path.parent}")
        sel = json.loads(sel_path.read_text())
        if sel["index_convention"] != "charts_logical":
            raise RuntimeError(f"unexpected index convention {sel['index_convention']}")
        cls.antennas = list(sel["selected_antennas"])
        cls.n_ant = len(cls.antennas)
        cls.weights, cls.pairs, cls.n_used = probe_weights(cls.antennas)
        cls.integration = run_integration_spectra(RUN_DIR)
        if cls.integration % T_RAW:
            raise RuntimeError(f"integration {cls.integration} is not a multiple of {T_RAW}")
        cls.bf = [replay_shard(RUN_DIR, EVENT, s, cls.antennas, cls.weights) for s in (0, 1)]
        cls.per_frame = np.concatenate([r["per_frame"] for r in cls.bf], axis=1)  # [N][672][64]
        cls.minus8 = np.concatenate([r["minus8"] for r in cls.bf], axis=1)        # [N][672][A]
        cls.power = cls.per_frame.sum(axis=0)                                      # [672][64]
        cls.seqs = cls.bf[0]["seqs"]
        start, end = cls.bf[0]["seq_first"], cls.bf[0]["seq_end"]

        index = [correlator_index(RUN_DIR, s) for s in (0, 1)]
        seq_c = index[0][1]
        inside = np.nonzero((seq_c >= start) & (seq_c + cls.integration <= end))[0]
        cls.mode = "exact" if inside.size else "statistical"
        if cls.mode == "exact":
            cls.ref = inside.tolist()
        else:
            cover = np.minimum(end, seq_c + cls.integration) - np.maximum(start, seq_c)
            if cover.max() <= 0:
                raise RuntimeError("no correlator integration overlaps the capture")
            cls.ref = [int(np.argmax(cover))]
        cls.ref_seq, cls.ref_lost, cls.corr = [], [], []
        for k in cls.ref:
            files = [index[s][0][k] for s in (0, 1)]
            cls.ref_seq.append([int(index[s][1][k]) for s in (0, 1)])
            cls.ref_lost.append([int(index[s][2][k]) for s in (0, 1)])
            cls.corr.append(np.concatenate([corr_matrix(read_corr_raw(p), cls.antennas)
                                            for p in files]))                   # [672][A][A]
        cls.metrics = {
            "run_dir": str(RUN_DIR), "event": EVENT, "mode": cls.mode,
            "antennas": cls.antennas,
            "pairs": [[cls.antennas[i], cls.antennas[j]] for i, j in cls.pairs],
            "integration_spectra": cls.integration,
            "beamformer": [{k: v for k, v in r.items()
                            if k not in ("per_frame", "minus8", "seqs")} for r in cls.bf],
            "correlator_integrations": cls.ref, "correlator_seq": cls.ref_seq,
            "correlator_lost_timesamples": cls.ref_lost,
            "minus8_fraction": (cls.minus8.sum(axis=(0, 1))
                                / (cls.minus8.shape[0] * T_RAW * 672)).tolist(),
        }
        if cls.mode == "statistical":
            cls.metrics["frame_series"] = cls.frame_series()
        if RESULTS:
            np.savez(RESULTS / "beam_sums.npz", power=cls.power, per_frame=cls.per_frame,
                     corr=np.array(cls.corr), antennas=cls.antennas, seqs=cls.seqs,
                     correlator_integrations=cls.ref)

    @classmethod
    def tearDownClass(cls):
        if RESULTS and hasattr(cls, "metrics"):
            (RESULTS / "metrics.json").write_text(json.dumps(cls.metrics, indent=1) + "\n")

    @classmethod
    def frame_series(cls):
        """Diagnostic only (no acceptance): per frame, median auto power per spectrum and
        signed median phase residual of each pair against the reference integration."""
        c = cls.corr[0]
        series = {"seconds_per_frame": T_RAW / 300000.0,
                  "auto_median_per_spectrum": np.round(
                      np.median(cls.per_frame[:, :, :cls.n_ant], axis=1) / T_RAW, 4).tolist(),
                  "signed_median_dphi": {}}
        for p, (i, j) in enumerate(cls.pairs):
            rc = c[:, i, j] / np.sqrt(c[:, i, i].real * c[:, j, j].real)
            use = np.abs(rc) >= COH_MIN
            f = cls.per_frame
            rb = cross_from_beams(f, cls.n_ant, p) / np.sqrt(f[:, :, i] * f[:, :, j])
            series["signed_median_dphi"][f"{cls.antennas[i]}-{cls.antennas[j]}"] = np.round(
                np.median(np.angle(rb[:, use] * np.conj(rc[use])), axis=1), 4).tolist()
        return series

    def require(self, mode):
        if self.mode != mode:
            self.skipTest(f"{mode} comparison not applicable: this capture is {self.mode}")

    # ------------------------------------------------------------ exact GPU checks
    def test_frames_and_metadata(self):
        for r in self.bf:
            self.assertEqual(r["rc"], 0)
            self.assertTrue(r["npy_loaded"])
            self.assertTrue(r["output_seqs_match"])
            self.assertTrue(r["contiguous"])
        self.assertEqual(self.bf[0]["seq_first"], self.bf[1]["seq_first"])
        for seq in self.ref_seq:
            self.assertEqual(seq[0], seq[1])

    def test_zero_weight_beams_exactly_zero(self):
        self.assertEqual(float(np.abs(self.power[:, self.n_used:]).max()), 0.0)

    def test_phase_quadruples_exact(self):
        ia = self.power[:, :self.n_ant]
        worst = 0.0
        for p, (i, j) in enumerate(self.pairs):
            b = self.n_ant + 4 * p
            want = 2 * (ia[:, i] + ia[:, j])
            for k0, k1 in ((0, 2), (1, 3)):
                got = self.power[:, b + k0] + self.power[:, b + k1]
                worst = max(worst, float(np.abs(got - want).max()))
        self.metrics["phase_quadruple_max_abs_error"] = worst
        self.assertEqual(worst, 0.0)

    def test_values_are_integers(self):
        used = self.power[:, :self.n_used]
        self.assertEqual(float(np.abs(used - np.rint(used)).max()), 0.0)
        self.assertGreater(float(self.power[:, :self.n_ant].min()), 0.0)

    # ------------------------------------------------------- exact correlator check
    def test_exact_equality_with_correlator(self):
        self.require("exact")
        rows = []
        for k, c in zip(self.ref, self.corr):
            s0 = int(self.ref_seq[self.ref.index(k)][0])
            sel = (self.seqs >= s0) & (self.seqs < s0 + self.integration)
            self.assertEqual(int(sel.sum()), self.integration // T_RAW,
                             "integration not aligned to the 15360-spectrum records")
            p = self.per_frame[sel].sum(axis=0)
            m8 = self.minus8[sel].sum(axis=0)                                 # [672][A]
            row = {"integration": k,
                   "auto_max_abs_diff": float(np.abs(
                       p[:, :self.n_ant] - np.real(np.einsum("fii->fi", c))).max()),
                   "pairs": []}
            for q, (i, j) in enumerate(self.pairs):
                cb = cross_from_beams(p, self.n_ant, q)
                clean = (m8[:, i] == 0) & (m8[:, j] == 0)
                d_im = np.abs(cb.imag - c[:, i, j].imag)
                row["pairs"].append({
                    "pair": [self.antennas[i], self.antennas[j]],
                    "real_max_abs_diff": float(np.abs(cb.real - c[:, i, j].real).max()),
                    "imag_max_abs_diff_without_minus8": float(d_im[clean].max()) if clean.any() else None,
                    "channels_without_minus8": int(clean.sum()),
                    "imag_mismatch_channels_with_minus8": int((d_im[~clean] > 0).sum()),
                    "control_conjugate_imag_max_abs_diff": float(np.abs(
                        -cb.imag - c[:, i, j].imag)[clean].max()) if clean.any() else None,
                })
            rows.append(row)
        self.metrics["exact"] = rows
        for row in rows:
            self.assertEqual(row["auto_max_abs_diff"], 0.0, row)
            for pr in row["pairs"]:
                self.assertEqual(pr["real_max_abs_diff"], 0.0, pr)
                if pr["channels_without_minus8"]:
                    self.assertEqual(pr["imag_max_abs_diff_without_minus8"], 0.0, pr)
                    self.assertGreater(pr["control_conjugate_imag_max_abs_diff"], 0.0, pr)

    # ------------------------------------------------------- statistical checks
    def rho(self):
        """Normalized cross terms, beamformer and correlator: [672][pairs]."""
        ia, c = self.power[:, :self.n_ant], self.corr[0]
        rb, rc = [], []
        for p, (i, j) in enumerate(self.pairs):
            rb.append(cross_from_beams(self.power, self.n_ant, p) / np.sqrt(ia[:, i] * ia[:, j]))
            rc.append(c[:, i, j] / np.sqrt(c[:, i, i].real * c[:, j, j].real))
        return np.array(rb).T, np.array(rc).T

    def test_statistical_reference_has_no_loss(self):
        self.require("statistical")
        # The replay carries no masks: valid only while the reference lost nothing.
        self.assertEqual(self.ref_lost[0], [0, 0])

    def test_auto_spectra_match_correlator(self):
        self.require("statistical")
        n_bf = self.bf[0]["spectra"]
        ia = self.power[:, :self.n_ant] / n_bf
        ca = np.real(np.einsum("fii->fi", self.corr[0])) / self.integration
        ratio = ia / ca
        n = self.n_ant
        confusion = np.corrcoef(np.log(ia).T, np.log(ca).T)[:n, n:] if n > 1 else np.ones((1, 1))
        m = {"median_ratio": np.median(ratio, axis=0).tolist(),
             "ratio_p05_p95": np.percentile(ratio, [5, 95], axis=0).tolist(),
             "log_spectrum_correlation": np.atleast_2d(confusion).tolist()}
        self.metrics["autos"] = m
        for a in range(n):
            self.assertGreater(m["median_ratio"][a], AUTO_RATIO_RANGE[0], m)
            self.assertLess(m["median_ratio"][a], AUTO_RATIO_RANGE[1], m)
            self.assertEqual(int(np.argmax(np.atleast_2d(confusion)[a])), a, m)

    def test_cross_phase_matches_correlator(self):
        self.require("statistical")
        if not self.pairs:
            self.skipTest("one antenna: no cross terms")
        rb, rc = self.rho()
        out = []
        for p, (i, j) in enumerate(self.pairs):
            use = np.abs(rc[:, p]) >= COH_MIN
            row = {"pair": [self.antennas[i], self.antennas[j]], "channels": int(use.sum())}
            if use.sum():
                d = phase_residual(rb[:, p], rc[:, p], use)
                row.update({
                    "median_abs_dphi": float(np.median(d)),
                    "p90_abs_dphi": float(np.percentile(d, 90)),
                    "median_abs_corr_phase": float(np.median(np.abs(np.angle(rc[use, p])))),
                    "coherence_ratio_median": float(np.median(np.abs(rb[use, p]) / np.abs(rc[use, p]))),
                    "control_conjugate_median": float(np.median(
                        phase_residual(np.conj(rb[:, p]), rc[:, p], use))),
                    "control_channel_reversed_median": float(np.median(
                        phase_residual(np.concatenate([rb[335::-1, p], rb[:335:-1, p]]),
                                       rc[:, p], use))),
                })
            out.append(row)
        self.metrics["cross"] = out
        tested = [r for r in out if r["channels"] >= 10]
        self.assertTrue(tested, out)
        for r in tested:
            self.assertLess(r["median_abs_dphi"], PHASE_MEDIAN_MAX, r)
            # Controls only discriminate when the phases are away from 0 and pi.
            if 0.3 < r["median_abs_corr_phase"] < np.pi - 0.3:
                self.assertGreater(r["control_conjugate_median"], CONTROL_MEDIAN_MIN, r)
            self.assertGreater(r["control_channel_reversed_median"], CONTROL_MEDIAN_MIN, r)


def main() -> int:
    global RESULTS, RUN_DIR, EVENT, WORK, BINARY
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path)
    ap.add_argument("--run-dir", type=Path, default=RUN_DIR)
    ap.add_argument("--event", type=int, default=EVENT)
    ap.add_argument("--work-dir", type=Path, default=WORK,
                    help="scratch for converted frames (≈ 330 MB per 51.2 ms frame per shard)")
    ap.add_argument("--kotekan-bin", type=Path, default=BINARY)
    args, rest = ap.parse_known_args()
    RUN_DIR, EVENT, WORK, BINARY = args.run_dir, args.event, args.work_dir, args.kotekan_bin.resolve()
    if args.results_dir:
        RESULTS = args.results_dir.resolve()
        RESULTS.mkdir(parents=True, exist_ok=True)
    res = unittest.main(argv=[sys.argv[0], *rest], exit=False, verbosity=2).result
    if RESULTS:
        (RESULTS / "summary.json").write_text(json.dumps({
            "tests": res.testsRun, "failures": len(res.failures), "errors": len(res.errors),
            "skipped": len(res.skipped)}, indent=1) + "\n")
    return 0 if res.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
