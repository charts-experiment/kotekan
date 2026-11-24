"""Unit tests covering CHARTS experimental spatial profiling, calibration, and RFI/ML modules."""

from __future__ import annotations

import math
import unittest
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from charts.experimental import (
    C_LIGHT_M_S,
    FREQS_MHZ,
    array_factor_power,
    load_antenna_positions_from_yaml,
)
from charts.experimental.auto_rfi_transient_detector import (
    CoherentCombCanceller,
    MatchedFilterTransientScorer,
    UnsupervisedRfiClassifier,
    extract_rfi_snapshot_features,
    fold_profile,
)
from charts.experimental.rfi_lm_profiler import (
    LiveRestLmScanner,
    MultiFrequencyLmProfiler,
    find_optimal_off_beams,
)
from charts.experimental.spatial_calibration import (
    AdaptiveSidelobeCanceller,
    BeamSpaceDelayCalibrator,
    decompose_tilt_and_closure,
)


def _synthetic_2x2_array() -> np.ndarray:
    """Return a clean 4-element 2D (~4.2 m) planar array matching CHARTS commissioned scale."""
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [-4.2, 0.0, 0.0],
            [0.1, 4.3, 0.0],
            [-4.1, 4.35, 0.0],
        ],
        dtype=np.float64,
    )


class TestExperimentalSuite(unittest.TestCase):
    """End-to-end synthetic verification of all charts/experimental components."""

    def test_load_antenna_positions_and_array_factor_peak(self) -> None:
        cfg_path = Path(__file__).resolve().parents[1] / "config" / "64antennas_direct_tracker_10beams.yaml"
        pos, active_ids, meta = load_antenna_positions_from_yaml(cfg_path)
        self.assertEqual(pos.shape, (len(active_ids), 3))
        self.assertGreaterEqual(len(active_ids), 4)
        self.assertIn("conjugate_gains", meta)

        p_peak = array_factor_power(pos, FREQS_MHZ[:16], 0.2, -0.15, 0.2, -0.15)
        self.assertTrue(np.allclose(p_peak, 1.0, atol=1e-12))

    def test_mfs_lm_profiler_localizes_sources_through_bandpass_ripple(self) -> None:
        pos = _synthetic_2x2_array()
        freqs = np.linspace(300.0, 500.0, 48)
        rng = np.random.default_rng(101)

        # Construct a sparse 12-beam spatial fan spanning both zenith and horizon angles
        angles = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
        ring_r = 0.55
        beam_l0 = np.concatenate([[0.0, -0.35, 0.55, 0.15], ring_r * np.cos(angles)])
        beam_m0 = np.concatenate([[0.0, 0.25, -0.20, -0.45], ring_r * np.sin(angles)])

        src1 = (-0.35, 0.25)
        src2 = (0.55, -0.20)
        af1 = array_factor_power(pos, freqs, src1[0], src1[1], beam_l0, beam_m0)
        af2 = array_factor_power(pos, freqs, src2[0], src2[1], beam_l0, beam_m0)

        # Arbitrary receiver bandpass ripple G(f) and isotropic system temperature T_sys(f)
        bandpass = 1.0 + 0.35 * np.sin(0.08 * freqs)
        t_sys = 12.0 + 2.0 * np.cos(0.03 * freqs)
        y = bandpass[None, :] * (t_sys[None, :] + 18.0 * af1 + 11.0 * af2)
        y += rng.normal(0.0, 0.08, size=y.shape)

        profiler = MultiFrequencyLmProfiler(pos, freqs_mhz=freqs, grid_n=41)
        res = profiler.reconstruct(beam_l0, beam_m0, y, ridge_lambda=0.04, l1_lambda=0.35)

        self.assertGreater(res.fit_r2, 0.85)
        self.assertGreaterEqual(len(res.peaks), 2)
        found = [(p["l"], p["m"]) for p in res.peaks[:2]]
        d1 = min(math.hypot(fl - src1[0], fm - src1[1]) for fl, fm in found)
        d2 = min(math.hypot(fl - src2[0], fm - src2[1]) for fl, fm in found)
        self.assertLessEqual(d1, 0.06)
        self.assertLessEqual(d2, 0.06)

    def test_find_optimal_off_beams_reduces_leakage(self) -> None:
        pos = _synthetic_2x2_array()
        freqs = np.linspace(300.0, 500.0, 48)
        cands = find_optimal_off_beams(
            pos,
            freqs,
            target_l=0.20,
            target_m=0.22,
            interferers=[(-0.40, 0.18)],
            min_sep_deg=18.0,
            max_zenith_deg=65.0,
            grid_n=51,
            top_k=3,
        )
        self.assertEqual(len(cands), 3)
        best = cands[0]
        self.assertGreaterEqual(best.target_sep_deg, 18.0)
        self.assertLess(best.band_power_ratio_db, -11.0)

    def test_live_rest_lm_scanner_with_mock_server(self) -> None:
        pos = _synthetic_2x2_array()
        freqs = np.linspace(305.0, 495.0, 32)
        rfi_src = (0.72, -0.18)
        state = {"l0": 0.0, "m0": 0.0}

        def _mock_request(
            method: str, url: str, payload: Optional[Dict[str, Any]]
        ) -> Dict[str, Any]:
            if method == "POST" and url.endswith("/set_target"):
                assert payload is not None
                state["l0"] = float(payload["l0"])
                state["m0"] = float(payload["m0"])
                return {"status": "ok"}
            if method == "GET" and url.endswith("/beam_power"):
                af = array_factor_power(
                    pos, freqs, rfi_src[0], rfi_src[1], state["l0"], state["m0"]
                )
                bp = [np.zeros_like(freqs).tolist() for _ in range(10)]
                bp[9] = (10.0 + 15.0 * af).tolist()
                return {"beam_power": bp}
            raise AssertionError(f"Unexpected mock REST call: {method} {url}")

        scanner = LiveRestLmScanner(
            pos, freqs_mhz=freqs, base_url="http://mock:12048", beam_id=9, requester=_mock_request
        )
        # 16 scan points distributed on concentric rings around sky + target
        pts = [(0.72, -0.18), (0.0, 0.0), (-0.5, 0.5), (-0.7, -0.2), (0.3, 0.7)] + [
            (0.65 * math.cos(t), 0.65 * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 11, endpoint=False)
        ]
        res, _, _ = scanner.raster_scan(pts, n_polls_per_point=2, grid_n=31)
        self.assertGreater(res.fit_r2, 0.90)
        top = res.peaks[0]
        self.assertLessEqual(math.hypot(top["l"] - rfi_src[0], top["m"] - rfi_src[1]), 0.09)

    def test_beam_space_delay_calibrator_recovers_delays(self) -> None:
        pos = _synthetic_2x2_array()
        freqs = np.linspace(300.0, 500.0, 64)
        true_delays_ns = np.array([0.0, 0.35, -0.45, 0.20], dtype=np.float64)
        true_tau_s = true_delays_ns * 1.0e-9

        # 3 snapshots of tracking Sun across slight sky drift
        src_l = np.array([-0.15, 0.05, 0.25])
        src_m = np.array([0.20, 0.18, 0.14])
        offsets = np.array(
            [[0.0, 0.0], [0.04, 0.0], [-0.04, 0.0], [0.08, 0.0], [0.0, 0.05], [0.0, -0.05], [-0.06, -0.04]]
        )
        fan_l0 = src_l[:, None] + offsets[None, :, 0]
        fan_m0 = src_m[:, None] + offsets[None, :, 1]
        off_l0 = np.array([-0.65, -0.60, -0.55])
        off_m0 = np.array([-0.60, -0.62, -0.64])

        af_fan = array_factor_power(
            pos, freqs, src_l[:, None], src_m[:, None], fan_l0, fan_m0, antenna_delays_s=true_tau_s
        )
        af_off = array_factor_power(
            pos, freqs, src_l, src_m, off_l0, off_m0, antenna_delays_s=true_tau_s
        )
        s_cal = 25.0 * (1.0 + 0.2 * np.cos(0.05 * freqs))
        t_sys = 5.0 + 0.01 * freqs
        p_fan = t_sys[None, None, :] + s_cal[None, None, :] * af_fan
        p_off = t_sys[None, :] + s_cal[None, :] * af_off

        cal = BeamSpaceDelayCalibrator(pos, freqs_mhz=freqs)
        res = cal.fit(p_fan, p_off, src_l, src_m, fan_l0, fan_m0, off_l0, off_m0)

        delay_rmse = float(np.sqrt(np.mean((res.delays_ns - true_delays_ns) ** 2)))
        self.assertLess(delay_rmse, 0.05)
        self.assertLess(res.after_rmse, res.before_rmse * 0.1)

        dl, dm, closure_ns = decompose_tilt_and_closure(pos, res.delays_ns)
        expected_dl, expected_dm, expected_closure = decompose_tilt_and_closure(pos, true_delays_ns)
        self.assertAlmostEqual(dl, expected_dl, places=3)
        self.assertAlmostEqual(dm, expected_dm, places=3)
        self.assertAlmostEqual(closure_ns, expected_closure, places=2)

    def test_adaptive_sidelobe_canceller_preserves_on_pulse(self) -> None:
        rng = np.random.default_rng(202)
        n_t, n_f = 1200, 16
        t = np.arange(n_t, dtype=np.float64)

        # Common-mode 1/f + slow Sun sidelobe variation coupled differently per frequency
        interferer = 2.0 * np.sin(2.0 * np.pi * t / 240.0) + 0.8 * np.cumsum(rng.normal(0, 0.02, n_t))
        interferer -= np.mean(interferer)
        cold_off = rng.normal(0.0, 0.05, (n_t, n_f))

        aux = np.zeros((n_t, 2, n_f))
        for f in range(n_f):
            gain_aux = 1.0 + 0.05 * f
            aux[:, 0, f] = 5.0 + gain_aux * interferer + rng.normal(0.0, 0.05, n_t)
            aux[:, 1, f] = 3.0 + cold_off[:, f]

        beam_on = np.zeros((n_t, n_f))
        for f in range(n_f):
            coupling = 0.4 + 0.02 * f
            beam_on[:, f] = 10.0 + coupling * interferer + 0.8 * cold_off[:, f] + rng.normal(0.0, 0.04, n_t)

        # Inject a bright ON-only astronomical pulse centered at t=600
        pulse = 4.0 * np.exp(-0.5 * ((t - 600.0) / 3.0) ** 2)
        beam_on += pulse[:, None]

        cube = np.stack([beam_on, aux[:, 0, :], aux[:, 1, :]], axis=1)  # (T, 3, F)
        canceller = AdaptiveSidelobeCanceller()
        res = canceller.cancel(cube, on_beam_idx=0, aux_beam_indices=[1, 2])

        self.assertGreater(abs(res.corr_before), 0.85)
        self.assertLess(abs(res.corr_after), 0.15)
        self.assertGreater(res.suppression_db, 15.0)

        cleaned_bb = np.median(res.cleaned_on, axis=1)
        pulse_recovered = float(cleaned_bb[600] - np.median(cleaned_bb))
        self.assertGreater(pulse_recovered, 3.5)

    def test_unsupervised_rfi_classifier_separates_ema_latch_and_rfi(self) -> None:
        rng = np.random.default_rng(303)
        n_chan = 200
        freqs = np.linspace(300.0, 499.0, n_chan)
        snapshots = []
        gt_regime = []

        for s in range(12):
            ch_list = []
            for ch in range(n_chan):
                p = 2.0 + rng.normal(0.0, 0.04)
                fz = rng.normal(0.0, 0.7)
                msk = 1.0 + rng.normal(0.0, 0.03)
                xsk = 1.2 + abs(rng.normal(0.0, 0.1))
                frac = 0.0
                bits = 0
                regime = "clean"

                if 20 <= ch < 45:
                    # EMA-latched passband channel: normal MAD z, normal SK, high flagged fraction
                    frac = 1.0
                    bits = 4
                    p = 2.4 + rng.normal(0.0, 0.04)
                    fz = rng.normal(0.8, 0.5)
                    regime = "ema_latch"
                elif 70 <= ch < 90:
                    # Narrowband CW: huge positive freq_mad_z, low SK
                    p = 15.0 + rng.normal(0.0, 0.5)
                    fz = 35.0 + rng.normal(0.0, 2.0)
                    msk = 0.08
                    xsk = 0.15
                    frac = 1.0
                    bits = 10
                    regime = "cw"
                elif 130 <= ch < 145:
                    # Impulsive burst: high max_sk
                    p = 4.5 + rng.normal(0.0, 0.2)
                    fz = 6.0
                    msk = 2.2
                    xsk = 7.5 + abs(rng.normal(0.0, 0.5))
                    frac = 0.4
                    bits = 1
                    regime = "impulsive"

                ch_list.append(
                    {
                        "freq_mhz": float(freqs[ch]),
                        "mean_power": max(0.1, float(p)),
                        "freq_mad_z": float(fz),
                        "mean_sk": max(0.01, float(msk)),
                        "max_sk": max(0.02, float(xsk)),
                        "flagged_chunk_fraction": float(frac),
                        "flag_bits": bits,
                    }
                )
                gt_regime.append(regime)
            snapshots.append({"data": {"channels": ch_list}})

        feats, meta = extract_rfi_snapshot_features(snapshots)
        clf = UnsupervisedRfiClassifier(n_clusters=4, seed=42)
        res = clf.fit_predict(feats, meta=meta)

        gt = np.asarray(gt_regime)
        latch_cids = [cid for cid, nm in res.cluster_names.items() if "latch" in nm]
        self.assertGreaterEqual(len(latch_cids), 1)
        is_latch_pred = np.isin(res.labels, latch_cids)
        precision = float(np.mean(gt[is_latch_pred] == "ema_latch"))
        recall = float(np.mean(is_latch_pred[gt == "ema_latch"]))
        self.assertGreater(precision, 0.95)
        self.assertGreater(recall, 0.95)

    def test_coherent_comb_canceller_preserves_pulsar_and_kills_comb(self) -> None:
        rng = np.random.default_rng(404)
        dt_s = 0.00256
        n_t = 10000
        t = np.arange(n_t, dtype=np.float64) * dt_s

        comb = 0.40 * np.sin(2.0 * np.pi * 9.765625 * t + 0.3) + 0.25 * np.sin(
            2.0 * np.pi * 50.10 * t + 1.1
        )
        # Localized ON-only Vela-like pulsar at P = 89.328 ms
        p_vela = 0.089328
        phase = np.mod(t, p_vela) / p_vela
        pulsar = 0.35 * np.exp(-0.5 * (((phase - 0.35 + 0.5) % 1.0 - 0.5) / 0.03) ** 2)

        off_ts = 3.0 + 0.65 * comb + rng.normal(0.0, 0.03, n_t)
        on_pure = 7.0 + pulsar + rng.normal(0.0, 0.03, n_t)
        on_ts = on_pure + 1.0 * comb

        canceller = CoherentCombCanceller(line_snr_thresh=10.0)
        res = canceller.cancel(on_ts, off_ts, dt_s=dt_s)

        self.assertGreater(res.comb_suppression_db, 15.0)
        prof_pure, _ = fold_profile(on_pure, dt_s, p_vela, n_bins=32)
        prof_clean, snr_clean = fold_profile(res.cleaned_on, dt_s, p_vela, n_bins=32)
        amp_pure = float(np.max(prof_pure) - np.median(prof_pure))
        amp_clean = float(np.max(prof_clean) - np.median(prof_clean))
        self.assertGreater(amp_clean / amp_pure, 0.92)
        self.assertGreater(snr_clean, 10.0)

    def test_matched_filter_transient_scorer_dm_and_off_veto(self) -> None:
        rng = np.random.default_rng(505)
        freqs = np.linspace(300.0, 500.0, 32)
        dt_s = 0.00256
        n_t = 400
        scorer = MatchedFilterTransientScorer(
            freqs_mhz=freqs, dm_trials=[0.0, 30.0, 68.0, 120.0], widths_samples=[1, 2, 4]
        )

        # Case A: Dispersed burst at DM = 68 pc cm^-3 present ONLY in ON beam
        on_spec = rng.normal(10.0, 0.25, (n_t, freqs.size))
        off_spec = rng.normal(5.0, 0.25, (n_t, freqs.size))
        f_top = float(np.max(freqs))
        for ch, f_mhz in enumerate(freqs):
            delay_s = 4148.74 * 68.0 * ((f_mhz ** -2) - (f_top ** -2))
            idx = int(round(120 + delay_s / dt_s))
            if 0 <= idx < n_t:
                on_spec[idx, ch] += 3.0

        cand_a = scorer.score(on_spec, off_spec, dt_s=dt_s)
        self.assertIsNotNone(cand_a)
        assert cand_a is not None
        self.assertFalse(cand_a.rfi_vetoed)
        self.assertAlmostEqual(cand_a.dm_pc_cm3, 68.0, delta=1e-6)
        self.assertGreater(cand_a.snr_on, 8.0)

        # Case B: Broad zero-DM impulse in BOTH ON and OFF -> must be vetoed
        on_rfi = rng.normal(10.0, 0.25, (n_t, freqs.size))
        off_rfi = rng.normal(5.0, 0.25, (n_t, freqs.size))
        on_rfi[200:202, :] += 4.0
        off_rfi[200:202, :] += 3.5
        cand_b = scorer.score(on_rfi, off_rfi, dt_s=dt_s)
        self.assertIsNotNone(cand_b)
        assert cand_b is not None
        self.assertTrue(cand_b.rfi_vetoed)


if __name__ == "__main__":
    unittest.main()
