#!/usr/bin/env python3
"""Unit tests for closed-loop telemetry and daemon helpers in self_tune_tracker.py."""

import json
import os
import socket
import struct
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import self_tune_tracker as stt


def _make_fake_bin_frame(seq: int, sec: int, usec: int, spectrum: np.ndarray) -> bytes:
    """Serialize one synthetic kotekan rawFileWrite beam_power frame."""
    meta = bytearray(3024)
    struct.pack_into("<Q", meta, 8, seq)
    struct.pack_into("<QQ", meta, 3008, sec, usec)
    payload = np.asarray(spectrum, dtype="<f4").tobytes()
    return struct.pack("<I", len(meta)) + bytes(meta) + payload


def _make_clean_channels(n_chan: int = stt.N_CHAN_TOTAL) -> list:
    """Generate clean 672-channel telemetry rows across SB0 and SB1."""
    rows = []
    for i in range(n_chan):
        sb = 0 if i < stt.N_CHAN_SUBBAND else 1
        lch = i if sb == 0 else (i - stt.N_CHAN_SUBBAND)
        f0 = stt.SUBBAND0_START_HZ if sb == 0 else stt.SUBBAND1_START_HZ
        freq_mhz = round((f0 + lch * stt.CHANNEL_WIDTH_HZ) / 1e6, 3)
        rows.append(
            {
                "freq_mhz": freq_mhz,
                "subband": sb,
                "local_channel": lch,
                "mean_power": 100.0 + (20.0 if lch == 100 else 0.0),
                "mean_sk": 1.0,
                "max_sk": 1.05,
                "freq_mad_z": 0.2,
                "flagged_chunk_fraction": 0.0,
                "flag_bits": 0,
                "gain": 1.0,
            }
        )
    return rows


class TestSelfTuneTrackerClosedLoop(unittest.TestCase):
    def test_compute_raw_rfi_gains_interpolates_rfi_channels(self) -> None:
        channels = _make_clean_channels()
        # Smooth bandpass slope in SB0 with two strong RFI spikes that must be excluded
        for i in range(stt.N_CHAN_SUBBAND):
            channels[i]["mean_power"] = 100.0 + 0.5 * i
        # Impulsive / CW / saturated RFI spikes should not perturb smooth gains
        channels[50]["mean_power"] = 5000.0
        channels[50]["flagged_chunk_fraction"] = 0.90
        channels[50]["flag_bits"] = stt.RFI_SK_CW_TONE | stt.RFI_FREQ_MAD
        channels[50]["freq_mad_z"] = 45.0

        channels[200]["mean_power"] = 9000.0
        channels[200]["flagged_chunk_fraction"] = 0.50
        channels[200]["flag_bits"] = stt.RFI_SATURATION_CLIP
        channels[200]["freq_mad_z"] = 1.0

        g_sb0, g_sb1, meta = stt.compute_raw_rfi_gains_from_channels(channels)
        self.assertIsNotNone(g_sb0)
        self.assertIsNotNone(g_sb1)
        assert g_sb0 is not None and g_sb1 is not None
        self.assertEqual(g_sb0.shape, (stt.N_CHAN_SUBBAND,))
        self.assertEqual(g_sb1.shape, (stt.N_CHAN_SUBBAND,))
        self.assertEqual(meta["clean_raw_channels"], stt.N_CHAN_TOTAL - 2)
        # Gains around ch 50 and ch 200 should stay close to neighbor gains (not jump toward sqrt(5000))
        self.assertLess(abs(float(g_sb0[50] - g_sb0[49])), 0.03)
        self.assertLess(abs(float(g_sb0[200] - g_sb0[199])), 0.03)

    def test_detect_latched_ema_channels_and_cooldown(self) -> None:
        channels = _make_clean_channels()
        # False EMA latch: full TIME_BURST|SIR flag, Gaussian SK ~1.0, normal MAD
        latched_f = channels[410]["freq_mhz"]
        channels[410].update(
            {
                "flagged_chunk_fraction": 1.0,
                "flag_bits": stt.RFI_TIME_BURST | stt.RFI_MORPHOLOGY_SIR,
                "mean_sk": 0.96,
                "freq_mad_z": 0.4,
            }
        )
        # Real CW tone also has FREQ_MAD bit and abnormal SK/MAD -> must NOT be reset as false latch
        cw_f = channels[420]["freq_mhz"]
        channels[420].update(
            {
                "flagged_chunk_fraction": 1.0,
                "flag_bits": stt.RFI_TIME_BURST | stt.RFI_FREQ_MAD | stt.RFI_SK_CW_TONE,
                "mean_sk": 0.05,
                "freq_mad_z": 18.0,
            }
        )

        streaks: dict[float, int] = {}
        last_reset: dict[float, float] = {}
        # Cycle 1 (streak=1 < 2): nothing returned yet
        res1 = stt.detect_latched_ema_channels(channels, streaks, last_reset, now=1000.0)
        self.assertEqual(res1, [])
        self.assertEqual(streaks[latched_f], 1)
        self.assertEqual(streaks[cw_f], 0)

        # Cycle 2 (streak=2 >= 2): latched channel triggers reset
        res2 = stt.detect_latched_ema_channels(channels, streaks, last_reset, now=1030.0)
        self.assertEqual(res2, [latched_f])
        self.assertEqual(last_reset[latched_f], 1030.0)

        # Within 180s cooldown, even after 2 more latched cycles it waits
        stt.detect_latched_ema_channels(channels, streaks, last_reset, now=1060.0)
        res3 = stt.detect_latched_ema_channels(channels, streaks, last_reset, now=1090.0)
        self.assertEqual(res3, [])

        # Past cooldown (1030 + 180 = 1210) it can fire again
        stt.detect_latched_ema_channels(channels, streaks, last_reset, now=1200.0)
        res4 = stt.detect_latched_ema_channels(channels, streaks, last_reset, now=1230.0)
        self.assertEqual(res4, [latched_f])

    def test_update_managed_notches_hysteresis_cap_and_manual_preservation(self) -> None:
        channels = _make_clean_channels()
        slow_occ = np.zeros(stt.N_CHAN_TOTAL, dtype=np.float64)
        manual_bands = [[362.0, 378.0], [413.8, 414.2]]
        bad_streaks: dict[float, int] = {}
        clean_streaks: dict[float, int] = {}
        managed: set[float] = set()

        # Make MAX_AUTO_NOTCH_CHANNELS + 4 severe persistent RFI channels to verify cap & clean margin
        n_bad = stt.MAX_AUTO_NOTCH_CHANNELS + 4
        bad_indices = list(range(500, 500 + n_bad))
        for idx in bad_indices:
            channels[idx].update(
                {
                    "flagged_chunk_fraction": 0.98,
                    "flag_bits": stt.RFI_SK_CW_TONE | stt.RFI_FREQ_MAD,
                    "mean_sk": 0.15,
                    "freq_mad_z": 12.0,
                }
            )
            slow_occ[idx] = 0.95

        combined = list(manual_bands)
        for _ in range(4):
            combined, changed, _ = stt.update_managed_notches(
                channels, slow_occ, combined, bad_streaks, clean_streaks, managed
            )
            self.assertFalse(changed)

        # 5th consecutive bad cycle promotes up to max_managed=24
        combined, changed, info = stt.update_managed_notches(
            channels, slow_occ, combined, bad_streaks, clean_streaks, managed
        )
        self.assertTrue(changed)
        self.assertEqual(len(info["promoted"]), stt.MAX_AUTO_NOTCH_CHANNELS)
        self.assertEqual(len(managed), stt.MAX_AUTO_NOTCH_CHANNELS)
        # Manual bands must remain untouched
        self.assertIn([362.0, 378.0], combined)
        self.assertIn([413.8, 414.2], combined)

        # Clean one promoted channel for 5 cycles -> demotes while preserving others
        demote_idx = bad_indices[0]
        demote_f = channels[demote_idx]["freq_mhz"]
        self.assertIn(demote_f, managed)
        channels[demote_idx].update(
            {
                "flagged_chunk_fraction": 0.0,
                "flag_bits": 0,
                "mean_sk": 1.0,
                "freq_mad_z": 0.1,
            }
        )
        slow_occ[demote_idx] = 0.0
        for _ in range(4):
            combined, changed, _ = stt.update_managed_notches(
                channels, slow_occ, combined, bad_streaks, clean_streaks, managed
            )
            self.assertFalse(changed)
        combined, changed, info = stt.update_managed_notches(
            channels, slow_occ, combined, bad_streaks, clean_streaks, managed
        )
        self.assertTrue(changed)
        self.assertEqual(info["demoted"], [demote_f])
        self.assertNotIn(demote_f, managed)
        self.assertIn([362.0, 378.0], combined)
        self.assertIn([413.8, 414.2], combined)

    def test_waterfall_tailer_rollover_and_symlink_reset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_a = Path(tmp) / "run_a"
            run_b = Path(tmp) / "run_b"
            run_a.mkdir()
            run_b.mkdir()
            cur = Path(tmp) / "current"
            cur.symlink_to(run_a)

            tailer = stt.WaterfallTailer(str(cur), max_frames=50)
            spec0 = np.full(stt.N_CHAN_SUBBAND, 100.0, dtype=np.float32)
            spec1 = np.full(stt.N_CHAN_SUBBAND, 120.0, dtype=np.float32)
            base_sec = 1_794_630_000

            # Write 3 frames to file _0000000.bin in run_a
            f0_0 = run_a / "beam_power_freq0_0000000.bin"
            f1_0 = run_a / "beam_power_freq1_0000000.bin"
            with open(f0_0, "wb") as fp0, open(f1_0, "wb") as fp1:
                for k in range(3):
                    seq = 1000 + k * stt.EXPECTED_ACCUM_DSEQ
                    fp0.write(_make_fake_bin_frame(seq, base_sec + k, 0, spec0))
                    fp1.write(_make_fake_bin_frame(seq, base_sec + k, 0, spec1))

            self.assertEqual(tailer.poll(), 3)
            self.assertEqual(tailer.total_aligned, 3)
            self.assertEqual(tailer.seq_gaps, 0)

            # Roll over to _0000001.bin and append an extra frame in _0000000.bin with a sequence gap
            f0_1 = run_a / "beam_power_freq0_0000001.bin"
            f1_1 = run_a / "beam_power_freq1_0000001.bin"
            seq3 = 1000 + 3 * stt.EXPECTED_ACCUM_DSEQ
            seq_gap = seq3 + 2 * stt.EXPECTED_ACCUM_DSEQ
            with open(f0_0, "ab") as fp0, open(f1_0, "ab") as fp1:
                fp0.write(_make_fake_bin_frame(seq3, base_sec + 3, 0, spec0))
                fp1.write(_make_fake_bin_frame(seq3, base_sec + 3, 0, spec1))
            with open(f0_1, "wb") as fp0, open(f1_1, "wb") as fp1:
                fp0.write(_make_fake_bin_frame(seq_gap, base_sec + 5, 0, spec0))
                fp1.write(_make_fake_bin_frame(seq_gap, base_sec + 5, 0, spec1))

            self.assertEqual(tailer.poll(), 2)
            self.assertEqual(tailer.total_aligned, 5)
            self.assertEqual(tailer.seq_gaps, 1)
            qa = tailer.compute_qa()
            self.assertEqual(qa["total_frames"], 5)
            self.assertAlmostEqual(qa["sb0_median_power"], 100.0, places=2)
            self.assertAlmostEqual(qa["sb1_median_power"], 120.0, places=2)

            # Repoint symlink current -> run_b; tailer must reset cleanly and read run_b
            cur.unlink()
            cur.symlink_to(run_b)
            with open(run_b / "beam_power_freq0_0000000.bin", "wb") as fp0, open(
                run_b / "beam_power_freq1_0000000.bin", "wb"
            ) as fp1:
                for k in range(2):
                    seq = 5000 + k * stt.EXPECTED_ACCUM_DSEQ
                    fp0.write(_make_fake_bin_frame(seq, base_sec + 100 + k, 0, spec0 * 2.0))
                    fp1.write(_make_fake_bin_frame(seq, base_sec + 100 + k, 0, spec1 * 2.0))

            self.assertEqual(tailer.poll(), 2)
            self.assertEqual(tailer.total_aligned, 2)
            self.assertEqual(tailer.seq_gaps, 0)

    def test_validate_gains_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ok_path = os.path.join(tmp, "ok_gains.npy")
            bad_nan = os.path.join(tmp, "bad_nan.npy")
            bad_shape = os.path.join(tmp, "bad_shape.npy")
            bad_range = os.path.join(tmp, "bad_range.npy")

            g = np.ones((stt.N_CHAN_TOTAL, 64), dtype=np.complex64) * (0.6 + 0.2j)
            np.save(ok_path, g)
            ok, msg = stt.validate_gains_file(ok_path)
            self.assertTrue(ok, msg)

            g_nan = g.copy()
            g_nan[10, stt.DEFAULT_SIGNAL_ANTS[0]] = np.nan + 0j
            np.save(bad_nan, g_nan)
            self.assertFalse(stt.validate_gains_file(bad_nan)[0])

            np.save(bad_shape, np.ones((100, 64), dtype=np.complex64))
            self.assertFalse(stt.validate_gains_file(bad_shape)[0])

            g_rng = g.copy()
            g_rng[5, stt.DEFAULT_SIGNAL_ANTS[0]] = 0.01 + 0j
            np.save(bad_range, g_rng)
            self.assertFalse(stt.validate_gains_file(bad_range)[0])

    def test_closed_loop_daemon_step_integration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_live"
            run_dir.mkdir()
            g_path = Path(tmp) / "gains_test.npy"
            np.save(g_path, np.ones((stt.N_CHAN_TOTAL, 64), dtype=np.complex64) * 0.6)

            channels = _make_clean_channels()
            posted = []

            def fake_get(_base_url: str, path: str, timeout: float = 5.0):
                if path == "/rfi_mitigation/status":
                    return {
                        "overall_flagged_chunk_fraction": 0.02,
                        "mean_flagged_percent": 2.0,
                        "static_notch_bands_mhz": [],
                    }
                if path == "/rfi_mitigation/channels":
                    return {"channels": channels}
                if path == "/direct_tracker/status":
                    return {
                        "gains_file": str(g_path),
                        "gains_generation": 1,
                        "active_physical_antennas": stt.DEFAULT_SIGNAL_ANTS,
                    }
                if path == "/antenna_mask/status":
                    return {
                        "active_physical_antennas": stt.DEFAULT_SIGNAL_ANTS,
                        "dead_antennas": 0,
                        "saturated_antennas": 0,
                        "antennas": [
                            {
                                "antenna_id": a,
                                "active": True,
                                "consecutive_healthy": 100,
                                "clip_fraction": 0.0,
                            }
                            for a in stt.DEFAULT_SIGNAL_ANTS
                        ],
                    }
                if path.startswith("/gpu_profile/"):
                    return {"copy_in": [{"time": 0.001}], "copy_out": [{"time": 0.001}]}
                return {}

            def fake_post(_base_url: str, path: str, payload: dict, timeout: float = 5.0):
                posted.append((path, payload))
                return {"status": "ok"}

            with mock.patch.object(stt, "http_get_json", side_effect=fake_get), mock.patch.object(
                stt, "http_post_json", side_effect=fake_post
            ):
                daemon = stt.ClosedLoopDaemon(
                    base_url="http://127.0.0.1:12048",
                    run_dir=str(run_dir),
                )
                status = daemon.step()

            status_file = run_dir / "self_tune_status.json"
            telemetry_file = run_dir / "telemetry_log.jsonl"
            self.assertTrue(status_file.is_file())
            self.assertTrue(telemetry_file.is_file())
            json.loads(status_file.read_text(encoding="utf-8"))
            rows = stt.load_telemetry_records(str(telemetry_file))
            self.assertEqual(len(rows), 1)
            self.assertTrue(status["rfi"]["raw_channels_api"])
            self.assertEqual(status["rfi"]["rfi_gains"]["source"], "rfi_mitigation/channels")
            self.assertAlmostEqual(status["rfi"]["overall_flagged_pct"], 2.0, places=2)
            self.assertEqual(status["pipeline"]["active_antennas"], stt.DEFAULT_SIGNAL_ANTS)
            self.assertIn("environment", status)
            self.assertTrue(any(p == "/rfi_mitigation/set_gains" for p, _ in posted))

    def test_daemon_restores_managed_notches_from_status_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_live"
            run_dir.mkdir()
            status_path = run_dir / "self_tune_status.json"
            status_path.write_text('{"rfi": {"managed_notch_freqs_mhz": [424.2, 469.2]}}')
            daemon = stt.ClosedLoopDaemon(
                base_url="http://127.0.0.1:12048",
                run_dir=str(run_dir),
            )
            self.assertEqual(daemon.managed_notch_freqs, {424.2, 469.2})

    def test_detrend_dynamic_spectrum_removes_differential_subband_tilts(self) -> None:
        rng = np.random.default_rng(42)
        n_t, n_f = 40, stt.N_CHAN_TOTAL
        freqs = stt._channel_freqs_mhz(n_f)
        f0 = freqs[: stt.N_CHAN_SUBBAND]
        f1 = freqs[stt.N_CHAN_SUBBAND :]
        u0 = (f0 - np.mean(f0)) / 100.0
        u1 = (f1 - np.mean(f1)) / 100.0

        drift = np.linspace(-1.0, 1.0, n_t)
        rel = rng.normal(0.0, 0.004, size=(n_t, n_f))
        rel[:, : stt.N_CHAN_SUBBAND] += 0.03 * drift[:, None] + (-0.06 * drift[:, None]) * u0[None, :]
        rel[:, stt.N_CHAN_SUBBAND :] += -0.02 * drift[:, None] + (0.05 * drift[:, None]) * u1[None, :]
        # Inject RFI outliers and NaN blanks that robust IRLS must ignore
        rel[20, 40:44] = 0.85
        rel[:, 420:425] = np.nan

        det_scalar, diag_s = stt.detrend_dynamic_spectrum(rel, freqs, mode="scalar")
        det_lin, diag_l = stt.detrend_dynamic_spectrum(rel, freqs, mode="subband_linear")
        det_quad, diag_q = stt.detrend_dynamic_spectrum(rel, freqs, mode="subband_quad")

        self.assertLess(diag_l["detrended_rms_pct"], diag_s["detrended_rms_pct"])
        self.assertLess(diag_q["sat_2pct"], diag_s["sat_2pct"])
        # Peak tilt at t=-1 (+6 %/100MHz) -> t=+1 (-6 %/100MHz) for SB0
        self.assertAlmostEqual(diag_l["sb0_tilt_pct_per_100mhz"][-1], -6.0, delta=0.6)
        self.assertAlmostEqual(diag_l["sb1_tilt_pct_per_100mhz"][-1], 5.0, delta=0.6)
        self.assertAlmostEqual(diag_l["sb0_common_pct"][-1], 3.0, delta=0.4)
        self.assertAlmostEqual(diag_l["sb1_common_pct"][-1], -2.0, delta=0.4)
        fin = np.isfinite(det_lin)
        self.assertTrue(np.allclose(det_lin[fin] + diag_l["fitted_baseline"][fin], rel[fin], atol=1e-10))
        self.assertTrue(np.isnan(det_scalar[:, 420]).all())

    def test_simulate_improved_rfi_excision_catches_wide_burst_shoulders_only(self) -> None:
        rng = np.random.default_rng(7)
        n_t, n_f = 60, stt.N_CHAN_TOTAL
        wf = rng.normal(100.0, 1.5, size=(n_t, n_f))
        # Wide 10-channel burst (431.7-434.4 analog: ch 439..448) + +18% shoulders (437,438,449,450)
        wf[20:, 439:449] *= 3.5
        wf[20:, [437, 438, 449, 450]] *= 1.18
        # Isolated single-channel CW tone at ch=100 must NOT drag clean neighbors 99, 101
        wf[:, 100] *= 3.0

        _, mask_sh = stt.simulate_improved_rfi_excision(wf, enable_freq_shoulder=True)
        for sh in (437, 438, 449, 450):
            self.assertGreater(float(np.mean(mask_sh[20:, sh])), 0.85)
        self.assertLess(float(np.mean(mask_sh[:, 99])), 0.05)
        self.assertLess(float(np.mean(mask_sh[:, 101])), 0.05)

    def test_detect_latched_ema_guards_against_real_shoulders_and_escalates_cooldown(self) -> None:
        channels = _make_clean_channels()
        # Real +30% neighbor of a bursting channel with low |mad_z| must NOT be reset
        sh_f = channels[430]["freq_mhz"]
        channels[431].update({"flagged_chunk_fraction": 0.99, "flag_bits": stt.RFI_TIME_BURST})
        channels[430].update(
            {
                "mean_power": 130.0,
                "flagged_chunk_fraction": 0.96,
                "flag_bits": stt.RFI_TIME_BURST | stt.RFI_MORPHOLOGY_SIR,
                "mean_sk": 1.0,
                "freq_mad_z": 1.8,
            }
        )
        # Isolated channel with unflagged +20% real power excess above neighbors must NOT reset
        channels[480].update(
            {
                "mean_power": 120.0,
                "flagged_chunk_fraction": 0.95,
                "flag_bits": stt.RFI_TIME_BURST,
                "mean_sk": 1.0,
                "freq_mad_z": 2.0,
            }
        )
        # Genuine false EMA latch with clean neighbors and normal power SHOULD reset & escalate cooldown
        latch_f = channels[390]["freq_mhz"]
        channels[390].update(
            {
                "mean_power": 100.0,
                "flagged_chunk_fraction": 1.0,
                "flag_bits": stt.RFI_TIME_BURST | stt.RFI_MORPHOLOGY_SIR,
                "mean_sk": 0.98,
                "freq_mad_z": 0.3,
            }
        )

        streaks: dict[float, int] = {}
        last_reset: dict[float, float] = {}
        counts: dict[float, int] = {}
        for t in (1000.0, 1030.0):
            res = stt.detect_latched_ema_channels(channels, streaks, last_reset, t, reset_counts=counts)
        self.assertEqual(res, [latch_f])
        self.assertEqual(counts[latch_f], 1)
        self.assertNotIn(sh_f, res)

        # Second reset allowed after base 180s cooldown -> escalates reset_count to 2
        stt.detect_latched_ema_channels(channels, streaks, last_reset, 1200.0, reset_counts=counts)
        res2 = stt.detect_latched_ema_channels(channels, streaks, last_reset, 1230.0, reset_counts=counts)
        self.assertEqual(res2, [latch_f])
        self.assertEqual(counts[latch_f], 2)

        # Third reset now requires 360s cooldown, so at +200s (1430s) it stays throttled
        stt.detect_latched_ema_channels(channels, streaks, last_reset, 1400.0, reset_counts=counts)
        res3 = stt.detect_latched_ema_channels(channels, streaks, last_reset, 1430.0, reset_counts=counts)
        self.assertEqual(res3, [])
        res4 = stt.detect_latched_ema_channels(channels, streaks, last_reset, 1600.0, reset_counts=counts)
        self.assertEqual(res4, [latch_f])

    def test_update_managed_notches_70pct_core_and_shoulder_hysteresis(self) -> None:
        channels = _make_clean_channels()
        slow_occ = np.zeros(stt.N_CHAN_TOTAL, dtype=np.float64)
        # 432.0 core at 78% occ (would fail old 90% gate) + adjacent +-2 shoulder at mad_z=3.4
        core_idx = 440
        sh_idx = 442
        clean_nbr_idx = 441
        channels[core_idx].update(
            {"flagged_chunk_fraction": 0.78, "flag_bits": stt.RFI_FREQ_MAD, "freq_mad_z": 9.5}
        )
        channels[sh_idx].update(
            {
                "flagged_chunk_fraction": 0.72,
                "flag_bits": stt.RFI_FREQ_SHOULDER,
                "freq_mad_z": 3.4,
            }
        )
        slow_occ[core_idx] = 0.70
        slow_occ[sh_idx] = 0.65

        bad_streaks: dict[float, int] = {}
        clean_streaks: dict[float, int] = {}
        managed: set[float] = set()
        combined: list[list[float]] = []
        for _ in range(5):
            combined, changed, info = stt.update_managed_notches(
                channels, slow_occ, combined, bad_streaks, clean_streaks, managed
            )
        self.assertTrue(changed)
        self.assertIn(channels[core_idx]["freq_mhz"], managed)
        self.assertIn(channels[sh_idx]["freq_mhz"], managed)
        self.assertNotIn(channels[clean_nbr_idx]["freq_mhz"], managed)

    def test_check_antennas_parses_rest(self) -> None:
        def fake_get(_url: str, path: str, timeout: float = 5.0):
            if path == "/antenna_mask/status":
                return {
                    "active_physical_antennas": [0, 4, 32, 36],
                    "dead_antennas": 0,
                    "saturated_antennas": 0,
                    "antennas": [{"antenna_id": 0, "active": True, "consecutive_healthy": 10, "clip_fraction": 0.0}],
                }
            return {"channels": [{"freq_mhz": 350.0, "power": 1.0}]}

        with mock.patch.object(stt, "http_get_json", side_effect=fake_get):
            res = stt.check_antennas("http://127.0.0.1:12048", verbose=False)
        self.assertEqual(res["status"]["active_physical_antennas"], [0, 4, 32, 36])
        self.assertEqual(len(res["channel_power"]["channels"]), 1)

    def test_environment_collector_rigol_prometheus_grafana_csv_and_offline_fallback(self) -> None:
        # Local ephemeral TCP server mocking Rigol DP8xx SCPI responder
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.bind(("127.0.0.1", 0))
        srv.listen(2)
        port = srv.getsockname()[1]

        def handle_rigol() -> None:
            conn, _ = srv.accept()
            with conn:
                buf = b""
                while True:
                    chunk = conn.recv(256)
                    if not chunk:
                        break
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        cmd = line.decode("ascii").strip()
                        if cmd.endswith("CH1"):
                            conn.sendall(b"12.012,0.4512,5.420\n")
                        else:
                            conn.sendall(b"5.001,0.1200,0.600\n")

        th = threading.Thread(target=handle_rigol, daemon=True)
        th.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                csv_path = Path(tmp) / "pequen_aht10.csv"
                csv_path.write_text(
                    "Series,Time,Value\n"
                    "pequen-aht10 temp,2026-11-15T00:05:00Z,18.45 °C\n"
                    "pequen-aht10 humidity,2026-11-15T00:05:00Z,61.2 %\n",
                    encoding="utf-8",
                )
                col = stt.EnvironmentTelemetryCollector(
                    rigol_host=f"127.0.0.1:{port}",
                    telemetry_csv=str(csv_path),
                    prometheus_url="http://127.0.0.1:9/unreachable",
                    include_local_hwmon=True,
                    timeout_s=0.4,
                )
                snap = col.sample()
                self.assertAlmostEqual(snap["metrics"]["lna_v"], 12.012, places=3)
                self.assertAlmostEqual(snap["metrics"]["lna_i"], 0.4512, places=4)
                self.assertAlmostEqual(snap["metrics"]["pequen_aht10_temp"], 18.45, places=2)
                self.assertTrue(snap["sources"]["prometheus"].startswith("offline:"))
                self.assertEqual(snap["sources"]["rigol_dp8xx"], "ok")
        finally:
            srv.close()

    def test_detect_gain_steps_and_correlate_telemetry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tlog = Path(tmp) / "telemetry_log.jsonl"
            env_csv = Path(tmp) / "outdoor_bias.csv"
            base_t = 1_794_680_000.0
            lines = []
            csv_rows = ["timestamp_unix,ambient_temp_c,lna_i\n"]
            for k in range(200):
                t = base_t + k * 2.0
                step = 1.6 if k >= 100 else 0.0
                sb0 = step + 0.03 * np.sin(k * 0.2)
                sb1 = 0.8 * step + 0.02 * np.cos(k * 0.2)
                temp = 15.0 + 0.01 * k + (1.2 if k >= 100 else 0.0)
                lna_i = 0.450 + (0.018 if k >= 100 else 0.0)
                lines.append(
                    json.dumps(
                        {
                            "timestamp_unix": t,
                            "sb0_common_pct": float(sb0),
                            "sb1_common_pct": float(sb1),
                        }
                    )
                )
                csv_rows.append(f"{t},{temp:.3f},{lna_i:.5f}\n")
            tlog.write_text("\n".join(lines) + "\n", encoding="utf-8")
            env_csv.write_text("".join(csv_rows), encoding="utf-8")

            report = stt.correlate_environment_telemetry(
                str(tlog), telemetry_path=str(env_csv), step_window_frames=25, min_step_pct=0.4
            )
            self.assertEqual(len(report["detected_steps_sb0"]), 1)
            self.assertAlmostEqual(report["detected_steps_sb0"][0]["step_pct"], 1.6, delta=0.15)
            corr_i = report["sensor_correlations"]["lna_i"]
            self.assertGreater(corr_i["r_level_sb0"], 0.90)
            self.assertEqual(len(corr_i["step_alignments"]), 2)
            self.assertAlmostEqual(corr_i["step_alignments"][0]["sensor_delta"], 0.018, delta=0.002)

    def test_find_active_transits_graceful_fallback(self) -> None:
        self.assertEqual(stt.find_active_transits(100.0, 200.0), [])
        self.assertEqual(stt.find_active_transits(1_794_680_000.0, 1_794_680_000.0), [])
        # When astropy is unavailable or returns empty, no exception is raised.
        with mock.patch.dict(sys.modules, {"astropy": None, "astropy.coordinates": None}):
            self.assertEqual(stt.find_active_transits(1_794_680_000.0, 1_794_690_000.0), [])

    def test_fetch_gpu_util_discovers_canonical_and_legacy_endpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            daemon = stt.ClosedLoopDaemon(base_url="http://127.0.0.1:12048", run_dir=tmp)

            # Canonical discovery via /endpoints (handles null kernel time safely).
            def fake_canonical(_base: str, path: str, timeout: float = 3.0):
                if path == "/endpoints":
                    return {"GET": ["/metrics", "/gpu_profile/gpu/gpu_0", "/gpu_profile/gpu/gpu_1"]}
                if path == "/gpu_profile/gpu/gpu_0":
                    return {
                        "copy_in": [{"time": 0.002}],
                        "copy_out": [{"time": 0.001}],
                        "kernel": [{"time": 0.003}, {"time": None}],
                    }
                if path == "/gpu_profile/gpu/gpu_1":
                    return {
                        "copy_in": [{"time": 0.004}],
                        "copy_out": [{"time": 0.002}],
                        "kernel": [{"time": 0.005}],
                    }
                raise AssertionError(f"unexpected path: {path}")

            with mock.patch.object(stt, "http_get_json", side_effect=fake_canonical):
                util = daemon._fetch_gpu_util()
            self.assertEqual(
                util,
                {
                    "gpu_0": {"copy_in_ms": 2.0, "copy_out_ms": 1.0, "kernel_ms": 3.0},
                    "gpu_1": {"copy_in_ms": 4.0, "copy_out_ms": 2.0, "kernel_ms": 5.0},
                },
            )

            # Fallback when /endpoints is unavailable and server uses legacy /stages/.
            def fake_legacy_fallback(_base: str, path: str, timeout: float = 3.0):
                if path == "/gpu_profile/stages/gpu_0":
                    return {"copy_in": [{"time": 0.001}], "copy_out": [{"time": 0.002}], "kernel": []}
                raise RuntimeError("404")

            with mock.patch.object(stt, "http_get_json", side_effect=fake_legacy_fallback):
                util_fb = daemon._fetch_gpu_util()
            self.assertEqual(
                util_fb,
                {"gpu_0": {"copy_in_ms": 1.0, "copy_out_ms": 2.0, "kernel_ms": 0.0}},
            )


if __name__ == "__main__":
    unittest.main()
