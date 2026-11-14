#!/usr/bin/env python3
"""Unit tests for closed-loop telemetry and daemon helpers in self_tune_tracker.py."""

import os
import struct
import sys
import tempfile
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

        # Make 26 severe persistent RFI channels (more than MAX_AUTO_NOTCH_CHANNELS=24)
        bad_indices = list(range(500, 526))
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
            self.assertTrue(status_file.is_file())
            self.assertTrue(status["rfi"]["raw_channels_api"])
            self.assertEqual(status["rfi"]["rfi_gains"]["source"], "rfi_mitigation/channels")
            self.assertAlmostEqual(status["rfi"]["overall_flagged_pct"], 2.0, places=2)
            self.assertEqual(status["pipeline"]["active_antennas"], stt.DEFAULT_SIGNAL_ANTS)
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


if __name__ == "__main__":
    unittest.main()
