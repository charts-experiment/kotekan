#!/usr/bin/env python3
"""Unit tests for closed-loop telemetry and daemon helpers in self_tune_tracker.py."""

import contextlib
import io
import math
import os
import struct
import sys
import tempfile
import unittest
import urllib.error
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


class TestSelfTuneTrackerSefdUtilsIntegration(unittest.TestCase):
    def _write_test_npz(
        self,
        path: Path,
        *,
        format_version: int = 2,
        n_freq: int = stt.N_CHAN_TOTAL,
    ) -> tuple[np.ndarray, list[int], np.ndarray]:
        rng = np.random.default_rng(1234)
        antennas = [0, 4, 32, 36]
        gains = (
            rng.uniform(0.5, 1.5, (n_freq, len(antennas)))
            * np.exp(1j * rng.uniform(-np.pi, np.pi, (n_freq, len(antennas))))
        ).astype(np.complex128)
        gains[10, 1] = np.nan + 0j
        gains[11, 2] = np.inf + 0j

        flagged = np.zeros(n_freq, dtype=bool)
        bursty = np.zeros(n_freq, dtype=bool)
        incoherent = np.zeros(n_freq, dtype=bool)
        flagged[20:24] = True
        bursty[100:103] = True
        incoherent[400:405] = True
        expected_bad = flagged | bursty | incoherent

        np.savez(
            path,
            format_version=np.asarray(format_version),
            gains=gains,
            antennas=np.asarray(antennas, dtype=int),
            freq_mhz=(stt.SUBBAND0_START_HZ + np.arange(n_freq) * stt.CHANNEL_WIDTH_HZ) / 1e6,
            flagged_channels=flagged,
            bursty_channels=bursty,
            incoherent_channels=incoherent,
        )
        return gains, antennas, expected_bad

    def test_expand_npz_gains_matches_charts_utils_and_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            npz_path = Path(tmp) / "gains_sun_test.npz"
            gains, antennas, expected_bad = self._write_test_npz(npz_path, format_version=2)

            out_lib = Path(tmp) / "out_lib.npy"
            npy_lib, info_lib = stt.expand_npz_gains_to_npy(
                str(npz_path), out_npy_path=str(out_lib), zero_flagged=False
            )
            arr_lib = np.load(npy_lib)

            with mock.patch.object(stt, "_import_charts_utils_cal", return_value=None):
                out_fb = Path(tmp) / "out_fb.npy"
                npy_fb, info_fb = stt.expand_npz_gains_to_npy(
                    str(npz_path), out_npy_path=str(out_fb), zero_flagged=False
                )
                arr_fb = np.load(npy_fb)

            self.assertEqual(info_lib["source"], "charts_utils.cal.gains")
            self.assertEqual(info_fb["source"], "numpy_fallback")
            self.assertEqual(arr_lib.shape, (stt.N_CHAN_TOTAL, 64))
            self.assertEqual(arr_lib.dtype, np.complex64)
            self.assertEqual(arr_fb.shape, (stt.N_CHAN_TOTAL, 64))
            self.assertEqual(arr_fb.dtype, np.complex64)
            np.testing.assert_allclose(arr_lib, arr_fb, rtol=1e-6, atol=1e-6)

            uncal = [a for a in range(64) if a not in antennas]
            self.assertTrue(np.all(arr_fb[:, uncal] == 0.0 + 0.0j))
            self.assertEqual(arr_fb[10, 4], 0.0 + 0.0j)
            self.assertEqual(arr_fb[11, 32], 0.0 + 0.0j)
            np.testing.assert_allclose(arr_fb[50, antennas], gains[50].astype(np.complex64))
            self.assertTrue(np.any(arr_fb[expected_bad, :][:, antennas] != 0.0))

            # With zero_flagged=True, flagged | bursty | incoherent channels must be zeroed
            out_zf = Path(tmp) / "out_zf.npy"
            npy_zf, info_zf = stt.expand_npz_gains_to_npy(
                str(npz_path), out_npy_path=str(out_zf), zero_flagged=True
            )
            arr_zf = np.load(npy_zf)
            self.assertEqual(info_zf["zeroed_channels"], int(np.count_nonzero(expected_bad)))
            self.assertTrue(np.all(arr_zf[expected_bad, :] == 0.0 + 0.0j))
            np.testing.assert_allclose(arr_zf[~expected_bad, :], arr_lib[~expected_bad, :])

            # Fallback zero_flagged=True must also match
            with mock.patch.object(stt, "_import_charts_utils_cal", return_value=None):
                out_zf_fb = Path(tmp) / "out_zf_fb.npy"
                npy_zf_fb, _ = stt.expand_npz_gains_to_npy(
                    str(npz_path), out_npy_path=str(out_zf_fb), zero_flagged=True
                )
                np.testing.assert_allclose(np.load(npy_zf_fb), arr_zf, rtol=1e-6, atol=1e-6)

    def test_expand_npz_format1_conjugates_consistently_in_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            npz_path = Path(tmp) / "gains_sun_v1.npz"
            self._write_test_npz(npz_path, format_version=1)

            with self.assertWarns(UserWarning):
                npy_lib, _ = stt.expand_npz_gains_to_npy(
                    str(npz_path), out_npy_path=str(Path(tmp) / "v1_lib.npy")
                )
            with mock.patch.object(stt, "_import_charts_utils_cal", return_value=None):
                npy_fb, _ = stt.expand_npz_gains_to_npy(
                    str(npz_path), out_npy_path=str(Path(tmp) / "v1_fb.npy")
                )
            np.testing.assert_allclose(np.load(npy_lib), np.load(npy_fb), rtol=1e-6, atol=1e-6)

    def test_reload_calibration_gains_cli_with_npz_and_zero_flagged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            npz_path = Path(tmp) / "gains_sun_20261116.npz"
            _, _, expected_bad = self._write_test_npz(npz_path, format_version=2)
            posted = []

            def fake_post(_url: str, path: str, payload: dict, timeout: float = 5.0):
                posted.append((path, payload))
                return {"status": "ok", "gains_generation": 3}

            with mock.patch.object(stt, "http_post_json", side_effect=fake_post), mock.patch.object(
                sys,
                "argv",
                [
                    "self_tune_tracker.py",
                    "--url",
                    "http://127.0.0.1:12048",
                    "--reload-cal-gains",
                    str(npz_path),
                    "--zero-flagged",
                ],
            ):
                rc = stt.main()

            self.assertEqual(rc, 0)
            self.assertEqual(len(posted), 1)
            self.assertEqual(posted[0][0], "/direct_tracker/reload_gains")
            reloaded_file = Path(posted[0][1]["gains_file"])
            self.assertTrue(reloaded_file.is_file())
            self.assertEqual(reloaded_file.suffix, ".npy")
            arr = np.load(reloaded_file)
            self.assertEqual(arr.shape, (stt.N_CHAN_TOTAL, 64))
            self.assertEqual(arr.dtype, np.complex64)
            self.assertTrue(np.all(arr[expected_bad, :] == 0.0 + 0.0j))

    def test_quantizer_diagnostics_charts_sefd_and_fallback_parity(self) -> None:
        powers = np.array([0.0, 0.2, 1.0, 4.0, 8.0, 16.0, 100.0], dtype=np.float64)
        diag_lib = stt.quantizer_diagnostics_from_power(powers)
        with mock.patch.object(stt, "_import_charts_sefd_quantization", return_value=None):
            diag_fb = stt.quantizer_diagnostics_from_power(powers)

        self.assertEqual(diag_fb["source"], "analytical_fallback")
        np.testing.assert_allclose(diag_lib["sigma_lsb"], diag_fb["sigma_lsb"], rtol=1e-12, equal_nan=True)
        np.testing.assert_allclose(diag_lib["eta_q"], diag_fb["eta_q"], rtol=1e-12, equal_nan=True)
        np.testing.assert_allclose(diag_lib["q_rms"], diag_fb["q_rms"], rtol=1e-12, equal_nan=True)
        np.testing.assert_array_equal(diag_lib["in_optimal_window"], diag_fb["in_optimal_window"])

        # p=4.0 LSB^2 -> sigma ~1.958 LSB, eta_q ~0.9583, inside optimal window [0.933, 2.639]
        scalar = stt.quantizer_diagnostics_from_power(4.0)
        self.assertAlmostEqual(scalar["sigma_lsb"], 1.9579, places=3)
        self.assertAlmostEqual(scalar["eta_q"], 0.9583, places=3)
        self.assertTrue(scalar["in_optimal_window"])
        self.assertTrue(math.isnan(stt.quantizer_diagnostics_from_power(0.0)["sigma_lsb"]))

    def test_check_antennas_prints_and_returns_quantizer_metrics(self) -> None:
        active_phys = [0, 4, 32, 36]
        ant_rows = []
        for raw_id in range(64):
            phys_id = 63 - raw_id
            is_act = phys_id in active_phys
            ant_rows.append(
                {
                    "raw_id": raw_id,
                    "physical_id": phys_id,
                    "pol": raw_id % 2,
                    "active": is_act,
                    "status": "HEALTHY" if is_act else "DEAD",
                    "power": 4.0 if is_act else 0.0,
                    "clip_fraction": 0.0001 if is_act else 0.0,
                    "coherence": 0.25 if is_act else 0.0,
                }
            )

        mat_sb0 = np.zeros((stt.N_CHAN_SUBBAND, 64), dtype=np.float64)
        mat_sb1 = np.zeros((stt.N_CHAN_SUBBAND, 64), dtype=np.float64)
        for p in active_phys:
            r = 63 - p
            mat_sb0[:, r] = 4.0
            mat_sb1[:, r] = 8.0

        def fake_get(_url: str, path: str, timeout: float = 5.0):
            if path == "/antenna_mask/status":
                return {
                    "active_physical_antennas": active_phys,
                    "dead_antennas": 60,
                    "saturated_antennas": 0,
                    "masked_antennas": 60,
                    "antennas": ant_rows,
                }
            if path == "/antenna_mask/channel_power":
                return {
                    "num_elements": 64,
                    "subbands": [
                        {
                            "subband": 0,
                            "freq_start_hz": stt.SUBBAND0_START_HZ,
                            "num_local_freq": stt.N_CHAN_SUBBAND,
                            "has_channel_power": True,
                            "mean_active_power": [4.0] * stt.N_CHAN_SUBBAND,
                            "channel_power": mat_sb0.tolist(),
                        },
                        {
                            "subband": 1,
                            "freq_start_hz": stt.SUBBAND1_START_HZ,
                            "num_local_freq": stt.N_CHAN_SUBBAND,
                            "has_channel_power": True,
                            "mean_active_power": [8.0] * stt.N_CHAN_SUBBAND,
                            "channel_power": mat_sb1.tolist(),
                        },
                    ],
                }
            raise AssertionError(f"Unexpected GET {path}")

        buf = io.StringIO()
        with mock.patch.object(stt, "http_get_json", side_effect=fake_get), contextlib.redirect_stdout(buf):
            summary = stt.check_antennas("http://127.0.0.1:12048")

        out = buf.getvalue()
        self.assertIn("sigma=", out)
        self.assertIn("eta_q=", out)
        self.assertIn("Subband 0", out)
        self.assertIn("Subband 1", out)
        self.assertEqual(summary["active_physical_antennas"], active_phys)
        self.assertEqual(len(summary["subbands"]), 2)
        self.assertAlmostEqual(summary["subbands"][0]["sigma_lsb"], 1.9579, places=3)
        self.assertAlmostEqual(summary["subbands"][1]["sigma_lsb"], 2.7997, places=3)

        active_enriched = [a for a in summary["antennas"] if a["active"]]
        self.assertEqual(len(active_enriched), 4)
        for row in active_enriched:
            self.assertAlmostEqual(row["sigma_lsb"], 1.9579, places=3)
            self.assertAlmostEqual(row["eta_q"], 0.9583, places=3)
            self.assertTrue(row["in_optimal_window"])
            self.assertAlmostEqual(row["subband_channel_power"][0], 4.0)
            self.assertAlmostEqual(row["subband_channel_power"][1], 8.0)

        # Also verify graceful behavior when /antenna_mask/channel_power returns 404
        def fake_get_404(_url: str, path: str, timeout: float = 5.0):
            if path == "/antenna_mask/channel_power":
                raise urllib.error.HTTPError(path, 404, "Not Found", hdrs=None, fp=None)
            return fake_get(_url, path, timeout=timeout)

        with mock.patch.object(stt, "http_get_json", side_effect=fake_get_404), contextlib.redirect_stdout(io.StringIO()):
            summary_404 = stt.check_antennas("http://127.0.0.1:12048")
        self.assertEqual(summary_404["subbands"], [])
        self.assertEqual(len([a for a in summary_404["antennas"] if a["active"]]), 4)


if __name__ == "__main__":
    unittest.main()
