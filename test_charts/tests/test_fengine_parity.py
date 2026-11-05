"""Unit test suite for CHARTS F-Engine reference simulation (sim/fengine.py).

Validates parity with Julia RadioTelescopeFEngine.jl:
  - Sinc-Hanning window (eq. 11, Erik convention) symmetry, bounds, and taper.
  - PFB channelization (4-tap, rfft, every-ntaps bin pick) tone recovery & leakage.
  - 4-bit quantizer clamp [-7, +7] and int4x2 packing / unpacking.
  - Quantization SNR loss calculation (< 0.5 dB AGENTS.md §3 equivalence bound).
  - Operating point optimization (~2.7 LSB per component).
  - End-to-end simulate_fengine_frame execution.
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

from sim.constants import DIGITIZER_NOMINAL_SIGMA_LSB
from sim.fengine import (
    FEngineConfig,
    ToneSource,
    measure_quantization_snr_loss,
    optimal_noise_sigma_lsb,
    pfb_channelize,
    quantize_int4x2,
    sinc_hanning_window,
    simulate_fengine_frame,
    synth_adc_stream,
)


class TestFEngineParity(unittest.TestCase):
    """Test suite for sim/fengine.py Julia-parity reference implementations."""

    def test_sinc_hanning_window(self):
        """sinc-Hanning window should be symmetric, peaked at center, and properly bounded."""
        ntaps = 4
        nsamples = 1024
        win = sinc_hanning_window(ntaps, nsamples)

        self.assertEqual(win.shape, (ntaps * nsamples,))
        self.assertAlmostEqual(float(np.max(win)), 1.0, places=3)
        self.assertGreaterEqual(float(np.min(win)), -0.1)

        # Symmetry: w[s] == w[N-1-s]
        np.testing.assert_allclose(win, win[::-1], atol=1e-12)

        # Edges should taper towards zero
        self.assertLess(abs(win[0]), 0.05)
        self.assertLess(abs(win[-1]), 0.05)

    def test_pfb_channelize_tone_recovery(self):
        """PFB channelizer should recover an injected monochromatic tone at the exact channel."""
        cfg = FEngineConfig(nsamples=512, ntaps=4, freq_ids=np.arange(10, 30))
        target_ch_idx = 7
        tone_freq = float(cfg.freqs_hz[target_ch_idx])

        n_samples = 4096
        t = np.arange(n_samples) * cfg.adc_dt_s
        adc_tone = np.sin(2.0 * math.pi * tone_freq * t)

        channels = pfb_channelize(adc_tone, config=cfg)
        self.assertEqual(channels.ndim, 2)
        self.assertEqual(channels.shape[1], len(cfg.freq_ids))

        # Channel power spectrum
        ch_power = np.mean(np.abs(channels) ** 2, axis=0)
        peak_idx = int(np.argmax(ch_power))
        self.assertEqual(peak_idx, target_ch_idx)

        # Leakage suppression: adjacent channels should be down significantly
        peak_pwr = ch_power[peak_idx]
        adj_pwr = max(ch_power[peak_idx - 1], ch_power[peak_idx + 1])
        leakage_ratio = adj_pwr / peak_pwr
        self.assertLess(
            leakage_ratio,
            0.01,
            f"Adjacent channel leakage ratio too high: {leakage_ratio}",
        )

    def test_pfb_channelize_multi_antenna(self):
        """pfb_channelize must support multi-antenna 2D input (n_samples, n_ant)."""
        cfg = FEngineConfig(nsamples=512, ntaps=4, freq_ids=np.arange(10, 20))
        n_samples = 4096
        n_ant = 4
        adc_multi = np.random.default_rng(42).standard_normal((n_samples, n_ant))

        channels = pfb_channelize(adc_multi, config=cfg)
        self.assertEqual(channels.shape, (channels.shape[0], len(cfg.freq_ids), n_ant))

    def test_pfb_channelize_error_on_short_stream(self):
        """pfb_channelize must raise ValueError when stream is shorter than ntaps*nsamples."""
        cfg = FEngineConfig(nsamples=1024, ntaps=4)
        short_adc = np.zeros(2048)
        with self.assertRaises(ValueError):
            pfb_channelize(short_adc, config=cfg)

    def test_quantize_int4x2_roundtrip_and_clamping(self):
        """quantize_int4x2 must clamp to [-7, +7] and pack Re/Im in 4-bit nibbles."""
        # Test values including beyond bounds
        vals = np.array(
            [
                0.0 + 0.0j,
                5.0 - 5.0j,
                7.0 + 7.0j,
                12.0 - 15.0j,  # Should clamp to 7 - 7j
                -10.0 + 20.0j,  # Should clamp to -7 + 7j
            ],
            dtype=np.complex64,
        )

        packed, quantized = quantize_int4x2(vals, scale=1.0)

        # Clamping check
        self.assertTrue(np.all(np.real(quantized) >= -7))
        self.assertTrue(np.all(np.real(quantized) <= 7))
        self.assertTrue(np.all(np.imag(quantized) >= -7))
        self.assertTrue(np.all(np.imag(quantized) <= 7))

        # Check clamp values specifically
        self.assertEqual(quantized[3], 7.0 - 7.0j)
        self.assertEqual(quantized[4], -7.0 + 7.0j)

        # Packed uint8 check: byte = (re & 0x0F) | ((im & 0x0F) << 4)
        for i in range(len(vals)):
            byte_val = int(packed[i])
            re_nibble = byte_val & 0x0F
            im_nibble = (byte_val >> 4) & 0x0F
            # Convert 4-bit unsigned to signed
            re_signed = re_nibble if re_nibble < 8 else re_nibble - 16
            im_signed = im_nibble if im_nibble < 8 else im_nibble - 16
            self.assertEqual(re_signed, int(np.real(quantized[i])))
            self.assertEqual(im_signed, int(np.imag(quantized[i])))

    def test_measure_quantization_snr_loss(self):
        """Quantization SNR loss should be 0 dB for unquantized and < 0.5 dB for nominal sigma."""
        rng = np.random.default_rng(123)
        n = 100_000
        # Perfect match -> 0 dB loss
        x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        loss_ideal = measure_quantization_snr_loss(x, x)
        self.assertAlmostEqual(loss_ideal, 0.0, places=4)

        # Realistic Gaussian noise at nominal digitizer sigma (2.0 LSB per component)
        sigma = DIGITIZER_NOMINAL_SIGMA_LSB
        x_scaled = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * sigma
        _, q = quantize_int4x2(x_scaled, scale=1.0)
        loss_db = measure_quantization_snr_loss(x_scaled, q)
        # Must be well below the AGENTS.md §3 equivalence bound (< 0.5 dB)
        self.assertLess(
            loss_db, 0.5, f"Quantization SNR loss {loss_db} dB exceeded 0.5 dB bound"
        )
        self.assertGreater(loss_db, 0.0)

    def test_optimal_noise_sigma_lsb(self):
        """optimal_noise_sigma_lsb must find a minimum loss sigma in [2.0, 3.5] LSB."""
        opt_sigma = optimal_noise_sigma_lsb(n_trials=50_000)
        self.assertGreaterEqual(opt_sigma, 2.0)
        self.assertLessEqual(opt_sigma, 3.5)

    def test_simulate_fengine_frame_end_to_end(self):
        """simulate_fengine_frame runs full ADC -> PFB -> quantize chain cleanly."""
        cfg = FEngineConfig(nsamples=512, ntaps=4, freq_ids=np.arange(10, 26))
        pos_x = np.array([0.0, 15.0, 30.0])
        pos_y = np.array([0.0, 0.0, 15.0])
        tones = [
            ToneSource(freq_hz=float(cfg.freqs_hz[5]), amp=0.8, l=0.0, m=0.0),
        ]

        n_spectra = 16
        packed, c_float, c_quant = simulate_fengine_frame(
            tone_sources=tones,
            noise_sigma_adc=2.0,
            pos_x_m=pos_x,
            pos_y_m=pos_y,
            t0_s=0.0,
            n_spectra=n_spectra,
            config=cfg,
            return_float=True,
            rng=np.random.default_rng(99),
        )

        self.assertEqual(packed.shape, (n_spectra, len(cfg.freq_ids), len(pos_x)))
        self.assertEqual(packed.dtype, np.uint8)
        self.assertEqual(c_float.shape, packed.shape)
        self.assertEqual(c_quant.shape, packed.shape)

        loss_db = measure_quantization_snr_loss(c_float, c_quant)
        self.assertLess(loss_db, 0.5)


if __name__ == "__main__":
    unittest.main()
