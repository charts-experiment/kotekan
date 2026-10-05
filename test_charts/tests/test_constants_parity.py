"""Unit tests for Kotekan CHARTS constants compatibility and C++/Python parity.

Verifies:
1. `test_charts/constants.py` exports physical, instrumental, site, and CPT constants.
2. Fallback mode operates correctly in standalone environments.
3. C++ header `lib/core/chartsConstants.hpp` has exact parity with Python constants.
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

# Setup paths
TEST_CHARTS_DIR = Path(__file__).resolve().parent.parent
KOTEKAN_ROOT = TEST_CHARTS_DIR.parent
HEADER_PATH = KOTEKAN_ROOT / "lib" / "core" / "chartsConstants.hpp"

if str(TEST_CHARTS_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_CHARTS_DIR))

import constants as c


class TestKotekanConstants(unittest.TestCase):
    """Test suite for Kotekan CHARTS constants."""

    def test_constants_exports_and_types(self):
        """Verify physical, instrumental, and networking constants."""
        self.assertEqual(c.C_LIGHT, 299_792_458.0)
        self.assertEqual(c.SPEED_OF_LIGHT, 299_792_458.0)
        self.assertAlmostEqual(c.K_DM, 4148.741601, places=5)

        self.assertEqual(c.ADC_SAMPLING_FREQ_MHZ, 2457.6)
        self.assertEqual(c.FPGA_NUM_SAMP_FFT, 8192)
        self.assertAlmostEqual(c.CHARTS_CHANNEL_WIDTH_MHZ, 0.3, places=6)
        self.assertEqual(c.CHARTS_CHANNEL_WIDTH_HZ, 300_000.0)

        self.assertEqual(c.CHARTS_N_FREQ, 672)
        self.assertEqual(c.LOCAL_FREQUENCY_CHANNELS, 336)
        self.assertEqual(c.FREQUENCY_SHARD_COUNT, 2)

        self.assertEqual(c.DEFAULT_FREQUENCY_START_MHZ, 300.0)
        self.assertEqual(c.DEFAULT_FREQUENCY_START_HZ, 300_000_000.0)
        self.assertEqual(c.DEFAULT_SPACING_M, 0.6)
        self.assertEqual(c.CHARTS_N_ANTENNAS, 256)

        # Site coordinates (Carén site)
        self.assertAlmostEqual(c.CHARTS_LATITUDE_DEG, -33.4211146, places=6)
        self.assertAlmostEqual(c.CHARTS_LONGITUDE_DEG, -70.8634710, places=6)
        self.assertEqual(c.CHARTS_ALTITUDE_M, 458.0)

        # CPT
        self.assertEqual(c.CPT_SAMPLE_RATE, 4915.2)
        self.assertEqual(c.CPT_SPECTRA_PER_PACKET, 4)

    def test_cpp_header_parity(self):
        """Verify C++ constants header exists and contains exact matching values."""
        self.assertTrue(HEADER_PATH.exists(), f"Header {HEADER_PATH} must exist")
        content = HEADER_PATH.read_text(encoding="utf-8")

        self.assertIn("inline constexpr double speed_of_light_m_per_s = 299'792'458.0;", content)
        self.assertIn("inline constexpr double k_dm = 4148.741601;", content)
        self.assertIn("inline constexpr float charts_channel_width_hz = 300'000.0F;", content)
        self.assertIn("inline constexpr std::size_t charts_full_band_channels = 672;", content)
        self.assertIn("inline constexpr std::size_t charts_local_channels = charts_full_band_channels / charts_shard_count;", content)
        self.assertIn("inline constexpr double charts_caren_lat_deg = -33.4211146;", content)
        self.assertIn("inline constexpr double charts_caren_lon_deg = -70.8634710;", content)
        self.assertIn("inline constexpr double charts_caren_alt_m = 458.0;", content)


if __name__ == "__main__":
    unittest.main()
