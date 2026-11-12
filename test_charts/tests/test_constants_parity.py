"""Unit tests for Kotekan CHARTS constants compatibility and C++/Python parity.

Verifies:
1. `test_charts/constants.py` exports physical, instrumental, site, and CPT constants.
2. Fallback mode operates correctly in standalone environments.
3. C++ header `lib/core/chartsConstants.hpp` has exact parity with Python constants.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import re
import sys
import unittest
from unittest import mock

# Setup paths
TEST_CHARTS_DIR = Path(__file__).resolve().parent.parent
KOTEKAN_ROOT = TEST_CHARTS_DIR.parent
HEADER_PATH = KOTEKAN_ROOT / "lib" / "core" / "chartsConstants.hpp"
SIM_CONSTANTS_PATH = TEST_CHARTS_DIR / "sim" / "constants.py"

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

        # Antenna array reference coordinates (Carén array center)
        self.assertAlmostEqual(c.CHARTS_ANTENNA_REF_LATITUDE_DEG, -33.4212167, places=6)
        self.assertAlmostEqual(c.CHARTS_ANTENNA_REF_LONGITUDE_DEG, -70.8637972, places=6)
        self.assertEqual(c.CHARTS_ANTENNA_REF_ALTITUDE_M, 458.0)
        self.assertEqual(
            c.CHARTS_ANTENNA_REF_LOCATION,
            (-33.4212167, -70.8637972, 458.0),
        )

        # Calán Observatory
        self.assertAlmostEqual(c.CALAN_LATITUDE_DEG, -33.39628, places=5)
        self.assertAlmostEqual(c.CALAN_LONGITUDE_DEG, -70.536695, places=5)
        self.assertEqual(c.CALAN_ALTITUDE_M, 860.0)

        # CPT
        self.assertEqual(c.CPT_SAMPLE_RATE, 4915.2)
        self.assertEqual(c.CPT_SPECTRA_PER_PACKET, 4)

    def test_cpp_header_parity(self):
        """Verify C++ constants header exists and contains exact matching values."""
        self.assertTrue(HEADER_PATH.exists(), f"Header {HEADER_PATH} must exist")
        content = HEADER_PATH.read_text(encoding="utf-8")

        self.assertIn(
            "inline constexpr double speed_of_light_m_per_s = 299'792'458.0;", content
        )
        self.assertIn("inline constexpr double k_dm = 4148.741601;", content)
        self.assertIn(
            "inline constexpr float charts_channel_width_hz = 300'000.0F;", content
        )
        self.assertIn(
            "inline constexpr std::size_t charts_full_band_channels = 672;", content
        )
        self.assertRegex(
            content,
            r"inline constexpr std::size_t charts_local_channels\s*=\s*charts_full_band_channels\s*/\s*charts_shard_count;",
        )
        self.assertIn(
            "inline constexpr double charts_caren_lat_deg = -33.4211146;", content
        )
        self.assertIn(
            "inline constexpr double charts_caren_lon_deg = -70.8634710;", content
        )
        self.assertIn("inline constexpr double charts_caren_alt_m = 458.0;", content)

        # Antenna array reference
        self.assertIn(
            "inline constexpr double charts_antenna_ref_lat_deg = -33.4212167;", content
        )
        self.assertIn(
            "inline constexpr double charts_antenna_ref_lon_deg = -70.8637972;", content
        )
        self.assertIn(
            "inline constexpr double charts_antenna_ref_alt_m = 458.0;", content
        )

        # Calán site
        self.assertIn(
            "inline constexpr double charts_calan_lat_deg = -33.39628;", content
        )
        self.assertIn(
            "inline constexpr double charts_calan_lon_deg = -70.536695;", content
        )
        self.assertIn("inline constexpr double charts_calan_alt_m = 860.0;", content)

    def test_charts_constants_cross_repo_drift(self):
        """Verify parity against central charts_constants package when available."""
        try:
            import charts_constants
        except ImportError:
            self.skipTest("charts_constants package not installed")

        # Compare Python constants with central package
        self.assertEqual(c.C_LIGHT, float(charts_constants.C_LIGHT))
        self.assertEqual(c.CHARTS_N_FREQ, int(charts_constants.CHARTS_N_FREQ))
        self.assertEqual(c.CHARTS_N_ANTENNAS, int(charts_constants.CHARTS_N_ANTENNAS))
        self.assertEqual(
            c.DEFAULT_SPACING_M, float(charts_constants.CHARTS_ANTENNA_SPACING_M)
        )
        self.assertEqual(c.CHARTS_CHANNEL_WIDTH_HZ, float(charts_constants.FPGA_FREQUENCY_HZ))
        self.assertAlmostEqual(c.FPGA_TIME_RESOLUTION_S, float(charts_constants.DELTA_TIME), places=15)
        self.assertEqual(c.DEFAULT_FREQUENCY_START_MHZ, float(charts_constants.CPT_FREQ_0_BAND1))
        self.assertEqual(c.UPCHANNELIZED_FACTOR, int(charts_constants.UPCHANNELIZED_FACTOR))
        self.assertEqual(c.UPCHANNELIZED_N_FREQ, int(charts_constants.UPCHANNELIZED_N_FREQ))
        for external_key in (
            "FPGA_FREQUENCY_HZ",
            "DELTA_TIME",
            "CPT_NUM_BANDS",
            "UPCHANNELIZED_FACTOR",
            "UPCHANNELIZED_N_FREQ",
            "CPT_FREQ_0_BAND1",
            "CHARTS_ANTENNA_SPACING_M",
        ):
            self.assertTrue(
                hasattr(charts_constants, external_key),
                f"charts_constants missing expected attribute {external_key}",
            )

        caren_lat, caren_lon, caren_alt = charts_constants.PRESET_LOCATIONS["Caren"]
        self.assertAlmostEqual(c.CHARTS_LATITUDE_DEG, float(caren_lat), places=6)
        self.assertAlmostEqual(c.CHARTS_LONGITUDE_DEG, float(caren_lon), places=6)
        self.assertEqual(c.CHARTS_ALTITUDE_M, float(caren_alt))

        calan_lat, calan_lon, calan_alt = charts_constants.PRESET_LOCATIONS["Calan"]
        self.assertAlmostEqual(c.CALAN_LATITUDE_DEG, float(calan_lat), places=5)
        self.assertAlmostEqual(c.CALAN_LONGITUDE_DEG, float(calan_lon), places=5)
        self.assertEqual(c.CALAN_ALTITUDE_M, float(calan_alt))

        ref_lat, ref_lon, ref_alt = charts_constants.CHARTS_ANTENNA_REF_LOCATION
        self.assertAlmostEqual(c.CHARTS_ANTENNA_REF_LATITUDE_DEG, float(ref_lat), places=6)
        self.assertAlmostEqual(c.CHARTS_ANTENNA_REF_LONGITUDE_DEG, float(ref_lon), places=6)
        self.assertEqual(c.CHARTS_ANTENNA_REF_ALTITUDE_M, float(ref_alt))

        # CPT constants
        self.assertEqual(c.CPT_SAMPLE_RATE, float(charts_constants.CPT_SAMPLE_RATE))
        self.assertEqual(
            c.CPT_SPECTRA_PER_PACKET, int(charts_constants.CPT_SPECTRA_PER_PACKET)
        )

        # Verify C++ header numerical values against central charts_constants
        content = HEADER_PATH.read_text(encoding="utf-8")

        def _extract_cpp_val(name: str) -> float:
            m = re.search(
                rf"inline\s+constexpr\s+[\w:]+\s+{name}\s*=\s*([+-]?[0-9][0-9'._eE+-]*)[fF]?\s*;",
                content,
            )
            self.assertIsNotNone(m, f"C++ constant {name} not found in {HEADER_PATH}")
            return float(m.group(1).replace("'", ""))

        self.assertEqual(_extract_cpp_val("speed_of_light_m_per_s"), float(charts_constants.C_LIGHT))
        self.assertEqual(_extract_cpp_val("charts_full_band_channels"), int(charts_constants.CHARTS_N_FREQ))
        self.assertEqual(_extract_cpp_val("charts_total_antennas"), int(charts_constants.CHARTS_N_ANTENNAS))
        self.assertAlmostEqual(
            _extract_cpp_val("charts_default_spacing_m"),
            float(charts_constants.CHARTS_ANTENNA_SPACING_M),
            places=6,
        )
        self.assertAlmostEqual(_extract_cpp_val("charts_caren_lat_deg"), float(caren_lat), places=6)
        self.assertAlmostEqual(_extract_cpp_val("charts_caren_lon_deg"), float(caren_lon), places=6)
        self.assertEqual(_extract_cpp_val("charts_caren_alt_m"), float(caren_alt))
        self.assertAlmostEqual(_extract_cpp_val("charts_antenna_ref_lat_deg"), float(ref_lat), places=6)
        self.assertAlmostEqual(_extract_cpp_val("charts_antenna_ref_lon_deg"), float(ref_lon), places=6)
        self.assertEqual(_extract_cpp_val("charts_antenna_ref_alt_m"), float(ref_alt))
        self.assertAlmostEqual(_extract_cpp_val("charts_calan_lat_deg"), float(calan_lat), places=5)
        self.assertAlmostEqual(_extract_cpp_val("charts_calan_lon_deg"), float(calan_lon), places=5)
        self.assertEqual(_extract_cpp_val("charts_calan_alt_m"), float(calan_alt))
        self.assertEqual(_extract_cpp_val("cpt_sample_rate_mhz"), float(charts_constants.CPT_SAMPLE_RATE))
        self.assertEqual(_extract_cpp_val("cpt_spectra_per_packet"), int(charts_constants.CPT_SPECTRA_PER_PACKET))

    def test_fallback_mode_parity(self):
        """Force embedded fallback mode and verify fallback values match active constants."""
        spec = importlib.util.spec_from_file_location(
            "_sim_constants_fallback_test", SIM_CONSTANTS_PATH
        )
        fallback_mod = importlib.util.module_from_spec(spec)
        with mock.patch("importlib.import_module", side_effect=ImportError("blocked")):
            spec.loader.exec_module(fallback_mod)

        self.assertTrue(fallback_mod.USING_FALLBACK_CONSTANTS)
        self.assertFalse(fallback_mod.USING_CHARTS_CONSTANTS_PACKAGE)
        self.assertEqual(fallback_mod.CONSTANTS_SOURCE, "embedded_fallback")

        for attr in (
            "C_LIGHT",
            "K_DM",
            "ADC_SAMPLING_FREQ_MHZ",
            "FPGA_NUM_SAMP_FFT",
            "CHARTS_CHANNEL_WIDTH_MHZ",
            "CHARTS_CHANNEL_WIDTH_HZ",
            "CHARTS_N_FREQ",
            "LOCAL_FREQUENCY_CHANNELS",
            "FREQUENCY_SHARD_COUNT",
            "UPCHANNELIZED_FACTOR",
            "UPCHANNELIZED_N_FREQ",
            "DEFAULT_FREQUENCY_START_MHZ",
            "DEFAULT_SPACING_M",
            "CHARTS_N_ANTENNAS",
            "CHARTS_LATITUDE_DEG",
            "CHARTS_LONGITUDE_DEG",
            "CHARTS_ALTITUDE_M",
            "CHARTS_ANTENNA_REF_LATITUDE_DEG",
            "CHARTS_ANTENNA_REF_LONGITUDE_DEG",
            "CHARTS_ANTENNA_REF_ALTITUDE_M",
            "CHARTS_ANTENNA_REF_LOCATION",
            "CALAN_LATITUDE_DEG",
            "CALAN_LONGITUDE_DEG",
            "CALAN_ALTITUDE_M",
            "CPT_SAMPLE_RATE",
            "CPT_SPECTRA_PER_PACKET",
        ):
            with self.subTest(attr=attr):
                val_fb = getattr(fallback_mod, attr)
                val_active = getattr(c, attr)
                if isinstance(val_fb, float):
                    self.assertAlmostEqual(val_fb, val_active, places=6)
                else:
                    self.assertEqual(val_fb, val_active)


if __name__ == "__main__":
    unittest.main()
