"""Validation tests for CHARTS production Kotekan pipeline configurations.

Verifies that all YAML pipeline configurations in `charts/` adhere to
physical, architectural, and instrumental invariants established by
`charts_constants` and the FPGA/RFSoC firmware specifications.
"""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import yaml

TEST_CHARTS_DIR = Path(__file__).resolve().parent.parent
KOTEKAN_ROOT = TEST_CHARTS_DIR.parent
CHARTS_DIR = KOTEKAN_ROOT / "charts"
CHARTS_CONFIG_SUBDIR = CHARTS_DIR / "config"

if str(TEST_CHARTS_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_CHARTS_DIR))

import constants as c


def _iter_nested_dicts(obj):
    if isinstance(obj, dict):
        yield obj
        for val in obj.values():
            yield from _iter_nested_dicts(val)
    elif isinstance(obj, list):
        for item in obj:
            yield from _iter_nested_dicts(item)


class TestChartsYamlConfigs(unittest.TestCase):
    """Test suite validating production CHARTS Kotekan YAML configurations."""

    @classmethod
    def setUpClass(cls):
        cls.yaml_files = sorted(
            list(CHARTS_DIR.glob("*.yaml")) + list(CHARTS_CONFIG_SUBDIR.glob("*.yaml"))
        )
        if not cls.yaml_files:
            raise FileNotFoundError(
                f"No YAML configs found in {CHARTS_DIR} or {CHARTS_CONFIG_SUBDIR}"
            )

    @staticmethod
    def _cfg_label(cfg_path: Path) -> str:
        return str(cfg_path.relative_to(KOTEKAN_ROOT))

    def test_minimum_expected_configs_present(self):
        """Ensure all primary production configurations exist."""
        expected_relpaths = [
            "charts/32antennas_3beams_record.yaml",
            "charts/32antennas_corr.yaml",
            "charts/32antennas_direct_tracker.yaml",
            "charts/32antennas_direct_tracker_optimal.yaml",
            "charts/32antennas_tracker.yaml",
            "charts/32antennas_tracker_record.yaml",
            "charts/64antennas_direct_tracker.yaml",
            "charts/64antennas_direct_tracker_bislc.yaml",
            "charts/64antennas_direct_tracker_bislc_transient.yaml",
            "charts/64antennas_direct_tracker_record.yaml",
            "charts/config/64antennas_direct_tracker.yaml",
            "charts/config/64antennas_corr.yaml",
            "charts/config/64antennas_charts_sim.yaml",
        ]
        found_relpaths = {self._cfg_label(p) for p in self.yaml_files}
        for relpath in expected_relpaths:
            self.assertIn(
                relpath,
                found_relpaths,
                f"Expected CHARTS YAML '{relpath}' not found",
            )

    def test_yaml_syntax_and_structure(self):
        """Validate YAML syntax and top-level type definition."""
        for cfg_path in self.yaml_files:
            with self.subTest(config=self._cfg_label(cfg_path)):
                with open(cfg_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                self.assertIsInstance(
                    data, dict, f"{self._cfg_label(cfg_path)} must parse to a dictionary"
                )
                self.assertEqual(
                    data.get("type"),
                    "config",
                    f"{self._cfg_label(cfg_path)} must declare type: config",
                )

    def test_frequency_channel_invariants(self):
        """Verify frequency channel counts match 672 full-band or 336 single-shard."""
        for cfg_path in self.yaml_files:
            label = self._cfg_label(cfg_path)
            with self.subTest(config=label):
                with open(cfg_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)

                if "num_local_freq" in data:
                    num_lf = int(data["num_local_freq"])
                    capture_bufs = [
                        k for k in data if k.startswith("network_capture_buf") and isinstance(data[k], dict)
                    ]
                    shard_count = max(len(capture_bufs), 1)
                    if shard_count == 1:
                        # Single-stream configs either carry the full 672-ch band or 1 shard (336-ch)
                        self.assertIn(
                            num_lf,
                            (c.CHARTS_N_FREQ, c.LOCAL_FREQUENCY_CHANNELS),
                            f"{label}: single-shard num_local_freq ({num_lf}) must equal {c.CHARTS_N_FREQ} or {c.LOCAL_FREQUENCY_CHANNELS}",
                        )
                    else:
                        self.assertEqual(
                            num_lf * shard_count,
                            c.CHARTS_N_FREQ,
                            f"{label}: num_local_freq ({num_lf}) * shards ({shard_count}) must equal {c.CHARTS_N_FREQ}",
                        )

                    if "local_subbands_per_packet" in data and "n_channels_per_packet" in data:
                        expected = (
                            data["local_subbands_per_packet"]
                            * data["n_channels_per_packet"]
                        )
                        self.assertEqual(
                            num_lf,
                            expected,
                            f"{label}: num_local_freq must equal local_subbands_per_packet * n_channels_per_packet",
                        )

                for block in _iter_nested_dicts(data):
                    if "freq_step_hz" in block:
                        self.assertAlmostEqual(
                            float(block["freq_step_hz"]),
                            c.CHARTS_CHANNEL_WIDTH_HZ,
                            places=3,
                            msg=f"{label}: freq_step_hz must match {c.CHARTS_CHANNEL_WIDTH_HZ}",
                        )

    def test_timing_and_sampling_invariants(self):
        """Verify sampling period conforms to 10/3 us RFSoC PFB cadence."""
        for cfg_path in self.yaml_files:
            label = self._cfg_label(cfg_path)
            with self.subTest(config=label):
                with open(cfg_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)

                if "sample_period_s" in data:
                    self.assertAlmostEqual(
                        float(data["sample_period_s"]),
                        c.FPGA_TIME_RESOLUTION_S,
                        places=15,
                        msg=f"{label}: sample_period_s must match FPGA_TIME_RESOLUTION_S ({c.FPGA_TIME_RESOLUTION_S})",
                    )

    def test_frame_batch_alignment_invariants(self):
        """Verify samples_per_data_set is divisible by CUDA warp size (32) and chunk alignment (256)."""
        for cfg_path in self.yaml_files:
            label = self._cfg_label(cfg_path)
            with self.subTest(config=label):
                with open(cfg_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)

                if "samples_per_data_set" in data:
                    nsamp = int(data["samples_per_data_set"])
                    self.assertEqual(
                        nsamp % 32,
                        0,
                        f"{label}: samples_per_data_set ({nsamp}) must be divisible by 32",
                    )
                    self.assertEqual(
                        nsamp % 256,
                        0,
                        f"{label}: samples_per_data_set ({nsamp}) must be divisible by 256",
                    )

    def test_array_geometry_and_site_invariants(self):
        """Verify physical spacing, site coordinates, and antenna count constraints."""
        for cfg_path in self.yaml_files:
            label = self._cfg_label(cfg_path)
            with self.subTest(config=label):
                with open(cfg_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)

                for block in _iter_nested_dicts(data):
                    if "spacing_m" in block:
                        self.assertAlmostEqual(
                            float(block["spacing_m"]),
                            c.DEFAULT_SPACING_M,
                            places=6,
                            msg=f"{label}: spacing_m must match DEFAULT_SPACING_M ({c.DEFAULT_SPACING_M})",
                        )
                    if "site_lat_deg" in block:
                        self.assertAlmostEqual(
                            float(block["site_lat_deg"]),
                            c.CHARTS_LATITUDE_DEG,
                            places=6,
                            msg=f"{label}: site_lat_deg must match CHARTS_LATITUDE_DEG ({c.CHARTS_LATITUDE_DEG})",
                        )
                    if "site_lon_deg" in block:
                        self.assertAlmostEqual(
                            float(block["site_lon_deg"]),
                            c.CHARTS_LONGITUDE_DEG,
                            places=6,
                            msg=f"{label}: site_lon_deg must match CHARTS_LONGITUDE_DEG ({c.CHARTS_LONGITUDE_DEG})",
                        )

                if "num_elements" in data:
                    num_elem = int(data["num_elements"])
                    self.assertLessEqual(
                        num_elem,
                        c.CHARTS_N_ANTENNAS,
                        f"{label}: num_elements ({num_elem}) exceeds max array capacity ({c.CHARTS_N_ANTENNAS})",
                    )
                    self.assertIn(
                        num_elem,
                        (32, 64, 128, 256),
                        f"{label}: num_elements ({num_elem}) expected to be 32, 64, 128, or 256",
                    )


if __name__ == "__main__":
    unittest.main()
