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
CHARTS_CONFIG_DIR = KOTEKAN_ROOT / "charts"

if str(TEST_CHARTS_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_CHARTS_DIR))

import constants as c


class TestChartsYamlConfigs(unittest.TestCase):
    """Test suite validating production CHARTS Kotekan YAML configurations."""

    @classmethod
    def setUpClass(cls):
        cls.yaml_files = sorted(CHARTS_CONFIG_DIR.glob("*.yaml"))
        if not cls.yaml_files:
            raise FileNotFoundError(
                f"No YAML configs found in {CHARTS_CONFIG_DIR}"
            )

    def test_minimum_expected_configs_present(self):
        """Ensure all primary production configurations exist."""
        expected_stems = [
            "32antennas_3beams_record",
            "32antennas_corr",
            "32antennas_direct_tracker",
            "32antennas_direct_tracker_optimal",
            "32antennas_tracker",
            "32antennas_tracker_record",
            "64antennas_direct_tracker",
            "64antennas_direct_tracker_bislc",
            "64antennas_direct_tracker_bislc_transient",
            "64antennas_direct_tracker_record",
        ]
        found_stems = {p.stem for p in self.yaml_files}
        for stem in expected_stems:
            self.assertIn(
                stem,
                found_stems,
                f"Expected production YAML '{stem}.yaml' not found in {CHARTS_CONFIG_DIR}",
            )

    def test_yaml_syntax_and_structure(self):
        """Validate YAML syntax and top-level type definition."""
        for cfg_path in self.yaml_files:
            with self.subTest(config=cfg_path.name):
                with open(cfg_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                self.assertIsInstance(
                    data, dict, f"{cfg_path.name} must parse to a dictionary"
                )
                self.assertEqual(
                    data.get("type"),
                    "config",
                    f"{cfg_path.name} must declare type: config",
                )

    def test_frequency_channel_invariants(self):
        """Verify frequency channel counts match 672 coarse channels or 336 per shard."""
        for cfg_path in self.yaml_files:
            with self.subTest(config=cfg_path.name):
                with open(cfg_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)

                if "num_local_freq" in data:
                    num_lf = int(data["num_local_freq"])
                    # Determine shard count from network capture buffers (1 for monolithic, 2 for dual-NIC)
                    capture_bufs = [
                        k for k in data if k.startswith("network_capture_buf") and isinstance(data[k], dict)
                    ]
                    shard_count = max(len(capture_bufs), 1)
                    self.assertEqual(
                        num_lf * shard_count,
                        c.CHARTS_N_FREQ,
                        f"{cfg_path.name}: num_local_freq ({num_lf}) * shards ({shard_count}) must equal {c.CHARTS_N_FREQ}",
                    )

                    if "local_subbands_per_packet" in data and "n_channels_per_packet" in data:
                        expected = (
                            data["local_subbands_per_packet"]
                            * data["n_channels_per_packet"]
                        )
                        self.assertEqual(
                            num_lf,
                            expected,
                            f"{cfg_path.name}: num_local_freq must equal local_subbands_per_packet * n_channels_per_packet",
                        )

    def test_timing_and_sampling_invariants(self):
        """Verify sampling period conforms to 10/3 us RFSoC PFB cadence."""
        for cfg_path in self.yaml_files:
            with self.subTest(config=cfg_path.name):
                with open(cfg_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)

                if "sample_period_s" in data:
                    self.assertAlmostEqual(
                        float(data["sample_period_s"]),
                        c.FPGA_TIME_RESOLUTION_S,
                        places=9,
                        msg=f"{cfg_path.name}: sample_period_s must match FPGA_TIME_RESOLUTION_S ({c.FPGA_TIME_RESOLUTION_S})",
                    )

    def test_frame_batch_alignment_invariants(self):
        """Verify samples_per_data_set is divisible by CUDA warp size (32) and chunk alignment (256)."""
        for cfg_path in self.yaml_files:
            with self.subTest(config=cfg_path.name):
                with open(cfg_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)

                if "samples_per_data_set" in data:
                    nsamp = int(data["samples_per_data_set"])
                    self.assertEqual(
                        nsamp % 32,
                        0,
                        f"{cfg_path.name}: samples_per_data_set ({nsamp}) must be divisible by 32",
                    )
                    self.assertEqual(
                        nsamp % 256,
                        0,
                        f"{cfg_path.name}: samples_per_data_set ({nsamp}) must be divisible by 256",
                    )

    def test_array_geometry_invariants(self):
        """Verify physical spacing and antenna count constraints."""
        for cfg_path in self.yaml_files:
            with self.subTest(config=cfg_path.name):
                with open(cfg_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)

                if "spacing_m" in data:
                    self.assertAlmostEqual(
                        float(data["spacing_m"]),
                        c.DEFAULT_SPACING_M,
                        places=6,
                        msg=f"{cfg_path.name}: spacing_m must match DEFAULT_SPACING_M ({c.DEFAULT_SPACING_M})",
                    )

                if "num_elements" in data:
                    num_elem = int(data["num_elements"])
                    self.assertLessEqual(
                        num_elem,
                        c.CHARTS_N_ANTENNAS,
                        f"{cfg_path.name}: num_elements ({num_elem}) exceeds max array capacity ({c.CHARTS_N_ANTENNAS})",
                    )
                    self.assertIn(
                        num_elem,
                        (32, 64),
                        f"{cfg_path.name}: num_elements ({num_elem}) expected to be 32 or 64",
                    )


if __name__ == "__main__":
    unittest.main()
