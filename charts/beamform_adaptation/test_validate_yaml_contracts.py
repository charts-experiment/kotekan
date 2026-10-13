"""Mutation checks for the offline contract auditor."""
import copy
import os
import unittest
from pathlib import Path
import yaml
from validate_yaml_contracts import audit, duplicates, number


class AuditTests(unittest.TestCase):
    def setUp(self):
        checkout = Path(os.environ.get("KOTEKAN_SOURCE_DIR", Path(__file__).resolve().parents[2]))
        self.config = yaml.safe_load((checkout / "charts/config/64antennas_record.yaml").read_text())

    def test_valid_record_contract(self):
        result = audit(self.config)
        self.assertEqual(result["raw_contract_errors"], [])
        self.assertEqual(result["shards"][0]["raw_bytes"], 330301440)

    def test_wrong_signed_zero(self):
        self.config["zero_samples"]["zero_value"] = 136
        self.assertTrue(any("0x00" in e for e in audit(self.config)["raw_contract_errors"]))

    def test_wrong_mask_length(self):
        self.config["mask_buffer_zeros_1"]["frame_size"] = 15359
        self.assertTrue(any("mask length" in e for e in audit(self.config)["raw_contract_errors"]))

    def test_swapped_frequency_shard(self):
        self.config["dpdk"]["handlers"][1]["coarse_freq_start"] = 0
        self.assertTrue(any("partition" in e for e in audit(self.config)["raw_contract_errors"]))

    def test_corr_time_length(self):
        config = copy.deepcopy(self.config)
        config["samples_per_data_set"] = config["spectra_per_dataset"] = 153600
        for i in range(2):
            config[f"mask_buffer_zeros_{i}"]["frame_size"] = "samples_per_data_set"
        result = audit(config)
        self.assertEqual(result["raw_contract_errors"], [])
        self.assertEqual(result["shards"][0]["cadence_ms"], 512)
        self.assertEqual(result["shards"][0]["fine_output_rows_K10"], 480)

    def test_duplicate_detection(self):
        self.assertEqual(len(duplicates(yaml.compose("x: 1\nx: 2\n"))), 1)

    def test_scoped_expression(self):
        self.assertEqual(number({"x": 20}, ({"x": 3, "size": "x * 2"},), "size"), 6)

    def test_unsafe_expression_rejected(self):
        with self.assertRaises(ValueError):
            number({}, ({"x": "__import__('os')"},), "x")


if __name__ == "__main__":
    unittest.main()
