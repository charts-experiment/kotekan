"""Unit and functional test for Kotekan Live Tracker Control CLI and REST endpoints."""

import os
import shutil
import subprocess
import sys
import time
import unittest
from pathlib import Path

TEST_CHARTS_DIR = Path(__file__).resolve().parent.parent
KOTEKAN_ROOT = TEST_CHARTS_DIR.parent
KOTEKAN_BIN = KOTEKAN_ROOT / "build" / "kotekan" / "kotekan"

if str(TEST_CHARTS_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_CHARTS_DIR))

from kotekan_tracker_control import KotekanTrackerClient, render_ascii_skymap


class TestKotekanLiveControl(unittest.TestCase):
    """Test suite for Kotekan Live Tracker Control REST client & CLI."""

    @classmethod
    def setUpClass(cls):
        """Start a persistent Kotekan instance with testDataGen for REST API tests."""
        cls.work_dir = Path("/tmp/test_kotekan_live_control")
        cls.work_dir.mkdir(parents=True, exist_ok=True)
        cls.config_path = cls.work_dir / "live_control_test.yaml"
        cls.port = 12055

        config_yaml = f"""type: config
log_level: info

cpu_affinity: [0, 1]

num_elements: 64
num_local_freq: 16
samples_per_data_set: 3200
integration_spectra: 320
spacing_m: 0.6
max_beams: 4
initial_active_beams: 1
buffer_depth: 3
sizeof_complex_float: 8

main_pool:
  kotekan_metadata_pool: chordMetadata
  num_metadata_objects: 20

network_capture_buf:
  kotekan_buffer: standard
  num_frames: buffer_depth
  frame_size: samples_per_data_set * num_local_freq * num_elements
  numa_node: 0
  metadata_pool: main_pool
  zero_new_frames: true
  mlock_frames: false

host_formed_beams_buffer:
  kotekan_buffer: standard
  num_frames: buffer_depth
  frame_size: samples_per_data_set * num_local_freq * max_beams * sizeof_complex_float
  numa_node: 0
  metadata_pool: main_pool
  zero_new_frames: true
  mlock_frames: false

gen_data:
  kotekan_stage: testDataGen
  type: random_signed
  seed: 42
  reuse_random: true
  wait: true
  num_frames: 1000
  end_interrupt: false
  out_buf: network_capture_buf

gpu:
  profiling: false
  commands: &command_list
    - name: cudaInputData
      in_buf: host_voltage
      gpu_mem: voltage
    - name: cudaSyncInput
    - name: cudaBeamTrackerCommand
      gpu_mem_voltage: voltage
      gpu_mem_formed_beams: formed_beams
      num_elements: 64
      num_local_freq: 16
      samples_per_data_set: 3200
      integration_spectra: 320
      spacing_m: 0.6
      max_beams: 4
      initial_active_beams: 1
      source_l0: 0.0
      source_m0: 0.0
    - name: cudaSyncOutput
    - name: cudaOutputData
      in_buf: host_voltage
      gpu_mem: formed_beams
      out_buf: host_formed_beams
  gpu_0:
    kotekan_stage: cudaProcess
    gpu_id: 0
    commands: *command_list
    in_buffers:
      host_voltage: network_capture_buf
    out_buffers:
      host_formed_beams: host_formed_beams_buffer

sink:
  kotekan_stage: dropAllFrames
  in_buf: host_formed_beams_buffer
"""
        cls.config_path.write_text(config_yaml, encoding="utf-8")

        if not KOTEKAN_BIN.exists():
            cls.proc = None
            raise unittest.SkipTest(
                f"Kotekan binary not found at {KOTEKAN_BIN}; skipping live binary tests."
            )

        # Launch Kotekan
        cmd = [
            str(KOTEKAN_BIN),
            "-c",
            str(cls.config_path),
            "-b",
            f"127.0.0.1:{cls.port}",
        ]
        cls.proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        time.sleep(1.0)
        cls.client = KotekanTrackerClient(port=cls.port)

    @classmethod
    def tearDownClass(cls):
        """Terminate Kotekan instance."""
        if cls.proc:
            cls.proc.terminate()
            try:
                cls.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                cls.proc.kill()
        if cls.work_dir.exists():
            shutil.rmtree(cls.work_dir)

    def test_01_query_status(self):
        """Test GET /beam_tracker/status."""
        status = self.client.get_status()
        self.assertEqual(status["total_elements"], 64)
        self.assertEqual(status["max_beams_capacity"], 8)
        self.assertEqual(status["num_active_beams"], 1)
        self.assertEqual(status["active_antennas"], 64)
        self.assertEqual(status["masked_antennas"], 0)
        self.assertAlmostEqual(status["site"]["lat_deg"], -33.4211146, places=5)
        self.assertAlmostEqual(status["site"]["lon_deg"], -70.8634710, places=5)

    def test_02_steer_lm(self):
        """Test POST /beam_tracker/set_trajectory."""
        resp = self.client.steer_lm(0, l0=0.123, m0=-0.456, dl=1.5e-5, dm=2.0e-5)
        self.assertIn("Trajectory updated", resp)

        status = self.client.get_status()
        b0 = status["trajectories"][0]
        self.assertAlmostEqual(b0["l0"], 0.123, places=3)
        self.assertAlmostEqual(b0["m0"], -0.456, places=3)
        self.assertAlmostEqual(b0["dl"], 1.5e-5, places=8)
        self.assertAlmostEqual(b0["dm"], 2.0e-5, places=8)

    def test_03_steer_celestial_radec(self):
        """Test POST /beam_tracker/set_celestial_target with astrometry."""
        resp = self.client.steer_radec(
            1, ra_deg=83.633, dec_deg=22.014, lst_hours=5.575
        )
        self.assertIn("Celestial target set", resp)

        status = self.client.get_status()
        b1 = status["trajectories"][1]
        self.assertTrue(b1["celestial_target"]["is_set"])
        self.assertAlmostEqual(b1["celestial_target"]["ra_deg"], 83.633, places=3)
        self.assertAlmostEqual(b1["celestial_target"]["dec_deg"], 22.014, places=3)

    def test_04_enable_dynamic_beams(self):
        """Test dynamic beam allocation (1..8)."""
        resp = self.client.enable_beams(3)
        self.assertIn("Active beams set to 3", resp)

        status = self.client.get_status()
        self.assertEqual(status["num_active_beams"], 3)

    def test_05_mask_antenna(self):
        """Test masking and unmasking antennas for fault-tolerance."""
        resp = self.client.mask_antenna(12, enabled=False)
        self.assertIn("DEAD/MASKED", resp)

        status = self.client.get_status()
        self.assertEqual(status["masked_antennas"], 1)
        self.assertIn(12, status["bad_elements"])

        resp_unmask = self.client.mask_antenna(12, enabled=True)
        self.assertIn("ACTIVE", resp_unmask)

        status = self.client.get_status()
        self.assertEqual(status["masked_antennas"], 0)

    def test_06_render_skymap(self):
        """Test ASCII skymap rendering."""
        trajectories = [
            {"l0": 0.0, "m0": 0.0},
            {"l0": 0.5, "m0": 0.5},
        ]
        map_str = render_ascii_skymap(trajectories, 2)
        self.assertIn("+", map_str)
        self.assertIn("0", map_str)
        self.assertIn("1", map_str)


if __name__ == "__main__":
    unittest.main()
