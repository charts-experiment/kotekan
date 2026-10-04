"""Unit and integration test suite for the centralized CHARTS simulation framework."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

# Setup paths
_test_charts_dir = Path(__file__).resolve().parent.parent
_repo_root = _test_charts_dir.parent
if str(_test_charts_dir) not in sys.path:
    sys.path.insert(0, str(_test_charts_dir))

import h5py
import numpy as np

from sim.astro import (
    datetime_to_lst_hours,
    find_visible_sources,
    parse_observation_time,
    resolve_beam_targets,
)
from sim.benchmark import (
    calculate_direct_tracker_vram,
    run_direct_tracker_benchmark,
)
from sim.constants import (
    CHARTS_CHANNEL_WIDTH_MHZ,
    CHARTS_LATITUDE_DEG,
    CHARTS_LONGITUDE_DEG,
    DEFAULT_FREQUENCY_START_MHZ,
    DEFAULT_SPACING_M,
    K_DM,
    get_antenna_positions,
)
from sim.generator import (
    SimulatedEvent,
    build_frame_selection_schedule,
    generate_simulation_window,
    schedule_random_events,
)
from sim.noise_model import AnalogChainParams, ChartsNoiseModel
from sim.pipeline import (
    create_beam_tracker_yaml,
    create_correlator_yaml,
    parse_beam_targets,
)
from sim.presets import SimulationConfig, get_preset_config
from sim.reference import (
    create_reference_manifest,
    get_reference_window,
    list_reference_windows,
    save_as_reference_window,
)


class TestChartsSimulationSuite(unittest.TestCase):
    """Test suite for centralized CHARTS simulation and orchestration."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="test_charts_sim_"))

    def tearDown(self):
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_preset_resolution(self):
        """Test preset resolution and profile defaults."""
        quick_cfg = get_preset_config("quick", "day")
        self.assertEqual(quick_cfg.preset, "quick")
        self.assertEqual(quick_cfg.profile, "day")
        self.assertEqual(quick_cfg.antennas, 64)
        self.assertEqual(quick_cfg.num_freq, 336)
        self.assertEqual(quick_cfg.duration_s, 2.0)
        self.assertEqual(quick_cfg.utc_hour, 15.0)
        self.assertIn("Crab Pulsar", quick_cfg.beam_targets)

        night_cfg = get_preset_config("1min", "night", antennas=256)
        self.assertEqual(night_cfg.preset, "1min")
        self.assertEqual(night_cfg.profile, "night")
        self.assertEqual(night_cfg.antennas, 256)
        self.assertEqual(night_cfg.utc_hour, 3.0)
        self.assertIn("Sgr A*", night_cfg.beam_targets)

    def test_noise_model_physics(self):
        """Test physical calculations of receiver chain, spillover, and Sun."""
        params = AnalogChainParams(ground_spillover_fraction=0.02)
        model = ChartsNoiseModel(params=params)

        # Receiver temperature from Friis cascade
        # Stage 1: NF=0.3 dB (~20.7 K), Stage 2: NF=0.65 dB (~46.8 K) / Gain=19.3 dB (~85.1x)
        # T_rx should be ~ 21 - 22 K
        t_rx = model.receiver_temperature()
        self.assertGreater(t_rx, 20.0)
        self.assertLess(t_rx, 25.0)

        # Ground spillover (2% of 290 K = 5.8 K)
        t_ground = model.ground_spillover_temperature()
        self.assertAlmostEqual(t_ground, 5.8, places=1)

        # Daytime vs Nighttime Sun elevation at Observatorio Carén
        import datetime
        dt_day = datetime.datetime(2026, 3, 20, 15, 0, 0, tzinfo=datetime.timezone.utc)
        dt_night = datetime.datetime(2026, 3, 20, 3, 0, 0, tzinfo=datetime.timezone.utc)

        day_sys = model.system_temperature(400.0, utc_dt=dt_day, include_sun=True)
        night_sys = model.system_temperature(400.0, utc_dt=dt_night, include_sun=True)

        self.assertGreater(day_sys["sun_elevation_deg"], 0.0)  # Sun UP
        self.assertLess(night_sys["sun_elevation_deg"], 0.0)   # Sun DOWN
        self.assertGreater(day_sys["t_sun_pb"], 0.0)
        self.assertEqual(night_sys["t_sun_pb"], 0.0)

    def test_beam_targets_parsing(self):
        """Test parsing of celestial beam target coordinates."""
        target_str = "Crab Pulsar:83.633,22.014;PSR J0437-4715:69.316,-47.252"
        beams = parse_beam_targets(target_str, max_beams=4, default_lst=5.5)

        self.assertEqual(len(beams), 4)
        self.assertEqual(beams[0]["name"], "Crab Pulsar")
        self.assertAlmostEqual(beams[0]["ra_deg"], 83.633)
        self.assertAlmostEqual(beams[0]["dec_deg"], 22.014)

        self.assertEqual(beams[1]["name"], "PSR J0437-4715")
        self.assertAlmostEqual(beams[1]["ra_deg"], 69.316)
        self.assertAlmostEqual(beams[1]["dec_deg"], -47.252)

        # Hex offset fill for remaining beams
        self.assertIn("Hex Offset", beams[2]["name"])
        self.assertIn("Hex Offset", beams[3]["name"])

    def test_correlator_yaml_generation(self):
        """Test generating Kotekan correlator YAML configuration."""
        yaml_out = self.temp_dir / "corr.yaml"
        create_correlator_yaml(
            yaml_path=yaml_out,
            baseband_dir=self.temp_dir,
            baseband_name="test_sim",
            correlator_dir=self.temp_dir / "corr",
            num_frames=10,
            num_elements=64,
            num_local_freq=336,
            samples_per_data_set=1536,
        )

        self.assertTrue(yaml_out.is_file())
        content = yaml_out.read_text(encoding="utf-8")
        self.assertIn("kotekan_stage: rawFileRead", content)
        self.assertIn("kotekan_stage: cudaShuffleAstron", content)
        self.assertIn("kotekan_stage: cudaCorrelatorAstron", content)
        self.assertIn("kotekan_stage: rawFileWrite", content)
        self.assertIn("num_elements: 64", content)
        self.assertIn("num_blocks: 528", content)

    def test_beam_tracker_yaml_generation(self):
        """Test generating Kotekan beam tracker YAML configuration."""
        yaml_out = self.temp_dir / "tracker.yaml"
        targets = parse_beam_targets("Crab:83.633,22.014", max_beams=4)
        create_beam_tracker_yaml(
            yaml_path=yaml_out,
            baseband_dir=self.temp_dir,
            baseband_name="test_sim",
            tracker_dir=self.temp_dir / "tracker",
            tracker_name="beams_test",
            num_frames=10,
            beam_targets=targets,
            max_beams=4,
        )

        self.assertTrue(yaml_out.is_file())
        content = yaml_out.read_text(encoding="utf-8")
        self.assertIn("kotekan_stage: cudaAntennaMask", content)
        self.assertIn("kotekan_stage: cudaDirectBeamTrackerCommand", content)
        self.assertIn("max_beams: 4", content)
        self.assertIn("Crab", content)

    def test_direct_tracker_cadence_and_power_benchmark(self):
        """Test Direct Beam Tracker benchmark under physical 5.12 ms cadence."""
        # 1. Byte-accurate VRAM layout calculation
        vram_info = calculate_direct_tracker_vram(n_ant=64, n_freq=672, n_time=1536, max_beams=4, buffer_depth=2)
        self.assertGreater(vram_info["total_vram_mb"], 400.0)
        self.assertLess(vram_info["total_vram_mb"], 1000.0)
        self.assertEqual(vram_info["input_frame_mb"], round(1536 * 672 * 64 * 1 / (1024.0 * 1024.0), 3))

        # 2. Benchmark run with cadence, power, and VRAM output
        res = run_direct_tracker_benchmark(
            ant_counts=[32, 64],
            beam_counts=[1, 4],
            num_freq=336,
            samples_per_frame=1536,
            gpu_profile_name="rtx4090",
        )
        self.assertEqual(res["target_gpu"], "NVIDIA GeForce RTX 4090 (Ada Lovelace)")
        self.assertAlmostEqual(res["cadence_ms"], 5.12, places=2)
        records = res["records"]
        self.assertEqual(len(records), 4)

        for r in records:
            self.assertGreater(r["headroom_factor"], 1.0)  # Must be faster than real-time
            self.assertLess(r["budget_utilization_pct"], 100.0)  # Must fit in budget
            self.assertGreater(r["active_kernel_power_w"], 100.0)  # Active power during burst
            self.assertLess(r["real_cadence_power_w"], 100.0)  # Continuous avg power under real cadence
            self.assertGreater(r["energy_per_frame_mj"], 0.0)
            self.assertGreater(r["vram_allocated_gb"], 0.0)

    def test_astro_dynamic_ephemeris_and_catalog(self):
        """Test astronomical coordinates, dynamic LST, and visible source ranking."""
        import datetime
        dt = parse_observation_time("2026-10-15T04:20:00Z")
        self.assertEqual(dt.year, 2026)
        self.assertEqual(dt.month, 10)
        self.assertEqual(dt.day, 15)

        lst = datetime_to_lst_hours(dt)
        self.assertGreaterEqual(lst, 0.0)
        self.assertLess(lst, 24.0)

        # Auto-targets selection finds visible sources
        visible = find_visible_sources(dt, min_elevation_deg=0.0)
        self.assertGreater(len(visible), 0)

        targets = resolve_beam_targets("auto", obs_time=dt, max_beams=4)
        self.assertEqual(len(targets), 4)
        for t in targets:
            self.assertIn("ra_deg", t)
            self.assertIn("dec_deg", t)
            self.assertIn("lst_hours", t)
            self.assertIn("l0", t)
            self.assertIn("m0", t)

    def test_reference_window_manifest_lifecycle(self):
        """Test creating, saving, discovering, and loading reference window manifests."""
        ref_dir = self.temp_dir / "reference_library"
        win_dir = self.temp_dir / "sample_window"
        win_dir.mkdir(parents=True, exist_ok=True)

        # Create dummy bin frames
        for i in range(5):
            (win_dir / f"sample_{i:07d}.bin").write_bytes(b"\x00" * 1024)

        manifest = create_reference_manifest(
            window_dir=win_dir,
            tag="test_window_tag",
            description="Test reference dataset",
            obs_time="2026-10-15T04:20:00Z",
            antennas=32,
            num_freq=112,
            samples_per_frame=1536,
        )
        self.assertEqual(manifest.tag, "test_window_tag")
        self.assertEqual(manifest.total_frames, 5)
        self.assertEqual(manifest.antennas, 32)

        # Save to reference library
        saved_dir = save_as_reference_window(
            src_window_dir=win_dir,
            tag="test_window_tag",
            description="Test reference dataset",
            reference_dir=ref_dir,
        )
        self.assertTrue((saved_dir / "window_manifest.json").is_file())

        # Discover reference windows
        discovered = list_reference_windows(reference_dir=ref_dir)
        self.assertEqual(len(discovered), 1)
        self.assertEqual(discovered[0]["tag"], "test_window_tag")

        # Load reference window
        retrieved_dir, retrieved_man = get_reference_window("test_window_tag", reference_dir=ref_dir)
        self.assertEqual(retrieved_man.tag, "test_window_tag")
        self.assertEqual(retrieved_man.total_frames, 5)

    def test_quick_window_generation(self):
        """Test generating small 1-second baseband window with metadata."""
        cfg = get_preset_config("quick", "day")
        cfg.duration_s = 0.05  # tiny test duration
        cfg.event_dense_s = 0.005
        cfg.scratch_dir = self.temp_dir
        cfg.window_name = "test_window"
        cfg.workers = 1

        res = generate_simulation_window(cfg)
        self.assertEqual(res["window_name"], "test_window")
        self.assertGreater(res["num_written"], 0)

        meta_path = res["meta_h5_path"]
        events_path = res["events_json_path"]
        self.assertTrue(meta_path.is_file())
        self.assertTrue(events_path.is_file())

        with h5py.File(meta_path, "r") as h5:
            self.assertEqual(h5.attrs["antennas"], 64)
            self.assertIn("frequencies_mhz", h5)
            self.assertIn("antenna_pos_x_m", h5)
            self.assertIn("frames/timestamp_s", h5)

        with open(events_path, "r", encoding="utf-8") as f:
            ev_list = json.load(f)
            self.assertIsInstance(ev_list, list)


if __name__ == "__main__":
    unittest.main()
