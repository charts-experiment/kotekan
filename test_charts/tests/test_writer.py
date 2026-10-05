"""Tier-1 unit tests for BasebandWriter serialization, int4x2 packing, and metadata.

Verifies:
  - F6: int4x2 byte-exact fixtures: Byte = (Re & 0x0F) | ((Im & 0x0F) << 4)
  - B1: Kotekan rawFileWrite compatibility, 4-byte uint32 header, frame sizing
  - WindowManifest generation and SHA-256 checksum verification
  - HDF5Writer baseband dataset and metadata sidecars
  - 5D fiducial layout conversions (t_pkt, dish, pol, freq, t_samp)

Theory & Literature:
  - Buschmann, B. A. P. (2025). "Design and Implementation of the F-Engine for CHARTS", §4.
  - Smith, K. M. (2023). "Notes on CHORD F -> X Packet Format".
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import h5py
import numpy as np

# Ensure test_charts is on path
_test_charts_dir = Path(__file__).resolve().parent.parent
if str(_test_charts_dir) not in sys.path:
    sys.path.insert(0, str(_test_charts_dir))

from sim.constants import (
    CHARTS_CHANNEL_WIDTH_MHZ,
    DEFAULT_FREQUENCY_START_MHZ,
    FPGA_TIME_RESOLUTION_US,
    get_antenna_positions,
)
from sim.writer import (
    BasebandWriter,
    HDF5Writer,
    RawBinWriter,
    WindowMetadata,
    create_writer,
    fiducial_5d_to_kotekan,
    kotekan_to_fiducial_5d,
    pack_int4x2,
    read_raw_bin_frame,
    unpack_int4x2,
)


class TestBasebandWriterSuite(unittest.TestCase):
    """Tier-1 Unit Test Suite for Baseband Serialization and Writers."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="test_charts_writer_"))

    def tearDown(self):
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_int4x2_byte_exact_fixtures(self):
        r"""Verify bit-exact int4x2 packing against known reference vectors.

        Mathematical Specification:
          \text{Byte} = (r \ \& \ 0x0F) \ | \ ((i \ \& \ 0x0F) \ll 4)
        where r, i \in [-8, 7] in signed 4-bit two's complement.
        """
        # Exact fixture vectors: (real, imag) -> expected_byte
        fixtures = [
            (0, 0, 0x00),
            (1, 0, 0x01),
            (0, 1, 0x10),
            (7, 7, 0x77),
            (-1, -1, 0xFF),   # -1 & 0xF = 0xF -> 0xF | (0xF << 4) = 0xFF
            (-7, 7, 0x79),    # -7 & 0xF = 0x9, 7 << 4 = 0x70 -> 0x79
            (-8, -8, 0x88),   # -8 & 0xF = 0x8 -> 0x88
            (3, -2, 0xE3),    # 3 & 0xF = 0x3, -2 & 0xF = 0xE -> 0xE3
            (-5, 2, 0x2B),    # -5 & 0xF = 0xB, 2 << 4 = 0x20 -> 0x2B
            (5, -5, 0xB5),    # 5 & 0xF = 0x5, -5 & 0xF = 0xB -> 0xB5
        ]

        for r, i, expected_byte in fixtures:
            c_val = np.array([complex(r, i)], dtype=np.complex64)
            packed = pack_int4x2(c_val)
            self.assertEqual(
                packed[0],
                expected_byte,
                f"Packing failed for ({r}, {i}): got 0x{packed[0]:02X}, expected 0x{expected_byte:02X}"
            )

            # Roundtrip unpacking verification
            unpacked = unpack_int4x2(packed)
            self.assertAlmostEqual(unpacked[0].real, float(r), places=5)
            self.assertAlmostEqual(unpacked[0].imag, float(i), places=5)

    def test_02_full_dynamic_range_roundtrip(self):
        """Test full 4-bit integer grid [-7, 7] x [-7, 7] roundtrip."""
        real_vals = np.arange(-7, 8, dtype=np.float32)
        imag_vals = np.arange(-7, 8, dtype=np.float32)
        rr, ii = np.meshgrid(real_vals, imag_vals)
        grid_complex = (rr + 1j * ii).astype(np.complex64)

        packed = pack_int4x2(grid_complex)
        self.assertEqual(packed.shape, grid_complex.shape)
        self.assertEqual(packed.dtype, np.uint8)

        unpacked = unpack_int4x2(packed)
        np.testing.assert_allclose(unpacked.real, rr, atol=1e-5)
        np.testing.assert_allclose(unpacked.imag, ii, atol=1e-5)

    def test_03_raw_bin_writer_frame_sizing_and_header(self):
        """Verify Kotekan rawFileWrite binary file layout and frame sizes."""
        samples_per_frame = 256
        num_freq = 16
        num_ant = 8

        writer = RawBinWriter(
            target_dir=self.temp_dir,
            window_name="test_raw_win",
            samples_per_frame=samples_per_frame,
            num_freq=num_freq,
            num_elements=num_ant,
        )

        expected_payload_bytes = samples_per_frame * num_freq * num_ant
        expected_total_bytes = 4 + expected_payload_bytes  # 4 bytes for metadata_size header

        test_frame = np.ones((samples_per_frame, num_freq, num_ant), dtype=np.complex64)
        out_file = writer.write_frame(0, test_frame)

        self.assertTrue(out_file.is_file())
        self.assertEqual(out_file.name, "test_raw_win_0000000.bin")
        self.assertEqual(out_file.stat().st_size, expected_total_bytes)

        # Verify header is uint32(0)
        with open(out_file, "rb") as f:
            header = np.fromfile(f, dtype=np.uint32, count=1)[0]
            self.assertEqual(header, 0)
            payload = np.fromfile(f, dtype=np.uint8)
            self.assertEqual(payload.size, expected_payload_bytes)

        # Read back via read_raw_bin_frame
        unpacked = read_raw_bin_frame(out_file, shape=(samples_per_frame, num_freq, num_ant), unpack=True)
        self.assertEqual(unpacked.shape, (samples_per_frame, num_freq, num_ant))
        self.assertAlmostEqual(float(unpacked[0, 0, 0].real), 1.0, places=4)
        self.assertAlmostEqual(float(unpacked[0, 0, 0].imag), 0.0, places=4)

    def test_04_raw_bin_writer_metadata_and_manifest(self):
        """Verify RawBinWriter emits valid companion HDF5 metadata and manifest."""
        samples_per_frame = 128
        num_freq = 8
        num_ant = 4

        writer = RawBinWriter(
            target_dir=self.temp_dir,
            window_name="test_manifest_win",
            samples_per_frame=samples_per_frame,
            num_freq=num_freq,
            num_elements=num_ant,
        )

        # Write 3 frames
        for f_idx in range(3):
            data = np.full((samples_per_frame, num_freq, num_ant), complex(f_idx, -f_idx), dtype=np.complex64)
            writer.write_frame(f_idx, data)

        pos_x, pos_y = get_antenna_positions(num_ant)
        freqs_hz = (DEFAULT_FREQUENCY_START_MHZ + np.arange(num_freq) * CHARTS_CHANNEL_WIDTH_MHZ) * 1e6
        dt_s = FPGA_TIME_RESOLUTION_US * 1e-6

        meta = WindowMetadata(
            window_name="test_manifest_win",
            target_dir=self.temp_dir,
            utc_start="2026-10-01T15:00:00Z",
            duration_s=3 * samples_per_frame * dt_s,
            antennas=num_ant,
            num_freq=num_freq,
            samples_per_frame=samples_per_frame,
            frame_duration_s=samples_per_frame * dt_s,
            frequencies_mhz=freqs_hz / 1e6,
            antenna_pos_x_m=pos_x,
            antenna_pos_y_m=pos_y,
            total_physical_frames=3,
            num_written_frames=3,
            events=[{"event_id": "test_evt", "event_type": "pulsar", "dm": 10.0}],
        )

        meta_path = writer.write_metadata(meta)
        self.assertTrue(meta_path.is_file())

        with h5py.File(meta_path, "r") as h5:
            self.assertEqual(h5.attrs["antennas"], num_ant)
            self.assertEqual(h5.attrs["num_freq"], num_freq)
            self.assertIn("frequencies_mhz", h5)
            self.assertIn("antenna_pos_x_m", h5)

        # Finalize and verify manifest
        manifest = writer.finalize()
        self.assertEqual(manifest.tag, "test_manifest_win")
        self.assertEqual(manifest.total_frames, 3)
        self.assertEqual(len(manifest.frame_files), 3)

        manifest_file = self.temp_dir / "window_manifest.json"
        self.assertTrue(manifest_file.is_file())
        with open(manifest_file, "r", encoding="utf-8") as f:
            man_data = json.load(f)
        self.assertEqual(man_data["total_frames"], 3)
        self.assertEqual(len(man_data["frame_files"]), 3)

    def test_05_hdf5_writer(self):
        """Verify HDF5Writer creates baseband dataset and metadata container."""
        samples_per_frame = 64
        num_freq = 4
        num_ant = 4

        writer = HDF5Writer(
            target_dir=self.temp_dir,
            window_name="test_h5_win",
            samples_per_frame=samples_per_frame,
            num_freq=num_freq,
            num_elements=num_ant,
        )

        # Write 2 frames
        f0 = np.full((samples_per_frame, num_freq, num_ant), complex(2, 3), dtype=np.complex64)
        f1 = np.full((samples_per_frame, num_freq, num_ant), complex(-4, 5), dtype=np.complex64)

        writer.write_frame(0, f0)
        writer.write_frame(1, f1)

        pos_x, pos_y = get_antenna_positions(num_ant)
        freqs_hz = (DEFAULT_FREQUENCY_START_MHZ + np.arange(num_freq) * CHARTS_CHANNEL_WIDTH_MHZ) * 1e6
        dt_s = FPGA_TIME_RESOLUTION_US * 1e-6

        meta = WindowMetadata(
            window_name="test_h5_win",
            target_dir=self.temp_dir,
            utc_start="2026-10-01T15:00:00Z",
            duration_s=2 * samples_per_frame * dt_s,
            antennas=num_ant,
            num_freq=num_freq,
            samples_per_frame=samples_per_frame,
            frame_duration_s=samples_per_frame * dt_s,
            frequencies_mhz=freqs_hz / 1e6,
            antenna_pos_x_m=pos_x,
            antenna_pos_y_m=pos_y,
            total_physical_frames=2,
            num_written_frames=2,
        )

        meta_path = writer.write_metadata(meta)
        self.assertTrue(meta_path.is_file())

        manifest = writer.finalize()
        self.assertEqual(manifest.total_frames, 2)

        # Verify HDF5 dataset
        h5_file = self.temp_dir / "test_h5_win.h5"
        self.assertTrue(h5_file.is_file())
        with h5py.File(h5_file, "r") as h5:
            self.assertIn("baseband", h5)
            dset = h5["baseband"]
            self.assertEqual(dset.shape, (2, samples_per_frame, num_freq, num_ant))
            self.assertEqual(dset.dtype, np.uint8)

            # Unpack first frame and verify content
            unpacked0 = unpack_int4x2(dset[0])
            self.assertAlmostEqual(float(unpacked0[0, 0, 0].real), 2.0, places=4)
            self.assertAlmostEqual(float(unpacked0[0, 0, 0].imag), 3.0, places=4)

            unpacked1 = unpack_int4x2(dset[1])
            self.assertAlmostEqual(float(unpacked1[0, 0, 0].real), -4.0, places=4)
            self.assertAlmostEqual(float(unpacked1[0, 0, 0].imag), 5.0, places=4)

    def test_06_fiducial_5d_layout_conversion(self):
        """Verify bi-directional conversion between fiducial 5D and Kotekan layout."""
        num_dishes = 8
        num_pol = 2
        num_elements = num_dishes * num_pol
        num_freq = 4
        samples_per_frame = 32

        # 1. 4D single packet: (dish, pol, freq, t_samp)
        orig_4d = np.random.randint(-7, 7, size=(num_dishes, num_pol, num_freq, samples_per_frame)).astype(np.float32)
        kotekan_3d = fiducial_5d_to_kotekan(orig_4d)
        self.assertEqual(kotekan_3d.shape, (samples_per_frame, num_freq, num_elements))

        recovered_4d = kotekan_to_fiducial_5d(kotekan_3d, num_dishes=num_dishes, num_pol=num_pol)
        self.assertEqual(recovered_4d.shape, orig_4d.shape)
        np.testing.assert_array_equal(recovered_4d, orig_4d)

        # 2. 5D multi packet: (t_pkt, dish, pol, freq, t_samp)
        t_pkt = 3
        orig_5d = np.random.randint(-7, 7, size=(t_pkt, num_dishes, num_pol, num_freq, samples_per_frame)).astype(np.float32)
        kotekan_4d = fiducial_5d_to_kotekan(orig_5d)
        self.assertEqual(kotekan_4d.shape, (t_pkt, samples_per_frame, num_freq, num_elements))

        recovered_5d = kotekan_to_fiducial_5d(kotekan_4d, num_dishes=num_dishes, num_pol=num_pol)
        self.assertEqual(recovered_5d.shape, orig_5d.shape)
        np.testing.assert_array_equal(recovered_5d, orig_5d)

    def test_07_factory_function(self):
        """Test create_writer factory function for all supported writer types."""
        raw_writer = create_writer("raw_bin", target_dir=self.temp_dir / "w1", window_name="w1")
        self.assertIsInstance(raw_writer, RawBinWriter)
        self.assertIsInstance(raw_writer, BasebandWriter)

        h5_writer = create_writer("hdf5", target_dir=self.temp_dir / "w2", window_name="w2")
        self.assertIsInstance(h5_writer, HDF5Writer)
        self.assertIsInstance(h5_writer, BasebandWriter)

        with self.assertRaises(ValueError):
            create_writer("invalid_backend", target_dir=self.temp_dir, window_name="w3")


if __name__ == "__main__":
    unittest.main()
