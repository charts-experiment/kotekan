#!/usr/bin/env python3
r"""Tier-2 integration test verifying writer independence from Kotekan.

Synthesizes physical baseband frames containing an injected celestial point source,
serializes them to disk using `RawBinWriter` (int4x2 packed), reads them back
using pure NumPy / Python, and computes cross-correlations without invoking Kotekan.

Verifies:
  - R2.1, R2.2: Complete decoupling of baseband generation/serialization from Kotekan.
  - F1, F2: Baseline geometric delay \tau_{ij} = (\vec{r}_j - \vec{r}_i) \cdot \hat{s} / c
    and fringe-stopping phase \phi_{ij}(f) = 2\pi f \tau_{ij} recovered accurately
    from quantized baseband frames via pure NumPy cross-correlation.
  - Parity with AGENTS.md §2.1 and §2.2 (no hardcoded constants, fiducial 5D ordering).

Theory & Literature:
  - Buschmann, B. A. P. (2025). "Design and Implementation of the F-Engine for CHARTS", §4.
  - Smith, K. M. (2024). "Note on CHIME FRB Beamforming", Eq. (phase alignment).
  - Ng, C., et al. (2017). "CHIME FRB: An Application of FFT Beamforming for a Radio Telescope".
"""

from __future__ import annotations

import math
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np

# Ensure test_charts is importable
_test_charts_dir = Path(__file__).resolve().parent.parent
if str(_test_charts_dir) not in sys.path:
    sys.path.insert(0, str(_test_charts_dir))

from sim.constants import (
    C_LIGHT,
    CHARTS_CHANNEL_WIDTH_MHZ,
    DEFAULT_FREQUENCY_START_MHZ,
    FPGA_TIME_RESOLUTION_US,
    get_antenna_positions,
)
from sim.generator import generate_simulation_window
from sim.presets import get_preset_config
from sim.writer import (
    BasebandWriter,
    RawBinWriter,
    WindowMetadata,
    pack_int4x2,
    read_raw_bin_frame,
    unpack_int4x2,
)


class TestWriterIndependence(unittest.TestCase):
    """Tier-2 Integration Suite: NumPy Cross-Correlation Recovery Without Kotekan."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="test_writer_indep_"))

    def tearDown(self):
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_injected_point_source_phase_recovery_numpy(self):
        r"""Verify that an injected point source serialized via RawBinWriter
        is accurately recovered via pure NumPy cross-correlation without Kotekan.

        Continuous Mathematical Model:
          Topocentric baseline: \vec{b}_{ij} = \vec{r}_j - \vec{r}_i
          Geometric delay: \tau_{ij} = \frac{\vec{b}_{ij} \cdot \hat{s}}{c} = \frac{(x_j - x_i) l + (y_j - y_i) m}{c}
          Theoretical fringe phase: \phi_{ij}(f) = 2\pi f \tau_{ij}
          Cross-correlation expectation:
            V_{ij}(f) = \frac{1}{T} \sum_{t=0}^{T-1} v_i(t, f) v_j^*(t, f) \propto e^{j \phi_{ij}(f)}
        """
        num_ant = 8
        num_freq = 4
        samples_per_frame = 1536
        dt_s = FPGA_TIME_RESOLUTION_US * 1e-6
        two_pi = 2.0 * math.pi
        c_inv = 1.0 / C_LIGHT

        pos_x, pos_y = get_antenna_positions(num_ant)
        freqs_hz = (
            DEFAULT_FREQUENCY_START_MHZ + np.arange(num_freq) * CHARTS_CHANNEL_WIDTH_MHZ
        ) * 1e6

        # Injected off-zenith point source direction cosines (l0, m0)
        l0 = 0.08
        m0 = -0.05
        source_amp = 4.5  # within int4 dynamic range [-7, 7]

        # Antenna delays relative to origin
        delays_s = (l0 * pos_x + m0 * pos_y) * c_inv  # shape: (num_ant,)

        t_indices = np.arange(samples_per_frame, dtype=np.float64)
        t_abs_s = t_indices * dt_s

        # Synthesize baseband voltages: (samples_per_frame, num_freq, num_ant)
        # Carrier base phase rotates through ~20 full cycles across the frame (5 kHz carrier)
        # ensuring the 4-bit quantizer uniformly samples phases around the unit circle
        carrier_phase = two_pi * (t_abs_s * 5000.0)  # (samples,)
        geom_phases = two_pi * (freqs_hz[:, None] * delays_s[None, :])  # (freq, ant)

        tot_phases = (
            carrier_phase[:, None, None] - geom_phases[None, :, :]
        )  # (samples, freq, ant)
        clean_voltages = (source_amp * np.exp(1j * tot_phases)).astype(np.complex64)

        # 1. Serialize using RawBinWriter
        writer = RawBinWriter(
            target_dir=self.temp_dir,
            window_name="indep_sim",
            samples_per_frame=samples_per_frame,
            num_freq=num_freq,
            num_elements=num_ant,
        )

        frame_file = writer.write_frame(0, clean_voltages)
        self.assertTrue(frame_file.is_file())

        meta = WindowMetadata(
            window_name="indep_sim",
            target_dir=self.temp_dir,
            utc_start="2026-10-01T15:00:00Z",
            duration_s=samples_per_frame * dt_s,
            antennas=num_ant,
            num_freq=num_freq,
            samples_per_frame=samples_per_frame,
            frame_duration_s=samples_per_frame * dt_s,
            frequencies_mhz=freqs_hz / 1e6,
            antenna_pos_x_m=pos_x,
            antenna_pos_y_m=pos_y,
            total_physical_frames=1,
            num_written_frames=1,
        )
        writer.write_metadata(meta)
        manifest = writer.finalize()
        self.assertEqual(manifest.total_frames, 1)

        # 2. Read back using pure NumPy reader (no Kotekan binary)
        read_voltages = read_raw_bin_frame(
            frame_file,
            shape=(samples_per_frame, num_freq, num_ant),
            unpack=True,
        )
        self.assertEqual(read_voltages.shape, (samples_per_frame, num_freq, num_ant))

        # 3. Compute cross-correlation in NumPy for all baselines (i, j)
        # Follows CHARTS visibility product convention (AGENTS.md §2.1, 2.2):
        # V_ij(f) = (1/T) sum_t v_i(t, f) * conj(v_j(t, f))
        for f_idx in range(num_freq):
            f_hz = freqs_hz[f_idx]
            v_chan = read_voltages[:, f_idx, :]  # shape: (samples, num_ant)

            # Correlation matrix: (num_ant, num_ant) with V_ij = <v_i * conj(v_j)>
            vis_matrix = np.dot(v_chan.T, v_chan.conj()) / samples_per_frame

            for i in range(num_ant):
                for j in range(i + 1, num_ant):
                    # Theoretical delay and fringe phase:
                    # v_i ~ exp(-j 2pi f tau_i), v_j ~ exp(-j 2pi f tau_j)
                    # V_ij = <v_i * conj(v_j)> ~ exp(-j 2pi f tau_i + j 2pi f tau_j) = exp(j 2pi f (tau_j - tau_i))
                    tau_ij = delays_s[j] - delays_s[i]
                    expected_phase = math.atan2(
                        math.sin(two_pi * f_hz * tau_ij),
                        math.cos(two_pi * f_hz * tau_ij),
                    )

                    measured_vis = vis_matrix[i, j]
                    measured_phase = np.angle(measured_vis)

                    # Phase difference modulo 2pi
                    phase_diff = np.angle(
                        np.exp(1j * (measured_phase - expected_phase))
                    )

                    # With int4 quantization ([-7, 7]) and 1536 time samples,
                    # phase error is < 0.05 radians (~2.8 degrees)
                    self.assertLess(
                        abs(phase_diff),
                        0.05,
                        f"Baseline ({i}, {j}) at {f_hz*1e-6:.1f} MHz phase error {abs(phase_diff):.4f} rad exceeds 0.05 rad",
                    )

    def test_02_end_to_end_generator_with_writer_independence(self):
        """Verify generate_simulation_window works seamlessly with standalone RawBinWriter."""
        cfg = get_preset_config("quick", "day", antennas=8, num_freq=16)
        cfg.duration_s = 0.05
        cfg.event_dense_s = 0.01
        cfg.scratch_dir = self.temp_dir
        cfg.window_name = "test_end_to_end_indep"
        cfg.workers = 1

        writer = RawBinWriter(
            target_dir=self.temp_dir / cfg.window_name,
            window_name=cfg.window_name,
            samples_per_frame=cfg.samples_per_frame,
            num_freq=cfg.num_freq,
            num_elements=cfg.antennas,
        )

        res = generate_simulation_window(cfg, writer=writer)
        self.assertEqual(res["window_name"], cfg.window_name)
        self.assertGreater(res["num_written"], 0)

        # Confirm binary frames exist and can be read with pure Python reader
        bin_files = sorted(res["target_dir"].glob("*.bin"))
        self.assertEqual(len(bin_files), res["num_written"])

        first_frame = read_raw_bin_frame(
            bin_files[0],
            shape=(cfg.samples_per_frame, cfg.num_freq, cfg.antennas),
            unpack=True,
        )
        self.assertEqual(
            first_frame.shape, (cfg.samples_per_frame, cfg.num_freq, cfg.antennas)
        )
        self.assertEqual(first_frame.dtype, np.complex64)


if __name__ == "__main__":
    unittest.main()
