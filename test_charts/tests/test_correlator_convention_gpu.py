"""GPU check of the correlator component convention against the charts-utils reader.

Run with /usr/bin/python3 -m test_charts.tests.test_correlator_convention_gpu
[--results-dir PATH] [--kotekan-bin PATH] [--charts-utils PATH].

Constant-in-time, element- and channel-dependent bytes go through the unchanged
rawFileRead -> cudaShuffleAstron -> cudaCorrelatorAstron -> rawFileWrite chain.
The frame is read with the official charts-utils reader (charts_logical order,
V_ab = x'_a·conj(x'_b)) and compared with the closed form
V_ab = T·x'_a·conj(x'_b), x' = high nibble + i·low nibble, element r = 63 − a.
Negative controls prove the comparison separates the plausible wrong conventions.
This is a synthetic verification of the TCC output order, not an on-sky result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import numpy as np

from test_charts.sim.packet_loss import write_frame
from test_charts.sim.pipeline import execute_kotekan

ROOT = Path(__file__).resolve().parents[2]
BINARY = ROOT / "build/kotekan/kotekan"
CHARTS_UTILS = Path.home() / "charts-utils"
RESULTS = None

N_ELEM, N_FREQ, T = 64, 336, 1536


def signed_nibble(v, lowest):
    span = 7 - lowest + 1
    return (v - lowest) % span + lowest


def make_bytes(with_minus_8=False):
    """Deterministic bytes [F][raw element]; asymmetric, with zero nibbles.

    Default range is −7…7 (the beamformer's clamp). `with_minus_8` spans −8…7 and
    pins raw element 7 to (−8, −8) to characterise the correlator at the extreme.
    """
    lowest = -8 if with_minus_8 else -7
    f = np.arange(N_FREQ)[:, None]
    r = np.arange(N_ELEM)[None, :]
    hi = signed_nibble(3 * r + 5 * f + 1, lowest)
    lo = signed_nibble(7 * r - 2 * f + 4, lowest)
    hi[:, 5] = 0          # purely low-nibble element
    lo[:, 6] = 0          # purely high-nibble element
    if with_minus_8:
        hi[:, 7], lo[:, 7] = -8, -8
    return (((hi & 0xF) << 4) | (lo & 0xF)).astype(np.uint8), hi, lo


CORRELATOR_YAML = """\
type: config
log_level: info
cpu_affinity: []
num_elements: {E}
num_local_freq: {F}
samples_per_data_set: {T}
num_data_sets: 1
block_size: 2
num_blocks: 528
elements_per_thread_block: 32
sizeof_int: 4
buffer_depth: 3
data_format: cuda_wmma
frame_arrival_period: samples_per_data_set / 300000.0

main_pool:
  kotekan_metadata_pool: chartsMetadata
  num_metadata_objects: 16

host_voltage_buf:
  kotekan_buffer: standard
  num_frames: buffer_depth
  frame_size: samples_per_data_set * num_local_freq * num_elements
  numa_node: 0
  metadata_pool: main_pool

host_correlation_buf:
  kotekan_buffer: standard
  num_frames: buffer_depth
  frame_size: num_local_freq * num_blocks * (block_size*block_size) * 2 * num_data_sets * sizeof_int
  numa_node: 0
  metadata_pool: main_pool

reader:
  kotekan_stage: rawFileRead
  buf: host_voltage_buf
  base_dir: {base}
  file_name: bb
  file_ext: bin
  prefix_hostname: false
  end_interrupt: false

gpu:
  kotekan_stage: cudaProcess
  gpu_id: 0
  buffer_depth: 3
  kernel_path: lib/cuda/kernels
  in_buffers:
    host_voltage: host_voltage_buf
  out_buffers:
    host_correlation: host_correlation_buf
  commands:
    - name: cudaInputData
      in_buf: host_voltage
      gpu_mem: voltage_0
    - name: cudaSyncInput
    - name: cudaShuffleAstron
      gpu_mem_voltage: voltage_0
      gpu_mem_ordered_voltage: ordered_voltage_0
    - name: cudaCorrelatorAstron
      gpu_mem_voltage: ordered_voltage_0
      gpu_mem_correlation_matrix: correlation_matrix_0
    - name: cudaSyncOutput
    - name: cudaOutputData
      in_buf: host_voltage
      gpu_mem: correlation_matrix_0
      out_buf: host_correlation

writer:
  kotekan_stage: rawFileWrite
  in_buf: host_correlation_buf
  base_dir: {out}
  file_name: corr
  file_ext: bin
  prefix_hostname: false
  num_frames_per_file: 1
  exit_after_n_files: 1
"""


def run_correlator(workdir, with_minus_8=False):
    byts, hi, lo = make_bytes(with_minus_8)
    # [T][F][E] with a time-constant sample, native chartsMetadata + payload.
    frame = np.broadcast_to(byts, (T, N_FREQ, N_ELEM)).copy()
    base, out = workdir / "in", workdir / "out"
    base.mkdir(parents=True, exist_ok=True)
    out.mkdir(exist_ok=True)
    write_frame(base / "bb_0000000.bin", frame, 0, 0)
    yaml_path = workdir / "corr.yaml"
    yaml_path.write_text(CORRELATOR_YAML.format(E=N_ELEM, F=N_FREQ, T=T, base=base, out=out))
    rc = execute_kotekan(yaml_path, BINARY, workdir / "kotekan.log", timeout_s=300)
    outs = sorted(out.glob("corr_*.bin"))
    if len(outs) != 1:
        raise RuntimeError(f"rc={rc}; expected one correlation frame, found {outs}; "
                           f"see {workdir / 'kotekan.log'}")
    return outs[0], hi, lo


def expected_visibilities(hi, lo, convention):
    """Closed form in charts_logical a: x' per `convention`, raw r = 63 − a."""
    hi, lo = hi[:, ::-1].astype(np.float64), lo[:, ::-1].astype(np.float64)
    x = {"high_real": hi + 1j * lo, "low_real": lo + 1j * hi}[convention]
    iu = np.triu_indices(N_ELEM)
    return T * x[:, iu[0]] * np.conj(x[:, iu[1]])


class CorrelatorConventionGpu(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not BINARY.is_file():
            raise unittest.SkipTest(f"no kotekan binary at {BINARY}")
        sys.path.insert(0, str(CHARTS_UTILS))
        from charts_utils.core import io
        cls.io = io
        cls.tmp = tempfile.TemporaryDirectory() if RESULTS is None else None
        root = Path(cls.tmp.name) if cls.tmp else RESULTS / "correlator_convention"

        def run(name, with_minus_8):
            out, hi, lo = run_correlator(root / name, with_minus_8)
            vis = io.load_gpu_visibilities(
                str(out), N_FREQ, num_antennas=N_ELEM, frame_format="prefixed"
            ).astype(np.complex128)
            return vis, hi, lo

        cls.vis, cls.hi, cls.lo = run("range_pm7", False)
        cls.vis8, cls.hi8, cls.lo8 = run("range_with_minus_8", True)

    @classmethod
    def tearDownClass(cls):
        if cls.tmp:
            cls.tmp.cleanup()

    def test_exact_match_high_nibble_real(self):
        want = expected_visibilities(self.hi, self.lo, "high_real")
        self.assertEqual(self.vis.shape, want.shape)
        np.testing.assert_array_equal(self.vis, want)

    def test_negative_control_low_nibble_real(self):
        wrong = expected_visibilities(self.hi, self.lo, "low_real")
        self.assertFalse(np.array_equal(self.vis, wrong))

    def test_negative_control_conjugate(self):
        want = expected_visibilities(self.hi, self.lo, "high_real")
        self.assertFalse(np.array_equal(self.vis, np.conj(want)))

    def test_negative_control_swapped_baseline_order(self):
        want = expected_visibilities(self.hi, self.lo, "high_real")
        swapped = np.conj(want)  # V_ba = conj(V_ab) read back as stored V_ab
        self.assertGreater(np.max(np.abs(swapped - want)), 0)
        self.assertFalse(np.array_equal(self.vis, swapped))

    def test_negative_control_element_order_not_reversed(self):
        # Same closed form without the raw = 63 − a reversal.
        hi, lo = self.hi.astype(np.float64), self.lo.astype(np.float64)
        x = hi + 1j * lo
        iu = np.triu_indices(N_ELEM)
        wrong = T * x[:, iu[0]] * np.conj(x[:, iu[1]])
        self.assertFalse(np.array_equal(self.vis, wrong))

    def test_minus_8_characterisation(self):
        """Record the −8 behaviour; this is a finding, not an acceptance of −8.

        The tensor-core correlator is exact for −7…7. With a −8 nibble the real
        part stays exact and the imaginary part can be wrong (negating −8 overflows
        int4). Every mismatching product must involve a −8 nibble.
        """
        want = expected_visibilities(self.hi8, self.lo8, "high_real")
        self.assertEqual(float(np.max(np.abs((self.vis8 - want).real))), 0.0)
        has8 = ((self.hi8 == -8) | (self.lo8 == -8))[:, ::-1]
        a, b = np.triu_indices(N_ELEM)
        involved = has8[:, a] | has8[:, b]
        bad = self.vis8 != want
        self.assertFalse(np.any(bad & ~involved))
        self.assertTrue(np.any(bad))
        if RESULTS:
            (RESULTS / "minus_8_characterisation.json").write_text(json.dumps({
                "mismatched_products": int(bad.sum()), "total_products": int(bad.size),
                "mismatch_without_minus_8": int((bad & ~involved).sum()),
                "max_real_abs_diff": float(np.max(np.abs((self.vis8 - want).real))),
            }, indent=2) + "\n")

    def test_beamformer_component_relation(self):
        """x = i·conj(x'): the beamformer's low-real decode of the same bytes."""
        x = self.lo[:, ::-1] + 1j * self.hi[:, ::-1]          # low + i·high, charts_logical
        xp = self.hi[:, ::-1] + 1j * self.lo[:, ::-1]
        np.testing.assert_array_equal(x, 1j * np.conj(xp))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--kotekan-bin", type=Path, default=BINARY)
    parser.add_argument("--charts-utils", type=Path, default=CHARTS_UTILS)
    args, rest = parser.parse_known_args()
    RESULTS = args.results_dir.resolve() if args.results_dir else None
    BINARY, CHARTS_UTILS = args.kotekan_bin.resolve(), args.charts_utils.resolve()
    res = unittest.main(argv=[sys.argv[0], *rest], exit=False).result
    summary = {"tests": res.testsRun, "failures": len(res.failures),
               "errors": len(res.errors), "skipped": len(res.skipped)}
    if RESULTS:
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / "correlator_convention_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    sys.exit(0 if not (res.failures or res.errors or res.skipped) else 1)
