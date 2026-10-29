"""GPU numerical and synthetic-science validation of the CHARTS beamformer.

Run with /usr/bin/python3 -m test_charts.tests.test_stage_c_numerics_gpu [--results-dir PATH].

Expectations are analytical, per fixture, never a port of the kernels:
* Beamforming: V[t,f,b] = sum_e W[b,f,e] x[t,f,e] with x = low nibble + i*high nibble,
  P = sum over the window of |V|^2 (sum-power, tails sum only the supplied spectra).
* Fine channelization: the documented section-2 equation, evaluated as a direct 128-tap sum
  Y[m,f,u,e] = sum_s h_s x[32m-96+s,f,e] exp(-2 pi i (u-15.5) s/32), zero history at startup
  or after a sequence jump, then the documented symmetric clamp -7..7 and round half to even.
  Components within TIE of a half-integer may round either way on the GPU (float32 FFT);
  each such component is allowed |dq| <= 1 and propagated with the section-5 item-3 bound
  sum_m (2|V_m| eps_m + eps_m^2), eps_m = sum_e |W_e| (number of ambiguous components).
* Float32 accumulation allowance: 1e-3 + 4 * 2^-24 * (E + K) * sum_m (sum_e |W_e||x_e|)^2.

Probe weights expose single elements without diagnostic dumps: beams 0..47 are one-hot on
elements 0..47 (W = 1), beams 48..63 are seeded non-unit complex combinations of all 64.
Synthetic verification only; nothing here is validated against the sky.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import yaml

from test_charts.sim.packet_loss import read_product, write_frame
from test_charts.sim.pipeline import create_beamformer_yaml, execute_kotekan

ROOT = Path(__file__).resolve().parents[2]
BINARY = ROOT / "build/kotekan/kotekan"
RESULTS = None

E = B = 64
F = 336
U = 32
FF = F * U
CLAMP = 7.0
TIE = 2e-4
EPS32 = 2.0 ** -24
N_ONEHOT = 48


# --------------------------------------------------------------------------- codec

def decode(raw: np.ndarray) -> np.ndarray:
    """Packed bytes -> complex128, x = low nibble + i*high nibble, both signed -8..7."""
    raw = raw.astype(np.int16)
    re = ((raw & 0xF) ^ 8) - 8
    im = (((raw >> 4) & 0xF) ^ 8) - 8
    return re + 1j * im


def encode(re: np.ndarray, im: np.ndarray) -> np.ndarray:
    re = np.asarray(re, np.int64)
    im = np.asarray(im, np.int64)
    if re.min() < -8 or re.max() > 7 or im.min() < -8 or im.max() > 7:
        raise ValueError("int4 components out of range")
    return ((re & 0xF) | ((im & 0xF) << 4)).astype(np.uint8)


def quantize_input(z: np.ndarray) -> np.ndarray:
    return encode(np.clip(np.rint(z.real), -8, 7), np.clip(np.rint(z.imag), -8, 7))


# --------------------------------------------------------------------------- PFB

def prototype() -> np.ndarray:
    s = np.arange(128)
    q = (2 * s - 127) / 258
    return np.cos(np.pi * q) ** 2 * np.sinc(4 * q) / 32


def analysis_matrix(alternating_sign: bool = True) -> np.ndarray:
    """[128][32]. alternating_sign=False is a negative control: it drops the (-1)^j that the
    complete-prototype phase contributes to a branch implementation (s = 32j + p)."""
    s = np.arange(128)[:, None]
    u = np.arange(U)[None, :]
    km = prototype()[:, None] * np.exp(-2j * np.pi * (u - 15.5) * s / 32)
    if not alternating_sign:
        km = km * ((-1.0) ** (s // 32))
    return km


def fine_quantized(raw_frame: np.ndarray, history: np.ndarray | None, *,
                   alternating_sign: bool = True, shift: int = 0):
    """Section-2 equation then clamp/round. Returns q [M][F*32][E] complex64, amb uint8.
    shift (negative control) moves the 128-sample support later by that many samples."""
    t_raw = raw_frame.shape[0]
    m_count = t_raw // 32
    x = decode(raw_frame)
    hist = np.zeros((96, F, E), complex) if history is None else decode(history)
    ext = np.concatenate([hist, x, np.zeros((shift, F, E), complex)], axis=0)[shift:]
    km = analysis_matrix(alternating_sign)
    q = np.empty((m_count, F, U, E), np.complex64)
    amb = np.empty((m_count, F, U, E), np.uint8)
    for f0 in range(0, F, 16):
        win = np.lib.stride_tricks.sliding_window_view(ext[:, f0:f0 + 16], 128, axis=0)[::32][:m_count]
        y = (win @ km).transpose(0, 1, 3, 2)                                   # [M][16][U][E]
        parts = []
        a = np.zeros(y.shape, np.uint8)
        for comp in (y.real, y.imag):
            c = np.clip(comp, -CLAMP, CLAMP)
            parts.append(np.rint(c))
            a += (np.abs(c - np.floor(c) - 0.5) < TIE).astype(np.uint8)
        q[:, f0:f0 + 16] = parts[0] + 1j * parts[1]
        amb[:, f0:f0 + 16] = a
    return q.reshape(m_count, FF, E), amb.reshape(m_count, FF, E)


# --------------------------------------------------------------------------- beam power

def expected_power(x: np.ndarray, amb: np.ndarray | None, w_bfe: np.ndarray, k: int):
    """x [T][F'][E] complex; w [B][F'][E], or [B][E] for frequency-independent probes.
    Returns (P, bound, ambiguous) as [ceil(T/K)][F'][B]; ambiguous marks windows whose
    expectation contains a near-half-integer rounding (the only place bound exceeds float)."""
    t, nf, _ = x.shape
    n_out = -(-t // k)
    p = np.zeros((n_out, nf, B))
    bound = np.zeros((n_out, nf, B))
    ambiguous = np.zeros((n_out, nf, B), bool)
    if w_bfe.ndim == 2:
        wt = w_bfe.T.astype(np.complex128)                                       # [E][B]
    else:
        wt = np.ascontiguousarray(w_bfe.transpose(1, 2, 0)).astype(np.complex128)  # [F'][E][B]
    aw = np.abs(wt)
    for wdx in range(n_out):
        sl = slice(wdx * k, min(t, (wdx + 1) * k))
        xs = x[sl].transpose(1, 0, 2).astype(np.complex128)                     # [F'][t][E]
        v = xs @ wt                                                              # [F'][t][B]
        av = np.abs(v)
        mag = np.abs(xs) @ aw
        p[wdx] = np.sum(av ** 2, axis=1)
        bnd = 1e-3 + 4 * EPS32 * (E + k) * np.sum(mag ** 2, axis=1)
        if amb is not None:
            eps = amb[sl].transpose(1, 0, 2).astype(float) @ aw
            bnd = bnd + np.sum(2 * av * eps + eps ** 2, axis=1)
            ambiguous[wdx] = np.sum(eps, axis=1) > 0
        bound[wdx] = bnd
    return p, bound, ambiguous


def probe_weights(seed: int) -> np.ndarray:
    """[B][E]: one-hot beams 0..47, seeded non-unit complex combinations 48..63."""
    rng = np.random.default_rng(seed)
    w = np.zeros((B, E), np.complex64)
    w[np.arange(N_ONEHOT), np.arange(N_ONEHOT)] = 1
    combo = rng.uniform(0.3, 1.7, (B - N_ONEHOT, E)) * np.exp(2j * np.pi * rng.random((B - N_ONEHOT, E)))
    combo[:, 9] = 0                                    # a disabled element in every combination
    w[N_ONEHOT:] = combo
    return w


def compare(got: np.ndarray, want: np.ndarray, bound: np.ndarray, ambiguous=None) -> dict:
    """Bound violations everywhere; the tolerance 1e-3 + 1e-5|P| where no rounding is
    ambiguous (elsewhere a legitimate float32 rounding flip changes P by whole codes)."""
    err = np.abs(got.astype(np.float64) - want)
    ratio = err / bound
    worst = np.unravel_index(int(np.argmax(ratio)), ratio.shape)
    plan = err > 1e-3 + 1e-5 * np.abs(want)
    clean = np.ones(err.shape, bool) if ambiguous is None else ~ambiguous
    return {"n": int(err.size), "violations": int(np.sum(err > bound)),
            "plan_tolerance_violations": int(np.sum(plan & clean)),
            "plan_tolerance_checked": int(np.sum(clean)),
            "ambiguous_outputs_bound_only": int(err.size - np.sum(clean)),
            "max_err": float(err.max()), "max_err_over_bound": float(ratio.max()),
            "worst_index": [int(i) for i in worst], "worst_want": float(want[worst]),
            "worst_got": float(got[worst]), "max_rel_err": float(np.max(err / np.maximum(want, 1)))}


# --------------------------------------------------------------------------- Kotekan

def run_kotekan(work: Path, frames, *, mode: str, t_raw: int, k: int, shard: int,
                weights_file: Path, depth: int = 3, sequences=None) -> list[np.ndarray]:
    inp, out = work / "in", work / "out"
    shutil.rmtree(work, ignore_errors=True)
    inp.mkdir(parents=True)
    out.mkdir(parents=True)
    sequences = sequences or [i * t_raw for i in range(len(frames))]
    for i, (frame, seq) in enumerate(zip(frames, sequences)):
        write_frame(inp / f"voltage.shard{shard}_{i:07d}.bin", frame, seq, shard)
    yaml_path = create_beamformer_yaml(
        work / "run.yaml", inp, f"voltage.shard{shard}", out, f"intensity.shard{shard}",
        len(frames), samples_per_data_set=t_raw, integration_spectra=k, input_mode=mode,
        output_format="float32", shard_id=shard, buffer_depth=depth, weights_file=weights_file)
    rc = execute_kotekan(yaml_path, BINARY, work / "kotekan.log", timeout_s=600)
    return read_outputs(out, shard, len(frames), rc, work, mode, t_raw, k)


def read_outputs(out, shard, n_frames, rc, work, mode, t_raw, k):
    files = sorted(out.glob(f"intensity.shard{shard}_*.bin"))
    if len(files) != n_frames:
        raise RuntimeError(f"rc={rc}; {len(files)} of {n_frames} outputs; see {work}/kotekan.log")
    n_spec, nf = (t_raw // U, FF) if mode == "fine" else (t_raw, F)
    shape = (-(-n_spec // k), nf, B)
    powers = []
    for i, path in enumerate(files):
        meta, power = read_product(path, "float32", shape)
        powers.append(power)
    return powers


def run_two_shards(work: Path, frames_by_shard, *, t_raw: int, weights_files):
    """Both shards concurrently in one Kotekan instance (fine, K=10, continuous frames)."""
    shutil.rmtree(work, ignore_errors=True)
    merged = {}
    n_frames = len(frames_by_shard[0])
    for shard, frames in enumerate(frames_by_shard):
        inp, out = work / f"in{shard}", work / "out"
        inp.mkdir(parents=True, exist_ok=True)
        out.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(frames):
            write_frame(inp / f"voltage.shard{shard}_{i:07d}.bin", frame, i * t_raw, shard)
        path = create_beamformer_yaml(
            work / f"shard{shard}.yaml", inp, f"voltage.shard{shard}", out,
            f"intensity.shard{shard}", n_frames, samples_per_data_set=t_raw,
            integration_spectra=10, input_mode="fine", output_format="float32",
            shard_id=shard, weights_file=weights_files[shard])
        cfg = yaml.safe_load(path.read_text())
        ren = {name: f"{name}_s{shard}" for name in
               ("voltage_buf", "intensity_buf", "read_voltage", "gpu_beamformer", "write_intensity")}
        for key, value in cfg.items():
            if key in ren:
                merged[ren[key]] = value
            else:
                merged.setdefault(key, value)
        merged[ren["read_voltage"]]["buf"] = ren["voltage_buf"]
        merged[ren["gpu_beamformer"]]["in_buffers"]["raw"] = ren["voltage_buf"]
        merged[ren["gpu_beamformer"]]["out_buffers"]["intensity"] = ren["intensity_buf"]
        merged[ren["write_intensity"]]["in_buf"] = ren["intensity_buf"]
    path = work / "run.yaml"
    path.write_text(yaml.safe_dump(merged, sort_keys=False))
    rc = execute_kotekan(path, BINARY, work / "kotekan.log", timeout_s=600)
    return [read_outputs(work / "out", s, n_frames, rc, work, "fine", t_raw, 10) for s in (0, 1)]


def write_weights(path: Path, w_bfe: np.ndarray) -> Path:
    with open(path, "wb") as fh:
        for b in range(w_bfe.shape[0]):
            np.ascontiguousarray(w_bfe[b], np.complex64).tofile(fh)
    return path


def record(name: str, payload: dict) -> None:
    if RESULTS:
        (RESULTS / f"{name}.json").write_text(json.dumps(payload, indent=2) + "\n")


class GpuCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not BINARY.is_file():
            raise unittest.SkipTest(f"no kotekan binary at {BINARY}")
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()


# =========================================================================== 1. codec/layout

def coarse_fixture(t_raw: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(t_raw)[:, None, None]
    f = np.arange(F)[None, :, None]
    e = np.arange(E)[None, None, :]
    raw = ((t * 37 + f * 11 + e * 101 + (t * f) % 7) % 256).astype(np.uint8)
    raw[:, 0] = 0                                                  # zero channel
    raw[:, 1] = 0x77                                               # constant (7, 7)
    raw[:, 2] = 0x88                                               # constant (-8, -8)
    raw[:, 3] = 0
    raw[min(5, t_raw - 1), 3, 17] = 0x3D                           # one-hot (-3 + 3i)
    raw[:, 4] = rng.choice(np.array([0x87, 0x78, 0x88, 0x77], np.uint8), (t_raw, E))
    raw[:, F - 1] = rng.integers(0, 256, (t_raw, E), dtype=np.uint8)
    return raw


def coarse_weights(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w = (rng.normal(size=(B, F, E)) + 1j * rng.normal(size=(B, F, E))).astype(np.complex64)
    w[:, :, 9] = 0                                                 # disabled element
    w[:8] = 0
    for b, el in enumerate((0, 1, 17, 31, 32, 62, 63, 40)):        # one-hot beams at boundaries
        w[b, :, el] = 1
    w[B - 1] *= 3.0                                                # non-unit last beam
    return w


class CodecLayoutGpu(GpuCase):
    """all 256 byte patterns, zero/constant/one-hot/extremes, K and tails."""

    def check(self, name, t_raw, k, shard):
        raw = coarse_fixture(t_raw, seed=t_raw + shard)
        self.assertEqual(len(np.unique(raw)), 256)
        w = coarse_weights(seed=7 + shard)
        wfile = write_weights(self.root / f"{name}_w.bin", w)
        got = run_kotekan(self.root / name, [raw], mode="coarse", t_raw=t_raw, k=k,
                          shard=shard, weights_file=wfile)[0]
        want, bound, _ = expected_power(decode(raw), None, w, k)
        m = compare(got, want, bound)
        m.update({"t_raw": t_raw, "K": k, "shard": shard, "windows": int(got.shape[0]),
                  "tail_spectra": int(t_raw - (got.shape[0] - 1) * k),
                  "zero_channel_max": float(got[:, 0].max()),
                  "byte_patterns": int(len(np.unique(raw)))})
        record(f"c1_{name}", m)
        self.assertEqual(m["violations"], 0, m)
        self.assertEqual(m["zero_channel_max"], 0.0)
        self.assertEqual(m["plan_tolerance_violations"], 0, m)
        # One-hot beams on integer inputs are exact integers in float32.
        self.assertTrue(np.array_equal(got[:, :, :8], want[:, :, :8].astype(np.float32)))

    def test_k320_partial_tail(self):
        self.check("k320_t650", 650, 320, 0)

    def test_k10_partial_tail(self):
        self.check("k10_t25", 25, 10, 0)

    def test_t_less_than_k(self):
        self.check("k320_t200", 200, 320, 1)

    def test_k10_shard1(self):
        self.check("k10_t333_s1", 333, 10, 1)


# =========================================================================== 2/3/6. PFB

TONE_BINS = (0, 15, 16, 31, 19)        # 19: the +3.5 dnu/32 tone


def fine_fixture(t_raw: int, seed: int, n0: int = 0,
                 frame_seed: int | None = None) -> tuple[np.ndarray, dict]:
    """[T][336][64] packed input; channel group = f % 12. n0 offsets the global sample index
    (tones are continuous across frames); frame_seed draws the random/extreme channels per
    frame so consecutive frames differ."""
    rng = np.random.default_rng(seed)
    frame_rng = rng if frame_seed is None else np.random.default_rng(frame_seed)
    n = (np.arange(t_raw) + n0)[:, None]
    raw = np.zeros((t_raw, F, E), np.uint8)
    phases = 2 * np.pi * rng.random((F, E))
    amp = rng.uniform(4.5, 5.5, (F, E))
    info = {}
    for f in range(F):
        g = f % 12

        def tone(offset_bins, a=amp[f]):
            return a[None, :] * np.exp(1j * (2 * np.pi * offset_bins * n / 32 + phases[f][None, :]))

        if g < 5:
            u0 = TONE_BINS[g]
            raw[:, f] = quantize_input(tone(u0 - 15.5))
            info[f] = ("center", u0)
        elif g == 5:
            raw[:, f] = quantize_input(tone(19.5 - 15.5))                # between 19 and 20
            info[f] = ("between", 19.5)
        elif g == 6:
            # Off-grid: 3.3125 - 15.5 bins is not a whole number of cycles per 640 or 1280.
            raw[:, f] = quantize_input(tone(3.3125 - 15.5))
            info[f] = ("offgrid", 3.3125)
        elif g == 7:
            raw[:, f] = quantize_input(tone(8 - 15.5, np.full(E, 7.4)))  # fine output clips
            info[f] = ("clip", 8)
        elif g == 8:
            raw[:, f] = frame_rng.choice(np.array([0x88, 0x77, 0x78, 0x87], np.uint8), (t_raw, E))
            info[f] = ("extremes", None)
        elif g == 9:
            raw[:, f] = frame_rng.integers(0, 256, (t_raw, E), dtype=np.uint8)
            info[f] = ("random", None)
        elif g == 10:
            z = tone(24 - 15.5)
            # Starts at n=320: window 0 (spectra 0..9) ends its support exactly at n=319.
            z[(n[:, 0] < 320)] = 0
            raw[:, f] = quantize_input(z)
            info[f] = ("late_start", 24)
        else:
            raw[:, f] = encode(np.full((t_raw, E), 3), np.full((t_raw, E), 2))   # DC 3+2i
            info[f] = ("dc", 15.5)
    return raw, info


class PfbCoefficients(unittest.TestCase):
    """prototype values, symmetry and sum; C++ table against the formula."""

    def test_prototype_formula(self):
        h = prototype()
        self.assertTrue(np.allclose(h, h[::-1], atol=0, rtol=0))
        self.assertAlmostEqual(float(h.sum()), 1.02061496, places=8)
        self.assertAlmostEqual(float(h[63]), 0.031233014, places=9)
        self.assertAlmostEqual(float(h[0]), -2.91355805e-07, places=14)
        self.assertEqual(int(np.argmax(h)), 63)
        record("c2_prototype", {"sum": float(h.sum()), "sum_sq": float(np.sum(h ** 2)),
                                "h0": float(h[0]), "h63": float(h[63]),
                                "overlap_corr_lag1": float(np.sum(h[32:] * h[:96]) / np.sum(h ** 2))})

    def test_cpp_table_matches_formula(self):
        cxx = shutil.which("g++")
        inc = [p for p in ("/usr/local/cuda/include",) if Path(p, "cuda_runtime.h").is_file()]
        if not cxx or not inc:
            self.skipTest("g++ or cuda_runtime.h not available")
        with tempfile.TemporaryDirectory() as d:
            src = Path(d, "h.cpp")
            src.write_text('#include "cudaChartsUpchannelizer.hpp"\n#include <cstdio>\n'
                           "int main(){for(std::size_t s=0;s<128;++s)"
                           'std::printf("%.17g\\n",(double)beamformer::sinc_hanning_pfb_weight(s));}\n')
            exe = Path(d, "h")
            subprocess.run([cxx, "-std=c++17", f"-I{ROOT / 'lib/cuda'}", f"-I{inc[0]}",
                            str(src), "-o", str(exe)], check=True)
            table = np.array([float(v) for v in subprocess.run(
                [str(exe)], check=True, capture_output=True, text=True).stdout.split()])
        ref = prototype().astype(np.float32).astype(np.float64)
        ulps = np.abs(table - ref) / np.spacing(np.abs(ref).astype(np.float32)).astype(np.float64)
        record("c2_cpp_table", {"max_float32_ulps_vs_formula": float(ulps.max()),
                                "exact_float32_matches": int(np.sum(ulps == 0))})
        self.assertLessEqual(float(ulps.max()), 1.0)


class PfbToneGpu(GpuCase):
    """fine U=32 output against the reference fine transform, one frame per shard."""

    T_RAW = 1280      # 40 spectra, 4 windows; window 0 holds startup spectra 0..2

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.w = probe_weights(seed=11)
        cls.wfile = write_weights(cls.root / "fine_w.bin",
                                  np.broadcast_to(cls.w[:, None, :], (B, FF, E)))
        cls.runs = {}

    def fixture(self, shard):
        return fine_fixture(self.T_RAW, seed=100 + shard)

    def run_shard(self, shard):
        if shard not in self.runs:
            raw, info = self.fixture(shard)
            got = run_kotekan(self.root / f"fine{shard}", [raw], mode="fine", t_raw=self.T_RAW,
                              k=10, shard=shard, weights_file=self.wfile)[0]
            q, amb = fine_quantized(raw, None)
            want, bound, ambig = expected_power(q, amb, self.w, 10)
            self.runs[shard] = (raw, info, got, want, bound, ambig, q, amb)
        return self.runs[shard]

    def check_shard(self, shard):
        raw, info, got, want, bound, ambig, q, amb = self.run_shard(shard)
        m = compare(got, want, bound, ambig)
        grid = got.reshape(-1, F, U, B)
        steady = grid[1:]                                              # windows 1..3
        loc, between, dc = [], [], []
        for f, (kind, u0) in info.items():
            onehot = steady[:, f, :, :N_ONEHOT]                        # [W][U][48]
            if kind == "center":
                loc.append(np.mean(np.argmax(onehot, axis=1) == u0))
            elif kind == "between":
                top2 = np.sort(np.argsort(onehot, axis=1)[:, -2:, :], axis=1)
                between.append(np.mean((top2[:, 0] == 19) & (top2[:, 1] == 20)))
            elif kind == "dc":
                dc.append(np.mean(np.isin(np.argmax(onehot, axis=1), (15, 16))))
        late = [f for f, (kind, _) in info.items() if kind == "late_start"]
        clipped = [f for f, (kind, _) in info.items() if kind == "clip"]
        self.assertEqual(q.shape[0] % 10, 0)
        exact_mask = ~ambig[..., :N_ONEHOT]
        exact_equal = bool(np.array_equal(got[..., :N_ONEHOT][exact_mask],
                                          want[..., :N_ONEHOT][exact_mask].astype(np.float32)))
        qc = q.reshape(-1, F, U, E)
        # Leakage of center tones into adjacent bins, predicted from the actual 128-tap filter
        # with quantization (the expectation) and measured on the GPU, steady windows.
        centers = [f for f, (kind, _) in info.items() if kind == "center" and 0 < info[f][1] < 31]
        wgrid = want.reshape(-1, F, U, B)[1:]

        def adjacent_ratio(g):
            main = np.array([g[:, f, info[f][1], :N_ONEHOT] for f in centers])
            adj = np.array([np.maximum(g[:, f, info[f][1] - 1, :N_ONEHOT],
                                       g[:, f, info[f][1] + 1, :N_ONEHOT]) for f in centers])
            return float((adj / main).max())

        h = prototype()
        sx = np.arange(128)
        m.update({
            "shard": shard,
            "ambiguous_components": int(amb.sum()),
            "ambiguous_fraction": float(amb.mean() / 2),
            "onehot_unambiguous_exact": exact_equal,
            "tone_center_argmax_fraction": float(np.mean(loc)),
            "tone_between_top2_19_20_fraction": float(np.mean(between)),
            "dc_argmax_15_16_fraction": float(np.mean(dc)),
            "late_start_window0_max": float(grid[0, late].max()),
            "late_start_window1_min_onehot": float(grid[1, late][..., :N_ONEHOT].sum(axis=1).min()),
            # Fixture coverage from the expectation (not a GPU measurement):
            "reference_clipped_components": int(np.sum(np.abs(qc[:, clipped].real) == 7)
                                                + np.sum(np.abs(qc[:, clipped].imag) == 7)),
            "onehot_max_per_spectrum": float(got[..., :N_ONEHOT].max() / 10),
            "adjacent_over_main_measured": adjacent_ratio(steady),
            "adjacent_over_main_predicted_quantized": adjacent_ratio(wgrid),
            "adjacent_over_main_predicted_unquantized": float(
                abs(np.sum(h * np.exp(2j * np.pi * sx / 32))) ** 2 / np.sum(h) ** 2),
            "rounding_ties_to_even": "not exercised: exact half-integers do not occur",
        })
        record(f"c2_fine_tones_shard{shard}", m)

        self.assertEqual(m["violations"], 0, m)
        self.assertEqual(m["plan_tolerance_violations"], 0, m)
        self.assertTrue(exact_equal)
        self.assertEqual(m["tone_center_argmax_fraction"], 1.0, m)
        self.assertEqual(m["tone_between_top2_19_20_fraction"], 1.0, m)
        self.assertEqual(m["dc_argmax_15_16_fraction"], 1.0, m)
        self.assertEqual(m["late_start_window0_max"], 0.0, m)
        self.assertGreater(m["late_start_window1_min_onehot"], 0.0, m)
        self.assertGreater(m["reference_clipped_components"], 0)
        self.assertLessEqual(m["onehot_max_per_spectrum"], 98.0)
        self.assertEqual(m["adjacent_over_main_measured"], m["adjacent_over_main_predicted_quantized"])

    def test_fine_tones_shard0(self):
        self.check_shard(0)

    def test_fine_tones_shard1(self):
        self.check_shard(1)

    def test_negative_controls(self):
        """The same comparison must fail for a mirrored bin order, a branch implementation
        without the (-1)^j tap sign, and a support shifted by one raw sample."""
        raw, info, got, want, bound, ambig, q, amb = self.run_shard(0)
        mirrored = got.reshape(-1, F, U, B)[:, :, ::-1].reshape(got.shape)
        counts = {"mirrored_bins": compare(mirrored, want, bound)["violations"]}
        for label, kwargs in (("no_alternating_sign", {"alternating_sign": False}),
                              ("support_shift_1", {"shift": 1})):
            w2, b2, _ = expected_power(*fine_quantized(raw, None, **kwargs), self.w, 10)
            counts[label] = compare(got, w2, b2)["violations"]
        record("c2_negative_controls", counts)
        for label, n in counts.items():
            self.assertGreater(n, 1000, (label, counts))


class TwoShardHistoryGpu(GpuCase):
    """both shards concurrently in one Kotekan instance, three continuous
    frames each with distinct content and distinct per-shard weights, against the reference fine transform."""

    T_RAW = 640
    N_FRAMES = 3

    def test_two_shards_concurrently(self):
        weights = [probe_weights(seed=21 + s) for s in (0, 1)]
        wfiles = [write_weights(self.root / f"w{s}.bin",
                                np.broadcast_to(weights[s][:, None, :], (B, FF, E))) for s in (0, 1)]
        frames = [[fine_fixture(self.T_RAW, seed=400 + s, n0=i * self.T_RAW,
                                frame_seed=4000 + 10 * s + i)[0] for i in range(self.N_FRAMES)]
                  for s in (0, 1)]
        got = run_two_shards(self.root / "both", frames, t_raw=self.T_RAW, weights_files=wfiles)
        report = {"frames_per_shard": self.N_FRAMES, "violations": {}, "plan_tolerance_violations": {},
                  "swapped_shard_violations": {}}
        for s in (0, 1):
            viol, plan, swapped = [], [], []
            for i in range(self.N_FRAMES):
                history = frames[s][i - 1][-96:] if i else None
                q, amb = fine_quantized(frames[s][i], history)
                m = compare(got[s][i], *expected_power(q, amb, weights[s], 10))
                viol.append(m["violations"])
                plan.append(m["plan_tolerance_violations"])
                # Sensitivity: the other shard's weights must not explain this output.
                swapped.append(compare(got[s][i], *expected_power(q, amb, weights[1 - s], 10))["violations"])
            report["violations"][s] = viol
            report["plan_tolerance_violations"][s] = plan
            report["swapped_shard_violations"][s] = swapped
        record("c7_two_shards_concurrent", report)
        for s in (0, 1):
            self.assertEqual(sum(report["violations"][s]), 0, report)
            self.assertEqual(sum(report["plan_tolerance_violations"][s]), 0, report)
            self.assertTrue(all(v > 1000 for v in report["swapped_shard_violations"][s]), report)


# =========================================================================== 4. history

class HistoryRingGpu(GpuCase):
    """continuous distinct frames, repeated ring wrap, a sequence jump."""

    T_RAW = 640
    N_FRAMES = 9

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.w = probe_weights(seed=12)
        cls.wfile = write_weights(cls.root / "fine_w.bin",
                                  np.broadcast_to(cls.w[:, None, :], (B, FF, E)))

    def frames(self, seed):
        return [fine_fixture(self.T_RAW, seed=seed, n0=i * self.T_RAW, frame_seed=seed * 100 + i)[0]
                for i in range(self.N_FRAMES)]

    def violations(self, got, frame, history):
        q, amb = fine_quantized(frame, history)
        return compare(got, *expected_power(q, amb, self.w, 10))

    def check(self, name, depth, jump_at=None):
        frames = self.frames(seed=200 + depth)
        self.assertFalse(any(np.array_equal(frames[i], frames[i + 1])
                             for i in range(self.N_FRAMES - 1)))
        seqs = [i * self.T_RAW + (self.T_RAW if jump_at is not None and i >= jump_at else 0)
                for i in range(self.N_FRAMES)]
        got = run_kotekan(self.root / name, frames, mode="fine", t_raw=self.T_RAW, k=10,
                          shard=0, weights_file=self.wfile, depth=depth, sequences=seqs)
        resets = {0} | ({jump_at} if jump_at is not None else set())
        per_frame, wrong = [], {"no_history": [], "own_tail": [], "frame_i_minus_2": [],
                                "continuous_at_jump": []}
        for i in range(self.N_FRAMES):
            history = None if i in resets else frames[i - 1][-96:]
            per_frame.append(self.violations(got[i], frames[i], history))
            # Wrong-history hypotheses (stale slot, ordering race, missing reset) must fail.
            if i not in resets:
                wrong["no_history"].append(self.violations(got[i], frames[i], None)["violations"])
                wrong["own_tail"].append(
                    self.violations(got[i], frames[i], frames[i][-96:])["violations"])
                if i >= 2 and i - 1 not in resets:
                    wrong["frame_i_minus_2"].append(
                        self.violations(got[i], frames[i], frames[i - 2][-96:])["violations"])
            elif i == jump_at:
                wrong["continuous_at_jump"].append(
                    self.violations(got[i], frames[i], frames[i - 1][-96:])["violations"])
        report = {"depth": depth, "frames": self.N_FRAMES, "wraps": self.N_FRAMES // depth,
                  "sequences": seqs, "jump_at": jump_at,
                  "violations_per_frame": [p["violations"] for p in per_frame],
                  "plan_tolerance_violations_per_frame": [p["plan_tolerance_violations"] for p in per_frame],
                  "max_err_over_bound_per_frame": [p["max_err_over_bound"] for p in per_frame],
                  "wrong_hypothesis_violations": wrong}
        record(f"c4_{name}", report)
        self.assertEqual(sum(report["violations_per_frame"]), 0, report)
        for label, counts in wrong.items():
            if label == "continuous_at_jump" and jump_at is None:
                continue
            self.assertTrue(counts and all(c > 0 for c in counts), (label, report))

    def test_continuous_depth2(self):
        self.check("continuous_depth2", 2)

    def test_continuous_depth4(self):
        self.check("continuous_depth4", 4)

    def test_sequence_jump_resets_history(self):
        self.check("jump_depth3", 3, jump_at=5)


# =========================================================================== 7. noise

def normal_cdf(x):
    return 0.5 * (1 + np.vectorize(math.erf)(np.asarray(x) / math.sqrt(2)))


def rounded_gaussian_pmf(sigma: float):
    k = np.arange(-8, 8)
    lo = np.where(k == -8, -np.inf, k - 0.5)
    hi = np.where(k == 7, np.inf, k + 0.5)
    return k, normal_cdf(hi / sigma) - normal_cdf(lo / sigma)


def quantized_square_moments(sigma: float, n_max: int = 12):
    """For g ~ N(0, sigma^2) and Q = round_half_even(clamp(g, -7, 7))^2: mean, variance and
    the Hermite coefficients c_n = E[Q He_n(g/sigma)] used for Cov at correlation r
    (Mehler: Cov = sum_{n even >= 2} r^n c_n^2 / n!)."""
    z = np.linspace(-12, 12, 400001)
    dz = z[1] - z[0]
    pdf = np.exp(-z ** 2 / 2) / math.sqrt(2 * math.pi)
    qv = np.rint(np.clip(sigma * z, -CLAMP, CLAMP)) ** 2
    he = [np.ones_like(z), z]
    for n in range(2, n_max + 1):
        he.append(z * he[-1] - (n - 1) * he[-2])
    mean = float(np.sum(qv * pdf) * dz)
    var = float(np.sum(qv ** 2 * pdf) * dz) - mean ** 2
    c = [float(np.sum(qv * hn * pdf) * dz) for hn in he]
    return mean, var, c


def cov_q2(r: float, c) -> float:
    return sum(r ** n * c[n] ** 2 / math.factorial(n) for n in range(2, len(c), 2))


class NoiseStatisticsGpu(GpuCase):
    """seeded noise, quantization-aware mean and covariance-aware K_eff."""

    T_RAW = 15360
    SIGMA_IN = 3.0

    def check(self, shard):
        rng = np.random.default_rng(300 + shard)
        k, pmf = rounded_gaussian_pmf(self.SIGMA_IN)
        cdf = np.cumsum(pmf / pmf.sum())
        cdf[-1] = 1.0
        raw = np.empty((self.T_RAW, F, E), np.uint8)
        for t0 in range(0, self.T_RAW, 512):
            idx = np.searchsorted(cdf, rng.random((2, 512, F, E), np.float32), side="right")
            comp = k[np.minimum(idx, k.size - 1)]
            raw[t0:t0 + 512] = encode(comp[0], comp[1])
        w = np.zeros((B, E), np.complex64)
        w[np.arange(N_ONEHOT), np.arange(N_ONEHOT)] = 1
        wfile = self.root / "noise_w.bin"
        if not wfile.exists():
            write_weights(wfile, np.broadcast_to(w[:, None, :], (B, FF, E)))
        got = run_kotekan(self.root / f"noise{shard}", [raw], mode="fine", t_raw=self.T_RAW,
                          k=10, shard=shard, weights_file=wfile)[0]
        s = got[1:, :, :N_ONEHOT].astype(np.float64)                  # steady windows [47][FF][48]

        var_in = float(np.sum(pmf * k ** 2) - np.sum(pmf * k) ** 2)
        h = prototype()
        sigma_y = math.sqrt(var_in * np.sum(h ** 2))
        mean_q2, var_q2, c = quantized_square_moments(sigma_y)
        r = [1.0] + [(-1) ** d * float(np.sum(h[32 * d:] * h[:128 - 32 * d]) / np.sum(h ** 2))
                     for d in (1, 2, 3)]
        kk = 10
        cov_p = [2 * var_q2] + [2 * cov_q2(r[d], c) for d in (1, 2, 3)]
        var_s = sum(cov_p[0] if a == b else (cov_p[abs(a - b)] if abs(a - b) <= 3 else 0.0)
                    for a in range(kk) for b in range(kk))
        mu_s = kk * 2 * mean_q2
        k_eff = mu_s ** 2 / var_s
        k_naive = mu_s ** 2 / (kk * cov_p[0])

        # Independent groups: distinct (coarse channel, element) pairs.
        groups = s.reshape(s.shape[0], F, U, N_ONEHOT)
        g_mean = groups.mean(axis=(0, 2))                                # [F][48]
        g_var = groups.var(axis=(0, 2), ddof=1)
        n_groups = g_mean.size
        mean_hat, mean_se = float(g_mean.mean()), float(g_mean.std(ddof=1) / math.sqrt(n_groups))
        var_hat, var_se = float(g_var.mean()), float(g_var.std(ddof=1) / math.sqrt(n_groups))
        k_eff_hat = mean_hat ** 2 / var_hat
        report = {
            "shard": shard, "sigma_in": self.SIGMA_IN, "var_in_per_component": var_in,
            "sigma_fine_per_component": sigma_y, "lag_correlation": r[1:],
            "mean_window_predicted": mu_s, "mean_window_measured": mean_hat,
            "mean_window_se": mean_se, "var_window_predicted": var_s,
            "var_window_measured": var_hat, "var_window_se": var_se,
            "k_eff_predicted": k_eff, "k_naive_independent": k_naive,
            "k_eff_measured": k_eff_hat,
            "mean_rel_dev": mean_hat / mu_s - 1, "var_rel_dev": var_hat / var_s - 1,
            # Significance against the statistical error only; nonzero z here is model error
            # (non-Gaussian fine voltage, -8 endpoint input mean), not sampling noise.
            "mean_z": (mean_hat - mu_s) / mean_se, "var_z": (var_hat - var_s) / var_se,
            "naive_var_z": (var_hat - kk * cov_p[0]) / var_se,
            "naive_var_rel_dev": var_hat / (kk * cov_p[0]) - 1,
            "groups": n_groups, "windows": int(s.shape[0]),
        }
        record(f"c7_noise_shard{shard}", report)
        # Gaussian model of the fine voltage (CLT over 128 taps): 1% is a model tolerance; the
        # measured deviations are statistically significant (see mean_z/var_z), so this is
        # agreement with a Gaussian approximation, not within the standard error. The independent-spectra (naive K) variance must be
        # rejected by the same data, so overlap covariance is actually being tested.
        self.assertLess(abs(report["mean_rel_dev"]), 0.01, report)
        self.assertLess(abs(report["var_rel_dev"]), 0.01, report)
        self.assertGreater(abs(report["naive_var_rel_dev"]), 0.02, report)
        self.assertGreater(mean_hat / mean_se, 1000)

    def test_noise_shard0(self):
        self.check(0)

    def test_noise_shard1(self):
        self.check(1)


def main():
    global RESULTS, BINARY
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--kotekan-bin", type=Path)
    args, rest = parser.parse_known_args()
    if args.results_dir:
        RESULTS = args.results_dir
        RESULTS.mkdir(parents=True, exist_ok=True)
    if args.kotekan_bin:
        BINARY = args.kotekan_bin
    unittest.main(argv=[sys.argv[0], *rest], verbosity=2)


if __name__ == "__main__":
    main()
