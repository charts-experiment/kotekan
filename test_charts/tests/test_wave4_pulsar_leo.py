"""Tests for Wave 4B: SVP lattice pulsar periodicity, FDAS, SGP4 LEO & beam holography."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

_KOTEKAN_ROOT = Path(__file__).resolve().parents[2]
_CHARTS_DIR = _KOTEKAN_ROOT / "charts"
for p in (str(_KOTEKAN_ROOT), str(_CHARTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from charts.leo_tracker import (
    Sgp4TopocentricPropagator,
    fit_beam_passage_envelope,
    lanczos8_resample,
    quadratic_doppler_matched_filter,
)
from charts.pulsar_svp_fdas import fdas_harmonic_search, lll_reduce, svp_lattice_period_search

# Representative ISS / LEO UHF TLE
_TEST_TLE1 = "1 25544U 98067A   26328.50000000  .00016717  00000-0  30250-3 0  9995"
_TEST_TLE2 = "2 25544  51.6400 200.0000 0004000 100.0000 260.0000 15.50000000 10000"


def test_lll_and_svp_lattice_period_recovery() -> None:
    rng = np.random.default_rng(101)
    # 1. Verify LLL shrinks off-orthogonal lattice vectors cleanly
    raw_lat = np.array([[1.0, 1.0, 1.0], [-1.0, 0.0, 2.0], [3.0, 5.0, 6.0]], dtype=np.float64)
    red = lll_reduce(raw_lat)
    assert np.linalg.norm(red[0]) <= np.linalg.norm(raw_lat[0]) + 1.0e-9

    # 2. Sparse noisy pulsar TOAs over 600 s at P0 = 0.089328385 s (Vela-like, f0 ~ 11.19465 Hz)
    f_true = 11.194650
    t_mid = 300.0
    # Sparse uneven integer pulse counts across 12 observing blocks
    k_true = np.array([0, 3, 19, 44, 102, 237, 411, 892, 1450, 2231, 3812, 6500], dtype=np.float64)
    k_true -= k_true[len(k_true) // 2]
    toas = t_mid + (k_true + 0.15) / f_true + rng.normal(0.0, 2.0e-5, size=k_true.size)

    res = svp_lattice_period_search(
        toas, f_min_hz=11.15, f_max_hz=11.24, sigma_toa_s=5.0e-5, include_fdot=False
    )
    assert abs(res["f_hz"] - f_true) < 1.0e-4
    assert res["phase_rms_cycles"] < 0.05


def test_fdas_harmonic_summing() -> None:
    rng = np.random.default_rng(202)
    dt_s = 0.0025
    n_t = 4096
    t = (np.arange(n_t, dtype=np.float64) - 0.5 * n_t) * dt_s
    f0 = 7.25
    fdot = 0.008
    phase = 2.0 * math.pi * (f0 * t + 0.5 * fdot * t * t)
    # Narrow duty-cycle pulse train: harmonics 1..4 add coherently
    pulse = np.zeros(n_t, dtype=np.float64)
    for h in (1, 2, 3, 4):
        pulse += 0.65 * np.cos(h * phase)
    ts = pulse + rng.normal(0.0, 2.5, size=n_t)

    res_k1 = fdas_harmonic_search(
        ts, dt_s=dt_s, f_min_hz=2.0, f_max_hz=20.0, fdot_grid_hz_s=[0.0, 0.008], max_harmonics=1
    )
    res_k4 = fdas_harmonic_search(
        ts, dt_s=dt_s, f_min_hz=2.0, f_max_hz=20.0, fdot_grid_hz_s=[0.0, 0.008], max_harmonics=4
    )
    assert abs(res_k4["f_hz"] - f0) < 0.15
    assert abs(res_k4["fdot_hz_s"] - fdot) < 1.0e-6
    assert res_k4["snr"] > 1.5 * res_k1["snr"]


def test_sgp4_leo_propagation_lanczos_and_holography() -> None:
    prop = Sgp4TopocentricPropagator(_TEST_TLE1, _TEST_TLE2, name="ISS")
    # Epoch near TLE 2026 day 328.5
    unix_t = 1795435200.0
    st = prop.propagate_unix(unix_t, carrier_hz=400.0e6)
    assert abs(st["l"] * st["l"] + st["m"] * st["m"] + st["n"] * st["n"] - 1.0) < 1.0e-9
    payload = prop.build_tle_target_payload(beam_id=1, unix_s=unix_t, frame_duration_s=0.0512)
    assert payload["norad_id"] == 25544 and payload["beam_id"] == 1
    assert "l0" in payload and "l1" in payload

    # 8-tap Lanczos Kronecker exactness at integers + fractional-delay accuracy on bandlimited sinusoid
    x = np.cos(2.0 * math.pi * 0.025 * np.arange(128))
    int_pts = np.arange(16.0, 90.0, 1.0)
    assert float(np.max(np.abs(lanczos8_resample(x, int_pts) - x[int_pts.astype(int)]))) < 1.0e-12
    pts = np.linspace(15.37, 90.37, 64)
    interp = lanczos8_resample(x, pts)
    truth = np.cos(2.0 * math.pi * 0.025 * pts)
    assert float(np.max(np.abs(interp - truth))) < 5.0e-3

    # Quadratic Doppler matched filter recovers C > 0.90
    dt_s = 1.0e-3
    t_ax = (np.arange(1024, dtype=np.float64) - 511.5) * dt_s
    sig = np.exp(1j * 2.0 * math.pi * (45.0 * t_ax - 0.5 * 32.0 * t_ax * t_ax))
    rng = np.random.default_rng(303)
    noisy = sig + 0.15 * (rng.normal(size=t_ax.size) + 1j * rng.normal(size=t_ax.size))
    qmf = quadratic_doppler_matched_filter(
        noisy, dt_s=dt_s, f_grid_hz=np.array([40.0, 45.0, 50.0]), fdot_grid_hz_s=np.array([-40.0, -32.0, 0.0])
    )
    assert qmf["coherent_fraction"] > 0.90
    assert abs(qmf["best_f_hz"] - 45.0) < 1.0e-6
    assert abs(qmf["best_fdot_hz_s"] + 32.0) < 1.0e-6

    # Beam-passage envelope holography recovers (dl, dm) within 0.002 across two crossing LEO passes
    s_par = np.linspace(-0.35, 0.35, 50)
    l_tr = np.concatenate([s_par * 0.8 + 0.05, -s_par * 0.6 + 0.08])
    m_tr = np.concatenate([s_par * 0.6 - 0.03, s_par * 0.8 - 0.01])
    dl_true, dm_true, w_true = 0.062, -0.024, 0.22
    p_true = 12.0 + 48.0 * np.exp(-4.0 * math.log(2.0) * ((l_tr - dl_true) ** 2 + (m_tr - dm_true) ** 2) / (w_true**2))
    p_noisy = p_true + rng.normal(0.0, 0.2, size=l_tr.size)
    holo = fit_beam_passage_envelope(l_tr, m_tr, p_noisy, fwhm_init_rad=0.20)
    assert abs(holo["dl_offset"] - dl_true) < 0.002
    assert abs(holo["dm_offset"] - dm_true) < 0.002
    assert abs(holo["fwhm_rad"] - w_true) < 0.005
