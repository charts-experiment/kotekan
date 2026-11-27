"""Tests for Wave 4C: Solar Radio Burst guard (RFI-3.4) & mutual coupling / az-tilt self-cal (CAL-2.4)."""

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

from charts.self_tune_tracker import evaluate_mutual_coupling_diagnostics, update_managed_notches
from charts.solar_burst_guard import classify_solar_radio_burst
from charts.spatial_calibration import (
    BeamSpaceDelayCalibrator,
    array_factor_power,
    compute_mutual_coupling_noise,
    decompose_tilt_and_closure,
    fit_array_azimuth_and_tilt,
)


def _synth_solar_burst(
    drift_mhz_s: float, sun_amp: float = 25.0, off_amp: float = 0.0, n_t: int = 80, n_f: int = 120
):
    rng = np.random.default_rng(404)
    freqs = np.linspace(300.0, 355.0, n_f)
    times = np.linspace(0.0, 8.0, n_t)
    sun = rng.normal(100.0, 1.0, size=(n_t, n_f))
    off = rng.normal(100.0, 1.0, size=(n_t, n_f))
    if abs(drift_mhz_s) > 0.01:
        f_mid = 345.0
        for fi, f in enumerate(freqs):
            if 310.0 <= f <= 350.0:
                t_c = 1.5 + (f - f_mid) / drift_mhz_s
                prof = np.exp(-0.5 * ((times - t_c) / 0.25) ** 2)
                sun[:, fi] += sun_amp * prof
                off[:, fi] += off_amp * prof
    else:
        # Stationary horizon RFI across narrow sub-band
        prof = np.exp(-0.5 * ((times - 4.0) / 0.5) ** 2)
        for fi, f in enumerate(freqs):
            if 320.0 <= f <= 325.0:
                sun[:, fi] += sun_amp * prof
                off[:, fi] += sun_amp * prof
    return sun, off, freqs, times


def test_solar_radio_burst_guard_and_notch_protection() -> None:
    # 1. Type III fast negative drift (-25 MHz/s) during daytime
    sun3, off3, f3, t3 = _synth_solar_burst(-12.0, sun_amp=30.0, off_amp=1.0)
    res3 = classify_solar_radio_burst(sun3, off3, f3, t3, sun_el_deg=45.0)
    assert res3["is_solar_burst"] is True
    assert res3["classification"] == "SOLAR_RADIO_BURST"
    assert res3["burst_type"] == "Type III"
    assert res3["drift_rate_mhz_s"] < -5.0
    assert len(res3["protected_ranges_mhz"]) == 1

    # 2. Type II slow negative drift (-2.0 MHz/s)
    sun2, off2, f2, t2 = _synth_solar_burst(-4.0, sun_amp=30.0, off_amp=0.5)
    # Use wider time span for slow drift across 40 MHz
    res2 = classify_solar_radio_burst(sun2, off2, f2, np.linspace(0.0, 16.0, t2.size), sun_el_deg=35.0)
    assert res2["is_solar_burst"] is True
    assert res2["burst_type"] == "Type II"

    # 3. Night-time (sun_el <= 10) and stationary horizon RFI must NOT trigger solar protection
    res_night = classify_solar_radio_burst(sun3, off3, f3, t3, sun_el_deg=-5.0)
    assert res_night["is_solar_burst"] is False
    sun_rfi, off_rfi, fr, tr = _synth_solar_burst(0.0, sun_amp=30.0, off_amp=30.0)
    res_rfi = classify_solar_radio_burst(sun_rfi, off_rfi, fr, tr, sun_el_deg=45.0)
    assert res_rfi["is_solar_burst"] is False

    # 4. update_managed_notches must NOT auto-notch inside solar_protected_ranges_mhz
    channels = [
        {"freq_mhz": 330.0, "subband": 0, "flagged_chunk_fraction": 0.95, "mean_sk": 3.5, "freq_mad_z": 9.0, "flag_bits": 0},
        {"freq_mhz": 370.0, "subband": 0, "flagged_chunk_fraction": 0.95, "mean_sk": 3.5, "freq_mad_z": 9.0, "flag_bits": 0},
    ]
    bad_streaks: dict[float, int] = {}
    clean_streaks: dict[float, int] = {}
    managed: set[float] = set()
    for _ in range(6):
        notches, _, _ = update_managed_notches(
            channels,
            None,
            [],
            bad_streaks,
            clean_streaks,
            managed,
            promote_streak=5,
            solar_protected_ranges_mhz=[[310.0, 350.0]],
        )
    assert 370.0 in managed
    assert 330.0 not in managed


def test_mutual_coupling_noise_and_array_az_tilt_cal() -> None:
    # 1. Exact analytical mutual-coupling diagnostics across 64 antennas, 8 channels
    n_ant = 64
    n_freq = 8
    covs = np.zeros((n_freq, n_ant, n_ant), dtype=np.complex128)
    w = np.full(n_ant, 1.0 / math.sqrt(n_ant), dtype=np.complex128)
    for f in range(n_freq):
        mat = np.eye(n_ant, dtype=np.complex128)
        # Uniform off-diagonal coupling sum -> w^H (R - I) w = 0.08 * (1 + delta)
        off_val = 0.08 / float(n_ant - 1) if f != 5 else 0.35 / float(n_ant - 1)
        mat += off_val * (np.ones((n_ant, n_ant)) - np.eye(n_ant))
        covs[f] = mat

    mc = compute_mutual_coupling_noise(
        covs, weights=w, t_rec_k=50.0, t_loss_k=5.0, t_sky_k=20.0, resonant_threshold_k=20.0
    )
    # Channel 0: off_sum = 0.08, total = 1.08 => p_mc = 0.08 / 1.08, T_mc = (0.08 / 1.0) * 75 = 6.0 K
    assert abs(float(mc["p_mc"][0]) - (0.08 / 1.08)) < 1.0e-10
    assert abs(float(mc["t_mc_k"][0]) - 6.0) < 1.0e-9
    # Channel 5 resonant spike: T_mc = 0.35 * 75 = 26.25 K > 20 K
    assert mc["resonant_spike_indices"] == [5]

    # 2. Planar tilt & azimuth rotation recovery (BURSTT 3.0 deg offset + planar slope)
    xs = []
    ys = []
    for row in range(8):
        for col in range(8):
            xs.append(-col * 0.6)
            ys.append(row * 0.6)
    pos = np.column_stack([xs, ys, np.zeros(n_ant)])

    c_m_s = 299792458.0
    dl_true, dm_true = 0.015, -0.012
    delays_ns = 0.25 + ((pos[:, 0] * dl_true + pos[:, 1] * dm_true) / c_m_s) * 1.0e9
    dl_rec, dm_rec, closure_ns = decompose_tilt_and_closure(pos, delays_ns)
    assert abs(dl_rec - dl_true) < 1.0e-4
    assert abs(dm_rec - dm_true) < 1.0e-4
    assert closure_ns < 1.0e-6

    exp_lm = np.array([[0.3, 0.1], [-0.25, 0.2], [0.1, -0.35], [-0.15, -0.2]], dtype=np.float64)
    rot_true_deg = 3.0
    th = math.radians(rot_true_deg)
    rot = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    meas_lm = exp_lm @ rot.T + np.array([[dl_true, dm_true]])

    geom = fit_array_azimuth_and_tilt(pos, delays_ns=delays_ns, measured_lm=meas_lm, expected_lm=exp_lm)
    assert abs(geom["azimuth_rot_deg"] - rot_true_deg) < 0.05
    assert abs(geom["dl_tilt"] - dl_true) < 1.0e-4

    # 3. End-to-end helper + BeamSpaceDelayCalibrator convergence check on 4-ant subgrid
    diag = evaluate_mutual_coupling_diagnostics(covs, pos, delays_ns=delays_ns, measured_lm=meas_lm, expected_lm=exp_lm)
    assert diag["resonant_spike_indices"] == [5]
    assert abs(diag["geometry"]["azimuth_rot_deg"] - rot_true_deg) < 0.05

    sub_pos = np.array([[0.0, 0.0], [-1.8, 0.0], [0.0, 1.8], [-1.8, 1.8]], dtype=np.float64)
    freqs_cal = np.linspace(320.0, 480.0, 12)
    cal = BeamSpaceDelayCalibrator(sub_pos, freqs_mhz=freqs_cal)
    fan_l0 = np.array([[-0.08, 0.0, 0.08]], dtype=np.float64)
    fan_m0 = np.zeros_like(fan_l0)
    off_l0 = np.array([0.45], dtype=np.float64)
    off_m0 = np.array([0.45], dtype=np.float64)
    tau_inj = np.array([0.0, 0.6, -0.4, 0.2], dtype=np.float64) * 1.0e-9
    fan_p = 100.0 + 40.0 * array_factor_power(sub_pos, freqs_cal, np.array([[0.02]]), np.array([[0.01]]), fan_l0, fan_m0, tau_inj)
    off_p = 100.0 + 40.0 * array_factor_power(sub_pos, freqs_cal, np.array([0.02]), np.array([0.01]), off_l0, off_m0, tau_inj)
    dres = cal.fit(fan_p, off_p, np.array([0.02]), np.array([0.01]), fan_l0, fan_m0, off_l0, off_m0)
    assert dres.after_rmse < dres.before_rmse
