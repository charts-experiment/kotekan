"""Unit tests for Wave 2C RFI mitigation and Pudahuel operations (RFI-3.2..3.3, OPS-6.1..6.2)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
CHARTS_DIR = REPO_ROOT / "charts"
if str(CHARTS_DIR) not in sys.path:
    sys.path.insert(0, str(CHARTS_DIR))

from rfi_mitigation_tools import (  # noqa: E402
    AdaptiveSidelobeCanceller,
    CoherentCombCanceller,
    FREQS_MHZ,
    PUDAHUEL_KNOWN_INTERFERERS,
    array_factor_power,
    find_optimal_off_beams,
)
import self_tune_tracker as stt  # noqa: E402


def test_coherent_comb_canceller_suppresses_timing_and_mains_preserves_pulse():
    """RFI-3.2: Suppress 9.7656 Hz + 50.1 Hz common-mode combs while keeping ON-only pulse."""
    rng = np.random.default_rng(12)
    dt_s = 0.00256
    n_t = 8192
    t = np.arange(n_t, dtype=np.float64) * dt_s

    comb = (
        1.5 * np.sin(2.0 * np.pi * 9.7656 * t)
        + 1.0 * np.sin(2.0 * np.pi * 19.5312 * t + 0.3)
        + 1.8 * np.sin(2.0 * np.pi * 50.10 * t - 0.5)
    )
    # Periodic pulsar-like train ONLY on ON beam at 11.2 Hz (orthogonal to 9.7656 Hz)
    pulsar_phase = np.mod(t * 11.2, 1.0)
    pulsar = 1.2 * np.exp(-0.5 * ((pulsar_phase - 0.5) / 0.04) ** 2)

    off_ts = 100.0 + 0.05 * rng.standard_normal(n_t) + 0.9 * comb
    on_ts = 105.0 + 0.05 * rng.standard_normal(n_t) + comb + pulsar

    res = CoherentCombCanceller().cancel(on_ts, off_ts, dt_s)
    assert res.comb_suppression_db >= 10.0, f"Low comb suppression: {res.comb_suppression_db} dB"
    assert any(abs(f - 9.7656) < 0.2 for f in res.comb_freqs_hz)
    assert any(abs(f - 50.10) < 0.2 for f in res.comb_freqs_hz)
    corr_raw = float(np.corrcoef(on_ts - np.median(on_ts), pulsar)[0, 1])
    corr_clean = float(np.corrcoef(res.cleaned_on - np.median(res.cleaned_on), pulsar)[0, 1])
    assert corr_clean > corr_raw + 0.45 and corr_clean > 0.70, (
        f"Unexpected correlation raw={corr_raw:.3f}, clean={corr_clean:.3f}"
    )
    ph = (np.mod(t * 11.2, 1.0) * 32).astype(int)
    prof = np.bincount(ph, weights=res.cleaned_on - np.median(res.cleaned_on)) / np.bincount(ph)
    folded_snr = float(np.max(prof) / max(np.std(prof[:10]), 1e-12))
    assert folded_snr > 100.0, f"Pulsar damaged by comb canceller: folded_snr={folded_snr:.2f}"


def test_find_optimal_off_beams_avoids_solar_grating_lobe():
    """RFI-3.3: Grating-lobe-safe OFF candidate beats deployed m=-0.7 (-3.3 dB) on 2x2 array."""
    pos4 = np.array(
        [
            [0.0, 0.0, 0.0],
            [-4.24775, 0.0275, 0.0],
            [0.15, 4.30625, 0.0],
            [-4.12725, 4.42725, 0.0],
        ],
        dtype=np.float64,
    )
    sun_l, sun_m = 0.0005, 0.2356
    freqs = FREQS_MHZ[::4]
    deployed_af = array_factor_power(pos4, freqs, sun_l, sun_m, 0.0, -0.70)
    deployed_db = 10.0 * np.log10(float(np.mean(deployed_af)))
    assert deployed_db > -6.0  # Known grating-lobe leakage at m=-0.7

    cands = find_optimal_off_beams(
        pos4,
        freqs,
        target_l=sun_l,
        target_m=sun_m,
        interferers=(PUDAHUEL_KNOWN_INTERFERERS["muos_3"],),
        min_sep_deg=18.0,
        top_k=2,
    )
    assert len(cands) >= 1
    assert cands[0].band_power_ratio_db < deployed_db - 6.0
    assert cands[0].band_power_ratio_db < -12.0


def test_adaptive_sidelobe_canceller_huber_irls():
    """RFI-3.3: Huber-IRLS subtracts auxiliary sidelobe leakage while preserving ON flare."""
    rng = np.random.default_rng(7)
    n_t, n_beams, n_f = 240, 3, 8
    interferer = 5.0 * np.sin(np.linspace(0, 12.0 * np.pi, n_t)) + rng.normal(0.0, 0.2, n_t)
    bp = np.ones((n_t, n_beams, n_f), dtype=np.float64) * 50.0
    bp[:, 1, :] += 1.0 * interferer[:, None] + rng.normal(0.0, 0.05, (n_t, n_f))
    bp[:, 2, :] += 0.6 * interferer[:, None] + rng.normal(0.0, 0.05, (n_t, n_f))
    bp[:, 0, :] += 0.35 * interferer[:, None] + rng.normal(0.0, 0.05, (n_t, n_f))
    # Inject bright ON-only transient flare at sample 120
    bp[119:122, 0, :] += 25.0

    res = AdaptiveSidelobeCanceller().cancel(bp, on_beam_idx=0, aux_beam_indices=[1, 2])
    assert res.suppression_db > 6.0
    assert abs(res.corr_after) < 0.2
    flare_height = float(np.mean(res.cleaned_on[120, :] - np.median(res.cleaned_on, axis=0)))
    assert flare_height > 20.0, f"Transient flare attenuated: {flare_height}"


def test_compute_adev_hdev_summary_linear_drift_invariance():
    """OPS-6.1: HDEV stays insensitive to linear drift that inflates ADEV."""
    dt_s = 1.0
    t = np.arange(400, dtype=np.float64)
    rng = np.random.default_rng(5)
    series = 1.0 + 2.0e-4 * t + 1.0e-5 * rng.standard_normal(len(t))
    stats = stt.compute_adev_hdev_summary(series, dt_s, taus_s=(1.0, 8.0, 16.0, 60.0))
    assert "adev_tau_60s" in stats and "hdev_tau_60s" in stats
    assert stats["hdev_tau_60s"] < 0.1 * stats["adev_tau_60s"]
