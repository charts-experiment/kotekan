"""Tests for Wave 4A: DBSCAN clustering, sidelobe confidence, and frb-voe VOEvent + NE2025 priors."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

_KOTEKAN_ROOT = Path(__file__).resolve().parents[2]
_CHARTS_DIR = _KOTEKAN_ROOT / "charts"
_TEST_CHARTS_DIR = _KOTEKAN_ROOT / "test_charts"
for p in (str(_KOTEKAN_ROOT), str(_CHARTS_DIR), str(_TEST_CHARTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from charts.frb_voevent import (
    build_frb_voevent_xml,
    compute_dm_excess,
    emit_frb_voevent_alert,
    equ_to_gal_deg,
    estimate_ne2025_galactic_prior,
)
from detect_frb_events import (
    MatchedFilterMultiBeamScorer,
    cluster_candidates_dbscan,
    compute_sidelobe_confidence,
    two_stage_iqr_normalize,
)


def test_two_stage_iqr_and_sidelobe_confidence() -> None:
    rng = np.random.default_rng(1234)
    # Slow exponential drift + unit noise
    t = np.arange(1024, dtype=np.float64)
    drift = 50.0 + 10.0 * np.sin(2.0 * math.pi * t / 800.0)
    wf = drift[None, :] + rng.normal(0.0, 2.0, size=(32, 1024))
    norm = two_stage_iqr_normalize(wf, short_win=64, long_win=256)
    assert norm.shape == wf.shape
    assert abs(float(np.median(norm))) < 0.15
    assert 0.7 < float(np.std(norm)) < 1.3

    # Localized beam vs common-mode sidelobe confidence
    c_loc, r_loc, b_loc = compute_sidelobe_confidence(18.0, np.array([0.4, 0.2, 20.0, 0.3]))
    assert b_loc == 2
    assert r_loc < 0.05
    assert c_loc > 17.0

    c_rfi, r_rfi, _ = compute_sidelobe_confidence(18.0, np.array([19.5, 20.0, 19.2, 19.8]))
    assert r_rfi > 0.95
    assert c_rfi < 1.0


def test_dbscan_clustering_and_multibeam_scorer() -> None:
    # Two clusters + separated candidate across beams
    raw = [
        {"dm": 345.0, "time_s": 1.000, "snr": 12.0, "beam_id": 0, "sidelobe_confidence": 11.5},
        {"dm": 350.0, "time_s": 1.002, "snr": 16.5, "beam_id": 0, "sidelobe_confidence": 16.0},
        {"dm": 355.0, "time_s": 1.004, "snr": 13.0, "beam_id": 0, "sidelobe_confidence": 12.1},
        {"dm": 120.0, "time_s": 2.500, "snr": 9.0, "beam_id": 1, "sidelobe_confidence": 8.2},
        {"dm": 124.0, "time_s": 2.501, "snr": 10.5, "beam_id": 1, "sidelobe_confidence": 9.9},
    ]
    reps = cluster_candidates_dbscan(raw, sigma_t_s=0.01, sigma_dm=15.0, eps=1.5, min_samples=2)
    assert len(reps) == 2
    assert reps[0]["beam_id"] == 0 and math.isclose(reps[0]["snr"], 16.5) and reps[0]["cluster_size"] == 3
    assert reps[1]["beam_id"] == 1 and math.isclose(reps[1]["snr"], 10.5) and reps[1]["cluster_size"] == 2

    # Multi-beam scorer accepts localized dispersed burst (DM=100) and rejects common-mode DM=0 RFI
    rng = np.random.default_rng(77)
    n_beams, n_freq, n_time = 4, 32, 1024
    freqs_mhz = np.linspace(300.0, 312.5, n_freq)
    dt_s = 1.0e-3
    bg = rng.normal(10.0, 1.0, size=(n_beams, n_freq, n_time))

    # Localized DM=100 burst in beam 2 near sample 400
    f_top = float(np.max(freqs_mhz))
    delays = np.rint(4.148808e3 * 100.0 * ((freqs_mhz**-2) - (f_top**-2)) / dt_s).astype(int)
    for ch in range(n_freq):
        idx = 300 + int(delays[ch])
        if 0 <= idx < n_time:
            bg[2, ch, idx : min(n_time, idx + 3)] += 4.5

    # Common-mode DM=0 RFI across ALL beams near sample 750
    bg[:, :, 750:752] += 10.0

    scorer = MatchedFilterMultiBeamScorer(
        freqs_mhz=freqs_mhz,
        dm_trials=(0.0, 50.0, 100.0, 150.0),
        widths_samples=(1, 2, 4),
        snr_threshold=6.0,
        max_sidelobe_ratio=0.70,
        min_sidelobe_confidence=4.0,
    )
    accepted, all_cands = scorer.score_multibeam(bg, dt_s=dt_s)
    assert any(c["rfi_vetoed"] and abs(c["dm"] - 0.0) < 1e-6 for c in all_cands)
    assert len(accepted) >= 1
    assert accepted[0]["beam_id"] == 2
    assert abs(accepted[0]["dm"] - 100.0) < 1e-6


def test_ne2025_priors_and_voevent_xml(tmp_path: Path) -> None:
    # SGP (~RA=12.86, Dec=-27.13) vs Galactic Center (RA=266.4168, Dec=-29.0078)
    _, b_sgp = equ_to_gal_deg(12.86, -27.13)
    _, b_gc = equ_to_gal_deg(266.4168, -29.0078)
    assert abs(b_sgp) > 85.0
    assert abs(b_gc) < 5.0

    p_sgp = estimate_ne2025_galactic_prior(12.86, -27.13, freq_mhz=400.0)
    p_gc = estimate_ne2025_galactic_prior(266.4168, -29.0078, freq_mhz=400.0)
    assert p_gc["dm_gal_total"] > 10.0 * p_sgp["dm_gal_total"]
    assert p_gc["tau_scatter_ms"] > p_sgp["tau_scatter_ms"]

    # Frequency scaling ~ nu^-4.0
    p_300 = estimate_ne2025_galactic_prior(12.86, -27.13, freq_mhz=300.0)
    p_600 = estimate_ne2025_galactic_prior(12.86, -27.13, freq_mhz=600.0)
    assert math.isclose(p_300["tau_scatter_ms"] / p_600["tau_scatter_ms"], 16.0, rel_tol=0.02)

    ex = compute_dm_excess(500.0, p_sgp)
    assert ex["is_extragalactic"] and ex["dm_excess"] > 400.0 and ex["p_extragalactic"] > 0.99

    cand = {
        "event_id": "test_evt_001",
        "ra_deg": 12.86,
        "dec_deg": -27.13,
        "freq_mhz": 400.0,
        "dm": 500.0,
        "snr": 19.4,
        "sidelobe_confidence": 18.8,
        "width_ms": 1.2,
        "beam_id": 3,
    }
    xml_out = tmp_path / "alert.xml"
    json_out = tmp_path / "alert.json"
    res = emit_frb_voevent_alert(cand, xml_path=xml_out, json_path=json_out)
    assert res["latency_ms"] < 100.0
    assert xml_out.exists() and json_out.exists()

    root = ET.fromstring(build_frb_voevent_xml(cand))
    assert root.attrib["version"] == "2.0"
    params = {p.attrib["name"]: p.attrib["value"] for p in root.findall(".//Param")}
    assert "dm_gal_ne2025" in params and "dm_excess" in params and "tau_scatter_ms" in params
