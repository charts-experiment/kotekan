#!/usr/bin/env python3
"""
High-Speed On-The-Fly Tracker Reconfiguration & Verification Suite v2
=====================================================================
Actively exercises the complete parameter space of the Kotekan Direct Beam Tracker
and Antenna Masking stages while the pipeline runs under full load:

- 11 Astronomical Celestial Targets (RA/Dec J2000, LST, Unix timestamps)
- Directional Cosines (l0, m0, l1, m1, dl, dm) across individual beams 0..3
- Window Tapers (uniform, cosine_pedestal, hamming, blackman, gaussian)
- Pedestal factors (0.00 .. 0.80)
- Subframe phase interpolation toggles (true/false)
- Active beam count multiplexing (1..4 beams)
- Physical vs Raw antenna element masking & unmasking (/direct_tracker and /antenna_mask)
- Comprehensive antenna mask auto-detection parameter sweeps (MAD, Kurtosis, DC lock, IQ)
- Dedicated queries to /beam_power, /telemetry, /targets, /antenna_mask/status
- Primary (/direct_tracker/*) and alias (/beam_tracker/*) REST endpoints
- Continuous latency and response auditing
"""

import time
import json
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path

REST_URL = "http://127.0.0.1:12048"
DATA_DIR = Path("/tmp/charts_1hour_run_data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_LOG = DATA_DIR / "on_the_fly_config_audit.jsonl"

TARGETS = [
    {"name": "Sagittarius A*", "ra_deg": 266.4168, "dec_deg": -29.0078},
    {"name": "Vela Pulsar", "ra_deg": 128.8361, "dec_deg": -45.1764},
    {"name": "The Sun (Transit)", "ra_deg": 198.3903, "dec_deg": -7.7872},
    {"name": "Centaurus A", "ra_deg": 201.3667, "dec_deg": -43.0192},
    {"name": "NGC 1316 (Fornax A)", "ra_deg": 50.6750, "dec_deg": -37.2083},
    {"name": "PSR B0826-34", "ra_deg": 127.0417, "dec_deg": -34.2900},
    {"name": "Crab Pulsar", "ra_deg": 83.6331, "dec_deg": 22.0145},
    {"name": "NGC 253 (Sculptor)", "ra_deg": 11.8875, "dec_deg": -25.2883},
    {"name": "Pictor A", "ra_deg": 79.7500, "dec_deg": -45.7667},
    {"name": "Hydra A", "ra_deg": 139.5000, "dec_deg": -12.0833},
    {"name": "NGC 612", "ra_deg": 23.4917, "dec_deg": -36.4933},
]

TAPERS = [
    {"taper_type": "cosine_pedestal", "taper_pedestal": 0.25},
    {"taper_type": "uniform", "taper_pedestal": 1.0},
    {"taper_type": "hamming", "taper_pedestal": 0.08},
    {"taper_type": "blackman_pedestal", "taper_pedestal": 0.0},
    {"taper_type": "gaussian", "taper_pedestal": 0.15},
    {"taper_type": "cosine_pedestal", "taper_pedestal": 0.50},
    {"taper_type": "gaussian", "taper_pedestal": 0.30},
    {"taper_type": "hamming", "taper_pedestal": 0.15},
]

COSINE_DIRS = [
    {"l0": 0.00, "m0": 0.00, "l1": 0.00, "m1": 0.00, "dl": 0.0, "dm": 0.0},
    {"l0": 0.05, "m0": -0.02, "l1": 0.055, "m1": -0.02, "dl": 5e-6, "dm": 0.0},
    {"l0": -0.12, "m0": 0.08, "l1": -0.115, "m1": 0.082, "dl": -5e-6, "dm": 2e-6},
    {"l0": 0.18, "m0": -0.22, "l1": 0.181, "m1": -0.221, "dl": 1e-6, "dm": -1e-6},
    {"l0": -0.25, "m0": -0.15, "l1": -0.248, "m1": -0.148, "dl": 2e-6, "dm": 2e-6},
    {"l0": 0.30, "m0": 0.10, "l1": 0.298, "m1": 0.10, "dl": -2e-6, "dm": 0.0},
]

MASK_CONFIGS = [
    {
        "auto_detect_enabled": True,
        "blank_voltages_enabled": True,
        "couple_pol_masks": False,
        "dead_power_threshold": 0.30,
        "sat_power_threshold": 80.0,
        "clip_fraction_threshold": 0.15,
        "enable_dc_lock_check": True,
        "max_dc_fraction": 0.30,
        "enable_iq_imbalance_check": True,
        "max_iq_imbalance": 0.50,
        "enable_kurtosis_check": True,
        "min_kurtosis_threshold": 1.15,
        "max_kurtosis_threshold": 4.50,
        "enable_mad_outlier_check": True,
        "mad_sigma_threshold": 6.0,
        "revival_frames": 5,
        "sample_stride": 1
    },
    {
        "auto_detect_enabled": True,
        "blank_voltages_enabled": True,
        "couple_pol_masks": False,
        "dead_power_threshold": 0.50,
        "sat_power_threshold": 90.0,
        "clip_fraction_threshold": 0.20,
        "enable_dc_lock_check": False,
        "enable_iq_imbalance_check": True,
        "max_iq_imbalance": 0.60,
        "enable_kurtosis_check": False,
        "enable_mad_outlier_check": True,
        "mad_sigma_threshold": 5.0,
        "revival_frames": 3,
        "sample_stride": 2
    },
    {
        "auto_detect_enabled": True,
        "blank_voltages_enabled": True,
        "couple_pol_masks": False,
        "dead_power_threshold": 1.00,
        "sat_power_threshold": 100.0,
        "clip_fraction_threshold": 0.10,
        "enable_dc_lock_check": True,
        "max_dc_fraction": 0.25,
        "enable_iq_imbalance_check": False,
        "enable_kurtosis_check": True,
        "min_kurtosis_threshold": 1.20,
        "max_kurtosis_threshold": 4.00,
        "enable_mad_outlier_check": True,
        "mad_sigma_threshold": 7.0,
        "revival_frames": 8,
        "sample_stride": 4
    }
]

def post_json(endpoint: str, payload: dict) -> dict:
    url = f"{REST_URL}{endpoint}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            lat = (time.time() - t0) * 1000.0
            rec = {"ts": time.time(), "endpoint": endpoint, "status": resp.status, "latency_ms": lat, "payload": payload, "reply": body[:80]}
            with open(AUDIT_LOG, "a") as f:
                f.write(json.dumps(rec) + "\n")
            return rec
    except Exception as e:
        lat = (time.time() - t0) * 1000.0
        rec = {"ts": time.time(), "endpoint": endpoint, "status": 500, "latency_ms": lat, "payload": payload, "error": str(e)}
        with open(AUDIT_LOG, "a") as f:
            f.write(json.dumps(rec) + "\n")
        return rec

def get_json(endpoint: str) -> dict:
    url = f"{REST_URL}{endpoint}"
    req = urllib.request.Request(url, method="GET")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            lat = (time.time() - t0) * 1000.0
            rec = {"ts": time.time(), "endpoint": endpoint, "status": resp.status, "latency_ms": lat, "reply_keys": list(data.keys())}
            with open(AUDIT_LOG, "a") as f:
                f.write(json.dumps(rec) + "\n")
            return data
    except Exception as e:
        lat = (time.time() - t0) * 1000.0
        rec = {"ts": time.time(), "endpoint": endpoint, "status": 500, "latency_ms": lat, "error": str(e)}
        with open(AUDIT_LOG, "a") as f:
            f.write(json.dumps(rec) + "\n")
        return {}

def run_step(step_idx: int):
    # Alternating prefix to test primary and alias endpoints
    prefix = "/direct_tracker" if (step_idx % 2 == 0) else "/beam_tracker"

    # 1. Steer Beams 0 and 1 to Celestial Targets (with simulated LST or current time)
    t0 = TARGETS[step_idx % len(TARGETS)]
    t1 = TARGETS[(step_idx + 4) % len(TARGETS)]
    post_json(f"{prefix}/set_celestial_target", {"beam_id": 0, "ra_deg": t0["ra_deg"], "dec_deg": t0["dec_deg"]})
    post_json(f"{prefix}/set_celestial_target", {"beam_id": 1, "ra_deg": t1["ra_deg"], "dec_deg": t1["dec_deg"], "lst_hours": (step_idx * 1.5) % 24.0})

    # 2. Steer Beams 2 and 3 to (l, m) Directional Cosines with trajectory extrapolation
    c2 = COSINE_DIRS[step_idx % len(COSINE_DIRS)]
    c3 = COSINE_DIRS[(step_idx + 2) % len(COSINE_DIRS)]
    post_json(f"{prefix}/set_target", {"beam_id": 2, "l0": c2["l0"], "m0": c2["m0"], "l1": c2["l1"], "m1": c2["m1"], "dl": c2["dl"], "dm": c2["dm"]})
    post_json(f"{prefix}/set_target", {"beam_id": 3, "l0": c3["l0"], "m0": c3["m0"], "l1": c3["l1"], "m1": c3["m1"], "dl": c3["dl"], "dm": c3["dm"]})

    # 3. Dynamic Beam Count Multiplexing (1..4 beams)
    active_b = (step_idx % 4) + 1
    post_json(f"{prefix}/enable_beam", {"num_active_beams": active_b})

    # 4. Dynamic Window Taper Variation
    taper = TAPERS[step_idx % len(TAPERS)].copy()
    taper["enable_weight_bislc"] = False
    post_json(f"{prefix}/set_taper", taper)

    # 5. Dynamic Subframe Phase Interpolation Toggle
    interp = (step_idx % 2 == 0)
    post_json(f"{prefix}/set_interpolation", {"enabled": interp})

    # 6. Physical vs Raw Antenna Masking
    ant_phys = [0, 4, 5, 6, 32, 37, 48][step_idx % 7]
    ant_raw = [63, 59, 58, 57, 31, 26, 15][step_idx % 7]
    # Mask physical antenna via tracker and antenna_mask
    post_json(f"{prefix}/mask_antenna", {"physical_id": ant_phys, "enabled": False})
    post_json("/antenna_mask/mask", {"antenna_id": ant_phys})
    # Mask raw element
    post_json(f"{prefix}/mask_antenna", {"raw_id": ant_raw, "is_raw": True, "enabled": False})
    time.sleep(0.05)
    # Unmask physical antenna and raw element
    post_json(f"{prefix}/mask_antenna", {"physical_id": ant_phys, "enabled": True})
    post_json("/antenna_mask/unmask", {"antenna_id": ant_phys})
    post_json(f"{prefix}/mask_antenna", {"raw_id": ant_raw, "is_raw": True, "enabled": True})

    # 7. Antenna Mask Detection Hyperparameter Sweep
    mask_cfg = MASK_CONFIGS[step_idx % len(MASK_CONFIGS)]
    post_json("/antenna_mask/configure", mask_cfg)

    # 8. Reset Mask Periodically
    if step_idx % 8 == 0:
        post_json("/antenna_mask/reset", {})

    # 9. Telemetry & Stitched Power Ingestion
    if step_idx % 3 == 0:
        get_json(f"{prefix}/telemetry")
        get_json(f"{prefix}/targets")
        get_json(f"{prefix}/beam_power")

if __name__ == "__main__":
    import sys
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    interval = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    for i in range(count):
        run_step(i)
        time.sleep(interval)
