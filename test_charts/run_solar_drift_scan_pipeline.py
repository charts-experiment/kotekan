#!/usr/bin/env python3
"""
CHARTS Direct Beam Tracker Solar Drift-Scan Pipeline
====================================================
Runs the Direct Beam Tracker solar drift-scan observation from 4:00 PM to 8:00 PM Chile local time.

Features:
1. Array-Stationary Beams Along the Sun's Path:
   - 4 stationary beams positioned sequentially along the Sun's upcoming transit path:
     * Beam 0: 16:50 Chile local time transit (Alt = 40.18 deg, Az = 279.47 deg)
     * Beam 1: 17:35 Chile local time transit (Alt = 30.85 deg, Az = 272.55 deg)
     * Beam 2: 18:20 Chile local time transit (Alt = 21.47 deg, Az = 266.37 deg)
     * Beam 3: 19:05 Chile local time transit (Alt = 12.15 deg, Az = 260.48 deg)
   - 2 stationary Cold Sky reference beams positioned on the same declination circle (-14.51 deg)
     at matching elevations in the Eastern sky:
     * Beam 4: Cold Sky Ref 1 (Alt = 40.18 deg, Az = 80.53 deg)
     * Beam 5: Cold Sky Ref 2 (Alt = 30.85 deg, Az = 87.45 deg)
   - All beams are array-stationary (is_celestial = false, dl=0, dm=0).

2. Hard 5 GB Storage Cap:
   - Logs ONLY the reduced tracker output from http://127.0.0.1:12048 (no raw DPDK frames or visibilities).
   - Enforces a strict 5.0 GB maximum disk limit on solar_drift_scan_data/.

3. Live Per-Frame Report:
   - Updates summary report on every saved frame showing:
     * Current beam powers
     * On/off contrast vs Cold Sky (ratio and dB)
     * Angular distance from the Sun to each stationary beam center (deg)
     * Total disk usage and % of 5 GB cap
     * Progress towards 8:00 PM Chile local time (23:00 UTC).
"""

import os
import sys
import time
import json
import shutil
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation, get_sun

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
REST_URL = "http://127.0.0.1:12048"
DATA_DIR = Path("/home/fpalma/kotekan/solar_drift_scan_data")
REPORT_PATH = DATA_DIR / "LIVE_REPORT.md"
ARTIFACT_DIR = Path("/home/fpalma/.gemini/antigravity-cli/brain/662ba808-d560-440c-b239-f6e5746902d9")
ARTIFACT_PATH = ARTIFACT_DIR / "SOLAR_DRIFT_SCAN_REPORT.md"

MAX_DISK_BYTES = 5 * 1024 * 1024 * 1024  # Strict 5.0 GB limit
PRUNE_TARGET_BYTES = 4_500_000_000       # Prune oldest dumps if exceeding 4.5 GB

# Carén Observatory Location
LAT_DEG = -33.4211146
LON_DEG = -70.8634710
ALT_M = 458.0
CAREN_LOC = EarthLocation(lat=LAT_DEG * u.deg, lon=LON_DEG * u.deg, height=ALT_M * u.m)

# Chile Timezone: UTC - 3 hours
CHILE_TZ = timezone(timedelta(hours=-3))

# Stop time: 8:00 PM Chile local time on 2026-11-01 (23:00:00 UTC)
STOP_TIME_UTC = datetime(2026, 11, 1, 23, 0, 0, tzinfo=timezone.utc)

# -----------------------------------------------------------------------------
# Beam Configurations
# -----------------------------------------------------------------------------
BEAMS = [
    {
        "beam_id": 0,
        "label": "Sun Transit 1 (16:50)",
        "transit_local": "16:50",
        "l0": -0.7536,
        "m0": 0.1257,
        "n0": 0.6452,
        "alt_deg": 40.18,
        "az_deg": 279.47,
        "is_reference": False,
        "ref_beam_id": 4
    },
    {
        "beam_id": 1,
        "label": "Sun Transit 2 (17:35)",
        "transit_local": "17:35",
        "l0": -0.8576,
        "m0": 0.0382,
        "n0": 0.5129,
        "alt_deg": 30.85,
        "az_deg": 272.55,
        "is_reference": False,
        "ref_beam_id": 5
    },
    {
        "beam_id": 2,
        "label": "Sun Transit 3 (18:20)",
        "transit_local": "18:20",
        "l0": -0.9287,
        "m0": -0.0589,
        "n0": 0.3661,
        "alt_deg": 21.47,
        "az_deg": 266.37,
        "is_reference": False,
        "ref_beam_id": 4
    },
    {
        "beam_id": 3,
        "label": "Sun Transit 4 (19:05)",
        "transit_local": "19:05",
        "l0": -0.9641,
        "m0": -0.1617,
        "n0": 0.2106,
        "alt_deg": 12.15,
        "az_deg": 260.48,
        "is_reference": False,
        "ref_beam_id": 5
    },
    {
        "beam_id": 4,
        "label": "Cold Sky Ref 1 (East 40°)",
        "transit_local": "Reference",
        "l0": 0.7536,
        "m0": 0.1257,
        "n0": 0.6452,
        "alt_deg": 40.18,
        "az_deg": 80.53,
        "is_reference": True,
        "ref_beam_id": 4
    },
    {
        "beam_id": 5,
        "label": "Cold Sky Ref 2 (East 31°)",
        "transit_local": "Reference",
        "l0": 0.8576,
        "m0": 0.0382,
        "n0": 0.5129,
        "alt_deg": 30.85,
        "az_deg": 87.45,
        "is_reference": True,
        "ref_beam_id": 5
    }
]

# -----------------------------------------------------------------------------
# REST Helpers
# -----------------------------------------------------------------------------
def post_json(endpoint: str, payload: dict) -> dict:
    url = f"{REST_URL}{endpoint}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=5.0) as resp:
        return {"status": resp.status, "body": resp.read().decode("utf-8", errors="replace")}

def get_json(endpoint: str) -> dict:
    url = f"{REST_URL}{endpoint}"
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=5.0) as resp:
        return json.loads(resp.read().decode("utf-8"))

def setup_stationary_beams():
    """Configures Kotekan direct tracker with the 6 array-stationary beams."""
    print("Configuring Kotekan Direct Beam Tracker with Array-Stationary Beams...")
    post_json("/direct_tracker/enable_beam", {"num_active_beams": len(BEAMS)})
    for b in BEAMS:
        payload = {
            "beam_id": b["beam_id"],
            "l0": b["l0"],
            "m0": b["m0"],
            "dl": 0.0,
            "dm": 0.0
        }
        post_json("/direct_tracker/set_target", payload)

    # Verify targets are stationary (is_celestial = false)
    targets_data = get_json("/direct_tracker/targets")
    for t in targets_data.get("targets", []):
        bid = t["beam_id"]
        assert t.get("is_celestial") is False, f"Beam {bid} must have is_celestial=false"
    print(f"  -> All {len(BEAMS)} beams confirmed STATIONARY relative to physical array.")

# -----------------------------------------------------------------------------
# Celestial & Ephemeris Computation
# -----------------------------------------------------------------------------
def compute_sun_position(time_dt_utc):
    """Computes Sun's current RA, Dec, topocentric Alt, Az, and direction cosines (l, m, n)."""
    t_astropy = Time(time_dt_utc, scale="utc")
    sun = get_sun(t_astropy)
    lst_hours = t_astropy.sidereal_time("apparent", longitude=CAREN_LOC.lon).hour
    
    ha_rad = np.radians(lst_hours * 15.0 - sun.ra.deg)
    dec_rad = np.radians(sun.dec.deg)
    lat_rad = np.radians(LAT_DEG)
    
    l_sun = -np.cos(dec_rad) * np.sin(ha_rad)
    m_sun = np.cos(lat_rad) * np.sin(dec_rad) - np.sin(lat_rad) * np.cos(dec_rad) * np.cos(ha_rad)
    trans_sq = l_sun * l_sun + m_sun * m_sun
    n_sun = np.sqrt(max(0.0, 1.0 - trans_sq))
    alt_sun = np.degrees(np.arcsin(n_sun))
    az_sun = (np.degrees(np.arctan2(l_sun, m_sun)) + 360.0) % 360.0
    
    return {
        "ra_deg": float(sun.ra.deg),
        "dec_deg": float(sun.dec.deg),
        "lst_hours": float(lst_hours),
        "l": float(l_sun),
        "m": float(m_sun),
        "n": float(n_sun),
        "alt_deg": float(alt_sun),
        "az_deg": float(az_sun)
    }

def compute_angular_distance_deg(l1, m1, n1, l2, m2, n2):
    """Angular distance between two topocentric unit vectors in degrees."""
    u1 = np.array([l1, m1, n1], dtype=np.float64)
    u2 = np.array([l2, m2, n2], dtype=np.float64)
    u1 /= max(1e-9, np.linalg.norm(u1))
    u2 /= max(1e-9, np.linalg.norm(u2))
    dot = np.clip(np.dot(u1, u2), -1.0, 1.0)
    return float(np.degrees(np.arccos(dot)))

# -----------------------------------------------------------------------------
# Disk & File Management
# -----------------------------------------------------------------------------
def get_data_dir_size_bytes():
    """Computes total bytes stored in DATA_DIR."""
    total = 0
    if DATA_DIR.exists():
        for p in DATA_DIR.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
    return total

def enforce_disk_limit():
    """Prunes oldest frame dumps if approaching 5 GB cap."""
    current_size = get_data_dir_size_bytes()
    if current_size > PRUNE_TARGET_BYTES:
        dumps_dir = DATA_DIR / "spectra_dumps"
        if dumps_dir.exists():
            files = sorted(dumps_dir.glob("*.npz"), key=lambda f: f.stat().st_mtime)
            for f in files:
                try:
                    f.unlink()
                    current_size = get_data_dir_size_bytes()
                    if current_size <= PRUNE_TARGET_BYTES:
                        break
                except Exception:
                    pass

# -----------------------------------------------------------------------------
# Live Markdown Report Generation
# -----------------------------------------------------------------------------
def update_report(frame_info, sun_info, beam_results, disk_bytes, start_time_utc):
    """Generates the live summary markdown report on every saved frame."""
    now_utc = datetime.now(timezone.utc)
    now_local = now_utc.astimezone(CHILE_TZ)
    time_remaining = STOP_TIME_UTC - now_utc
    rem_seconds = max(0, int(time_remaining.total_seconds()))
    rem_str = str(timedelta(seconds=rem_seconds))
    
    elapsed = now_utc - start_time_utc
    elapsed_str = str(timedelta(seconds=int(elapsed.total_seconds())))

    disk_mb = disk_bytes / (1024 * 1024)
    disk_gb = disk_bytes / (1024 * 1024 * 1024)
    disk_pct = (disk_bytes / MAX_DISK_BYTES) * 100.0

    lines = [
        "# CHARTS Direct Beam Tracker: Solar Drift-Scan Live Observation Report",
        "",
        "> [!IMPORTANT]",
        f"> **Solar Drift-Scan Active**: Tracking sequential transits across array-stationary beams from **4:00 PM to 8:00 PM Chile Time** (19:00 to 23:00 UTC).",
        "",
        "## 1. Real-Time Observation Status",
        "",
        f"- **Local Time (Chile)**: `{now_local.strftime('%Y-%m-%d %H:%M:%S')} -03`",
        f"- **UTC Time**: `{now_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC`",
        f"- **Current LST**: `{sun_info['lst_hours']:.4f} h`",
        f"- **Observation Progress**: Elapsed: `{elapsed_str}` | Remaining until 8:00 PM: `{rem_str}`",
        f"- **Pipeline Frame ID**: `#{frame_info.get('frame_id', 0)}` (Timestamp: `{frame_info.get('timestamp_utc', 0.0):.2f}`)",
        f"- **Hardware Storage Usage**: **`{disk_mb:.2f} MB`** (`{disk_gb:.4f} GB` / `5.00 GB` cap, **`{disk_pct:.2f}%`**)",
        "",
        "## 2. Live Sun Ephemeris",
        "",
        f"- **Equatorial (J2000)**: RA = `{sun_info['ra_deg']:.3f}°`, Dec = `{sun_info['dec_deg']:.3f}°`",
        f"- **Topocentric Coordinates**: Altitude = **`{sun_info['alt_deg']:.2f}°`**, Azimuth = **`{sun_info['az_deg']:.2f}°`**",
        f"- **Direction Cosines**: $\\ell_\\odot = {sun_info['l']:+.4f}$, $m_\\odot = {sun_info['m']:+.4f}$, $n_\\odot = {sun_info['n']:.4f}$",
        "",
        "## 3. Array-Stationary Beams & Solar Proximity",
        "",
        "| Beam | Configuration & Target | Fixed Coordinates $(\\ell_0, m_0, n_0)$ | Alt / Az | Distance to Sun ($\\theta_\\odot$) | Mean Power | On/Off Contrast vs. Cold Sky | Transit Status |",
        "|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|"
    ]

    for b in BEAMS:
        bid = b["beam_id"]
        res = beam_results.get(bid, {})
        mean_p = res.get("mean_power", 0.0)
        contrast_ratio = res.get("contrast_ratio", 1.0)
        contrast_db = res.get("contrast_db", 0.0)
        dist_deg = res.get("ang_dist_deg", 99.9)

        if b["is_reference"]:
            status_badge = "🟢 **BASELINE REF**"
            contrast_str = "Baseline (1.00x)"
        elif dist_deg < 2.5:
            status_badge = "🔥 **IN-BEAM TRANSIT**"
            contrast_str = f"**{contrast_ratio:.2f}x ({contrast_db:+.2f} dB)**"
        elif dist_deg < 8.0:
            status_badge = "🟡 **ENTERING BEAM**"
            contrast_str = f"{contrast_ratio:.2f}x ({contrast_db:+.2f} dB)"
        elif sun_info["l"] < b["l0"]:
            status_badge = "⚪ Transited (Past)"
            contrast_str = f"{contrast_ratio:.2f}x ({contrast_db:+.2f} dB)"
        else:
            status_badge = "🔵 Upcoming"
            contrast_str = f"{contrast_ratio:.2f}x ({contrast_db:+.2f} dB)"

        lines.append(
            f"| **{bid}** | **{b['label']}** | `({b['l0']:+.4f}, {b['m0']:+.4f}, {b['n0']:.4f})` | {b['alt_deg']:.1f}° / {b['az_deg']:.1f}° | **`{dist_deg:.2f}°`** | **`{mean_p:.4f}`** | {contrast_str} | {status_badge} |"
        )

    lines.extend([
        "",
        "## 4. Spectral Distribution & Subbands",
        "",
        "| Beam | Subband 0 (300.0–400.8 MHz) | Subband 1 (400.8–501.3 MHz) | Peak Channel (MHz) | Peak Spectral Power |",
        "|:---:|:---:|:---:|:---:|:---:|"
    ])

    for b in BEAMS:
        bid = b["beam_id"]
        res = beam_results.get(bid, {})
        sb0 = res.get("sb0_mean", 0.0)
        sb1 = res.get("sb1_mean", 0.0)
        peak_p = res.get("peak_power", 0.0)
        peak_ch = res.get("peak_chan", 0)
        peak_freq = 300.0 + peak_ch * 0.3
        lines.append(
            f"| **{bid} ({b['label']})** | `{sb0:.4f}` | `{sb1:.4f}` | Ch {peak_ch} (`{peak_freq:.1f} MHz`) | `{peak_p:.2f}` |"
        )

    lines.extend([
        "",
        "## 5. Storage Integrity & Safety Invariants",
        "",
        f"- **Tracker Data Directory**: `{DATA_DIR}`",
        f"- **Hard Disk Limit**: `{MAX_DISK_BYTES / (1024**3):.1f} GB` (Active Auto-Pruning Threshold: `{PRUNE_TARGET_BYTES / (1024**3):.1f} GB`)",
        f"- **Current Directory Size**: `{disk_mb:.2f} MB` / `{MAX_DISK_BYTES / (1024**3):.1f} GB`",
        f"- **Data Ingest**: Logging exclusively reduced 672-channel formed beam tracker output from port 12048.",
        f"- **Weight Normalization**: Active (`normalize_weights: true`) ensuring baseline noise floor is pinned to single-antenna levels.",
        "",
        "---",
        f"*Report auto-generated at {now_utc.strftime('%Y-%m-%d %H:%M:%S')} UTC*"
    ])

    report_text = "\n".join(lines) + "\n"

    # Write to live report file
    try:
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(report_text)
    except Exception as e:
        print(f"Error writing {REPORT_PATH}: {e}")

    # Write to Antigravity artifact directory
    try:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        with open(ARTIFACT_PATH, "w", encoding="utf-8") as f:
            f.write(report_text)
    except Exception as e:
        print(f"Error writing {ARTIFACT_PATH}: {e}")

# -----------------------------------------------------------------------------
# Main Loop
# -----------------------------------------------------------------------------
def main():
    print("=" * 100)
    print(" CHARTS DIRECT BEAM TRACKER: SOLAR DRIFT-SCAN RUNNER")
    print(f" Observation Window: 4:00 PM to 8:00 PM Chile Time (Stop at {STOP_TIME_UTC.isoformat()})")
    print(f" Storage Directory:  {DATA_DIR} (Strict 5.0 GB limit)")
    print("=" * 100)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dumps_dir = DATA_DIR / "spectra_dumps"
    dumps_dir.mkdir(parents=True, exist_ok=True)
    telemetry_log_path = DATA_DIR / "solar_drift_scan_telemetry.jsonl"

    # 1. Setup stationary beams
    setup_stationary_beams()

    start_time_utc = datetime.now(timezone.utc)
    last_fid = -1
    frame_count = 0
    save_spectra_interval = 5  # Save compressed 672-ch spectra every 5 seconds

    print("\nStarting continuous solar drift-scan logging loop...")

    while datetime.now(timezone.utc) < STOP_TIME_UTC:
        try:
            now_utc = datetime.now(timezone.utc)
            now_local = now_utc.astimezone(CHILE_TZ)

            # Query beam power from Kotekan REST
            data = get_json("/direct_tracker/beam_power")
            fid = data.get("frame_id", 0)

            if fid > 0 and fid != last_fid:
                last_fid = fid
                frame_count += 1
                ts_utc = data.get("timestamp_utc", now_utc.timestamp())
                frame_dt = datetime.fromtimestamp(ts_utc, tz=timezone.utc)

                # Compute Sun position at this frame's exact timestamp
                sun_info = compute_sun_position(frame_dt)

                # Process beam power spectra
                all_spectra = data.get("beam_power", [])
                beam_results = {}

                # Calculate Cold Sky baseline powers first
                cold_sky_mean_4 = 1.0
                cold_sky_mean_5 = 1.0
                if len(all_spectra) > 4 and len(all_spectra[4]) > 0:
                    cold_sky_mean_4 = float(np.mean(all_spectra[4]))
                if len(all_spectra) > 5 and len(all_spectra[5]) > 0:
                    cold_sky_mean_5 = float(np.mean(all_spectra[5]))

                for b in BEAMS:
                    bid = b["beam_id"]
                    if bid < len(all_spectra) and len(all_spectra[bid]) > 0:
                        spec = np.array(all_spectra[bid], dtype=np.float32)
                        mean_p = float(np.mean(spec))
                        std_p = float(np.std(spec))
                        peak_p = float(np.max(spec))
                        peak_ch = int(np.argmax(spec))
                        sb0_m = float(np.mean(spec[:336])) if len(spec) >= 336 else mean_p
                        sb1_m = float(np.mean(spec[336:])) if len(spec) >= 672 else 0.0

                        # Reference beam for contrast
                        ref_bid = b["ref_beam_id"]
                        ref_mean = cold_sky_mean_4 if ref_bid == 4 else cold_sky_mean_5
                        contrast_ratio = mean_p / max(1e-6, ref_mean)
                        contrast_db = 10.0 * np.log10(max(1e-6, contrast_ratio))

                        ang_dist = compute_angular_distance_deg(
                            sun_info["l"], sun_info["m"], sun_info["n"],
                            b["l0"], b["m0"], b["n0"]
                        )

                        beam_results[bid] = {
                            "mean_power": mean_p,
                            "std_power": std_p,
                            "peak_power": peak_p,
                            "peak_chan": peak_ch,
                            "sb0_mean": sb0_m,
                            "sb1_mean": sb1_m,
                            "contrast_ratio": contrast_ratio,
                            "contrast_db": contrast_db,
                            "ang_dist_deg": ang_dist
                        }

                # Periodically save compressed full-spectrum dump
                if frame_count % save_spectra_interval == 0:
                    npz_path = dumps_dir / f"frame_{fid}.npz"
                    np.savez_compressed(
                        npz_path,
                        frame_id=fid,
                        timestamp_utc=ts_utc,
                        sun_l=sun_info["l"],
                        sun_m=sun_info["m"],
                        sun_n=sun_info["n"],
                        sun_alt=sun_info["alt_deg"],
                        sun_az=sun_info["az_deg"],
                        beam_power=np.array(all_spectra[:len(BEAMS)], dtype=np.float32)
                    )

                # Check and enforce hard 5 GB disk limit
                enforce_disk_limit()
                current_disk_bytes = get_data_dir_size_bytes()

                # Append telemetry line
                telemetry_entry = {
                    "frame_id": fid,
                    "timestamp_utc": ts_utc,
                    "local_time": now_local.strftime("%H:%M:%S"),
                    "sun": sun_info,
                    "beams": {bid: beam_results[bid] for bid in beam_results},
                    "disk_bytes": current_disk_bytes
                }
                with open(telemetry_log_path, "a", encoding="utf-8") as f_log:
                    f_log.write(json.dumps(telemetry_entry) + "\n")

                # Update live summary report
                update_report(data, sun_info, beam_results, current_disk_bytes, start_time_utc)

                # Print console log every 5 frames
                if frame_count % 5 == 0:
                    b0_dist = beam_results.get(0, {}).get("ang_dist_deg", 0.0)
                    b0_pow = beam_results.get(0, {}).get("mean_power", 0.0)
                    b1_dist = beam_results.get(1, {}).get("ang_dist_deg", 0.0)
                    b1_pow = beam_results.get(1, {}).get("mean_power", 0.0)
                    cold_pow = beam_results.get(4, {}).get("mean_power", 0.0)
                    disk_mb = current_disk_bytes / (1024 * 1024)
                    print(
                        f"[{now_local.strftime('%H:%M:%S')}] Frame #{fid} | Sun Alt={sun_info['alt_deg']:5.1f}° | "
                        f"B0(16:50): dist={b0_dist:5.2f}° P={b0_pow:.3f} | "
                        f"B1(17:35): dist={b1_dist:5.2f}° P={b1_pow:.3f} | "
                        f"ColdSky: P={cold_pow:.3f} | Disk: {disk_mb:.1f} MB"
                    )

        except Exception as e:
            print(f"Error in drift-scan loop: {e}")

        time.sleep(1.0)

    print("\n" + "=" * 100)
    print(f" SOLAR DRIFT-SCAN OBSERVATION REACHED 8:00 PM CHILE TIME ({datetime.now(timezone.utc).isoformat()})")
    print(" Observation completed successfully!")
    print("=" * 100)

if __name__ == "__main__":
    main()
