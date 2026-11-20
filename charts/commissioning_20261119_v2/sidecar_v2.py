#!/usr/bin/env python3
"""Live beam-steering & REST/GPU/ring-buffer telemetry sidecar for CHARTS
v2 commissioning & chromaticity campaign on ms-x01."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import signal
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import astropy.units as u
import numpy as np
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_sun
from astropy.time import Time
from skyfield.api import EarthSatellite, load, wgs84

from charts_constants import CHARTS_ALTITUDE_M, CHARTS_LATITUDE_DEG, CHARTS_LONGITUDE_DEG

STOP_REQUESTED = False


def _handle_sig(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


signal.signal(signal.SIGTERM, _handle_sig)
signal.signal(signal.SIGINT, _handle_sig)


def http_get_json(base_url: str, path: str, timeout: float = 3.0) -> Any:
    req = urllib.request.Request(base_url.rstrip("/") + path, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_post_json(base_url: str, path: str, payload: Dict[str, Any], timeout: float = 3.0) -> Any:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8").strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"message": raw}


def altaz_to_lm(alt_deg: float, az_deg: float) -> Tuple[float, float]:
    el_r = np.deg2rad(alt_deg)
    az_r = np.deg2rad(az_deg)
    return float(np.cos(el_r) * np.sin(az_r)), float(np.cos(el_r) * np.cos(az_r))


def clamp_lm(l_val: float, m_val: float, max_r2: float = 0.96) -> Tuple[float, float]:
    r2 = l_val * l_val + m_val * m_val
    if r2 > max_r2:
        scale = math.sqrt(max_r2 / r2)
        return float(l_val * scale), float(m_val * scale)
    return float(l_val), float(m_val)


def sky_lm(coord: SkyCoord, frame: AltAz) -> Tuple[float, float, float, float]:
    aa = coord.transform_to(frame)
    el_d = float(aa.alt.deg)
    az_d = float(aa.az.deg)
    l0, m0 = clamp_lm(*altaz_to_lm(el_d, az_d))
    return el_d, az_d, l0, m0


def load_tles(tle_path: Path, ts) -> Dict[int, EarthSatellite]:
    sats: Dict[int, EarthSatellite] = {}
    if not tle_path.exists():
        return sats
    lines = [ln.strip() for ln in tle_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    for i in range(0, len(lines) - 2, 3):
        name, l1, l2 = lines[i], lines[i + 1], lines[i + 2]
        if l1.startswith("1 ") and l2.startswith("2 "):
            try:
                nid = int(l1[2:7].strip())
                sats[nid] = EarthSatellite(l1, l2, name, ts)
            except ValueError:
                pass
    return sats


def sat_altaz_lm(sat: EarthSatellite, observer, sf_time) -> Tuple[float, float, float, float]:
    alt, az, _ = (sat - observer).at(sf_time).altaz()
    el_d = float(alt.degrees)
    az_d = float(az.degrees)
    l0, m0 = clamp_lm(*altaz_to_lm(el_d, az_d))
    return el_d, az_d, l0, m0


def poll_gpu_smi() -> Dict[str, Any]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=3,
        ).strip()
        parts = [float(x.strip()) for x in out.splitlines()[0].split(",")]
        return {
            "gpu_util_pct": parts[0],
            "mem_util_pct": parts[1],
            "vram_used_mib": parts[2],
            "vram_total_mib": parts[3],
            "temp_c": parts[4],
            "power_w": parts[5],
        }
    except Exception as exc:
        return {"error": str(exc)}


def summarize_buffers(buf_json: Dict[str, Any]) -> Dict[str, Any]:
    summary = {}
    for name, info in buf_json.items():
        frames = info.get("frames", [])
        n_frames = len(frames)
        in_use = sum(1 for f in frames if f != 0)
        consumers = {
            c: {
                "acq": cv.get("last_frame_acquired"),
                "rel": cv.get("last_frame_released"),
            }
            for c, cv in info.get("consumers", {}).items()
        }
        summary[name] = {
            "num_frames": n_frames,
            "nonzero_frames": in_use,
            "occupancy_pct": round(100.0 * in_use / n_frames, 1) if n_frames else 0.0,
            "consumers": consumers,
        }
    return summary


def summarize_gpu_profile(base_url: str) -> Dict[str, Any]:
    out = {}
    for idx in (0, 1):
        for ep in (f"/gpu_profile/stages/gpu_{idx}", f"/gpu_profile/gpu/gpu_{idx}"):
            try:
                gp = http_get_json(base_url, ep, timeout=2.0)
                cmds = {}
                for bucket in ("copy_in", "kernel", "copy_out"):
                    for entry in gp.get(bucket, []):
                        cname = entry.get("name", bucket)
                        cmds[f"{bucket}:{cname}"] = round(float(entry.get("time") or 0.0) * 1e3, 3)
                tot_ms = round(sum(cmds.values()), 3)
                out[f"gpu_{idx}"] = {"total_ms": tot_ms, "commands_ms": cmds}
                break
            except Exception:
                continue
    return out


def dir_size_bytes(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for fn in files:
            try:
                total += (Path(root) / fn).stat().st_size
            except OSError:
                pass
    return total


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mode",
        required=True,
        choices=[
            "corr_monitor",
            "muos_rfi",
            "fornax_chromaticity",
            "multi_source",
            "pulsar_static",
            "sun_onoff",
            "cena_dual",
            "solar_chromaticity_2d",
            "galactic_center",
            "stress",
        ],
    )
    p.add_argument("--url", default="http://127.0.0.1:12048")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--tle-file", default="/data/commissioning_20261119_v2/tle_cache.tle")
    p.add_argument("--rfi-targets", default="/data/commissioning_20261119_v2/localized_rfi_targets.json")
    p.add_argument("--duration-s", type=float, default=900.0)
    p.add_argument("--poll-interval-s", type=float, default=5.0)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    telem_path = out_dir / "telemetry_log.jsonl"
    pointing_path = out_dir / "beam_pointing_log.jsonl"
    rfi_snap_path = out_dir / "rfi_channel_snapshots.jsonl"

    rfi_locs = {}
    for rp in (Path(args.rfi_targets), Path("/data/commissioning_20261119/localized_rfi_targets.json")):
        if rp.exists():
            rfi_locs = json.loads(rp.read_text(encoding="utf-8"))
            break
    rfi_425 = rfi_locs.get("RFI_425.4", {}).get("peak_el_ge_25", {"l": 0.008, "m": -0.9031})
    rfi_472 = rfi_locs.get("RFI_472.5", {}).get("peak_el_ge_25", {"l": -0.5035, "m": -0.1838})

    ts = load.timescale()
    caren_sf = wgs84.latlon(CHARTS_LATITUDE_DEG, CHARTS_LONGITUDE_DEG, elevation_m=CHARTS_ALTITUDE_M)
    loc_ap = EarthLocation(
        lat=CHARTS_LATITUDE_DEG * u.deg,
        lon=CHARTS_LONGITUDE_DEG * u.deg,
        height=CHARTS_ALTITUDE_M * u.m,
    )
    sats = load_tles(Path(args.tle_file), ts)

    fornax_sc = SkyCoord(50.6737 * u.deg, -37.2083 * u.deg)
    pictor_sc = SkyCoord(79.9571 * u.deg, -45.7789 * u.deg)
    orion_sc = SkyCoord(83.8221 * u.deg, -5.3911 * u.deg)
    lmc_sc = SkyCoord(84.6767 * u.deg, -69.1008 * u.deg)
    crab_sc = SkyCoord(83.6329 * u.deg, 22.0144 * u.deg)
    cena_sc = SkyCoord(201.3650 * u.deg, -43.0192 * u.deg)
    gal_cuts = [(b, SkyCoord(l=0.0 * u.deg, b=float(b) * u.deg, frame="galactic").icrs) for b in (0, 4, -4, 8, -8, 15, -15)]
    gal_cold_sc = SkyCoord(318.0 * u.deg, -35.0 * u.deg)

    # Wait up to 60s for Kotekan REST server to come up
    ready_ep = "/buffers" if args.mode == "corr_monitor" else "/direct_tracker/status"
    ready = False
    t_wait = time.time() + 60.0
    while time.time() < t_wait and not STOP_REQUESTED:
        try:
            http_get_json(args.url, ready_ep, timeout=2.0)
            ready = True
            break
        except Exception:
            time.sleep(1.0)
    if not ready:
        raise RuntimeError(f"Kotekan REST {args.url}{ready_ep} did not become ready within 60s")

    t_end = time.time() + args.duration_s
    last_steer = 0.0
    steer_cadence = 10.0 if args.mode in ("sun_onoff", "solar_chromaticity_2d") else 5.0
    if args.mode in ("corr_monitor", "pulsar_static"):
        steer_cadence = 999999.0
    cycle = 0

    while not STOP_REQUESTED and time.time() < t_end:
        t0_loop = time.time()
        now_dt = datetime.now(timezone.utc)
        sf_t = ts.from_datetime(now_dt)
        t_ap = Time(now_dt)
        frame = AltAz(obstime=t_ap, location=loc_ap)

        # 1. Dynamic Steering (skip corr_monitor and pulsar_static)
        if args.mode not in ("corr_monitor", "pulsar_static") and (t0_loop - last_steer) >= steer_cadence:
            last_steer = t0_loop
            steer_cmds: List[Dict[str, Any]] = []
            send_mask: Dict[int, bool] = {}

            if args.mode == "muos_rfi":
                m1 = sat_altaz_lm(sats[38093], caren_sf, sf_t) if 38093 in sats else (40, 315, -0.54, 0.54)
                m5 = sat_altaz_lm(sats[41622], caren_sf, sf_t) if 41622 in sats else (35, 310, -0.63, 0.53)
                m3 = sat_altaz_lm(sats[40374], caren_sf, sf_t) if 40374 in sats else (18, 65, 0.86, 0.41)
                leo_pool = [25544, 28654, 54361, 50466, 44909, 38771, 43689, 39086]
                best_sat = None
                for nid in leo_pool:
                    if nid in sats:
                        el_d, az_d, ls, ms = sat_altaz_lm(sats[nid], caren_sf, sf_t)
                        if el_d >= 15.0 and (best_sat is None or el_d > best_sat[0]):
                            best_sat = (el_d, az_d, ls, ms, sats[nid].name, nid)
                if best_sat is not None:
                    b4 = {"beam_id": 4, "label": f"LEO_{best_sat[4]}_{best_sat[5]}", "l0": best_sat[2], "m0": best_sat[3], "el_deg": best_sat[0], "az_deg": best_sat[1]}
                else:
                    g19 = sat_altaz_lm(sats[60133], caren_sf, sf_t) if 60133 in sats else (50.9, 352.2, -0.086, 0.625)
                    b4 = {"beam_id": 4, "label": "GOES-19_468.8MHz", "l0": g19[2], "m0": g19[3], "el_deg": g19[0], "az_deg": g19[1]}
                steer_cmds = [
                    {"beam_id": 0, "label": "MUOS-1", "l0": m1[2], "m0": m1[3], "el_deg": m1[0], "az_deg": m1[1]},
                    {"beam_id": 1, "label": "MUOS-5", "l0": m5[2], "m0": m5[3], "el_deg": m5[0], "az_deg": m5[1]},
                    {"beam_id": 2, "label": "MUOS-3", "l0": m3[2], "m0": m3[3], "el_deg": m3[0], "az_deg": m3[1]},
                    {"beam_id": 3, "label": "OFF_MUOS_Ctrl", "l0": 0.45, "m0": -0.45},
                    b4,
                    {"beam_id": 5, "label": "Loc_425.4MHz", "l0": float(rfi_425["l"]), "m0": float(rfi_425["m"])},
                    {"beam_id": 6, "label": "Loc_472.5MHz", "l0": float(rfi_472["l"]), "m0": float(rfi_472["m"])},
                    {"beam_id": 7, "label": "Zenith_Ctrl", "l0": 0.0, "m0": 0.0},
                ]

            elif args.mode == "fornax_chromaticity":
                f_el, f_az, f_l, f_m = sky_lm(fornax_sc, frame)
                p_el, p_az, p_l, p_m = sky_lm(pictor_sc, frame)
                l_p3, m_p3 = clamp_lm(f_l + 0.052, f_m)
                l_m3, m_m3 = clamp_lm(f_l - 0.052, f_m)
                l_p6, m_p6 = clamp_lm(f_l + 0.105, f_m)
                l_m6, m_m6 = clamp_lm(f_l - 0.105, f_m)
                steer_cmds = [
                    {"beam_id": 0, "label": "FornaxA_Cal_ON", "l0": f_l, "m0": f_m, "el_deg": f_el, "az_deg": f_az, "calibrated": True},
                    {"beam_id": 1, "label": "FornaxA_ParkedDrift", "l0": 0.0, "m0": -0.0644, "parked": True},
                    {"beam_id": 2, "label": "FornaxA_E+3deg", "l0": l_p3, "m0": m_p3},
                    {"beam_id": 3, "label": "FornaxA_W-3deg", "l0": l_m3, "m0": m_m3},
                    {"beam_id": 4, "label": "FornaxA_E+6deg", "l0": l_p6, "m0": m_p6},
                    {"beam_id": 5, "label": "FornaxA_W-6deg", "l0": l_m6, "m0": m_m6},
                    {"beam_id": 6, "label": "FornaxA_Uncal_ON", "l0": f_l, "m0": f_m, "calibrated": False},
                    {"beam_id": 7, "label": "PictorA", "l0": p_l, "m0": p_m, "el_deg": p_el, "az_deg": p_az},
                    {"beam_id": 8, "label": "Zenith", "l0": 0.0, "m0": 0.0, "parked": True},
                    {"beam_id": 9, "label": "Deep_OFF", "l0": -0.50, "m0": 0.50, "parked": True},
                ]
                send_mask = {0: True, 1: False, 2: True, 3: True, 4: True, 5: True, 6: True, 7: True, 8: False, 9: False}

            elif args.mode == "multi_source":
                pic = sky_lm(pictor_sc, frame)
                ori = sky_lm(orion_sc, frame)
                lmc = sky_lm(lmc_sc, frame)
                crb = sky_lm(crab_sc, frame)
                l_olmc, m_olmc = clamp_lm(*altaz_to_lm(lmc[0], (lmc[1] - 55.0) % 360.0))
                l_ocrb, m_ocrb = clamp_lm(*altaz_to_lm(crb[0], (crb[1] + 55.0) % 360.0))
                op = sky_lm(SkyCoord(35.0 * u.deg, -45.8 * u.deg), frame)
                oo = sky_lm(SkyCoord(39.0 * u.deg, -5.4 * u.deg), frame)
                steer_cmds = [
                    {"beam_id": 0, "label": "PictorA", "l0": pic[2], "m0": pic[3], "el_deg": pic[0], "az_deg": pic[1]},
                    {"beam_id": 1, "label": "OrionA", "l0": ori[2], "m0": ori[3], "el_deg": ori[0], "az_deg": ori[1]},
                    {"beam_id": 2, "label": "LMC_30Dor", "l0": lmc[2], "m0": lmc[3], "el_deg": lmc[0], "az_deg": lmc[1]},
                    {"beam_id": 3, "label": "Crab_TauA", "l0": crb[2], "m0": crb[3], "el_deg": crb[0], "az_deg": crb[1]},
                    {"beam_id": 4, "label": "OFF_Pictor", "l0": op[2], "m0": op[3], "el_deg": op[0], "az_deg": op[1]},
                    {"beam_id": 5, "label": "OFF_Orion", "l0": oo[2], "m0": oo[3], "el_deg": oo[0], "az_deg": oo[1]},
                    {"beam_id": 6, "label": "OFF_LMC_SameEl", "l0": l_olmc, "m0": m_olmc, "el_deg": lmc[0], "az_deg": (lmc[1] - 55.0) % 360.0},
                    {"beam_id": 7, "label": "OFF_Crab_SameEl", "l0": l_ocrb, "m0": m_ocrb, "el_deg": crb[0], "az_deg": (crb[1] + 55.0) % 360.0},
                    {"beam_id": 8, "label": "PictorA_Uncal", "l0": pic[2], "m0": pic[3], "calibrated": False},
                    {"beam_id": 9, "label": "Zenith", "l0": 0.0, "m0": 0.0, "parked": True},
                ]
                send_mask = {0: True, 1: True, 2: True, 3: True, 4: True, 5: True, 6: True, 7: True, 8: True, 9: False}

            elif args.mode == "sun_onoff":
                sun_el, sun_az, l_sun, m_sun = sky_lm(get_sun(t_ap), frame)
                l_off_n, m_off_n = clamp_lm(*altaz_to_lm(sun_el, (sun_az - 42.0) % 360.0))
                l_off_s, m_off_s = clamp_lm(*altaz_to_lm(sun_el, (sun_az + 42.0) % 360.0))
                steer_cmds = [
                    {"beam_id": 0, "label": "Sun_Cal_ON", "l0": l_sun, "m0": m_sun, "el_deg": sun_el, "az_deg": sun_az, "calibrated": True},
                    {"beam_id": 1, "label": "OFF_North_SameEl", "l0": l_off_n, "m0": m_off_n, "el_deg": sun_el, "az_deg": (sun_az - 42.0) % 360.0, "calibrated": True},
                    {"beam_id": 2, "label": "OFF_South_SameEl", "l0": l_off_s, "m0": m_off_s, "el_deg": sun_el, "az_deg": (sun_az + 42.0) % 360.0, "calibrated": True},
                    {"beam_id": 3, "label": "Sun_Uncal_ON", "l0": l_sun, "m0": m_sun, "el_deg": sun_el, "az_deg": sun_az, "calibrated": False},
                ]

            elif args.mode == "cena_dual":
                cen_el, cen_az, cen_l, cen_m = sky_lm(cena_sc, frame)
                sun_el, sun_az, s_l, s_m = sky_lm(get_sun(t_ap), frame)
                lp5, mp5 = clamp_lm(cen_l + 0.0872, cen_m)
                lm5, mm5 = clamp_lm(cen_l - 0.0872, cen_m)
                steer_cmds = [
                    {"beam_id": 0, "label": "CenA_Cal_ON", "l0": cen_l, "m0": cen_m, "el_deg": cen_el, "az_deg": cen_az, "calibrated": True},
                    {"beam_id": 1, "label": "CenA_Uncal_ON", "l0": cen_l, "m0": cen_m, "calibrated": False},
                    {"beam_id": 2, "label": "CenA_E+5deg", "l0": lp5, "m0": mp5},
                    {"beam_id": 3, "label": "CenA_W-5deg", "l0": lm5, "m0": mm5},
                    {"beam_id": 4, "label": "Sun_Monitor", "l0": s_l, "m0": s_m, "el_deg": sun_el, "az_deg": sun_az},
                    {"beam_id": 5, "label": "Deep_OFF", "l0": -0.55, "m0": 0.45, "parked": True},
                ]
                send_mask = {0: True, 1: True, 2: True, 3: True, 4: True, 5: False}

            elif args.mode == "solar_chromaticity_2d":
                s_el, s_az, s_l, s_m = sky_lm(get_sun(t_ap), frame)
                l2, m2 = clamp_lm(s_l + 0.035, s_m)
                l3, m3 = clamp_lm(s_l + 0.070, s_m)
                l4, m4 = clamp_lm(s_l + 0.122, s_m)
                l5, m5 = clamp_lm(s_l - 0.070, s_m)
                l6, m6 = clamp_lm(s_l, s_m + 0.070)
                l7, m7 = clamp_lm(s_l, s_m - 0.122)
                steer_cmds = [
                    {"beam_id": 0, "label": "Sun_Cal_ON", "l0": s_l, "m0": s_m, "el_deg": s_el, "az_deg": s_az, "calibrated": True},
                    {"beam_id": 1, "label": "Sun_ParkedDrift", "l0": 0.0, "m0": 0.2356, "parked": True},
                    {"beam_id": 2, "label": "Sun_E+2deg", "l0": l2, "m0": m2},
                    {"beam_id": 3, "label": "Sun_E+4deg", "l0": l3, "m0": m3},
                    {"beam_id": 4, "label": "Sun_E+7deg", "l0": l4, "m0": m4},
                    {"beam_id": 5, "label": "Sun_W-4deg", "l0": l5, "m0": m5},
                    {"beam_id": 6, "label": "Sun_N+4deg", "l0": l6, "m0": m6},
                    {"beam_id": 7, "label": "Sun_S-7deg", "l0": l7, "m0": m7},
                    {"beam_id": 8, "label": "Sun_Uncal_ON", "l0": s_l, "m0": s_m, "calibrated": False},
                    {"beam_id": 9, "label": "Deep_South_OFF", "l0": 0.0, "m0": -0.70, "parked": True},
                ]
                send_mask = {0: True, 1: False, 2: True, 3: True, 4: True, 5: True, 6: True, 7: True, 8: True, 9: False}

            elif args.mode == "galactic_center":
                steer_cmds = []
                for bi, (bdeg, sc) in enumerate(gal_cuts):
                    el_d, az_d, lg, mg = sky_lm(sc, frame)
                    steer_cmds.append({"beam_id": bi, "label": f"Gal_l0_b{bdeg:+d}", "l0": lg, "m0": mg, "el_deg": el_d, "az_deg": az_d})
                s_el, s_az, s_l, s_m = sky_lm(get_sun(t_ap), frame)
                c_el, c_az, c_l, c_m = sky_lm(gal_cold_sc, frame)
                steer_cmds.extend([
                    {"beam_id": 7, "label": "SgrA_ParkedDrift", "l0": 0.0, "m0": 0.0768, "parked": True},
                    {"beam_id": 8, "label": "Sun_Monitor", "l0": s_l, "m0": s_m, "el_deg": s_el, "az_deg": s_az},
                    {"beam_id": 9, "label": "Cold_OFF", "l0": c_l, "m0": c_m, "el_deg": c_el, "az_deg": c_az},
                ])
                send_mask = {i: (i != 7) for i in range(10)}

            elif args.mode == "stress":
                s_el, s_az, s_l, s_m = sky_lm(get_sun(t_ap), frame)
                best_leo = None
                for nid in (26931, 67683, 27431, 57191, 25544):
                    if nid in sats:
                        el_d, az_d, ls, ms = sat_altaz_lm(sats[nid], caren_sf, sf_t)
                        if el_d >= 15.0 and (best_leo is None or el_d > best_leo[0]):
                            best_leo = (el_d, az_d, ls, ms, sats[nid].name, nid)
                b5 = (
                    {"beam_id": 5, "label": f"LEO_{best_leo[4]}", "l0": best_leo[2], "m0": best_leo[3], "el_deg": best_leo[0], "az_deg": best_leo[1]}
                    if best_leo
                    else {"beam_id": 5, "label": "MUOS-5_Fallback", "l0": -0.65, "m0": 0.48}
                )
                steer_cmds = [
                    {"beam_id": 0, "label": "Sun", "l0": s_l, "m0": s_m, "el_deg": s_el, "az_deg": s_az},
                    b5,
                ]

            if steer_cmds:
                for cmd in steer_cmds:
                    bid = cmd["beam_id"]
                    if send_mask and not send_mask.get(bid, True):
                        continue
                    payload: Dict[str, Any] = {"beam_id": bid, "l0": cmd["l0"], "m0": cmd["m0"]}
                    if "calibrated" in cmd:
                        payload["calibrated"] = cmd["calibrated"]
                    try:
                        http_post_json(args.url, "/direct_tracker/set_target", payload, timeout=2.0)
                    except Exception as exc:
                        cmd["error"] = str(exc)
                with open(pointing_path, "a", encoding="utf-8") as pf:
                    pf.write(json.dumps({"utc": now_dt.isoformat(), "unix": t0_loop, "targets": steer_cmds}) + "\n")

        # 2. Telemetry Snapshot
        rec: Dict[str, Any] = {"utc": now_dt.isoformat(), "unix": round(t0_loop, 3), "cycle": cycle}
        if args.mode != "corr_monitor":
            for key, ep in (
                ("tracker_status", "/direct_tracker/status"),
                ("tracker_telemetry", "/direct_tracker/telemetry"),
                ("rfi_status", "/rfi_mitigation/status"),
                ("antenna_mask", "/antenna_mask/status"),
            ):
                try:
                    rec[key] = http_get_json(args.url, ep, timeout=2.5)
                except Exception as exc:
                    rec[key] = {"error": str(exc)}

        try:
            rec["buffers"] = summarize_buffers(http_get_json(args.url, "/buffers", timeout=2.5))
        except Exception as exc:
            rec["buffers"] = {"error": str(exc)}

        rec["gpu_profile"] = summarize_gpu_profile(args.url)
        rec["nvidia_smi"] = poll_gpu_smi()
        rec["run_dir_bytes"] = dir_size_bytes(out_dir)
        rec["root_free_gb"] = round(shutil.disk_usage("/").free / (1024 ** 3), 2)

        with open(telem_path, "a", encoding="utf-8") as tf:
            tf.write(json.dumps(rec) + "\n")

        if args.mode in ("muos_rfi", "fornax_chromaticity", "solar_chromaticity_2d", "galactic_center", "stress") and (cycle % 3 == 0):
            try:
                ch_data = http_get_json(args.url, "/rfi_mitigation/channels", timeout=3.0)
                with open(rfi_snap_path, "a", encoding="utf-8") as rf:
                    rf.write(json.dumps({"utc": now_dt.isoformat(), "unix": round(t0_loop, 3), "data": ch_data}) + "\n")
            except Exception:
                pass

        cycle += 1
        sleep_s = max(0.2, args.poll_interval_s - (time.time() - t0_loop))
        time.sleep(sleep_s)


if __name__ == "__main__":
    main()
