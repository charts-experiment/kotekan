#!/usr/bin/env python3
"""Kotekan CUDA Beam Tracker Live Steering & Health Telemetry CLI.

Connects to Kotekan's live REST server (default http://127.0.0.1:12048) to:
1. Steer beams in direction cosine space (l, m) or equatorial coordinates (RA, Dec).
2. Dynamically enable/disable tracking beam slots (1..8).
3. Mask/unmask bad or noisy antenna elements.
4. Watch live pipeline health, sky map coordinates, and antenna array status.

Usage:
    python kotekan_tracker_control.py status
    python kotekan_tracker_control.py steer-lm --beam 0 --l0 0.05 --m0 -0.02 --dl 1e-5
    python kotekan_tracker_control.py steer-radec --beam 0 --ra 83.633 --dec 22.014 --now
    python kotekan_tracker_control.py enable-beams --count 4
    python kotekan_tracker_control.py mask-antenna --id 14 --disable
    python kotekan_tracker_control.py watch --interval 1.0
    python kotekan_tracker_control.py interactive
"""

from __future__ import annotations

import argparse
try:
    import curses
except ImportError:
    curses = None
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional
import urllib.error
import urllib.request
import numpy as np

# Setup paths for constants
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

try:
    from constants import (
        C_LIGHT,
        CHARTS_LATITUDE_DEG,
        CHARTS_LONGITUDE_DEG,
        DEFAULT_SPACING_M,
        LOCAL_FREQUENCY_CHANNELS,
        get_default_charts_h5_path,
    )
except ImportError:
    CHARTS_LATITUDE_DEG = -33.4211146
    CHARTS_LONGITUDE_DEG = -70.8634710
    CHARTS_ALTITUDE_M = 458.0
    DEFAULT_SPACING_M = 0.6
    LOCAL_FREQUENCY_CHANNELS = 336
    def get_default_charts_h5_path():
        from pathlib import Path
        return Path("baseband_virtual.h5")



class KotekanTrackerClient:
    """REST Client for Kotekan Beam Tracker endpoints."""

    def __init__(self, host: str = "127.0.0.1", port: int = 12048, timeout_s: float = 3.0):
        self.base_url = f"http://{host}:{port}"
        self.timeout_s = timeout_s

    def _get(self, endpoint: str) -> Dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        req = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                data = resp.read().decode("utf-8")
                return json.loads(data)
        except urllib.error.URLError as e:
            raise ConnectionError(f"Failed to connect to Kotekan at {url}: {e}")

    def _post(self, endpoint: str, payload: Dict[str, Any]) -> str:
        url = f"{self.base_url}{endpoint}"
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return resp.read().decode("utf-8").strip()
        except urllib.error.URLError as e:
            raise ConnectionError(f"Failed to connect to Kotekan at {url}: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Query /beam_tracker/status."""
        return self._get("/beam_tracker/status")

    def get_inspect_frame(self, buffer_name: str = "host_formed_beams_buffer") -> bytes:
        """Fetch raw binary frame from /inspect_frame/<buffer_name>."""
        url = f"{self.base_url}/inspect_frame/{buffer_name}"
        req = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return resp.read()
        except urllib.error.URLError as e:
            raise ConnectionError(f"Failed to fetch inspect frame from {url}: {e}")

    def steer_lm(self, beam_id: int, l0: float, m0: float, dl: float = 0.0, dm: float = 0.0) -> str:
        """Set beam trajectory via direction cosines (l, m)."""
        payload = {
            "beam_id": beam_id,
            "l0": float(l0),
            "m0": float(m0),
            "dl": float(dl),
            "dm": float(dm),
        }
        return self._post("/beam_tracker/set_trajectory", payload)

    def steer_radec(
        self,
        beam_id: int,
        ra_deg: float,
        dec_deg: float,
        lst_hours: Optional[float] = None,
        unix_timestamp_s: Optional[float] = None,
    ) -> str:
        """Set celestial target via (RA, Dec)."""
        payload: Dict[str, Any] = {
            "beam_id": beam_id,
            "ra_deg": float(ra_deg),
            "dec_deg": float(dec_deg),
        }
        if lst_hours is not None:
            payload["lst_hours"] = float(lst_hours)
        elif unix_timestamp_s is not None:
            payload["unix_timestamp_s"] = float(unix_timestamp_s)
        else:
            payload["unix_timestamp_s"] = time.time()
        return self._post("/beam_tracker/set_celestial_target", payload)

    def enable_beams(self, num_active_beams: int) -> str:
        """Dynamically set number of active beams (1..8)."""
        payload = {"num_active_beams": int(num_active_beams)}
        return self._post("/beam_tracker/enable_beam", payload)

    def get_antenna_mask_status(self) -> Dict[str, Any]:
        """Query /antenna_mask/status."""
        return self._get("/antenna_mask/status")

    def mask_antenna_stage(self, antenna_id: int) -> str:
        """Mask antenna via /antenna_mask/mask."""
        return self._post("/antenna_mask/mask", {"antenna_id": int(antenna_id)})

    def unmask_antenna_stage(self, antenna_id: int) -> str:
        """Unmask antenna via /antenna_mask/unmask."""
        return self._post("/antenna_mask/unmask", {"antenna_id": int(antenna_id)})

    def reset_antenna_masks(self) -> str:
        """Reset all antenna masks via /antenna_mask/reset."""
        return self._post("/antenna_mask/reset", {})

    def mask_antenna(self, antenna_id: int, enabled: bool) -> str:
        """Mask (disable) or unmask (enable) a specific antenna."""
        try:
            if enabled:
                return self.unmask_antenna_stage(antenna_id)
            else:
                return self.mask_antenna_stage(antenna_id)
        except Exception:
            payload = {"antenna_id": int(antenna_id), "enabled": bool(enabled)}
            return self._post("/beam_tracker/mask_antenna", payload)

    def set_antenna_mask(
        self,
        bad_elements: Optional[List[int]] = None,
        active_elements: Optional[List[int]] = None,
    ) -> str:
        """Batch update antenna mask."""
        payload: Dict[str, Any] = {}
        if bad_elements is not None:
            payload["bad_elements"] = bad_elements
        if active_elements is not None:
            payload["active_elements"] = active_elements
        return self._post("/beam_tracker/set_antenna_mask", payload)

    def auto_mask(self, h5_path: Optional[str] = None, power_threshold: float = 0.05) -> str:
        """Automatically detects unplugged/dead antennas from baseband data and masks them."""
        bad_elements = []
        active_elements = []
        if h5_path and os.path.exists(h5_path):
            import h5py
            with h5py.File(h5_path, "r") as f:
                dset = f["baseband"]
                raw = dset[:, :min(84, dset.shape[1]), :min(2048, dset.shape[2])]
                r = (raw & 0x0F).astype(np.int8)
                i = (raw >> 4).astype(np.int8)
                r[r >= 8] -= 16
                i[i >= 8] -= 16
                c = r.astype(np.float32) + 1j * i.astype(np.float32)
                powers = np.mean(np.abs(c) ** 2, axis=(1, 2))
                for a in range(len(powers)):
                    if powers[a] < power_threshold:
                        bad_elements.append(int(a))
                    else:
                        active_elements.append(int(a))
        else:
            # Default auto-masking: activate channels 0..7, mask channels 8..63
            active_elements = list(range(8))
            bad_elements = list(range(8, 64))

        payload = {"bad_elements": bad_elements, "active_elements": active_elements}
        return self._post("/beam_tracker/auto_mask", payload)

    def set_working_antennas(self, active_elements: List[int]) -> Dict[str, Any]:
        """Activate specified raw antenna elements and mask all others."""
        payload = {"active_elements": active_elements}
        return self._post("/beam_tracker/set_antenna_mask", payload)


def render_ascii_skymap(trajectories: List[Dict[str, Any]], active_beams: int) -> str:
    """Render a 2D ASCII hemisphere sky map with beam positions."""
    grid_size = 17  # 17x17 grid
    center = grid_size // 2
    canvas = [[" " for _ in range(grid_size)] for _ in range(grid_size)]

    # Draw horizon circle (l^2 + m^2 = 1.0)
    for r in range(grid_size):
        for c in range(grid_size):
            y = (center - r) / center
            x = (c - center) / center
            d_sq = x * x + y * y
            if 0.9 <= d_sq <= 1.1:
                canvas[r][c] = "·"
            elif d_sq < 0.9:
                canvas[r][c] = " "

    # Draw Zenith cross
    canvas[center][center] = "+"

    # Plot beams
    for b in range(min(active_beams, len(trajectories))):
        traj = trajectories[b]
        l0 = traj.get("l0", 0.0)
        m0 = traj.get("m0", 0.0)
        c = int(round(center + l0 * center))
        r = int(round(center - m0 * center))
        if 0 <= r < grid_size and 0 <= c < grid_size:
            canvas[r][c] = str(b)

    lines = ["    N (+m)"]
    for row_idx, row in enumerate(canvas):
        prefix = "W " if row_idx == center else "  "
        suffix = " E" if row_idx == center else ""
        lines.append(f"{prefix} {''.join(row)} {suffix}")
    lines.append("    S (-m)")
    return "\n".join(lines)


def print_status_dashboard(status: Dict[str, Any], latency_ms: float):
    """Print full formatted status dashboard to console."""
    print("=" * 80)
    print(" CHARTS Kotekan Beam Tracker Live Status & Health Dashboard")
    print("=" * 80)
    print(f" REST Connection Latency : {latency_ms:.2f} ms")
    print(f" Output Format            : {status.get('output_format', 'N/A')}")
    print(f" Total Array Elements    : {status.get('total_elements', 32)}")
    print(f" Active Antennas (Alive) : {status.get('active_antennas', 0)} / {status.get('total_elements', 32)}")
    print(f" Masked Antennas (Dead)  : {status.get('masked_antennas', 0)}")
    active_raw = status.get("active_raw_elements", [])
    active_phys = status.get("active_physical_antennas", [])
    if active_raw:
        print(f"   -> Active Raw Elements: {active_raw}")
        print(f"   -> Physical Antennas  : {active_phys}")
    site = status.get("site", {})
    print(f" Site Location           : Lat {site.get('lat_deg', 0.0):.4f}°, Lon {site.get('lon_deg', 0.0):.4f}°, Alt {site.get('alt_m', 0.0):.1f} m")
    print(f" Active Beams / Capacity : {status.get('num_active_beams', 1)} / {status.get('max_beams_capacity', 8)}")
    print(f" Integration Spectra     : {status.get('integration_spectra', 320)} (~1.07 ms)")
    print(f" Element Spacing         : {status.get('spacing_m', 0.6):.2f} m")
    print("-" * 80)

    # Beams table
    trajectories = status.get("trajectories", [])
    active_count = status.get("num_active_beams", 1)
    print(" ACTIVE BEAMS & TRAJECTORY STATE:")
    print(" | Beam | State  |   l0 (East) |   m0 (North)|   n0 (Zenith)|  dl/dt (1/s) |  dm/dt (1/s) | Target (RA, Dec) |")
    print(" +------+--------+-------------+-------------+--------------+--------------+--------------+-------------------+")
    for b, traj in enumerate(trajectories):
        state = "ACTIVE" if b < active_count else "IDLE  "
        cel = traj.get("celestial_target", {})
        cel_str = f"({cel.get('ra_deg', 0.0):.2f}°, {cel.get('dec_deg', 0.0):.2f}°)" if cel.get("is_set", False) else "Manual (l,m)"
        print(f" | {b:4d} | {state} | {traj.get('l0', 0.0):11.5f} | {traj.get('m0', 0.0):11.5f} | {traj.get('n0', 0.0):12.5f} | {traj.get('dl', 0.0):12.4e} | {traj.get('dm', 0.0):12.4e} | {cel_str:17s} |")
    print(" +------+--------+-------------+-------------+--------------+--------------+--------------+-------------------+")

    print("\n SKY RECEPTIVITY FOOTPRINT (Hemisphere Topocentric Projection):")
    print(render_ascii_skymap(trajectories, active_count))
    print("=" * 80)


def print_antenna_mask_dashboard(mask_status: Dict[str, Any]):
    """Print a clean terminal dashboard of antenna health and masking telemetry."""
    print("=" * 80)
    print(" CHARTS ANTENNA HEALTH & MASKING STAGE TELEMETRY")
    print("=" * 80)
    n_ant = mask_status.get("num_elements", 0)
    active = mask_status.get("active_antennas", 0)
    masked = mask_status.get("masked_antennas", 0)
    dead = mask_status.get("dead_antennas", 0)
    sat = mask_status.get("saturated_antennas", 0)
    manual = mask_status.get("manual_masked_antennas", 0)

    print(f" Version                : {mask_status.get('version', 'N/A')}")
    print(f" Elements (Total/Active): {n_ant} / {active} (Masked: {masked})")
    print(f" Breakdown              : Dead: {dead} | Saturated: {sat} | Manual Mask: {manual}")
    thresh = mask_status.get("thresholds", {})
    clip_pct = thresh.get('clip_fraction_threshold', 0.02) * 100.0
    print(f" Thresholds             : Dead <= {thresh.get('dead_power_threshold', 0.05)} | Sat >= {thresh.get('sat_power_threshold', 80.0)} | Clip >= {clip_pct:.1f}%")
    print("-" * 80)
    print(" | Ant ID | State       | Active | Mean Power | Clip Frac (%) | Clip Count | Healthy Frames |")
    print(" +--------+-------------+--------+------------+---------------+------------+----------------+")
    for ant in mask_status.get("antennas", []):
        aid = ant.get("id", 0)
        st = ant.get("status", "UNKNOWN")
        act = "YES" if ant.get("active", False) else "NO "
        pwr = ant.get("power", 0.0)
        clip_f = ant.get("clipping_fraction", 0.0) * 100.0
        clip_c = ant.get("clipped_count", 0)
        rec = ant.get("consecutive_healthy", 0)
        print(f" | {aid:6d} | {st:11s} | {act:6s} | {pwr:10.4f} | {clip_f:12.2f}% | {clip_c:10d} | {rec:14d} |")
    print(" +--------+-------------+--------+------------+---------------+------------+----------------+")
    print("=" * 80)


def watch_loop(client: KotekanTrackerClient, interval_s: float = 1.0):
    """Continuously poll and display dashboard."""
    try:
        while True:
            os.system("clear" if os.name == "posix" else "cls")
            t0 = time.perf_counter()
            try:
                status = client.get_status()
                t1 = time.perf_counter()
                print_status_dashboard(status, (t1 - t0) * 1000.0)
            except Exception as e:
                print(f"[ERROR] Connection to Kotekan failed: {e}")
                print("Make sure Kotekan is running with REST server enabled (e.g. -b 127.0.0.1:12048)")
            time.sleep(interval_s)
    except KeyboardInterrupt:
        print("\nExiting watch mode.")


def stream_loop(client: KotekanTrackerClient, buffer_name: str = "host_formed_beams_buffer", interval_s: float = 1.0):
    """Continuously poll and display live formed beam RF stream metrics in terminal."""
    print("=" * 80)
    print(f" CHARTS LIVE FORMED BEAM RF STREAM MONITOR (/inspect_frame/{buffer_name})")
    print(f" Endpoint: {client.base_url}/inspect_frame/{buffer_name}")
    print(" Press Ctrl+C to stop.")
    print("=" * 80)

    frame_idx = 0
    try:
        while True:
            t0 = time.perf_counter()
            try:
                raw_data = client.get_inspect_frame(buffer_name)
                t_fetch_ms = (time.perf_counter() - t0) * 1000.0

                data = np.frombuffer(raw_data, dtype=np.complex64)
                n_samples = len(data)

                try:
                    st = client.get_status()
                    n_beams = st.get("num_active_beams", 2)
                except Exception:
                    n_beams = 2

                now_str = time.strftime("%H:%M:%S")
                mb_size = len(raw_data) / (1024.0 * 1024.0)
                print(f"\n[{now_str}] Frame #{frame_idx:04d} | Payload: {mb_size:.2f} MB ({n_samples:,} complex float2) | REST: {t_fetch_ms:.1f} ms")

                samples_per_beam = n_samples // max(1, n_beams)
                for b in range(n_beams):
                    b_slice = data[b * samples_per_beam : (b + 1) * samples_per_beam]
                    if len(b_slice) == 0:
                        continue
                    power = np.abs(b_slice) ** 2
                    mean_p = float(np.mean(power))
                    rms_v = float(np.sqrt(mean_p))
                    peak_p = float(np.max(power))
                    db_p = 10.0 * math.log10(mean_p + 1e-12)

                    bar_len = 24
                    bar_filled = min(bar_len, max(0, int(rms_v * 12)))
                    bar_str = "█" * bar_filled + "░" * (bar_len - bar_filled)

                    print(f"  Beam {b} | RMS: {rms_v:8.4f} | Power: {db_p:6.1f} dB | [{bar_str}] Peak: {peak_p:8.4f}")

                frame_idx += 1
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    print(f"[{time.strftime('%H:%M:%S')}] Waiting for beam tracker to produce first frame... (HTTP 404)")
                else:
                    print(f"[ERROR] HTTP {e.code}: {e.reason}")
            except Exception as e:
                print(f"[{time.strftime('%H:%M:%S')}] Waiting for stream / frame arrival: {e}")

            time.sleep(interval_s)
    except KeyboardInterrupt:
        print("\nExiting stream monitor.")


def render_ascii_spectrum(
    db_values: np.ndarray,
    height: int = 10,
    width: int = 60,
    freq_start_mhz: float = 300.0,
    freq_step_mhz: float = 0.3,
    bin_offset: int = 0,
) -> str:
    """Render a 2D ASCII power spectrum graph (dB vs Frequency Bins)."""
    n = len(db_values)
    width = min(width, n)
    indices = np.linspace(0, n - 1, width).astype(int)
    vals = np.array(db_values)[indices]

    max_db = float(np.max(vals))
    min_db = float(max(np.min(vals), max_db - 35.0))
    if max_db - min_db < 1.0:
        min_db = max_db - 10.0

    lines = []
    lines.append("   dB |" + " " * width)

    db_step = (max_db - min_db) / max(1, height)
    for row in range(height, -1, -1):
        level_db = min_db + row * db_step
        row_chars = []
        for v in vals:
            h = (v - min_db) / (max_db - min_db) * height
            if h >= row:
                row_chars.append("█")
            elif h >= row - 0.5:
                row_chars.append("▄")
            else:
                row_chars.append(" ")
        lines.append(f"{level_db:5.1f} |" + "".join(row_chars))

    lines.append("      +" + "-" * width + ">")

    # Tick marks for frequency and bins
    b_start = bin_offset
    b_end = bin_offset + n - 1
    b_mid = bin_offset + n // 2
    f_start = freq_start_mhz + b_start * freq_step_mhz
    f_end = freq_start_mhz + b_end * freq_step_mhz
    f_mid = 0.5 * (f_start + f_end)

    tick_line = f"  Bin: {b_start}" + f"{b_mid}".center(width - 10) + f"{b_end}"
    freq_line = f"  MHz: {f_start:.1f}" + f"{f_mid:.1f} MHz".center(width - 18) + f"{f_end:.1f}"
    lines.append(tick_line)
    lines.append(freq_line)
    return "\n".join(lines)


def render_horizontal_spectrum(
    db_values: np.ndarray,
    n_bands: int = 16,
    freq_start_mhz: float = 300.0,
    freq_step_mhz: float = 0.3,
) -> str:
    """Render horizontal ASCII power breakdown by frequency sub-bands."""
    n = len(db_values)
    band_size = n // n_bands
    lines = []
    lines.append("  " + "-" * 72)
    lines.append(f"  {'Bin Range':<15} {'Center Freq':<14} {'Power (dB)':<12} {'ASCII Power Bar'}")
    lines.append("  " + "-" * 72)

    max_db = float(np.max(db_values))
    min_db = float(max(np.min(db_values), max_db - 35.0))
    if max_db - min_db < 1.0:
        min_db = max_db - 10.0

    peak_band_idx = -1
    highest_band_p = -1e9
    band_records = []

    for b in range(n_bands):
        b_start = b * band_size
        b_end = (b + 1) * band_size if b < n_bands - 1 else n
        sub_vals = db_values[b_start:b_end]
        mean_p = float(np.mean(sub_vals))
        f_center = freq_start_mhz + (b_start + b_end) * 0.5 * freq_step_mhz
        band_records.append((b_start, b_end - 1, f_center, mean_p))
        if mean_p > highest_band_p:
            highest_band_p = mean_p
            peak_band_idx = b

    for idx, (b0, b1, fc, p_db) in enumerate(band_records):
        norm = (p_db - min_db) / (max_db - min_db)
        bar_len = 24
        filled = min(bar_len, max(0, int(norm * bar_len)))
        bar = "█" * filled + "░" * (bar_len - filled)
        tag = " [PEAK]" if idx == peak_band_idx else ""
        lines.append(f"  Bin {b0:03d}..{b1:03d}     [{fc:6.1f} MHz]    {p_db:6.1f} dB    | {bar}{tag}")

    lines.append("  " + "-" * 72)
    return "\n".join(lines)


def spectrum_loop(
    client: KotekanTrackerClient,
    beam_id: int = 0,
    buffer_name: str = "host_formed_beams_buffer",
    interval_s: float = 1.0,
    height: int = 10,
    width: int = 60,
    horizontal: bool = False,
    once: bool = False,
    freq_start_mhz: float = 300.0,
    freq_step_mhz: float = 0.3,
    bins: Optional[str] = None,
):
    """Continuously poll formed beams and display ASCII spectrum with dB and frequency bins."""
    try:
        while True:
            t0 = time.perf_counter()
            try:
                raw_data = client.get_inspect_frame(buffer_name)
                t_fetch_ms = (time.perf_counter() - t0) * 1000.0

                data = np.frombuffer(raw_data, dtype=np.complex64)
                n_samples = len(data)

                try:
                    st = client.get_status()
                    n_beams = st.get("num_active_beams", 4)
                except Exception:
                    n_beams = 4

                if beam_id >= n_beams:
                    beam_to_plot = 0
                else:
                    beam_to_plot = beam_id

                # Data layout: [time, freq (672), max_beams (4)]
                n_freq = 672
                max_beams = 4
                n_time = n_samples // (n_freq * max_beams)

                if n_time > 0:
                    reshaped = data[: n_time * n_freq * max_beams].reshape((n_time, n_freq, max_beams))
                    beam_v = reshaped[:, :, beam_to_plot]
                    power_per_freq = np.mean(np.abs(beam_v) ** 2, axis=0)
                else:
                    n_freq = min(n_samples // max(1, n_beams), 672)
                    beam_slice = data[:n_freq]
                    power_per_freq = np.abs(beam_slice) ** 2

                db_per_freq = 10.0 * np.log10(power_per_freq + 1e-12)

                # Optional zoom into specific bin range
                bin_offset = 0
                if bins:
                    try:
                        b_parts = bins.split(":")
                        b0 = max(0, int(b_parts[0]))
                        b1 = min(len(db_per_freq), int(b_parts[1])) if len(b_parts) > 1 and b_parts[1] else len(db_per_freq)
                        if b1 > b0:
                            db_per_freq = db_per_freq[b0:b1]
                            bin_offset = b0
                    except Exception as ex:
                        print(f"[WARN] Invalid --bins '{bins}': {ex}")

                peak_bin_rel = int(np.argmax(db_per_freq))
                peak_bin = bin_offset + peak_bin_rel
                peak_db = float(db_per_freq[peak_bin_rel])
                mean_db = float(np.mean(db_per_freq))
                min_db = float(np.min(db_per_freq))

                peak_freq_mhz = freq_start_mhz + peak_bin * freq_step_mhz

                if not once:
                    os.system("clear" if os.name == "posix" else "cls")

                now_str = time.strftime("%H:%M:%S")
                print("=" * 80)
                print(" CHARTS 32-ANTENNA BEAM TRACKER ASCII FREQUENCY SPECTRUM")
                zoom_info = f" | Bins: {bin_offset}..{bin_offset+len(db_per_freq)-1}" if bins else ""
                print(f" Time: {now_str} | Beam: {beam_to_plot} of {n_beams} active{zoom_info} | Frame: {len(raw_data)/1024/1024:.2f} MB | Latency: {t_fetch_ms:.1f} ms")
                print("=" * 80)

                # Render 2D Vertical Spectrum Graph
                print(render_ascii_spectrum(db_per_freq, height=height, width=width, freq_start_mhz=freq_start_mhz, freq_step_mhz=freq_step_mhz, bin_offset=bin_offset))

                # Render Horizontal Table if requested
                if horizontal:
                    print()
                    print(render_horizontal_spectrum(db_per_freq, n_bands=min(16, len(db_per_freq)), freq_start_mhz=freq_start_mhz + bin_offset * freq_step_mhz, freq_step_mhz=freq_step_mhz))

                print()
                print(f"  [Summary] Peak Bin : {peak_bin:03d} ({peak_freq_mhz:6.1f} MHz) -> {peak_db:6.1f} dB")
                print(f"            Mean/RMS : {mean_db:6.1f} dB | Floor: {min_db:6.1f} dB | Dynamic Range: {peak_db - min_db:5.1f} dB")
                print("=" * 80)
                if not once:
                    print(" Press Ctrl+C to stop. Refreshing every {:.1f}s...".format(interval_s))

            except urllib.error.HTTPError as e:
                if e.code == 404:
                    print(f"[{time.strftime('%H:%M:%S')}] Waiting for beam tracker formed beam frames... (HTTP 404)")
                else:
                    print(f"[ERROR] HTTP {e.code}: {e.reason}")
            except Exception as e:
                print(f"[{time.strftime('%H:%M:%S')}] Waiting for spectrum stream: {e}")

            if once:
                break
            time.sleep(interval_s)
    except KeyboardInterrupt:
        print("\nExiting spectrum monitor.")


def interactive_mode(client: KotekanTrackerClient):
    """Interactive steering prompt."""
    print("=" * 80)
    print(" Kotekan Beam Tracker Interactive Steering Console")
    print(" Commands:")
    print("   l <beam_id> <l0> <m0> [dl] [dm]   : Steer beam in direction cosines")
    print("   radec <beam_id> <ra> <dec>        : Steer beam to celestial RA/Dec")
    print("   nbeams <count>                    : Set number of active beams (1..8)")
    print("   mask <ant_id> [0|1]               : Enable/disable antenna (1=alive, 0=dead)")
    print("   status                            : Query status")
    print("   quit                              : Exit")
    print("=" * 80)

    while True:
        try:
            line = input("kotekan-tracker> ").strip()
            if not line:
                continue
            parts = line.split()
            cmd = parts[0].lower()

            if cmd in ("quit", "exit", "q"):
                break
            elif cmd == "status":
                t0 = time.perf_counter()
                st = client.get_status()
                t1 = time.perf_counter()
                print_status_dashboard(st, (t1 - t0) * 1000.0)
            elif cmd == "l":
                if len(parts) < 4:
                    print("Usage: l <beam_id> <l0> <m0> [dl] [dm]")
                    continue
                beam_id = int(parts[1])
                l0 = float(parts[2])
                m0 = float(parts[3])
                dl = float(parts[4]) if len(parts) > 4 else 0.0
                dm = float(parts[5]) if len(parts) > 5 else 0.0
                resp = client.steer_lm(beam_id, l0, m0, dl, dm)
                print(f" -> {resp}")
            elif cmd == "radec":
                if len(parts) < 4:
                    print("Usage: radec <beam_id> <ra_deg> <dec_deg>")
                    continue
                beam_id = int(parts[1])
                ra = float(parts[2])
                dec = float(parts[3])
                resp = client.steer_radec(beam_id, ra, dec)
                print(f" -> {resp}")
            elif cmd == "nbeams":
                if len(parts) < 2:
                    print("Usage: nbeams <count>")
                    continue
                resp = client.enable_beams(int(parts[1]))
                print(f" -> {resp}")
            elif cmd == "mask":
                if len(parts) < 3:
                    print("Usage: mask <ant_id> <0|1>")
                    continue
                ant_id = int(parts[1])
                enabled = bool(int(parts[2]))
                resp = client.mask_antenna(ant_id, enabled)
                print(f" -> {resp}")
            else:
                print(f"Unknown command: {cmd}. Type 'status', 'l', 'radec', 'nbeams', 'mask', or 'quit'.")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Kotekan Beam Tracker Live Steering & Health Monitor")
    parser.add_argument("--host", default="127.0.0.1", help="Kotekan host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=12048, help="Kotekan REST port (default: 12048)")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Status command
    subparsers.add_parser("status", help="Query and display live status")

    # Steer (l, m) command
    p_lm = subparsers.add_parser("steer-lm", help="Set trajectory using direction cosines (l, m)")
    p_lm.add_argument("--beam", type=int, default=0, help="Beam ID (0..7)")
    p_lm.add_argument("--l0", type=float, required=True, help="Direction cosine l0 (East)")
    p_lm.add_argument("--m0", type=float, required=True, help="Direction cosine m0 (North)")
    p_lm.add_argument("--dl", type=float, default=0.0, help="Direction rate dl/sample")
    p_lm.add_argument("--dm", type=float, default=0.0, help="Direction rate dm/sample")

    # Steer (RA, Dec) command
    p_radec = subparsers.add_parser("steer-radec", help="Set trajectory using celestial coordinates (RA, Dec)")
    p_radec.add_argument("--beam", type=int, default=0, help="Beam ID (0..7)")
    p_radec.add_argument("--ra", type=float, required=True, help="Right Ascension in degrees")
    p_radec.add_argument("--dec", type=float, required=True, help="Declination in degrees")
    p_radec.add_argument("--lst", type=float, default=None, help="Local Sidereal Time in hours")
    p_radec.add_argument("--now", action="store_true", help="Use current UTC time for LST calculation")

    # Enable beams command
    p_eb = subparsers.add_parser("enable-beams", help="Set number of active beams")
    p_eb.add_argument("--count", type=int, required=True, help="Active beam count (1..8)")

    # Mask antenna command
    p_ma = subparsers.add_parser("mask-antenna", help="Mask or unmask antenna element")
    p_ma.add_argument("--id", type=int, required=True, help="Antenna ID (0..255)")
    p_ma.add_argument("--enable", action="store_true", default=True, help="Enable/unmask antenna")
    p_ma.add_argument("--disable", action="store_false", dest="enable", help="Disable/mask antenna")

    # Auto-mask command
    p_am = subparsers.add_parser("auto-mask", help="Automatically detect and mask dead/unplugged antennas")
    p_am.add_argument("--h5-path", type=str, default=str(get_default_charts_h5_path()), help="Path to baseband data")
    p_am.add_argument("--threshold", type=float, default=0.05, help="Power threshold for dead antenna detection")

    # Set working antennas command
    p_swa = subparsers.add_parser("set-working-antennas", help="Activate working antennas (e.g. raw 24:31 or physical 0:7)")
    p_swa.add_argument("--raw", type=str, default="24:31", help="Raw elements range (e.g. 24:31) or comma-separated list")
    p_swa.add_argument("--physical", type=str, default=None, help="Physical antennas range (e.g. 0:7) or comma-separated list")

    # Watch command
    p_watch = subparsers.add_parser("watch", help="Watch status in live-updating dashboard")
    p_watch.add_argument("--interval", type=float, default=1.0, help="Update interval in seconds")

    # Stream command
    p_stream = subparsers.add_parser("stream", help="Stream live formed beam RF metrics and powers in terminal")
    p_stream.add_argument("--interval", type=float, default=1.0, help="Poll interval in seconds (default: 1.0)")
    p_stream.add_argument("--buffer", type=str, default="host_formed_beams_buffer", help="Buffer name to inspect")

    # Spectrum command
    p_spec = subparsers.add_parser("spectrum", help="Display live ASCII frequency spectrum (dB vs frequency bins)")
    p_spec.add_argument("--beam", type=int, default=0, help="Beam index to plot (default: 0)")
    p_spec.add_argument("--interval", type=float, default=1.0, help="Refresh interval in seconds (default: 1.0)")
    p_spec.add_argument("--height", type=int, default=10, help="Graph height in text lines (default: 10)")
    p_spec.add_argument("--width", type=int, default=60, help="Graph width in columns (default: 60)")
    p_spec.add_argument("--bins", type=str, default=None, help="Zoom into bin range e.g. 300:380 (1-channel resolution)")
    p_spec.add_argument("--horizontal", action="store_true", help="Also display horizontal frequency sub-band table")
    p_spec.add_argument("--once", action="store_true", help="Print single snapshot and exit")
    p_spec.add_argument("--freq-start", type=float, default=300.0, help="Band start frequency in MHz (default: 300.0)")
    p_spec.add_argument("--freq-step", type=float, default=0.3, help="Channel width in MHz (default: 0.3)")
    p_spec.add_argument("--buffer", type=str, default="host_formed_beams_buffer", help="Buffer name to inspect")

    # Interactive command
    subparsers.add_parser("interactive", help="Start interactive steering console")

    # Mask status command
    subparsers.add_parser("mask-status", help="Display live antenna health and masking metrics (/antenna_mask/status)")

    # Reset mask command
    subparsers.add_parser("reset-mask", help="Reset all antenna masks to active (/antenna_mask/reset)")

    args = parser.parse_args()
    client = KotekanTrackerClient(host=args.host, port=args.port)

    if args.command == "status" or args.command is None:
        t0 = time.perf_counter()
        st = client.get_status()
        t1 = time.perf_counter()
        print_status_dashboard(st, (t1 - t0) * 1000.0)
    elif args.command == "mask-status":
        try:
            m_status = client.get_antenna_mask_status()
            print_antenna_mask_dashboard(m_status)
        except Exception as e:
            print(f"[ERROR] Failed to query /antenna_mask/status: {e}")
    elif args.command == "reset-mask":
        try:
            resp = client.reset_antenna_masks()
            print(f"Success: {resp}")
        except Exception as e:
            print(f"[ERROR] Failed to reset masks: {e}")
    elif args.command == "steer-lm":
        resp = client.steer_lm(args.beam, args.l0, args.m0, args.dl, args.dm)
        print(f"Success: {resp}")
    elif args.command == "steer-radec":
        resp = client.steer_radec(args.beam, args.ra, args.dec, args.lst)
        print(f"Success: {resp}")
    elif args.command == "enable-beams":
        resp = client.enable_beams(args.count)
        print(f"Success: {resp}")
    elif args.command == "mask-antenna":
        resp = client.mask_antenna(args.id, args.enable)
        print(f"Success: {resp}")
    elif args.command == "auto-mask":
        resp = client.auto_mask(args.h5_path, args.threshold)
        print(f"Success: {resp}")
    elif args.command == "set-working-antennas":
        if args.physical:
            if ":" in args.physical:
                p0, p1 = [int(x) for x in args.physical.split(":")]
                phys_list = list(range(p0, p1 + 1))
            else:
                phys_list = [int(x) for x in args.physical.split(",")]
            raw_list = [31 - p for p in phys_list]
        else:
            if ":" in args.raw:
                r0, r1 = [int(x) for x in args.raw.split(":")]
                raw_list = list(range(r0, r1 + 1))
            else:
                raw_list = [int(x) for x in args.raw.split(",")]
        resp = client.set_working_antennas(raw_list)
        print(f"Success: Activated raw elements {raw_list} -> physical antennas {[31 - r for r in raw_list]}")
    elif args.command == "watch":
        watch_loop(client, args.interval)
    elif args.command == "stream":
        stream_loop(client, args.buffer, args.interval)
    elif args.command == "spectrum":
        spectrum_loop(
            client,
            beam_id=args.beam,
            buffer_name=args.buffer,
            interval_s=args.interval,
            height=args.height,
            width=args.width,
            horizontal=args.horizontal,
            once=args.once,
            freq_start_mhz=args.freq_start,
            freq_step_mhz=args.freq_step,
            bins=args.bins,
        )
    elif args.command == "interactive":
        interactive_mode(client)


if __name__ == "__main__":
    main()
