#!/usr/bin/env python3
"""CHARTS Baseband Simulation Generator.

Generates physically realistic complex baseband voltage streams (int4x2 format)
for the CHARTS array:
  - Daytime / Nighttime astronomical backgrounds via `ChartsNoiseModel`
  - Randomized transient events: Fast Radio Bursts (dispersed chirps), pulsars,
    persistent narrowband site RFI, and fast LEO satellite sweeps
  - Multiprocessing chunked rendering with smart decimation schedule
  - Outputs sequential Kotekan-compatible .bin frames, companion HDF5 metadata,
    and event catalog JSON.
"""

from __future__ import annotations

import datetime
import json
import math
import multiprocessing as mp
from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np

from .constants import (
    C_LIGHT,
    CHARTS_CHANNEL_WIDTH_MHZ,
    CHARTS_LATITUDE_DEG,
    CHARTS_LONGITUDE_DEG,
    DEFAULT_FREQUENCY_START_MHZ,
    DEFAULT_SPACING_M,
    DIGITIZER_NOMINAL_SIGMA_LSB,
    FPGA_TIME_RESOLUTION_US,
    K_DM,
    LOCAL_FREQUENCY_CHANNELS,
    get_antenna_positions,
)
from .astro import datetime_to_lst_hours, parse_observation_time
from .noise_model import AnalogChainParams, ChartsNoiseModel
from .presets import SimulationConfig
from .sky import (
    direction_cosines_track,
    geometric_phase_track,
    load_verified_catalog,
    resolve_window_start,
)
from .writer import (
    BasebandWriter,
    RawBinWriter,
    WindowMetadata,
)


@dataclass
class SimulatedEvent:
    event_id: str
    event_type: str  # 'frb', 'pulsar', 'rfi_narrow', 'rfi_leo'
    t_start_s: float
    duration_s: float
    dm: float = 0.0
    nominal_amp: float = 3.0
    l0: float = 0.0
    m0: float = 0.0
    drift_dl: float = 0.0
    drift_dm: float = 0.0
    pulse_period_s: float = 0.0
    pulse_width_ms: float = 1.0
    channels: Optional[List[int]] = None
    is_persistent: bool = False
    description: str = ""
    ra_deg: Optional[float] = None
    dec_deg: Optional[float] = None
    source_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def schedule_random_events(
    duration_s: float = 60.0,
    num_events: int = 3,
    allowed_types: Optional[List[str]] = None,
    seed: int = 42,
    persistent_rfi_channels: Optional[List[int]] = None,
    persistent_rfi_amp: float = 7.0,
    num_freq: int = LOCAL_FREQUENCY_CHANNELS,
    obs_time: Optional[Any] = None,
) -> List[SimulatedEvent]:
    """Generates randomized schedule of transient events plus site RFI."""
    rng = np.random.default_rng(seed)
    if allowed_types is None:
        allowed_types = ["frb", "pulsar", "rfi_narrow", "rfi_leo"]

    unix_t0: Optional[float] = None
    if obs_time is not None:
        try:
            unix_t0 = resolve_window_start(obs_time).timestamp()
        except Exception:
            unix_t0 = None

    events: List[SimulatedEvent] = []

    # 1. Site Persistent Narrowband RFI
    if persistent_rfi_channels:
        az = float(rng.uniform(0.0, 2.0 * math.pi))
        l0 = float(0.96 * math.sin(az))
        m0 = float(0.96 * math.cos(az))
        chans = sorted(persistent_rfi_channels)
        freqs_str = ", ".join(f"{DEFAULT_FREQUENCY_START_MHZ + ch * CHARTS_CHANNEL_WIDTH_MHZ:.1f} MHz" for ch in chans)
        events.append(
            SimulatedEvent(
                event_id="site_persistent_rfi",
                event_type="rfi_narrow",
                t_start_s=0.0,
                duration_s=float(duration_s),
                nominal_amp=float(persistent_rfi_amp),
                l0=l0,
                m0=m0,
                channels=chans,
                is_persistent=True,
                description=f"Site continuous RFI on channels {chans} ({freqs_str}) (amp={persistent_rfi_amp:.1f} LSB, 100% duty)",
            )
        )

    # 2. Transient Events
    margin_start = min(10.0, duration_s * 0.05)
    margin_end = max(margin_start + 0.01, duration_s - min(15.0, duration_s * 0.1))
    min_separation = max(0.01, (margin_end - margin_start) / max(1, num_events + 1))
    t_candidates = []
    for _ in range(500):
        if len(t_candidates) >= num_events:
            break
        t_prop = rng.uniform(margin_start, margin_end)
        if all(abs(t_prop - tc) >= min_separation for tc in t_candidates):
            t_candidates.append(t_prop)

    t_candidates.sort()

    for idx, t0 in enumerate(t_candidates):
        ev_type = rng.choice(allowed_types)
        ev_id = f"evt_{idx + 1:02d}_{ev_type}"

        if ev_type == "frb":
            dm = float(rng.uniform(100.0, 350.0))
            sweep_s = 4148.8 * dm * (1.0 / (300.0 ** 2) - 1.0 / (400.5 ** 2))
            width_ms = float(rng.uniform(1.2, 3.5))
            amp = float(rng.uniform(2.8, 4.5))
            ra_deg = float(rng.uniform(0.0, 360.0))
            dec_deg = float(rng.uniform(-65.0, 10.0))
            if unix_t0 is not None:
                l_arr, m_arr, _ = direction_cosines_track(ra_deg, dec_deg, np.array([unix_t0 + t0]))
                l0 = float(l_arr[0])
                m0 = float(m_arr[0])
            else:
                l0 = float(rng.normal(0.0, 0.08))
                m0 = float(rng.normal(0.0, 0.08))

            events.append(
                SimulatedEvent(
                    event_id=ev_id,
                    event_type="frb",
                    t_start_s=float(t0),
                    duration_s=float(sweep_s + 0.5),
                    dm=dm,
                    nominal_amp=amp,
                    l0=l0,
                    m0=m0,
                    pulse_width_ms=width_ms,
                    ra_deg=ra_deg,
                    dec_deg=dec_deg,
                    source_name="Synthetic FRB",
                    description=f"Synthetic FRB (DM={dm:.1f} pc/cm^3, width={width_ms:.2f} ms, sweep={sweep_s:.2f} s)",
                )
            )

        elif ev_type == "pulsar":
            is_msp = rng.choice([True, False])
            if is_msp:
                period_s = 0.005757  # PSR J0437-4715
                dm = 2.64
                width_ms = 0.18
                amp = float(rng.uniform(1.8, 2.8))
                duration = min(duration_s - t0, 15.0)
                desc = f"PSR J0437-4715 MSP train (P={period_s*1e3:.2f} ms, DM={dm:.2f})"
                ra_deg = 69.316
                dec_deg = -47.252
                src_name = "PSR J0437-4715"
            else:
                period_s = 0.08933  # Vela Pulsar
                dm = 67.99
                width_ms = 2.1
                amp = float(rng.uniform(2.2, 3.2))
                duration = min(duration_s - t0, 15.0)
                desc = f"Vela Pulsar pulse train (P={period_s*1e3:.2f} ms, DM={dm:.2f})"
                ra_deg = 128.836
                dec_deg = -45.176
                src_name = "Vela SNR / PSR B0833-45"

            if unix_t0 is not None:
                l_arr, m_arr, _ = direction_cosines_track(ra_deg, dec_deg, np.array([unix_t0 + t0]))
                l0 = float(l_arr[0])
                m0 = float(m_arr[0])
            else:
                l0 = float(rng.normal(0.0, 0.06))
                m0 = float(rng.normal(-0.15, 0.06))

            events.append(
                SimulatedEvent(
                    event_id=ev_id,
                    event_type="pulsar",
                    t_start_s=float(t0),
                    duration_s=duration,
                    dm=dm,
                    nominal_amp=amp,
                    l0=l0,
                    m0=m0,
                    pulse_period_s=period_s,
                    pulse_width_ms=width_ms,
                    ra_deg=ra_deg,
                    dec_deg=dec_deg,
                    source_name=src_name,
                    description=desc,
                )
            )

        elif ev_type == "rfi_narrow":
            num_bad_chans = int(rng.integers(1, min(4, max(2, num_freq))))
            chans = sorted(rng.choice(num_freq, size=min(num_bad_chans, num_freq), replace=False).tolist())
            duration = float(min(duration_s - t0, rng.uniform(15.0, 30.0)))
            amp = float(rng.uniform(3.5, 7.5))
            az = rng.uniform(0.0, 2.0 * math.pi)
            l0 = float(0.96 * math.sin(az))
            m0 = float(0.96 * math.cos(az))

            events.append(
                SimulatedEvent(
                    event_id=ev_id,
                    event_type="rfi_narrow",
                    t_start_s=float(t0),
                    duration_s=duration,
                    nominal_amp=amp,
                    l0=l0,
                    m0=m0,
                    channels=chans,
                    description=f"Narrowband transient RFI on channels {chans} (amp={amp:.1f} LSB, dur={duration:.1f} s)",
                )
            )

        elif ev_type == "rfi_leo":
            duration = float(min(duration_s - t0, rng.uniform(4.0, 7.0)))
            amp = float(rng.uniform(10.0, 14.0))
            drift_speed = float(rng.uniform(0.015, 0.035))
            drift_ang = rng.uniform(0.0, 2.0 * math.pi)
            dl = float(drift_speed * math.cos(drift_ang))
            dm = float(drift_speed * math.sin(drift_ang))
            l0 = float(rng.uniform(-0.1, 0.1))
            m0 = float(rng.uniform(-0.1, 0.1))

            events.append(
                SimulatedEvent(
                    event_id=ev_id,
                    event_type="rfi_leo",
                    t_start_s=float(t0),
                    duration_s=duration,
                    nominal_amp=amp,
                    l0=l0,
                    m0=m0,
                    drift_dl=dl,
                    drift_dm=dm,
                    description=f"LEO Satellite RFI sweep (amp={amp:.1f} LSB, dur={duration:.1f} s, drift={drift_speed:.3f}/s)",
                )
            )

    return events


def build_frame_selection_schedule(
    total_frames: int,
    frame_duration_s: float,
    events: List[SimulatedEvent],
    background_cadence_s: float = 2.0,
    event_dense_s: float = 1.0,
    event_sparse_cadence_ms: float = 51.2,
) -> Tuple[List[int], Dict[int, List[str]]]:
    """Determines which frame indices to write to disk."""
    selected_set = set()
    frame_events: Dict[int, List[str]] = {}

    # 1. Periodic background
    bg_step = max(1, int(round(background_cadence_s / frame_duration_s)))
    for k in range(0, total_frames, bg_step):
        selected_set.add(k)

    # 2. Event captures
    dense_frames_count = max(1, int(round(event_dense_s / frame_duration_s)))
    sparse_step = max(1, int(round((event_sparse_cadence_ms * 1e-3) / frame_duration_s)))

    for ev in events:
        if ev.is_persistent:
            continue

        t_start = ev.t_start_s
        t_end = t_start + ev.duration_s
        k_start = max(0, int(math.floor(t_start / frame_duration_s)))
        k_end = min(total_frames, int(math.ceil(t_end / frame_duration_s)))

        k_dense_end = min(k_end, k_start + dense_frames_count)
        for k in range(k_start, k_dense_end):
            selected_set.add(k)
            frame_events.setdefault(k, []).append(ev.event_id)

        for k in range(k_dense_end, k_end, sparse_step):
            selected_set.add(k)
            frame_events.setdefault(k, []).append(ev.event_id)

    persistent_evs = [ev.event_id for ev in events if ev.is_persistent]
    for k in selected_set:
        for p_id in persistent_evs:
            frame_events.setdefault(k, []).append(p_id)

    sorted_indices = sorted(selected_set)
    return sorted_indices, frame_events


def render_and_write_frame(job: Dict[str, Any]) -> Tuple[int, int, float, float, float]:
    """Renders 4-bit complex baseband voltage for one frame and writes .bin directly."""
    out_idx = job["out_idx"]
    frame_idx = job["frame_idx"]
    out_file_path = job["out_file_path"]

    num_ant = job["num_antennas"]
    num_freq = job["num_freq"]
    samples_per_frame = job["samples_per_frame"]
    dt_s = job["dt_s"]
    t0_s = job["t_start_s"]
    unix_start_s = float(job.get("unix_start_s", 0.0))
    lat_deg = float(job.get("site_lat_deg", CHARTS_LATITUDE_DEG))
    lon_deg = float(job.get("site_lon_deg", CHARTS_LONGITUDE_DEG))
    freqs_hz = job["freqs_hz"]
    pos_x = job["pos_x"]
    pos_y = job["pos_y"]
    sigma_ant = job["sigma_ant"]
    bandpass = job["bandpass"]
    sun_info = job["sun_info"]
    active_events = job["active_events"]

    rng = np.random.default_rng(job["seed"])
    c_inv = np.float32(1.0 / C_LIGHT)
    two_pi = np.float32(2.0 * np.pi)

    chunk_size = 512
    packed_frame = np.empty((samples_per_frame, num_freq, num_ant), dtype=np.uint8)

    total_power_sum = 0.0
    max_power_val = 0.0
    total_clipped = 0
    total_elements = samples_per_frame * num_freq * num_ant * 2

    noise_sigma = (sigma_ant[None, None, :] * bandpass[None, :, None]).astype(np.float32)
    f_top_hz = freqs_hz[-1]
    f_top_ghz = f_top_hz / 1e9

    for c0 in range(0, samples_per_frame, chunk_size):
        c1 = min(samples_per_frame, c0 + chunk_size)
        n_chunk = c1 - c0

        t_indices = np.arange(c0, c1, dtype=np.float64)
        t_abs_s = t0_s + t_indices * dt_s
        t_utc_s = unix_start_s + t_abs_s

        # 1. Independent receiver & sky thermal noise (direct float32)
        v_real = rng.standard_normal(size=(n_chunk, num_freq, num_ant), dtype=np.float32) * noise_sigma
        v_imag = rng.standard_normal(size=(n_chunk, num_freq, num_ant), dtype=np.float32) * noise_sigma

        # 2. Coherent Sun emission if up
        if sun_info is not None and sun_info["amp"] > 0.005:
            sun_amp = np.float32(sun_info["amp"])
            sun_ra = sun_info.get("ra_deg")
            sun_dec = sun_info.get("dec_deg")
            if sun_ra is not None and sun_dec is not None and unix_start_s > 0:
                sun_l, sun_m, _ = direction_cosines_track(sun_ra, sun_dec, t_utc_s, lat_deg, lon_deg)
                sun_delays = (sun_l[:, None] * pos_x[None, :] + sun_m[:, None] * pos_y[None, :]) * c_inv
                geom_phase = two_pi * (sun_delays[:, None, :] * freqs_hz[None, :, None])
            else:
                sun_l = np.float32(sun_info["l"])
                sun_m = np.float32(sun_info["m"])
                sun_delays = (sun_l * pos_x + sun_m * pos_y) * c_inv
                geom_phase = two_pi * (sun_delays[None, None, :] * freqs_hz[None, :, None])
            tot_sun_phase = -geom_phase.astype(np.float32)
            v_real += sun_amp * np.cos(tot_sun_phase)
            v_imag += sun_amp * np.sin(tot_sun_phase)

        # 3. Active Transient Events
        for ev in active_events:
            ev_type = ev["event_type"]
            ev_amp = np.float32(ev["nominal_amp"])
            ev_ra = ev.get("ra_deg")
            ev_dec = ev.get("dec_deg")

            if ev_ra is not None and ev_dec is not None and unix_start_s > 0:
                ev_l, ev_m, _ = direction_cosines_track(ev_ra, ev_dec, t_utc_s, lat_deg, lon_deg)
                delays = (ev_l[:, None] * pos_x[None, :] + ev_m[:, None] * pos_y[None, :]) * c_inv
                geom_phases = two_pi * (delays[:, None, :] * freqs_hz[None, :, None])
            elif ev_type == "rfi_leo":
                t_rel = (t_abs_s - ev["t_start_s"]).astype(np.float32)
                l_t = (np.float32(ev["l0"]) + np.float32(ev["drift_dl"]) * t_rel).astype(np.float32)
                m_t = (np.float32(ev["m0"]) + np.float32(ev["drift_dm"]) * t_rel).astype(np.float32)
                delays = (l_t[:, None] * pos_x[None, :] + m_t[:, None] * pos_y[None, :]) * c_inv
                geom_phases = two_pi * (delays[:, None, :] * freqs_hz[None, :, None])
            else:
                l_ev = np.float32(ev["l0"])
                m_ev = np.float32(ev["m0"])
                delays = (l_ev * pos_x + m_ev * pos_y) * c_inv
                geom_phases = two_pi * (delays[None, None, :] * freqs_hz[None, :, None])

            tot_phase = -geom_phases.astype(np.float32)

            if ev_type == "frb":
                dm = ev["dm"]
                width_s = np.float32(ev["pulse_width_ms"] * 1e-3)
                t_arrival_top = ev["t_start_s"]
                f_ghz = freqs_hz / 1e9
                dm_delays_s = ((K_DM * 1e-6) * dm * (1.0 / (f_ghz ** 2) - 1.0 / (f_top_ghz ** 2))).astype(np.float32)

                t_chan_center = t_arrival_top + dm_delays_s[None, :]
                t_diff = (t_abs_s[:, None] - t_chan_center).astype(np.float32)
                envelope = np.exp(-0.5 * (t_diff / width_s) ** 2).astype(np.float32)

                v_real += ev_amp * envelope[:, :, None] * np.cos(tot_phase)
                v_imag += ev_amp * envelope[:, :, None] * np.sin(tot_phase)

            elif ev_type == "pulsar":
                period_s = ev["pulse_period_s"]
                width_s = np.float32(ev["pulse_width_ms"] * 1e-3)
                dm = ev["dm"]
                f_ghz = freqs_hz / 1e9
                dm_delays_s = ((K_DM * 1e-6) * dm * (1.0 / (f_ghz ** 2) - 1.0 / (f_top_ghz ** 2))).astype(np.float32)

                t_min = t_abs_s[0] - dm_delays_s[-1] - 3.0 * width_s
                t_max = t_abs_s[-1] + 3.0 * width_s
                k_min = int(math.floor((t_min - ev["t_start_s"]) / period_s))
                k_max = int(math.ceil((t_max - ev["t_start_s"]) / period_s))

                for p_idx in range(k_min, k_max + 1):
                    t_pulse_0 = ev["t_start_s"] + p_idx * period_s
                    t_chan = t_pulse_0 + dm_delays_s[None, :]
                    t_diff = (t_abs_s[:, None] - t_chan).astype(np.float32)
                    env = np.exp(-0.5 * (t_diff / width_s) ** 2).astype(np.float32)
                    v_real += ev_amp * env[:, :, None] * np.cos(tot_phase)
                    v_imag += ev_amp * env[:, :, None] * np.sin(tot_phase)

            elif ev_type == "rfi_narrow":
                chans = [ch for ch in (ev["channels"] or []) if ch < num_freq]
                for ch in chans:
                    v_real[:, ch, :] += ev_amp * np.cos(tot_phase[:, ch, :])
                    v_imag[:, ch, :] += ev_amp * np.sin(tot_phase[:, ch, :])

            elif ev_type == "rfi_leo":
                v_real += ev_amp * np.cos(tot_phase)
                v_imag += ev_amp * np.sin(tot_phase)

        pwr_chunk = v_real ** 2 + v_imag ** 2
        total_power_sum += float(np.sum(pwr_chunk))
        max_power_val = max(max_power_val, float(np.max(pwr_chunk)))
        total_clipped += int(np.sum(np.abs(v_real) > 7.5)) + int(np.sum(np.abs(v_imag) > 7.5))

        r_quant = np.clip(np.round(v_real), -7, 7).astype(np.int8)
        i_quant = np.clip(np.round(v_imag), -7, 7).astype(np.int8)

        r_nibble = (r_quant & 0x0F).astype(np.uint8)
        i_nibble = ((i_quant & 0x0F) << 4).astype(np.uint8)

        packed_frame[c0:c1] = r_nibble | i_nibble

    with open(out_file_path, "wb") as f:
        np.uint32(0).tofile(f)
        packed_frame.tofile(f)

    mean_power = total_power_sum / (samples_per_frame * num_freq * num_ant)
    clip_frac = total_clipped / total_elements
    return out_idx, frame_idx, mean_power, max_power_val, clip_frac


def compute_analytic_lightcurve(
    duration_s: float,
    noise_power_base: float,
    sun_power_series: Union[np.ndarray, float],
    events: List[SimulatedEvent],
    freqs_hz: np.ndarray,
    cadence_hz: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Computes a 10 Hz analytic power lightcurve for the entire observation window."""
    num_points = max(1, int(round(duration_s * cadence_hz)))
    t_points = np.linspace(0.0, duration_s, num_points, endpoint=False)
    p_total = np.full(num_points, noise_power_base, dtype=np.float64)

    if np.isscalar(sun_power_series):
        p_total += float(sun_power_series)
    elif isinstance(sun_power_series, np.ndarray):
        if len(sun_power_series) == num_points:
            p_total += sun_power_series
        elif len(sun_power_series) > 0:
            sun_interp = np.interp(
                np.linspace(0, 1, num_points),
                np.linspace(0, 1, len(sun_power_series)),
                sun_power_series,
            )
            p_total += sun_interp

    f_top_hz = freqs_hz[-1]
    f_top_ghz = f_top_hz / 1e9
    f_ghz = freqs_hz / 1e9

    for ev in events:
        amp_sq = ev.nominal_amp ** 2
        t0 = ev.t_start_s
        dur = ev.duration_s

        if ev.event_type == "frb":
            dm = ev.dm
            width_s = ev.pulse_width_ms * 1e-3
            dm_delays_s = (K_DM * 1e-6) * dm * (1.0 / (f_ghz ** 2) - 1.0 / (f_top_ghz ** 2))
            for idx, t in enumerate(t_points):
                if t0 - 0.5 <= t <= t0 + dur + 0.5:
                    t_chan = t0 + dm_delays_s
                    env_sq = np.exp(-((t - t_chan) / width_s) ** 2)
                    p_total[idx] += amp_sq * float(np.mean(env_sq))

        elif ev.event_type == "pulsar":
            period_s = ev.pulse_period_s
            width_s = ev.pulse_width_ms * 1e-3
            duty_cycle = min(1.0, width_s / period_s)
            mask = (t_points >= t0) & (t_points <= t0 + dur)
            p_total[mask] += amp_sq * duty_cycle

        elif ev.event_type == "rfi_narrow":
            frac_chans = len(ev.channels or [1]) / len(freqs_hz)
            if ev.is_persistent:
                p_total += amp_sq * frac_chans
            else:
                mask = (t_points >= t0) & (t_points <= t0 + dur)
                p_total[mask] += amp_sq * frac_chans

        elif ev.event_type == "rfi_leo":
            mask = (t_points >= t0) & (t_points <= t0 + dur)
            p_total[mask] += amp_sq

    return t_points, p_total


def generate_simulation_window(
    config: SimulationConfig,
    writer: Optional[BasebandWriter] = None,
) -> Dict[str, Any]:
    """Master generator function creating baseband frames, HDF5 metadata, and event JSON."""
    if writer is not None and hasattr(writer, "target_dir"):
        target_dir = Path(writer.target_dir)
    else:
        target_dir = Path(config.scratch_dir) / config.window_name
    target_dir.mkdir(parents=True, exist_ok=True)

    if writer is None:
        writer = RawBinWriter(
            target_dir=target_dir,
            window_name=config.window_name,
            samples_per_frame=config.samples_per_frame,
            num_freq=config.num_freq,
            num_elements=config.antennas,
        )

    obs_time_val = getattr(config, "start_time", None) or config.utc_hour
    dt_start = resolve_window_start(obs_time_val, duration_s=config.duration_s)
    config.initial_lst_hours = datetime_to_lst_hours(dt_start)
    unix_start_s = dt_start.timestamp()
    dt_s = FPGA_TIME_RESOLUTION_US * 1e-6
    frame_duration_s = config.samples_per_frame * dt_s
    total_window_frames = int(math.floor(config.duration_s / frame_duration_s))

    freqs_hz = (config.frequency_start_mhz + np.arange(config.num_freq) * CHARTS_CHANNEL_WIDTH_MHZ) * 1e6
    pos_x, pos_y = get_antenna_positions(num_antennas=config.antennas)

    seed = int(1000 * config.utc_hour + 42)
    noise_model = ChartsNoiseModel()
    sigma_ant_base = noise_model.generate_antenna_gain_dispersion(num_antennas=config.antennas, seed=seed)
    bandpass = noise_model.generate_bandpass_shape(freqs_hz)

    # Solar state
    dt_mid = dt_start + datetime.timedelta(seconds=config.duration_s * 0.5)
    sun_state = noise_model.system_temperature(400.0, utc_dt=dt_mid, include_sun=True)
    sun_el = float(sun_state["sun_elevation_deg"])
    sun_amp = float(noise_model.temp_to_adc_sigma(sun_state["t_sun_pb"]))
    sun_l = float(sun_state["sun_l"])
    sun_m = float(sun_state["sun_m"])

    sun_is_up = sun_el > 0.0

    # Physical Sun RA/Dec
    doy = dt_mid.timetuple().tm_yday
    gamma = 2.0 * math.pi * (doy - 1) / 365.0
    decl_deg = (
        0.006918 - 0.399912 * math.cos(gamma) + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2.0 * gamma) + 0.000907 * math.sin(2.0 * gamma)
        - 0.002697 * math.cos(3.0 * gamma) + 0.001480 * math.sin(3.0 * gamma)
    ) * (180.0 / math.pi)
    eot_min = 229.18 * (
        0.000075 + 0.001868 * math.cos(gamma) - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2.0 * gamma) - 0.040849 * math.sin(2.0 * gamma)
    )
    hour_frac = dt_mid.hour + dt_mid.minute / 60.0 + dt_mid.second / 3600.0 + dt_mid.microsecond / 3.6e9
    solar_time_h = (hour_frac + (CHARTS_LONGITUDE_DEG / 15.0) + (eot_min / 60.0)) % 24.0
    ha_deg = (solar_time_h - 12.0) * 15.0
    lst_deg = datetime_to_lst_hours(dt_mid) * 15.0
    ra_sun_deg = (lst_deg - ha_deg) % 360.0

    sun_info = (
        {
            "l": sun_l,
            "m": sun_m,
            "amp": sun_amp,
            "el_deg": sun_el,
            "t_pb_k": float(sun_state["t_sun_pb"]),
            "ra_deg": float(ra_sun_deg),
            "dec_deg": float(decl_deg),
        }
        if sun_is_up
        else None
    )

    t_noise_incoherent = float(sun_state["t_noise_incoherent"])
    sigma_noise_base = float(noise_model.temp_to_adc_sigma(t_noise_incoherent))
    sigma_ant = sigma_ant_base * sigma_noise_base

    # Resolve persistent RFI channels (filtered to valid channels within band)
    rfi_channels = config.persistent_rfi_channels
    if config.persistent_rfi_freqs:
        # Convert frequencies to channel indices
        rfi_channels = [
            int(round((f - config.frequency_start_mhz) / CHARTS_CHANNEL_WIDTH_MHZ))
            for f in config.persistent_rfi_freqs
        ]
    rfi_channels = [ch for ch in rfi_channels if 0 <= ch < config.num_freq]

    events = schedule_random_events(
        duration_s=config.duration_s,
        num_events=config.num_events,
        seed=seed + 77,
        persistent_rfi_channels=rfi_channels,
        persistent_rfi_amp=config.persistent_rfi_amp,
        allowed_types=getattr(config, "allowed_event_types", None),
        num_freq=config.num_freq,
        obs_time=obs_time_val,
    )

    written_indices, frame_events_map = build_frame_selection_schedule(
        total_frames=total_window_frames,
        frame_duration_s=frame_duration_s,
        events=events,
        background_cadence_s=config.background_cadence_s,
        event_dense_s=config.event_dense_s,
        event_sparse_cadence_ms=config.event_sparse_cadence_ms,
    )

    if getattr(config, "dry_run", False):
        written_indices = written_indices[:1]

    num_written = len(written_indices)

    print("=" * 76)
    print(f" CHARTS BASEBAND SIMULATION: {config.window_name} ({dt_start.strftime('%Y-%m-%d %H:%M:%S UTC')}, LST={config.initial_lst_hours:.3f}h)")
    print("=" * 76)
    print(f" Target Directory        : {target_dir}")
    print(f" Duration                : {config.duration_s:.1f} s ({total_window_frames} physical frames)")
    print(f" Written Frames          : {num_written} frames{' [DRY-RUN sample]' if getattr(config, 'dry_run', False) else ''}")
    print(f" Antennas                : {config.antennas}")
    print(f" Frequency Channels      : {config.num_freq} ({config.frequency_start_mhz:.1f} - {config.frequency_start_mhz + config.num_freq*CHARTS_CHANNEL_WIDTH_MHZ:.1f} MHz)")
    print(f" Sun Elevation           : {sun_el:+.2f} deg ({'UP' if sun_is_up else 'DOWN'})")
    print(f" Active Events Scheduled : {len(events)}")
    print(f" Workers                 : {config.workers}")
    print("=" * 76)

    jobs = []
    for out_idx, frame_idx in enumerate(written_indices):
        t_start_s = frame_idx * frame_duration_s
        out_bin_file = target_dir / f"{config.window_name}_{out_idx:07d}.bin"

        active_evs = []
        ev_ids = frame_events_map.get(frame_idx, [])
        for ev in events:
            if ev.event_id in ev_ids:
                active_evs.append(ev.to_dict())

        jobs.append({
            "out_idx": out_idx,
            "frame_idx": frame_idx,
            "out_file_path": str(out_bin_file),
            "num_antennas": config.antennas,
            "num_freq": config.num_freq,
            "samples_per_frame": config.samples_per_frame,
            "dt_s": dt_s,
            "t_start_s": t_start_s,
            "unix_start_s": unix_start_s,
            "site_lat_deg": CHARTS_LATITUDE_DEG,
            "site_lon_deg": CHARTS_LONGITUDE_DEG,
            "freqs_hz": freqs_hz,
            "pos_x": pos_x,
            "pos_y": pos_y,
            "sigma_ant": sigma_ant,
            "bandpass": bandpass,
            "sun_info": sun_info,
            "active_events": active_evs,
            "seed": seed + frame_idx,
        })

    t0_gen = time.perf_counter()
    pool_workers = min(config.workers, mp.cpu_count() or 4)

    stats_list = []
    if pool_workers > 1 and len(jobs) > 1:
        with mp.Pool(processes=pool_workers, maxtasksperchild=100) as pool:
            for res in pool.imap_unordered(render_and_write_frame, jobs, chunksize=1):
                stats_list.append(res)
                if len(stats_list) % max(1, num_written // 10) == 0 or len(stats_list) == num_written:
                    pct = (len(stats_list) / num_written) * 100.0
                    print(f"  Progress: {len(stats_list):4d}/{num_written} frames ({pct:.1f}%)", flush=True)
    else:
        for idx, job in enumerate(jobs):
            res = render_and_write_frame(job)
            stats_list.append(res)
            if (idx + 1) % max(1, num_written // 5) == 0 or (idx + 1) == num_written:
                pct = ((idx + 1) / num_written) * 100.0
                print(f"  Progress: {idx + 1:4d}/{num_written} frames ({pct:.1f}%)", flush=True)

    t_gen_s = time.perf_counter() - t0_gen
    stats_list.sort(key=lambda x: x[0])

    # Compute analytic lightcurve
    noise_pwr_base = float(np.mean(2.0 * (sigma_ant ** 2)) * np.mean(bandpass ** 2))
    sun_pwr_series = np.full(int(round(config.duration_s * 10.0)), sun_amp ** 2 if sun_is_up else 0.0)

    lc_t, lc_power = compute_analytic_lightcurve(
        duration_s=config.duration_s,
        noise_power_base=noise_pwr_base,
        sun_power_series=sun_pwr_series,
        events=events,
        freqs_hz=freqs_hz,
        cadence_hz=10.0,
    )

    meta_h5_path = target_dir / f"{config.window_name}_meta.h5"
    events_json_path = target_dir / f"{config.window_name}_events.json"

    written_frames_arr = np.array(written_indices, dtype=np.int32)
    written_out_arr = np.arange(num_written, dtype=np.int32)
    written_timestamps_s = written_frames_arr.astype(np.float64) * frame_duration_s

    mean_powers_arr = np.array([s[2] for s in stats_list], dtype=np.float32)
    max_powers_arr = np.array([s[3] for s in stats_list], dtype=np.float32)
    clip_fracs_arr = np.array([s[4] for s in stats_list], dtype=np.float32)

    meta = WindowMetadata(
        window_name=config.window_name,
        target_dir=target_dir,
        utc_start=dt_start.isoformat(),
        duration_s=config.duration_s,
        antennas=config.antennas,
        num_freq=config.num_freq,
        samples_per_frame=config.samples_per_frame,
        frame_duration_s=frame_duration_s,
        frequencies_mhz=freqs_hz / 1e6,
        antenna_pos_x_m=pos_x,
        antenna_pos_y_m=pos_y,
        sun_elevation_deg=sun_el,
        sun_is_up=sun_is_up,
        total_physical_frames=total_window_frames,
        num_written_frames=num_written,
        antenna_sigma_base=sigma_ant,
        bandpass_shape=bandpass,
        physical_frame_indices=written_frames_arr,
        output_file_indices=written_out_arr,
        timestamps_s=written_timestamps_s,
        mean_powers_lsb2=mean_powers_arr,
        max_powers_lsb2=max_powers_arr,
        clip_fractions=clip_fracs_arr,
        lightcurve_time_s=lc_t,
        lightcurve_power=lc_power,
        events=[ev.to_dict() for ev in events],
    )

    if writer is not None and hasattr(writer, "_written_frames"):
        for out_idx in range(num_written):
            out_bin_file = target_dir / f"{config.window_name}_{out_idx:07d}.bin"
            writer._written_frames[out_idx] = out_bin_file

    if writer is not None and hasattr(writer, "write_metadata"):
        writer.write_metadata(meta)

    manifest = None
    if writer is not None and hasattr(writer, "finalize"):
        manifest = writer.finalize()

    with open(events_json_path, "w", encoding="utf-8") as f:
        json.dump([ev.to_dict() for ev in events], f, indent=2)

    # Ensure companion h5 is written if writer didn't write it
    if not meta_h5_path.is_file():
        with h5py.File(meta_h5_path, "w") as h5:
            h5.attrs["utc_start"] = dt_start.isoformat()
            h5.attrs["duration_s"] = config.duration_s
            h5.attrs["antennas"] = config.antennas
            h5.attrs["num_freq"] = config.num_freq
            h5.attrs["samples_per_frame"] = config.samples_per_frame
            h5.attrs["frame_duration_s"] = frame_duration_s
            h5.attrs["sun_elevation_deg"] = sun_el
            h5.attrs["sun_is_up"] = sun_is_up
            h5.attrs["total_physical_frames"] = total_window_frames
            h5.attrs["num_written_frames"] = num_written

            h5.create_dataset("frequencies_mhz", data=freqs_hz / 1e6)
            h5.create_dataset("antenna_pos_x_m", data=pos_x)
            h5.create_dataset("antenna_pos_y_m", data=pos_y)
            h5.create_dataset("antenna_sigma_base", data=sigma_ant)
            h5.create_dataset("bandpass_shape", data=bandpass)

            frames_grp = h5.create_group("frames")
            frames_grp.create_dataset("physical_frame_index", data=written_frames_arr)
            frames_grp.create_dataset("output_file_index", data=written_out_arr)
            frames_grp.create_dataset("timestamp_s", data=written_timestamps_s)
            frames_grp.create_dataset("mean_power_lsb2", data=mean_powers_arr)
            frames_grp.create_dataset("max_power_lsb2", data=max_powers_arr)
            frames_grp.create_dataset("clip_fraction", data=clip_fracs_arr)

            lc_grp = h5.create_group("lightcurve")
            lc_grp.create_dataset("time_s", data=lc_t)
            lc_grp.create_dataset("power_analytic", data=lc_power)

    return {
        "target_dir": target_dir,
        "window_name": config.window_name,
        "meta_h5_path": meta_h5_path,
        "events_json_path": events_json_path,
        "num_written": num_written,
        "duration_s": config.duration_s,
        "gen_time_s": t_gen_s,
        "events": events,
        "manifest": manifest,
        "writer": writer,
    }
