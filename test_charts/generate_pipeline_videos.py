#!/usr/bin/env python3
"""
CHARTS Pipeline Video Generator
================================
Generates 2-minute (120-second) MP4 videos from CHARTS pipeline outputs (covering the full window):
  1. Baseband 8x8 spectrogram video  — frequency power per antenna over time
  2. Correlator matrix video         — 64x64 visibility matrix evolution
  3. Beam tracker output video       — formed beam power over time/frequency

Uses matplotlib animation with FFMpegWriter to produce .mp4 files.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Setup paths
_test_charts_dir = Path(__file__).resolve().parent
_kotekan_root = _test_charts_dir.parent
if str(_test_charts_dir) not in sys.path:
    sys.path.insert(0, str(_test_charts_dir))

from constants import (
    CHARTS_CHANNEL_WIDTH_MHZ,
    DEFAULT_FREQUENCY_START_MHZ,
    DEFAULT_SPACING_M,
    LOCAL_FREQUENCY_CHANNELS,
)

import multiprocessing as mp

# 4-bit two's complement LUT
INT4_LUT = np.array(
    [0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1],
    dtype=np.float64,
)

# 4-bit two's complement square LUT
INT4_SQUARES = np.array(
    [0, 1, 4, 9, 16, 25, 36, 49, 64, 49, 36, 25, 16, 9, 4, 1],
    dtype=np.float32,
)
# Precompute 256-element byte power LUT: |v|^2 = real^2 + imag^2
BYTE_POWER_LUT = np.array(
    [INT4_SQUARES[b >> 4] + INT4_SQUARES[b & 0x0F] for b in range(256)],
    dtype=np.float32,
)


def _worker_load_baseband(args):
    fpath, num_ant, num_freq, samples = args
    raw = np.fromfile(fpath, dtype=np.uint8)
    if raw.size < 4:
        raise ValueError(f"File too small: {fpath}")
    meta_size = int(np.frombuffer(raw[:4].tobytes(), dtype="<u4", count=1)[0])
    expected_bytes = samples * num_freq * num_ant
    packed = raw[4 + meta_size : 4 + meta_size + expected_bytes]
    power = BYTE_POWER_LUT[packed].reshape(samples, num_freq, num_ant)
    mean_power = np.mean(power, axis=0)  # (num_freq, num_ant)
    spec_db = 10.0 * np.log10(mean_power.T + 1e-12)  # (num_ant, num_freq)
    mean_db = np.mean(spec_db, axis=0)
    return spec_db, mean_db


def _worker_load_tracker(args):
    fpath, samples, num_freq, max_beams = args
    raw = np.fromfile(fpath, dtype=np.uint8)
    if raw.size < 4:
        raise ValueError(f"File too small: {fpath}")
    meta_size = int(np.frombuffer(raw[:4].tobytes(), dtype="<u4", count=1)[0])
    expected_floats = samples * num_freq * max_beams * 2
    expected_bytes = expected_floats * 4
    floats = np.frombuffer(
        raw[4 + meta_size : 4 + meta_size + expected_bytes], dtype="<f4"
    )
    shaped = floats.reshape(samples, num_freq, max_beams, 2)
    power = shaped[..., 0] ** 2 + shaped[..., 1] ** 2
    mean_pwr = np.mean(power, axis=0)  # (freq, beams)
    integ_pwr = np.sum(mean_pwr, axis=0)  # (beams,)
    power_db = 10.0 * np.log10(mean_pwr + 1e-12)
    integ_db = 10.0 * np.log10(integ_pwr + 1e-12)
    return power_db, integ_db


# ---------------------------------------------------------------------------
# Baseband frame loading
# ---------------------------------------------------------------------------
def load_baseband_frame(
    bin_path: Path,
    num_antennas: int = 64,
    num_freq: int = 672,
    samples_per_frame: int = 1536,
) -> np.ndarray:
    """
    Loads a raw int4x2 baseband frame and returns complex voltage array.
    Returns shape: (samples_per_frame, num_freq, num_antennas) complex128
    """
    raw = np.fromfile(bin_path, dtype=np.uint8)
    # Skip 4-byte metadata header + metadata payload
    if raw.size < 4:
        raise ValueError(f"File too small: {bin_path}")
    meta_size = int(np.frombuffer(raw[:4].tobytes(), dtype="<u4", count=1)[0])
    payload = raw[4 + meta_size :]

    expected_bytes = samples_per_frame * num_freq * num_antennas
    if payload.size < expected_bytes:
        raise ValueError(
            f"Payload {payload.size} < expected {expected_bytes} in {bin_path}"
        )

    # Each byte packs two 4-bit values: high nibble = real, low nibble = imag
    packed = payload[:expected_bytes]
    real_idx = (packed >> 4) & 0x0F
    imag_idx = packed & 0x0F
    voltages = INT4_LUT[real_idx] + 1j * INT4_LUT[imag_idx]
    return voltages.reshape(samples_per_frame, num_freq, num_antennas)


def compute_baseband_spectrogram(
    voltages: np.ndarray,
    num_antennas: int = 64,
) -> np.ndarray:
    """
    Computes per-antenna frequency power (averaged over time).
    Input: (samples, freq, antennas) complex
    Returns: (num_antennas, num_freq) power in dB
    """
    power = np.mean(np.abs(voltages) ** 2, axis=0)  # (freq, antennas)
    power_db = 10.0 * np.log10(power.T + 1e-12)  # (antennas, freq)
    return power_db


# ---------------------------------------------------------------------------
# Correlator frame loading
# ---------------------------------------------------------------------------
def load_correlator_frame(
    bin_path: Path,
    num_elements: int = 64,
    num_channels: int = 672,
    polarizations: int = 2,
) -> np.ndarray:
    """
    Loads a cudaCorrelatorAstron binary dump.
    Returns: (num_channels, num_elements, num_elements) complex128
    """
    from inspect_correlator_dump import load_astron_correlator_dump

    return load_astron_correlator_dump(
        bin_path,
        num_elements=num_elements,
        num_channels=num_channels,
        polarizations=polarizations,
    )


# ---------------------------------------------------------------------------
# Beam tracker frame loading
# ---------------------------------------------------------------------------
def load_beam_tracker_frame(
    bin_path: Path,
    samples_per_data_set: int = 1536,
    num_freq: int = 672,
    max_beams: int = 8,
) -> np.ndarray:
    """
    Loads a beam tracker output frame (complex float32 formed beams).
    Layout: [time][freq][beam] of float2 (real, imag)
    Returns: (samples_per_data_set, num_freq, max_beams) complex128
    """
    raw = np.fromfile(bin_path, dtype=np.uint8)
    if raw.size < 4:
        raise ValueError(f"File too small: {bin_path}")
    meta_size = int(np.frombuffer(raw[:4].tobytes(), dtype="<u4", count=1)[0])
    payload = raw[4 + meta_size :]

    expected_floats = samples_per_data_set * num_freq * max_beams * 2
    expected_bytes = expected_floats * 4
    if payload.size < expected_bytes:
        raise ValueError(
            f"Payload {payload.size} < expected {expected_bytes} in {bin_path}"
        )

    floats = np.frombuffer(payload[:expected_bytes], dtype="<f4")
    # Reshape to (time, freq, beams, 2) then combine to complex
    shaped = floats.reshape(samples_per_data_set, num_freq, max_beams, 2)
    return shaped[..., 0].astype(np.float64) + 1j * shaped[..., 1].astype(np.float64)


# ---------------------------------------------------------------------------
# Video 1: Baseband Frequency Power Line Spectrum Video
# ---------------------------------------------------------------------------
def generate_baseband_spectrogram_video(
    window_dir: Path,
    base_name: str,
    output_path: Path,
    num_antennas: int = 64,
    num_freq: int = 672,
    samples_per_frame: int = 1536,
    duration_s: float = 60.0,
    fps: int = 10,
    max_frames: Optional[int] = None,
    workers: int = 16,
):
    """
    Generates a video showing baseband frequency power (dB) across all antennas:
      - Top panel: Array-average spectrum (dB vs MHz) with dynamic peak marker and callout.
      - Bottom panel: 8×8 grid of individual antenna line spectra following dB power,
        making RFI spikes and transient bursts immediately visible.
    Optimized with parallel precomputation across workers for rapid rendering.
    """
    bin_files = sorted(window_dir.glob(f"{base_name}_*.bin"))
    if not bin_files:
        print(
            f"[WARNING] No baseband files found matching {base_name}_*.bin in {window_dir}"
        )
        return

    total_available = len(bin_files)
    if max_frames:
        bin_files = bin_files[:max_frames]
        total_available = len(bin_files)

    # Ensure the video spans the full window duration (duration_s seconds at given fps)
    total_video_frames = int(duration_s * fps)
    if total_available > 0:
        indices = np.linspace(0, total_available - 1, total_video_frames).astype(int)
        sampled_files = [bin_files[i] for i in indices]
    else:
        sampled_files = []
        total_video_frames = 0

    print(
        f"  Baseband video: {total_video_frames} frames ({duration_s:.1f} s @ {fps} fps) "
        f"sampled across {total_available} files spanning the entire window"
    )

    freq_mhz = (
        DEFAULT_FREQUENCY_START_MHZ + np.arange(num_freq) * CHARTS_CHANNEL_WIDTH_MHZ
    )

    # Parallel pre-extraction of all spectra across workers
    print(
        f"  Pre-extracting {total_video_frames} baseband spectra in parallel using {workers} workers..."
    )
    tasks = [(f, num_antennas, num_freq, samples_per_frame) for f in sampled_files]
    if workers > 1:
        with mp.Pool(processes=min(workers, len(tasks) or 1)) as pool:
            results = pool.map(_worker_load_baseband, tasks)
    else:
        results = [_worker_load_baseband(t) for t in tasks]

    all_specs = [r[0] for r in results]
    all_means = [r[1] for r in results]

    # Compute baseline from first frame
    first_spec = all_specs[0]  # (antennas, freq) in dB
    first_mean = all_means[0]

    # Dynamic dB limits with headroom for RFI peaks
    ymin = float(np.percentile(first_spec, 1) - 3.0)
    ymax = float(max(np.percentile(first_spec, 99.5) + 18.0, ymin + 28.0))

    # Setup figure with GridSpec: Top row = Array Mean, Bottom rows = 8x8 antenna grid
    fig = plt.figure(figsize=(20, 13), facecolor="#0a0a1a")
    fig.suptitle(
        "CHARTS Baseband Frequency Power Spectrum — Array Mean & 8×8 Antenna Grid",
        fontsize=15,
        color="white",
        y=0.98,
    )

    gs = fig.add_gridspec(
        nrows=9,
        ncols=8,
        height_ratios=[2.4, 1, 1, 1, 1, 1, 1, 1, 1],
        hspace=0.35,
        wspace=0.18,
        top=0.94,
        bottom=0.05,
        left=0.05,
        right=0.96,
    )

    # Top: Array Average Spectrum
    ax_mean = fig.add_subplot(gs[0, :])
    ax_mean.set_facecolor("#111122")
    (mean_line,) = ax_mean.plot(
        freq_mhz,
        first_mean,
        color="#00FFCC",
        linewidth=1.6,
        label="Array Mean Spectrum",
    )
    peak_idx = int(np.argmax(first_mean))
    peak_dot = ax_mean.scatter(
        [freq_mhz[peak_idx]], [first_mean[peak_idx]], color="#FFD700", s=65, zorder=5
    )
    peak_text = ax_mean.text(
        0.02,
        0.82,
        f"Array Peak: {freq_mhz[peak_idx]:.1f} MHz ({first_mean[peak_idx]:.1f} dB)",
        transform=ax_mean.transAxes,
        color="#FFD700",
        fontsize=11,
        fontweight="bold",
    )
    ax_mean.set_xlim(freq_mhz[0], freq_mhz[-1])
    ax_mean.set_ylim(ymin, ymax)
    ax_mean.set_xlabel("Frequency (MHz)", color="#C9D1D9", fontsize=10)
    ax_mean.set_ylabel("Power (dB)", color="#C9D1D9", fontsize=10)
    ax_mean.tick_params(colors="#8B949E")
    ax_mean.grid(True, color="#21262D", linestyle="--", alpha=0.7)
    for sp in ax_mean.spines.values():
        sp.set_color("#30363D")
    ax_mean.legend(
        loc="upper right",
        facecolor="#161B22",
        edgecolor="#30363D",
        labelcolor="#C9D1D9",
        fontsize=9,
    )

    # Bottom: 8×8 Antenna Line Graph Grid
    ant_lines = []
    for ant in range(num_antennas):
        row = 7 - (ant >> 3)  # Row 7 at top of antenna grid
        col = ant & 7
        ax = fig.add_subplot(gs[row + 1, col])
        ax.set_facecolor("#111122")
        (line,) = ax.plot(freq_mhz, first_spec[ant], color="#38ef7d", linewidth=0.75)
        ant_lines.append(line)
        ax.set_xlim(freq_mhz[0], freq_mhz[-1])
        ax.set_ylim(ymin, ymax)
        ax.text(
            0.06,
            0.78,
            f"A{ant}",
            transform=ax.transAxes,
            fontsize=6.5,
            color="white",
            fontweight="bold",
            alpha=0.85,
        )
        ax.grid(True, color="#21262D", linestyle=":", alpha=0.5)
        for sp in ax.spines.values():
            sp.set_color("#30363D")

        # Show ticks only on outer border
        if row == 7:  # Bottom row
            ax.tick_params(colors="#8B949E", labelsize=6)
        else:
            ax.set_xticks([])
        if col == 0:  # Leftmost column
            ax.tick_params(colors="#8B949E", labelsize=6)
        else:
            ax.set_yticks([])

    time_text = fig.text(0.5, 0.015, "", ha="center", fontsize=12, color="cyan")

    def update(frame_idx):
        cur_spec = all_specs[frame_idx]
        cur_mean = all_means[frame_idx]

        mean_line.set_ydata(cur_mean)
        p_idx = int(np.argmax(cur_mean))
        peak_text.set_text(
            f"Array Peak: {freq_mhz[p_idx]:.1f} MHz ({cur_mean[p_idx]:.1f} dB)"
        )
        peak_dot.set_offsets([[freq_mhz[p_idx], cur_mean[p_idx]]])

        for a in range(num_antennas):
            ant_lines[a].set_ydata(cur_spec[a])

        t_s = frame_idx * (duration_s / max(1, total_video_frames))
        mins = int(t_s // 60)
        secs = int(t_s % 60)
        tot_mins = int(duration_s // 60)
        tot_secs = int(duration_s % 60)
        time_text.set_text(
            f"Frame {frame_idx + 1}/{total_video_frames}  |  "
            f"Window Time: {mins:02d}:{secs:02d} / {tot_mins:02d}:{tot_secs:02d} ({t_s:.1f}s / {duration_s:.0f}s)"
        )
        return [mean_line, peak_dot, peak_text, time_text] + ant_lines

    anim = animation.FuncAnimation(fig, update, frames=total_video_frames, blit=False)
    writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(output_path), writer=writer)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Video 2: Correlator Matrix Video (2-Channel / Interesting Channel)
# ---------------------------------------------------------------------------
def find_interesting_correlator_channels(
    corr_dir: Path,
    corr_name: str,
    num_elements: int = 64,
    num_channels: int = 672,
) -> Tuple[int, int]:
    """
    Identifies:
      1. Active / interesting channel: maximum off-diagonal coherence ratio (RFI, celestial source, or event).
      2. Quiet reference channel: pure thermal noise floor baseline.
    """
    # 1. Check if events.json exists in parent or window dir
    events_file = None
    for candidate_dir in [corr_dir.parent, corr_dir]:
        ev_files = list(candidate_dir.glob("*_events.json"))
        if ev_files:
            events_file = ev_files[0]
            break

    known_rfi_channels = []
    if events_file and events_file.exists():
        try:
            import json

            with open(events_file, "r") as f:
                ev_data = json.load(f)
            for ev in ev_data:
                chans = ev.get("channels") or []
                for c in chans:
                    if 0 <= c < num_channels and c not in known_rfi_channels:
                        known_rfi_channels.append(c)
        except Exception:
            pass

    # 2. Inspect first correlator dump to compute coherence ratio across all channels
    bin_files = sorted(corr_dir.glob(f"{corr_name}_*.bin"))
    if not bin_files:
        ch_act = (
            known_rfi_channels[0] if known_rfi_channels else min(147, num_channels - 1)
        )
        ch_qui = num_channels // 2
        return ch_act, ch_qui

    first_vis = load_correlator_frame(
        bin_files[0], num_elements, num_channels
    )  # (channels, ant, ant)
    mag_sq = np.abs(first_vis) ** 2
    diag_sq = np.diagonal(mag_sq, axis1=1, axis2=2)  # (channels, ant)
    auto_pwr = np.sum(diag_sq, axis=1)
    tot_pwr = np.sum(mag_sq, axis=(1, 2))
    off_diag_pwr = tot_pwr - auto_pwr

    coherence = off_diag_pwr / np.maximum(1e-12, auto_pwr)

    # Active channel: prioritize known RFI channel if it has elevated coherence, else global maximum
    if known_rfi_channels:
        ch_active = max(known_rfi_channels, key=lambda c: coherence[c])
    else:
        ch_active = int(np.argmax(coherence))

    # Quiet channel: 25th percentile of coherence (clean noise)
    sorted_by_coh = np.argsort(coherence)
    ch_quiet = int(sorted_by_coh[len(sorted_by_coh) // 4])

    if ch_active == ch_quiet:
        ch_quiet = (ch_active + num_channels // 2) % num_channels

    return ch_active, ch_quiet


def generate_correlator_video(
    corr_dir: Path,
    corr_name: str,
    output_path: Path,
    num_elements: int = 64,
    num_channels: int = 672,
    duration_s: float = 120.0,
    fps: int = 10,
    freq_channel_idx: Optional[int] = None,
    freq_channels: Optional[List[int]] = None,
    max_frames: Optional[int] = None,
    separate_videos: bool = False,
):
    """
    Generates correlator visibility matrix evolution video:
      - By default, displays 2 channels simultaneously:
          Row 1: Active Channel (coherent RFI / sky source emission, showing off-diagonal fringes)
          Row 2: Quiet Reference Channel (pure thermal noise baseline, showing diagonal auto-power)
      - Each row displays Magnitude |V_ij| (left) and Phase arg(V_ij) (right).
    """
    bin_files = sorted(corr_dir.glob(f"{corr_name}_*.bin"))
    if not bin_files:
        print(
            f"[WARNING] No correlator files found matching {corr_name}_*.bin in {corr_dir}"
        )
        return

    total_available = len(bin_files)
    if max_frames:
        bin_files = bin_files[:max_frames]
        total_available = len(bin_files)

    # Ensure the video spans the full window duration (duration_s seconds at given fps)
    total_video_frames = int(duration_s * fps)
    if total_available > 0:
        indices = np.linspace(0, total_available - 1, total_video_frames).astype(int)
        sampled_files = [bin_files[i] for i in indices]
    else:
        sampled_files = []
        total_video_frames = 0

    # Determine frequency channels to visualize
    if freq_channels and len(freq_channels) >= 2:
        channels_to_plot = freq_channels[:2]
    elif freq_channel_idx is not None:
        channels_to_plot = [freq_channel_idx]
    else:
        ch_act, ch_qui = find_interesting_correlator_channels(
            corr_dir, corr_name, num_elements, num_channels
        )
        channels_to_plot = [ch_act, ch_qui]

    freqs_mhz = [
        DEFAULT_FREQUENCY_START_MHZ + ch * CHARTS_CHANNEL_WIDTH_MHZ
        for ch in channels_to_plot
    ]

    # Load first frame for color scales
    first_vis = load_correlator_frame(sampled_files[0], num_elements, num_channels)

    if len(channels_to_plot) >= 2:
        ch_A, ch_B = channels_to_plot[0], channels_to_plot[1]
        freq_A, freq_B = freqs_mhz[0], freqs_mhz[1]

        print(
            f"  Correlator video (2-channel mode): {total_video_frames} frames ({duration_s:.1f} s @ {fps} fps) "
            f"spanning the full window from {total_available} files"
        )
        print(f"    Channel A (Active) : Ch {ch_A} ({freq_A:.1f} MHz)")
        print(f"    Channel B (Quiet)  : Ch {ch_B} ({freq_B:.1f} MHz)")

        mag_A = np.abs(first_vis[ch_A])
        ph_A = np.angle(first_vis[ch_A])
        vmax_A = float(np.percentile(mag_A, 99.5))

        mag_B = np.abs(first_vis[ch_B])
        ph_B = np.angle(first_vis[ch_B])
        vmax_B = float(np.percentile(mag_B, 99.5))

        fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor="#0a0a1a")
        fig.suptitle(
            f"CHARTS Correlator Visibility Matrix — Dual Channel Comparison\n"
            f"Active: Ch {ch_A} ({freq_A:.1f} MHz)  vs  Quiet: Ch {ch_B} ({freq_B:.1f} MHz)",
            fontsize=14,
            color="white",
            y=0.98,
        )

        # Row 0: Active Channel
        ax_mag_0 = axes[0, 0]
        ax_mag_0.set_facecolor("#111122")
        im_mag_0 = ax_mag_0.imshow(
            mag_A, origin="lower", cmap="inferno", vmin=0.0, vmax=vmax_A
        )
        ax_mag_0.set_title(
            f"Active Ch {ch_A} ({freq_A:.1f} MHz) — |V_ij| (Magnitude)",
            color="#FFD700",
            fontsize=11,
            fontweight="bold",
        )
        ax_mag_0.set_xlabel("Antenna j", color="#C9D1D9", fontsize=9)
        ax_mag_0.set_ylabel("Antenna i", color="#C9D1D9", fontsize=9)
        ax_mag_0.tick_params(colors="#8B949E")
        cb0 = plt.colorbar(im_mag_0, ax=ax_mag_0, shrink=0.82, pad=0.02)
        cb0.ax.tick_params(colors="#8B949E")

        ax_ph_0 = axes[0, 1]
        ax_ph_0.set_facecolor("#111122")
        im_ph_0 = ax_ph_0.imshow(
            ph_A, origin="lower", cmap="twilight", vmin=-np.pi, vmax=np.pi
        )
        ax_ph_0.set_title(
            f"Active Ch {ch_A} ({freq_A:.1f} MHz) — arg(V_ij) (Phase)",
            color="#FFD700",
            fontsize=11,
            fontweight="bold",
        )
        ax_ph_0.set_xlabel("Antenna j", color="#C9D1D9", fontsize=9)
        ax_ph_0.set_ylabel("Antenna i", color="#C9D1D9", fontsize=9)
        ax_ph_0.tick_params(colors="#8B949E")
        cb1 = plt.colorbar(im_ph_0, ax=ax_ph_0, shrink=0.82, pad=0.02)
        cb1.ax.tick_params(colors="#8B949E")

        # Row 1: Quiet Channel
        ax_mag_1 = axes[1, 0]
        ax_mag_1.set_facecolor("#111122")
        im_mag_1 = ax_mag_1.imshow(
            mag_B, origin="lower", cmap="inferno", vmin=0.0, vmax=vmax_B
        )
        ax_mag_1.set_title(
            f"Quiet Ch {ch_B} ({freq_B:.1f} MHz) — |V_ij| (Magnitude)",
            color="#40C4FF",
            fontsize=11,
            fontweight="bold",
        )
        ax_mag_1.set_xlabel("Antenna j", color="#C9D1D9", fontsize=9)
        ax_mag_1.set_ylabel("Antenna i", color="#C9D1D9", fontsize=9)
        ax_mag_1.tick_params(colors="#8B949E")
        cb2 = plt.colorbar(im_mag_1, ax=ax_mag_1, shrink=0.82, pad=0.02)
        cb2.ax.tick_params(colors="#8B949E")

        ax_ph_1 = axes[1, 1]
        ax_ph_1.set_facecolor("#111122")
        im_ph_1 = ax_ph_1.imshow(
            ph_B, origin="lower", cmap="twilight", vmin=-np.pi, vmax=np.pi
        )
        ax_ph_1.set_title(
            f"Quiet Ch {ch_B} ({freq_B:.1f} MHz) — arg(V_ij) (Phase)",
            color="#40C4FF",
            fontsize=11,
            fontweight="bold",
        )
        ax_ph_1.set_xlabel("Antenna j", color="#C9D1D9", fontsize=9)
        ax_ph_1.set_ylabel("Antenna i", color="#C9D1D9", fontsize=9)
        ax_ph_1.tick_params(colors="#8B949E")
        cb3 = plt.colorbar(im_ph_1, ax=ax_ph_1, shrink=0.82, pad=0.02)
        cb3.ax.tick_params(colors="#8B949E")

        time_text = fig.text(0.5, 0.02, "", ha="center", fontsize=12, color="cyan")
        fig.tight_layout(rect=[0, 0.04, 1, 0.94])

        def update(frame_idx):
            vis = load_correlator_frame(
                sampled_files[frame_idx], num_elements, num_channels
            )
            im_mag_0.set_data(np.abs(vis[ch_A]))
            im_ph_0.set_data(np.angle(vis[ch_A]))
            im_mag_1.set_data(np.abs(vis[ch_B]))
            im_ph_1.set_data(np.angle(vis[ch_B]))

            t_s = frame_idx * (duration_s / max(1, total_video_frames))
            mins = int(t_s // 60)
            secs = int(t_s % 60)
            tot_mins = int(duration_s // 60)
            tot_secs = int(duration_s % 60)
            time_text.set_text(
                f"Frame {frame_idx + 1}/{total_video_frames}  |  "
                f"Window Time: {mins:02d}:{secs:02d} / {tot_mins:02d}:{tot_secs:02d} ({t_s:.1f}s / {duration_s:.0f}s)"
            )
            return [im_mag_0, im_ph_0, im_mag_1, im_ph_1, time_text]

        anim = animation.FuncAnimation(
            fig, update, frames=total_video_frames, blit=False
        )
        writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        anim.save(str(output_path), writer=writer)
        plt.close(fig)
        print(f"  Saved 2-channel correlator video: {output_path}")

        # Optional separate videos for each channel
        if separate_videos:
            for ch_idx, fr_mhz, lbl in [
                (ch_A, freq_A, "active"),
                (ch_B, freq_B, "quiet"),
            ]:
                sep_path = (
                    output_path.parent / f"{output_path.stem}_{lbl}_ch{ch_idx}.mp4"
                )
                generate_single_channel_correlator_video(
                    bin_files=bin_files,
                    ch=ch_idx,
                    freq_mhz=fr_mhz,
                    output_path=sep_path,
                    num_elements=num_elements,
                    num_channels=num_channels,
                    duration_s=duration_s,
                    fps=fps,
                    total_video_frames=total_video_frames,
                )

    else:
        # Single channel fallback
        ch = channels_to_plot[0]
        freq = freqs_mhz[0]
        generate_single_channel_correlator_video(
            bin_files=bin_files,
            ch=ch,
            freq_mhz=freq,
            output_path=output_path,
            num_elements=num_elements,
            num_channels=num_channels,
            duration_s=duration_s,
            fps=fps,
            total_video_frames=total_video_frames,
        )


def generate_single_channel_correlator_video(
    bin_files: List[Path],
    ch: int,
    freq_mhz: float,
    output_path: Path,
    num_elements: int,
    num_channels: int,
    duration_s: float = 120.0,
    fps: int = 10,
    total_video_frames: Optional[int] = None,
):
    """Renders a 1-channel |V_ij| + arg(V_ij) correlator matrix video."""
    total_available = len(bin_files)
    if total_video_frames is None:
        total_video_frames = int(duration_s * fps)

    if total_available > 0:
        indices = np.linspace(0, total_available - 1, total_video_frames).astype(int)
        sampled_files = [bin_files[i] for i in indices]
    else:
        sampled_files = []
        total_video_frames = 0

    first_vis = load_correlator_frame(sampled_files[0], num_elements, num_channels)
    first_mag = np.abs(first_vis[ch])
    vmin = 0.0
    vmax = float(np.percentile(first_mag, 99.5))

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor="#0a0a1a")
    fig.suptitle(
        f"CHARTS Correlator Visibility Matrix — Channel {ch} ({freq_mhz:.1f} MHz)",
        fontsize=14,
        color="white",
    )

    ax_mag = axes[0]
    ax_mag.set_facecolor("#111122")
    im_mag = ax_mag.imshow(
        first_mag, origin="lower", cmap="inferno", vmin=vmin, vmax=vmax
    )
    ax_mag.set_title("|V_ij| (Magnitude)", color="white", fontsize=12)
    ax_mag.set_xlabel("Antenna j", color="white")
    ax_mag.set_ylabel("Antenna i", color="white")
    ax_mag.tick_params(colors="white")
    plt.colorbar(im_mag, ax=ax_mag, shrink=0.8)

    ax_ph = axes[1]
    ax_ph.set_facecolor("#111122")
    first_phase = np.angle(first_vis[ch])
    im_ph = ax_ph.imshow(
        first_phase, origin="lower", cmap="twilight", vmin=-np.pi, vmax=np.pi
    )
    ax_ph.set_title("arg(V_ij) (Phase)", color="white", fontsize=12)
    ax_ph.set_xlabel("Antenna j", color="white")
    ax_ph.set_ylabel("Antenna i", color="white")
    ax_ph.tick_params(colors="white")
    plt.colorbar(im_ph, ax=ax_ph, shrink=0.8)

    time_text = fig.text(0.5, 0.02, "", ha="center", fontsize=12, color="cyan")
    fig.tight_layout(rect=[0, 0.05, 1, 0.93])

    def update(frame_idx):
        vis = load_correlator_frame(
            sampled_files[frame_idx], num_elements, num_channels
        )
        mag = np.abs(vis[ch])
        phase = np.angle(vis[ch])
        im_mag.set_data(mag)
        im_ph.set_data(phase)
        t_s = frame_idx * (duration_s / max(1, total_video_frames))
        mins = int(t_s // 60)
        secs = int(t_s % 60)
        tot_mins = int(duration_s // 60)
        tot_secs = int(duration_s % 60)
        time_text.set_text(
            f"Frame {frame_idx + 1}/{total_video_frames}  |  "
            f"Window Time: {mins:02d}:{secs:02d} / {tot_mins:02d}:{tot_secs:02d} ({t_s:.1f}s / {duration_s:.0f}s)"
        )
        return [im_mag, im_ph, time_text]

    anim = animation.FuncAnimation(fig, update, frames=total_video_frames, blit=False)
    writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(output_path), writer=writer)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Video 3: Beam Tracker Output Video (All Beams + Target Legend)
# ---------------------------------------------------------------------------
def generate_beam_tracker_video(
    tracker_dir: Path,
    tracker_name: str,
    output_path: Path,
    samples_per_data_set: int = 1536,
    num_freq: int = 672,
    max_beams: int = 8,
    duration_s: float = 60.0,
    fps: int = 10,
    max_frames: Optional[int] = None,
    beam_targets_str: Optional[str] = None,
    workers: int = 16,
):
    """
    Generates a video showing formed-beam power evolution:
      - Top panel: Formed beam power vs frequency across ALL active beams,
        with individual high-contrast colors and tracked object names in the legend.
      - Bottom panel: Integrated formed beam power bar chart per beam.
    Optimized with parallel precomputation across workers for rapid rendering.
    """
    bin_files = sorted(tracker_dir.glob(f"{tracker_name}_*.bin"))
    if not bin_files:
        print(
            f"[WARNING] No tracker files found matching {tracker_name}_*.bin in {tracker_dir}"
        )
        return

    total_available = len(bin_files)
    if max_frames:
        bin_files = bin_files[:max_frames]
        total_available = len(bin_files)

    # Ensure the video spans the full window duration (duration_s seconds at given fps)
    total_video_frames = int(duration_s * fps)
    if total_available > 0:
        indices = np.linspace(0, total_available - 1, total_video_frames).astype(int)
        sampled_files = [bin_files[i] for i in indices]
    else:
        sampled_files = []
        total_video_frames = 0

    print(
        f"  Tracker video: {total_video_frames} frames ({duration_s:.1f} s @ {fps} fps) "
        f"spanning the full window from {total_available} files, {max_beams} beams"
    )

    freq_mhz = (
        DEFAULT_FREQUENCY_START_MHZ + np.arange(num_freq) * CHARTS_CHANNEL_WIDTH_MHZ
    )

    # Parallel pre-extraction across workers
    print(
        f"  Pre-extracting {total_video_frames} beam tracker spectra in parallel using {workers} workers..."
    )
    tasks = [(f, samples_per_data_set, num_freq, max_beams) for f in sampled_files]
    if workers > 1:
        with mp.Pool(processes=min(workers, len(tasks) or 1)) as pool:
            results = pool.map(_worker_load_tracker, tasks)
    else:
        results = [_worker_load_tracker(t) for t in tasks]

    all_powers = [r[0] for r in results]
    all_integs = [r[1] for r in results]

    # 1. Load beam target metadata for object labels in legend
    target_metadata = {}
    targets_json = tracker_dir / f"{tracker_name}_targets.json"
    if targets_json.exists():
        try:
            import json

            with open(targets_json, "r") as f:
                tgt_list = json.load(f)
            for item in tgt_list:
                b_idx = item.get("beam", 0)
                target_metadata[b_idx] = item
        except Exception:
            pass

    if not target_metadata and beam_targets_str:
        raw_items = [s.strip() for s in beam_targets_str.split(";") if s.strip()]
        for idx, item in enumerate(raw_items):
            if ":" in item:
                nm, coords = item.split(":", 1)
                target_metadata[idx] = {"name": nm.strip(), "coords": coords.strip()}
            else:
                target_metadata[idx] = {"name": f"Target {idx}", "coords": item.strip()}

    # High-contrast vibrant colors for distinct beams
    vibrant_colors = [
        "#00FFCC",  # Cyan (Beam 0)
        "#FF5252",  # Red / Coral (Beam 1)
        "#FFD700",  # Gold / Yellow (Beam 2)
        "#FF4081",  # Hot Pink (Beam 3)
        "#7C4DFF",  # Purple / Indigo (Beam 4)
        "#00E676",  # Bright Green (Beam 5)
        "#FF9100",  # Orange (Beam 6)
        "#40C4FF",  # Light Blue (Beam 7)
    ]
    while len(vibrant_colors) < max_beams:
        vibrant_colors.append(plt.cm.tab10(len(vibrant_colors) % 10))

    # First frame for setup
    first_power = all_powers[0]  # (freq, beams) in dB
    first_integ = all_integs[0]  # (beams,) in dB

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(16, 10),
        facecolor="#0a0a1a",
        gridspec_kw={"height_ratios": [3, 2]},
    )
    fig.suptitle(
        "CHARTS Beam Tracker Output — Formed Beam Power Across Tracked Targets",
        fontsize=14,
        color="white",
    )

    # Top: Power vs Frequency per beam
    ax_spec = axes[0]
    ax_spec.set_facecolor("#111122")
    lines = []
    for b in range(max_beams):
        tgt_info = target_metadata.get(b, {})
        tgt_name = tgt_info.get("name", f"Beam {b}")
        if "ra_deg" in tgt_info and "dec_deg" in tgt_info:
            label = f"B{b}: {tgt_name} ({tgt_info['ra_deg']:.2f}°, {tgt_info['dec_deg']:+.2f}°)"
        elif "coords" in tgt_info:
            label = f"B{b}: {tgt_name} ({tgt_info['coords']})"
        else:
            label = f"B{b}: {tgt_name}"

        (line,) = ax_spec.plot(
            freq_mhz,
            first_power[:, b],
            color=vibrant_colors[b],
            linewidth=1.4,
            label=label,
            alpha=0.9,
        )
        lines.append(line)

    ax_spec.set_xlabel("Frequency (MHz)", color="#C9D1D9", fontsize=10)
    ax_spec.set_ylabel("Formed Power (dB)", color="#C9D1D9", fontsize=10)
    ax_spec.legend(
        loc="upper right",
        fontsize=8,
        ncol=2,
        facecolor="#1a1a2e",
        labelcolor="white",
        edgecolor="#30363D",
    )
    ax_spec.tick_params(colors="#8B949E")
    ax_spec.grid(True, alpha=0.25, color="gray", linestyle="--")
    for sp in ax_spec.spines.values():
        sp.set_color("#30363D")

    # Bottom: Integrated power bar chart
    ax_bar = axes[1]
    ax_bar.set_facecolor("#111122")
    bars = ax_bar.bar(
        range(max_beams),
        first_integ,
        color=vibrant_colors[:max_beams],
        edgecolor="white",
        linewidth=0.6,
    )
    ax_bar.set_xlabel("Formed Beam Target", color="#C9D1D9", fontsize=10)
    ax_bar.set_ylabel("Integrated Power (dB)", color="#C9D1D9", fontsize=10)
    ax_bar.set_xticks(range(max_beams))

    bar_labels = []
    for b in range(max_beams):
        tgt_name = target_metadata.get(b, {}).get("name", f"B{b}")
        short_name = tgt_name.split("(")[0].strip()
        bar_labels.append(f"B{b}\n{short_name}")

    ax_bar.set_xticklabels(bar_labels, color="white", fontsize=8)
    ax_bar.tick_params(colors="#8B949E")
    ax_bar.grid(True, alpha=0.25, color="gray", axis="y", linestyle="--")
    for sp in ax_bar.spines.values():
        sp.set_color("#30363D")

    time_text = fig.text(0.5, 0.015, "", ha="center", fontsize=12, color="cyan")
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])

    def update(frame_idx):
        cur_power = all_powers[frame_idx]
        cur_integ = all_integs[frame_idx]
        for b in range(max_beams):
            lines[b].set_ydata(cur_power[:, b])
            bars[b].set_height(cur_integ[b])
        t_s = frame_idx * (duration_s / max(1, total_video_frames))
        mins = int(t_s // 60)
        secs = int(t_s % 60)
        tot_mins = int(duration_s // 60)
        tot_secs = int(duration_s % 60)
        time_text.set_text(
            f"Frame {frame_idx + 1}/{total_video_frames}  |  "
            f"Window Time: {mins:02d}:{secs:02d} / {tot_mins:02d}:{tot_secs:02d} ({t_s:.1f}s / {duration_s:.0f}s)"
        )
        return lines + list(bars) + [time_text]

    anim = animation.FuncAnimation(fig, update, frames=total_video_frames, blit=False)
    writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(output_path), writer=writer)
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="CHARTS Pipeline Video Generator")
    subparsers = parser.add_subparsers(dest="command", help="Video type to generate")

    # --- baseband ---
    p_bb = subparsers.add_parser(
        "baseband", help="Baseband frequency power line spectrum video"
    )
    p_bb.add_argument("--window-dir", type=str, required=True)
    p_bb.add_argument("--base-name", type=str, required=True)
    p_bb.add_argument("--output", type=str, required=True)
    p_bb.add_argument("--antennas", type=int, default=64)
    p_bb.add_argument("--num-freq", type=int, default=672)
    p_bb.add_argument("--samples-per-frame", type=int, default=1536)
    p_bb.add_argument(
        "--duration-s",
        type=float,
        default=60.0,
        help="Video duration in seconds (default: 60.0 = 1 minute)",
    )
    p_bb.add_argument("--fps", type=int, default=10)
    p_bb.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Parallel workers for frame pre-extraction (default: 16)",
    )
    p_bb.add_argument("--max-frames", type=int, default=None)

    # --- correlator ---
    p_corr = subparsers.add_parser(
        "correlator", help="Correlator matrix video (dual-channel active vs quiet)"
    )
    p_corr.add_argument("--corr-dir", type=str, required=True)
    p_corr.add_argument("--corr-name", type=str, required=True)
    p_corr.add_argument("--output", type=str, required=True)
    p_corr.add_argument("--num-elements", type=int, default=64)
    p_corr.add_argument("--num-channels", type=int, default=672)
    p_corr.add_argument(
        "--duration-s",
        type=float,
        default=60.0,
        help="Video duration in seconds (default: 60.0 = 1 minute)",
    )
    p_corr.add_argument("--fps", type=int, default=10)
    p_corr.add_argument(
        "--freq-channel", type=int, default=None, help="Single frequency channel index"
    )
    p_corr.add_argument(
        "--freq-channels",
        type=str,
        default=None,
        help="Comma-separated frequency channels or 'auto'",
    )
    p_corr.add_argument(
        "--separate-videos",
        action="store_true",
        help="Also generate separate MP4s for each channel",
    )
    p_corr.add_argument("--max-frames", type=int, default=None)

    # --- tracker ---
    p_trk = subparsers.add_parser("tracker", help="Beam tracker output video")
    p_trk.add_argument("--tracker-dir", type=str, required=True)
    p_trk.add_argument("--tracker-name", type=str, required=True)
    p_trk.add_argument("--output", type=str, required=True)
    p_trk.add_argument("--samples-per-data-set", type=int, default=1536)
    p_trk.add_argument("--num-freq", type=int, default=672)
    p_trk.add_argument("--max-beams", type=int, default=8)
    p_trk.add_argument(
        "--duration-s",
        type=float,
        default=60.0,
        help="Video duration in seconds (default: 60.0 = 1 minute)",
    )
    p_trk.add_argument("--fps", type=int, default=10)
    p_trk.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Parallel workers for frame pre-extraction (default: 16)",
    )
    p_trk.add_argument(
        "--beam-targets",
        type=str,
        default=None,
        help="Semicolon-separated target names/coordinates",
    )
    p_trk.add_argument("--max-frames", type=int, default=None)

    args = parser.parse_args()

    if args.command == "baseband":
        generate_baseband_spectrogram_video(
            window_dir=Path(args.window_dir),
            base_name=args.base_name,
            output_path=Path(args.output),
            num_antennas=args.antennas,
            num_freq=args.num_freq,
            samples_per_frame=args.samples_per_frame,
            duration_s=args.duration_s,
            fps=args.fps,
            workers=args.workers,
            max_frames=args.max_frames,
        )
    elif args.command == "correlator":
        f_channels = None
        if args.freq_channels:
            if args.freq_channels.strip().lower() == "auto":
                f_channels = None
            else:
                f_channels = [
                    int(x.strip())
                    for x in args.freq_channels.split(",")
                    if x.strip().isdigit()
                ]

        generate_correlator_video(
            corr_dir=Path(args.corr_dir),
            corr_name=args.corr_name,
            output_path=Path(args.output),
            num_elements=args.num_elements,
            num_channels=args.num_channels,
            duration_s=args.duration_s,
            fps=args.fps,
            freq_channel_idx=args.freq_channel,
            freq_channels=f_channels,
            max_frames=args.max_frames,
            separate_videos=args.separate_videos,
        )
    elif args.command == "tracker":
        generate_beam_tracker_video(
            tracker_dir=Path(args.tracker_dir),
            tracker_name=args.tracker_name,
            output_path=Path(args.output),
            samples_per_data_set=args.samples_per_data_set,
            num_freq=args.num_freq,
            max_beams=args.max_beams,
            duration_s=args.duration_s,
            fps=args.fps,
            workers=args.workers,
            beam_targets_str=args.beam_targets,
            max_frames=args.max_frames,
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
