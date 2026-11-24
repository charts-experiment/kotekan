#!/usr/bin/env python3
"""Generate schedule manifest and all 14 YAML configs for the uncapped
CHARTS 4-antenna commissioning & chromaticity campaign (commissioning_20261119_v2).

Supports --rebuild-beam-configs after Phase 1 produces fresh localized_rfi_targets.json
and gains_corr_1119_0_4_32_36_fringehp_672x64.npy.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import astropy.units as u
import numpy as np
import yaml
from astropy.coordinates import AltAz, EarthLocation, SkyCoord, get_sun
from astropy.time import Time

from charts_constants import CHARTS_ALTITUDE_M, CHARTS_LATITUDE_DEG, CHARTS_LONGITUDE_DEG

_CHARTS_DIR = str(Path(__file__).resolve().parents[1])
if _CHARTS_DIR not in sys.path:
    sys.path.insert(0, _CHARTS_DIR)

from rfi_mitigation_tools import (  # noqa: E402
    FREQS_MHZ,
    PUDAHUEL_KNOWN_INTERFERERS,
    find_optimal_off_beams,
)

V2_DATA_ROOT = Path("/data/commissioning_20261119_v2")
V2_SCRIPT_DIR = Path("/home/fpalma/kotekan/charts/commissioning_20261119_v2")
CONFIG_OUT_DIR = V2_SCRIPT_DIR / "configs"
CONTINUATION_CONFIG_DIR = V2_SCRIPT_DIR / "configs_20261121"
CONTINUATION_MANIFEST_PATH = V2_DATA_ROOT / "schedule_manifest_20261121.json"
FULL_20261121_CONFIG_DIR = V2_SCRIPT_DIR / "configs_20261121_full"
FULL_20261121_MANIFEST_PATH = V2_DATA_ROOT / "schedule_manifest_20261121_full.json"
GAINS_FILE = "/data/calibration/corr_1119/gains_corr_1119_0_4_32_36_fringehp_672x64.npy"
LOCALIZED_RFI_PATH = V2_DATA_ROOT / "localized_rfi_targets.json"
FALLBACK_RFI_PATH = Path("/data/commissioning_20261119/localized_rfi_targets.json")
MANIFEST_PATH = V2_DATA_ROOT / "schedule_manifest.json"
TLE_V2_PATH = V2_DATA_ROOT / "tle_cache.tle"
TLE_V1_PATH = Path("/data/commissioning_20261119/tle_cache.tle")

SITE_LAT_DEG = float(CHARTS_LATITUDE_DEG)
SITE_LON_DEG = float(CHARTS_LONGITUDE_DEG)
SITE_ALT_M = float(CHARTS_ALTITUDE_M)

ACTIVE_ANTENNAS = [0, 4, 32, 36]
FITTED_POSITIONS = {
    0: [0.0, 0.0, 0.0],
    4: [-4.24775, 0.0275, 0.0],
    32: [0.15, 4.30625, 0.0],
    36: [-4.12725, 4.42725, 0.0],
}


def build_antenna_positions() -> List[List[float]]:
    pos = [[0.0, 0.0, 0.0] for _ in range(64)]
    for ant, xyz in FITTED_POSITIONS.items():
        pos[ant] = list(xyz)
    return pos


def altaz_to_lm(alt_deg: float, az_deg: float) -> tuple[float, float]:
    el_r = np.deg2rad(alt_deg)
    az_r = np.deg2rad(az_deg)
    return float(np.cos(el_r) * np.sin(az_r)), float(np.cos(el_r) * np.cos(az_r))


def build_corr_config(
    *,
    base_dir: str,
    exit_after_n_files: int,
    integration_frames: int = 20,
    samples_per_data_set: int = 153600,
    buffer_depth: int = 20,
) -> Dict[str, Any]:
    n_local_freq = 336
    bytes_per_dataset = 168 * 2 * samples_per_data_set
    num_blocks = 528
    block_size = 2
    sizeof_int = 4
    corr_frame_size = n_local_freq * num_blocks * (block_size * block_size) * 2 * sizeof_int
    cfg: Dict[str, Any] = {
        "type": "config",
        "log_level": "info",
        "cpu_affinity": [6, 7, 8, 9],
        "num_elements": 64,
        "n_channels_per_packet": 168,
        "local_subbands_per_packet": 2,
        "num_local_freq": n_local_freq,
        "buffer_depth": buffer_depth,
        "samples_per_data_set": samples_per_data_set,
        "sample_period_s": 3.333333333333e-6,
        "frame_arrival_period": samples_per_data_set * 3.333333333333e-6,
        "bytes_per_dataset": bytes_per_dataset,
        "num_data_sets": 1,
        "block_size": block_size,
        "num_blocks": num_blocks,
        "sizeof_int": sizeof_int,
        "data_format": "cuda_wmma",
        "integration_frames": integration_frames,
        "capture_n_frames": 0,
        "main_pool": {
            "kotekan_metadata_pool": "chartsMetadata",
            "num_metadata_objects": buffer_depth * 20,
        },
        "dpdk": {
            "kotekan_stage": "dpdkCore",
            "lcore_start_delay": 1,
            "lcore_cpu_map": [3, 5],
            "main_lcore_cpu": 4,
            "lcore_port_map": [[0], [1]],
            "pcie_block_list": ["0000:70:00.0"],
            "max_rx_pkt_len": 9000,
            "rx_ring_size": 4096,
            "num_mbufs": 65536,
            "mbuf_data_size": 9216,
            "handlers": [
                {
                    "dpdk_handler": "rfsocHandlerShuffle",
                    "out_buffer": "network_capture_buf_0",
                    "alignment": 15360,
                    "mask_buf": "mask_buffer_zeros_0",
                    "first_subband": 0,
                    "coarse_freq_start": 0,
                    "num_local_freq": n_local_freq,
                },
                {
                    "dpdk_handler": "rfsocHandlerShuffle",
                    "out_buffer": "network_capture_buf_1",
                    "alignment": 15360,
                    "mask_buf": "mask_buffer_zeros_1",
                    "first_subband": 2,
                    "coarse_freq_start": 336,
                    "num_local_freq": n_local_freq,
                },
            ],
        },
        "run_metadata": {
            "kotekan_stage": "runMetadataWriter",
            "base_dir": base_dir,
            "num_antennas": 64,
            "signal_antennas": list(ACTIVE_ANTENNAS),
            "position_outliers": [4, 32, 36],
        },
    }

    for sb in (0, 1):
        cfg[f"network_capture_buf_{sb}"] = {
            "kotekan_buffer": "standard",
            "num_frames": buffer_depth,
            "frame_size": bytes_per_dataset * 64,
            "numa_node": 0,
            "metadata_pool": "main_pool",
            "zero_new_frames": True,
            "mlock_frames": False,
        }
        cfg[f"mask_buffer_zeros_{sb}"] = {
            "kotekan_buffer": "standard",
            "num_frames": buffer_depth,
            "frame_size": samples_per_data_set,
            "numa_node": 0,
            "metadata_pool": "main_pool",
        }
        cfg[f"host_correlation_buffer_{sb}"] = {
            "kotekan_buffer": "standard",
            "num_frames": buffer_depth,
            "frame_size": corr_frame_size,
            "metadata_pool": "main_pool",
        }
        cfg[f"integrated_correlation_buffer_{sb}"] = {
            "kotekan_buffer": "standard",
            "num_frames": buffer_depth,
            "frame_size": corr_frame_size,
            "metadata_pool": "main_pool",
        }

    cfg["zero_samples"] = {
        "kotekan_stage": "zeroSamples",
        "out_buf": "network_capture_buf_0",
        "lost_samples_buf": "mask_buffer_zeros_0",
        "zero_value": 0,
        "sample_size": 21504,
    }
    cfg["zero_samples_1"] = {
        "kotekan_stage": "zeroSamples",
        "out_buf": "network_capture_buf_1",
        "lost_samples_buf": "mask_buffer_zeros_1",
        "zero_value": 0,
        "sample_size": 21504,
    }

    gpu_block: Dict[str, Any] = {"profiling": True, "kernel_path": "lib/cuda/kernels"}
    for sb in (0, 1):
        cmds = [
            {"name": "cudaInputData", "in_buf": "host_voltage", "gpu_mem": f"voltage_{sb}"},
            {"name": "cudaSyncInput"},
            {"name": "cudaShuffleAstron", "gpu_mem_voltage": f"voltage_{sb}", "gpu_mem_ordered_voltage": f"ordered_voltage_{sb}"},
            {"name": "cudaCorrelatorAstron", "gpu_mem_voltage": f"ordered_voltage_{sb}", "gpu_mem_correlation_matrix": f"correlation_matrix_{sb}"},
            {"name": "cudaSyncOutput"},
            {"name": "cudaOutputData", "in_buf": "host_voltage", "gpu_mem": f"correlation_matrix_{sb}", "out_buf": "host_correlation"},
        ]
        gpu_block[f"commands_{sb}"] = cmds
        gpu_block[f"gpu_{sb}"] = {
            "kotekan_stage": "cudaProcess",
            "gpu_id": 0,
            "cuda_stream_base": sb,
            "num_cuda_streams": 6,
            "buffer_depth": 3,
            "commands": cmds,
            "in_buffers": {"host_voltage": f"network_capture_buf_{sb}"},
            "out_buffers": {"host_correlation": f"host_correlation_buffer_{sb}"},
        }
    cfg["gpu"] = gpu_block

    for sb in (0, 1):
        cfg[f"charts_accumulate_{sb}"] = {
            "kotekan_stage": "chartsAccumulate",
            "in_buf": f"host_correlation_buffer_{sb}",
            "out_buf": f"integrated_correlation_buffer_{sb}",
            "num_frames_to_accumulate": integration_frames,
        }
        cfg[f"inspect_frame_{sb}"] = {
            "kotekan_stage": "restInspectFrame",
            "in_buf": f"host_correlation_buffer_{sb}",
        }
        cfg[f"raw_file_write_{sb}"] = {
            "kotekan_stage": "rawFileWrite",
            "in_buf": f"integrated_correlation_buffer_{sb}",
            "base_dir": base_dir,
            "create_base_dir": True,
            "file_name": f"corr_freq_{sb}",
            "file_ext": "bin",
            "prefix_hostname": False,
            "num_frames_per_file": 1,
            "exit_after_n_files": exit_after_n_files,
        }
    return cfg


def build_rfi_cmd(
    *,
    sb_idx: int,
    samples_per_data_set: int,
    enabled: bool,
    blank_voltages: bool,
) -> Dict[str, Any]:
    freq_start_hz = 300000000.0 if sb_idx == 0 else 400800000.0
    return {
        "name": "cudaRfiMitigationCommand",
        "gpu_mem_voltage": f"voltage_{sb_idx}",
        "gpu_mem_rfi_mask": f"rfi_mask_{sb_idx}",
        "num_elements": 64,
        "num_local_freq": 336,
        "samples_per_data_set": samples_per_data_set,
        "buffer_depth": 3 if samples_per_data_set >= 15360 else 4,
        "antenna_order": "descending",
        "active_antennas": list(ACTIVE_ANTENNAS),
        "freq_start_hz": freq_start_hz,
        "freq_step_hz": 300000.0,
        "enabled": enabled,
        "blank_voltages_enabled": blank_voltages,
        "time_chunk_size": 256,
        "sample_stride": 1,
        "enable_sk_check": True,
        "sk_sigma_threshold": 4.5,
        "enable_time_burst_check": True,
        "burst_sigma_threshold": 5.0,
        "enable_freq_mad_check": True,
        "freq_window_half_width": 8,
        "mad_sigma_threshold": 6.0,
        "enable_saturation_check": True,
        "saturation_fraction_threshold": 0.02,
        "enable_morphology_sir": True,
        "enable_freq_shoulder_check": True,
        "freq_shoulder_half_width": 2,
        "shoulder_sigma_threshold": 2.5,
        "edge_channels_to_ignore": 0,
        "enable_persistent_ema": True,
        "ema_alpha": 0.02,
    }


def build_tracker_cmd(
    *,
    sb_idx: int,
    samples_per_data_set: int,
    gpu_buffer_depth: int,
    max_beams: int,
    conjugate_gains: bool,
    uncalibrated_beams: List[int],
    beam_targets: List[Dict[str, float]],
    use_rfi_mask: bool,
) -> Dict[str, Any]:
    freq_start_hz = 300000000.0 if sb_idx == 0 else 400800000.0
    cmd: Dict[str, Any] = {
        "name": "cudaDirectBeamTrackerCommand",
        "gpu_mem_voltage": f"voltage_{sb_idx}",
        "gpu_mem_formed_beams": f"formed_beams_{sb_idx}",
        "gpu_mem_beam_power": f"beam_power_{sb_idx}",
        "num_elements": 64,
        "num_local_freq": 336,
        "samples_per_data_set": samples_per_data_set,
        "spacing_m": 0.6,
        "antenna_order": "descending",
        "active_antennas": list(ACTIVE_ANTENNAS),
        "custom_antenna_positions": build_antenna_positions(),
        "max_beams": max_beams,
        "initial_active_beams": max_beams,
        "buffer_depth": gpu_buffer_depth,
        "time_chunk_size": 256,
        "time_unroll": 4,
        "taper_type": "none",
        "normalize_weights": True,
        "beam_tile_size": 4,
        "enable_cuda_graph": False,
        "gains_file": GAINS_FILE,
        "conjugate_gains": conjugate_gains,
        "enable_grid_mode": False,
        "grid_step": 0.02,
        "site_lat_deg": SITE_LAT_DEG,
        "site_lon_deg": SITE_LON_DEG,
        "site_alt_m": SITE_ALT_M,
        "freq_start_hz": freq_start_hz,
        "freq_step_hz": 300000.0,
    }
    if use_rfi_mask:
        cmd["gpu_mem_rfi_mask"] = f"rfi_mask_{sb_idx}"
    if uncalibrated_beams:
        cmd["uncalibrated_beams"] = list(uncalibrated_beams)

    for b_idx, tgt in enumerate(beam_targets):
        sfx = "" if b_idx == 0 else f"_{b_idx}"
        if "ra_deg" in tgt and "dec_deg" in tgt:
            cmd[f"source_ra_deg{sfx}"] = float(tgt["ra_deg"])
            cmd[f"source_dec_deg{sfx}"] = float(tgt["dec_deg"])
        else:
            cmd[f"source_l0{sfx}"] = float(tgt.get("l0", 0.0))
            cmd[f"source_m0{sfx}"] = float(tgt.get("m0", 0.0))
    return cmd


def build_beam_config(
    *,
    run_id: str,
    samples_per_data_set: int,
    buffer_depth: int,
    gpu_buffer_depth: int,
    max_beams: int,
    conjugate_gains: bool,
    uncalibrated_beams: List[int],
    beam_targets: List[Dict[str, float]],
    rfi_enabled: bool,
    rfi_blank_voltages: bool,
    accumulate_frames: int,
    num_frames_per_file: int,
    exit_after_n_files: int,
    skip_frames: int,
    gpu_profiling: bool = True,
    copy_formed_beams_to_host: bool = False,
) -> Dict[str, Any]:
    out_dir = str(V2_DATA_ROOT / run_id)
    n_local_freq = 336
    bytes_per_dataset = 168 * 2 * samples_per_data_set
    cfg: Dict[str, Any] = {
        "type": "config",
        "log_level": "info",
        "cpu_affinity": [6, 7, 8, 9],
        "num_elements": 64,
        "n_channels_per_packet": 168,
        "local_subbands_per_packet": 2,
        "num_local_freq": n_local_freq,
        "sample_period_s": 3.333333333333e-6,
        "samples_per_data_set": samples_per_data_set,
        "frame_arrival_period": samples_per_data_set * 3.333333333333e-6,
        "bytes_per_dataset": bytes_per_dataset,
        "capture_n_frames": 0,
        "buffer_depth": buffer_depth,
        "gpu_buffer_depth": gpu_buffer_depth,
        "spacing_m": 0.6,
        "max_beams": max_beams,
        "initial_active_beams": max_beams,
        "sizeof_complex_float": 8,
        "sizeof_float": 4,
        "main_pool": {
            "kotekan_metadata_pool": "chartsMetadata",
            "num_metadata_objects": buffer_depth * 20,
        },
        "dpdk": {
            "kotekan_stage": "dpdkCore",
            "lcore_start_delay": 1,
            "lcore_cpu_map": [3, 5],
            "main_lcore_cpu": 4,
            "lcore_port_map": [[0], [1]],
            "pcie_block_list": ["0000:70:00.0"],
            "max_rx_pkt_len": 9000,
            "rx_ring_size": 4096,
            "num_mbufs": 65536,
            "mbuf_data_size": 9216,
            "handlers": [
                {
                    "dpdk_handler": "rfsocHandlerShuffle",
                    "out_buffer": "network_capture_buf_0",
                    "alignment": 15360,
                    "mask_buf": "mask_buffer_zeros_0",
                    "first_subband": 0,
                    "coarse_freq_start": 0,
                    "num_local_freq": n_local_freq,
                },
                {
                    "dpdk_handler": "rfsocHandlerShuffle",
                    "out_buffer": "network_capture_buf_1",
                    "alignment": 15360,
                    "mask_buf": "mask_buffer_zeros_1",
                    "first_subband": 2,
                    "coarse_freq_start": 336,
                    "num_local_freq": n_local_freq,
                },
            ],
        },
        "run_metadata": {
            "kotekan_stage": "runMetadataWriter",
            "base_dir": out_dir,
            "num_antennas": 64,
            "signal_antennas": list(ACTIVE_ANTENNAS),
        },
    }

    for sb in (0, 1):
        cfg[f"network_capture_buf_{sb}"] = {
            "kotekan_buffer": "standard",
            "num_frames": buffer_depth,
            "frame_size": bytes_per_dataset * 64,
            "numa_node": 0,
            "metadata_pool": "main_pool",
            "zero_new_frames": True,
            "mlock_frames": False,
        }
        cfg[f"mask_buffer_zeros_{sb}"] = {
            "kotekan_buffer": "standard",
            "num_frames": buffer_depth,
            "frame_size": samples_per_data_set,
            "numa_node": 0,
            "metadata_pool": "main_pool",
        }
        if copy_formed_beams_to_host:
            cfg[f"host_formed_beams_buffer_{sb}"] = {
                "kotekan_buffer": "standard",
                "num_frames": buffer_depth,
                "frame_size": samples_per_data_set * n_local_freq * max_beams * 8,
                "numa_node": 0,
                "metadata_pool": "main_pool",
                "zero_new_frames": True,
                "mlock_frames": False,
            }
        cfg[f"host_beam_power_buffer_{sb}"] = {
            "kotekan_buffer": "standard",
            "num_frames": buffer_depth,
            "frame_size": max_beams * n_local_freq * 4,
            "numa_node": 0,
            "metadata_pool": "main_pool",
            "zero_new_frames": True,
            "mlock_frames": False,
        }
        if accumulate_frames > 1:
            cfg[f"integrated_beam_power_buffer_{sb}"] = {
                "kotekan_buffer": "standard",
                "num_frames": buffer_depth,
                "frame_size": max_beams * n_local_freq * 4,
                "numa_node": 0,
                "metadata_pool": "main_pool",
                "zero_new_frames": True,
                "mlock_frames": False,
            }
        cfg[f"zero_samples_{sb}"] = {
            "kotekan_stage": "zeroSamples",
            "out_buf": f"network_capture_buf_{sb}",
            "lost_samples_buf": f"mask_buffer_zeros_{sb}",
            "zero_value": 0,
            "sample_size": 64 * n_local_freq,
        }

    gpu_block: Dict[str, Any] = {"profiling": gpu_profiling}
    for sb in (0, 1):
        cmds: List[Dict[str, Any]] = [
            {"name": "cudaInputData", "in_buf": "host_voltage", "gpu_mem": f"voltage_{sb}"},
            {"name": "cudaSyncInput"},
            {
                "name": "cudaAntennaMaskCommand",
                "gpu_mem_voltage": f"voltage_{sb}",
                "num_elements": 64,
                "num_local_freq": n_local_freq,
                "samples_per_data_set": samples_per_data_set,
                "buffer_depth": gpu_buffer_depth,
                "sample_stride": 4,
                "antenna_order": "descending",
                "active_antennas": list(ACTIVE_ANTENNAS),
                "auto_detect_enabled": True,
                "blank_voltages_enabled": False,
                "dead_power_threshold": 1.0,
                "sat_power_threshold": 80.0,
                "clip_fraction_threshold": 0.15,
                "revival_frames": 5,
            },
        ]
        if rfi_enabled:
            cmds.append(
                build_rfi_cmd(
                    sb_idx=sb,
                    samples_per_data_set=samples_per_data_set,
                    enabled=True,
                    blank_voltages=rfi_blank_voltages,
                )
            )
        cmds.append(
            build_tracker_cmd(
                sb_idx=sb,
                samples_per_data_set=samples_per_data_set,
                gpu_buffer_depth=gpu_buffer_depth,
                max_beams=max_beams,
                conjugate_gains=conjugate_gains,
                uncalibrated_beams=uncalibrated_beams,
                beam_targets=beam_targets,
                use_rfi_mask=rfi_enabled and rfi_blank_voltages,
            )
        )
        cmds.append({"name": "cudaSyncOutput"})
        if copy_formed_beams_to_host:
            cmds.append(
                {
                    "name": "cudaOutputData",
                    "in_buf": "host_voltage",
                    "gpu_mem": f"formed_beams_{sb}",
                    "out_buf": "host_formed_beams",
                }
            )
        cmds.append(
            {
                "name": "cudaOutputData",
                "in_buf": "host_voltage",
                "gpu_mem": f"beam_power_{sb}",
                "out_buf": "host_beam_power",
            }
        )
        gpu_out_buffers = {"host_beam_power": f"host_beam_power_buffer_{sb}"}
        if copy_formed_beams_to_host:
            gpu_out_buffers = {
                "host_formed_beams": f"host_formed_beams_buffer_{sb}",
                **gpu_out_buffers,
            }
        gpu_block[f"commands_{sb}"] = cmds
        gpu_block[f"gpu_{sb}"] = {
            "kotekan_stage": "cudaProcess",
            "gpu_id": 0,
            "cuda_stream_base": sb,
            "num_cuda_streams": 6,
            "buffer_depth": gpu_buffer_depth,
            "commands": cmds,
            "in_buffers": {"host_voltage": f"network_capture_buf_{sb}"},
            "out_buffers": gpu_out_buffers,
        }
    cfg["gpu"] = gpu_block

    for sb in (0, 1):
        if copy_formed_beams_to_host:
            cfg[f"inspect_beams_{sb}"] = {
                "kotekan_stage": "restInspectFrame",
                "in_buf": f"host_formed_beams_buffer_{sb}",
                "len": 4194304,
            }
        power_buf = f"host_beam_power_buffer_{sb}"
        if accumulate_frames > 1:
            cfg[f"accumulate_power_{sb}"] = {
                "kotekan_stage": "chartsAccumulate",
                "in_buf": f"host_beam_power_buffer_{sb}",
                "out_buf": f"integrated_beam_power_buffer_{sb}",
                "data_type": "float32",
                "average": True,
                "num_frames_to_accumulate": accumulate_frames,
            }
            power_buf = f"integrated_beam_power_buffer_{sb}"
        cfg[f"raw_file_write_power_{sb}"] = {
            "kotekan_stage": "rawFileWrite",
            "in_buf": power_buf,
            "base_dir": out_dir,
            "create_base_dir": True,
            "file_name": f"beam_power_freq{sb}",
            "file_ext": "bin",
            "prefix_hostname": False,
            "skip_frames": skip_frames,
            "num_frames_per_file": num_frames_per_file,
            "exit_after_n_files": exit_after_n_files,
        }
    return cfg


def load_rfi_targets() -> Dict[str, Any]:
    if LOCALIZED_RFI_PATH.exists():
        return json.loads(LOCALIZED_RFI_PATH.read_text(encoding="utf-8"))
    if FALLBACK_RFI_PATH.exists():
        return json.loads(FALLBACK_RFI_PATH.read_text(encoding="utf-8"))
    return {}


def compute_gal_b_icrs(b_deg: float) -> Dict[str, float]:
    sc = SkyCoord(l=0.0 * u.deg, b=float(b_deg) * u.deg, frame="galactic").icrs
    return {"ra_deg": round(float(sc.ra.deg), 5), "dec_deg": round(float(sc.dec.deg), 5)}


def compute_solar_transit_m0(date_str: str) -> float:
    """Compute Sun meridional m0 at local transit (~16:30 UTC) from Carén site."""
    loc = EarthLocation(lat=SITE_LAT_DEG * u.deg, lon=SITE_LON_DEG * u.deg, height=SITE_ALT_M * u.m)
    t_ap = Time(f"{date_str}T16:30:00Z")
    aa = get_sun(t_ap).transform_to(AltAz(obstime=t_ap, location=loc))
    _, m0 = altaz_to_lm(float(aa.alt.deg), float(aa.az.deg))
    return round(float(m0), 4)


def patch_tracker_solar_m0(cfg: Dict[str, Any], m0_val: float) -> None:
    """Update meridional m0 targets inside cudaDirectBeamTrackerCommand entries."""
    for sb in ("0", "1"):
        for cmd in cfg.get("gpu", {}).get(f"commands_{sb}", []):
            if cmd.get("name") == "cudaDirectBeamTrackerCommand":
                for k in (
                    "source_m0",
                    "source_m0_1",
                    "source_m0_2",
                    "source_m0_3",
                    "source_m0_4",
                    "source_m0_5",
                    "source_m0_8",
                ):
                    cmd[k] = m0_val
                cmd["source_m0_6"] = round(m0_val + 0.070, 4)
                cmd["source_m0_7"] = round(m0_val - 0.122, 4)


def select_grating_safe_off_lm(
    target_l: float,
    target_m: float,
    positions_m: np.ndarray | None = None,
    interferers: Tuple[Tuple[float, float], ...] | None = None,
    min_sep_deg: float = 18.0,
) -> Tuple[float, float]:
    """Select an OFF (l, m) coordinate minimizing grating-lobe leakage across 300-501 MHz."""
    pos = np.asarray(build_antenna_positions(), dtype=np.float64)[ACTIVE_ANTENNAS] if positions_m is None else np.asarray(positions_m, dtype=np.float64)
    inter = (
        tuple(PUDAHUEL_KNOWN_INTERFERERS.values())
        if interferers is None
        else tuple(interferers)
    )
    cands = find_optimal_off_beams(
        pos,
        FREQS_MHZ[::4],
        target_l=float(target_l),
        target_m=float(target_m),
        interferers=inter,
        min_sep_deg=min_sep_deg,
        max_zenith_deg=65.0,
        grid_n=49,
        top_k=1,
    )
    return round(cands[0].l0, 4), round(cands[0].m0, 4)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rebuild-beam-configs", action="store_true", help="Rebuild YAML configs only, preserving manifest timing")
    p.add_argument(
        "--optimize-off-beams",
        action="store_true",
        help="Use grating-lobe-safe OFF-beam coordinates via find_optimal_off_beams()",
    )
    p.add_argument(
        "--generate-continuation-20261121",
        action="store_true",
        help="Write the 9 unfinished tests (test05-test11) into configs_20261121/ and schedule_manifest_20261121.json",
    )
    p.add_argument(
        "--generate-run-20261121-full",
        action="store_true",
        help="Write the 11-run full suite (test11/12a/12b/10 tonight + test05-09b tomorrow) into configs_20261121_full/",
    )
    args = p.parse_args()

    V2_DATA_ROOT.mkdir(parents=True, exist_ok=True)
    if args.generate_run_20261121_full:
        cfg_dir = FULL_20261121_CONFIG_DIR
    elif args.generate_continuation_20261121:
        cfg_dir = CONTINUATION_CONFIG_DIR
    else:
        cfg_dir = CONFIG_OUT_DIR
    cfg_dir.mkdir(parents=True, exist_ok=True)
    if not TLE_V2_PATH.exists() and TLE_V1_PATH.exists():
        shutil.copy2(TLE_V1_PATH, TLE_V2_PATH)

    rfi_locs = load_rfi_targets()
    rfi_425 = rfi_locs.get("RFI_425.4", {}).get("peak_el_ge_25", {"l": 0.008, "m": -0.9031})
    rfi_472 = rfi_locs.get("RFI_472.5", {}).get("peak_el_ge_25", {"l": -0.5035, "m": -0.1838})

    solar_chr_off_l, solar_chr_off_m = (
        select_grating_safe_off_lm(0.0005, 0.2356) if args.optimize_off_beams else (0.0, -0.70)
    )
    cena_off_l, cena_off_m = (
        select_grating_safe_off_lm(0.0, -0.1691) if args.optimize_off_beams else (-0.55, 0.45)
    )

    configs: Dict[str, Dict[str, Any]] = {
        "00_phase1_corr_cal.yaml": build_corr_config(
            base_dir="/data/corr/corr_1119_cal",
            exit_after_n_files=480,  # 480 * 10.24s = 81.92 min (~5.2 GB)
        ),
        "01_test01_muos_rfi_425_475.yaml": build_beam_config(
            run_id="test01_muos_rfi_425_475",
            samples_per_data_set=15360,
            buffer_depth=20,
            gpu_buffer_depth=3,
            max_beams=8,
            conjugate_gains=True,
            uncalibrated_beams=[],
            beam_targets=[
                {"l0": -0.54, "m0": 0.54},                                  # 0: MUOS-1 (updated live)
                {"l0": -0.63, "m0": 0.53},                                  # 1: MUOS-5 (updated live)
                {"l0": 0.86, "m0": 0.41},                                   # 2: MUOS-3 (updated live)
                {"l0": 0.45, "m0": -0.45},                                  # 3: OFF-MUOS control
                {"l0": -0.086, "m0": 0.625},                                # 4: LEO / GOES-19 (updated live)
                {"l0": float(rfi_425["l"]), "m0": float(rfi_425["m"])},     # 5: Loc 425 MHz
                {"l0": float(rfi_472["l"]), "m0": float(rfi_472["m"])},     # 6: Loc 472.5 MHz
                {"l0": 0.0, "m0": 0.0},                                     # 7: Zenith
            ],
            rfi_enabled=True,
            rfi_blank_voltages=False,  # Monitor-only so 380/425/475 MHz signals remain unblanked
            accumulate_frames=20,
            num_frames_per_file=585,
            exit_after_n_files=6,  # 59.9 min
            skip_frames=1,
        ),
        "02_test02_fornax_a_night_chromaticity.yaml": build_beam_config(
            run_id="test02_fornax_a_night_chromaticity",
            samples_per_data_set=15360,
            buffer_depth=20,
            gpu_buffer_depth=3,
            max_beams=10,
            conjugate_gains=True,
            uncalibrated_beams=[6],
            beam_targets=[
                {"ra_deg": 50.6737, "dec_deg": -37.2083},                   # 0: Fornax A Tracked Cal
                {"l0": 0.0, "m0": -0.0644},                                 # 1: Parked meridian drift at transit
                {"l0": 0.052, "m0": -0.0644},                               # 2: Fornax A +3 deg E (updated live)
                {"l0": -0.052, "m0": -0.0644},                              # 3: Fornax A -3 deg W (updated live)
                {"l0": 0.105, "m0": -0.0644},                               # 4: Fornax A +6 deg E (updated live)
                {"l0": -0.105, "m0": -0.0644},                              # 5: Fornax A -6 deg W (updated live)
                {"ra_deg": 50.6737, "dec_deg": -37.2083},                   # 6: Fornax A Uncalibrated
                {"ra_deg": 79.9571, "dec_deg": -45.7789},                   # 7: Pictor A
                {"l0": 0.0, "m0": 0.0},                                     # 8: Zenith
                {"l0": -0.50, "m0": 0.50},                                  # 9: Deep OFF (>45 deg away)
            ],
            rfi_enabled=True,
            rfi_blank_voltages=True,
            accumulate_frames=20,
            num_frames_per_file=585,
            exit_after_n_files=10,  # 99.8 min
            skip_frames=1,
        ),
        "03_test03_pictor_orion_lmc_crab.yaml": build_beam_config(
            run_id="test03_pictor_orion_lmc_crab",
            samples_per_data_set=15360,
            buffer_depth=20,
            gpu_buffer_depth=3,
            max_beams=10,
            conjugate_gains=True,
            uncalibrated_beams=[8],
            beam_targets=[
                {"ra_deg": 79.9571, "dec_deg": -45.7789},                   # 0: Pictor A
                {"ra_deg": 83.8221, "dec_deg": -5.3911},                    # 1: Orion A
                {"ra_deg": 84.6767, "dec_deg": -69.1008},                   # 2: LMC 30 Dor
                {"ra_deg": 83.6329, "dec_deg": 22.0144},                    # 3: Crab / Tau A
                {"ra_deg": 35.0, "dec_deg": -45.8},                         # 4: OFF-Pictor (~31 deg away)
                {"ra_deg": 39.0, "dec_deg": -5.4},                          # 5: OFF-Orion (~45 deg away)
                {"l0": 0.45, "m0": -0.35},                                  # 6: OFF-LMC (updated live same-El)
                {"l0": 0.65, "m0": 0.45},                                   # 7: OFF-Crab (updated live same-El)
                {"ra_deg": 79.9571, "dec_deg": -45.7789},                   # 8: Pictor A Uncalibrated
                {"l0": 0.0, "m0": 0.0},                                     # 9: Zenith
            ],
            rfi_enabled=True,
            rfi_blank_voltages=True,
            accumulate_frames=20,
            num_frames_per_file=585,
            exit_after_n_files=7,  # 69.9 min
            skip_frames=1,
        ),
        "04_test04_vela_pulsar_transit.yaml": build_beam_config(
            run_id="test04_vela_pulsar_transit",
            samples_per_data_set=768,  # 2.56 ms
            buffer_depth=256,
            gpu_buffer_depth=4,
            max_beams=3,
            conjugate_gains=True,
            uncalibrated_beams=[1],
            beam_targets=[
                {"ra_deg": 128.8359, "dec_deg": -45.1764},                  # 0: Vela Cal ON
                {"ra_deg": 128.8359, "dec_deg": -45.1764},                  # 1: Vela Uncal ON
                {"ra_deg": 93.8359, "dec_deg": -45.1764},                   # 2: OFF-Vela (~25 deg away)
            ],
            rfi_enabled=True,
            rfi_blank_voltages=True,
            accumulate_frames=1,
            num_frames_per_file=30000,  # 76.8 s/file
            exit_after_n_files=16,      # 20.48 min
            skip_frames=200,
        ),
        "05_test05_psr_b0950_transit.yaml": build_beam_config(
            run_id="test05_psr_b0950_transit",
            samples_per_data_set=768,  # 2.56 ms
            buffer_depth=256,
            gpu_buffer_depth=4,
            max_beams=3,
            conjugate_gains=True,
            uncalibrated_beams=[1],
            beam_targets=[
                {"ra_deg": 148.2888, "dec_deg": 7.9266},                    # 0: PSR B0950+08 Cal ON
                {"ra_deg": 148.2888, "dec_deg": 7.9266},                    # 1: PSR B0950+08 Uncal ON
                {"ra_deg": 118.2888, "dec_deg": 7.9266},                    # 2: OFF-B0950 (~30 deg away)
            ],
            rfi_enabled=True,
            rfi_blank_voltages=True,
            accumulate_frames=1,
            num_frames_per_file=30000,
            exit_after_n_files=16,      # 20.48 min
            skip_frames=200,
        ),
        "06_test06a_sun_onoff_conj_true.yaml": build_beam_config(
            run_id="test06a_sun_onoff_conj_true",
            samples_per_data_set=15360,
            buffer_depth=20,
            gpu_buffer_depth=3,
            max_beams=4,
            conjugate_gains=True,
            uncalibrated_beams=[3],
            beam_targets=[
                {"l0": 0.78, "m0": 0.01},                                   # 0: Sun Cal ON (updated live)
                {"l0": 0.55, "m0": 0.56},                                   # 1: OFF North same-El
                {"l0": 0.55, "m0": -0.54},                                  # 2: OFF South same-El
                {"l0": 0.78, "m0": 0.01},                                   # 3: Sun Uncal ON
            ],
            rfi_enabled=True,
            rfi_blank_voltages=True,
            accumulate_frames=20,
            num_frames_per_file=585,
            exit_after_n_files=6,       # 59.9 min
            skip_frames=1,
        ),
        "07_test06b_sun_onoff_conj_false.yaml": build_beam_config(
            run_id="test06b_sun_onoff_conj_false",
            samples_per_data_set=15360,
            buffer_depth=20,
            gpu_buffer_depth=3,
            max_beams=4,
            conjugate_gains=False,
            uncalibrated_beams=[3],
            beam_targets=[
                {"l0": 0.69, "m0": 0.14},
                {"l0": 0.42, "m0": 0.57},
                {"l0": 0.55, "m0": -0.45},
                {"l0": 0.69, "m0": 0.14},
            ],
            rfi_enabled=True,
            rfi_blank_voltages=True,
            accumulate_frames=20,
            num_frames_per_file=580,
            exit_after_n_files=1,       # 9.9 min
            skip_frames=1,
        ),
        "08_test07_cena_dual_source.yaml": build_beam_config(
            run_id="test07_cena_dual_source",
            samples_per_data_set=15360,
            buffer_depth=20,
            gpu_buffer_depth=3,
            max_beams=6,
            conjugate_gains=True,
            uncalibrated_beams=[1],
            beam_targets=[
                {"ra_deg": 201.3650, "dec_deg": -43.0192},                  # 0: Cen A Cal ON
                {"ra_deg": 201.3650, "dec_deg": -43.0192},                  # 1: Cen A Uncal
                {"l0": 0.0872, "m0": -0.1691},                              # 2: Cen A +5 deg E (updated live)
                {"l0": -0.0872, "m0": -0.1691},                             # 3: Cen A -5 deg W (updated live)
                {"l0": 0.56, "m0": 0.21},                                   # 4: Sun simultaneous monitor (updated live)
                {"l0": cena_off_l, "m0": cena_off_m},                       # 5: Deep OFF
            ],
            rfi_enabled=True,
            rfi_blank_voltages=True,
            accumulate_frames=20,
            num_frames_per_file=585,
            exit_after_n_files=6,       # 59.9 min
            skip_frames=1,
        ),
        "09_test08_solar_chromaticity_2d.yaml": build_beam_config(
            run_id="test08_solar_chromaticity_2d",
            samples_per_data_set=15360,
            buffer_depth=20,
            gpu_buffer_depth=3,
            max_beams=10,
            conjugate_gains=True,
            uncalibrated_beams=[8],
            beam_targets=[
                {"l0": 0.0005, "m0": 0.2356},                               # 0: Tracked Sun Cal ON
                {"l0": 0.0, "m0": 0.2356},                                  # 1: Parked meridian drift at transit
                {"l0": 0.0355, "m0": 0.2356},                               # 2: E +2 deg
                {"l0": 0.0705, "m0": 0.2356},                               # 3: E +4 deg
                {"l0": 0.1225, "m0": 0.2356},                               # 4: E +7 deg
                {"l0": -0.0695, "m0": 0.2356},                              # 5: W -4 deg
                {"l0": 0.0005, "m0": 0.3056},                               # 6: N +4 deg
                {"l0": 0.0005, "m0": 0.1136},                               # 7: S -7 deg
                {"l0": 0.0005, "m0": 0.2356},                               # 8: Uncalibrated Tracked Sun ON
                {"l0": solar_chr_off_l, "m0": solar_chr_off_m},             # 9: Deep South OFF (grating-safe when optimized)
            ],
            rfi_enabled=True,
            rfi_blank_voltages=True,
            accumulate_frames=20,
            num_frames_per_file=585,
            exit_after_n_files=9,       # 89.9 min
            skip_frames=1,
        ),
        "10_test09a_galactic_center_plane.yaml": build_beam_config(
            run_id="test09a_galactic_center_plane",
            samples_per_data_set=15360,
            buffer_depth=20,
            gpu_buffer_depth=3,
            max_beams=10,
            conjugate_gains=True,
            uncalibrated_beams=[],
            beam_targets=[
                compute_gal_b_icrs(0.0),                                    # 0: l_gal=0, b=0 (Sgr A*)
                compute_gal_b_icrs(4.0),                                    # 1: b=+4 deg
                compute_gal_b_icrs(-4.0),                                   # 2: b=-4 deg
                compute_gal_b_icrs(8.0),                                    # 3: b=+8 deg
                compute_gal_b_icrs(-8.0),                                   # 4: b=-8 deg
                compute_gal_b_icrs(15.0),                                   # 5: b=+15 deg
                compute_gal_b_icrs(-15.0),                                  # 6: b=-15 deg
                {"l0": 0.0, "m0": 0.0768},                                  # 7: Parked zenith drift at Sgr A* transit
                {"l0": -0.45, "m0": 0.18},                                  # 8: Simultaneous Sun monitor (updated live)
                {"ra_deg": 318.0, "dec_deg": -35.0},                        # 9: Cold-sky OFF (|b_gal|~43.5 deg)
            ],
            rfi_enabled=True,
            rfi_blank_voltages=True,
            accumulate_frames=20,
            num_frames_per_file=585,
            exit_after_n_files=7,       # 69.9 min
            skip_frames=1,
        ),
        "11_test09b_psr_b1749_28_zenith.yaml": build_beam_config(
            run_id="test09b_psr_b1749_28_zenith",
            samples_per_data_set=1536,  # 5.12 ms
            buffer_depth=128,
            gpu_buffer_depth=4,
            max_beams=3,
            conjugate_gains=True,
            uncalibrated_beams=[1],
            beam_targets=[
                {"ra_deg": 268.2445, "dec_deg": -28.1104},                  # 0: PSR B1749-28 Cal ON
                {"ra_deg": 268.2445, "dec_deg": -28.1104},                  # 1: PSR B1749-28 Uncal ON
                {"ra_deg": 228.2445, "dec_deg": -28.1104},                  # 2: OFF (~35 deg away)
            ],
            rfi_enabled=True,
            rfi_blank_voltages=True,
            accumulate_frames=1,
            num_frames_per_file=15000,  # 76.8 s/file
            exit_after_n_files=16,      # 20.48 min
            skip_frames=100,
        ),
        "12_test10_production_stress_10beams.yaml": build_beam_config(
            run_id="test10_production_stress_10beams",
            samples_per_data_set=15360,
            buffer_depth=20,
            gpu_buffer_depth=3,
            max_beams=10,
            conjugate_gains=True,
            uncalibrated_beams=[],
            beam_targets=[
                {"l0": -0.60, "m0": 0.25},                                  # 0: Sun (updated live)
                {"ra_deg": 201.3650, "dec_deg": -43.0192},                  # 1: Cen A
                {"l0": -0.086, "m0": 0.625},                                # 2: GOES-19
                {"l0": -0.54, "m0": 0.54},                                  # 3: MUOS-1
                {"l0": 0.86, "m0": 0.41},                                   # 4: MUOS-3
                {"l0": -0.30, "m0": 0.10},                                  # 5: Active LEO (updated live)
                {"ra_deg": 266.4168, "dec_deg": -29.0078},                  # 6: Sgr A*
                {"ra_deg": 50.6737, "dec_deg": -37.2083},                   # 7: Fornax A
                {"l0": 0.0, "m0": 0.0},                                     # 8: Zenith
                {"l0": 0.45, "m0": -0.55},                                  # 9: OFF sky control
            ],
            rfi_enabled=True,
            rfi_blank_voltages=True,
            accumulate_frames=20,
            num_frames_per_file=585,
            exit_after_n_files=6,       # 59.9 min
            skip_frames=1,
            copy_formed_beams_to_host=True,
        ),
        "13_test11_24h_phase_drift_corr.yaml": build_corr_config(
            base_dir="/data/corr/corr_1121_drift" if (args.generate_continuation_20261121 or args.generate_run_20261121_full) else "/data/corr/corr_1120_drift",
            exit_after_n_files=120,     # 20.48 min (~1.31 GB)
        ),
        "14_test12a_geo_satellites_survey.yaml": build_beam_config(
            run_id="test12a_geo_satellites_survey",
            samples_per_data_set=15360,
            buffer_depth=20,
            gpu_buffer_depth=3,
            max_beams=10,
            conjugate_gains=True,
            uncalibrated_beams=[],
            beam_targets=[
                {"l0": -0.5520, "m0": 0.4914},  # 0: MUOS-1 (updated live)
                {"l0": 0.8871, "m0": 0.2608},   # 1: MUOS-3 (updated live)
                {"l0": -0.0861, "m0": 0.6250},  # 2: GOES-19 (updated live)
                {"l0": -0.9533, "m0": 0.2316},  # 3: GOES-18 (updated live)
                {"l0": 0.8115, "m0": 0.2686},   # 4: UFO-10 (updated live)
                {"l0": 0.8601, "m0": 0.2906},   # 5: SKYNET 5C (updated live)
                {"l0": -0.3696, "m0": 0.3483},  # 6: USA 115 / MILSTAR (updated live)
                {"l0": 0.3252, "m0": 0.4411},   # 7: INMARSAT 3-F5 (updated live)
                {"l0": 0.0, "m0": -0.8192},     # 8: OFF-South El 35 deg control
                {"l0": 0.0, "m0": 0.0},         # 9: Zenith control
            ],
            rfi_enabled=True,
            rfi_blank_voltages=False,
            accumulate_frames=20,
            num_frames_per_file=585,
            exit_after_n_files=3,       # 29.95 min
            skip_frames=1,
            copy_formed_beams_to_host=False,
        ),
        "15_test12b_airport_santiago_horizon.yaml": build_beam_config(
            run_id="test12b_airport_santiago_horizon",
            samples_per_data_set=15360,
            buffer_depth=20,
            gpu_buffer_depth=3,
            max_beams=10,
            conjugate_gains=True,
            uncalibrated_beams=[8],
            beam_targets=[
                {"l0": 0.9176, "m0": 0.3975},   # 0: SCL Airport Horizon (Az 66.58, El 0.5)
                {"l0": 0.9037, "m0": 0.3914},   # 1: SCL Approach (Az 66.58, El 10.0)
                {"l0": 0.9958, "m0": -0.0912},  # 2: Santiago Centro Plaza (Az 95.23, El 0.5)
                {"l0": 0.9996, "m0": -0.0229},  # 3: Cerro San Cristobal Cal (Az 91.31, El 1.0)
                {"l0": 0.9835, "m0": -0.0515},  # 4: Santiago Elevated (Az 93.0, El 10.0)
                {"l0": -0.99996, "m0": 0.0},    # 5: Quiet Horizon West (Az 270, El 0.5)
                {"l0": 0.0, "m0": -0.99996},    # 6: Quiet Horizon South (Az 180, El 0.5)
                {"l0": -0.5000, "m0": 0.8660},  # 7: Quiet Horizon NW (Az 330, El 0.5)
                {"l0": 0.9996, "m0": -0.0229},  # 8: San Cristobal Uncal (Az 91.31, El 1.0)
                {"l0": 0.0, "m0": 0.0},         # 9: Zenith control
            ],
            rfi_enabled=True,
            rfi_blank_voltages=False,
            accumulate_frames=20,
            num_frames_per_file=585,
            exit_after_n_files=3,       # 29.95 min
            skip_frames=1,
            copy_formed_beams_to_host=False,
        ),
    }

    if args.generate_continuation_20261121:
        patch_tracker_solar_m0(configs["09_test08_solar_chromaticity_2d.yaml"], compute_solar_transit_m0("2026-11-21"))
        cont_keys = {
            "05_test05_psr_b0950_transit.yaml",
            "06_test06a_sun_onoff_conj_true.yaml",
            "07_test06b_sun_onoff_conj_false.yaml",
            "08_test07_cena_dual_source.yaml",
            "09_test08_solar_chromaticity_2d.yaml",
            "10_test09a_galactic_center_plane.yaml",
            "11_test09b_psr_b1749_28_zenith.yaml",
            "12_test10_production_stress_10beams.yaml",
            "13_test11_24h_phase_drift_corr.yaml",
        }
        configs = {k: v for k, v in configs.items() if k in cont_keys}
    elif args.generate_run_20261121_full:
        patch_tracker_solar_m0(configs["09_test08_solar_chromaticity_2d.yaml"], compute_solar_transit_m0("2026-11-22"))
        full_keys = {
            "05_test05_psr_b0950_transit.yaml",
            "06_test06a_sun_onoff_conj_true.yaml",
            "07_test06b_sun_onoff_conj_false.yaml",
            "08_test07_cena_dual_source.yaml",
            "09_test08_solar_chromaticity_2d.yaml",
            "10_test09a_galactic_center_plane.yaml",
            "11_test09b_psr_b1749_28_zenith.yaml",
            "12_test10_production_stress_10beams.yaml",
            "13_test11_24h_phase_drift_corr.yaml",
            "14_test12a_geo_satellites_survey.yaml",
            "15_test12b_airport_santiago_horizon.yaml",
        }
        configs = {k: v for k, v in configs.items() if k in full_keys}

    for fname, data in configs.items():
        p_out = cfg_dir / fname
        header = f"---\n# Auto-generated ms-x01 4-antenna v2 config: {fname}\n"
        p_out.write_text(header + yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        print(f"Wrote {p_out}")

    if args.rebuild_beam_configs and MANIFEST_PATH.exists() and not args.generate_continuation_20261121 and not args.generate_run_20261121_full:
        print("Preserving existing schedule_manifest.json (--rebuild-beam-configs).")
        return

    now_dt = datetime.now(timezone.utc)
    p1_start = now_dt
    p1_end = p1_start + timedelta(minutes=92)

    runs = [
        {
            "id": "phase1_corr_cal",
            "title": "Phase 1: 82-min GPU Correlator Calibration Capture (corr_1119_cal)",
            "start_utc": p1_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end_utc": p1_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "duration_s": 5100,
            "disk_cap_mb": 7200,
            "config_path": str(cfg_dir / "00_phase1_corr_cal.yaml"),
            "output_dir": "/data/corr/corr_1119_cal",
            "sidecar_mode": "corr_monitor",
            "verifier": None,
            "post_cal_handoff": True,
        },
        {
            "id": "test01_muos_rfi_425_475",
            "title": "Test 1: MUOS 360-380 MHz + 425/475 MHz Unblanked RFI & Satellite Tracking",
            "start_utc": "2026-11-19T20:55:00Z",
            "end_utc": "2026-11-19T21:55:00Z",
            "duration_s": 3600,
            "disk_cap_mb": 512,
            "config_path": str(cfg_dir / "01_test01_muos_rfi_425_475.yaml"),
            "output_dir": str(V2_DATA_ROOT / "test01_muos_rfi_425_475"),
            "sidecar_mode": "muos_rfi",
            "verifier": "verify_muos_rfi.py",
        },
        {
            "id": "test02_fornax_a_night_chromaticity",
            "title": "Test 2: Zero-Sun Nighttime Fornax A Beam Chromaticity & Meridian Drift Scan",
            "start_utc": "2026-11-20T03:20:00Z",
            "end_utc": "2026-11-20T05:00:00Z",
            "duration_s": 6000,
            "disk_cap_mb": 768,
            "config_path": str(cfg_dir / "02_test02_fornax_a_night_chromaticity.yaml"),
            "output_dir": str(V2_DATA_ROOT / "test02_fornax_a_night_chromaticity"),
            "sidecar_mode": "fornax_chromaticity",
            "verifier": "verify_chromaticity.py",
        },
        {
            "id": "test03_pictor_orion_lmc_crab",
            "title": "Test 3: Deep-Night 10-Beam Multi-Source (Pictor A, Orion A, LMC, Crab)",
            "start_utc": "2026-11-20T05:40:00Z",
            "end_utc": "2026-11-20T06:50:00Z",
            "duration_s": 4200,
            "disk_cap_mb": 512,
            "config_path": str(cfg_dir / "03_test03_pictor_orion_lmc_crab.yaml"),
            "output_dir": str(V2_DATA_ROOT / "test03_pictor_orion_lmc_crab"),
            "sidecar_mode": "multi_source",
            "verifier": "verify_multi_source.py",
        },
        {
            "id": "test04_vela_pulsar_transit",
            "title": "Test 4: Vela Pulsar (PSR B0833-45) 89.3 ms Transit Fold @ 2.56 ms",
            "start_utc": "2026-11-20T09:10:00Z",
            "end_utc": "2026-11-20T09:31:00Z",
            "duration_s": 1245,
            "disk_cap_mb": 7200,
            "config_path": str(cfg_dir / "04_test04_vela_pulsar_transit.yaml"),
            "output_dir": str(V2_DATA_ROOT / "test04_vela_pulsar_transit"),
            "sidecar_mode": "pulsar_static",
            "verifier": "verify_pulsar_generic.py",
            "verifier_args": ["--target", "vela"],
        },
        {
            "id": "test05_psr_b0950_transit",
            "title": "Test 5: Northern/Low-DM PSR B0950+08 Transit Fold @ 2.56 ms",
            "start_utc": "2026-11-21T10:25:00Z" if args.generate_continuation_20261121 else "2026-11-20T10:25:00Z",
            "end_utc": "2026-11-21T10:46:00Z" if args.generate_continuation_20261121 else "2026-11-20T10:46:00Z",
            "duration_s": 1245,
            "disk_cap_mb": 7200,
            "config_path": str(cfg_dir / "05_test05_psr_b0950_transit.yaml"),
            "output_dir": str(V2_DATA_ROOT / "test05_psr_b0950_transit"),
            "sidecar_mode": "pulsar_static",
            "verifier": "verify_pulsar_generic.py",
            "verifier_args": ["--target", "b0950"],
        },
        {
            "id": "test06a_sun_onoff_conj_true",
            "title": "Test 6A: Morning Sun ON/OFF Allan Variance (conjugate_gains=true)",
            "start_utc": "2026-11-21T12:15:00Z" if args.generate_continuation_20261121 else "2026-11-20T12:15:00Z",
            "end_utc": "2026-11-21T13:15:00Z" if args.generate_continuation_20261121 else "2026-11-20T13:15:00Z",
            "duration_s": 3600,
            "disk_cap_mb": 512,
            "config_path": str(cfg_dir / "06_test06a_sun_onoff_conj_true.yaml"),
            "output_dir": str(V2_DATA_ROOT / "test06a_sun_onoff_conj_true"),
            "sidecar_mode": "sun_onoff",
            "verifier": "verify_sun_onoff.py",
        },
        {
            "id": "test06b_sun_onoff_conj_false",
            "title": "Test 6B: Morning Sun Sign Test (conjugate_gains=false)",
            "start_utc": "2026-11-21T13:16:00Z" if args.generate_continuation_20261121 else "2026-11-20T13:15:30Z",
            "end_utc": "2026-11-21T13:26:30Z" if args.generate_continuation_20261121 else "2026-11-20T13:26:00Z",
            "duration_s": 610,
            "disk_cap_mb": 256,
            "config_path": str(cfg_dir / "07_test06b_sun_onoff_conj_false.yaml"),
            "output_dir": str(V2_DATA_ROOT / "test06b_sun_onoff_conj_false"),
            "sidecar_mode": "sun_onoff",
            "verifier": "verify_sun_onoff.py",
        },
        {
            "id": "test07_cena_dual_source",
            "title": "Test 7: Centaurus A Transit + Simultaneous Sun Monitor",
            "start_utc": "2026-11-21T13:40:00Z" if args.generate_continuation_20261121 else "2026-11-20T13:45:00Z",
            "end_utc": "2026-11-21T14:40:00Z" if args.generate_continuation_20261121 else "2026-11-20T14:45:00Z",
            "duration_s": 3600,
            "disk_cap_mb": 512,
            "config_path": str(cfg_dir / "08_test07_cena_dual_source.yaml"),
            "output_dir": str(V2_DATA_ROOT / "test07_cena_dual_source"),
            "sidecar_mode": "cena_dual",
            "verifier": "verify_multi_source.py",
        },
        {
            "id": "test08_solar_chromaticity_2d",
            "title": "Test 8: Flagship 2D Solar Chromaticity & Drift-Scan Transit Suite",
            "start_utc": "2026-11-21T15:40:00Z" if args.generate_continuation_20261121 else "2026-11-20T15:40:00Z",
            "end_utc": "2026-11-21T17:10:00Z" if args.generate_continuation_20261121 else "2026-11-20T17:10:00Z",
            "duration_s": 5400,
            "disk_cap_mb": 768,
            "config_path": str(cfg_dir / "09_test08_solar_chromaticity_2d.yaml"),
            "output_dir": str(V2_DATA_ROOT / "test08_solar_chromaticity_2d"),
            "sidecar_mode": "solar_chromaticity_2d",
            "verifier": "verify_chromaticity.py",
        },
        {
            "id": "test09a_galactic_center_plane",
            "title": "Test 9A: Galactic Center (Sgr A*) Latitude Cut & Zenith Drift Scan",
            "start_utc": "2026-11-21T17:40:00Z" if args.generate_continuation_20261121 else "2026-11-20T17:45:00Z",
            "end_utc": "2026-11-21T18:50:00Z" if args.generate_continuation_20261121 else "2026-11-20T18:55:00Z",
            "duration_s": 4200,
            "disk_cap_mb": 768,
            "config_path": str(cfg_dir / "10_test09a_galactic_center_plane.yaml"),
            "output_dir": str(V2_DATA_ROOT / "test09a_galactic_center_plane"),
            "sidecar_mode": "galactic_center",
            "verifier": "verify_galactic_center.py",
        },
        {
            "id": "test09b_psr_b1749_28_zenith",
            "title": "Test 9B: Galactic Plane PSR B1749-28 High-El Fold @ 5.12 ms",
            "start_utc": "2026-11-21T18:51:00Z" if args.generate_continuation_20261121 else "2026-11-20T18:56:00Z",
            "end_utc": "2026-11-21T19:12:00Z" if args.generate_continuation_20261121 else "2026-11-20T19:17:00Z",
            "duration_s": 1245,
            "disk_cap_mb": 3800,
            "config_path": str(cfg_dir / "11_test09b_psr_b1749_28_zenith.yaml"),
            "output_dir": str(V2_DATA_ROOT / "test09b_psr_b1749_28_zenith"),
            "sidecar_mode": "pulsar_static",
            "verifier": "verify_pulsar_generic.py",
            "verifier_args": ["--target", "b1749"],
        },
        {
            "id": "test10_production_stress_10beams",
            "title": "Test 10: 60-min 10-Beam Production Stress + RFI (No Triggers/Dumps)",
            "start_utc": "2026-11-21T19:25:00Z" if args.generate_continuation_20261121 else "2026-11-20T19:30:00Z",
            "end_utc": "2026-11-21T20:25:00Z" if args.generate_continuation_20261121 else "2026-11-20T20:30:00Z",
            "duration_s": 3600,
            "disk_cap_mb": 512,
            "config_path": str(cfg_dir / "12_test10_production_stress_10beams.yaml"),
            "output_dir": str(V2_DATA_ROOT / "test10_production_stress_10beams"),
            "sidecar_mode": "stress",
            "verifier": "verify_stress.py",
        },
        {
            "id": "test11_24h_phase_drift_corr",
            "title": "Test 11: 24-Hour Post-Epoch Phase & Delay Drift Correlator Audit",
            "start_utc": "2026-11-21T20:35:00Z" if args.generate_continuation_20261121 else "2026-11-20T20:45:00Z",
            "end_utc": "2026-11-21T20:57:00Z" if args.generate_continuation_20261121 else "2026-11-20T21:07:00Z",
            "duration_s": 1260,
            "disk_cap_mb": 2048,
            "config_path": str(cfg_dir / "13_test11_24h_phase_drift_corr.yaml"),
            "output_dir": "/data/corr/corr_1121_drift" if args.generate_continuation_20261121 else "/data/corr/corr_1120_drift",
            "sidecar_mode": "corr_monitor",
            "verifier": "calibrate_corr_epoch.py",
            "verifier_args": [
                "--corr-dir",
                "/data/corr/corr_1121_drift" if args.generate_continuation_20261121 else "/data/corr/corr_1120_drift",
                "--out-dir",
                str(V2_DATA_ROOT / "test11_24h_phase_drift_corr"),
                "--verify-only-gains",
                GAINS_FILE,
                "--min-sun-el",
                "15.0",
            ],
        },
    ]

    if args.generate_run_20261121_full:
        t0 = now_dt + timedelta(minutes=2)

        def fmt(dt: datetime) -> str:
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        runs = [
            {
                "id": "test11_24h_phase_drift_corr",
                "title": "Test 11: 48-Hour Post-Epoch Phase & Delay Drift Correlator Audit",
                "start_utc": fmt(t0),
                "end_utc": fmt(t0 + timedelta(minutes=22)),
                "duration_s": 1260,
                "disk_cap_mb": 2048,
                "config_path": str(cfg_dir / "13_test11_24h_phase_drift_corr.yaml"),
                "output_dir": "/data/corr/corr_1121_drift",
                "sidecar_mode": "corr_monitor",
                "verifier": "calibrate_corr_epoch.py",
                "verifier_args": [
                    "--corr-dir",
                    "/data/corr/corr_1121_drift",
                    "--out-dir",
                    str(V2_DATA_ROOT / "test11_24h_phase_drift_corr"),
                    "--verify-only-gains",
                    GAINS_FILE,
                    "--min-sun-el",
                    "15.0",
                ],
            },
            {
                "id": "test12a_geo_satellites_survey",
                "title": "Test 12A: Multi-GEO Satellite UHF Survey (MUOS-1/3, GOES-18/19, UFO-10, SKYNET-5C, USA-115, INMARSAT-3F5)",
                "start_utc": fmt(t0 + timedelta(minutes=24)),
                "end_utc": fmt(t0 + timedelta(minutes=55)),
                "duration_s": 1820,
                "disk_cap_mb": 512,
                "config_path": str(cfg_dir / "14_test12a_geo_satellites_survey.yaml"),
                "output_dir": str(V2_DATA_ROOT / "test12a_geo_satellites_survey"),
                "sidecar_mode": "geo_satellites",
                "verifier": "verify_geo_horizon.py",
                "verifier_args": ["--mode", "geo"],
            },
            {
                "id": "test12b_airport_santiago_horizon",
                "title": "Test 12B: SCL Airport, Santiago Centro & Cerro San Cristobal Horizon Directional Power Study",
                "start_utc": fmt(t0 + timedelta(minutes=57)),
                "end_utc": fmt(t0 + timedelta(minutes=88)),
                "duration_s": 1820,
                "disk_cap_mb": 512,
                "config_path": str(cfg_dir / "15_test12b_airport_santiago_horizon.yaml"),
                "output_dir": str(V2_DATA_ROOT / "test12b_airport_santiago_horizon"),
                "sidecar_mode": "airport_santiago_horizon",
                "verifier": "verify_geo_horizon.py",
                "verifier_args": ["--mode", "horizon"],
            },
            {
                "id": "test10_production_stress_10beams",
                "title": "Test 10: 60-min 10-Beam Production Stress + RFI (No Triggers/Dumps)",
                "start_utc": fmt(t0 + timedelta(minutes=90)),
                "end_utc": fmt(t0 + timedelta(minutes=151)),
                "duration_s": 3620,
                "disk_cap_mb": 512,
                "config_path": str(cfg_dir / "12_test10_production_stress_10beams.yaml"),
                "output_dir": str(V2_DATA_ROOT / "test10_production_stress_10beams"),
                "sidecar_mode": "stress",
                "verifier": "verify_stress.py",
            },
            {
                "id": "test05_psr_b0950_transit",
                "title": "Test 5: Northern/Low-DM PSR B0950+08 Transit Fold @ 2.56 ms",
                "start_utc": "2026-11-22T10:21:00Z",
                "end_utc": "2026-11-22T10:42:00Z",
                "duration_s": 1245,
                "disk_cap_mb": 7200,
                "config_path": str(cfg_dir / "05_test05_psr_b0950_transit.yaml"),
                "output_dir": str(V2_DATA_ROOT / "test05_psr_b0950_transit"),
                "sidecar_mode": "pulsar_static",
                "verifier": "verify_pulsar_generic.py",
                "verifier_args": ["--target", "b0950"],
            },
            {
                "id": "test06a_sun_onoff_conj_true",
                "title": "Test 6A: Morning Sun ON/OFF Allan Variance (conjugate_gains=true)",
                "start_utc": "2026-11-22T12:15:00Z",
                "end_utc": "2026-11-22T13:15:00Z",
                "duration_s": 3600,
                "disk_cap_mb": 512,
                "config_path": str(cfg_dir / "06_test06a_sun_onoff_conj_true.yaml"),
                "output_dir": str(V2_DATA_ROOT / "test06a_sun_onoff_conj_true"),
                "sidecar_mode": "sun_onoff",
                "verifier": "verify_sun_onoff.py",
            },
            {
                "id": "test06b_sun_onoff_conj_false",
                "title": "Test 6B: Morning Sun Sign Test (conjugate_gains=false)",
                "start_utc": "2026-11-22T13:16:00Z",
                "end_utc": "2026-11-22T13:26:30Z",
                "duration_s": 610,
                "disk_cap_mb": 256,
                "config_path": str(cfg_dir / "07_test06b_sun_onoff_conj_false.yaml"),
                "output_dir": str(V2_DATA_ROOT / "test06b_sun_onoff_conj_false"),
                "sidecar_mode": "sun_onoff",
                "verifier": "verify_sun_onoff.py",
            },
            {
                "id": "test07_cena_dual_source",
                "title": "Test 7: Centaurus A Transit + Simultaneous Sun Monitor",
                "start_utc": "2026-11-22T13:36:00Z",
                "end_utc": "2026-11-22T14:36:00Z",
                "duration_s": 3600,
                "disk_cap_mb": 512,
                "config_path": str(cfg_dir / "08_test07_cena_dual_source.yaml"),
                "output_dir": str(V2_DATA_ROOT / "test07_cena_dual_source"),
                "sidecar_mode": "cena_dual",
                "verifier": "verify_multi_source.py",
            },
            {
                "id": "test08_solar_chromaticity_2d",
                "title": "Test 8: Flagship 2D Solar Chromaticity & Drift-Scan Transit Suite",
                "start_utc": "2026-11-22T15:40:00Z",
                "end_utc": "2026-11-22T17:10:00Z",
                "duration_s": 5400,
                "disk_cap_mb": 768,
                "config_path": str(cfg_dir / "09_test08_solar_chromaticity_2d.yaml"),
                "output_dir": str(V2_DATA_ROOT / "test08_solar_chromaticity_2d"),
                "sidecar_mode": "solar_chromaticity_2d",
                "verifier": "verify_chromaticity.py",
            },
            {
                "id": "test09a_galactic_center_plane",
                "title": "Test 9A: Galactic Center (Sgr A*) Latitude Cut & Zenith Drift Scan",
                "start_utc": "2026-11-22T17:36:00Z",
                "end_utc": "2026-11-22T18:46:00Z",
                "duration_s": 4200,
                "disk_cap_mb": 768,
                "config_path": str(cfg_dir / "10_test09a_galactic_center_plane.yaml"),
                "output_dir": str(V2_DATA_ROOT / "test09a_galactic_center_plane"),
                "sidecar_mode": "galactic_center",
                "verifier": "verify_galactic_center.py",
            },
            {
                "id": "test09b_psr_b1749_28_zenith",
                "title": "Test 9B: Galactic Plane PSR B1749-28 High-El Fold @ 5.12 ms",
                "start_utc": "2026-11-22T18:47:00Z",
                "end_utc": "2026-11-22T19:08:00Z",
                "duration_s": 1245,
                "disk_cap_mb": 3800,
                "config_path": str(cfg_dir / "11_test09b_psr_b1749_28_zenith.yaml"),
                "output_dir": str(V2_DATA_ROOT / "test09b_psr_b1749_28_zenith"),
                "sidecar_mode": "pulsar_static",
                "verifier": "verify_pulsar_generic.py",
                "verifier_args": ["--target", "b1749"],
            },
        ]
        manifest_out = FULL_20261121_MANIFEST_PATH
        campaign_name = "commissioning_20261119_v2_run_20261121_full"
        stop_utc = "2026-11-23T00:00:00Z"
    elif args.generate_continuation_20261121:
        cont_ids = {
            "test05_psr_b0950_transit",
            "test06a_sun_onoff_conj_true",
            "test06b_sun_onoff_conj_false",
            "test07_cena_dual_source",
            "test08_solar_chromaticity_2d",
            "test09a_galactic_center_plane",
            "test09b_psr_b1749_28_zenith",
            "test10_production_stress_10beams",
            "test11_24h_phase_drift_corr",
        }
        runs = [r for r in runs if r["id"] in cont_ids]
        manifest_out = CONTINUATION_MANIFEST_PATH
        campaign_name = "commissioning_20261119_v2_continuation_20261121"
        stop_utc = "2026-11-22T02:00:00Z"
    else:
        manifest_out = MANIFEST_PATH
        campaign_name = "commissioning_20261119_v2"
        stop_utc = "2026-11-21T06:00:00Z"

    manifest = {
        "campaign": campaign_name,
        "generated_utc": now_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "site": {"lat_deg": SITE_LAT_DEG, "lon_deg": SITE_LON_DEG, "alt_m": SITE_ALT_M},
        "active_antennas": ACTIVE_ANTENNAS,
        "gains_file": GAINS_FILE,
        "min_free_disk_gb": 100.0,
        "scheduler_stop_utc": stop_utc,
        "runs": runs,
    }
    manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_out} ({len(runs)} runs)")


if __name__ == "__main__":
    main()
