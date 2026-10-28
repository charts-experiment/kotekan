#!/usr/bin/env python3
"""Replay helpers for the Kotekan correlator with the visibility calibrator.

Builds the YAML for ``rawFileRead -> cudaProcess(Shuffle, Astron correlator,
[cudaVisibilityCalibrator]) -> sink``, runs Kotekan on it, and decodes the
``int32`` correlation frames it writes. Shared by the visibility-calibrator
tests and benchmark.

Environment:
  CHARTS_TEST_LD_LIBRARY_PATH  Extra library directories searched first when
      launching Kotekan, e.g. an NVRTC that matches the installed GPU driver.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from .presets import find_kotekan_binary

BLOCK_SIZE = 2
FRAME_HEADER_BYTES = 4  # rawFileWrite prefixes the uint32 metadata size


def num_blocks_for(num_elements: int) -> int:
    """Triangular 2x2 blocks of the Astron correlator (528 for 64 antennas)."""
    n = num_elements // BLOCK_SIZE
    return n * (n + 1) // 2


def kotekan_env(repo_root: Path) -> Dict[str, str]:
    """Environment with the n2k and optional user library directories first."""
    env = os.environ.copy()
    paths = [p for p in env.get("CHARTS_TEST_LD_LIBRARY_PATH", "").split(":") if p]
    paths.append(str(repo_root / "build" / "external" / "n2k"))
    if env.get("LD_LIBRARY_PATH"):
        paths.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = ":".join(paths)
    return env


def create_replay_yaml(
    yaml_path: Path,
    baseband_dir: Path,
    window_name: str,
    out_dir: Optional[Path],
    num_elements: int = 64,
    num_local_freq: int = 336,
    samples_per_data_set: int = 1536,
    buffer_depth: int = 4,
    calibrator: Optional[Dict[str, Any]] = None,
    num_frames: Optional[int] = None,
    end_interrupt: bool = True,
    frame_arrival_period: Optional[float] = None,
) -> Path:
    """Write the replay config.

    ``calibrator`` is ``None`` to leave the command out, or a dict of its extra
    options (``{}`` gives a pass-through command, ``{"calibration_file": ...,
    "freq_offset": ...}`` a real one). With ``out_dir`` the correlation frames
    are written there; with ``None`` they go to a ``restInspectFrame`` sink,
    which makes no disk traffic and is what the benchmark uses.
    """
    num_blocks = num_blocks_for(num_elements)
    if frame_arrival_period is None:
        frame_arrival_period = samples_per_data_set * (10.0 / 3.0) * 1e-6  # 3.33 us/sample

    commands: List[Dict[str, Any]] = [
        {"name": "cudaInputData", "in_buf": "host_voltage", "gpu_mem": "voltage_0"},
        {"name": "cudaSyncInput"},
        {"name": "cudaShuffleAstron", "gpu_mem_voltage": "voltage_0",
         "gpu_mem_ordered_voltage": "ordered_voltage_0"},
        {"name": "cudaCorrelatorAstron", "gpu_mem_voltage": "ordered_voltage_0",
         "gpu_mem_correlation_matrix": "correlation_matrix_0"},
    ]
    if calibrator is not None:
        commands.append({"name": "cudaVisibilityCalibrator",
                         "gpu_mem_correlation_matrix": "correlation_matrix_0",
                         **{k: (str(v) if isinstance(v, Path) else v)
                            for k, v in calibrator.items()}})
    commands += [
        {"name": "cudaSyncOutput"},
        {"name": "cudaOutputData", "in_buf": "host_voltage", "gpu_mem": "correlation_matrix_0",
         "out_buf": "host_correlation"},
    ]

    reader: Dict[str, Any] = {
        "kotekan_stage": "rawFileRead", "buf": "network_capture_buf",
        "base_dir": str(baseband_dir), "file_name": window_name, "file_ext": "bin",
        "end_interrupt": end_interrupt,
    }
    if out_dir is not None:
        sink: Dict[str, Any] = {
            "kotekan_stage": "rawFileWrite", "in_buf": "host_correlation_buffer",
            "base_dir": str(out_dir), "file_name": "corr", "file_ext": "bin",
            "num_frames_per_file": 1, "prefix_hostname": False,
        }
        if num_frames is not None:
            sink["exit_after_n_files"] = num_frames
    else:
        sink = {"kotekan_stage": "restInspectFrame", "in_buf": "host_correlation_buffer"}

    config = {
        "type": "config",
        "log_level": "info",
        "cpu_affinity": [0, 1, 2, 3],
        "num_elements": num_elements,
        "num_local_freq": num_local_freq,
        "samples_per_data_set": samples_per_data_set,
        "num_data_sets": 1,
        "block_size": BLOCK_SIZE,
        "num_blocks": num_blocks,
        "sizeof_int": 4,
        "buffer_depth": buffer_depth,
        "frame_arrival_period": frame_arrival_period,
        "main_pool": {"kotekan_metadata_pool": "chordMetadata", "num_metadata_objects": 60},
        "network_capture_buf": {
            "kotekan_buffer": "standard", "num_frames": "buffer_depth",
            "frame_size": "samples_per_data_set * num_local_freq * num_elements",
            "numa_node": 0, "metadata_pool": "main_pool",
        },
        "host_correlation_buffer": {
            "kotekan_buffer": "standard", "num_frames": "buffer_depth",
            "frame_size": "num_local_freq * num_blocks * (block_size*block_size) * 2 "
                          "* num_data_sets * sizeof_int",
            "metadata_pool": "main_pool",
        },
        "baseband_reader": reader,
        "gpu": {
            "profiling": True,
            "kernel_path": "lib/cuda/kernels",
            "gpu_0": {
                "kotekan_stage": "cudaProcess", "gpu_id": 0, "buffer_depth": 3,
                "commands": commands,
                "in_buffers": {"host_voltage": "network_capture_buf"},
                "out_buffers": {"host_correlation": "host_correlation_buffer"},
            },
        },
        "sink": sink,
    }
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return yaml_path


def run_kotekan_replay(
    yaml_path: Path,
    log_path: Path,
    repo_root: Path,
    kotekan_bin: Optional[Path] = None,
    timeout_s: float = 300.0,
) -> int:
    """Run Kotekan to completion on ``yaml_path`` from the repo root; return its exit code."""
    binary = kotekan_bin or find_kotekan_binary()
    if binary is None:
        raise FileNotFoundError("kotekan binary not found; build it or pass kotekan_bin")
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.run([str(binary), "-c", str(yaml_path)], cwd=repo_root,
                              env=kotekan_env(repo_root), stdout=log, stderr=subprocess.STDOUT,
                              timeout=timeout_s)
    return proc.returncode


def load_corr_frame(path: Path, num_local_freq: int, num_blocks: int) -> np.ndarray:
    """Decode one written frame to complex ``(freq, block, pol_a, pol_b)``.

    Values are complex128 so that integer counts up to 2**31 stay exact.
    """
    with open(path, "rb") as f:
        meta_size = int(np.frombuffer(f.read(FRAME_HEADER_BYTES), dtype="<u4")[0])
    count = num_local_freq * num_blocks * BLOCK_SIZE * BLOCK_SIZE * 2
    raw = np.fromfile(path, dtype="<i4", count=count, offset=FRAME_HEADER_BYTES + meta_size)
    if raw.size != count:
        raise ValueError(f"{path} holds {raw.size} int32 values; expected {count}")
    raw = raw.reshape(num_local_freq, num_blocks, BLOCK_SIZE, BLOCK_SIZE, 2)
    return raw[..., 0].astype(np.float64) + 1j * raw[..., 1].astype(np.float64)
