"""Stage A replay fixtures and configuration; no voltage-processing CPU oracle."""

from __future__ import annotations

import ctypes as ct
import json
import os
from pathlib import Path
import struct
import tempfile

import numpy as np
import yaml


class NativeChartsMetadata(ct.Structure):
    """Native rawFileRead ABI from chartsMetadataFormat (Linux LP64 test fixtures).

    This is the existing native format, not a portable interchange format. Host
    tests check its size against the actual C++ serializer before GPU acceptance.
    """

    _fields_ = [
        ("frame_counter", ct.c_int32), ("fpga_seq_num", ct.c_int64),
        ("timestamp_step", ct.c_int32), ("name", ct.c_char * 20),
        ("type", ct.c_int32), ("dims", ct.c_int32), ("dim", ct.c_int32 * 8),
        ("dim_name", (ct.c_char * 20) * 8), ("stride", ct.c_int64 * 8),
        ("offset", ct.c_int64), ("nfreq", ct.c_int32),
        ("coarse_freq", ct.c_int32 * 672), ("lost_timesamples", ct.c_int32),
        ("first_packet_recv_time", ct.c_long * 2), ("time0_fpga", ct.c_uint64),
    ]


def write_frame(path: Path, payload: np.ndarray, sequence: int, shard: int,
                *, mask: bool = False, timestamp_step: int = 1) -> None:
    """Write one packed voltage or original handler mask frame with provenance."""
    if payload.dtype != np.uint8:
        raise ValueError("fixture payload must be bytes")
    meta = NativeChartsMetadata()
    meta.fpga_seq_num = sequence
    meta.timestamp_step = timestamp_step
    meta.frame_counter = -1
    meta.dims = -1  # Raw provenance only; transformed shape is in the associated sidecar.
    if not mask:
        meta.nfreq = 336
        for f in range(336):
            meta.coarse_freq[f] = shard * 336 + f
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("=I", ct.sizeof(meta)) + bytes(meta) + payload.tobytes())


def read_product(path: Path, dtype: str, shape: tuple[int, ...]):
    """Read one native rawFileWrite record without interpreting provenance as shape."""
    data = path.read_bytes()
    (size,) = struct.unpack("=I", data[:4])
    if size != ct.sizeof(NativeChartsMetadata):
        raise ValueError("unexpected native chartsMetadata size")
    expected = int(np.prod(shape)) * np.dtype(dtype).itemsize
    if len(data) != 4 + size + expected:
        raise ValueError("product payload length disagrees with associated shape")
    meta = NativeChartsMetadata.from_buffer_copy(data[4:4 + size])
    return meta, np.frombuffer(data, dtype=dtype, offset=4 + size).reshape(shape)


def create_packet_loss_yaml(yaml_path: Path, input_dir: Path, output_dir: Path,
                            frames: int, *, mode: str = "fine", raw_n_time: int = 640,
                            k: int | None = None, gpu_depth: int = 3,
                            raw_depth: int = 4, mask_depth: int = 5,
                            output_depth: int = 4, num_beams: int = 64) -> Path:
    """Two shards with pre-zeroed voltage and original masks via existing cudaInputData.

    All output is float32 for numerical checks. The associated exposure descriptor
    is a validation sidecar; each serialized record carries its raw sequence ID.
    Host ring depths must cover GPU depth because existing copy constructors index
    their instance's host frame when registering memory.

    Separate file readers cannot reproduce the live handler's voltage-before-mask
    publication contract with zeroSamples. Therefore replay fixtures MUST already
    zero missing spectra. Live ingest retains the unchanged zeroSamples stage;
    its ordered host handoff is exercised in test_packet_loss.cpp.
    """
    from .pipeline import create_beamformer_yaml

    k = (10 if mode == "fine" else 320) if k is None else k
    if mode not in ("fine", "coarse") or (mode == "fine" and k != 10):
        raise ValueError("fine requires K=10; mode must be fine or coarse")
    if mode == "coarse" and k not in (10, 320):
        raise ValueError("coarse requires K=10 or 320")
    if raw_n_time <= 0 or (mode == "fine" and (raw_n_time < 96 or raw_n_time % 32)):
        raise ValueError("fine requires T_raw >= 96 divisible by 32")
    if frames < 1 or gpu_depth < 1 or min(raw_depth, mask_depth, output_depth) < gpu_depth:
        raise ValueError("positive frame count and host depths >= GPU depth are required")
    if not 1 <= num_beams <= 128:
        raise ValueError("num_beams must be between 1 and 128")
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    n_time = raw_n_time // 32 if mode == "fine" else raw_n_time
    n_integrated = (n_time + k - 1) // k
    count_dtype = "uint8" if mode == "fine" else "uint16"
    descriptor = {"schema": "charts-exposure-validation-v1", "input_mode": mode,
                  "T_raw": raw_n_time, "U": 32 if mode == "fine" else 1, "K": k,
                  "shape": [n_integrated, 336], "dtype": count_dtype,
                  "byte_order": "native", "num_elements": 64, "num_beams": num_beams,
                  "intensity_dtype": "float32",
                  "intensity_shape": [n_integrated, 10752 if mode == "fine" else 336, num_beams],
                  "weight_identity": "generated default geometry; coherent beam=num_beams//2",
                  "association": ["configured_shard", "FPGA_SEQ_NUM", "T_raw"],
                  "timestamp_step": 1, "raw_tick_seconds": [10, 3000000],
                  "mask_polarity": "1=missing, 0=valid",
                  "voltage_preparation": "replay fixtures pre-zero missing spectra",
                  "fine_bins_per_count": 32 if mode == "fine" else 1,
                  "startup_incomplete_fine_spectra": 3 if mode == "fine" else 0,
                  "power_convention": "sum-power of supplied zeroed samples; no normalization",
                  "tail_supplied_spectra": min(k, n_time - (n_integrated - 1) * k),
                  "shards": []}
    config = {}
    with tempfile.TemporaryDirectory() as tmp:
        for shard in (0, 1):
            temp_yaml = Path(tmp) / f"{shard}.yaml"
            create_beamformer_yaml(temp_yaml, input_dir, f"voltage.shard{shard}", output_dir,
                                   f"intensity.shard{shard}", frames, num_beams=num_beams,
                                   samples_per_data_set=raw_n_time, integration_spectra=k,
                                   input_mode=mode, output_format="float32", shard_id=shard,
                                   buffer_depth=gpu_depth)
            part = yaml.safe_load(temp_yaml.read_text())
            if shard == 0:
                config.update({key: value for key, value in part.items()
                               if key not in ("voltage_buf", "intensity_buf", "read_voltage",
                                              "gpu_beamformer", "write_intensity")})
                config["main_pool"]["num_metadata_objects"] = 128
                config["cpu_affinity"] = sorted(os.sched_getaffinity(0))
            raw, missing = f"voltage_{shard}", f"missing_{shard}"
            intensity, exposure = f"intensity_{shard}", f"exposure_{shard}"
            config[raw] = dict(part["voltage_buf"], num_frames=raw_depth)
            config[missing] = dict(part["voltage_buf"], num_frames=mask_depth,
                                   frame_size=raw_n_time)
            config[intensity] = dict(part["intensity_buf"], num_frames=output_depth)
            config[exposure] = dict(config[intensity], frame_size=
                                     n_integrated * 336 * np.dtype(count_dtype).itemsize)
            config[f"read_voltage_{shard}"] = dict(part["read_voltage"], buf=raw)
            config[f"read_missing_{shard}"] = dict(part["read_voltage"], buf=missing,
                                                    file_name=f"missing.shard{shard}")
            gpu = part["gpu_beamformer"]
            gpu["in_buffers"] = {"raw": raw, "missing": missing}
            gpu["out_buffers"] = {"intensity": intensity, "exposure": exposure}
            commands = gpu["commands"]
            commands.insert(1, {"name": "cudaInputData", "in_buf": "missing",
                                "gpu_mem": f"charts.missing.shard{shard}"})
            command = next(c for c in commands if c["name"] == "cudaChartsBeamformerCommand")
            command.update(in_buf_missing="missing", out_buf_exposure="exposure",
                           mask_shard_id=shard, gpu_mem_missing=f"charts.missing.shard{shard}",
                           gpu_mem_exposure=f"charts.exposure.shard{shard}")
            commands.append({"name": "cudaOutputData", "in_buf": "raw",
                             "gpu_mem": f"charts.exposure.shard{shard}", "out_buf": "exposure"})
            # Existing cudaSyncInput/Output and final-signal ownership use these roles.
            # Both D2H products finish on stream 1; the final event covers both.
            for c in commands:
                if c["name"] == "cudaInputData":
                    c["cuda_stream"] = 0
                elif c["name"] == "cudaChartsBeamformerCommand":
                    c["cuda_stream"] = 2
                elif c["name"] == "cudaOutputData":
                    c["cuda_stream"] = 1
            config[f"gpu_{shard}"] = gpu
            config[f"write_intensity_{shard}"] = dict(part["write_intensity"], in_buf=intensity)
            config[f"write_exposure_{shard}"] = dict(part["write_intensity"], in_buf=exposure,
                                                     file_name=f"exposure.shard{shard}")
            descriptor["shards"].append({"shard": shard, "coarse_origin": 336 * shard,
                                         "file_prefix": f"exposure.shard{shard}"})
    yaml_path.write_text(yaml.safe_dump(config, sort_keys=False))
    (output_dir / "exposure.v1.json").write_text(json.dumps(descriptor, indent=2) + "\n")
    return yaml_path
