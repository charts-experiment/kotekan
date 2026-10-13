"""Compact coarse replay through real Kotekan; compare existing CPU reference output.

This is numerical validation, not a CPU performance benchmark. Requires PyYAML.
Input metadata fixtures are serialized by Kotekan's C++ classes to avoid assuming ABI.
"""
import argparse
import array
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys

import yaml


def run(command, log, timeout=120):
    environment = os.environ.copy()
    # Kotekan invokes python3 for YAML/JSON loading. Reuse this driver's
    # interpreter environment, where PyYAML has already been imported.
    environment["PATH"] = str(Path(sys.executable).parent) + os.pathsep + environment.get("PATH", "")
    with log.open("w") as stream:
        result = subprocess.run([str(x) for x in command], stdout=stream, stderr=subprocess.STDOUT,
                                timeout=timeout, env=environment)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}); see {log}")


def read_product(path):
    data = path.read_bytes()
    if len(data) < 4:
        raise ValueError(f"Truncated product: {path}")
    length = struct.unpack_from("<I", data)[0]
    if not 0 < length <= 4 * 1024 * 1024 or 4 + length > len(data):
        raise ValueError("Invalid product metadata byte range")
    metadata = json.loads(data[4:4 + length])
    if metadata["schema"] != "charts.beam.intensity.v1" or metadata["datatype"] != "float32_le":
        raise ValueError("Unsupported product")
    payload = data[4 + length:]
    if len(payload) != metadata["payload_bytes"] or len(payload) != math.prod(metadata["shape"]) * 4:
        raise ValueError("Payload/shape mismatch")
    result = array.array("f")
    result.frombytes(payload)
    if sys.byteorder != "little":
        result.byteswap()
    return metadata, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kotekan", type=Path, required=True)
    parser.add_argument("--fixture-generator", type=Path, required=True,
                        help="test_charts_beamformer executable")
    parser.add_argument("--cpu-reference", type=Path, required=True,
                        help="existing beamform_project beamformer_cpu executable")
    parser.add_argument("--output-dir", type=Path, required=True, help="new, empty validation directory")
    parser.add_argument("--template", type=Path,
                        default=Path(__file__).resolve().parents[1] / "config/64antennas_beamform_replay.yaml")
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    run([args.fixture_generator, "--run_test=emit_replay_fixture", "--", "--fixture-dir", root],
        root / "fixtures.log")
    config = yaml.safe_load(args.template.read_text())
    config.update(samples_per_data_set=11, num_elements=64, num_beams=4,
                  integration_spectra=10, replay_frames=6)
    for shard in range(2):
        for kind in ("voltage", "missing"):
            config[f"read_{kind}_{shard}"]["base_dir"] = str(root)
        config[f"write_{shard}"]["base_dir"] = str(root)
        command = config[f"gpu_{shard}"]["commands"][2]
        command["weights_file"] = str(root / f"weights.shard{shard}.bin")
        command["weights_manifest"] = str(root / f"weights.shard{shard}.json")
    config_file = root / "pipeline.json"
    config_file.write_text(json.dumps(config, indent=2))
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    run([args.kotekan, "-c", config_file, "-b", f"127.0.0.1:{port}"], root / "kotekan.log")
    records = []
    for shard in range(2):
        for frame in range(6):
            suffix = f".shard{shard}_{frame:07d}.bin"
            reference = root / ("reference" + suffix)
            run([args.cpu_reference, "--input-mode", "raw", "--shard-id", shard,
                 "--n-time", 11, "--n-ant", 64, "--n-beams", 4, "--integration-spectra", 10,
                 "--input", root / ("packed" + suffix), "--weights", root / f"weights.shard{shard}.bin",
                 "--output", reference], root / f"reference.shard{shard}_{frame:07d}.log")
            metadata, actual = read_product(root / ("intensity" + suffix))
            expected = array.array("f")
            expected.frombytes(reference.read_bytes())
            if sys.byteorder != "little":
                expected.byteswap()
            assert len(actual) == len(expected) == 2 * 336 * 4
            maximum_error = 0.0
            for a, e in zip(actual, expected):
                assert math.isfinite(a) and math.isfinite(e)
                error = abs(a - e)
                assert error <= 1e-3 + 1e-5 * abs(e), (shard, frame, a, e)
                maximum_error = max(maximum_error, error)
            assert metadata["shape"] == [2, 336, 4]
            assert metadata["shard_id"] == shard and metadata["coarse_origin"] == shard * 336
            assert metadata["frequency_first_hz"] == 300000000 + shard * 100800000
            assert metadata["sequence_start"] == 15360 + frame * 11
            assert metadata["raw_frame_id"] == frame and metadata["sequence_gap_ticks"] == 0
            assert metadata["supplied_spectra"] == [10, 1]
            assert metadata["valid_spectra"] == ([9, 1] if shard == 0 else [9, 0])
            assert metadata["tick_period_us"] == [10, 3] and metadata["timestamp_step_ticks"] == 10
            manifest = json.loads((root / f"weights.shard{shard}.json").read_text())
            assert metadata["weights_id"] == manifest["digest"]
            records.append(dict(shard=shard, frame=frame, max_absolute_error=maximum_error,
                                valid_spectra=metadata["valid_spectra"],
                                product_sha256=hashlib.sha256((root / ("intensity" + suffix)).read_bytes()).hexdigest()))
    report = dict(scope="Phase 2 compact coarse numerical replay; no sustained cadence claim",
                  frames_per_shard=6, gpu_buffer_depth=3, records=records,
                  template_sha256=hashlib.sha256(args.template.read_bytes()).hexdigest())
    (root / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"PASS: {len(records)} independent-shard frames, CPU/GPU parity, metadata and ring reuse")
    print(f"Evidence: {root / 'validation.json'}")


if __name__ == "__main__":
    main()
