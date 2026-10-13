"""Offline audit of the two CHARTS YAMLs; does not start Kotekan or CUDA.

Requires PyYAML (available with /usr/bin/python3 on the audited machine).
Only arithmetic used by these YAMLs is supported; this is not a replacement Config.
"""
import argparse
import ast
import hashlib
import json
import operator
from pathlib import Path

import yaml


def duplicates(node, path="", ancestors=()):
    result = []
    if id(node) in ancestors:
        return result
    ancestors = (*ancestors, id(node))
    if isinstance(node, yaml.MappingNode):
        seen = set()
        for key, value in node.value:
            if key.value in seen:
                result.append(f"{path}/{key.value} (line {key.start_mark.line + 1})")
            seen.add(key.value)
            result.extend(duplicates(value, f"{path}/{key.value}", ancestors))
    elif isinstance(node, yaml.SequenceNode):
        for i, value in enumerate(node.value):
            result.extend(duplicates(value, f"{path}/{i}", ancestors))
    return result


def number(config, scopes, key, default=None, stack=()):
    value = next((scope[key] for scope in (*scopes, config) if key in scope), default)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    if not isinstance(value, str) or key in stack:
        raise ValueError(f"Missing or recursive numeric field: {key}")
    ops = {ast.Add: operator.add, ast.Sub: operator.sub,
           ast.Mult: operator.mul, ast.Div: operator.truediv}

    def evaluate(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.Name):
            return number(config, scopes, node.id, stack=(*stack, key))
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](evaluate(node.left), evaluate(node.right))
        raise ValueError(f"Unsupported arithmetic in {key}")
    return evaluate(ast.parse(value, mode="eval").body)


def audit(config):
    errors, shards = [], []

    def require(condition, message):
        if not condition:
            errors.append(message)

    handlers = config["dpdk"]["handlers"]
    require(len(handlers) == 2, "Expected two independent NIC shards")
    for shard, handler in enumerate(handlers):
        raw = config[handler["out_buffer"]]
        mask = config[handler["mask_buf"]]
        scopes = (handler, config["dpdk"])
        n = lambda key, default=None: number(config, scopes, key, default)
        elements, freq = n("num_elements"), n("num_local_freq")
        raw_bytes = number(config, (raw,), "frame_size")
        samples = raw_bytes / (elements * freq)
        require(handler["dpdk_handler"] == "rfsocHandlerShuffle", "Unexpected handler")
        require(elements == 64 and freq == 336, f"shard {shard}: raw E/F mismatch")
        require(n("num_rfsocs", 2) == 2 and n("num_elements_per_rfsoc", 32) == 32,
                f"shard {shard}: RFSoC element order mismatch")
        require(n("n_channels_per_packet") * n("subbands_per_nic", 2) == freq,
                f"shard {shard}: packet frequency coverage mismatch")
        require(n("first_subband") == shard * 2 and n("coarse_freq_start") == shard * 336,
                f"shard {shard}: frequency partition mismatch")
        require(samples.is_integer() and samples > 0 and samples % 32 == 0,
                f"shard {shard}: invalid fine-compatible frame length")
        require(number(config, (mask,), "frame_size") == samples,
                f"shard {shard}: mask length mismatch")
        require(raw.get("metadata_pool") == "main_pool" and mask.get("metadata_pool") == "main_pool"
                and config["main_pool"]["kotekan_metadata_pool"] == "chartsMetadata",
                f"shard {shard}: identity metadata pool mismatch")
        require(n("alignment", 0) > 0, f"shard {shard}: missing startup alignment")
        stages = [v for v in config.values() if isinstance(v, dict)
                  and v.get("kotekan_stage") == "zeroSamples" and v.get("out_buf") == handler["out_buffer"]]
        require(len(stages) == 1, f"shard {shard}: expected one zeroSamples producer")
        if stages:
            stage = stages[0]
            require(stage["lost_samples_buf"] == handler["mask_buf"], f"shard {shard}: wrong zero mask")
            require(number(config, (stage,), "zero_value", 0x88) == 0,
                    f"shard {shard}: signed int4 zero must be 0x00")
            require(number(config, (stage,), "sample_size", 2048) == freq * elements,
                    f"shard {shard}: zero-fill stride mismatch")
        require(samples == number(config, (), "samples_per_data_set"),
                f"shard {shard}: frame length disagrees with consumers")
        fine_time = int(samples) // 32
        shards.append({"shard": shard, "raw_spectra": int(samples), "raw_bytes": int(raw_bytes),
                       "cadence_ms": samples / 300, "alignment": n("alignment"),
                       "host_depth": number(config, (raw,), "num_frames"),
                       "fine_spectra": fine_time, "fine_output_rows_K10": (fine_time + 9) // 10,
                       "fine_output_bytes_K10": ((fine_time + 9) // 10) * 10752 * 64 * 4})
    return {"raw_contract_errors": errors, "shards": shards}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    args = parser.parse_args()
    results = {}
    for name in ("64antennas_record.yaml", "64antennas_corr.yaml"):
        path = args.checkout / "charts/config" / name
        data = path.read_bytes()
        config = yaml.safe_load(data)
        result = audit(config)
        result["yaml_duplicate_keys"] = duplicates(yaml.compose(data))
        result["sha256"] = hashlib.sha256(data).hexdigest()
        results[name] = result
    print(json.dumps({"scope": "static YAML raw-contract audit; not runtime validation",
                      "results": results}, indent=2))
    # Duplicate keys are configuration failures even if PyYAML keeps the last value.
    return int(any(r["raw_contract_errors"] or r["yaml_duplicate_keys"] for r in results.values()))


if __name__ == "__main__":
    raise SystemExit(main())
