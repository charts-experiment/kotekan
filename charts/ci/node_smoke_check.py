#!/usr/bin/env python3
"""Helper for charts/ci/node_smoke_test.sh.

Subcommands:

- ``prepare``: write a copy of a config with every output path pointing to one
  directory, the antenna lists replaced and, optionally, the calibration table
  replaced. Prints ``key=value`` lines the shell script reads.
- ``metrics``: check 4, DPDK packet counters between two ``/metrics`` samples.
- ``trackers``: check 9, GPU command utilisation from a ``/trackers`` dump.
- ``files``: check 6, cadence of the ``rawFileWrite`` outputs.
- ``data``: checks 7 and 8 with charts-utils, skipped if it is not importable.

Check subcommands print one ``<check>\\t<PASS|FAIL|WARN|SKIP>\\t<detail>`` line
each. Only ``prepare`` needs pyyaml; ``data`` needs charts-utils.
"""

import argparse
import ast
import json
import operator
import re
import statistics
import sys
from pathlib import Path

OUTPUT_PATH_KEYS = ("output_dir", "base_dir", "root_path")
ANTENNA_KEYS = ("signal_antennas", "selected_antennas")
# Lost, and rejected (rx_error_total), packets over received plus lost, between the
# start and end samples. Production sees a steady trickle of rejected packets.
MAX_BAD_FRACTION = 1e-3
# Sum of the mean utilisation of the commands of one cudaProcess stage.
MAX_GPU_UTILISATION = 0.8
# Allowed deviation of the median file period from the configured one.
CADENCE_TOLERANCE = 0.1

_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def report(check, status, detail):
    print(f"{check}\t{status}\t{detail}")


def walk(node, seen=None):
    """Yield (parent, key) for every mapping entry of a parsed config.

    A yaml alias loads as the same object as its anchor, so each container is
    visited once.
    """
    seen = set() if seen is None else seen
    if not isinstance(node, (dict, list)) or id(node) in seen:
        return
    seen.add(id(node))
    if isinstance(node, dict):
        for key, value in node.items():
            yield node, key
            yield from walk(value, seen)
    else:
        for value in node:
            yield from walk(value, seen)


def evaluate(cfg, value, depth=0):
    """Evaluate a top-level kotekan config expression such as ``a / 300000.0``."""
    if isinstance(value, (int, float)):
        return value
    if depth > 10:
        raise ValueError(f"config expression nests too deep: {value!r}")

    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.Name):
            return evaluate(cfg, cfg[node.id], depth + 1)
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
            return _BINARY_OPS[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -visit(node.operand)
        raise ValueError(f"unsupported config expression: {value!r}")

    return visit(ast.parse(str(value), mode="eval"))


def parse_antennas(text):
    """Return a list of antennas from ``0,4,32`` or from the first ``signal_antennas`` of a yaml."""
    if Path(text).is_file():
        import yaml

        with open(text) as stream:
            cfg = yaml.safe_load(stream)
        for parent, key in walk(cfg):
            if key == "signal_antennas":
                return [int(a) for a in parent[key]]
        raise ValueError(f"no signal_antennas in {text}")
    return [int(a) for a in text.split(",") if a.strip()]


def cmd_prepare(args):
    import yaml

    with open(args.config) as stream:
        cfg = yaml.safe_load(stream)

    antennas = parse_antennas(args.antennas) if args.antennas else None
    selected = [p[key] for p, key in walk(cfg) if key == "selected_antennas"]
    if (
        selected
        and "num_saved_antennas" in cfg
        and len(selected[0]) != cfg["num_saved_antennas"]
    ):
        print(
            f"warning: {args.config}: num_saved_antennas is {cfg['num_saved_antennas']} but "
            f"selected_antennas has {len(selected[0])} entries; BasebandWriter refuses this",
            file=sys.stderr,
        )

    handlers = 0
    file_prefixes = []
    baseband_endpoint = ""
    for parent, key in list(walk(cfg)):
        value = parent[key]
        if key in OUTPUT_PATH_KEYS and isinstance(value, str):
            parent[key] = args.output_dir
        elif key in ANTENNA_KEYS and antennas is not None:
            parent[key] = list(antennas)
        elif key == "calibration_file" and args.calibration_file:
            if args.calibration_file == "none":
                del parent[key]
            else:
                parent[key] = args.calibration_file
        elif key == "dpdk_handler" and value == "rfsocHandlerShuffle":
            handlers += 1
        elif key == "kotekan_stage" and value == "rawFileWrite":
            file_prefixes.append(parent["file_name"])
        elif key == "kotekan_stage" and value == "chartsTriggeredBasebandReadout":
            baseband_endpoint = parent.get("endpoint", "/charts/baseband")
    if antennas is not None and "num_saved_antennas" in cfg and selected:
        cfg["num_saved_antennas"] = len(antennas)

    missing = [
        parent[key]
        for parent, key in walk(cfg)
        if key == "calibration_file" and not Path(parent[key]).is_file()
    ]
    if missing:
        print(
            f"error: calibration_file not found: {', '.join(sorted(set(missing)))}",
            file=sys.stderr,
        )
        return 1

    cadence = 0.0
    if "integration_frames" in cfg and "frame_arrival_period" in cfg:
        cadence = evaluate(cfg, cfg["integration_frames"]) * evaluate(
            cfg, cfg["frame_arrival_period"]
        )

    with open(args.output, "w") as stream:
        yaml.safe_dump(cfg, stream, sort_keys=False, default_flow_style=None)

    print(f"handlers={handlers}")
    print(f"file_prefixes={','.join(file_prefixes)}")
    print(f"cadence={cadence:.6f}")
    print(
        f"n_freq={evaluate(cfg, cfg.get('num_local_freq', 0)) * max(len(file_prefixes), 1)}"
    )
    print(f"baseband_endpoint={baseband_endpoint}")
    print(f"antennas={','.join(str(a) for a in antennas) if antennas else ''}")
    return 0


def read_metrics(path):
    """Return {(metric, port): (value, timestamp_ms)} for the kotekan_dpdk_rx_* gauges."""
    line_re = re.compile(
        r'^(kotekan_dpdk_rx_\w+)\{[^}]*port="(\d+)"[^}]*\}\s+(\S+)(?:\s+(\d+))?'
    )
    samples = {}
    with open(path) as stream:
        for line in stream:
            match = line_re.match(line)
            if match:
                name, port, value, stamp = match.groups()
                samples[(name, int(port))] = (float(value), int(stamp or 0))
    return samples


def counter_delta(start, end, name, port):
    """Return the increase of a gauge and the seconds between its two samples."""
    end_value, end_ms = end.get((name, port), (0.0, 0))
    start_value, start_ms = start.get((name, port), (0.0, end_ms))
    return end_value - start_value, (end_ms - start_ms) / 1e3


def cmd_metrics(args):
    start, end = read_metrics(args.start), read_metrics(args.end)
    ports = sorted(
        port for name, port in end if name == "kotekan_dpdk_rx_packets_total"
    )
    if len(ports) < args.ports:
        report(
            4,
            "FAIL",
            f"rx_packets_total found for ports {ports}, expected {args.ports}",
        )
        return 0

    failed = False
    details = []
    for port in ports:
        packets, seconds = counter_delta(
            start, end, "kotekan_dpdk_rx_packets_total", port
        )
        lost, _ = counter_delta(start, end, "kotekan_dpdk_rx_lost_packets_total", port)
        errors, _ = counter_delta(start, end, "kotekan_dpdk_rx_error_total", port)
        if seconds <= 0:
            details.append(
                f"port {port}: start and end samples are from the same update"
            )
            failed = True
            continue
        total = packets + lost
        lost_fraction = lost / total if total > 0 else 0.0
        error_fraction = errors / total if total > 0 else 0.0
        details.append(
            f"port {port}: {packets:.0f} pkts ({packets / seconds:.0f}/s), "
            f"lost {lost:.0f} ({lost_fraction:.1e}), errors {errors:.0f} "
            f"({errors / seconds:.2f}/s, {error_fraction:.1e})"
        )
        if packets <= 0 or max(lost_fraction, error_fraction) > MAX_BAD_FRACTION:
            failed = True
    report(4, "FAIL" if failed else "PASS", "; ".join(details))
    return 0


def cmd_zero_ports(args):
    """Print the ports whose rx_packets_total is zero, for the check 2 message."""
    samples = read_metrics(args.sample)
    print(
        ",".join(
            str(port)
            for (name, port), (value, _) in sorted(samples.items())
            if name == "kotekan_dpdk_rx_packets_total" and value == 0
        )
    )
    return 0


def cmd_trackers(args):
    with open(args.trackers) as stream:
        trackers = json.load(stream) or {}
    per_stage = {}
    worst = 0.0
    for stage, stage_trackers in trackers.items():
        if "/commands/" not in stage:
            continue
        gpu_stage = stage.split("/commands/")[0]
        for name, tracker in stage_trackers.items():
            if name.endswith("_u") and tracker.get("samples"):
                per_stage[gpu_stage] = per_stage.get(gpu_stage, 0.0) + tracker["avg"]
                worst = max(worst, tracker["max"])
    if not per_stage:
        report(9, "SKIP", "no GPU command trackers (is profiling enabled?)")
        return 0
    detail = ", ".join(f"{stage} {u:.2f}" for stage, u in sorted(per_stage.items()))
    failed = max(per_stage.values()) >= MAX_GPU_UTILISATION or worst >= 1.0
    report(
        9,
        "FAIL" if failed else "PASS",
        f"summed mean utilisation per stage: {detail}; worst single command {worst:.2f}",
    )
    return 0


def cmd_files(args):
    if not args.prefixes:
        report(6, "SKIP", "config has no rawFileWrite stage")
        return 0
    details = []
    failed = False
    counts = []
    for prefix in args.prefixes.split(","):
        files = sorted(
            Path(args.dir).glob(f"*{prefix}_*"), key=lambda p: p.stat().st_mtime
        )
        files = [f for f in files if f.stat().st_size > 0]
        counts.append(len(files))
        if len(files) < (2 if args.cadence > 0 else 1):
            details.append(f"{prefix}: {len(files)} files")
            failed = True
            continue
        if len(files) == 1:
            details.append(f"{prefix}: 1 file")
            continue
        mtimes = [f.stat().st_mtime for f in files]
        period = statistics.median(b - a for a, b in zip(mtimes, mtimes[1:]))
        if (
            args.cadence > 0
            and abs(period - args.cadence) > CADENCE_TOLERANCE * args.cadence
        ):
            failed = True
        details.append(f"{prefix}: {len(files)} files, median period {period:.2f} s")
    if max(counts) - min(counts) > 1:
        failed = True
    report(
        6,
        "FAIL" if failed else "PASS",
        "; ".join(details) + f" (expected {args.cadence:.2f} s)",
    )
    return 0


def antenna_levels(group, antennas, n_freq):
    """Return the median autocorrelation of each antenna and the median coherence of each pair."""
    import numpy as np
    from charts_utils.core.io import extract_gpu_visibility

    autos = {a: extract_gpu_visibility(group, a, a, n_freq).real for a in antennas}
    levels = {a: float(np.median(autos[a])) for a in antennas}
    coherence = {}
    for i, a in enumerate(antennas):
        for b in antennas[i + 1 :]:
            cross = extract_gpu_visibility(group, a, b, n_freq)
            norm = np.sqrt(np.abs(autos[a] * autos[b]))
            valid = norm > 0
            coherence[(a, b)] = (
                float(np.median(np.abs(cross[valid]) / norm[valid]))
                if valid.any()
                else 0.0
            )
    return levels, coherence


def last_full_group(groups):
    """Return the newest frame group whose files are as large as the first group's."""
    sizes = [Path(p).stat().st_size for p in groups[0].paths]
    for group in reversed(groups):
        if [Path(p).stat().st_size for p in group.paths] == sizes:
            return group
    return groups[0]


def cmd_data(args):
    try:
        import numpy as np
        from charts_utils.core.gpu_frames import (
            discover_gpu_corr_frame_groups,
            read_gpu_corr_frame_header,
        )
    except ImportError as error:
        report(7, "SKIP", f"charts-utils not importable ({error})")
        report(8, "SKIP", f"charts-utils not importable ({error})")
        return 0

    try:
        # Raises if a frame index lacks a shard or the shard sequence numbers differ.
        groups = discover_gpu_corr_frame_groups(args.dir, allow_incomplete_last=True)
        offsets = []
        for group in groups:
            times = [read_gpu_corr_frame_header(p).time0_fpga for p in group.paths]
            offsets.append(max(times) - min(times))
        report(
            7,
            "PASS",
            f"{len(groups)} frame groups, fpga_seq_num equal across shards, "
            f"max time0 offset {max(offsets)} us",
        )
    except (OSError, ValueError) as error:
        report(7, "FAIL", str(error))
        report(8, "SKIP", "no aligned frame groups")
        return 0

    antennas = [int(a) for a in args.antennas.split(",")]
    try:
        levels, coherence = antenna_levels(
            last_full_group(groups), antennas, args.n_freq
        )
    except (OSError, ValueError) as error:
        report(8, "FAIL", str(error))
        return 0
    detail = "autos " + ", ".join(f"{a}: {v:.3g}" for a, v in levels.items())
    detail += "; coherence " + ", ".join(
        f"{a}-{b}: {c:.3f}" for (a, b), c in coherence.items()
    )
    failed = any(not np.isfinite(v) or v <= 0 for v in levels.values())
    status = "FAIL" if failed else "PASS"

    if args.reference and not failed:
        try:
            reference = discover_gpu_corr_frame_groups(
                args.reference, allow_incomplete_last=True
            )
            ref_levels, _ = antenna_levels(
                last_full_group(reference), antennas, args.n_freq
            )
            ratios = {
                a: levels[a] / ref_levels[a] for a in antennas if ref_levels[a] > 0
            }
            detail += "; ratio to reference " + ", ".join(
                f"{a}: {r:.2f}" for a, r in ratios.items()
            )
            if any(not 0.5 <= r <= 2.0 for r in ratios.values()):
                status = "WARN"
        except (OSError, ValueError) as error:
            detail += f"; reference not usable ({error})"
            status = "WARN"
    report(8, status, detail)
    return 0


def main(argv):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare")
    p.add_argument("config")
    p.add_argument("output", help="path of the config copy to write")
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--antennas",
        help="comma-separated list, or a yaml to take signal_antennas from",
    )
    p.add_argument(
        "--calibration-file", help="replacement calibration table, or 'none' to drop it"
    )
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("metrics")
    p.add_argument("start")
    p.add_argument("end")
    p.add_argument("--ports", type=int, default=2)
    p.set_defaults(func=cmd_metrics)

    p = sub.add_parser("zero-ports")
    p.add_argument("sample")
    p.set_defaults(func=cmd_zero_ports)

    p = sub.add_parser("trackers")
    p.add_argument("trackers")
    p.set_defaults(func=cmd_trackers)

    p = sub.add_parser("files")
    p.add_argument("dir")
    p.add_argument("--prefixes", default="")
    p.add_argument("--cadence", type=float, default=0.0)
    p.set_defaults(func=cmd_files)

    p = sub.add_parser("data")
    p.add_argument("dir")
    p.add_argument("--antennas", required=True)
    p.add_argument("--n-freq", type=int, required=True)
    p.add_argument(
        "--reference", help="production output directory to compare levels against"
    )
    p.set_defaults(func=cmd_data)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
