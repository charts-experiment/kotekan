#!/usr/bin/env python3
"""Static checks of the CHARTS configs in charts/config/ against the source tree.

Complements ``kotekan --check-config`` (see kotekan/kotekan.cpp), which needs a
built binary and checks stage types, duplicate buffers and metadata pools, but
not what is looked up by name at construction time. This script needs no build:

- every ``kotekan_stage`` is registered with REGISTER_KOTEKAN_STAGE in lib/;
- every command of a ``cudaProcess`` stage is registered with
  REGISTER_CUDA_COMMAND[_WITH_STATE] in lib/;
- every ``dpdk_handler`` of a ``dpdkCore`` stage is one dpdkCore.cpp dispatches;
- the ``coarse_freq_start``/``num_local_freq`` shards of a dpdkCore stage do not
  overlap and stay inside the full band (charts_full_band_channels in
  lib/core/chartsConstants.hpp);
- every ``freq_offset`` of a GPU command is the start of one of those shards.

Exits non-zero and prints one line per problem. ``--self-test`` checks that the
validator catches the faults in SELF_TEST_CASES.

Usage: charts/ci/validate_charts_configs.py [config.yaml ...]
"""

import re
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO / "charts" / "config"

STAGE_RE = re.compile(r"REGISTER_KOTEKAN_STAGE\(\s*(\w+)")
CUDA_RE = re.compile(r"REGISTER_CUDA_COMMAND(?:_WITH_STATE)?\(\s*(\w+)")
HANDLER_RE = re.compile(r'handler_name\s*==\s*"(\w+)"')
FULL_BAND_RE = re.compile(r"charts_full_band_channels\s*=\s*(\d+)")


def scan_source(repo=REPO):
    """Collect the names the binary can build from the source tree."""
    stages, cuda = set(), set()
    for path in (repo / "lib").rglob("*"):
        if path.suffix not in (".cpp", ".hpp", ".cu", ".h"):
            continue
        text = path.read_text(errors="ignore")
        stages.update(STAGE_RE.findall(text))
        cuda.update(CUDA_RE.findall(text))
    dpdk = set(HANDLER_RE.findall((repo / "lib/dpdk/dpdkCore.cpp").read_text()))
    full_band = int(FULL_BAND_RE.search((repo / "lib/core/chartsConstants.hpp").read_text()).group(1))
    return {"stages": stages, "cuda": cuda, "dpdk": dpdk, "full_band": full_band}


def stage_blocks(node, path=""):
    """Yield (path, block) for every dict with a kotekan_stage key."""
    if isinstance(node, dict):
        if "kotekan_stage" in node:
            yield path or "/", node
        for key, value in node.items():
            yield from stage_blocks(value, f"{path}/{key}")


def check_config(cfg, src):
    """Return the list of problems in one parsed config."""
    problems = []
    for path, block in stage_blocks(cfg):
        stage = block["kotekan_stage"]
        if stage not in src["stages"]:
            problems.append(f"{path}: unknown stage '{stage}'")

        if stage == "cudaProcess":
            for i, cmd in enumerate(block.get("commands") or []):
                name = cmd.get("name") if isinstance(cmd, dict) else None
                if name not in src["cuda"]:
                    problems.append(f"{path}/commands[{i}]: unknown GPU command '{name}'")

        if stage == "dpdkCore":
            shards = []
            for i, handler in enumerate(block.get("handlers") or []):
                name = handler.get("dpdk_handler")
                if name not in src["dpdk"]:
                    problems.append(f"{path}/handlers[{i}]: unknown dpdk_handler '{name}'")
                start, n = handler.get("coarse_freq_start"), handler.get("num_local_freq")
                if isinstance(start, int) and isinstance(n, int):
                    shards.append((start, n, i))
            shards.sort()
            for start, n, i in shards:
                if start < 0 or start + n > src["full_band"]:
                    problems.append(f"{path}/handlers[{i}]: channels {start}..{start + n - 1} "
                                    f"outside the {src['full_band']}-channel band")
            for (s0, n0, i0), (s1, _, i1) in zip(shards, shards[1:]):
                if s1 < s0 + n0:
                    problems.append(f"{path}/handlers[{i0}] and handlers[{i1}]: overlapping "
                                    f"frequency shards ({s0}+{n0} and {s1})")
            starts = {s for s, _, _ in shards}
            if starts:
                for gpath, gblock in stage_blocks(cfg):
                    if gblock["kotekan_stage"] != "cudaProcess":
                        continue
                    for i, cmd in enumerate(gblock.get("commands") or []):
                        off = cmd.get("freq_offset") if isinstance(cmd, dict) else None
                        if isinstance(off, int) and off not in starts:
                            problems.append(f"{gpath}/commands[{i}]: freq_offset {off} is not the "
                                            f"start of a shard {sorted(starts)}")
    return problems


def check_file(path, src):
    try:
        cfg = yaml.safe_load(Path(path).read_text())
    except yaml.YAMLError as ex:
        return [f"invalid YAML: {ex}"]
    if not isinstance(cfg, dict):
        return ["config is not a YAML mapping"]
    return check_config(cfg, src)
    "cuda": {
        "cudaInputData",
        "cudaCorrelatorAstron",
        "cudaAntennaMaskCommand",
        "cudaDirectBeamTrackerCommand",
    },

SELF_TEST_SRC = {
    "stages": {"dpdkCore", "cudaProcess", "rawFileWrite"},
    "cuda": {"cudaInputData", "cudaCorrelatorAstron"},
    "dpdk": {"rfsocHandlerShuffle"},
    "full_band": 672,
}


def _cfg(handlers=None, commands=None, extra=None):
    cfg = {
        "dpdk": {"kotekan_stage": "dpdkCore", "handlers": handlers if handlers is not None else [
            {"name": "cudaInputData"},
            {"name": "cudaAntennaMaskCommand"},
            {"name": "cudaDirectBeamTrackerCommand"},
            {"name": "cudaCorrelatorAstron", "freq_offset": 336},
            {"dpdk_handler": "rfsocHandlerShuffle", "coarse_freq_start": 336, "num_local_freq": 336},
        ]},
        "gpu": {"kotekan_stage": "cudaProcess", "commands": commands if commands is not None else [
            {"name": "cudaInputData"}, {"name": "cudaCorrelatorAstron", "freq_offset": 336},
        ]},
    }
    cfg.update(extra or {})
    return cfg


# (description, config, substring every expected problem must contain; None = no problems)
SELF_TEST_CASES = [
    ("valid config", _cfg(), None),
    ("unknown stage", _cfg(extra={"w": {"kotekan_stage": "noSuchStage"}}), "unknown stage 'noSuchStage'"),
    ("unknown GPU command", _cfg(commands=[{"name": "cudaFoo"}]), "unknown GPU command 'cudaFoo'"),
    ("unknown dpdk handler", _cfg(handlers=[{"dpdk_handler": "rfsocHandlerFoo"}]),
     "unknown dpdk_handler 'rfsocHandlerFoo'"),
    ("overlapping shards", _cfg(handlers=[
        {"dpdk_handler": "rfsocHandlerShuffle", "coarse_freq_start": 0, "num_local_freq": 336},
        {"dpdk_handler": "rfsocHandlerShuffle", "coarse_freq_start": 300, "num_local_freq": 336},
    ], commands=[]), "overlapping"),
    ("shard outside band", _cfg(handlers=[
        {"dpdk_handler": "rfsocHandlerShuffle", "coarse_freq_start": 400, "num_local_freq": 336},
    ], commands=[]), "outside the 672-channel band"),
    ("bad freq_offset", _cfg(commands=[{"name": "cudaCorrelatorAstron", "freq_offset": 100}]),
     "freq_offset 100"),
]


def self_test():
    failed = 0
    for desc, cfg, expect in SELF_TEST_CASES:
        problems = check_config(cfg, SELF_TEST_SRC)
        ok = not problems if expect is None else bool(problems) and all(expect in p for p in problems)
        if not ok:
            failed += 1
            print(f"self-test FAILED: {desc}: {problems}")
    print(f"self-test: {len(SELF_TEST_CASES) - failed}/{len(SELF_TEST_CASES)} passed")
    return 1 if failed else 0


def main(argv):
    if argv[1:] == ["--self-test"]:
        return self_test()
    files = [Path(p).resolve() for p in argv[1:]] or sorted(CONFIG_DIR.glob("*.yaml"))
    src = scan_source()
    total = 0
    for path in files:
        problems = check_file(path, src)
        total += len(problems)
        for problem in problems:
            shown = path.relative_to(REPO) if path.resolve().is_relative_to(REPO) else path
            print(f"{shown}: {problem}")
    print(f"checked {len(files)} configs: {total} problem(s)")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
