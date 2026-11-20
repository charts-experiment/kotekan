#!/usr/bin/env python3
"""Preflight static, --check-config, and --dry-run auditor for all 14 v2
commissioning & chromaticity configs on ms-x01."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

import numpy as np
import yaml

KOTEKAN_BIN = "/home/fpalma/kotekan/build/kotekan/kotekan"
MANIFEST_PATH = Path("/data/commissioning_20261119_v2/schedule_manifest.json")

FORBIDDEN_STRESS_STAGES = (
    "cudaTransientTrigger",
    "cudaTransientTriggerCommand",
    "chartsTriggeredBasebandReadout",
    "BasebandWriter",
    "chartsBasebandReadout",
)


def audit_yaml_safety(cfg_path: Path) -> List[str]:
    errs: List[str] = []
    raw_text = cfg_path.read_text(encoding="utf-8")
    cfg = yaml.safe_load(raw_text)

    for bad in FORBIDDEN_STRESS_STAGES:
        if bad in raw_text:
            errs.append(f"{cfg_path.name}: FORBIDDEN trigger/baseband string '{bad}' found")

    pcie_block = cfg.get("dpdk", {}).get("pcie_block_list", [])
    if "0000:70:00.0" not in pcie_block:
        errs.append(f"{cfg_path.name}: dpdk.pcie_block_list missing 0000:70:00.0 (got {pcie_block})")

    gains_paths = set()
    for sb in ("commands_0", "commands_1"):
        for cmd in cfg.get("gpu", {}).get(sb, []):
            if cmd.get("name") == "cudaDirectBeamTrackerCommand":
                gf = cmd.get("gains_file", "")
                gains_paths.add(gf)
                if cmd.get("active_antennas") != [0, 4, 32, 36]:
                    errs.append(f"{cfg_path.name}: {sb} active_antennas != [0,4,32,36]")
                if cmd.get("antenna_order") != "descending":
                    errs.append(f"{cfg_path.name}: {sb} antenna_order != descending")

    for gf in gains_paths:
        p = Path(gf)
        if not p.exists():
            errs.append(f"{cfg_path.name}: missing gains_file {gf}")
        else:
            arr = np.load(p)
            if arr.shape != (672, 64) or not np.all(np.isfinite(arr)):
                errs.append(f"{cfg_path.name}: invalid gains_file shape/finiteness {arr.shape}")

    has_raw_write = False
    for k, v in cfg.items():
        if isinstance(v, dict) and v.get("kotekan_stage") == "rawFileWrite":
            has_raw_write = True
            if int(v.get("exit_after_n_files", 0)) <= 0:
                errs.append(f"{cfg_path.name}: {k}.exit_after_n_files must be > 0")
            in_buf = v.get("in_buf", "")
            if "formed_beams" in in_buf or "network_capture" in in_buf or "alive_voltages" in in_buf:
                errs.append(f"{cfg_path.name}: {k} writes forbidden high-rate voltage buffer {in_buf}")
    if not has_raw_write:
        errs.append(f"{cfg_path.name}: missing rawFileWrite stage")

    if "test01_muos" in cfg_path.name:
        for sb in ("commands_0", "commands_1"):
            for cmd in cfg.get("gpu", {}).get(sb, []):
                if cmd.get("name") == "cudaRfiMitigationCommand":
                    if cmd.get("blank_voltages_enabled") is not False:
                        errs.append(f"{cfg_path.name}: {sb} blank_voltages_enabled must be false in Test 01")

    for k, v in cfg.items():
        if isinstance(v, dict) and v.get("kotekan_stage") == "chartsAccumulate":
            if "beam_power" in v.get("in_buf", ""):
                if v.get("data_type") != "float32" or v.get("average") is not True:
                    errs.append(f"{cfg_path.name}: {k} must set data_type=float32 and average=true")

    return errs


def run_kotekan_validation(cfg_path: Path) -> List[str]:
    errs: List[str] = []
    for flag in ("--check-config", "--dry-run"):
        proc = subprocess.run(
            [KOTEKAN_BIN, flag, "--config", str(cfg_path)],
            cwd="/home/fpalma/kotekan",
            capture_output=True,
            text=True,
            timeout=60,
        )
        if proc.returncode != 0:
            tail = "\n".join((proc.stdout + "\n" + proc.stderr).splitlines()[-25:])
            errs.append(f"{cfg_path.name} {flag} failed (rc={proc.returncode}):\n{tail}")
    return errs


def main():
    all_errs: List[str] = []
    free_gb = shutil.disk_usage("/").free / (1024 ** 3)
    print(f"Root filesystem free: {free_gb:.1f} GB")
    if free_gb < 100.0:
        all_errs.append(f"Insufficient free disk space on /: {free_gb:.1f} GB < 100 GB")

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    cfg_paths = [Path(r["config_path"]) for r in manifest["runs"]]
    for cp in cfg_paths:
        print(f"Auditing {cp.name} ...", flush=True)
        if not cp.exists():
            all_errs.append(f"Missing config {cp}")
            continue
        all_errs.extend(audit_yaml_safety(cp))
        all_errs.extend(run_kotekan_validation(cp))

    if all_errs:
        print("PREFLIGHT V2 AUDIT FAILED:")
        for e in all_errs:
            print("  ERROR:", e)
        sys.exit(1)
    print(f"ALL {len(cfg_paths)} V2 COMMISSIONING CONFIGS PASSED STATIC SAFETY, --check-config, AND --dry-run.")


if __name__ == "__main__":
    main()
