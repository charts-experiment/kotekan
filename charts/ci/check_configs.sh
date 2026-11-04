#!/bin/bash
# Run `kotekan --check-config` on every CHARTS config.
#
# The check is static (see validate_config_static in kotekan/kotekan.cpp): it opens
# no device, allocates no buffer and binds no port, so it is safe in CI and next to
# a live pipeline. It catches unknown stage types, duplicate buffers and dangling
# metadata pools; charts/ci/validate_charts_configs.py covers GPU commands, DPDK
# handlers and frequency shards.
#
# Usage: charts/ci/check_configs.sh <path to kotekan binary> [config.yaml ...]

set -uo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <kotekan binary> [config.yaml ...]" >&2
    exit 2
fi
kotekan=$1
shift
if [[ ! -x "$kotekan" ]]; then
    echo "::error::kotekan binary not found or not executable: $kotekan" >&2
    exit 2
fi

repo=$(cd "$(dirname "$0")/../.." && pwd)
if [[ $# -gt 0 ]]; then
    configs=("$@")
else
    configs=("$repo"/charts/config/*.yaml)
fi

log=$(mktemp)
trap 'rm -f "$log"' EXIT

failed=()
for cfg in "${configs[@]}"; do
    if ! "$kotekan" --check-config -c "$cfg" > "$log" 2>&1; then
        failed+=("$cfg")
        echo "::error file=${cfg#"$repo"/}::kotekan --check-config failed"
        grep -E "config check|FAILED|ERROR" "$log"
    fi
done

echo "kotekan --check-config: $((${#configs[@]} - ${#failed[@]}))/${#configs[@]} configs passed"
[[ ${#failed[@]} -eq 0 ]]
