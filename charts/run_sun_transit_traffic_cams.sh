#!/usr/bin/env bash
# ==============================================================================
# Runner for CHARTS Sun Transit Traffic Cameras Direct Beam Tracker Observation
# ==============================================================================
# Usage:
#   sudo ./charts/run_sun_transit_traffic_cams.sh [--dry-run | --check | --run]
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KOTEKAN_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${KOTEKAN_ROOT}/build"
KOTEKAN_BIN="${BUILD_DIR}/kotekan/kotekan"
CONFIG_FILE="${CONFIG_FILE:-${KOTEKAN_ROOT}/charts/config/64antennas_direct_tracker_sun_transit_traffic_cams.yaml}"
DUMP_DIR="${DUMP_DIR:-/data/tracker/dump_sun_transit}"
LOG_FILE="${KOTEKAN_ROOT}/kotekan_sun_transit.log"
REST_BIND="127.0.0.1:12048"

MODE="${1:---check}"

echo "=============================================================================="
echo " CHARTS Solar Transit Traffic Cameras Observation Pipeline"
echo "=============================================================================="
echo " Mode          : ${MODE}"
echo " Kotekan Bin   : ${KOTEKAN_BIN}"
echo " Config File   : ${CONFIG_FILE}"
echo " Gains File    : ${KOTEKAN_ROOT}/charts/config/gains/gains_sun_transit_1027_interp.npy"
echo " Dump Dir      : ${DUMP_DIR}"
echo " Log File      : ${LOG_FILE}"
echo " REST Endpoint : http://${REST_BIND}"
echo "=============================================================================="

if [[ ! -x "${KOTEKAN_BIN}" ]]; then
    echo "Error: Kotekan binary not found at ${KOTEKAN_BIN}. Build it first." >&2
    exit 1
fi

if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "Error: Config file not found at ${CONFIG_FILE}." >&2
    exit 1
fi

# Pre-flight Check: YAML Config Syntax and Pipeline Topology
echo "--> Pre-flight Check: Kotekan config validation..."
"${KOTEKAN_BIN}" -c "${CONFIG_FILE}" --check-config
echo "--> Config check passed successfully."

if [[ "${MODE}" == "--check" ]]; then
    echo "Check completed. To launch live observation: sudo $0 --run"
    exit 0
fi

if [[ "${MODE}" == "--dry-run" ]]; then
    echo "--> Running dry-run..."
    "${KOTEKAN_BIN}" -c "${CONFIG_FILE}" -b "${REST_BIND}" --dry-run
    exit 0
fi

if [[ "${MODE}" == "--run" ]]; then
    if [[ $EUID -ne 0 ]]; then
        echo "Error: Live observation requires root privileges (sudo) for DPDK 100GbE NIC access." >&2
        exit 1
    fi

    # Check for active instances
    if pgrep -f "${KOTEKAN_BIN}" > /dev/null; then
        echo "Warning: An active Kotekan instance is already running!" >&2
        pgrep -fl "${KOTEKAN_BIN}"
        echo "Please terminate or wait for the other instance before launching." >&2
        exit 1
    fi

    echo "--> Preparing output dump directory: ${DUMP_DIR}"
    mkdir -p "${DUMP_DIR}"
    chmod 777 "${DUMP_DIR}"

    echo "--> Cleaning up stale DPDK hugepages..."
    rm -f /dev/hugepages/rtemap_*

    echo "--> Launching Kotekan Sun Transit Traffic Cameras pipeline..."
    echo "    (Logs streamed to ${LOG_FILE})"
    "${KOTEKAN_BIN}" -c "${CONFIG_FILE}" -b "${REST_BIND}" 2>&1 | tee "${LOG_FILE}"
fi
