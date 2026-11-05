#!/usr/bin/env bash
# ==============================================================================
# CHARTS 32-Antenna Live DPDK Ingest & GPU Beam Tracker Pipeline Launcher
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KOTEKAN_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
KOTEKAN_BIN="${KOTEKAN_ROOT}/build/kotekan/kotekan"
# Mode selection: default is live in-memory (no disk writing). Pass --direct, --direct-512ms, or --record.
MODE="live"
for arg in "$@"; do
    case "${arg}" in
        --record)
            MODE="record"
            ;;
        --direct|--direct-optimal|--fast)
            MODE="direct"
            ;;
        --direct-512ms|--direct-legacy)
            MODE="direct_512ms"
            ;;
        --live|--no-write)
            MODE="live"
            ;;
        --help|-h)
            echo "Usage: $0 [--live | --direct | --direct-512ms | --record]"
            echo "  --direct       : (Recommended) Run optimal low-latency (12.8ms / 3,840-sample, ~495MB VRAM) Direct Beam Tracker"
            echo "  --direct-512ms : Run large 512ms batch (153,600-sample, 13.2GB VRAM) Direct Beam Tracker"
            echo "  --live         : Run legacy V5 integrated beam tracking in RAM/GPU"
            echo "  --record       : Stream formed beam complex voltages to NVMe disk (/data/tracker/tracker_32ant)"
            exit 0
            ;;
    esac
done

if [ "${MODE}" = "record" ]; then
    CONFIG_FILE="${SCRIPT_DIR}/32antennas_tracker_record.yaml"
    MODE_DESC="RECORDING TO DISK (/data/tracker/tracker_32ant)"
    OUTPUT_DIR="/data/tracker/tracker_32ant"
    mkdir -p "${OUTPUT_DIR}" 2>/dev/null || {
        echo "[WARN] Cannot write to ${OUTPUT_DIR}. Falling back to /tmp/tracker_32ant"
        OUTPUT_DIR="/tmp/tracker_32ant"
        mkdir -p "${OUTPUT_DIR}"
    }
elif [ "${MODE}" = "direct" ]; then
    CONFIG_FILE="${SCRIPT_DIR}/32antennas_direct_tracker_optimal.yaml"
    MODE_DESC="OPTIMAL DIRECT BEAM TRACKER (12.8ms / 3,840 Samples, ~495MB VRAM)"
    OUTPUT_DIR="None (RAM / REST snapshot only)"
elif [ "${MODE}" = "direct_512ms" ]; then
    CONFIG_FILE="${SCRIPT_DIR}/32antennas_direct_tracker.yaml"
    MODE_DESC="DIRECT BEAM TRACKER 512ms BATCH (153,600 Samples, 13.2GB VRAM)"
    OUTPUT_DIR="None (RAM / REST snapshot only)"
else
    CONFIG_FILE="${SCRIPT_DIR}/32antennas_tracker.yaml"
    MODE_DESC="LIVE IN-MEMORY V5 (NO DISK WRITING)"
    OUTPUT_DIR="None (RAM / REST snapshot only)"
fi

PORT="${KOTEKAN_REST_PORT:-12048}"
PID_FILE="/tmp/kotekan_32ant_tracker.pid"
LOG_FILE="/tmp/kotekan_32ant_tracker.log"

echo "================================================================================"
echo " CHARTS 32-ANTENNA LIVE DPDK BEAM TRACKER PIPELINE LAUNCHER"
echo " Mode: ${MODE_DESC}"
echo "================================================================================"

if [ ! -f "${KOTEKAN_BIN}" ]; then
    echo "[ERROR] Kotekan binary not found at ${KOTEKAN_BIN}."
    echo "  -> Please compile first in build/ using: make -j$(nproc)"
    exit 1
fi

# Check if already running
if [ -f "${PID_FILE}" ]; then
    PID=$(cat "${PID_FILE}")
    if kill -0 "${PID}" 2>/dev/null; then
        echo "[INFO] Kotekan 32-antenna tracker is already running with PID: ${PID}"
        echo "  -> REST API: http://127.0.0.1:${PORT}"
        exit 0
    else
        rm -f "${PID_FILE}"
    fi
fi

echo "[1/2] Launching 32-antenna Beam Tracker with DPDK ingest..."
echo "  Configuration: ${CONFIG_FILE}"
echo "  Mode         : ${MODE_DESC}"
echo "  REST API Port: ${PORT}"
echo "  Log File     : ${LOG_FILE}"

nohup "${KOTEKAN_BIN}" -c "${CONFIG_FILE}" -b "127.0.0.1:${PORT}" > "${LOG_FILE}" 2>&1 &
PID=$!
echo "${PID}" > "${PID_FILE}"

sleep 1.0
if kill -0 "${PID}" 2>/dev/null; then
    echo "================================================================================"
    echo " [SUCCESS] 32-Antenna Beam Tracker Pipeline is RUNNING!"
    echo "================================================================================"
    echo " Process PID       : ${PID}"
    echo " REST API Server   : http://127.0.0.1:${PORT}"
    echo " Output Directory  : ${OUTPUT_DIR}"
    echo " Log Output        : tail -f ${LOG_FILE}"
    echo "--------------------------------------------------------------------------------"
    echo " Live Monitoring & Steering:"
    echo "   Web Dashboard   : python test_charts/kotekan_tracker_dashboard.py"
    echo "   CLI Telemetry   : python test_charts/kotekan_tracker_control.py status"
    echo "   Live Watch CLI  : python test_charts/kotekan_tracker_control.py watch"
    echo "   Steer Beam 0    : python test_charts/kotekan_tracker_control.py steer-lm --beam 0 --l0 0.05 --m0 -0.02"
    echo "   Stop Pipeline   : kill ${PID}"
    echo "================================================================================"
else
    echo "[ERROR] Kotekan failed to start. Last 25 lines of log:"
    tail -n 25 "${LOG_FILE}"
    exit 1
fi
