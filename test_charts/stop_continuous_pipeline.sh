#!/usr/bin/env bash
# ==============================================================================
# CHARTS Kotekan Continuous Beam Tracker Pipeline Terminator
# ==============================================================================
set -euo pipefail

WORK_DIR="/tmp/kotekan_continuous_tracker"
PID_FILE="${WORK_DIR}/kotekan.pid"

if [ -f "${PID_FILE}" ]; then
    PID=$(cat "${PID_FILE}")
    if kill -0 "${PID}" 2>/dev/null; then
        echo "[INFO] Terminating Kotekan pipeline (PID: ${PID})..."
        kill -15 "${PID}" || true
        for i in {1..10}; do
            if kill -0 "${PID}" 2>/dev/null; then
                sleep 0.2
            else
                break
            fi
        done
        if kill -0 "${PID}" 2>/dev/null; then
            echo "[WARN] Force killing Kotekan..."
            kill -9 "${PID}" || true
        fi
        echo "[SUCCESS] Kotekan stopped."
    else
        echo "[INFO] Kotekan is not running."
    fi
    rm -f "${PID_FILE}"
else
    echo "[INFO] No Kotekan PID file found at ${PID_FILE}."
fi
