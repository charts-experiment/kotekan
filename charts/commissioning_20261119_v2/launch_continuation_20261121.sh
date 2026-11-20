#!/usr/bin/env bash
set -euo pipefail

# Launch the grouped 2026-11-21 continuation batch (test05-test11) only after verifying no
# active kotekan or scheduler process is running on ms-x01.

if pgrep -x kotekan >/dev/null 2>&1; then
    echo "ERROR: A kotekan process is already running; aborting continuation launch." >&2
    exit 1
fi
if pgrep -f "run_v2_campaign.py" >/dev/null 2>&1; then
    echo "ERROR: run_v2_campaign.py is already running; aborting continuation launch." >&2
    exit 1
fi

LOG_FILE="/data/commissioning_20261119_v2/scheduler_20261121.log"
MANIFEST="/data/commissioning_20261119_v2/schedule_manifest_20261121.json"
STATUS="/data/commissioning_20261119_v2/campaign_status_20261121.json"
REPORT="/data/commissioning_20261119_v2/CAMPAIGN_20261121_REPORT.md"

if [[ ! -f "${MANIFEST}" ]]; then
    /home/fpalma/charts-utils/.venv/bin/python \
        /home/fpalma/kotekan/charts/commissioning_20261119_v2/generate_v2_campaign.py \
        --generate-continuation-20261121
fi

nohup /home/fpalma/charts-utils/.venv/bin/python \
    /home/fpalma/kotekan/charts/commissioning_20261119_v2/run_v2_campaign.py \
    --manifest "${MANIFEST}" \
    --status "${STATUS}" \
    --report "${REPORT}" \
    > "${LOG_FILE}" 2>&1 &

echo "Launched 2026-11-21 continuation scheduler PID $! (log: ${LOG_FILE})"
