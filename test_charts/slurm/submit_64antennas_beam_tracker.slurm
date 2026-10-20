#!/bin/bash
#SBATCH --job-name=kotekan_tracker_64
#SBATCH --output=slurm_tracker_64_%j.out
#SBATCH --error=slurm_tracker_64_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --partition=gpu

set -euo pipefail

echo "======================================================================"
echo " Launching CHARTS 64-Antenna Beam Tracker Pipeline on Trillium"
echo " Hostname       : $(hostname)"
echo " CUDA Device    : ${CUDA_VISIBLE_DEVICES:-0}"
echo " Job ID         : ${SLURM_JOB_ID}"
echo "======================================================================"

# Load modules
if command -v module >/dev/null 2>&1; then
    module load cuda/13.3 || module load cuda/12.6 || module load cuda || true
    module load gcc/12.3 || module load gcc || true
    module load python/3.10 || module load python || true
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
CONFIG_FILE="${REPO_ROOT}/test_charts/config/64antennas_beam_tracker_test.yaml"
KOTEKAN_BIN="${REPO_ROOT}/build/kotekan/kotekan"
INSPECT_SCRIPT="${REPO_ROOT}/test_charts/inspect_tracker_output.py"
OUT_DIR="/tmp/test_charts_tracker"

mkdir -p "${OUT_DIR}"
rm -f "${OUT_DIR}"/*.bin

echo "Starting Kotekan Pipeline in background..."
"${KOTEKAN_BIN}" -c "${CONFIG_FILE}" &
KOTEKAN_PID=$!

echo "Kotekan PID: ${KOTEKAN_PID}"

# Wait for REST API to initialize
echo "Waiting for Kotekan REST API to come up on port 8080..."
for i in {1..30}; do
    if curl -s http://localhost:8080/beam_tracker/status >/dev/null 2>&1; then
        echo "REST API is active!"
        break
    fi
    sleep 0.5
done

# Query status
echo ">>> Initial Beam Tracker Status:"
curl -s http://localhost:8080/beam_tracker/status | python3 -m json.tool || true

# Update Celestial Target to Crab Pulsar (RA=83.633 deg, Dec=22.014 deg)
echo ">>> Setting Celestial Tracking Target via REST API..."
curl -s -X POST http://localhost:8080/beam_tracker/set_celestial_target \
     -H "Content-Type: application/json" \
     -d '{"beam_index": 0, "ra_deg": 83.633, "dec_deg": 22.014, "lst_hours": 5.575}' | python3 -m json.tool || true

# Wait for Kotekan process to complete writing dumped frames
echo "Waiting for pipeline execution..."
wait "${KOTEKAN_PID}"

echo "======================================================================"
echo " Inspecting Complex Voltages Output Frames"
echo "======================================================================"
python3 "${INSPECT_SCRIPT}"

echo "======================================================================"
echo " 64-Antenna Beam Tracker SLURM Job Completed Successfully!"
echo "======================================================================"
