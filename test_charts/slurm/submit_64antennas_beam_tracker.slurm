#!/bin/bash
#SBATCH --job-name=kotekan_tracker_64
#SBATCH --account=def-vanderli
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=slurm_tracker_64_%j.out
#SBATCH --error=slurm_tracker_64_%j.err

set -euo pipefail

echo "======================================================================"
echo " Launching CHARTS 64-Antenna Beam Tracker Pipeline on Trillium"
echo " Hostname       : $(hostname)"
echo " CUDA Device    : ${CUDA_VISIBLE_DEVICES:-0}"
echo " Job ID         : ${SLURM_JOB_ID:-manual}"
echo " Date           : $(date)"
echo "======================================================================"

# 1. Load Cluster Environment Modules
if command -v module >/dev/null 2>&1; then
    echo ">>> Loading Trillium Environment Modules..."
    module load StdEnv/2023 2>/dev/null || true
    module load gcc/12.3 2>/dev/null || module load gcc 2>/dev/null || true
    module load cuda/12.6 2>/dev/null || module load cuda/13.3 2>/dev/null || module load cuda 2>/dev/null || true
    module load cmake 2>/dev/null || true
    module load boost 2>/dev/null || true
    module load python/3.11 2>/dev/null || module load python/3.10 2>/dev/null || module load python 2>/dev/null || true
    module load scipy-stack 2>/dev/null || true
    module load hdf5 2>/dev/null || true
    module load ffmpeg 2>/dev/null || true
fi

nvidia-smi || true

# 2. Resolve Repository Root (Slurm-safe: sbatch executes a spool copy)
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${SUBMIT_DIR}/CMakeLists.txt" ] && [ -d "${SUBMIT_DIR}/test_charts" ]; then
    REPO_ROOT="${SUBMIT_DIR}"
elif [ -f "${SUBMIT_DIR}/../CMakeLists.txt" ] && [ -d "${SUBMIT_DIR}/../test_charts" ]; then
    REPO_ROOT="$(cd "${SUBMIT_DIR}/.." && pwd)"
elif [ -f "${SUBMIT_DIR}/../../CMakeLists.txt" ] && [ -d "${SUBMIT_DIR}/../../test_charts" ]; then
    REPO_ROOT="$(cd "${SUBMIT_DIR}/../.." && pwd)"
elif [ -f "${SCRIPT_DIR}/../../CMakeLists.txt" ] && [ -d "${SCRIPT_DIR}/../../test_charts" ]; then
    REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
else
    REPO_ROOT="${PWD}"
fi
cd "${REPO_ROOT}"

BUILD_TYPE="${BUILD_TYPE:-Release}"
CUDA_ARCH="${CUDA_ARCH:-native}"
BUILD_DIR="${REPO_ROOT}/build"
KOTEKAN_BIN="${BUILD_DIR}/kotekan/kotekan"
BUILD_KOTEKAN="${BUILD_KOTEKAN:-0}"
CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/test_charts/config/64antennas_beam_tracker_test.yaml}"
REST_PORT="${REST_PORT:-12048}"
INSPECT_SCRIPT="${REPO_ROOT}/test_charts/inspect_tracker_output.py"
OUT_DIR="/tmp/test_charts_tracker"

# 3. Build Kotekan if requested or missing
if [ "${BUILD_KOTEKAN}" = "1" ] || [ ! -x "${KOTEKAN_BIN}" ]; then
    echo ">>> Compiling Kotekan with CUDA enabled (CUDA_ARCH=${CUDA_ARCH})..."
    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"
    cmake .. \
        -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
        -DUSE_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
        -DUSE_NUMA=ON \
        -DWERROR=OFF \
        -DCMAKE_SKIP_RPATH=TRUE
    make -j"${SLURM_CPUS_PER_TASK:-16}" kotekan
    if command -v setrpaths.sh >/dev/null 2>&1; then
        echo "Applying Alliance setrpaths.sh to build tree..."
        setrpaths.sh --path "${BUILD_DIR}" --add_origin 2>/dev/null || true
    fi
    cd "${REPO_ROOT}"
fi

export LD_LIBRARY_PATH="${BUILD_DIR}/external/n2k:${BUILD_DIR}/lib:${BUILD_DIR}/kotekan:${LD_LIBRARY_PATH:-}"

# 4. Validate configuration
echo ">>> Validating config: ${CONFIG_FILE}"
"${KOTEKAN_BIN}" --check-config -c "${CONFIG_FILE}"

mkdir -p "${OUT_DIR}"
rm -f "${OUT_DIR}"/*.bin

echo "Starting Kotekan Pipeline in background (REST 127.0.0.1:${REST_PORT})..."
"${KOTEKAN_BIN}" -c "${CONFIG_FILE}" -b "127.0.0.1:${REST_PORT}" &
KOTEKAN_PID=$!

echo "Kotekan PID: ${KOTEKAN_PID}"

# Wait for REST API to initialize (break early if Kotekan finishes quickly)
REST_READY=0
echo "Waiting for Kotekan REST API to come up on port ${REST_PORT}..."
for i in $(seq 1 30); do
    if ! kill -0 "${KOTEKAN_PID}" >/dev/null 2>&1; then
        echo "Kotekan process completed before REST polling finished."
        break
    fi
    if curl -sf "http://127.0.0.1:${REST_PORT}/beam_tracker/status" >/dev/null 2>&1; then
        echo "REST API is active!"
        REST_READY=1
        break
    fi
    sleep 0.5
done

if [ "${REST_READY}" = "1" ] && kill -0 "${KOTEKAN_PID}" >/dev/null 2>&1; then
    echo ">>> Initial Beam Tracker Status:"
    curl -sf "http://127.0.0.1:${REST_PORT}/beam_tracker/status" | python3 -m json.tool || true

    # Update Celestial Target to Crab Pulsar (RA=83.633 deg, Dec=22.014 deg)
    echo ">>> Setting Celestial Tracking Target via REST API..."
    curl -sf -X POST "http://127.0.0.1:${REST_PORT}/beam_tracker/set_celestial_target" \
         -H "Content-Type: application/json" \
         -d '{"beam_index": 0, "ra_deg": 83.633, "dec_deg": 22.014, "lst_hours": 5.575}' | python3 -m json.tool || true
fi

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
