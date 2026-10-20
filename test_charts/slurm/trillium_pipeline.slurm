#!/bin/bash
#SBATCH --job-name=charts_pipeline
#SBATCH --account=def-vanderli
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=04:00:00
#SBATCH --output=slurm_charts_pipeline_%j.log

# ==============================================================================
# CHARTS Unified Pipeline Launcher for Trillium / Compute Canada Cluster
# ==============================================================================
# Runs the centralized CHARTS simulation & Kotekan pipeline:
#   1. Generates physical baseband data (Sun, thermal noise, RFI, transients)
#   2. Correlates baseband stream via GPU Tensor Core correlator
#   3. Forms and tracks 8 steerable beams
#   4. Reconstructs and inspects Hermitian correlation matrices
#   5. Exports CASM correlation matrices, multi-baseline waterfalls, and videos
#
# Usage:
#   # Standard 1-minute daytime run:
#   sbatch test_charts/slurm/trillium_pipeline.slurm
#
#   # Custom preset and nighttime profile:
#   sbatch --export=ALL,PRESET=5min,PROFILE=night test_charts/slurm/trillium_pipeline.slurm
#
#   # Pass arbitrary CLI flags directly to charts_sim.py:
#   sbatch test_charts/slurm/trillium_pipeline.slurm --antennas 64 --max-beams 8
# ==============================================================================

set -euo pipefail

echo "======================================================================"
echo " CHARTS Unified Pipeline: Trillium HPC Launcher"
echo " Hostname       : $(hostname)"
echo " CUDA Device    : ${CUDA_VISIBLE_DEVICES:-0}"
echo " Job ID         : ${SLURM_JOB_ID:-manual}"
echo " Date           : $(date)"
echo "======================================================================"

# 1. Load Cluster Environment Modules
if command -v module >/dev/null 2>&1; then
    echo ">>> Loading Trillium Environment Modules..."
    module load StdEnv/2023 2>/dev/null || true
    module load gcc/12.3 2>/dev/null || module load gcc || true
    module load cuda/12.6 2>/dev/null || module load cuda || true
    module load cmake 2>/dev/null || true
    module load boost 2>/dev/null || true
    module load python/3.11 2>/dev/null || module load python/3.10 2>/dev/null || module load python || true
    module load scipy-stack 2>/dev/null || true
    module load hdf5 2>/dev/null || true
    module load ffmpeg 2>/dev/null || true
fi

# 2. Resolve Repository Root
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
if [ -f "${SUBMIT_DIR}/test_charts/charts_sim.py" ]; then
    REPO_ROOT="${SUBMIT_DIR}"
elif [ -f "${SUBMIT_DIR}/../../test_charts/charts_sim.py" ]; then
    REPO_ROOT="$(cd "${SUBMIT_DIR}/../.." && pwd)"
else
    REPO_ROOT="${PWD}"
fi
cd "${REPO_ROOT}"

# 3. Resolve Fast Scratch Storage (prevent /dev/shm RAM exhaustion)
if [ -n "${SCRATCH_DIR:-}" ]; then
    SCRATCH_BASE="${SCRATCH_DIR}/charts_${SLURM_JOB_ID:-manual}"
elif [ -n "${SLURM_TMPDIR:-}" ] && [[ "${SLURM_TMPDIR}" != /dev/shm* ]]; then
    SCRATCH_BASE="${SLURM_TMPDIR}/charts_${SLURM_JOB_ID:-manual}"
elif [ -n "${SCRATCH:-}" ]; then
    SCRATCH_BASE="${SCRATCH}/kotekan_scratch/charts_${SLURM_JOB_ID:-manual}"
else
    SCRATCH_BASE="${REPO_ROOT}/scratch_charts_${SLURM_JOB_ID:-manual}"
fi
mkdir -p "${SCRATCH_BASE}"

# 4. Build Kotekan if not already compiled
KOTEKAN_BIN="${REPO_ROOT}/build/kotekan/kotekan"
BUILD_KOTEKAN="${BUILD_KOTEKAN:-0}"
if [ "${BUILD_KOTEKAN}" = "1" ] || [ ! -x "${KOTEKAN_BIN}" ]; then
    echo ">>> Compiling Kotekan with CUDA..."
    mkdir -p build && cd build
    cmake .. -DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="90;80" -DUSE_NUMA=ON -DWERROR=OFF
    make -j"${SLURM_CPUS_PER_TASK:-8}" kotekan
    cd "${REPO_ROOT}"
fi

# 5. Execute Unified Pipeline CLI
PRESET="${PRESET:-1min}"
PROFILE="${PROFILE:-day}"
WORKERS="${SLURM_CPUS_PER_TASK:-16}"

echo ">>> Launching charts_sim.py pipeline..."
exec python3 "${REPO_ROOT}/test_charts/charts_sim.py" pipeline \
    --preset "${PRESET}" \
    --profile "${PROFILE}" \
    --scratch-dir "${SCRATCH_BASE}" \
    --output-dir "${REPO_ROOT}/test_charts/data/sim_output" \
    --kotekan-bin "${KOTEKAN_BIN}" \
    --workers "${WORKERS}" \
    "$@"
