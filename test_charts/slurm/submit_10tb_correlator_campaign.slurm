#!/bin/bash
#SBATCH --job-name=charts_10tb_corr
#SBATCH --account=def-vanderli
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=slurm_10tb_corr_%A_%a.log
#SBATCH --error=slurm_10tb_corr_%A_%a.err
#SBATCH --array=0-15

# ==============================================================================
# CHARTS 10-Terabyte Diverse Correlator Simulation Campaign (Trillium Cluster)
# ==============================================================================
# Executes a scientifically diverse, 10-TB campaign of realistic baseband
# generation and Kotekan GPU Tensor Core Correlator (cudaCorrelatorAstron) replay.
#
# Simulation Matrix (16 Diverse Runs = 9.995 TB):
#   - Task 0:  S1_transit_vela          (396.4 GB | 64 ant, 336 ch | Vela SNR meridian transit)
#   - Task 1:  S1_transit_crab          (396.4 GB | 64 ant, 336 ch | Crab Nebula/pulsar transit)
#   - Task 2:  S1_transit_sgra          (396.4 GB | 64 ant, 336 ch | Sgr A* / Galactic Center transit)
#   - Task 3:  S1_transit_cena          (396.4 GB | 64 ant, 336 ch | Centaurus A radio galaxy transit)
#   - Task 4:  S1_transit_puppisa       (396.4 GB | 64 ant, 336 ch | Puppis A supernova remnant transit)
#   - Task 5:  S2_frb_low_dm            (495.5 GB | 64 ant, 336 ch | Low-DM High-SNR FRB)
#   - Task 6:  S2_frb_high_dm_wide      (660.6 GB | 64 ant, 672 ch | High-DM 672-channel full band FRB)
#   - Task 7:  S2_frb_extreme_256ant    (528.5 GB | 256 ant, 336 ch| Extreme FRB across 256 antennas)
#   - Task 8:  S3_pulsar_millisecond    (495.5 GB | 64 ant, 336 ch | Fast ms pulsar train PSR J0437)
#   - Task 9:  S3_pulsar_canonical      (495.5 GB | 64 ant, 336 ch | Canonical 89ms slow pulsar train)
#   - Task 10: S4_rfi_multiline         (495.5 GB | 64 ant, 336 ch | Multi-line terrestrial stationary RFI)
#   - Task 11: S4_rfi_leo_sweep         (495.5 GB | 64 ant, 336 ch | Rapid non-celestial LEO satellite sweep)
#   - Task 12: S5_complex_night_64ant   (1057.0 GB | 64 ant, 672 ch| Deep night 2 FRBs + Pulsar + RFI)
#   - Task 13: S5_complex_day_sun_64ant (1057.0 GB | 64 ant, 672 ch| Active Sun Day FRB + LEO + Solar burst)
#   - Task 14: S5_complex_256ant_336ch  (1123.0 GB | 256 ant, 336 ch| Full 256-ant array complex scene)
#   - Task 15: S5_complex_256ant_672ch  (1109.8 GB | 256 ant, 672 ch| Full 256-ant 672-ch dual shard benchmark)
#
# Usage:
#   # 1. Submit full 10-TB campaign as a 16-task Slurm Job Array (parallel across nodes):
#   sbatch test_charts/slurm/submit_10tb_correlator_campaign.slurm
#
#   # 2. Run single task (e.g. Task 0: Vela transit, or Task 12: Complex Night 1 TB):
#   sbatch --array=0 test_charts/slurm/submit_10tb_correlator_campaign.slurm
#   sbatch --array=12 test_charts/slurm/submit_10tb_correlator_campaign.slurm
#
#   # 3. Dry-run mode (verifies configurations & YAML generation without writing 10 TB):
#   sbatch --export=ALL,DRY_RUN=1 test_charts/slurm/submit_10tb_correlator_campaign.slurm
#
#   # 4. Sequential single-node execution of all 16 simulations:
#   sbatch --array=0 --export=ALL,RUN_ALL=1 test_charts/slurm/submit_10tb_correlator_campaign.slurm
# ==============================================================================

set -euo pipefail

echo "======================================================================"
echo " CHARTS 10-TERABYTE CORRELATOR CAMPAIGN: TRILLIUM HPC LAUNCHER"
echo " Hostname       : $(hostname)"
echo " CUDA Device    : ${CUDA_VISIBLE_DEVICES:-0}"
echo " Slurm Job ID   : ${SLURM_JOB_ID:-manual}"
echo " Array Task ID  : ${SLURM_ARRAY_TASK_ID:-0}"
echo " CPUs Allocated : ${SLURM_CPUS_PER_TASK:-16}"
echo " Memory         : ${SLURM_MEM_PER_NODE:-186 GiB (Trillium GPU default)}"
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
if [ -f "${SUBMIT_DIR}/test_charts/run_10tb_correlator_campaign.py" ]; then
    REPO_ROOT="${SUBMIT_DIR}"
elif [ -f "${SUBMIT_DIR}/../../test_charts/run_10tb_correlator_campaign.py" ]; then
    REPO_ROOT="$(cd "${SUBMIT_DIR}/../.." && pwd)"
else
    REPO_ROOT="${PWD}"
fi
cd "${REPO_ROOT}"

# 3. Resolve Fast Scratch Storage (NVMe / Node-Local Scratch)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
JOB_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"

if [ -n "${SCRATCH_DIR:-}" ]; then
    SCRATCH_BASE="${SCRATCH_DIR}/charts_10tb_${JOB_ID}_task${TASK_ID}"
elif [ -n "${SLURM_TMPDIR:-}" ] && [[ "${SLURM_TMPDIR}" != /dev/shm* ]]; then
    SCRATCH_BASE="${SLURM_TMPDIR}/charts_10tb_task${TASK_ID}"
elif [ -n "${SCRATCH:-}" ]; then
    SCRATCH_BASE="${SCRATCH}/kotekan_10tb/${JOB_ID}_task${TASK_ID}"
else
    SCRATCH_BASE="${REPO_ROOT}/scratch_10tb_${JOB_ID}_task${TASK_ID}"
fi
mkdir -p "${SCRATCH_BASE}"

# Resolve Permanent Output Directory
if [ -n "${OUTPUT_DIR:-}" ]; then
    DEST_OUTPUT="${OUTPUT_DIR}"
elif [ -n "${SCRATCH:-}" ]; then
    DEST_OUTPUT="${SCRATCH}/charts_10tb_campaign_results"
else
    DEST_OUTPUT="${REPO_ROOT}/test_charts/data/campaign_10tb_output"
fi
mkdir -p "${DEST_OUTPUT}"

# 4. Build / Locate Kotekan Binary
KOTEKAN_BIN="${REPO_ROOT}/build/kotekan/kotekan"
BUILD_KOTEKAN="${BUILD_KOTEKAN:-0}"
if [ "${BUILD_KOTEKAN}" = "1" ] || [ ! -x "${KOTEKAN_BIN}" ]; then
    echo ">>> Compiling Kotekan with CUDA support..."
    mkdir -p build && cd build
    cmake .. -DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="90;80" -DUSE_NUMA=ON -DWERROR=OFF
    make -j"${SLURM_CPUS_PER_TASK:-8}" kotekan
    cd "${REPO_ROOT}"
fi

# 5. Prepare Campaign Execution Arguments
WORKERS="${SLURM_CPUS_PER_TASK:-16}"
DRY_RUN_FLAG=""
if [ "${DRY_RUN:-0}" = "1" ]; then
    DRY_RUN_FLAG="--dry-run"
fi

CLEANUP_FLAG=""
if [ "${CLEANUP_BASEBAND:-0}" = "1" ]; then
    CLEANUP_FLAG="--cleanup-baseband"
fi

RUN_MODE_FLAG="--run-idx ${TASK_ID}"
if [ "${RUN_ALL:-0}" = "1" ]; then
    RUN_MODE_FLAG="--all"
fi

echo "----------------------------------------------------------------------"
echo " EXECUTION ENVIRONMENT DETAILS:"
echo " Scratch Base   : ${SCRATCH_BASE}"
echo " Output Base    : ${DEST_OUTPUT}"
echo " Kotekan Bin    : ${KOTEKAN_BIN}"
echo " Worker Threads : ${WORKERS}"
echo " Run Mode       : ${RUN_MODE_FLAG}"
echo " Dry Run        : ${DRY_RUN:-0}"
echo " Cleanup Bin    : ${CLEANUP_BASEBAND:-0}"
echo "----------------------------------------------------------------------"

# 6. Execute Campaign Driver
python3 "${REPO_ROOT}/test_charts/run_10tb_correlator_campaign.py" \
    ${RUN_MODE_FLAG} \
    --scratch-dir "${SCRATCH_BASE}" \
    --output-dir "${DEST_OUTPUT}" \
    --kotekan-bin "${KOTEKAN_BIN}" \
    --workers "${WORKERS}" \
    ${DRY_RUN_FLAG} \
    ${CLEANUP_FLAG} \
    "$@"

echo "======================================================================"
echo " [COMPLETE] Task ${TASK_ID} finished successfully at $(date)"
echo " Results available in: ${DEST_OUTPUT}"
echo "======================================================================"
