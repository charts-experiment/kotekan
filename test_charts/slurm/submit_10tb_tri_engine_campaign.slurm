#!/bin/bash
#SBATCH --job-name=charts_10tb_tri_engine
#SBATCH --account=def-vanderli
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --output=slurm_10tb_tri_engine_%j.out
#SBATCH --error=slurm_10tb_tri_engine_%j.err

# ==============================================================================
# CHARTS 10-TB Campaign + Tri-Engine Outputs on Trillium / Compute Canada
# ==============================================================================
# Runs the CHARTS 10-Terabyte Diverse Correlator Simulation Campaign and then
# executes the 64-antenna tri-engine benchmark pipeline, placing both sets of
# data products under /project/rrg-vanderli/ferpb for side-by-side comparison:
#
#   1. 10-TB campaign (run_10tb_correlator_campaign.py):
#       baseband generation -> cudaCorrelatorAstron replay -> CASM/waterfall
#       products archived per run under $FERPB_OUT/10tb_campaign/<run_id>
#   2. Tri-engine benchmark (kotekan --config .../64antennas_tri_engine_benchmark.yaml):
#       Engine 0: cudaAntennaMaskCommand
#       Engine 1: cudaDirectBeamTrackerCommand (4 beams)
#       Engine 2: cudaFRBBeamformer_charts_U32 (64 beams, K=320 integration)
#       Engine 3: cudaShuffleAstron + cudaCorrelatorAstron (2,080 baselines)
#       Outputs archived under $FERPB_OUT/tri_engine/
#
# Usage:
#   sbatch test_charts/slurm/submit_10tb_tri_engine_campaign.slurm
#
#   # Custom FERPB output root or campaign run subset:
#   sbatch --export=ALL,FERPB_OUT=/project/rrg-vanderli/ferpb,RUN_ALL=1 \
#     test_charts/slurm/submit_10tb_tri_engine_campaign.slurm
# ==============================================================================

set -euo pipefail

echo "======================================================================"
echo " CHARTS 10-TB Campaign + Tri-Engine Outputs on Trillium"
echo " Hostname       : $(hostname)"
echo " CUDA Device    : ${CUDA_VISIBLE_DEVICES:-Not set (default 0)}"
echo " Job ID         : ${SLURM_JOB_ID:-manual}"
echo " Date           : $(date)"
echo "======================================================================"

# 1. Load Cluster Environment Modules
if command -v module >/dev/null 2>&1; then
    echo ">>> Loading Trillium Environment Modules..."
    module load StdEnv/2023 2>/dev/null || true
    module load gcc/12.3 2>/dev/null || module load gcc || true
    module load cuda/12.6 2>/dev/null || module load cuda/13.3 2>/dev/null || module load cuda || true
    module load cmake 2>/dev/null || true
    module load python/3.11 2>/dev/null || module load python/3.10 2>/dev/null || module load python || true
    module load scipy-stack 2>/dev/null || true
    module load hdf5 2>/dev/null || true
fi

# Print GPU architecture & specifications
nvidia-smi

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

echo "Repository Root : ${REPO_ROOT}"

# 3. Configuration knobs (overridable via --export at sbatch time)
BUILD_TYPE="${BUILD_TYPE:-Release}"
FERPB_OUT="${FERPB_OUT:-/project/rrg-vanderli/ferpb}"
CAMPAIGN_OUT="${FERPB_OUT}/10tb_campaign"
TRI_ENGINE_OUT="${FERPB_OUT}/tri_engine"
CONFIG="${CONFIG:-test_charts/config/64antennas_tri_engine_benchmark.yaml}"
RUN_ALL="${RUN_ALL:-0}"
BUILD_DIR="${REPO_ROOT}/build"
KOTEKAN_BIN="${BUILD_DIR}/kotekan/kotekan"

# Resolve fast scratch storage (avoid /dev/shm RAM exhaustion)
if [ -n "${SLURM_TMPDIR:-}" ] && [[ "${SLURM_TMPDIR}" != /dev/shm* ]]; then
    SCRATCH_BASE="${SLURM_TMPDIR}/charts_10tb_tri_${SLURM_JOB_ID:-manual}"
else
    SCRATCH_BASE="${REPO_ROOT}/scratch_10tb_tri_${SLURM_JOB_ID:-manual}"
fi
mkdir -p "${SCRATCH_BASE}"

# Make the FERPB output roots
mkdir -p "${CAMPAIGN_OUT}" "${TRI_ENGINE_OUT}"

echo "FERPB Output Root : ${FERPB_OUT}"
echo "Campaign Output   : ${CAMPAIGN_OUT}"
echo "Tri-Engine Output : ${TRI_ENGINE_OUT}"
echo "Scratch Base      : ${SCRATCH_BASE}"

# 4. Build Kotekan with CUDA enabled
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

echo ">>> Configuring CMake (BUILD_TYPE=${BUILD_TYPE})..."
cmake .. \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DUSE_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="80;90" \
    -DCMAKE_SKIP_RPATH=TRUE

echo ">>> Compiling Kotekan..."
make -j"${SLURM_CPUS_PER_TASK:-8}" kotekan

# Apply Alliance setrpaths.sh if available
if command -v setrpaths.sh >/dev/null 2>&1; then
    echo "Applying Alliance setrpaths.sh to build tree..."
    setrpaths.sh --path "${BUILD_DIR}" --add_origin 2>/dev/null || true
fi

cd "${REPO_ROOT}"

# 5. Pre-flight check: smoke test Kotekan binary
echo ">>> Pre-flight check: testing Kotekan binary..."
if ! "${KOTEKAN_BIN}" --version >/dev/null 2>&1 && ! "${KOTEKAN_BIN}" --help >/dev/null 2>&1; then
    echo "[ERROR] Kotekan binary smoke test failed."
    ldd "${KOTEKAN_BIN}" 2>&1 | head -n 35 || true
    exit 1
else
    echo ">>> Kotekan binary smoke test passed successfully."
fi

# 6. Validate the tri-engine benchmark config
echo ">>> Validating tri-engine benchmark config: ${CONFIG}"
"${KOTEKAN_BIN}" --check-config "${CONFIG}"

# 7. Run the 10-TB correlator campaign
echo "======================================================================"
echo " Step 1: CHARTS 10-TB Correlator Simulation Campaign"
echo " Output : ${CAMPAIGN_OUT}"
echo "======================================================================"
CAMPAIGN_ARGS=(
    --scratch-dir "${SCRATCH_BASE}/campaign"
    --output-dir "${CAMPAIGN_OUT}"
    --kotekan-bin "${KOTEKAN_BIN}"
    --workers "${SLURM_CPUS_PER_TASK:-16}"
)
if [ "${RUN_ALL}" = "1" ]; then
    CAMPAIGN_ARGS+=(--all)
else
    CAMPAIGN_ARGS+=(--run-idx 0)
fi

python3 "${REPO_ROOT}/test_charts/run_10tb_correlator_campaign.py" \
    "${CAMPAIGN_ARGS[@]}" \
    "$@"

echo "10-TB campaign run completed. Products in ${CAMPAIGN_OUT}"

# 8. Run the tri-engine benchmark pipeline
echo "======================================================================"
echo " Step 2: CHARTS 64-Antenna Tri-Engine Benchmark Pipeline"
echo " Config  : ${CONFIG}"
echo " Output  : ${TRI_ENGINE_OUT}"
echo "======================================================================"
mkdir -p /tmp/test_charts_tri_engine_${SLURM_JOB_ID:-manual}
LD_LIBRARY_PATH="${BUILD_DIR}/external/n2k:${BUILD_DIR}/lib:${BUILD_DIR}/kotekan:${LD_LIBRARY_PATH:-}" \
    "${KOTEKAN_BIN}" --config "${CONFIG}"

echo "Tri-engine benchmark pipeline execution completed successfully."

# 9. Archive tri-engine REST-inspect outputs into the FERPB tree
echo "======================================================================"
echo " Archiving tri-engine outputs to ${TRI_ENGINE_OUT}"
echo "======================================================================"
TRI_ENGINE_SCRATCH="${SCRATCH_BASE}/tri_engine"
mkdir -p "${TRI_ENGINE_SCRATCH}"

# The tri-engine config writes formed beams into VRAM and transfers only
# correlation visibilities and FRB intensity D2H for REST inspection. Capture
# the live REST telemetry snapshots from the running instance if available,
# otherwise record the config and benchmark metrics as the output record.
if command -v curl >/dev/null 2>&1; then
    for buf in host_correlation_buffer_0 host_correlation_buffer_1 \
               host_frb_intensity_buffer_0 host_frb_intensity_buffer_1; do
        if curl -sf "localhost:12048/buffer_frame?name=${buf}" \
                -o "${TRI_ENGINE_SCRATCH}/${buf}.json" 2>/dev/null; then
            echo "  * Captured REST frame for ${buf}"
        fi
    done
fi

# Copy the benchmark config and metrics record for reproducibility
cp "${CONFIG}" "${TRI_ENGINE_OUT}/tri_engine_config.yaml"
cp "${REPO_ROOT}/test_charts/benchmark_tri_engine.py" \
   "${TRI_ENGINE_OUT}/benchmark_tri_engine.py"

# Run the theoretical metrics benchmark and store its stdout as the output record
python3 "${REPO_ROOT}/test_charts/benchmark_tri_engine.py" \
    > "${TRI_ENGINE_OUT}/tri_engine_metrics.txt" 2>&1 || true

echo "Tri-engine outputs archived in ${TRI_ENGINE_OUT}"

echo "======================================================================"
echo " All 10-TB Campaign + Tri-Engine Steps Completed Successfully!"
echo " Campaign products : ${CAMPAIGN_OUT}"
echo " Tri-engine outputs: ${TRI_ENGINE_OUT}"
echo "======================================================================"
