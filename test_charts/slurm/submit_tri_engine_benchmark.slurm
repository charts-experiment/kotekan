#!/bin/bash
#SBATCH --job-name=charts_tri_engine
#SBATCH --account=def-vanderli
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --output=slurm_tri_engine_%j.out
#SBATCH --error=slurm_tri_engine_%j.err

# ==============================================================================
# CHARTS 64-Antenna Tri-Engine Benchmark on Trillium / Compute Canada
# ==============================================================================
# Builds Kotekan with CUDA and runs the 64-antenna tri-engine benchmark
# pipeline, exercising all three concurrent DSP engines on a single GPU:
#
#   Engine 0: In-Place GPU Antenna Health Inspection & Voltage Blanking
#             (cudaAntennaMaskCommand)
#   Engine 1: CHARTS Direct Astrometric Multi-Beam Tracker
#             (cudaDirectBeamTrackerCommand, 4 beams)
#   Engine 2: CHARTS 64-Beam Time-Integrated FRB Intensity Beamformer
#             (cudaFRBBeamformer_charts_U32, 8x8 grid, K=320 integration)
#   Engine 3: Astron WMMA Tensor Core Correlator (all 2,080 baselines)
#
# Ingestion: In-process Synthetic Astrophysical Baseband Generator
#            (chartsFEngineSim)
# Zero-Disk Mode: Formed beams stay in VRAM; only correlation visibilities
#                 and FRB intensity are transferred D2H for REST inspection.
#
# Usage:
#   sbatch test_charts/slurm/submit_tri_engine_benchmark.slurm
#
#   # Custom build type or config override:
#   sbatch --export=ALL,BUILD_TYPE=Release,CONFIG=charts/config/64antennas_tri_engine_dpdk.yaml \
#     test_charts/slurm/submit_tri_engine_benchmark.slurm
# ==============================================================================

set -euo pipefail

echo "======================================================================"
echo " CHARTS 64-Antenna Tri-Engine Benchmark on Trillium"
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
if [ -f "${SUBMIT_DIR}/test_charts/benchmark_tri_engine.py" ]; then
    REPO_ROOT="${SUBMIT_DIR}"
elif [ -f "${SUBMIT_DIR}/../../test_charts/benchmark_tri_engine.py" ]; then
    REPO_ROOT="$(cd "${SUBMIT_DIR}/../.." && pwd)"
else
    REPO_ROOT="${PWD}"
fi
cd "${REPO_ROOT}"

echo "Repository Root : ${REPO_ROOT}"

# 3. Configuration knobs (overridable via --export at sbatch time)
BUILD_TYPE="${BUILD_TYPE:-Release}"
CONFIG="${CONFIG:-test_charts/config/64antennas_tri_engine_benchmark.yaml}"
BUILD_DIR="${REPO_ROOT}/build"
KOTEKAN_BIN="${BUILD_DIR}/kotekan/kotekan"

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

# 5. Pre-flight check: smoke test Kotekan binary execution and dynamic linker
echo ">>> Pre-flight check: testing Kotekan binary..."
if ! "${KOTEKAN_BIN}" --version >/dev/null 2>&1 && ! "${KOTEKAN_BIN}" --help >/dev/null 2>&1; then
    echo "[WARNING] Kotekan binary smoke test failed. Inspecting dependencies with ldd:"
    ldd "${KOTEKAN_BIN}" 2>&1 | head -n 35 || true
    echo ">>> Direct execution attempt output:"
    "${KOTEKAN_BIN}" --version || true
    exit 1
else
    echo ">>> Kotekan binary smoke test passed successfully."
fi

# 6. Validate the benchmark config before running
echo ">>> Validating tri-engine benchmark config: ${CONFIG}"
"${KOTEKAN_BIN}" --check-config "${CONFIG}"

# 7. Run the tri-engine benchmark pipeline
echo "======================================================================"
echo " Running CHARTS 64-Antenna Tri-Engine Benchmark Pipeline"
echo " Config : ${CONFIG}"
echo "======================================================================"
mkdir -p /tmp/test_charts_tri_engine_${SLURM_JOB_ID:-manual}
LD_LIBRARY_PATH="${BUILD_DIR}/external/n2k:${BUILD_DIR}/lib:${BUILD_DIR}/kotekan:${LD_LIBRARY_PATH:-}" \
    "${KOTEKAN_BIN}" --config "${CONFIG}"

echo "Tri-engine benchmark pipeline execution completed successfully."

# 8. Run the theoretical metrics benchmark (Python validation)
echo "======================================================================"
echo " Running Tri-Engine Theoretical Metrics Benchmark"
echo "======================================================================"
python3 "${REPO_ROOT}/test_charts/benchmark_tri_engine.py"

echo "======================================================================"
echo " All Tri-Engine Benchmark Tests Completed Successfully!"
echo "======================================================================"
