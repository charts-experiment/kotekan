#!/bin/bash
#SBATCH --job-name=charts_tri_engine
#SBATCH --account=def-vanderli
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
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
# Available Configurations:
#   - Single-subband synthetic benchmark (default):
#       test_charts/config/64antennas_tri_engine_benchmark.yaml
#   - Dual-subband synthetic benchmark (672 channels):
#       charts/config/64antennas_tri_engine_benchmark.yaml
#   - Dual-subband DPDK production config:
#       charts/config/64antennas_tri_engine_dpdk.yaml
#
# Ingestion: In-process Synthetic Astrophysical Baseband Generator
#            (chartsFEngineSim)
# Zero-Disk Mode: Formed beams stay in VRAM; only correlation visibilities
#                 and FRB intensity are transferred D2H for REST inspection.
#
# Usage:
#   sbatch test_charts/slurm/submit_tri_engine_benchmark.slurm
#
#   # Run dual-subband synthetic benchmark or DPDK config override:
#   sbatch --export=ALL,CONFIG=charts/config/64antennas_tri_engine_benchmark.yaml \
#     test_charts/slurm/submit_tri_engine_benchmark.slurm
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
    module load gcc/12.3 2>/dev/null || module load gcc 2>/dev/null || true
    module load cuda/12.6 2>/dev/null || module load cuda/13.3 2>/dev/null || module load cuda 2>/dev/null || true
    module load cmake 2>/dev/null || true
    module load boost 2>/dev/null || true
    module load python/3.11 2>/dev/null || module load python/3.10 2>/dev/null || module load python 2>/dev/null || true
    module load scipy-stack 2>/dev/null || true
    module load hdf5 2>/dev/null || true
    module load ffmpeg 2>/dev/null || true
fi

# Print GPU architecture & specifications
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

echo "Repository Root : ${REPO_ROOT}"

# 3. Configuration knobs (overridable via --export at sbatch time)
BUILD_TYPE="${BUILD_TYPE:-Release}"
CUDA_ARCH="${CUDA_ARCH:-native}"
CONFIG="${CONFIG:-test_charts/config/64antennas_tri_engine_benchmark.yaml}"
BUILD_KOTEKAN="${BUILD_KOTEKAN:-1}"
BUILD_DIR="${REPO_ROOT}/build"
KOTEKAN_BIN="${BUILD_DIR}/kotekan/kotekan"

# 4. Build Kotekan with CUDA enabled
if [ "${BUILD_KOTEKAN}" = "1" ] || [ ! -x "${KOTEKAN_BIN}" ]; then
    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"

    echo ">>> Configuring CMake (BUILD_TYPE=${BUILD_TYPE}, CUDA_ARCH=${CUDA_ARCH})..."
    cmake .. \
        -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
        -DUSE_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
        -DUSE_NUMA=ON \
        -DWERROR=OFF \
        -DCMAKE_SKIP_RPATH=TRUE

    echo ">>> Compiling Kotekan..."
    make -j"${SLURM_CPUS_PER_TASK:-16}" kotekan

    # Apply Alliance setrpaths.sh if available
    if command -v setrpaths.sh >/dev/null 2>&1; then
        echo "Applying Alliance setrpaths.sh to build tree..."
        setrpaths.sh --path "${BUILD_DIR}" --add_origin 2>/dev/null || true
    fi

    cd "${REPO_ROOT}"
fi

export LD_LIBRARY_PATH="${BUILD_DIR}/external/n2k:${BUILD_DIR}/lib:${BUILD_DIR}/kotekan:${LD_LIBRARY_PATH:-}"

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
"${KOTEKAN_BIN}" --check-config -c "${CONFIG}"

# 7. Run the tri-engine benchmark pipeline
echo "======================================================================"
echo " Running CHARTS 64-Antenna Tri-Engine Benchmark Pipeline"
echo " Config : ${CONFIG}"
echo "======================================================================"
mkdir -p "/tmp/test_charts_tri_engine_${SLURM_JOB_ID:-manual}"
"${KOTEKAN_BIN}" --config "${CONFIG}"

echo "Tri-engine benchmark pipeline execution completed successfully."

# 8. Print the theoretical metrics audit (without re-running Kotekan)
echo "======================================================================"
echo " Tri-Engine Theoretical Metrics Audit"
echo "======================================================================"
python3 -c 'import sys; sys.path.insert(0, "test_charts"); import benchmark_tri_engine as b; b.print_theoretical_audit(b.calculate_theoretical_metrics())'

echo "======================================================================"
echo " All Tri-Engine Benchmark Tests Completed Successfully!"
echo "======================================================================"
