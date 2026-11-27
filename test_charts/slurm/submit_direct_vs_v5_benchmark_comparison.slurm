#!/bin/bash
#SBATCH --job-name=direct_vs_v5_tracker
#SBATCH --account=def-vanderli
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=slurm_direct_vs_v5_%j.out
#SBATCH --error=slurm_direct_vs_v5_%j.err

set -euo pipefail

echo "======================================================================"
echo " Starting Head-to-Head Benchmark: Direct Beam Tracker vs Beam Tracker V5"
echo " Hostname       : $(hostname)"
echo " CUDA Device    : ${CUDA_VISIBLE_DEVICES:-Not set (default 0)}"
echo " Job ID         : ${SLURM_JOB_ID:-manual}"
echo " Date           : $(date)"
echo "======================================================================"

# 1. Load environment modules (Compute Canada / Trillium / standard HPC cluster)
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

# Setup scratch python environment inheriting cluster modules
VENV_DIR="${SLURM_TMPDIR:-/tmp}/kotekan_pyenv_${SLURM_JOB_ID:-manual}"
if [ ! -d "${VENV_DIR}" ]; then
    python3 -m venv --system-site-packages "${VENV_DIR}" || true
fi
if [ -f "${VENV_DIR}/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
fi
pip install --no-index pyyaml jinja2 future 2>/dev/null || true

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

BUILD_TYPE="${BUILD_TYPE:-Release}"
CUDA_ARCH="${CUDA_ARCH:-native}"
BUILD_DIR="${REPO_ROOT}/build"
KOTEKAN_BIN="${BUILD_DIR}/kotekan/kotekan"
BENCHMARK_BIN="${BUILD_DIR}/test_charts/benchmark_tracker_v5_vs_direct"
BUILD_KOTEKAN="${BUILD_KOTEKAN:-1}"

echo "Repository Root : ${REPO_ROOT}"
echo "Build Directory : ${BUILD_DIR}"
echo "CUDA Arch       : ${CUDA_ARCH}"

# 3. Build Kotekan and Comparison Benchmark target
if [ "${BUILD_KOTEKAN}" = "1" ] || [ ! -x "${KOTEKAN_BIN}" ] || [ ! -x "${BENCHMARK_BIN}" ]; then
    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"

    echo ">>> Configuring CMake with CUDA enabled..."
    cmake .. \
        -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
        -DUSE_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
        -DUSE_NUMA=ON \
        -DWERROR=OFF \
        -DCMAKE_SKIP_RPATH=TRUE

    echo ">>> Compiling Head-to-Head Comparative Benchmark..."
    make -j"${SLURM_CPUS_PER_TASK:-16}" benchmark_tracker_v5_vs_direct kotekan

    # Apply Alliance setrpaths.sh if available
    if command -v setrpaths.sh >/dev/null 2>&1; then
        echo "Applying Alliance setrpaths.sh to build tree..."
        setrpaths.sh --path "${BUILD_DIR}" --add_origin 2>/dev/null || true
    fi

    cd "${REPO_ROOT}"
fi

export LD_LIBRARY_PATH="${BUILD_DIR}/external/n2k:${BUILD_DIR}/lib:${BUILD_DIR}/kotekan:${LD_LIBRARY_PATH:-}"

# 4. Run Head-to-Head Comparative Benchmark
echo "======================================================================"
echo " Running Comparative Benchmark Matrix (32, 64, 128, 256 Antennas x 1, 2, 4 Beams)"
echo "======================================================================"
"${BENCHMARK_BIN}"

# 5. Generate Analysis and Markdown Summary
if [ -f "tracker_benchmark_comparison.json" ]; then
    echo "======================================================================"
    echo " Generating Speedup Analysis & Markdown Summary"
    echo "======================================================================"
    python3 test_charts/compare_tracker_benchmarks.py tracker_benchmark_comparison.json \
        --md-out "tracker_benchmark_report_${SLURM_JOB_ID:-manual}.md"
fi

echo "======================================================================"
echo " Direct vs V5 Comparative Benchmark Completed Successfully!"
echo "======================================================================"
