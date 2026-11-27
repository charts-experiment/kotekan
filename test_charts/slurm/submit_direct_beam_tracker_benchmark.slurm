#!/bin/bash
#SBATCH --job-name=direct_beam_tracker
#SBATCH --account=def-vanderli
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=slurm_direct_tracker_%j.out
#SBATCH --error=slurm_direct_tracker_%j.err

set -euo pipefail

echo "======================================================================"
echo " Starting CHARTS Direct Beam Tracker Test Suite & Benchmark on Slurm"
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
pip install --no-index pyyaml jinja2 future h5py 2>/dev/null || true

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
BENCHMARK_BIN="${BUILD_DIR}/test_charts/benchmark_direct_beam_tracker"
MASKING_BENCH_BIN="${BUILD_DIR}/test_charts/benchmark_pipeline_masking_direct"
BUILD_KOTEKAN="${BUILD_KOTEKAN:-1}"
CONFIG="${CONFIG:-test_charts/config/64antennas_direct_tracker_test.yaml}"

echo "Repository Root : ${REPO_ROOT}"
echo "Build Directory : ${BUILD_DIR}"
echo "CUDA Arch       : ${CUDA_ARCH}"

# 3. Build Kotekan and Direct Beam Tracker benchmark targets
if [ "${BUILD_KOTEKAN}" = "1" ] || [ ! -x "${KOTEKAN_BIN}" ] || [ ! -x "${BENCHMARK_BIN}" ] || [ ! -x "${MASKING_BENCH_BIN}" ]; then
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

    echo ">>> Compiling Direct Beam Tracker benchmark targets..."
    make -j"${SLURM_CPUS_PER_TASK:-16}" kotekan benchmark_direct_beam_tracker benchmark_pipeline_masking_direct

    # Apply Alliance setrpaths.sh if available
    if command -v setrpaths.sh >/dev/null 2>&1; then
        echo "Applying Alliance setrpaths.sh to build tree..."
        setrpaths.sh --path "${BUILD_DIR}" --add_origin 2>/dev/null || true
    fi

    cd "${REPO_ROOT}"
fi

export LD_LIBRARY_PATH="${BUILD_DIR}/external/n2k:${BUILD_DIR}/lib:${BUILD_DIR}/kotekan:${LD_LIBRARY_PATH:-}"

# 4. Run Standalone Direct Beam Tracker & Masking CUDA Benchmarks
echo "======================================================================"
echo " 1. Running Direct Beam Tracker (Zero Integration Window) Benchmarks"
echo "======================================================================"
"${BENCHMARK_BIN}" --json-output "${REPO_ROOT}/direct_tracker_benchmark_${SLURM_JOB_ID:-manual}.json"

echo ">>> Running Masking + Direct Tracker Pipeline Benchmark..."
"${MASKING_BENCH_BIN}"

# 5. Run F-Engine Simulation Test for Direct Beam Tracker (Python validation)
echo "======================================================================"
echo " 2. Running F-Engine Simulation & Direct Beam Tracker Tests (64 & 256 Antennas)"
echo "======================================================================"
mkdir -p test_charts/data
python3 test_charts/generate_fengine_sim_data.py --antennas 64 --scenario zenith --output test_charts/data/voltage_sim_64ant.h5
python3 test_charts/generate_fengine_sim_data.py --antennas 256 --scenario zenith --output test_charts/data/voltage_sim_256ant.h5

python3 test_charts/test_fengine_direct_tracker.py test_charts/data/voltage_sim_64ant.h5
python3 test_charts/test_fengine_direct_tracker.py test_charts/data/voltage_sim_256ant.h5

# 6. Validate and Run Live Streaming Pipeline with Direct Beam Tracker
echo "======================================================================"
echo " 3. Running Kotekan Streaming Pipeline with Direct Beam Tracker"
echo "======================================================================"
"${KOTEKAN_BIN}" --check-config -c "${CONFIG}"
mkdir -p /tmp/test_charts_direct_tracker_64
"${KOTEKAN_BIN}" -c "${CONFIG}"
echo "Direct Beam Tracker streaming pipeline execution completed successfully."

echo "======================================================================"
echo " All Direct Beam Tracker Tests & Benchmarks Completed Successfully!"
echo "======================================================================"
