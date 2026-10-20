#!/bin/bash
#SBATCH --job-name=direct_beam_tracker
#SBATCH --output=slurm_direct_tracker_%j.out
#SBATCH --error=slurm_direct_tracker_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --partition=gpu

set -euo pipefail

echo "======================================================================"
echo " Starting CHARTS Direct Beam Tracker Test Suite & Benchmark on Slurm"
echo " Hostname       : $(hostname)"
echo " CUDA Device    : ${CUDA_VISIBLE_DEVICES:-Not set (default 0)}"
echo " Date           : $(date)"
echo "======================================================================"

# 1. Load environment modules (Compute Canada / Trillium / standard HPC cluster)
if command -v module >/dev/null 2>&1; then
    module load cuda/12.6 2>/dev/null || module load cuda/13.3 2>/dev/null || module load cuda || true
    module load gcc/12.3 2>/dev/null || module load gcc || true
    module load cmake 2>/dev/null || true
    module load python/3.11 2>/dev/null || module load python/3.10 2>/dev/null || module load python || true
    module load scipy-stack 2>/dev/null || true
    module load hdf5 2>/dev/null || true
fi

# Setup scratch python environment inheriting cluster modules
VENV_DIR="${SLURM_TMPDIR:-/tmp}/kotekan_pyenv_${SLURM_JOB_ID:-manual}"
if [ ! -d "${VENV_DIR}" ]; then
    python3 -m venv --system-site-packages "${VENV_DIR}" || true
fi
if [ -f "${VENV_DIR}/bin/activate" ]; then
    source "${VENV_DIR}/bin/activate"
fi
pip install --no-index pyyaml jinja2 future 2>/dev/null || true

# Print GPU architecture & specifications
nvidia-smi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUILD_DIR="${REPO_ROOT}/build"

echo "Repository Root : ${REPO_ROOT}"
echo "Build Directory : ${BUILD_DIR}"
cd "${REPO_ROOT}"

# 2. Build Kotekan and Direct Beam Tracker targets
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

echo ">>> Configuring CMake with CUDA enabled..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="80;89;90;100;120" \
    -DUSE_NUMA=ON \
    -DWERROR=OFF \
    -DCMAKE_SKIP_RPATH=TRUE

echo ">>> Compiling Direct Beam Tracker benchmark targets..."
make -j${SLURM_CPUS_PER_TASK:-8} kotekan benchmark_direct_beam_tracker || true

# Apply Alliance setrpaths.sh if available
if command -v setrpaths.sh >/dev/null 2>&1; then
    echo "Applying Alliance setrpaths.sh to build tree..."
    setrpaths.sh --path "${BUILD_DIR}" --add_origin 2>/dev/null || true
fi

cd "${REPO_ROOT}"

# Standalone compilation via NVCC directly
echo ">>> Building standalone benchmark_direct_beam_tracker via NVCC..."
nvcc -std=c++17 -O3 --use_fast_math \
    -arch=native \
    -Xcompiler -fopenmp \
    -I lib/cuda -I lib/metadata -I lib/core -I lib/utils -I lib/stages -I lib/ \
    -I external/json -I external/fmt \
    test_charts/benchmark_direct_beam_tracker.cpp \
    lib/cuda/cudaDirectBeamTracker.cu \
    lib/core/kotekanLogging.cpp \
    lib/core/errors.c \
    -o benchmark_direct_beam_tracker

if [ -f "${REPO_ROOT}/benchmark_direct_beam_tracker" ]; then
    BENCHMARK_BIN="${REPO_ROOT}/benchmark_direct_beam_tracker"
else
    BENCHMARK_BIN="${BUILD_DIR}/test_charts/benchmark_direct_beam_tracker"
fi

# 3. Run Standalone Direct Beam Tracker CUDA Benchmarks (32, 64, 128, 256 antennas)
echo "======================================================================"
echo " 1. Running Direct Beam Tracker (Zero Integration Window) Benchmarks"
echo "======================================================================"
"${BENCHMARK_BIN}"

# 4. Run F-Engine Simulation Test for Direct Beam Tracker (Python validation)
echo "======================================================================"
echo " 2. Running F-Engine Simulation & Direct Beam Tracker Tests (64 & 256 Antennas)"
echo "======================================================================"
mkdir -p test_charts/data
python3 test_charts/generate_fengine_sim_data.py --antennas 64 --scenario zenith --output test_charts/data/voltage_sim_64ant.h5
python3 test_charts/generate_fengine_sim_data.py --antennas 256 --scenario zenith --output test_charts/data/voltage_sim_256ant.h5

python3 test_charts/test_fengine_direct_tracker.py test_charts/data/voltage_sim_64ant.h5
python3 test_charts/test_fengine_direct_tracker.py test_charts/data/voltage_sim_256ant.h5

# 5. Run Live Streaming Pipeline with Direct Beam Tracker
echo "======================================================================"
echo " 3. Running Kotekan Streaming Pipeline with Direct Beam Tracker"
echo "======================================================================"
if [ -f "${BUILD_DIR}/kotekan/kotekan" ]; then
    mkdir -p /tmp/test_charts_direct_tracker_64
    LD_LIBRARY_PATH="${BUILD_DIR}/external/n2k:${LD_LIBRARY_PATH:-}" \
        "${BUILD_DIR}/kotekan/kotekan" --config test_charts/config/64antennas_direct_tracker_test.yaml
    echo "Direct Beam Tracker streaming pipeline execution completed successfully."
fi

echo "======================================================================"
echo " All Direct Beam Tracker Tests & Benchmarks Completed Successfully!"
echo "======================================================================"
