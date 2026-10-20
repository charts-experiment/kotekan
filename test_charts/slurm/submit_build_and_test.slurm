#!/bin/bash
#SBATCH --job-name=kotekan_tracker_tests
#SBATCH --output=slurm_tracker_tests_%j.out
#SBATCH --error=slurm_tracker_tests_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --partition=gpu

set -euo pipefail

echo "======================================================================"
echo " Starting Kotekan Beam Tracker Compilation & Verification on Trillium"
echo " Hostname       : $(hostname)"
echo " CUDA Device    : ${CUDA_VISIBLE_DEVICES:-Not set (default 0)}"
echo " Date           : $(date)"
echo "======================================================================"

# 1. Load required modules on Trillium / Compute Canada (adjust version names if needed)
if command -v module >/dev/null 2>&1; then
    module load cuda/13.3 || module load cuda/12.6 || module load cuda || true
    module load gcc/12.3 || module load gcc || true
    module load cmake || true
    module load python/3.10 || module load python || true
fi

# Print GPU info
nvidia-smi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUILD_DIR="${REPO_ROOT}/build"

echo "Repository Root : ${REPO_ROOT}"
echo "Build Directory : ${BUILD_DIR}"

# 2. Build Kotekan with CUDA enabled
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

echo ">>> Configuring CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="80;90" \
    -DCMAKE_SKIP_RPATH=TRUE

echo ">>> Compiling Kotekan & Test Targets..."
make -j${SLURM_CPUS_PER_TASK:-8} kotekan test_beam_tracker_suite test_chaos_engineering benchmark_kotekan_cuda_tracker_v5 benchmark_pipeline_stages benchmark_direct_beam_tracker benchmark_tracker_v5_vs_direct

# Apply Alliance setrpaths.sh if available
if command -v setrpaths.sh >/dev/null 2>&1; then
    echo "Applying Alliance setrpaths.sh to build tree..."
    setrpaths.sh --path "${BUILD_DIR}" --add_origin 2>/dev/null || true
fi

echo "======================================================================"
echo " 1. Running Comprehensive Beam Tracker Test Suite (64, 128, 256 Ants)"
echo "======================================================================"
./test_charts/test_beam_tracker_suite

echo "======================================================================"
echo " 2. Running Chaos Engineering & Fault Injection Test Suite"
echo "======================================================================"
./test_charts/test_chaos_engineering

echo "======================================================================"
echo " 3. Running Kotekan CUDA V5 Complex Voltage Benchmarks"
echo "======================================================================"
./test_charts/benchmark_kotekan_cuda_tracker_v5

echo "======================================================================"
echo " 4. Running End-to-End Pipeline Stage Latency Benchmarks"
echo "======================================================================"
./test_charts/benchmark_pipeline_stages

echo "======================================================================"
echo " 5. Running Direct Beam Tracker (Zero-Window) Benchmarks"
echo "======================================================================"
./test_charts/benchmark_direct_beam_tracker

echo "======================================================================"
echo " 6. Running Head-to-Head Comparative Benchmark (Direct vs V5)"
echo "======================================================================"
./test_charts/benchmark_tracker_v5_vs_direct

echo "======================================================================"
echo " All Trillium Batch Jobs & Verification Steps Completed Successfully!"
echo "======================================================================"
