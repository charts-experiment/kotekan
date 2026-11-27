#!/bin/bash
#SBATCH --job-name=kotekan_tracker_tests
#SBATCH --account=def-vanderli
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=slurm_tracker_tests_%j.out
#SBATCH --error=slurm_tracker_tests_%j.err

set -euo pipefail

echo "======================================================================"
echo " Starting Kotekan CHARTS Compilation & Verification on Trillium"
echo " Hostname       : $(hostname)"
echo " CUDA Device    : ${CUDA_VISIBLE_DEVICES:-Not set (default 0)}"
echo " Job ID         : ${SLURM_JOB_ID:-manual}"
echo " Date           : $(date)"
echo "======================================================================"

# 1. Load required modules on Trillium / Compute Canada
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

# Print GPU info
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
RUN_BENCHMARKS="${RUN_BENCHMARKS:-1}"
RUN_PYTHON_TESTS="${RUN_PYTHON_TESTS:-1}"

echo "Repository Root : ${REPO_ROOT}"
echo "Build Directory : ${BUILD_DIR}"
echo "CUDA Arch       : ${CUDA_ARCH}"

# Setup scratch python environment inheriting cluster modules
VENV_DIR="${SLURM_TMPDIR:-/tmp}/kotekan_pyenv_${SLURM_JOB_ID:-manual}"
if [ ! -d "${VENV_DIR}" ]; then
    python3 -m venv --system-site-packages "${VENV_DIR}" || true
fi
if [ -f "${VENV_DIR}/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
fi
pip install --no-index pyyaml jinja2 future pytest 2>/dev/null || true

# 3. Build Kotekan and all test_charts targets with CUDA enabled
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

TEST_TARGETS=(
    test_beam_tracker_suite
    test_chaos_engineering
    test_antenna_masking_stage
    test_bislc_suite
    test_phase_interpolation_suite
    test_transient_trigger_suite
    test_chained_mask_tracker_mvdr_suite
    test_phase1_gains_loader
    test_rfi_mitigation_stage
    test_pirate_beam_quantize_suite
    test_wave3_gpu_scaling_suite
    test_beam_upchannelize_suite
    test_transient_trigger_multibeam_suite
)

BENCHMARK_TARGETS=(
    benchmark_kotekan_cuda_tracker_v5
    benchmark_pipeline_stages
    benchmark_direct_beam_tracker
    benchmark_tracker_v5_vs_direct
    benchmark_pipeline_masking_direct
)

echo ">>> Compiling Kotekan & all 18 test_charts C++/CUDA targets..."
make -j"${SLURM_CPUS_PER_TASK:-16}" kotekan "${TEST_TARGETS[@]}" "${BENCHMARK_TARGETS[@]}"

# Apply Alliance setrpaths.sh if available
if command -v setrpaths.sh >/dev/null 2>&1; then
    echo "Applying Alliance setrpaths.sh to build tree..."
    setrpaths.sh --path "${BUILD_DIR}" --add_origin 2>/dev/null || true
fi

# Run all verification from repository root so relative asset paths resolve
cd "${REPO_ROOT}"
export LD_LIBRARY_PATH="${BUILD_DIR}/external/n2k:${BUILD_DIR}/lib:${BUILD_DIR}/kotekan:${LD_LIBRARY_PATH:-}"

echo "======================================================================"
echo " 1. Validating CHARTS YAML Configurations"
echo "======================================================================"
python3 charts/ci/validate_charts_configs.py --self-test
python3 charts/ci/validate_charts_configs.py
bash charts/ci/check_configs.sh "${KOTEKAN_BIN}"

echo "======================================================================"
echo " 2. Running All 13 C++/CUDA Test Suites"
echo "======================================================================"
for target in "${TEST_TARGETS[@]}"; do
    echo ">>> Running ${target}..."
    "${BUILD_DIR}/test_charts/${target}"
done

if [ "${RUN_BENCHMARKS}" = "1" ]; then
    echo "======================================================================"
    echo " 3. Running All 5 C++/CUDA Benchmarks"
    echo "======================================================================"
    for target in "${BENCHMARK_TARGETS[@]}"; do
        echo ">>> Running ${target}..."
        "${BUILD_DIR}/test_charts/${target}"
    done
else
    echo ">>> Skipping C++/CUDA benchmarks (RUN_BENCHMARKS=${RUN_BENCHMARKS})."
fi

if [ "${RUN_PYTHON_TESTS}" = "1" ]; then
    echo "======================================================================"
    echo " 4. Running CHARTS Python Test Suites"
    echo "======================================================================"
    export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/test_charts:${PYTHONPATH:-}"
    if python3 -c "import pytest" >/dev/null 2>&1; then
        python3 -m pytest -q test_charts/tests
    else
        python3 -m unittest discover -s test_charts/tests
    fi
else
    echo ">>> Skipping Python tests (RUN_PYTHON_TESTS=${RUN_PYTHON_TESTS})."
fi

echo "======================================================================"
echo " All Trillium Build & Verification Steps Completed Successfully!"
echo "======================================================================"
