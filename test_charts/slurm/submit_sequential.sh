#!/bin/bash
# ==============================================================================
# Submit CHARTS Direct Beam Tracker and Comparison Benchmarks Sequentially
# Uses SLURM Job Dependencies (--dependency=afterok) to prevent resource overlap
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

ACCOUNT_FLAG=""
if [ -n "${SLURM_ACCOUNT:-}" ]; then
    ACCOUNT_FLAG="--account=${SLURM_ACCOUNT}"
elif [ -n "${1:-}" ]; then
    ACCOUNT_FLAG="--account=$1"
fi

echo "======================================================================"
echo " Submitting Benchmarks Sequentially via SLURM Dependencies"
echo " Working Directory: ${REPO_ROOT}"
[ -n "${ACCOUNT_FLAG}" ] && echo " Allocation Account: ${ACCOUNT_FLAG}"
echo "======================================================================"

# 1. Submit Direct Beam Tracker test & benchmark job
JOB1=$(sbatch --parsable ${ACCOUNT_FLAG} test_charts/slurm/submit_direct_beam_tracker_benchmark.slurm)
echo "✓ [1/2] Submitted Direct Beam Tracker Job: ID ${JOB1}"

# 2. Submit Head-to-Head Comparison benchmark (only executes after Job 1 succeeds)
JOB2=$(sbatch --parsable --dependency=afterok:${JOB1} ${ACCOUNT_FLAG} test_charts/slurm/submit_direct_vs_v5_benchmark_comparison.slurm)
echo "✓ [2/2] Queued Direct vs V5 Comparison Job: ID ${JOB2} (Dependency: afterok:${JOB1})"

echo ""
echo "======================================================================"
echo " Jobs queued successfully without resource overlap!"
echo " Monitor live queue: squeue -u \$USER"
echo " Cancel pipeline   : scancel ${JOB1} ${JOB2}"
echo "======================================================================"
