#!/usr/bin/env bash
# ==============================================================================
# Direct Beam Tracker Deployment Runner — Grus Quartet (NGC 7582)
# Run A (no masking, raw 64-ant baseband + formed beams)
# Run B (cudaAntennaMaskCommand ON, 7-ant alive baseband + formed beams)
# ==============================================================================
set -uo pipefail

cd /home/fpalma/kotekan

mkdir -p /data/tracker/dump_A /data/tracker/dump_B
rm -f /data/tracker/dump_A/* /data/tracker/dump_B/* 2>/dev/null || true

echo "=== [1/4] Pre-flight System & Clock Check ==="
df -h /data/tracker
timedatectl status 2>/dev/null | head -n 8 || true
chronyc tracking 2>/dev/null | head -n 5 || true
nvidia-smi --query-gpu=name,driver_version,temperature.gpu,utilization.gpu,memory.used --format=csv
dpdk-devbind.py -s | head -n 10

echo ""
echo "=== [2/4] Caching sudo & Running Config Validation / Dry-Run (Run A & Run B) ==="
sudo -v
for r in A B; do
  echo "--- Checking config $r ---"
  build/kotekan/kotekan --check-config -c "charts/config/64antennas_direct_tracker_tonight_${r}.yaml"
  sudo rm -f /dev/hugepages/rtemap_* 2>/dev/null || true
  sudo build/kotekan/kotekan --dry-run -c "charts/config/64antennas_direct_tracker_tonight_${r}.yaml"
done

run_with_metrics() {
  local r=$1
  local out="/data/tracker/dump_${r}"
  echo ""
  echo "=============================================================================="
  echo " Launching Run ${r} -> ${out} at $(date -u '+%Y-%m-%d %H:%M:%S.%3N UTC') ($(TZ=America/Santiago date '+%H:%M:%S.%3N CLST'))"
  echo "=============================================================================="

  sudo rm -f /dev/hugepages/rtemap_* 2>/dev/null || true

  nvidia-smi dmon -s pucvmet -d 1 > "${out}/gpu_dmon.log" 2>&1 &
  local smi_pid=$!

  (
    while true; do
      ts=$(date +%s.%N)
      curl -sf http://127.0.0.1:12048/metrics > "${out}/metrics_${ts}.prom" 2>/dev/null || rm -f "${out}/metrics_${ts}.prom"
      curl -sf http://127.0.0.1:12048/buffers > "${out}/buffers_${ts}.json" 2>/dev/null || rm -f "${out}/buffers_${ts}.json"
      curl -sf http://127.0.0.1:12048/direct_tracker/status >> "${out}/direct_tracker_status.jsonl" 2>/dev/null && echo "" >> "${out}/direct_tracker_status.jsonl" || true
      curl -sf http://127.0.0.1:12048/antenna_mask/status >> "${out}/antenna_mask_status.jsonl" 2>/dev/null && echo "" >> "${out}/antenna_mask_status.jsonl" || true
      /home/fpalma/kotekan/.venv/bin/python test_charts/kotekan_tracker_control.py status >> "${out}/tracker_status.log" 2>/dev/null || true
      sleep 0.2
    done
  ) &
  local poll_pid=$!

  sudo build/kotekan/kotekan -c "charts/config/64antennas_direct_tracker_tonight_${r}.yaml" \
       -b 127.0.0.1:12048 2>&1 | tee "${out}/run.log"

  kill $poll_pid $smi_pid 2>/dev/null || true
  wait $poll_pid $smi_pid 2>/dev/null || true
  echo "Run ${r} finished at $(TZ=America/Santiago date '+%H:%M:%S.%3N CLST'). Files in ${out}:"
  ls -lh "${out}"
}

echo ""
echo "=== [3/4] Executing Run A and Run B ==="
# Cache sudo credential once before starting
sudo -v
run_with_metrics A
run_with_metrics B

echo ""
echo "=== [4/4] Running Posterior Benchmarking & Physics Analysis ==="
/home/fpalma/kotekan/.venv/bin/python charts/analyze_tonight_grus.py --dump-a /data/tracker/dump_A --dump-b /data/tracker/dump_B --out-dir /data/tracker/analysis_tonight
