#!/usr/bin/env bash
# ==============================================================================
# CHARTS Kotekan Continuous Beam Tracker Pipeline & REST Server Launcher
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KOTEKAN_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
KOTEKAN_BIN="${KOTEKAN_ROOT}/build/kotekan/kotekan"
if [ -z "${PYTHON_BIN:-}" ]; then
    if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
        PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
    elif [ -f "${KOTEKAN_ROOT}/.venv/bin/python" ]; then
        PYTHON_BIN="${KOTEKAN_ROOT}/.venv/bin/python"
    elif [ -f "${KOTEKAN_ROOT}/.venv/Scripts/python.exe" ]; then
        PYTHON_BIN="${KOTEKAN_ROOT}/.venv/Scripts/python.exe"
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python3)"
    elif command -v python >/dev/null 2>&1; then
        PYTHON_BIN="$(command -v python)"
    else
        PYTHON_BIN="python"
    fi
fi

PORT="${KOTEKAN_REST_PORT:-12048}"
WORK_DIR="/tmp/kotekan_continuous_tracker"
INPUT_DIR="${WORK_DIR}/input"
CONFIG_FILE="${WORK_DIR}/continuous_tracker.yaml"
PID_FILE="${WORK_DIR}/kotekan.pid"
LOG_FILE="${WORK_DIR}/kotekan.log"

ANTENNAS="${KOTEKAN_ANTENNAS:-64}"
CHANNELS="${KOTEKAN_CHANNELS:-16}"
TIME_SAMPLES="${KOTEKAN_TIME_SAMPLES:-3840}"
INTEGRATION_SPECTRA="${KOTEKAN_INTEGRATION_SPECTRA:-320}"
BEAMS="${KOTEKAN_BEAMS:-4}"
BUFFER_DEPTH="${KOTEKAN_BUFFER_DEPTH:-3}"

mkdir -p "${INPUT_DIR}"

echo "================================================================================"
echo " CHARTS KOTEKAN CONTINUOUS BEAM TRACKER PIPELINE LAUNCHER"
echo "================================================================================"

# Check if already running
if [ -f "${PID_FILE}" ]; then
    PID=$(cat "${PID_FILE}")
    if kill -0 "${PID}" 2>/dev/null; then
        echo "[INFO] Kotekan is already running with PID: ${PID}"
        echo "  -> REST Server : http://127.0.0.1:${PORT}"
        echo "  -> Stop command: ${SCRIPT_DIR}/stop_continuous_pipeline.sh"
        exit 0
    else
        rm -f "${PID_FILE}"
    fi
fi

# 1. Generate Continuous Baseband Input Buffer if missing
INPUT_FILE="${INPUT_DIR}/window_replay_0000000.bin"
if [ ! -f "${INPUT_FILE}" ]; then
    echo "[1/3] Generating Continuous Replay Baseband Buffer (${ANTENNAS} Ant x ${CHANNELS} Freq x ${TIME_SAMPLES} Time)..."
    "${PYTHON_BIN}" -c "
import struct, numpy as np
n_ant, n_freq, n_time = ${ANTENNAS}, ${CHANNELS}, ${TIME_SAMPLES}
rng = np.random.default_rng(42)
r = rng.integers(-4, 4, size=(n_time, n_freq, n_ant), dtype=np.int8)
i = rng.integers(-4, 4, size=(n_time, n_freq, n_ant), dtype=np.int8)
packed = ((r & 0x0F) | ((i & 0x0F) << 4)).astype(np.uint8)
with open('${INPUT_FILE}', 'wb') as f:
    f.write(struct.pack('<I', 0))
    f.write(packed.tobytes())
print('  -> Wrote ' + str(packed.nbytes / 1e6) + ' MB baseband replay buffer.')
"
fi

# 2. Write Continuous Kotekan Configuration
echo "[2/3] Generating Continuous Pipeline Configuration: ${CONFIG_FILE}..."
cat <<EOF > "${CONFIG_FILE}"
type: config
log_level: error

cpu_affinity: [0, 1]

num_elements: ${ANTENNAS}
num_local_freq: ${CHANNELS}
samples_per_data_set: ${TIME_SAMPLES}
integration_spectra: ${INTEGRATION_SPECTRA}
spacing_m: 0.6
max_beams: 8
initial_active_beams: ${BEAMS}
buffer_depth: ${BUFFER_DEPTH}
sizeof_complex_float: 8

main_pool:
  kotekan_metadata_pool: chordMetadata
  num_metadata_objects: 30

network_capture_buf:
  kotekan_buffer: standard
  num_frames: buffer_depth
  frame_size: samples_per_data_set * num_local_freq * num_elements
  numa_node: 0
  metadata_pool: main_pool
  zero_new_frames: true
  mlock_frames: false

host_formed_beams_buffer:
  kotekan_buffer: standard
  num_frames: buffer_depth
  frame_size: samples_per_data_set * num_local_freq * max_beams * sizeof_complex_float
  numa_node: 0
  metadata_pool: main_pool
  zero_new_frames: true
  mlock_frames: false

reader:
  kotekan_stage: rawFileRead
  base_dir: "${INPUT_DIR}"
  file_name: "window_replay"
  file_ext: "bin"
  prefix_hostname: false
  loop_files: true
  max_repeats: -1
  end_interrupt: false
  buf: network_capture_buf

gpu:
  profiling: false
  commands: &command_list
    - name: cudaInputData
      in_buf: host_voltage
      gpu_mem: voltage
    - name: cudaSyncInput
    - name: cudaBeamTrackerCommand
      gpu_mem_voltage: voltage
      gpu_mem_formed_beams: formed_beams
      num_elements: ${ANTENNAS}
      num_local_freq: ${CHANNELS}
      samples_per_data_set: ${TIME_SAMPLES}
      integration_spectra: ${INTEGRATION_SPECTRA}
      spacing_m: 0.6
      max_beams: 8
      initial_active_beams: ${BEAMS}
      source_l0: 0.05
      source_m0: -0.02
      source_dl: 1.0e-5
      source_dm: 0.0
    - name: cudaSyncOutput
    - name: cudaOutputData
      in_buf: host_voltage
      gpu_mem: formed_beams
      out_buf: host_formed_beams
  gpu_0:
    kotekan_stage: cudaProcess
    gpu_id: 0
    commands: *command_list
    in_buffers:
      host_voltage: network_capture_buf
    out_buffers:
      host_formed_beams: host_formed_beams_buffer

sink:
  kotekan_stage: dropAllFrames
  in_buf: host_formed_beams_buffer
EOF

# 3. Launch Kotekan in background
echo "[3/3] Launching Kotekan (PID stored in ${PID_FILE})..."
nohup "${KOTEKAN_BIN}" -c "${CONFIG_FILE}" -b "127.0.0.1:${PORT}" > "${LOG_FILE}" 2>&1 &
PID=$!
echo "${PID}" > "${PID_FILE}"

sleep 0.8
if kill -0 "${PID}" 2>/dev/null; then
    echo "================================================================================"
    echo " [SUCCESS] Kotekan Continuous Beam Tracker Pipeline is RUNNING!"
    echo "================================================================================"
    echo " Process PID       : ${PID}"
    echo " REST API Server   : http://127.0.0.1:${PORT}"
    echo " Log File          : ${LOG_FILE}"
    echo "--------------------------------------------------------------------------------"
    echo " Quick Bash Controls:"
    echo "   Status Query    : ${PYTHON_BIN} ${SCRIPT_DIR}/kotekan_tracker_control.py status"
    echo "   Live Watch CLI  : ${PYTHON_BIN} ${SCRIPT_DIR}/kotekan_tracker_control.py watch"
    echo "   Steer Beam 0    : ${PYTHON_BIN} ${SCRIPT_DIR}/kotekan_tracker_control.py steer-lm --beam 0 --l0 0.1 --m0 -0.2"
    echo "   Launch Plotly UI: ${PYTHON_BIN} ${SCRIPT_DIR}/kotekan_tracker_dashboard.py"
    echo "   Stop Pipeline   : ${SCRIPT_DIR}/stop_continuous_pipeline.sh"
    echo "================================================================================"
else
    echo "[ERROR] Kotekan failed to start. Showing last 20 lines of log:"
    tail -n 20 "${LOG_FILE}"
    exit 1
fi
