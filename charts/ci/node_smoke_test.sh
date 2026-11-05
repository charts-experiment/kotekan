#!/bin/bash
# Smoke test of kotekan on the CHARTS GPU node with live RFSoC traffic.
#
# For each config: write a copy with every output path under OUTDIR/<config> and the
# antenna lists replaced, run --check-config, start kotekan, wait for the
# rfsocHandlerShuffle startup alignment, sample /metrics, run for -t seconds (triggering
# a baseband capture if the config has chartsTriggeredBasebandReadout), stop it and run
# the checks. Results go to stdout and OUTDIR/summary.md; exits non-zero if any check
# fails.
#
# Only one kotekan can own the NICs, so this refuses to start while another kotekan
# runs; it never stops one itself. kotekan is started from the repo root, which the
# configs' kernel_path is relative to.
#
# Usage: sudo charts/ci/node_smoke_test.sh [-t SECONDS] [-o OUTDIR] [-b KOTEKAN_BIN]
#            [-a ANTENNAS] [-k CAL_FILE] [-r REF_DIR] [-p PYTHON] [-w SECONDS] [config.yaml ...]
#   -t  run time per config after alignment (default 120)
#   -o  output directory (default /data/smoke_test/<date>_<commit>)
#   -b  kotekan binary (default <repo>/build/kotekan/kotekan)
#   -a  antennas: comma-separated list, or a config to take signal_antennas from
#       (default: the production config below, if it exists)
#   -k  calibration table for cudaVisibilityCalibrator, or 'none' for pass-through; also
#       adds the calibrator configs to the default list
#   -r  production output directory to compare autocorrelation levels against
#   -p  python with pyyaml, and charts-utils for checks 7-8 (default: charts-utils venv)
#   -w  alignment timeout in seconds from launch (default 180)

set -uo pipefail

repo=$(cd "$(dirname "$0")/../.." && pwd)
helper="$repo/charts/ci/node_smoke_check.py"
user_home=$(getent passwd "${SUDO_USER:-$USER}" | cut -d: -f6)
production_config="$user_home/kotekan_charts_next/charts/config/64antennas_corr.yaml"
rest="http://localhost:12048"
stop_grace=180
baseband_settle=15
noise_re='Invalid subband number|outside configured range|Invalid RFSoC id'
error_re='^(EMERGENCY|ALERT|CRITICAL|ERROR):|FATAL'

run_seconds=120
outdir=""
kotekan="$repo/build/kotekan/kotekan"
antennas=""
cal_file=""
reference=""
python="$user_home/charts-utils/.venv/bin/python"
align_timeout=180

usage() {
    sed -n '/^# Usage:/,/^$/s/^# \{0,1\}//p' "$0" >&2
    exit 2
}

while getopts "t:o:b:a:k:r:p:w:h" opt; do
    case $opt in
        t) run_seconds=$OPTARG ;;
        o) outdir=$OPTARG ;;
        b) kotekan=$OPTARG ;;
        a) antennas=$OPTARG ;;
        k) cal_file=$OPTARG ;;
        r) reference=$OPTARG ;;
        p) python=$OPTARG ;;
        w) align_timeout=$OPTARG ;;
        *) usage ;;
    esac
done
shift $((OPTIND - 1))

if [[ $# -gt 0 ]]; then
    configs=("$@")
else
    configs=("$repo"/charts/config/{64antennas_corr,64antennas_corr_triggered_baseband,64antennas_direct_tracker}.yaml)
    if [[ -n "$cal_file" ]]; then
        configs+=("$repo"/charts/config/{64antennas_corr_cal,64antennas_corr_trigger_cal}.yaml)
    fi
fi
if [[ ! -x "$python" ]]; then
    python=python3
fi
if [[ -z "$antennas" && -f "$production_config" ]]; then
    antennas=$production_config
fi
if [[ ! -x "$kotekan" ]]; then
    echo "error: kotekan binary not found or not executable: $kotekan" >&2
    exit 2
fi

# Packet capture keeps up with the RFSoC rate only with optimisation (-O3).
cmake_cache="$(dirname "$(dirname "$kotekan")")/CMakeCache.txt"
build_type=$(sed -n 's/^CMAKE_BUILD_TYPE:STRING=//p' "$cmake_cache" 2> /dev/null)
if [[ "$build_type" != Release ]]; then
    echo "warning: $kotekan is a '${build_type:-unknown}' build, not Release; expect packet loss" >&2
fi

# The commit tested is the one the binary was built from, which differs from this script's
# checkout when -b points to another build.
source_dir=$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "$cmake_cache" 2> /dev/null)
source_dir=${source_dir:-$repo}
git_repo=(git -c "safe.directory=$source_dir" -C "$source_dir")
commit=$("${git_repo[@]}" rev-parse --short HEAD)
if [[ -n $("${git_repo[@]}" status --porcelain --untracked-files=no) ]]; then
    commit="$commit-dirty"
fi
outdir=${outdir:-/data/smoke_test/$(date +%Y%m%d_%H%M%S)_$commit}

other_kotekan() {
    pgrep -x kotekan > /dev/null || curl -s -m 2 -o /dev/null "$rest/status"
}

if other_kotekan; then
    echo "error: another kotekan is running (or $rest answers); stop it first" >&2
    exit 2
fi
if [[ $EUID -ne 0 ]]; then
    echo "warning: not root; DPDK and hugepages usually need sudo" >&2
fi
mkdir -p "$outdir" || exit 2
# kotekan runs from the repo root, so every path it reads must be absolute.
outdir=$(realpath "$outdir")
if [[ -n "$cal_file" && "$cal_file" != none ]]; then
    cal_file=$(realpath -m "$cal_file")
fi

pid=""
summary_antennas=""
stop_kotekan() {
    [[ -n "$pid" ]] || return 0
    curl -s -m "$stop_grace" -o /dev/null "$rest/stop"
    kill -TERM "$pid" 2> /dev/null
    for _ in $(seq "$stop_grace"); do
        kill -0 "$pid" 2> /dev/null || break
        sleep 1
    done
    if kill -0 "$pid" 2> /dev/null; then
        echo "warning: kotekan $pid did not exit after ${stop_grace}s, killing it" >&2
        kill -KILL "$pid" 2> /dev/null
    fi
    wait "$pid" 2> /dev/null
    exit_code=$?
    pid=""
}

cleanup() {
    stop_kotekan
    if [[ -n "${SUDO_USER:-}" ]]; then
        chown -R "$SUDO_USER": "$outdir" 2> /dev/null
    fi
}
trap cleanup EXIT
trap 'echo "interrupted" >&2; exit 130' INT TERM

# Results, one "<config>\t<check>\t<status>\t<detail>" line each.
results="$outdir/results.tsv"
: > "$results"

record() {
    printf '%s\t%s\t%s\t%s\n' "$name" "$1" "$2" "$3" | tee -a "$results"
}

# Read the "<check>\t<status>\t<detail>" lines of the helper into the results.
record_helper() {
    local check status detail
    while IFS=$'\t' read -r check status detail; do
        record "$check" "$status" "$detail"
    done
}

# Start line of each rfsocHandlerShuffle port: "locked to common start spectrum S" since the
# shared start-up alignment, "selected start spectrum S" before it.
start_re='Port [0-9]+ received spectrum [0-9]+, (locked to common|selected) start spectrum [0-9]+'

wait_for_alignment() {
    local start=$SECONDS started_ports
    while ((SECONDS - start < align_timeout)); do
        if ! kill -0 "$pid" 2> /dev/null; then
            return 1
        fi
        started_ports=$(grep -oE "$start_re" "$log" | awk '{print $2}' | sort -u | wc -l)
        if ((started_ports >= handlers)); then
            align_seconds=$((SECONDS - start))
            return 0
        fi
        sleep 1
    done
    return 1
}

check_alignment() {
    local lines spectrum others
    lines=$(grep -oE "$start_re" "$log")
    spectrum=$(head -n 1 <<< "$lines" | awk '{print $NF}')
    others=$(awk -v s="$spectrum" '$NF != s' <<< "$lines" | wc -l)
    if ((others > 0)); then
        record 1 FAIL "ports started at different spectra: $(tr '\n' ';' <<< "$lines")"
    else
        record 1 PASS "all $handlers ports started at spectrum $spectrum after ${align_seconds}s"
    fi
}

check_baseband() {
    local event_id=$1 dir sizes
    dir="$data_dir/baseband_raw_$event_id"
    sizes=$(find "$dir" -type f -name "baseband_${event_id}_*.data" -printf '%s\n' 2> /dev/null | sort -u)
    if [[ -z "$sizes" ]]; then
        record 10 FAIL "trigger $event_id: no files in $dir: $(tail -n 3 "$outdir/$name.trigger.log" | tr '\n' ' ')"
    elif (($(wc -l <<< "$sizes") > 1)) || [[ "$sizes" == 0 ]]; then
        record 10 FAIL "trigger $event_id: file sizes differ or are zero: $(tr '\n' ' ' <<< "$sizes")"
    else
        record 10 PASS "trigger $event_id: $(find "$dir" -type f -name "baseband_${event_id}_*.data" | wc -l) files of $sizes bytes"
    fi
}

run_config() {
    local cfg=$1 prepared event_id="" zero
    name=$(basename "$cfg" .yaml)
    data_dir="$outdir/$name"
    log="$outdir/$name.log"
    mkdir -p "$data_dir"
    echo "== $name"

    if other_kotekan; then
        record 0 FAIL "another kotekan is running; not started"
        return
    fi
    local prepare_args=(prepare "$cfg" "$outdir/$name.yaml" --output-dir "$data_dir")
    [[ -n "$antennas" ]] && prepare_args+=(--antennas "$antennas")
    [[ -n "$cal_file" ]] && prepare_args+=(--calibration-file "$cal_file")
    if ! prepared=$("$python" "$helper" "${prepare_args[@]}" 2> "$outdir/$name.prepare.log"); then
        record 0 FAIL "config copy failed: $(tr '\n' ' ' < "$outdir/$name.prepare.log")"
        return
    fi
    local handlers gpu_stages file_prefixes cadence n_freq baseband_endpoint used_antennas
    handlers=$(sed -n 's/^handlers=//p' <<< "$prepared")
    gpu_stages=$(sed -n 's/^gpu_stages=//p' <<< "$prepared")
    file_prefixes=$(sed -n 's/^file_prefixes=//p' <<< "$prepared")
    cadence=$(sed -n 's/^cadence=//p' <<< "$prepared")
    n_freq=$(sed -n 's/^n_freq=//p' <<< "$prepared")
    baseband_endpoint=$(sed -n 's/^baseband_endpoint=//p' <<< "$prepared")
    used_antennas=$(sed -n 's/^antennas=//p' <<< "$prepared")
    summary_antennas=${used_antennas:-$summary_antennas}
    if [[ -s "$outdir/$name.prepare.log" ]]; then
        record 0 WARN "$(tr '\n' ' ' < "$outdir/$name.prepare.log")"
    fi
    if ! "$kotekan" --check-config -c "$outdir/$name.yaml" > "$outdir/$name.check.log" 2>&1; then
        record 0 FAIL "--check-config failed, see $outdir/$name.check.log"
        return
    fi

    (cd "$repo" && exec "$kotekan" -c "$outdir/$name.yaml") > "$log" 2>&1 &
    pid=$!
    local started=$SECONDS align_seconds=0 aligned=1
    exit_code=""

    if ((handlers > 0)); then
        if wait_for_alignment; then
            check_alignment
            record 2 PASS "aligned within ${align_timeout}s"
        else
            aligned=0
            curl -s -m 5 "$rest/metrics" > "$outdir/$name.metrics_timeout.txt"
            zero=$("$python" "$helper" zero-ports "$outdir/$name.metrics_timeout.txt")
            record 1 FAIL "no alignment line in the log"
            if kill -0 "$pid" 2> /dev/null; then
                record 2 FAIL "no alignment after ${align_timeout}s; ports with rx_packets_total = 0: ${zero:-none}"
            else
                record 2 FAIL "kotekan exited before aligning, see $log"
            fi
        fi
    else
        record 1 SKIP "no rfsocHandlerShuffle handlers"
        record 2 SKIP "no rfsocHandlerShuffle handlers"
    fi

    curl -s -m 5 "$rest/metrics" > "$outdir/$name.metrics_start.txt"
    if ((aligned)) && [[ -n "$baseband_endpoint" ]] && kill -0 "$pid" 2> /dev/null; then
        sleep "$baseband_settle"
        event_id=$(date +%s)
        "$python" "$repo/charts/trigger_baseband.py" --url "$rest$baseband_endpoint" \
            --event-id "$event_id" --duration 1 --wait --wait-timeout 120 \
            > "$outdir/$name.trigger.log" 2>&1
    fi
    while ((aligned && SECONDS - started < run_seconds + align_seconds)) && kill -0 "$pid" 2> /dev/null; do
        sleep 5
    done
    local died=0
    kill -0 "$pid" 2> /dev/null || died=1
    curl -s -m 5 "$rest/metrics" > "$outdir/$name.metrics_end.txt"
    curl -s -m 5 "$rest/trackers" > "$outdir/$name.trackers.json"
    local run_time=$((SECONDS - started))
    stop_kotekan

    local errors
    errors=$(grep -cE "$error_re" "$log")
    if ((died)) && [[ "$exit_code" == 0 ]]; then
        record 3 WARN "kotekan exited cleanly by itself after ${run_time}s"
    elif ((died)); then
        record 3 FAIL "kotekan exited by itself after ${run_time}s (code $exit_code): $(grep -m1 -E "$error_re" "$log")"
    elif ((errors > 0)); then
        record 3 FAIL "$errors error lines, first: $(grep -m1 -E "$error_re" "$log")"
    else
        record 3 PASS "no error lines in ${run_time}s, exit code $exit_code"
    fi

    if ((handlers > 0)); then
        "$python" "$helper" metrics "$outdir/$name.metrics_start.txt" "$outdir/$name.metrics_end.txt" \
            --ports "$handlers" | record_helper
        local noise
        noise=$(grep -cE "$noise_re" "$log")
        if ((noise > 0)); then
            record 5 WARN "$noise invalid packet lines, first: $(grep -m1 -E "$noise_re" "$log")"
        else
            record 5 PASS "no invalid packet lines"
        fi
    else
        record 4 SKIP "no rfsocHandlerShuffle handlers"
        record 5 SKIP "no rfsocHandlerShuffle handlers"
    fi

    "$python" "$helper" files "$data_dir" --prefixes "$file_prefixes" --cadence "$cadence" | record_helper
    if [[ "$file_prefixes" == *corr_freq* && -n "$used_antennas" ]]; then
        local data_args=(data "$data_dir" --antennas "$used_antennas" --n-freq "$n_freq")
        [[ -n "$reference" ]] && data_args+=(--reference "$reference")
        "$python" "$helper" "${data_args[@]}" | record_helper
    else
        record 7 SKIP "no corr_freq output or no antenna list"
        record 8 SKIP "no corr_freq output or no antenna list"
    fi
    if ((gpu_stages == 0)); then
        record 9 SKIP "no cudaProcess stage"
    elif ((died)); then
        record 9 SKIP "kotekan was not running at the end, see check 3"
    elif [[ -s "$outdir/$name.trackers.json" ]]; then
        "$python" "$helper" trackers "$outdir/$name.trackers.json" | record_helper
    else
        record 9 FAIL "no /trackers reply"
    fi
    if [[ -n "$event_id" ]]; then
        check_baseband "$event_id"
    fi
}

for cfg in "${configs[@]}"; do
    run_config "$cfg"
done

summary="$outdir/summary.md"
{
    echo "# Node smoke test, commit $commit"
    echo
    echo "- Date: $(date -Is)"
    echo "- Host: $(hostname)"
    echo "- Binary: $kotekan (CMAKE_BUILD_TYPE ${build_type:-unknown}, source $source_dir)"
    echo "- Run time per config: ${run_seconds}s after alignment"
    echo "- Antennas: ${summary_antennas:-from each config}${antennas:+ (from $antennas)}"
    echo "- Calibration file: ${cal_file:-from each config}"
    echo "- Reference: ${reference:-none}"
    echo
    echo "| Config | Check | Status | Detail |"
    echo "|---|---|---|---|"
    awk -F'\t' '{gsub(/\|/, "\\|", $4); printf "| %s | %s | %s | %s |\n", $1, $2, $3, $4}' "$results"
} > "$summary"

echo
echo "Summary: $summary"
awk -F'\t' '{n[$3]++} END {for (s in n) printf "%s: %d\n", s, n[s]}' "$results"
! grep -qP '\tFAIL\t' "$results"
