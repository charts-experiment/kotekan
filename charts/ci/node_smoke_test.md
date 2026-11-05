# Node smoke test

`charts/ci/node_smoke_test.sh` runs CHARTS configs on the GPU node with live RFSoC traffic and
checks the pipeline end to end: DPDK capture, port start-up alignment, GPU load, output files,
shard timestamps and visibility levels. It is the node validation that a release needs before
the PR `kotekan_charts_next → kotekan_charts` (see [CONTRIBUTING.md §5](../CONTRIBUTING.md)).
`charts/ci/node_smoke_check.py` holds the config rewriting and the checks that parse metrics,
trackers and data files.

The GitHub CI (`.github/workflows/charts_ci.yaml`) only builds kotekan and checks configs
statically; it never opens a NIC or a GPU. This test is the one that runs them.

## Before running

- Only one kotekan can own the NICs (DPDK), so production must be stopped first. The script
  refuses to start if a `kotekan` process is running or `localhost:12048` answers; it never
  stops one itself. Note how production was started so it can be relaunched exactly.
- It needs `sudo` (DPDK, hugepages).
- Run it from a test clone, not from the production checkout. kotekan is started from the
  clone's root, which the configs' `kernel_path` is relative to.
- Nothing in the repo is modified. Each config is copied to `OUTDIR/<config>.yaml` with every
  output path pointing to `OUTDIR/<config>/`.

## Usage

```bash
sudo charts/ci/node_smoke_test.sh [-t SECONDS] [-o OUTDIR] [-b KOTEKAN_BIN] [-a ANTENNAS]
    [-k CAL_FILE] [-r REF_DIR] [-p PYTHON] [-w SECONDS] [config.yaml ...]
```

| Option | Default | Meaning |
|---|---|---|
| `-t` | 120 | Run time per config, counted from the alignment. |
| `-o` | `/data/smoke_test/<date>_<commit>` | Output directory. |
| `-b` | `<repo>/build/kotekan/kotekan` | kotekan binary. |
| `-a` | `signal_antennas` of `~/kotekan_charts_next/charts/config/64antennas_corr.yaml` | Antennas: `0,4,32,36`, or a config to take `signal_antennas` from. |
| `-k` | none | Calibration table for `cudaVisibilityCalibrator`, or `none` for pass-through. Also adds `64antennas_corr_cal` and `64antennas_corr_trigger_cal` to the default configs. |
| `-r` | none | Production output directory to compare autocorrelation levels against (check 8). |
| `-p` | `~/charts-utils/.venv/bin/python` | Python with pyyaml, and charts-utils for checks 7 and 8. |
| `-w` | 180 | Alignment timeout in seconds, counted from launch. |

Without config arguments it runs `64antennas_corr`, `64antennas_corr_triggered_baseband` and
`64antennas_direct_tracker`. Typical sequence:

```bash
cd ~/kotekan_next_test
# script check without DPDK or RFSoC (~2-3 min)
sudo charts/ci/node_smoke_test.sh -t 60 -o /data/smoke_test/sim charts/config/64antennas_charts_sim.yaml
# live traffic, default configs (~12 min), optionally with the calibrator (+~8 min)
sudo charts/ci/node_smoke_test.sh -r /data/corr/<recent production run>
sudo charts/ci/node_smoke_test.sh -r /data/corr/<recent production run> -k <table.npy>
```

Each config takes about 4 minutes with the default `-t 120`: 1 to 1.5 min to start and align,
the run, then up to 2 min to stop (the baseband config drains pending captures). 120 s gives
about 11 integrated frames per shard and about 1.4e8 packets per port. A longer `-t` gives a
longer packet-loss sample; below 60 s check 6 may not see two files.

## What it does for each config

1. Write the config copy. The helper replaces `output_dir`, `base_dir` and `root_path`; sets
   `signal_antennas` and `selected_antennas` to the antenna list and `num_saved_antennas` to its
   length; replaces or drops `calibration_file` if `-k` is given. It warns when
   `num_saved_antennas` does not match `selected_antennas` in the original, and fails the
   config if a `calibration_file` does not exist. Other antenna lists, such as the beamformer's
   `active_antennas` and `gains_antennas`, are left as they are.
2. Run `kotekan --check-config` on the copy.
3. Start kotekan, logging to `OUTDIR/<config>.log`, and wait for the alignment lines of
   `rfsocHandlerShuffle`. If the ports do not align within `-w` seconds, stop at once and report
   which ports received no packets.
4. Sample `/metrics`. For a config with `chartsTriggeredBasebandReadout`, wait 15 s for the
   history buffer to fill, then trigger a 1 s capture with `charts/trigger_baseband.py --wait`.
5. Run for `-t` seconds, then sample `/metrics` and `/trackers`.
6. Stop: `GET /stop`, then SIGTERM, then SIGKILL if kotekan has not exited after 180 s. The same
   happens if the script is interrupted.
7. Run the checks below.

## Checks

Each check reports PASS, FAIL, WARN (worth a look, does not fail the run) or SKIP (does not
apply to the config, or charts-utils is not available). The thresholds are constants at the top
of `node_smoke_check.py`.

| # | What | How | PASS | Otherwise |
|---|---|---|---|---|
| 0 | Setup | config copy and `--check-config` | not reported | FAIL: the config is not started. WARN: helper warning, e.g. `num_saved_antennas` mismatch. |
| 1 | Port alignment | log lines `Port p received spectrum ..., locked to common start spectrum S` (`selected start spectrum S` in older builds), one per port | all N ports started at the same S | FAIL. SKIP if the config has no `rfsocHandlerShuffle`. |
| 2 | No hang | check 1 under the `-w` timeout | aligned in time | FAIL, listing the ports with `rx_packets_total` = 0 (a port that never receives data makes kotekan wait forever without a message). |
| 3 | No errors | log lines starting with `ERROR:`, `CRITICAL:`, `ALERT:` or `EMERGENCY:`, or containing `FATAL`; whether kotekan exited by itself | none, and kotekan was still running at the end | FAIL with the first error line. WARN if it exited cleanly by itself (e.g. `exit_after_n_files`). |
| 4 | Packets | `kotekan_dpdk_rx_*` gauges per port, start and end samples | packets grow on every port; lost and rejected (`rx_error_total`) packets are at most 1e-3 of the total | FAIL. The rates are always reported; production shows a steady trickle of rejected packets. |
| 5 | Invalid packets | log lines `Invalid subband number`, `outside configured range`, `Invalid RFSoC id` | none | WARN with the count. |
| 6 | Output cadence | mtimes of the `rawFileWrite` files of each prefix | at least 2 files per prefix, prefixes within one file of each other, median period within 10% of `integration_frames × frame_arrival_period` (10.24 s) | FAIL. SKIP without `rawFileWrite` (the beamformer config writes no files). |
| 7 | Shard timestamps | `discover_gpu_corr_frame_groups` (charts-utils) on `corr_freq_*` | every frame index has both shards with the same `fpga_seq_num` | FAIL with the first mismatch or missing shard. |
| 8 | Visibility levels | `extract_gpu_visibility` on the last complete frame: median autocorrelation per antenna, median coherence `abs(V_ab) / sqrt(V_aa V_bb)` per pair | autocorrelations finite and positive | FAIL. WARN if `-r` is given and a level differs from the reference by more than ×2 (expected with a calibration table, which rescales the levels). |
| 9 | GPU load | `/trackers`: utilisation (`_u`, execute time over `frame_arrival_period`) of each GPU command | per `cudaProcess` stage, the summed mean utilisation is below 0.8 and no single command reaches 1 | FAIL. SKIP without GPU trackers (`profiling: false`). |
| 10 | Baseband capture | files `baseband_raw_<event>/baseband_<event>_<freq>.data` | files written, all the same non-zero size | FAIL with the end of the trigger output. Only for configs with `chartsTriggeredBasebandReadout`. |

The coherence in check 8 is reported, not judged. On live sky the connected antennas show about
0.03 for a 10.24 s frame and unconnected inputs about 0.001, so it shows which inputs carry
signal; the autocorrelation levels do not.

## Output

`OUTDIR/` holds, for each config:

- `<config>.yaml`: the copy that was run;
- `<config>.log`, `.check.log`, `.prepare.log`, `.trigger.log`: kotekan, `--check-config`,
  helper and trigger output;
- `<config>.metrics_start.txt`, `.metrics_end.txt`, `.trackers.json`: the REST samples;
- `<config>/`: the pipeline output.

`results.tsv` has one line per config and check, and `summary.md` is the same table with the
commit, binary, antennas, calibration table and reference. Attach `summary.md` to the release
PR. The script exits non-zero if any check fails.

## Not covered

- Whether a calibration table is right for the current antennas. The test only shows that the
  calibrator loads the table and keeps up. A table solved for other inputs is applied without
  complaint.
- Science output of the beamformer configs, which write no files by default.
- Long-term stability: memory growth, slow drift or rare faults need hours, not minutes.
