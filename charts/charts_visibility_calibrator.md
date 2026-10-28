# On-node visibility calibration: `cudaVisibilityCalibrator`

State as of 2026-10-28. Notion task: *Calibración en nodo* (section *Calibración de correlator*), in the task log. Summary for resuming work on another PC: `CLAUDE.md` at the repo root.

**Status:** implemented, tested in simulation and deployed on the node. On real data the fringes look good and similar to the uncalibrated ones; the quantitative phase validation and the benchmark on the real system are still missing (§8).

## 1. Goal and design decisions

Apply the instrumental gains to the visibilities on the GPU, right after the N2 Astron correlator, without touching the correlator kernel or the baseband (which is calibrated offline).

| Decision | Reason |
|---|---|
| Calibrate the visibility matrix, not the voltages | Voltages are `int4+int4` and the Astron kernel already works with that format. Converting to float and re-quantising would add error. The correlator output is already `int32`, which can be operated on. |
| One new command, `cudaVisibilityCalibrator`, between `cudaCorrelatorAstron` and `cudaSyncOutput` | Keeps the architecture modular. No existing stage or config changes, and whoever does not add the command sees no change. |
| Baseline products `1/(g_i g_j*)` are computed offline in charts-utils | The GPU only does one complex multiplication per visibility. |
| In-place `int32` calibration, `complex64` factor, round to nearest with saturation | `chartsAccumulate` and `rawFileWrite` do not change (same frame size). Agreed with the user. |
| Time-averaged gains (`nanmean`), one static table | Recalibration happens every ~2 days, not in real time. Check `phase_scatter_deg` and `amp_scatter` from `cal_pipeline_gpu.py` to confirm the average is safe. |
| NaN or zero gains give factor 0 | Flagged data (`--flag-freqs`) is zeroed instead of propagating NaN. |
| Antennas without gains keep gain 1 | They pass uncalibrated. Note: a baseline between a calibrated and an uncalibrated antenna ends up half-calibrated. |

## 2. Data layout (must be respected)

- Correlator frame: `[freq_local][baseline][pol_a][pol_b][re, im]`, all `int32`.
- `baseline = recv_a*(recv_a+1)/2 + recv_b` with `recv_a >= recv_b` (lower triangle). With 64 antennas: 32 receivers × 2 pols = 528 blocks, equal to `num_blocks`.
- Raw element `r`: receiver `r//2`, pol `r%2`.
- **Reversed antenna order:** CHARTS antenna `a` is raw element `N-1-a` (`charts_utils/core/antenna_indices.py`).
- Convention `V_ab = x_a · conj(x_b)`. `apply_gains` computes `V/(g_i g_j*)`.
- Two frequency shards (`corr_freq_0` and `corr_freq_1`), 336 channels each. The table covers the full band (672 channels) and each command takes its slice with `freq_offset` (0 and 336).

## 3. What was implemented

### kotekan (`/home/jpcontreras/kotekan`)
- `lib/cuda/cudaVisibilityCalibrator.{hpp,cpp,cu}` and its registration in `lib/cuda/CMakeLists.txt`.
  - At start-up it reads a C-contiguous `complex64` `.npy` of shape `(n_freq, num_blocks, 2, 2)`, validates the shape and channel range, and uploads only its slice to the GPU.
  - Kernel: one thread per visibility, `float32` with `fmaf`, `__float2int_rn` rounding with saturation to `[-2^31, 2147483520]`.
  - Without `calibration_file` no kernel is launched (pass-through).
  - Requires `block_size: 2` and `num_data_sets: 1`; fails at start-up otherwise.
  - Parameters: `gpu_mem_correlation_matrix`, `num_local_freq`, `num_blocks`, `calibration_file` (optional), `freq_offset` (default 0).
- `charts/config/64antennas_corr_cal.yaml`: copy of `64antennas_corr.yaml` with the command added to both command lists, with `freq_offset` 0 and 336 and `calibration_file: /data/calibration/vis_cal_64.npy` (example path, adjust). The original config was not modified.
- `charts/config/64antennas_corr_trigger_cal.yaml`: correlator plus triggered baseband, with the calibrator in `commands_0/1`.

### charts-utils (`/home/jpcontreras/charts-utils`)
- `charts_utils/cal/kotekan_calibration.py`:
  - `expand_gains(gains, antennas, n_ant)` averages the time axis if present and fills unresolved antennas with 1.
  - `gains_to_visibility_calibration(gains)` returns the `complex64 (n_freq, 528, 2, 2)` table in the GPU frame layout.
- `scripts/kotekan_vis_calibrator.py` (CLI: `--gains`, `--antennas`, `--num-antennas`, `--output`).
- `tests/test_kotekan_calibration.py` (6 tests; uses `apply_gains` and `gpu_corr_to_visibilities` as the oracle).

### Tests and benchmark (`test_charts/`)
- `sim/corr_cal.py`: builds the replay YAML (`rawFileRead → cudaProcess → sink`), launches kotekan and decodes frames.
- `tests/test_visibility_calibrator.py`: 8 end-to-end tests against the real binary and the GPU.
- `sim/benchmark_vis_calibrator.py` and the `charts_sim.py bench-vis-cal` subcommand.

## 4. How to use it

```bash
# 1. Solve gains (charts-utils) and save them averaged
python scripts/cal_pipeline_gpu.py --input <frames_dir> --save-avg-gains -o gains.npy ...

# 2. Build the GPU table (antennas in the file, as CHARTS indices)
#    Real gains used so far: gains_sun_transit_1027_interp.npy, (672, 4), antennas 0,5,32,37, reference antenna 5
PYTHONPATH=~/charts-utils python ~/charts-utils/scripts/kotekan_vis_calibrator.py \
  -g gains_sun_transit_1027_interp.npy -ant 0,5,32,37 --num-antennas 64 -o /data/calibration/vis_cal_64.npy

# 3. Set the path in the config (calibration_file, in both commands) and run on the node
sudo build/kotekan/kotekan -c charts/config/64antennas_corr_trigger_cal.yaml
```

To disable calibration, remove `calibration_file` (pass-through). Gain and table `.npy` files are not versioned: copy them with `scp`.

**Scope of the table.** Only 4 antennas are connected today (0, 5, 32, 37) and they are the ones with gains, so the table covers every product that carries signal: 10 counting the autos. Products with disconnected antennas are noise and their factor does not matter. A baseline between a calibrated antenna and one without gains would be multiplied by `1/g_i` (half-calibrated), but that only matters once more antennas are connected.

**What to expect.** The table multiplies by up to ×3.25 in amplitude and corrects each antenna's phase relative to antenna 5. If the fringes look identical with and without calibration, measure it: see the quantitative validation in §8.1.

## 5. How it was validated

All with the real binary on the RTX 5090.

| Check | Result |
|---|---|
| No table vs. no command at all | Bit-exact |
| GPU vs. `round(vis × table)` | Maximum error 0.71 counts (`float32` limit with values of ~10⁷) |
| GPU vs. charts-utils `apply_gains`, `freq_offset` 0 and 336 | Relative error ~4×10⁻⁶ |
| Saturation | Saturates at the `int32` limits, no wrap-around |
| Wrong table shape or out-of-range offset | kotekan stops at start-up with a clear message |

Run the tests:

```bash
export CHARTS_TEST_LD_LIBRARY_PATH=<dir with a libnvrtc matching the driver>   # see note 7.1
python -m pytest test_charts/tests/test_visibility_calibrator.py -q
python -m pytest tests/test_kotekan_calibration.py -q      # in charts-utils
```

## 6. Performance

`python test_charts/charts_sim.py bench-vis-cal --frames 4000 --repetitions 3 --output-dir <dir>` measures with kotekan's trackers (`/trackers`, needs `gpu.profiling: true`; it keeps only the last 100 samples, so it is polled repeatedly) and with `nvidia-smi` at 100 ms. Results (RTX 5090, 64 antennas, 336 channels, 1536 samples/frame):

| | Without calibrator | With calibrator |
|---|---|---|
| Astron correlator | 194.4 µs | 194.3 µs |
| **Calibrator** | n/a | **5.8 µs** (p99 5.8, max 7.2) |
| Kernel time per frame | 225.8 µs | 231.5 µs |
| Pipeline rate (real time: 195.3 Hz) | 238.6 Hz | 238.3 Hz |
| GPU util / power / SM clock | 22 % / 104 W / 2625 MHz | 22 % / 104 W / 2627 MHz |
| Max GPU memory | 671 MiB | 689 MiB |

The calibrator uses 0.11 % of the frame period and 3 % of the correlator time. The +18 MiB is the table.

Caveats when reading these numbers:
- `cudaSync*`, `cudaInputData` and `cudaOutputData` measure waits and PCIe copies, not compute; kernels are therefore summed separately.
- The benchmark repeats the same frame, so the table and data probably fit in L2 (96 MB on the 5090). With real traffic it could rise somewhat (estimated up to ~10 µs) without changing the conclusion.

## 7. Known environment and repo issues

1. **NVRTC newer than the driver.** On this machine the driver is CUDA 13.2 and the toolkit 13.3. The Astron correlator, which compiles at runtime, fails with *"PTX compiled with an unsupported toolchain"*, even without the calibrator. For the tests I used a 13.0 `libnvrtc` through `LD_LIBRARY_PATH` (`CHARTS_TEST_LD_LIBRARY_PATH`). A permanent fix is still missing: match driver and toolkit, or compile to SASS instead of PTX.
2. **`create_correlator_yaml` in `test_charts/sim/pipeline.py` is outdated.** It uses old `rawFileRead` keys (`prefix`, `out_buf`) and separate stages instead of `cudaProcess`, so `charts_sim.py correlate` and `pipeline` do not run with the current kotekan. `sim/corr_cal.py` generates a current config that can be reused.
3. **Replay frames carry a 4-byte prefix with metadata size 0**, and `charts_utils.core.io` expects ≥16 bytes. `load_corr_frame` in `sim/corr_cal.py` reads them with that offset.
4. **`test_live_control_cli.py::test_01_query_status` fails** with connection refused, because it needs a live kotekan. It was not checked whether it already failed before these changes.
5. **`test_charts/scratch_charts_sim/` shows up untracked** in git (`.gitignore` does not cover it from that level).

### 7.6 Node deployment: what failed and how it was solved

1. `ERROR: Unrecognized Stage! (chartsTriggeredBasebandReadout)`: the branch was cut from an old `kotekan_charts`. `origin/kotekan_charts` was merged into `feat/vis-calibrator`. The merge also brought in the int32 overflow fix in `cudaShuffleAstron.cu`, needed for production frames (≈3.3 GB per shard).
2. `Cannot find source file: chartsBasebandCapture.cpp` in `lib/stages/CMakeLists.txt`: an upstream error (the real file is `chartsBasebandCaptureSelect.cpp`). Fixed in commit `45324c173`. The fix still has to be ported to `kotekan_charts` and derived branches.
3. Node notes: CUDA 12.8, NVCC 12.8.93, DPDK ON, HDF5 OFF, WERROR OFF.

## 8. Next steps

In priority order.

1. **Physical validation on real data.** Done by eye: the fringes look good and similar to the uncalibrated ones, which with 4 calibrated antennas should be checked with numbers (amplitudes should change by up to ×3). The quantitative test is still missing. The simulation does not inject per-antenna gains, so it only validated arithmetic and layout.
   - Use the calibration source (the Sun, `gains_sun_transit_1027_interp.npy`) and the products among antennas 0, 5, 32, 37.
   - Record with and without `calibration_file` using the same config, or apply the table offline to uncalibrated frames.
   - Criteria: source phase ≈0 with the table (and not without it); amplitude consistent with `apply_gains`; difference from the offline calibration ≈ quantisation noise (≤1 count per visibility).
2. **Real gains.** Confirmed: antennas 0, 5, 32, 37, reference antenna 5, amplitude 0.55–3.09 (max factor ~3.25 in the table). No int32 saturation risk with those amplitudes. The old file `gains_E_2026-10-19_with0.npy` `(672, 4, 7)` has unidentified antennas and amplitudes 0.0018–18; do not use it without confirming them.
3. **Node test.** First without `calibration_file` (must equal the run without the calibrator), then with the table. Check the start-up log (table shape, `freq_offset`) and that `chartsAccumulate`, `rawFileWrite` and the charts-utils scripts (`plot_triangular_vis.py`, `cal_pipeline_gpu.py`) work with the calibrated output.
4. **Benchmark on the real system.** Repeat `bench-vis-cal` (or read `/trackers` with `gpu.profiling: true` on the production kotekan) with the production config: 2 shards, 15360/153600 samples per frame, data that differs between frames to rule out the L2 effect. Record:
   - calibrator and correlator kernel time (µs/frame) and its ratio to the frame period;
   - GPU utilisation, power and memory (`nvidia-smi`);
   - dropped frames and buffer occupancy with and without the calibrator;
   - the node's GPU and driver, and the correlator's fixed `-arch=compute_86` against the real GPU.
   Write the numbers in the §6 table with the date and node.
5. **Hot reload over REST.** Today recalibrating means restarting kotekan. If needed, add an endpoint with a double buffer on the GPU and a swap at the start of a frame.
6. **Apply the same idea to beamforming.** Per the Notion task, in the tracking beam and multi-beamforming the gains are multiplied at each stage, after the internal quantisation. This is separate work.
7. **Review the gain average.** With long windows, a complex `nanmean` can reduce the amplitude if the phase drifts. Define a `phase_scatter_deg` threshold to accept a calibration.
8. **Repo hygiene.**
   - Fix or remove `create_correlator_yaml` (7.2) and resolve NVRTC (7.1).
   - Add `scratch_charts_sim/` to `.gitignore` (7.5).
   - Commit charts-utils, which has uncommitted changes unrelated to this (`pyproject.toml`, `export_directions.py`).
   - Update the *Calibración en nodo* task in Notion with this state.
