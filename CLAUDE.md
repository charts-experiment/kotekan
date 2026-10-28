# CLAUDE.md — kotekan (CHARTS)

Context for resuming work on another PC or on the node. The user (Juan Contreras) speaks Spanish; reply in Spanish, but keep repo docs, code and commit messages in English. Full technical detail: `charts/charts_visibility_calibrator.md`.

## Working rules
- **Git: do not run `git commit`, `git push` or create branches unless the user explicitly asks for it in that request.** By default, give the exact commands instead. Always `git add` explicit paths, never `git add .`.
- Never commit files unrelated to the task: `charts/config/64antennas_record.yaml` (user's local change), `charts/old_results_sync/`, `test_charts/scratch_charts_sim/`, `gains_*.npy`.
- Task log in Notion: page "Registro de Tareas" → task *Calibración en nodo* (section *Calibración de correlator*). Keep it up to date.
- Remotes: `charts-experiment/kotekan` and `charts-experiment/charts-utils` (sibling repo at `~/charts-utils`).

## Current state (2026-10-28)
Branch `feat/vis-calibrator` (pushed; includes the merge of `origin/kotekan_charts` and the CMake fix). Notion task: in progress, due 2026-10-30.

**Implemented: `cudaVisibilityCalibrator`**, a `cudaProcess` command between `cudaCorrelatorAstron` and `cudaSyncOutput`. It multiplies every `int32` visibility in place by a `complex64` factor from a `.npy` table of shape `(n_freq, 528, 2, 2)` (float32 math, round to nearest, int32 saturation). Without `calibration_file` it is a bit-exact pass-through. Parameters: `gpu_mem_correlation_matrix`, `num_local_freq`, `num_blocks`, `calibration_file`, `freq_offset` (0 and 336 for the two 336-channel shards of the 672-channel band).

Files:
- kotekan: `lib/cuda/cudaVisibilityCalibrator.{hpp,cpp,cu}`; configs `charts/config/64antennas_corr_cal.yaml` and `64antennas_corr_trigger_cal.yaml`; tests `test_charts/tests/test_visibility_calibrator.py`; helpers `test_charts/sim/corr_cal.py`; benchmark `test_charts/sim/benchmark_vis_calibrator.py` (`python test_charts/charts_sim.py bench-vis-cal ...`).
- charts-utils (**not pushed yet as of this note; check `git status` there**): `charts_utils/cal/kotekan_calibration.py`, `scripts/kotekan_vis_calibrator.py`, `tests/test_kotekan_calibration.py`. Without them, tables cannot be built on another PC.

**Validated**
- Simulation, RTX 5090: bit-exact pass-through; error ≤0.71 counts against `round(vis × table)`; ~4e-6 relative against charts-utils `apply_gains`; saturation; a wrong table shape or offset stops kotekan.
- Simulated performance: calibrator 5.8 µs/frame (0.11 % of the 5.12 ms frame period).
- **Real data (user, 2026-10-28): fringes look good and similar to the uncalibrated ones.** Still to be confirmed with numbers (see "To do"): with 4 connected antennas, all calibrated, the effect should be measurable.

**Real gains used so far:** `gains_sun_transit_1027_interp.npy`, (672, 4) complex128; CHARTS antennas **0, 5, 32, 37**, reference antenna 5 (gain 1, phase 0). Build the table with:
```bash
PYTHONPATH=~/charts-utils python ~/charts-utils/scripts/kotekan_vis_calibrator.py \
  -g gains_sun_transit_1027_interp.npy -ant 0,5,32,37 --num-antennas 64 -o /data/calibration/vis_cal_64.npy
```
Gain and table `.npy` files are not versioned: copy them with `scp`.

## Non-obvious facts
- Frame layout: `[freq_local 336][baseline 528 = a*(a+1)/2+b, a>=b][pol_a][pol_b][re,im]`, int32. Raw element `r` = receiver `r//2`, pol `r%2`. **CHARTS antenna `a` = raw element `N-1-a`.** `V_ab = x_a·conj(x_b)`; calibrating is `V/(g_i g_j*)`.
- Voltages are int4+int4; the correlator output is int32; the calibrator is the only float32→int32 step. `chartsAccumulate` accumulates in int64 internally.
- **Only 4 antennas are connected today (0, 5, 32, 37), the same ones that have gains.** The other 60 carry no signal, so the table covers everything useful: the 10 products among those 4 (autos included). Products with disconnected antennas are noise. Using factor 1 for products with antennas that have no gains only matters once more antennas are connected.
- **Consequence for validation:** if the calibrated fringes look *identical* to the uncalibrated ones, that is not obviously normal. The gains have amplitude 0.55–3.09 (the table multiplies by up to ×3.25), so amplitudes should change, and phases too unless the gain phases are small. Check with numbers before accepting the calibration.
- The trigger configs use `selected_antennas`/`signal_antennas` `[0, 4, 5, 6]`: antennas 4 and 6 are **not connected**, while 32 and 37 are. Change to `[0, 5, 32, 37]` if the triggered baseband should use the antennas that carry signal.
- Node (`~/kotekan_vis_cal`, host ms-x00): CUDA 12.8, NVCC 12.8.93, DPDK ON, HDF5 OFF. Run with `sudo build/kotekan/kotekan -c charts/config/64antennas_corr_trigger_cal.yaml`.
- Upstream bug fixed: `lib/stages/CMakeLists.txt` listed `chartsBasebandCapture.cpp` (does not exist) instead of `chartsBasebandCaptureSelect.cpp`. Port the fix to `kotekan_charts` and derived branches (`kotekan_charts_next`, `beamform-64`, `kotekan_charts_trillium`) if it is not there yet.
- The branch was cut from an old `kotekan_charts`; the merge brought in the int32 overflow fix in `cudaShuffleAstron.cu` (production frames are ≈3.3 GB per shard).
- Local test environment: driver CUDA 13.2 vs NVRTC 13.3 → "PTX compiled with an unsupported toolchain" (also without the calibrator). Workaround: `CHARTS_TEST_LD_LIBRARY_PATH=<dir with libnvrtc 13.0>`.
- Tests: `python -m pytest test_charts/tests/test_visibility_calibrator.py -q` (needs the built binary and a GPU); in charts-utils `python -m pytest tests/test_kotekan_calibration.py -q` (use `PYTHONPATH=~/charts-utils` if needed).

## To do (in order)
1. **Quantitative validation on real data:** on the calibration source (the Sun), the phase of the products among antennas 0, 5, 32, 37 should be ≈0 with the table and not with it disabled. Compare against the offline calibration (`apply_gains`). Record the same segment with and without `calibration_file` (same config), or apply the table offline to uncalibrated frames and compare. Expected: auto-correlation ratio ≈ `1/|g_i|²`; cross-product phase change `-(φ_i − φ_j)` per channel.
2. Run on the node without `calibration_file` (must equal a run without the calibrator) and then with the table; check the start-up log (shape, `freq_offset`).
3. **Benchmark on the real system:** `bench-vis-cal` with the production config (2 shards, 15360/153600 samples per frame) and real traffic that differs between frames (the simulation repeats one frame, which favours L2). Record kernel time (`/trackers`, `gpu.profiling: true`), GPU utilisation, and dropped frames/buffer overflows. Write the results in `charts/charts_visibility_calibrator.md` §6.
4. Check the correlator's fixed `-arch=compute_86` against the node's GPU.
5. Commit/push charts-utils (separate branch) and repo hygiene: ignore `test_charts/scratch_charts_sim/`, regenerate `64antennas_corr_cal.yaml` from the current `64antennas_corr.yaml`, decide `64antennas_corr_trigger.yaml` vs `64antennas_corr_triggered_baseband.yaml`.
6. Optional: factor 1 for products with antennas without gains (only relevant once more antennas are connected); hot reload over REST; a `phase_scatter_deg` threshold; calibration in the tracking beam/multi-beamforming; fix `create_correlator_yaml` in `test_charts/sim/pipeline.py`; confirm the int4 nibble order (the doc says real in the low nibble, charts-utils `unpack_4bit_complex` takes real from the high one; it does not affect the calibrator).
7. Update the Notion task after each step.
