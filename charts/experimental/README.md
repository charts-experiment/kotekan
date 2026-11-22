# CHARTS Experimental Software Prototypes

Self-contained Python prototypes addressing 2x2 commissioning constraints (periodic grating lobes,
no continuous visibility correlator, and dense terrestrial/periodic RFI) without requiring new
hardware.

## Modules

- `__init__.py`: Shared constants sourced from `charts_constants` (`FREQ`, `C_LIGHT`, `DELTA_TIME`,
  `K_DM`), YAML antenna coordinate loader (`load_antenna_positions_from_yaml`), `rawFileWrite` binary
  beam-power readers, and exact 3D coherent array-factor evaluation.
- `rfi_lm_profiler.py`:
  - `MultiFrequencyLmProfiler`: Elastic-net (L1 + L2) non-negative least-squares deconvolution in
    direction-cosine `(l, m)` coordinates across `300-501 MHz`, canceling unknown receiver passband
    gain and isotropic `T_sys` via cross-beam differencing.
  - `find_optimal_off_beams`: Searches `(l, m)` for OFF reference coordinates minimizing band-averaged
    and worst-channel grating-lobe leakage from target and known interferer directions.
  - `LiveRestLmScanner`: Drives one spare beam through `/direct_tracker/set_target` and
    `/direct_tracker/beam_power` to map local horizon RFI live.
- `spatial_calibration.py`:
  - `BeamSpaceDelayCalibrator`: Fits relative per-antenna residual delays (ns) and decomposes planar
    pointing tilt `(dl, dm)` from non-planar closure skew using normalized multi-beam power ratios.
  - `AdaptiveSidelobeCanceller`: Per-channel Huber-IRLS regression subtracting correlated `1/f` and
    interferer sidelobe leakage while masking positive ON-beam astronomical flares.
- `auto_rfi_transient_detector.py`:
  - `UnsupervisedRfiClassifier`: Standardized 7-feature `k-means++` clustering and per-cluster
    Mahalanobis anomaly scoring on `rfi_channel_snapshots.jsonl`, automatically separating clean
    channels, CW tones, impulsive bursts, and persistent EMA-latch candidates.
  - `CoherentCombCanceller`: Identifies common-mode periodic combs (e.g. `9.7656 Hz` harmonics and
    `~50.1 Hz` mains) from OFF-beam spectral templates and subtracts them with non-negative island
    gains, leaving isolated ON-beam pulsars intact.
  - `MatchedFilterTransientScorer`: Incoherent dedispersion + multi-width MAD boxcar scoring with
    ON-vs-OFF spatial coincidence RFI veto.

## Running Tests & Commissioning Benchmarks

```bash
PYTHONPATH=$PWD python3 -m unittest discover -s charts/experimental -p 'test_*.py' -v
PYTHONPATH=$PWD python3 -m charts.experimental.benchmark_commissioning_v2
```
