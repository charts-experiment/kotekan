# CHARTS Simulation, Processing & Verification Suite

A centralized, bloat-free framework for simulating, processing, verifying, and visualizing CHARTS (*CHilean Astronomical Radio Telescope System*) baseband voltage data with Kotekan.

Designed to execute seamlessly on both **local workstations / PCs** (e.g. NVIDIA GeForce RTX 4090 / RTX 5090) and **HPC clusters** (e.g. Trillium / Compute Canada via Slurm).

---

## 1. Architecture & Design Principles

The simulation suite has been reorganized into a modular, centralized architecture:

```text
test_charts/
├── charts_sim.py                 # Central CLI & unified simulation orchestrator
├── sim/                          # Core modular simulation package
│   ├── __init__.py               # Package exports & public API
│   ├── constants.py              # Physical, instrumental, and site constants (Carén)
│   ├── noise_model.py            # Analog RF chain, Friis cascade, Haslam sky & solar model
│   ├── generator.py              # Baseband window generation & transient event engine
│   ├── pipeline.py               # Kotekan orchestration & dynamic YAML generation
│   ├── inspector.py              # Correlator Hermitian verification & tracker diagnostics
│   ├── visualizer.py             # CASM matrix, multi-baseline waterfalls, and MP4 animations
│   └── presets.py                # Hardware & observation profiles (quick, 1min, 5min, day, night)
├── config/                       # Kotekan stage YAML pipeline templates
├── slurm/                        # Centralized Slurm cluster job scripts
│   ├── trillium_pipeline.slurm   # Parameterized cluster launcher
│   └── submit_build_and_test.slurm
├── tests/                        # Comprehensive unit & integration tests
│   ├── test_constants_parity.py  # C++/Python physical constants parity
│   ├── test_charts_sim.py        # Pipeline & simulation suite test cases
│   └── test_live_control_cli.py  # REST steering & control tests
├── kotekan_tracker_control.py    # Live REST steering CLI
├── kotekan_tracker_dashboard.py  # Interactive browser telemetry dashboard
└── CMakeLists.txt                # C++ CUDA test suites & kernel benchmarks
```

---

## 2. Quickstart on a Local PC

No Slurm cluster required. Works out of the box on Linux workstations equipped with an NVIDIA GPU (RTX 3080/4090/5090 or modern Ada/Blackwell GPUs).

### A. Run End-to-End Pipeline (Quick Test)
Simulates a rapid 2-second test window, runs Kotekan correlator and beam tracking, validates Hermitian symmetry, and exports plots:
```bash
python test_charts/charts_sim.py pipeline --preset quick
```

### B. Dry-Run Verification (No GPU / Pre-Build)
Tests the simulation data generation, physical noise model, and YAML configuration generation without executing the compiled Kotekan binary:
```bash
python test_charts/charts_sim.py pipeline --preset quick --dry-run
```

### C. Standard 1-Minute Science Verification
Generates a realistic 60-second window at Observatorio Carén with 8 celestial targets and 8 injected transients:
```bash
# Daytime Window (15:00 UTC, Sun UP):
python test_charts/charts_sim.py pipeline --preset 1min --profile day

# Nighttime Window (03:00 UTC, Sun DOWN):
python test_charts/charts_sim.py pipeline --preset 1min --profile night
```

---

## 3. Execution on HPC Clusters (Trillium / Slurm)

All cluster execution is driven by a single parameterized launcher (`test_charts/slurm/trillium_pipeline.slurm`). The root of the repository is completely clean of one-off `.sbatch` scripts.

```bash
# Standard 1-minute daytime run:
sbatch test_charts/slurm/trillium_pipeline.slurm

# Full 5-minute nighttime run:
sbatch --export=ALL,PRESET=5min,PROFILE=night test_charts/slurm/trillium_pipeline.slurm

# Pass arbitrary flags directly to charts_sim.py:
sbatch test_charts/slurm/trillium_pipeline.slurm --antennas 64 --max-beams 8 --workers 24
```

---

## 4. CLI Subcommand Reference

### 1. `pipeline` (Full Simulation Pipeline)
```bash
python test_charts/charts_sim.py pipeline [options]
  --preset {quick, 1min, 5min}   # Standard observation scale (default: quick)
  --profile {day, night, both}   # 15:00 UTC (Sun UP) or 03:00 UTC (Sun DOWN)
  --antennas N                   # Number of elements (64 or 256)
  --num-freq N                   # Frequency channels (336 or 672)
  --max-beams N                  # Formed beam count (default: 4 for quick, 8 for 1min)
  --scratch-dir DIR              # Fast scratch directory for runtime buffers
  --output-dir DIR               # Destination for permanent figures and metadata
  --kotekan-bin PATH             # Explicit path to compiled Kotekan binary
  --dry-run                      # Run generation and configure stages without binary
```

### 2. `generate` (Baseband Data Window & Reference Saving)
```bash
python test_charts/charts_sim.py generate \
    --duration-s 60 \
    --antennas 64 \
    --num-freq 672 \
    --start-time "2026-10-15T04:20:00Z" \
    --beam-targets "Vela;auto:3" \
    --save-reference ref_oct15_vela_64ant \
    --num-events 8
```

### 3. `reference` (Reference Baseband Library Management)
```bash
# List all saved reference windows:
python test_charts/charts_sim.py reference list

# Inspect exact manifest details for a reference window:
python test_charts/charts_sim.py reference info --tag ref_oct15_vela_64ant
```

### 4. `correlate` (Correlator Replay Only)
```bash
# Replay from a directory or reference tag:
python test_charts/charts_sim.py correlate --reference ref_oct15_vela_64ant
```

### 5. `track` (Direct Beam Tracker Replay: cudaDirectBeamTracker)
```bash
# Replay baseband frames through Kotekan's upstream Direct Beam Tracker:
python test_charts/charts_sim.py track \
    --reference ref_oct15_vela_64ant \
    --beam-targets "Vela;PSR_J0437-4715;auto:2" \
    --max-beams 4
```

### 6. `benchmark` (Direct Beam Tracker: Real Cadence, VRAM & GPU Power)
Dedicated performance, memory, and wall-power analysis for Kotekan's upstream `cudaDirectBeamTracker`:
```bash
# Benchmark across array scales on target GPU (e.g. RTX 4090):
python test_charts/charts_sim.py benchmark \
    --target-gpu rtx4090 \
    --antennas 32 64 128 256 \
    --beams 1 4 8 \
    --num-freq 672 \
    --samples-per-frame 1536 \
    --out-json direct_tracker_report.json \
    --out-md direct_tracker_report.md
```

### 7. `inspect` (Verification & Signal Diagnostics)
```bash
python test_charts/charts_sim.py inspect --file ./scratch/win15UTC_64ant/correlator/corr_0000000.bin --type corr
python test_charts/charts_sim.py inspect --file ./scratch/win15UTC_64ant/tracker/beams_0000000.bin --type tracker
```

### 8. `visualize` (Generate Plots & Waterfall Figures)
```bash
python test_charts/charts_sim.py visualize --window-dir ./scratch/win15UTC_64ant --type casm
python test_charts/charts_sim.py visualize --window-dir ./scratch/win15UTC_64ant --type waterfall
python test_charts/charts_sim.py visualize --window-dir ./scratch/win15UTC_64ant --type tracker
```

---

## 5. Physical Simulation Features

- **Observatory Coordinates**: Observatorio Carén, Chile (Lat: -33.4211° S, Lon: -70.8635° W, Alt: 458 m).
- **Receiver Chain**: 2-stage Friis cascade (ULNA QPL9547 + PSA4-5043+), providing $T_{\text{rx}} \approx 21.3\text{ K}$.
- **Ground Spillover**: Realistic beam-averaged 2% ground spillover ($T_{\text{spill}} \approx 5.8\text{ K}$) for a wide-beam zenith-pointed patch antenna over a ground plane.
- **Topocentric Solar Ephemeris**: Continuous calculation of Sun elevation, azimuth, and direction cosines $(l_\odot, m_\odot, n_\odot)$ with primary beam attenuation.
- **Transient Injection**:
  - Dispersed Fast Radio Bursts (chirps satisfying $\Delta t = k_{\text{DM}} \text{DM} (\nu_1^{-2} - \nu_2^{-2})$).
  - Southern sky pulsars (e.g. PSR J0437-4715 millisecond pulsar and Vela-like pulse trains).
  - Continuous site RFI lines (e.g. channels 94, 133, 147 -> 328.2, 339.9, 344.1 MHz).
  - Fast LEO satellite sweeps with drifting $(l(t), m(t))$ trajectory.
- **Quantization**: Realistic 4-bit two's complement integer mapping ($-7$ to $+7$) with per-antenna gain dispersion and analog bandpass ripple.

---

## 6. Running Unit & Integration Tests

```bash
# Run all Python test suites:
python -m unittest discover -s test_charts/tests

# Run individual test suites:
python -m unittest test_charts/tests/test_constants_parity.py
python -m unittest test_charts/tests/test_charts_sim.py
python -m unittest test_charts/tests/test_live_control_cli.py
```
