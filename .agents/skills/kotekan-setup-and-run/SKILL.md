---
name: kotekan-setup-and-run
description: >-
  Complete step-by-step guide to clone, install dependencies, compile, and execute
  Kotekan from scratch on a new PC or server node (Linux / CUDA 12/13 / RTX 4090 / RTX 5090).
---

# Kotekan Setup and Execution from Scratch

This guide provides the complete end-to-end workflow for configuring, compiling, and running Kotekan on a fresh machine (such as a Linux workstation or cluster node equipped with modern GPUs like the NVIDIA GeForce RTX 4090 or RTX 5090).

---

## 1. System Prerequisites

Ensure essential build tools, compilers, and NUMA libraries are installed on the host OS:

```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    libnuma-dev \
    libboost-all-dev \
    pkg-config
```

Verify the NVIDIA driver and CUDA Toolkit:
```bash
nvidia-smi
nvcc --version   # Recommended: CUDA 12.4+ or CUDA 13.1+
```

---

## 2. Directory Layout & Cloning

`kotekan` and `charts-constants` must be cloned side-by-side in the same parent directory so that local package paths resolve properly:

```bash
mkdir -p ~/charts_workspace && cd ~/charts_workspace

# 1. Clone kotekan (charts development branch)
git clone -b kotekan_charts_next https://github.com/charts-experiment/kotekan.git

# 2. Clone charts-constants into the same directory
git clone https://github.com/charts-experiment/charts-constants.git
```

Resulting directory tree:
```text
charts_workspace/
+-- charts-constants/
+-- kotekan/
```

---

## 3. Python Virtual Environment (`uv`)

Use `uv` to build an isolated, fast Python virtual environment:

```bash
cd ~/charts_workspace/kotekan

# Install uv if not already present
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.cargo/env

# Create and activate virtual environment
uv venv .venv
source .venv/bin/activate

# Install all dependencies (installs ../charts-constants in editable mode)
uv pip install -r requirements.txt
```

---

## 4. C++ & CUDA Build Configuration

### GPU Architecture Targeting
- **RTX 4090 (Ada Lovelace)**: `sm_89` (`-DCMAKE_CUDA_ARCHITECTURES=89`)
- **RTX 5090 (Blackwell)**: `sm_120` (`-DCMAKE_CUDA_ARCHITECTURES=120`)
- **Automatic Host Detection**: `-DCMAKE_CUDA_ARCHITECTURES=native`

> **Note for CUDA 13+**: NVIDIA dropped the legacy `compute_61` (Pascal) architecture in CUDA 13.0. Always pass `-DCMAKE_CUDA_ARCHITECTURES=89` (or `120`/`native`) to compile directly for your GPU and drastically speed up `nvcc` compile times.

### Build Commands:
```bash
cd ~/charts_workspace/kotekan
mkdir -p build && cd build

# Clean any existing CMake cache
rm -rf CMakeCache.txt CMakeFiles/

# Configure CMake (for RTX 4090)
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=89 \
    -DWERROR=OFF

# Compile using all CPU cores
make -j$(nproc)
```

Verify binary generation:
```bash
./kotekan/kotekan --help
```

---

## 5. Pipeline Execution Modes

### Mode A: Offline Continuous Replay Pipeline (Local Test without NIC/RFSoC)
Simulates continuous RFSoC baseband data streamed in a circular loop through the CUDA Beam Tracker:

```bash
cd ~/charts_workspace/kotekan
source .venv/bin/activate

# 1. Launch replay pipeline in background (REST server on 127.0.0.1:12048)
bash test_charts/start_continuous_pipeline.sh

# 2. Open Web Telemetry & Steering Dashboard
python test_charts/kotekan_tracker_dashboard.py
# -> Open browser to http://127.0.0.1:8050

# 3. Stop pipeline when finished
bash test_charts/stop_continuous_pipeline.sh
```

### Mode B: Live 32-Antenna Online Pipeline (DPDK + RFSoC)
Ingests live network packets over DPDK port 0, zeroes dropped samples, runs GPU beam tracking, and writes complex voltages:

```bash
cd ~/charts_workspace/kotekan
source .venv/bin/activate

# 1. Launch live 32-antenna tracker
bash charts/start_32antennas_tracker.sh

# Or run directly in the foreground:
./build/kotekan/kotekan -c charts/32antennas_tracker.yaml -b 127.0.0.1:12048

# 2. Monitor logs
tail -f /tmp/kotekan_32ant_tracker.log
```

### Mode C: Centralized Simulation & Verification Suite (PC & Trillium)
Simulates realistic physical astronomical backgrounds (Sun UP / Sun DOWN), transient events (FRBs, pulsars, RFI), and correlates/tracks formed beams:

```bash
cd ~/charts_workspace/kotekan
source .venv/bin/activate

# 1. Fast verification test on local PC (< 1 min run time):
python test_charts/charts_sim.py pipeline --preset quick

# 2. Full 1-minute realistic observation (64 antennas, 8 formed beams, Carén site):
python test_charts/charts_sim.py pipeline --preset 1min --profile day

# 3. Dry-run test (verifies data generation and configs without GPU execution):
python test_charts/charts_sim.py pipeline --preset quick --dry-run

# 4. Slurm execution on Trillium cluster (the test_charts/slurm/ launchers exist only on the
#    kotekan_charts_trillium branch; they are not present on kotekan_charts_next):
git checkout kotekan_charts_trillium
sbatch test_charts/slurm/trillium_pipeline.slurm
```

---

## 6. Live Steering & Telemetry CLI

Dynamically inspect and steer active beam slots over the REST API without stopping the pipeline:

```bash
# Check pipeline status & active beam slots
python test_charts/kotekan_tracker_control.py status

# Watch live metrics in terminal (updates at 4 Hz)
python test_charts/kotekan_tracker_control.py watch

# Steer Beam 0 in direction cosines (l, m)
python test_charts/kotekan_tracker_control.py steer-lm --beam 0 --l0 0.05 --m0 -0.02

# Steer Beam 1 to celestial coordinates (RA, Dec)
python test_charts/kotekan_tracker_control.py steer-radec --beam 1 --ra 83.633 --dec 22.014

# Mask/isolate a faulty antenna element
python test_charts/kotekan_tracker_control.py mask-antenna --element 14 --state bad
```

---

## 7. Common Troubleshooting

| Issue | Cause | Fix |
| :--- | :--- | :--- |
| `nvcc fatal: Unsupported gpu architecture 'compute_61'` | CUDA 13+ dropped Pascal architecture (`compute_61`). | Pass `-DCMAKE_CUDA_ARCHITECTURES=89` (or `120`/`native`) when running CMake. |
| `Unable to synchronously open file (file signature not found)` | `h5py` attempted to open a raw `.bin` file as HDF5. | Guarded automatically by `h5py.is_hdf5()`; ensure `baseband_virtual.h5` exists or rely on `.bin` fallback. |
| `cudaErrorMemoryAllocation: out of memory` | GPU buffer depth or beam count too large for VRAM. | Set `gpu_buffer_depth: 2` and `max_beams: 4` in YAML config for 24GB RTX 4090. RTX 5090 (32GB) can use `max_beams: 8`. |
| `dpdkCore: Port 0 not available` | Hugepages not allocated or NIC unbound. | Setup 2MB/1GB hugepages and bind the network interface to `vfio-pci` or `uio_pci_generic`. |
