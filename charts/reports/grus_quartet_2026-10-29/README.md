# Direct Beam Tracker Deployment & Posterior Analysis — Grus Quartet (NGC 7582)

## 1. Hardware Mapping & Root-Cause Diagnosis of the Original CUDA Masking Kernel

### 1.1 First-Principles Hardware Mapping Verification ([`rfsocHandlerShuffle.hpp`](../../../lib/dpdk/rfsocHandlerShuffle.hpp#L472-L480))
In [`rfsocHandlerShuffle.hpp`](../../../lib/dpdk/rfsocHandlerShuffle.hpp#L472-L480), the DPDK packet handler places each RFSoC's 32 channels into the 64-element frame buffer in strictly descending order (`antenna_output_offset = (num_rfsocs - 1 - rf_id) * 32`, `phys_ant = 63 - raw_elem`):
- **RFSoC 0 (`rf_id = 0`)**: written to `raw_elem = 32..63` (`ADC_D` chains `0..7` $\to$ `phys_ant = 0..7` $\to$ `raw_elem = 63..56`).
- **RFSoC 1 (`rf_id = 1`)**: written to `raw_elem = 0..31` (`ADC_D` chains `0..7` $\to$ `phys_ant = 32..39` $\to$ `raw_elem = 31..24`; `ADC_C` chains `0..7` $\to$ `phys_ant = 40..47` $\to$ `raw_elem = 23..16`).

| Feed Label | Hardware Port / ADC Chain | Physical Antenna (`phys_ant`) | Kotekan Raw Buffer Index (`raw_elem = 63 - phys_ant`) | In-Band 75th-Pct Passband Power ($\text{ADC}^2$) | Cross-Antenna Spatial Coherence $\langle|\gamma_{ab}|\rangle$ | Status |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **C0** | `rfsoc 0 adc d chain 0` | **`0`** | **`63`** | `3.883` | `0.2604` | **ALIVE** |
| **C1** | `rfsoc 0 adc d chain 5` | **`5`** | **`58`** | `4.088` | `0.2607` | **ALIVE** |
| **C6** | `rfsoc 1 adc d chain 0` | **`32`** | **`31`** | `3.796` | `0.2481` | **ALIVE** |
| **C7** | `rfsoc 1 adc d chain 5` | **`37`** | **`26`** | `3.762` | `0.2654` | **ALIVE** |
| **CHORD** | `rfsoc 1 adc c chain 0` | **`40`** | **`23`** | `4.188` | `0.1817` | **ALIVE** |
| *59 Unconnected Inputs* | *Remaining RFSoC 0 & 1 chains* | *All others* | *All others* | `3.10–3.55` (median `3.30`) | `0.132–0.153` (median `0.1365`) | **DEAD (Masked)** |

---

### 1.2 Three Flaws Identified & Fixed in [`cudaAntennaMask.cu`](../../../lib/cuda/cudaAntennaMask.cu) and [`cudaAntennaMaskCommand.cpp`](../../../lib/cuda/cudaAntennaMaskCommand.cpp)

1. **Broadband Unweighted Sums Masked Narrower Analog Filters & Let Narrowband ADC Spurs Leak Through**:
   - Unterminated 4-bit ADC chains sit at a quantized thermal floor around `3.00–3.45 ADC^2`. Several unconnected chains (`raw = 15, 17, 21, 22, 52`) exhibit narrowband internal ADC clock/interleaving spurs in a few channels (`power up to 43.5 ADC^2`), inflating their broadband mean above `3.48–5.22 ADC^2`.
   - Meanwhile, the **CHORD** feed (`pos 40, raw = 23`) has a steeper analog bandpass roll-off at the band edges (`q10 = 2.19 ADC^2`), pulling its unweighted 336-channel mean down to `3.47–3.53 ADC^2` even though its active passband power (`q75 = 4.188–5.227 ADC^2`) is the highest in the array.
2. **Subband 1 (`400.8–501.3 MHz`, Out-of-Band) Overwrote the Shared Mask of Subband 0 (`300.0–400.5 MHz`, In-Band)**:
   - Both `gpu_0` and `gpu_1` wrote to static `_shared_mask` and `_shared_metrics` without subband awareness, allowing out-of-band noise above the 400 MHz analog filter to overwrite the in-band mask.
3. **Zero-Filled Missing Packets Biased Broadband Moment Normalization**:
   - Dividing raw sums by `total_spectra` without scaling by the fraction of non-zero frequency channels artificially depressed `elem_power` on partially filled startup frames.

> [!IMPORTANT]
> **Physics-Based Solution Implemented**:
> Added [`inspect_channel_coherence_kernel`](../../../lib/cuda/cudaAntennaMask.cu#L417-L535) (`336` blocks $\times$ `64` threads) to [`cudaAntennaMask.cu`](../../../lib/cuda/cudaAntennaMask.cu) and updated [`cudaAntennaMaskCommand::execute()`](../../../lib/cuda/cudaAntennaMaskCommand.cpp#L759-L1085) to compute **per-channel 75th-percentile passband power** and **cross-antenna spatial coherence** (excluding same-RFSoC ADC-tile interleaving crosstalk $|a - b| \bmod 8 == 0$ and high-kurtosis spur channels), while making in-band Subband 0 authoritative for `_shared_mask`. No antenna indices or positions are hardcoded anywhere.

---

## 2. Run Summary & Data Budget Verification (`/data/tracker/dump_A` & `/data/tracker/dump_B`)

| Run | File | Frames | Exact Size (Bytes) | Status |
|:---:|:---|:---:|---:|:---:|
| **Run A** (No Mask) | `raw_baseband_64ant_freq0_0000000.bin` | 2 | `660,602,882` (~660.6 MB) | `CLEAN_EXIT` |
| **Run A** (No Mask) | `formed_beams_64ant_freq0_0000000.bin` | 4 | `165,150,724` (~165.2 MB) | `CLEAN_EXIT` |
| **Run A** (No Mask) | `formed_beams_64ant_freq1_0000000.bin` | 4 | `165,150,724` (~165.2 MB) | `CLEAN_EXIT` |
| **Run B** (Mask ON) | `alive_baseband_64ant_freq0_0000000.bin` | 9 | `325,140,489` (~325.1 MB) | `CLEAN_EXIT` |
| **Run B** (Mask ON) | `alive_baseband_64ant_freq1_0000000.bin` | 9 | `325,140,489` (~325.1 MB) | `CLEAN_EXIT` |
| **Run B** (Mask ON) | `formed_beams_64ant_freq0_0000000.bin` | 4 | `165,150,724` (~165.2 MB) | `CLEAN_EXIT` |
| **Run B** (Mask ON) | `formed_beams_64ant_freq1_0000000.bin` | 4 | `165,150,724` (~165.2 MB) | `CLEAN_EXIT` |

---

## 3. Section 7 — Kotekan & GPU Kernel Benchmarks (vs. 51.2 ms Frame Budget)

![Section 7 GPU Pipeline Benchmarks](fig0_gpu_pipeline_benchmarks.png)

| Benchmark Metric | Run A (No Mask, 64-Ant) | Run B (Mask ON, 5 Alive / 7 Slots) | Real-Time Budget / Target |
|:---|:---:|:---:|:---:|
| **End-to-End GPU Frame Pipeline (`GPU0` / `GPU1`)** | `7.67 ms` / `13.12 ms` (`15.0%`–`25.6%` duty) | `14.38 ms` / `14.98 ms` (`28.1%`–`29.3%` duty) | `< 51.2 ms` (`100%` budget) |
| **NVMe Disk Write (`formed_beams`, 41.3 MB/frame)** | `12.25 ms` | `11.40 ms` (S0) / `11.54 ms` (S1) | `< 51.2 ms` |
| **NVMe Disk Write (`alive_baseband`, 36.1 MB/frame)** | N/A | `11.95 ms` (S0) / `16.11 ms` (S1) | `< 51.2 ms` |
| **Peak Ring Buffer Occupancy (`network_capture_buf`)** | `3 / 20` frames (`15%`) | `9 / 20` frames (`45%`) | `< 20` frames |
| **RTX 5090 Peak SM / Mem / Power (`nvidia-smi dmon`)** | `27% SM` / `3% Mem` / `103 W` | `35% SM` / `5% Mem` / `124 W` | Headroom `> 65%` |
| **Peak PCIe H2D RX Throughput (`rxpci`)** | `34.78 GB/s` | `30.72 GB/s` | PCIe Gen5 x16 (`~64 GB/s`) |

---

## 4. Section 8 — Posterior Data Analysis & Physics Validation (`dump_A` vs. `dump_B`)

````carousel
![Run A 64-Antenna Health & Histograms](fig1_runA_64ant_health_histograms.png)
<!-- slide -->
![Section 8.2 Automatic 2D Passband & Coherence Separation](fig4_masking_verification_runA_vs_runB.png)
<!-- slide -->
![Run B Compacted Alive Baseband PSD & Waterfalls](fig2_bandpass_and_waterfalls.png)
<!-- slide -->
![7x7 Zero-Lag Cross-Correlation Coherence Matrix](fig3_7x7_coherence_matrix.png)
<!-- slide -->
![Astrometric Trajectory & Geometric Delays on Grus Quartet](fig5_astrometry_and_delays.png)
<!-- slide -->
![Formed Beam Waterfalls & Band-Averaged Power P(t)](fig6_formed_beam_waterfall_and_power.png)
<!-- slide -->
![Coherent vs. Incoherent Power Ratio eta(f)](fig7_coherent_incoherent_ratio_eta.png)
<!-- slide -->
![CPU Golden-Reference Delay-and-Sum Replay vs. GPU Formed Beams](fig8_cpu_golden_replay_residual.png)
````

### Key Physics & Verification Results
1. **100% Exact Match Between Offline Python Detector & Live CUDA Kernel (Section 8.2)**:
   - Both the offline Python detector (on Run A raw 64-antenna baseband) and the live [`cudaAntennaMaskCommand`](../../../lib/cuda/cudaAntennaMaskCommand.cpp) stage in Run B automatically isolated the exact 5 physically connected antennas:
     - **Raw indices**: `[23, 26, 31, 58, 63]`
     - **Physical antennas**: `[40, 37, 32, 5, 0]` (`CHORD, C7, C6, C1, C0`)
   - In Run B (`max_alive_antennas: 7`), slots `0..4` contain the extracted 4-bit complex voltages of `[23, 26, 31, 58, 63]` (`mean power = 3.37–4.31 ADC^2`), while slots `5..6` are cleanly zero-padded (`0.00 ADC^2`).
2. **Astrometric & Geometric Delay Verification on Grus Quartet (Section 8.3a)**:
   - At observation epoch `UTC = 1793322700.83` (`LST = 23.029 h`), the Grus Quartet (`RA = 349.5979°, Dec = -42.3706°`) was near transit at **altitude `80.50°`**, **azimuth `159.20°`**.
   - Logged Kotekan direction cosines `(l0, m0, n0) = (0.05363, -0.15664, 0.98620)` match the C++ J2000 reference `(0.05367, -0.15664, 0.98620)` to `< 4.2e-5` (`0.002°`), and match ICRS-to-observed `astropy` `(0.05863, -0.15433)` within J2000-to-apparent precession/aberration (`~0.3°`, well inside the `~14°` primary beam).
3. **Formed-Beam SNR Boost & Coherent-to-Incoherent Ratio $\eta(f)$ (Section 8.3b–c)**:
   - Masking out the 59 unterminated ADC noise channels increased Stream 0 mean formed-beam power from **`3.146`** (Run A, diluted by 59 noise channels) to **`3.417`** (Run B, 5 live antennas).
   - Across Stream 0 (`300.0–400.5 MHz`, in-band), the median coherent-to-incoherent power ratio is **$\text{median}\,\eta(f) = 1.0228 > 1.0$**, demonstrating net positive coherent addition across the 5 live feeds, while out-of-band Stream 1 (`400.8–501.3 MHz`) sits at the incoherent baseline ($\text{median}\,\eta(f) = 0.9794 \approx 1.0$).
4. **CPU Golden-Reference Delay-and-Sum Replay (Section 8.3d)**:
   - **Run A (64-antenna raw baseband $\to$ formed beam, `seq = 83083468800`)**: `rel_rms_err = 3.9623e-07`, `max_abs_err = 2.7804e-06`.
   - **Run B (5-antenna compacted alive baseband $\to$ formed beam, `seq = 83868549120`)**: `rel_rms_err = 4.5413e-07`, `max_abs_err = 7.5281e-06` — proving to float32 machine precision that the GPU beamformer summed **only** the 5 automatically detected live antennas (`[23, 26, 31, 58, 63]`, `phys = [40, 37, 32, 5, 0]`).
