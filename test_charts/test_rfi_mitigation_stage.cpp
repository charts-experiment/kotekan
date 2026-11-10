#include "cudaDirectBeamTracker.hpp"
#include "cudaRfiMitigation.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>

using namespace kotekan;

namespace {

inline int4x2_t pack_int4x2_signed(int re, int im) {
    // 15-level quantizer: symmetric clipping to [-7, +7] (F-Engine quantizer.v)
    re = std::max(-7, std::min(7, re));
    im = std::max(-7, std::min(7, im));
    std::uint8_t lo = static_cast<std::uint8_t>(re & 0x0F);
    std::uint8_t hi = static_cast<std::uint8_t>((im & 0x0F) << 4);
    int4x2_t out;
    out.val = static_cast<std::uint8_t>(hi | lo);
    return out;
}

} // namespace

int main() {
    std::printf("====================================================================================================\n");
    std::printf(" CHARTS Radio Telescope: Lightweight Real-Time GPU RFI Mitigation Stage Test & Benchmark Suite\n");
    std::printf(" Grounded in F-Engine Quantization Bias Theory (Mena-Parra et al. 2018; Target q-RMS=3.5)\n");
    std::printf(" Verifying: Scale-Invariant SK, Temporal Burst MAD, Outlier-Rejected Passband MAD,\n");
    std::printf("            Quantizer Saturation Clipping, Pre-Gain Equalization, and Voltage Blanking\n");
    std::printf("====================================================================================================\n\n");

    const std::size_t n_time = 15360;
    const std::size_t n_freq = 336;
    const std::size_t n_elem = 64;
    const std::size_t chunk_size = 256;
    const std::size_t n_chunks = n_time / chunk_size; // 60 chunks
    const std::size_t total_cells = n_chunks * n_freq;

    // 4 calibrated corner antennas [0, 4, 32, 36] -> raw descending elements [63, 59, 31, 27]
    const std::vector<int> active_raw = {63, 59, 31, 27};
    const int num_active = static_cast<int>(active_raw.size());
    // F-Engine operating point: target q-RMS = 3.5 -> sigma = 1.70 total, sigma_c = 1.20 LSB per component
    const float ant_sigmas[4] = {1.10f, 1.20f, 1.35f, 1.50f}; // Per-antenna gains around optimal window

    std::printf("Configuration:\n");
    std::printf("  - Time Samples/Frame  : %zu (51.20 ms frame)\n", n_time);
    std::printf("  - Frequency Channels  : %zu (100.8 MHz subband)\n", n_freq);
    std::printf("  - Total Elements      : %zu (Active calibrated subset: 4 corner elements [63, 59, 31, 27])\n", n_elem);
    std::printf("  - SK Time Chunk Size  : %zu samples (853.3 us, %zu chunks/frame)\n", chunk_size, n_chunks);
    std::printf("  - F-Engine Baseline   : 15-level quantizer [-7..+7], target q-RMS=3.5 (sigma_c ~ 1.20 LSB)\n\n");

    const std::size_t frame_elems = n_time * n_freq * n_elem;
    std::vector<int4x2_t> h_voltages(frame_elems);
    std::memset(h_voltages.data(), 0, frame_elems * sizeof(int4x2_t));

    std::mt19937 rng(20261107);
    std::normal_distribution<float> norm(0.0f, 1.0f);

    // 1. Synthesize clean thermal noise with unequal per-antenna gains + realistic passband slope
    for (std::size_t t = 0; t < n_time; ++t) {
        for (std::size_t f = 0; f < n_freq; ++f) {
            const float passband_scale = 1.0f + 0.15f * std::sin(3.14159265f * float(f) / float(n_freq));
            const std::size_t base = (t * n_freq + f) * n_elem;
            for (int a = 0; a < num_active; ++a) {
                const int raw_e = active_raw[a];
                const float sig = ant_sigmas[a] * passband_scale;
                const int re = static_cast<int>(std::round(norm(rng) * sig));
                const int im = static_cast<int>(std::round(norm(rng) * sig));
                h_voltages[base + raw_e] = pack_int4x2_signed(re, im);
            }
        }
    }

    // Allocate device buffers
    int4x2_t* d_voltages = nullptr;
    int* d_active_antennas = nullptr;
    float* d_chunk_power = nullptr;
    float* d_chunk_sk = nullptr;
    float* d_chunk_sat_frac = nullptr;
    std::uint8_t* d_static_notch = nullptr;
    float* d_channel_gains = nullptr;
    std::uint8_t* d_chunk_rfi_mask = nullptr;
    RfiChannelStats* d_channel_stats = nullptr;

    cudaMalloc(&d_voltages, frame_elems * sizeof(int4x2_t));
    cudaMalloc(&d_active_antennas, num_active * sizeof(int));
    cudaMalloc(&d_chunk_power, total_cells * sizeof(float));
    cudaMalloc(&d_chunk_sk, total_cells * sizeof(float));
    cudaMalloc(&d_chunk_sat_frac, total_cells * sizeof(float));
    cudaMalloc(&d_static_notch, n_freq * sizeof(std::uint8_t));
    cudaMalloc(&d_channel_gains, n_freq * sizeof(float));
    cudaMalloc(&d_chunk_rfi_mask, total_cells * sizeof(std::uint8_t));
    cudaMalloc(&d_channel_stats, n_freq * sizeof(RfiChannelStats));

    cudaMemcpy(d_voltages, h_voltages.data(), frame_elems * sizeof(int4x2_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_active_antennas, active_raw.data(), num_active * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_static_notch, 0, n_freq * sizeof(std::uint8_t));
    cudaMemset(d_channel_gains, 0, n_freq * sizeof(float));

    RfiMitigationConfig cfg;
    cfg.enabled = true;
    cfg.blank_voltages_enabled = true;
    cfg.time_chunk_size = chunk_size;
    cfg.sample_stride = 1;
    cfg.enable_sk_check = true;
    cfg.sk_sigma_threshold = 4.5f;
    cfg.enable_time_burst_check = true;
    cfg.burst_sigma_threshold = 5.0f;
    cfg.enable_freq_mad_check = true;
    cfg.freq_window_half_width = 8;
    cfg.mad_sigma_threshold = 6.0f;
    cfg.enable_saturation_check = true;
    cfg.saturation_fraction_threshold = 0.02f;
    cfg.enable_morphology_sir = true;
    cfg.edge_channels_to_ignore = 16;

    // >>> TEST 1: Scale-Invariant Thermal Noise Baseline (Target q-RMS=3.5) <<<
    launch_inspect_rfi_chunks(d_voltages, d_active_antennas, num_active, d_chunk_power, d_chunk_sk,
                              d_chunk_sat_frac, n_time, n_freq, n_elem, chunk_size, 1, nullptr, false);
    launch_evaluate_rfi_flags(d_chunk_power, d_chunk_sk, d_chunk_sat_frac, d_static_notch,
                              d_channel_gains, d_chunk_rfi_mask, d_channel_stats, n_chunks, n_freq,
                              num_active, chunk_size, cfg, nullptr);
    cudaDeviceSynchronize();

    std::vector<RfiChannelStats> h_stats(n_freq);
    std::vector<std::uint8_t> h_mask(total_cells);
    cudaMemcpy(h_stats.data(), d_channel_stats, n_freq * sizeof(RfiChannelStats), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mask.data(), d_chunk_rfi_mask, total_cells * sizeof(std::uint8_t), cudaMemcpyDeviceToHost);

    double mean_thermal_sk = 0.0;
    int false_positives = 0;
    for (std::size_t f = 0; f < n_freq; ++f) {
        mean_thermal_sk += h_stats[f].mean_sk;
    }
    mean_thermal_sk /= double(n_freq);
    for (std::size_t i = 0; i < total_cells; ++i) {
        if (h_mask[i] != 0)
            false_positives++;
    }

    std::printf(">>> TEST 1: Scale-Invariant Thermal Noise Baseline (Target q-RMS=3.5) <<<\n");
    std::printf("  Measured Mean Spectral Kurtosis : %.4f (Expected ~0.90 - 1.02 for 15-level Gaussian)\n", mean_thermal_sk);
    std::printf("  False Positive Cells Flagged    : %d / %zu (%.3f%%)\n", false_positives, total_cells,
                100.0 * double(false_positives) / double(total_cells));
    if (std::abs(mean_thermal_sk - 0.95) > 0.08 || false_positives > 5) {
        std::printf("  Result: [FAILED]\n");
        return 1;
    }
    std::printf("  Result: [PASSED] Scale-invariant per-antenna SK is immune to inter-antenna gain spread!\n\n");

    // >>> TEST 2: Multi-Modal RFI Injection (Impulsive, Telemetry, CW, Narrowband, Notch) <<<
    const std::size_t f_impulsive = 45;  // Sub-chunk radar pulse in chunk 12
    const std::size_t c_impulsive = 12;

    const std::size_t f_burst = 130;     // Full-chunk telemetry burst in chunk 33
    const std::size_t c_burst = 33;

    const std::size_t f_cw = 210;        // Constant-modulus phase-rotating CW carrier
    const std::size_t f_narrow = 280;    // Stationary narrowband noise jammer
    const std::size_t f_notch = 310;     // Static notch channel
    const std::size_t f_sat = 90;        // Heavy saturating radar burst hitting +-7
    const std::size_t c_sat = 20;

    // Inject sub-chunk impulsive burst (20 samples hitting +7/-7 inside chunk 12 of channel 45)
    for (std::size_t dt = 40; dt < 60; ++dt) {
        const std::size_t t = c_impulsive * chunk_size + dt;
        const std::size_t base = (t * n_freq + f_impulsive) * n_elem;
        for (int a = 0; a < num_active; ++a) {
            h_voltages[base + active_raw[a]] = pack_int4x2_signed(7, -7);
        }
    }

    // Inject heavy saturating radar burst in chunk 20 of channel 90 (35 samples saturating at +-7)
    for (std::size_t dt = 10; dt < 45; ++dt) {
        const std::size_t t = c_sat * chunk_size + dt;
        const std::size_t base = (t * n_freq + f_sat) * n_elem;
        for (int a = 0; a < num_active; ++a) {
            h_voltages[base + active_raw[a]] = pack_int4x2_signed(7, 7);
        }
    }

    // Inject full-chunk telemetry burst in chunk 33 of channel 130
    for (std::size_t dt = 0; dt < chunk_size; ++dt) {
        const std::size_t t = c_burst * chunk_size + dt;
        const std::size_t base = (t * n_freq + f_burst) * n_elem;
        for (int a = 0; a < num_active; ++a) {
            const int re = static_cast<int>(std::round(norm(rng) * 5.5f));
            const int im = static_cast<int>(std::round(norm(rng) * 5.5f));
            h_voltages[base + active_raw[a]] = pack_int4x2_signed(re, im);
        }
    }

    // Inject constant-envelope rotating CW tone on channel 210
    for (std::size_t t = 0; t < n_time; ++t) {
        const float phase = 0.37f * static_cast<float>(t);
        const int re = static_cast<int>(std::round(5.5f * std::cos(phase)));
        const int im = static_cast<int>(std::round(5.5f * std::sin(phase)));
        const std::size_t base = (t * n_freq + f_cw) * n_elem;
        for (int a = 0; a < num_active; ++a) {
            h_voltages[base + active_raw[a]] = pack_int4x2_signed(re, im);
        }
    }

    // Inject stationary narrowband wideband-modulated noise jammer on channel 280
    for (std::size_t t = 0; t < n_time; ++t) {
        const std::size_t base = (t * n_freq + f_narrow) * n_elem;
        for (int a = 0; a < num_active; ++a) {
            const int re = static_cast<int>(std::round(norm(rng) * 5.0f));
            const int im = static_cast<int>(std::round(norm(rng) * 5.0f));
            h_voltages[base + active_raw[a]] = pack_int4x2_signed(re, im);
        }
    }

    std::vector<std::uint8_t> h_notch(n_freq, 0);
    h_notch[f_notch] = 1;

    cudaMemcpy(d_voltages, h_voltages.data(), frame_elems * sizeof(int4x2_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_static_notch, h_notch.data(), n_freq * sizeof(std::uint8_t), cudaMemcpyHostToDevice);

    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    cudaEventRecord(ev_start);
    launch_inspect_rfi_chunks(d_voltages, d_active_antennas, num_active, d_chunk_power, d_chunk_sk,
                              d_chunk_sat_frac, n_time, n_freq, n_elem, chunk_size, 1, nullptr, false);
    launch_evaluate_rfi_flags(d_chunk_power, d_chunk_sk, d_chunk_sat_frac, d_static_notch,
                              d_channel_gains, d_chunk_rfi_mask, d_channel_stats, n_chunks, n_freq,
                              num_active, chunk_size, cfg, nullptr);
    launch_blank_rfi_chunks(d_voltages, d_chunk_rfi_mask, n_time, n_freq, n_elem, chunk_size,
                            nullptr, false);
    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop);

    cudaMemcpy(h_stats.data(), d_channel_stats, n_freq * sizeof(RfiChannelStats), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mask.data(), d_chunk_rfi_mask, total_cells * sizeof(std::uint8_t), cudaMemcpyDeviceToHost);

    const std::uint8_t imp_flag = h_mask[c_impulsive * n_freq + f_impulsive];
    const std::uint8_t sat_flag = h_mask[c_sat * n_freq + f_sat];
    const std::uint8_t burst_flag = h_mask[c_burst * n_freq + f_burst];
    const std::uint32_t cw_flags = h_stats[f_cw].flag_bits;
    const std::uint32_t narrow_flags = h_stats[f_narrow].flag_bits;
    const std::uint32_t notch_flags = h_stats[f_notch].flag_bits;

    std::printf(">>> TEST 2: Multi-Modal RFI Detection & Classification <<<\n");
    std::printf("  1. Sub-Chunk Impulsive Pulse (ch=%zu, chunk=%zu): max_SK=%.2f, flag=0x%02x (RFI_SK_IMPULSIVE=%d)\n",
                f_impulsive, c_impulsive, h_stats[f_impulsive].max_sk, imp_flag,
                (imp_flag & RFI_SK_IMPULSIVE) != 0);
    std::printf("  2. Quantizer Saturation Radar (ch=%zu, chunk=%zu): flag=0x%02x (RFI_SATURATION_CLIP=%d)\n",
                f_sat, c_sat, sat_flag, (sat_flag & RFI_SATURATION_CLIP) != 0);
    std::printf("  3. Telemetry Chunk Burst     (ch=%zu, chunk=%zu): flag=0x%02x (RFI_TIME_BURST=%d)\n",
                f_burst, c_burst, burst_flag, (burst_flag & RFI_TIME_BURST) != 0);
    std::printf("  4. Constant-Modulus CW Tone  (ch=%zu)          : mean_SK=%.3f, MAD_Z=%.1f, flags=0x%02x\n",
                f_cw, h_stats[f_cw].mean_sk, h_stats[f_cw].freq_mad_z, cw_flags);
    std::printf("  5. Narrowband Noise Jammer   (ch=%zu)          : MAD_Z=%.1f, flagged_frac=%.2f, flags=0x%02x\n",
                f_narrow, h_stats[f_narrow].freq_mad_z, h_stats[f_narrow].flagged_chunk_fraction, narrow_flags);
    std::printf("  6. Static Notch Band         (ch=%zu)          : flagged_frac=%.2f, flags=0x%02x\n",
                f_notch, h_stats[f_notch].flagged_chunk_fraction, notch_flags);

    if ((imp_flag & RFI_SK_IMPULSIVE) == 0 || (sat_flag & RFI_SATURATION_CLIP) == 0
        || (burst_flag & RFI_TIME_BURST) == 0
        || (cw_flags & (RFI_SK_CW_TONE | RFI_FREQ_MAD)) == 0
        || (narrow_flags & RFI_FREQ_MAD) == 0 || (notch_flags & RFI_STATIC_NOTCH) == 0) {
        std::printf("  Result: [FAILED] One or more RFI injections were not detected!\n");
        return 1;
    }
    std::printf("  Result: [PASSED] 100%% detection across all 6 RFI classes!\n\n");

    // >>> TEST 3: In-Place Blanking & Surgical Preservation <<<
    std::vector<int4x2_t> h_cleaned(frame_elems);
    cudaMemcpy(h_cleaned.data(), d_voltages, frame_elems * sizeof(int4x2_t), cudaMemcpyDeviceToHost);

    long long imp_chunk_energy_after = 0;
    for (std::size_t dt = 0; dt < chunk_size; ++dt) {
        const std::size_t t = c_impulsive * chunk_size + dt;
        const std::size_t base = (t * n_freq + f_impulsive) * n_elem;
        for (std::size_t e = 0; e < n_elem; ++e) {
            imp_chunk_energy_after += h_cleaned[base + e].val;
        }
    }

    long long clean_chunk_energy_after = 0;
    for (std::size_t dt = 0; dt < chunk_size; ++dt) {
        const std::size_t t = (c_impulsive + 1) * chunk_size + dt;
        const std::size_t base = (t * n_freq + f_impulsive) * n_elem;
        for (int a = 0; a < num_active; ++a) {
            clean_chunk_energy_after += std::abs(int(h_cleaned[base + active_raw[a]].val));
        }
    }

    std::printf(">>> TEST 3: Surgical In-Place Blanking & Real-Time GPU Latency <<<\n");
    std::printf("  Contaminated Chunk Residual Bytes : %lld (Expected 0)\n", imp_chunk_energy_after);
    std::printf("  Adjacent Clean Chunk Bytes Sum    : %lld (Expected > 0, surgical preservation)\n",
                clean_chunk_energy_after);
    std::printf("  Full 3-Pass GPU Execution Time    : %.3f ms (%.2f%% of 51.20 ms frame budget)\n",
                elapsed_ms, 100.0 * elapsed_ms / 51.20);

    if (imp_chunk_energy_after != 0 || clean_chunk_energy_after == 0) {
        std::printf("  Result: [FAILED] Blanking verification failed!\n");
        return 1;
    }
    std::printf("  Result: [PASSED] Surgical chunk blanking verified at %.3f ms/frame!\n\n", elapsed_ms);

    std::printf("====================================================================================================\n");
    std::printf(" RFI MITIGATION STAGE TEST RESULT: ALL TESTS PASSED (100%% OPERATIONAL)\n");
    std::printf("====================================================================================================\n");

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    cudaFree(d_voltages);
    cudaFree(d_active_antennas);
    cudaFree(d_chunk_power);
    cudaFree(d_chunk_sk);
    cudaFree(d_chunk_sat_frac);
    cudaFree(d_static_notch);
    cudaFree(d_channel_gains);
    cudaFree(d_chunk_rfi_mask);
    cudaFree(d_channel_stats);
    return 0;
}
