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
    std::fflush(stdout);

    const std::size_t frame_elems = n_time * n_freq * n_elem;
    std::vector<int4x2_t> h_voltages(frame_elems);
    std::memset(h_voltages.data(), 0, frame_elems * sizeof(int4x2_t));

    std::mt19937 rng(20261107);
    std::normal_distribution<float> norm(0.0f, 1.0f);

    std::vector<float> passband_scale(n_freq);
    for (std::size_t f = 0; f < n_freq; ++f) {
        passband_scale[f] = 1.0f + 0.15f * std::sin(3.14159265f * float(f) / float(n_freq));
    }

    // 1. Synthesize clean thermal noise with unequal per-antenna gains + realistic passband slope
    for (std::size_t t = 0; t < n_time; ++t) {
        for (std::size_t f = 0; f < n_freq; ++f) {
            const float pb_scale = passband_scale[f];
            const std::size_t base = (t * n_freq + f) * n_elem;
            for (int a = 0; a < num_active; ++a) {
                const int raw_e = active_raw[a];
                const float sig = ant_sigmas[a] * pb_scale;
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
        std::fflush(stdout);
        return 1;
    }
    std::printf("  Result: [PASSED] Scale-invariant per-antenna SK is immune to inter-antenna gain spread!\n\n");
    std::fflush(stdout);

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
        std::fflush(stdout);
        return 1;
    }
    std::printf("  Result: [PASSED] 100%% detection across all 6 RFI classes!\n\n");
    std::fflush(stdout);

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
        std::fflush(stdout);
        return 1;
    }
    std::printf("  Result: [PASSED] Surgical chunk blanking verified at %.3f ms/frame!\n\n", elapsed_ms);
    std::fflush(stdout);

    // >>> TEST 4: Excision Exempt Channels & Beam Power Blanking Compensation <<<
    std::vector<std::uint8_t> h_exempt(n_freq, 0);
    h_exempt[f_burst] = 1; // Exempt channel 130 from blanking
    std::uint8_t* d_exempt = nullptr;
    cudaMalloc(&d_exempt, n_freq * sizeof(std::uint8_t));
    cudaMemcpy(d_exempt, h_exempt.data(), n_freq * sizeof(std::uint8_t), cudaMemcpyHostToDevice);

    launch_evaluate_rfi_flags(d_chunk_power, d_chunk_sk, d_chunk_sat_frac, d_static_notch,
                              d_channel_gains, d_chunk_rfi_mask, d_channel_stats, n_chunks, n_freq,
                              num_active, chunk_size, cfg, nullptr, d_exempt);
    cudaDeviceSynchronize();

    cudaMemcpy(h_mask.data(), d_chunk_rfi_mask, total_cells * sizeof(std::uint8_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_stats.data(), d_channel_stats, n_freq * sizeof(RfiChannelStats), cudaMemcpyDeviceToHost);

    const std::uint8_t exempt_chunk_flag = h_mask[c_burst * n_freq + f_burst];
    const std::uint32_t exempt_chan_flags = h_stats[f_burst].flag_bits;

    std::printf(">>> TEST 4: Excision Exempt Channels & Beam Power Blanking Compensation <<<\n");
    std::printf("  1. Exempt Telemetry Channel (%zu): chunk_mask=0x%02x (Expected 0x00 / CLEAN), stats_flags=0x%02x (Expected TIME_BURST)\n",
                f_burst, exempt_chunk_flag, exempt_chan_flags);

    if (exempt_chunk_flag != RFI_CLEAN || (exempt_chan_flags & RFI_TIME_BURST) == 0) {
        std::printf("  Result: [FAILED] Excision exempt channel handling failed!\n");
        std::fflush(stdout);
        return 1;
    }

    // Verify beam power reduction with blanking compensation
    const std::size_t max_beams = 1;
    const std::size_t n_pol = 1;
    const std::size_t beam_samples = n_time * n_freq * max_beams * n_pol;
    std::vector<float2> h_formed_beams(beam_samples);
    const float true_pwr = 4.0f; // amplitude 2.0 -> |v|^2 = 4.0
    for (std::size_t t = 0; t < n_time; ++t) {
        for (std::size_t f = 0; f < n_freq; ++f) {
            h_formed_beams[t * n_freq + f] = make_float2(2.0f, 0.0f);
        }
    }

    // Blank 10 out of 60 chunks in channel 0 in the mask
    std::vector<std::uint8_t> h_comp_mask(total_cells, 0);
    const std::size_t f_test = 0;
    const std::size_t blanked_chunks = 10;
    for (std::size_t c = 0; c < blanked_chunks; ++c) {
        h_comp_mask[c * n_freq + f_test] = RFI_TIME_BURST;
    }
    // Zero out those blanked chunk samples in formed beams
    for (std::size_t c = 0; c < blanked_chunks; ++c) {
        for (std::size_t dt = 0; dt < chunk_size; ++dt) {
            const std::size_t t = c * chunk_size + dt;
            h_formed_beams[t * n_freq + f_test] = make_float2(0.0f, 0.0f);
        }
    }

    float2* d_formed = nullptr;
    float* d_power = nullptr;
    std::uint8_t* d_comp_mask = nullptr;
    cudaMalloc(&d_formed, beam_samples * sizeof(float2));
    cudaMalloc(&d_power, max_beams * n_freq * sizeof(float));
    cudaMalloc(&d_comp_mask, total_cells * sizeof(std::uint8_t));

    cudaMemcpy(d_formed, h_formed_beams.data(), beam_samples * sizeof(float2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_comp_mask, h_comp_mask.data(), total_cells * sizeof(std::uint8_t), cudaMemcpyHostToDevice);

    launch_reduce_beam_power(d_formed, d_power, n_time, n_freq, 1, 1, nullptr, 1, d_comp_mask, chunk_size);
    cudaDeviceSynchronize();

    std::vector<float> h_reduced_power(n_freq);
    cudaMemcpy(h_reduced_power.data(), d_power, n_freq * sizeof(float), cudaMemcpyDeviceToHost);

    const float measured_pwr = h_reduced_power[f_test];
    std::printf("  2. Blanking Compensated Power on ch=%zu (%zu/60 chunks blanked): %.4f (Expected %.4f, Uncompensated would be %.4f)\n",
                f_test, blanked_chunks, measured_pwr, true_pwr, true_pwr * (50.0f / 60.0f));

    if (std::abs(measured_pwr - true_pwr) > 1e-3f) {
        std::printf("  Result: [FAILED] Beam power blanking compensation failed!\n");
        std::fflush(stdout);
        return 1;
    }
    std::printf("  Result: [PASSED] Exempt channels and beam power blanking compensation verified!\n\n");
    std::fflush(stdout);

    cudaFree(d_exempt);
    cudaFree(d_formed);
    cudaFree(d_power);
    cudaFree(d_comp_mask);

    // >>> TEST 5: Edge Channels (edge_channels_to_ignore=0) & Persistent Cross-Frame EMA <<<
    std::printf(">>> TEST 5: Edge-Channel Excision & Persistent Cross-Frame EMA (Multi-Second Step) <<<\n");

    const std::size_t f_edge = 5;    // Inside old edge_channels_to_ignore=16 blind spot
    const std::size_t f_steady = 160; // Analog of 414.0 MHz multi-second steady burst

    // Synthesize fresh clean noise frame
    std::vector<int4x2_t> h_clean_base(frame_elems);
    std::memset(h_clean_base.data(), 0, frame_elems * sizeof(int4x2_t));
    for (std::size_t t = 0; t < n_time; ++t) {
        for (std::size_t f = 0; f < n_freq; ++f) {
            const float pb_scale = passband_scale[f];
            const std::size_t base = (t * n_freq + f) * n_elem;
            for (int a = 0; a < num_active; ++a) {
                const int raw_e = active_raw[a];
                const float sig = ant_sigmas[a] * pb_scale;
                const int re = static_cast<int>(std::round(norm(rng) * sig));
                const int im = static_cast<int>(std::round(norm(rng) * sig));
                h_clean_base[base + raw_e] = pack_int4x2_signed(re, im);
            }
        }
    }

    // 5A: Verify edge channel burst (ch=5) is caught when edge_channels_to_ignore=0
    std::vector<int4x2_t> h_edge_burst = h_clean_base;
    for (std::size_t t = 0; t < n_time; ++t) {
        const std::size_t base = (t * n_freq + f_edge) * n_elem;
        for (int a = 0; a < num_active; ++a) {
            const int raw_e = active_raw[a];
            const float sig = ant_sigmas[a] * 2.5f;
            const int re = static_cast<int>(std::round(norm(rng) * sig));
            const int im = static_cast<int>(std::round(norm(rng) * sig));
            h_edge_burst[base + raw_e] = pack_int4x2_signed(re, im);
        }
    }

    RfiMitigationConfig cfg_edge = cfg;
    cfg_edge.edge_channels_to_ignore = 0;
    cfg_edge.enable_persistent_ema = false;

    cudaMemset(d_static_notch, 0, n_freq * sizeof(std::uint8_t));
    cudaMemcpy(d_voltages, h_edge_burst.data(), frame_elems * sizeof(int4x2_t),
               cudaMemcpyHostToDevice);
    launch_inspect_rfi_chunks(d_voltages, d_active_antennas, num_active, d_chunk_power, d_chunk_sk,
                              d_chunk_sat_frac, n_time, n_freq, n_elem, chunk_size, 1, nullptr,
                              false);
    launch_evaluate_rfi_flags(d_chunk_power, d_chunk_sk, d_chunk_sat_frac, d_static_notch,
                              d_channel_gains, d_chunk_rfi_mask, d_channel_stats, n_chunks, n_freq,
                              num_active, chunk_size, cfg_edge, nullptr);
    cudaDeviceSynchronize();
    cudaMemcpy(h_stats.data(), d_channel_stats, n_freq * sizeof(RfiChannelStats),
               cudaMemcpyDeviceToHost);

    const std::uint32_t edge_flags = h_stats[f_edge].flag_bits;
    std::printf("  1. Edge Channel (%zu) flags with edge_channels_to_ignore=0: 0x%02x (Expected FREQ_MAD 0x08)\n",
                f_edge, edge_flags);
    if ((edge_flags & RFI_FREQ_MAD) == 0) {
        std::printf("  Result: [FAILED] Edge channel burst was ignored!\n");
        std::fflush(stdout);
        return 1;
    }

    // 5B: Full-frame steady +step burst across 10 consecutive frames (414 MHz CW/step analog)
    std::vector<int4x2_t> h_step_burst = h_clean_base;
    for (std::size_t t = 0; t < n_time; ++t) {
        const std::size_t base = (t * n_freq + f_steady) * n_elem;
        for (int a = 0; a < num_active; ++a) {
            const int raw_e = active_raw[a];
            const float sig = ant_sigmas[a] * passband_scale[f_steady] * 2.0f;
            const int re = static_cast<int>(std::round(norm(rng) * sig));
            const int im = static_cast<int>(std::round(norm(rng) * sig));
            h_step_burst[base + raw_e] = pack_int4x2_signed(re, im);
        }
    }

    // Isolate temporal detection: without EMA, a steady full-frame burst hides inside single-frame median
    RfiMitigationConfig cfg_no_ema = cfg;
    cfg_no_ema.edge_channels_to_ignore = 0;
    cfg_no_ema.enable_freq_mad_check = false;
    cfg_no_ema.enable_persistent_ema = false;

    cudaMemcpy(d_voltages, h_step_burst.data(), frame_elems * sizeof(int4x2_t),
               cudaMemcpyHostToDevice);
    launch_inspect_rfi_chunks(d_voltages, d_active_antennas, num_active, d_chunk_power, d_chunk_sk,
                              d_chunk_sat_frac, n_time, n_freq, n_elem, chunk_size, 1, nullptr,
                              false);
    launch_evaluate_rfi_flags(d_chunk_power, d_chunk_sk, d_chunk_sat_frac, d_static_notch,
                              d_channel_gains, d_chunk_rfi_mask, d_channel_stats, n_chunks, n_freq,
                              num_active, chunk_size, cfg_no_ema, nullptr);
    cudaDeviceSynchronize();
    cudaMemcpy(h_stats.data(), d_channel_stats, n_freq * sizeof(RfiChannelStats),
               cudaMemcpyDeviceToHost);
    const std::uint32_t no_ema_flags = h_stats[f_steady].flag_bits;

    // Now allocate persistent EMA buffers, warm up on 5 clean frames, then feed 10 consecutive step frames
    float* d_ema_power = nullptr;
    float* d_ema_mad = nullptr;
    cudaMalloc(&d_ema_power, n_freq * sizeof(float));
    cudaMalloc(&d_ema_mad, n_freq * sizeof(float));
    cudaMemset(d_ema_power, 0, n_freq * sizeof(float));
    cudaMemset(d_ema_mad, 0, n_freq * sizeof(float));

    RfiMitigationConfig cfg_ema = cfg_no_ema;
    cfg_ema.enable_persistent_ema = true;
    cfg_ema.ema_alpha = 0.02f;

    cudaMemcpy(d_voltages, h_clean_base.data(), frame_elems * sizeof(int4x2_t),
               cudaMemcpyHostToDevice);
    for (int warmup = 0; warmup < 5; ++warmup) {
        launch_inspect_rfi_chunks(d_voltages, d_active_antennas, num_active, d_chunk_power,
                                  d_chunk_sk, d_chunk_sat_frac, n_time, n_freq, n_elem, chunk_size,
                                  1, nullptr, false);
        launch_evaluate_rfi_flags(d_chunk_power, d_chunk_sk, d_chunk_sat_frac, d_static_notch,
                                  d_channel_gains, d_chunk_rfi_mask, d_channel_stats, n_chunks,
                                  n_freq, num_active, chunk_size, cfg_ema, nullptr, nullptr,
                                  d_ema_power, d_ema_mad);
    }
    cudaDeviceSynchronize();
    float ema_before = 0.0f;
    cudaMemcpy(&ema_before, d_ema_power + f_steady, sizeof(float), cudaMemcpyDeviceToHost);

    cudaMemcpy(d_voltages, h_step_burst.data(), frame_elems * sizeof(int4x2_t),
               cudaMemcpyHostToDevice);
    float min_flagged_frac = 1.0f;
    std::uint32_t ema_flags = 0;
    for (int burst_f = 0; burst_f < 10; ++burst_f) {
        launch_inspect_rfi_chunks(d_voltages, d_active_antennas, num_active, d_chunk_power,
                                  d_chunk_sk, d_chunk_sat_frac, n_time, n_freq, n_elem, chunk_size,
                                  1, nullptr, false);
        launch_evaluate_rfi_flags(d_chunk_power, d_chunk_sk, d_chunk_sat_frac, d_static_notch,
                                  d_channel_gains, d_chunk_rfi_mask, d_channel_stats, n_chunks,
                                  n_freq, num_active, chunk_size, cfg_ema, nullptr, nullptr,
                                  d_ema_power, d_ema_mad);
        cudaDeviceSynchronize();
        cudaMemcpy(h_stats.data(), d_channel_stats, n_freq * sizeof(RfiChannelStats),
                   cudaMemcpyDeviceToHost);
        ema_flags = h_stats[f_steady].flag_bits;
        min_flagged_frac = std::min(min_flagged_frac, h_stats[f_steady].flagged_chunk_fraction);
    }
    float ema_after = 0.0f;
    cudaMemcpy(&ema_after, d_ema_power + f_steady, sizeof(float), cudaMemcpyDeviceToHost);

    // Measure full 3-pass latency with persistent EMA enabled
    RfiMitigationConfig cfg_bench = cfg;
    cfg_bench.edge_channels_to_ignore = 0;
    cfg_bench.enable_persistent_ema = true;
    cudaEventRecord(ev_start);
    launch_inspect_rfi_chunks(d_voltages, d_active_antennas, num_active, d_chunk_power, d_chunk_sk,
                              d_chunk_sat_frac, n_time, n_freq, n_elem, chunk_size, 1, nullptr,
                              false);
    launch_evaluate_rfi_flags(d_chunk_power, d_chunk_sk, d_chunk_sat_frac, d_static_notch,
                              d_channel_gains, d_chunk_rfi_mask, d_channel_stats, n_chunks, n_freq,
                              num_active, chunk_size, cfg_bench, nullptr, nullptr, d_ema_power,
                              d_ema_mad);
    launch_blank_rfi_chunks(d_voltages, d_chunk_rfi_mask, n_time, n_freq, n_elem, chunk_size,
                            nullptr, false);
    cudaEventRecord(ev_stop);
    cudaEventSynchronize(ev_stop);
    float ema_elapsed_ms = 0.0f;
    cudaEventElapsedTime(&ema_elapsed_ms, ev_start, ev_stop);

    std::printf("  2. Full-frame step burst ch=%zu WITHOUT EMA: flags=0x%02x (Expected TIME_BURST=0, escaped)\n",
                f_steady, no_ema_flags);
    std::printf("  3. Full-frame step burst ch=%zu WITH Persistent EMA (10 frames): flags=0x%02x, min_flagged=%.2f, EMA=%.4f -> %.4f\n",
                f_steady, ema_flags, min_flagged_frac, ema_before, ema_after);
    std::printf("  4. 3-Pass Latency with EMA enabled: %.3f ms (%.2f%% of 51.20 ms frame budget)\n",
                ema_elapsed_ms, 100.0 * ema_elapsed_ms / 51.20);

    if ((no_ema_flags & RFI_TIME_BURST) != 0 || (ema_flags & RFI_TIME_BURST) == 0
        || min_flagged_frac < 0.90f || std::abs(ema_after - ema_before) > 1.0e-5f) {
        std::printf("  Result: [FAILED] Persistent cross-frame EMA failed!\n");
        std::fflush(stdout);
        return 1;
    }
    std::printf("  Result: [PASSED] Edge-channel excision and persistent EMA step-burst detection verified!\n\n");
    std::fflush(stdout);

    cudaFree(d_ema_power);
    cudaFree(d_ema_mad);

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
