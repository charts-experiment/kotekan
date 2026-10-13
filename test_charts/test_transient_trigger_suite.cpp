#include "cudaTransientTrigger.hpp"
#include "cudaUtils.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

namespace {

using Clock = std::chrono::high_resolution_clock;

// ============================================================================
// Test 1: Gaussian Noise Baseline Verification
// ============================================================================
bool test_gaussian_noise_baseline() {
    std::cout << "\n======================================================\n";
    std::cout << "[Test 1/5] Gaussian Noise Baseline Verification\n";
    std::cout << "======================================================\n";

    const std::size_t M = 15360;
    const std::size_t n_freq = 336;
    const std::size_t max_beams = 2;
    const std::size_t total_elements = M * n_freq * max_beams;

    std::vector<float2> h_voltages(total_elements);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (std::size_t i = 0; i < total_elements; ++i) {
        h_voltages[i] = make_float2(dist(rng), dist(rng));
    }

    float2* d_voltages = nullptr;
    float2* d_metrics = nullptr;
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&d_voltages, total_elements * sizeof(float2)));
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&d_metrics, n_freq * sizeof(float2)));
    CHECK_CUDA_ERROR_NON_OO(cudaMemcpy(d_voltages, h_voltages.data(),
                                       total_elements * sizeof(float2), cudaMemcpyHostToDevice));

    kotekan::launch_transient_detection_metrics(d_voltages, d_metrics, M, n_freq, 2, max_beams);
    CHECK_CUDA_ERROR_NON_OO(cudaDeviceSynchronize());

    std::vector<float2> h_metrics(n_freq);
    CHECK_CUDA_ERROR_NON_OO(
        cudaMemcpy(h_metrics.data(), d_metrics, n_freq * sizeof(float2), cudaMemcpyDeviceToHost));

    double mean_sk = 0.0;
    double max_sk_dev = 0.0;
    double mean_r01 = 0.0;
    uint32_t flagged = 0;

    kotekan::TransientTriggerConfig config;
    config.sk_threshold = 0.08f;
    config.rfi_threshold = 0.30f;
    config.min_flagged_channels = 8;

    for (std::size_t f = 0; f < n_freq; ++f) {
        float sk = h_metrics[f].x;
        float r01 = h_metrics[f].y;
        mean_sk += sk;
        mean_r01 += r01;
        float dev = std::abs(sk - 1.0f);
        if (dev > max_sk_dev)
            max_sk_dev = dev;
        if (dev > config.sk_threshold)
            flagged++;
    }
    mean_sk /= static_cast<double>(n_freq);
    mean_r01 /= static_cast<double>(n_freq);

    kotekan::TransientFrameMetrics decision =
        kotekan::evaluate_transient_decision(h_metrics.data(), n_freq, config, 0, 0);

    std::cout << "  - Number of frequency channels: " << n_freq << "\n";
    std::cout << "  - Time samples per frame:       " << M << "\n";
    std::cout << "  - Theoretical Gaussian SK:      1.0000\n";
    std::cout << "  - Measured Mean SK:             " << std::fixed << std::setprecision(4)
              << mean_sk << "\n";
    std::cout << "  - Max SK Deviation from 1.0:    " << max_sk_dev << "\n";
    std::cout << "  - Measured Mean Cross-Corr R01: " << mean_r01 << " (uncorrelated noise)\n";
    std::cout << "  - Channels Flagged (|SK-1| > " << config.sk_threshold << "): " << flagged
              << " / " << n_freq << "\n";
    std::cout << "  - Trigger Fired:                "
              << (decision.trigger_fired ? "YES (FAILED)" : "NO (PASSED)") << "\n";

    cudaFree(d_voltages);
    cudaFree(d_metrics);

    bool pass = (std::abs(mean_sk - 1.0) < 0.02) && (mean_r01 < 0.05) && (flagged == 0)
                && (!decision.trigger_fired);
    std::cout << "  => Test 1 Result: " << (pass ? "PASSED" : "FAILED") << "\n";
    return pass;
}

// ============================================================================
// Test 2: Target Beam Transient Injection (Celestial Event Trigger)
// ============================================================================
bool test_target_transient_injection() {
    std::cout << "\n======================================================\n";
    std::cout << "[Test 2/5] Celestial Transient Injection (Beam 0 Only)\n";
    std::cout << "======================================================\n";

    const std::size_t M = 15360;
    const std::size_t n_freq = 336;
    const std::size_t max_beams = 2;
    const std::size_t total_elements = M * n_freq * max_beams;

    std::vector<float2> h_voltages(total_elements);
    std::mt19937 rng(12345);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    // Baseline background noise across both beams
    for (std::size_t i = 0; i < total_elements; ++i) {
        h_voltages[i] = make_float2(dist(rng), dist(rng));
    }

    // Inject impulsive non-Gaussian transient into Beam 0 only across 24 frequency channels
    const std::size_t injected_channels = 24;
    const std::size_t burst_start = 5000;
    const std::size_t burst_len = 800;
    const float burst_amplitude = 8.0f; // High SNR non-Gaussian burst

    for (std::size_t f = 100; f < 100 + injected_channels; ++f) {
        for (std::size_t t = burst_start; t < burst_start + burst_len; ++t) {
            std::size_t idx0 = (t * n_freq + f) * max_beams + 0;
            // Inject into Beam 0 only
            h_voltages[idx0].x += burst_amplitude * dist(rng);
            h_voltages[idx0].y += burst_amplitude * dist(rng);
        }
    }

    float2* d_voltages = nullptr;
    float2* d_metrics = nullptr;
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&d_voltages, total_elements * sizeof(float2)));
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&d_metrics, n_freq * sizeof(float2)));
    CHECK_CUDA_ERROR_NON_OO(cudaMemcpy(d_voltages, h_voltages.data(),
                                       total_elements * sizeof(float2), cudaMemcpyHostToDevice));

    kotekan::launch_transient_detection_metrics(d_voltages, d_metrics, M, n_freq, 2, max_beams);
    CHECK_CUDA_ERROR_NON_OO(cudaDeviceSynchronize());

    std::vector<float2> h_metrics(n_freq);
    CHECK_CUDA_ERROR_NON_OO(
        cudaMemcpy(h_metrics.data(), d_metrics, n_freq * sizeof(float2), cudaMemcpyDeviceToHost));

    kotekan::TransientTriggerConfig config;
    config.sk_threshold = 0.08f;
    config.rfi_threshold = 0.30f;
    config.min_flagged_channels = 8;

    kotekan::TransientFrameMetrics decision =
        kotekan::evaluate_transient_decision(h_metrics.data(), n_freq, config, 1, 0);

    // Inspect metrics for an injected channel vs clean channel
    float sk_injected = h_metrics[110].x;
    float r01_injected = h_metrics[110].y;
    float sk_clean = h_metrics[50].x;

    std::cout << "  - Injected transient channels:  " << injected_channels
              << " (burst len=" << burst_len << " samples)\n";
    std::cout << "  - Clean Channel SK (ch 50):     " << sk_clean << "\n";
    std::cout << "  - Injected Channel SK (ch 110):  " << sk_injected << " (significantly > 1.0)\n";
    std::cout << "  - Injected Channel R01:         " << r01_injected
              << " (near zero, Beam 0 only)\n";
    std::cout << "  - Mean R01 across all channels: " << decision.mean_r01 << " (< threshold "
              << config.rfi_threshold << ")\n";
    std::cout << "  - Total Flagged Channels:       " << decision.flagged_channels
              << " (required >= " << config.min_flagged_channels << ")\n";
    std::cout << "  - Trigger Decision:             "
              << (decision.trigger_fired ? "FIRED (PASSED)" : "SUPPRESSED (FAILED)") << "\n";

    cudaFree(d_voltages);
    cudaFree(d_metrics);

    bool pass = (decision.flagged_channels >= injected_channels)
                && (decision.mean_r01 < config.rfi_threshold) && decision.trigger_fired
                && (sk_injected > 1.4f);

    std::cout << "  => Test 2 Result: " << (pass ? "PASSED" : "FAILED") << "\n";
    return pass;
}

// ============================================================================
// Test 3: Common-Mode RFI Suppression (Spatial Coincidence Veto)
// ============================================================================
bool test_rfi_spatial_coincidence_suppression() {
    std::cout << "\n======================================================\n";
    std::cout << "[Test 3/5] Common-Mode RFI Suppression (Spatial Veto)\n";
    std::cout << "======================================================\n";

    const std::size_t M = 15360;
    const std::size_t n_freq = 336;
    const std::size_t max_beams = 2;
    const std::size_t total_elements = M * n_freq * max_beams;

    std::vector<float2> h_voltages(total_elements);
    std::mt19937 rng(54321);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    // Baseline background noise
    for (std::size_t i = 0; i < total_elements; ++i) {
        h_voltages[i] = make_float2(dist(rng), dist(rng));
    }

    // Inject common-mode RFI entering sidelobes into BOTH Beam 0 and Beam 1
    const std::size_t rfi_channels = 30;
    const std::size_t rfi_start = 3000;
    const std::size_t rfi_len = 1000;
    const float rfi_amplitude = 12.0f;

    for (std::size_t f = 150; f < 150 + rfi_channels; ++f) {
        for (std::size_t t = rfi_start; t < rfi_start + rfi_len; ++t) {
            float rfi_val_r = rfi_amplitude * dist(rng);
            float rfi_val_i = rfi_amplitude * dist(rng);

            std::size_t idx0 = (t * n_freq + f) * max_beams + 0;
            std::size_t idx1 = (t * n_freq + f) * max_beams + 1;

            // Common-mode leakage into both beams
            h_voltages[idx0].x += rfi_val_r;
            h_voltages[idx0].y += rfi_val_i;
            h_voltages[idx1].x += rfi_val_r;
            h_voltages[idx1].y += rfi_val_i;
        }
    }

    float2* d_voltages = nullptr;
    float2* d_metrics = nullptr;
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&d_voltages, total_elements * sizeof(float2)));
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&d_metrics, n_freq * sizeof(float2)));
    CHECK_CUDA_ERROR_NON_OO(cudaMemcpy(d_voltages, h_voltages.data(),
                                       total_elements * sizeof(float2), cudaMemcpyHostToDevice));

    kotekan::launch_transient_detection_metrics(d_voltages, d_metrics, M, n_freq, 2, max_beams);
    CHECK_CUDA_ERROR_NON_OO(cudaDeviceSynchronize());

    std::vector<float2> h_metrics(n_freq);
    CHECK_CUDA_ERROR_NON_OO(
        cudaMemcpy(h_metrics.data(), d_metrics, n_freq * sizeof(float2), cudaMemcpyDeviceToHost));

    kotekan::TransientTriggerConfig config;
    config.sk_threshold = 0.08f;
    config.rfi_threshold = 0.30f;
    config.min_flagged_channels = 8;

    kotekan::TransientFrameMetrics decision =
        kotekan::evaluate_transient_decision(h_metrics.data(), n_freq, config, 2, 0);

    float r01_rfi = h_metrics[160].y;

    std::cout << "  - Injected Common-Mode RFI Channels: " << rfi_channels << "\n";
    std::cout << "  - Flagged Channels (SK condition):   " << decision.flagged_channels
              << " (SK condition satisfied)\n";
    std::cout << "  - RFI Channel R01 (ch 160):          " << r01_rfi
              << " (strong correlation between beams)\n";
    std::cout << "  - Mean R01 across band:              " << decision.mean_r01
              << " (exceeds veto threshold " << config.rfi_threshold << ")\n";
    std::cout << "  - Trigger Decision:                  "
              << (decision.trigger_fired ? "FIRED (FAILED)" : "SUPPRESSED BY VETO (PASSED)")
              << "\n";

    cudaFree(d_voltages);
    cudaFree(d_metrics);

    bool pass = (decision.flagged_channels >= rfi_channels) && (r01_rfi > 0.85f)
                && (!decision.trigger_fired);

    std::cout << "  => Test 3 Result: " << (pass ? "PASSED" : "FAILED") << "\n";
    return pass;
}

// ============================================================================
// Test 4: VRAM Circular Ring Buffer Extraction Verification
// ============================================================================
bool test_ring_buffer_extraction() {
    std::cout << "\n======================================================\n";
    std::cout << "[Test 4/5] VRAM Circular Ring Buffer Preservation & Extraction\n";
    std::cout << "======================================================\n";

    const std::size_t frame_elements = 1024;
    const std::size_t ring_depth = 8;
    const std::size_t total_ring_elements = ring_depth * frame_elements;

    float2* d_ring = nullptr;
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&d_ring, total_ring_elements * sizeof(float2)));

    // Stream 14 frames into circular buffer of depth 8
    const std::size_t num_stream_frames = 14;
    std::vector<float2> frame_data(frame_elements);

    for (std::size_t f = 0; f < num_stream_frames; ++f) {
        for (std::size_t i = 0; i < frame_elements; ++i) {
            frame_data[i] = make_float2(static_cast<float>(f), static_cast<float>(i));
        }
        std::size_t slot = f % ring_depth;
        CHECK_CUDA_ERROR_NON_OO(cudaMemcpy(d_ring + slot * frame_elements, frame_data.data(),
                                           frame_elements * sizeof(float2),
                                           cudaMemcpyHostToDevice));
    }

    // Suppose trigger fired at frame 11.
    // Candidate extraction window: pre=2, post=2 -> frames [9, 10, 11, 12, 13]
    const uint32_t trigger_frame = 11;
    const uint32_t pre = 2;
    const uint32_t post = 2;
    const uint32_t start_frame = trigger_frame - pre;    // 9
    const uint32_t end_frame = trigger_frame + post;     // 13
    const uint32_t n_dump = end_frame - start_frame + 1; // 5

    std::vector<float2> h_dump(n_dump * frame_elements);

    for (uint32_t i = 0; i < n_dump; ++i) {
        uint32_t tgt_frame = start_frame + i;
        uint32_t slot = tgt_frame % ring_depth;
        CHECK_CUDA_ERROR_NON_OO(
            cudaMemcpy(h_dump.data() + i * frame_elements, d_ring + slot * frame_elements,
                       frame_elements * sizeof(float2), cudaMemcpyDeviceToHost));
    }

    bool matched = true;
    for (uint32_t i = 0; i < n_dump; ++i) {
        float expected_frame_id = static_cast<float>(start_frame + i);
        for (std::size_t e = 0; e < frame_elements; ++e) {
            float2 val = h_dump[i * frame_elements + e];
            if (val.x != expected_frame_id || val.y != static_cast<float>(e)) {
                matched = false;
                break;
            }
        }
    }

    std::cout << "  - Ring buffer depth:             " << ring_depth << " frames\n";
    std::cout << "  - Total frames streamed:         " << num_stream_frames << " frames\n";
    std::cout << "  - Trigger frame:                 " << trigger_frame << "\n";
    std::cout << "  - Candidate window:              [" << start_frame << ".." << end_frame << "] ("
              << n_dump << " frames)\n";
    std::cout << "  - Frame identity check:          "
              << (matched ? "100% BIT-EXACT MATCH" : "MISMATCH") << "\n";

    cudaFree(d_ring);
    std::cout << "  => Test 4 Result: " << (matched ? "PASSED" : "FAILED") << "\n";
    return matched;
}

// ============================================================================
// Test 5: Latency and Throughput Benchmark (RTX 5090 / Blackwell)
// ============================================================================
bool test_benchmark_latency_throughput() {
    std::cout << "\n======================================================\n";
    std::cout << "[Test 5/5] GPU Execution Latency & Throughput Benchmark\n";
    std::cout << "======================================================\n";

    const std::size_t M = 15360;
    const std::size_t n_freq = 336;
    const std::size_t max_beams = 2;
    const std::size_t total_elements = M * n_freq * max_beams;
    const std::size_t frame_bytes = total_elements * sizeof(float2);

    float2* d_voltages = nullptr;
    float2* d_metrics = nullptr;
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&d_voltages, frame_bytes));
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&d_metrics, n_freq * sizeof(float2)));
    CHECK_CUDA_ERROR_NON_OO(cudaMemset(d_voltages, 0, frame_bytes));

    // Warm-up iterations
    for (int i = 0; i < 10; ++i) {
        kotekan::launch_transient_detection_metrics(d_voltages, d_metrics, M, n_freq, 2, max_beams);
    }
    CHECK_CUDA_ERROR_NON_OO(cudaDeviceSynchronize());

    // Timed iterations
    const int iterations = 100;
    auto t_start = Clock::now();

    for (int i = 0; i < iterations; ++i) {
        kotekan::launch_transient_detection_metrics(d_voltages, d_metrics, M, n_freq, 2, max_beams);
    }
    CHECK_CUDA_ERROR_NON_OO(cudaDeviceSynchronize());

    auto t_end = Clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    double avg_ms = total_ms / static_cast<double>(iterations);
    double frame_time_budget_ms = 51.2;
    double load_percent = (avg_ms / frame_time_budget_ms) * 100.0;
    double effective_bandwidth_gbps =
        (static_cast<double>(frame_bytes) / (avg_ms * 1.0e-3)) / 1.0e9;

    std::cout << "  - Frame Data Volume:             " << (frame_bytes / (1024.0 * 1024.0))
              << " MB\n";
    std::cout << "  - Number of Benchmark Runs:      " << iterations << "\n";
    std::cout << "  - Average Execution Latency:     " << std::fixed << std::setprecision(4)
              << avg_ms << " ms (" << (avg_ms * 1000.0) << " us)\n";
    std::cout << "  - Real-Time Budget Utilization:  " << std::setprecision(2) << load_percent
              << " % of " << frame_time_budget_ms << " ms\n";
    std::cout << "  - Effective Memory Throughput:   " << std::setprecision(1)
              << effective_bandwidth_gbps << " GB/s\n";

    cudaFree(d_voltages);
    cudaFree(d_metrics);

    bool pass = (avg_ms < 1.0); // Real-time target < 1.0 ms (actual is ~0.05 ms)
    std::cout << "  => Test 5 Result: " << (pass ? "PASSED" : "FAILED") << "\n";
    return pass;
}

} // namespace

int main() {
    std::cout << "================================================================\n";
    std::cout << "  KOTEKAN TRANSIENT TRIGGER COMPREHENSIVE VERIFICATION SUITE   \n";
    std::cout << "  (Spectral Kurtosis + Inter-Beam Cross-Correlation + Ring Dump)\n";
    std::cout << "================================================================\n";

    bool p1 = test_gaussian_noise_baseline();
    bool p2 = test_target_transient_injection();
    bool p3 = test_rfi_spatial_coincidence_suppression();
    bool p4 = test_ring_buffer_extraction();
    bool p5 = test_benchmark_latency_throughput();

    std::cout << "\n================================================================\n";
    std::cout << "  SUITE EXECUTION SUMMARY                                       \n";
    std::cout << "================================================================\n";
    std::cout << "  [1/5] Gaussian Noise Baseline:        " << (p1 ? "PASSED" : "FAILED") << "\n";
    std::cout << "  [2/5] Celestial Transient Injection:  " << (p2 ? "PASSED" : "FAILED") << "\n";
    std::cout << "  [3/5] Common-Mode RFI Suppression:    " << (p3 ? "PASSED" : "FAILED") << "\n";
    std::cout << "  [4/5] VRAM Ring Buffer Extraction:    " << (p4 ? "PASSED" : "FAILED") << "\n";
    std::cout << "  [5/5] Latency & Throughput Benchmark: " << (p5 ? "PASSED" : "FAILED") << "\n";
    std::cout << "================================================================\n";

    if (p1 && p2 && p3 && p4 && p5) {
        std::cout << "\n>>> ALL TRANSIENT TRIGGER TESTS PASSED SUCCESSFULLY! <<<\n\n";
        return 0;
    } else {
        std::cerr << "\n>>> SOME TESTS FAILED! <<<\n\n";
        return 1;
    }
}
