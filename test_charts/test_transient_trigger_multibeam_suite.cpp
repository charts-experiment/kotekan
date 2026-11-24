#include "cudaTransientTrigger.hpp"
#include "cudaUtils.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>

namespace {

bool test_single_beam_localized_burst() {
    std::cout << "\n[Test 1/3] Single-Beam Localized Burst in B=8 Beams\n";

    constexpr std::size_t n_time = 2048;
    constexpr std::size_t n_freq = 16;
    constexpr std::size_t num_beams = 8;
    constexpr std::size_t max_beams_stride = 8;
    constexpr std::size_t target_beam = 5;
    constexpr std::size_t burst_t0 = 1024;
    constexpr std::size_t burst_t1 = 1280;

    std::mt19937 rng(2026);
    std::normal_distribution<float> noise(0.0f, 1.0f);

    std::vector<float2> h_v(n_time * n_freq * max_beams_stride);
    for (std::size_t t = 0; t < n_time; ++t) {
        const float amp = (t >= burst_t0 && t < burst_t1) ? 4.5f : 1.0f;
        for (std::size_t f = 0; f < n_freq; ++f) {
            for (std::size_t b = 0; b < num_beams; ++b) {
                const float s = (b == target_beam) ? amp : 1.0f;
                h_v[(t * n_freq + f) * max_beams_stride + b] =
                    make_float2(s * noise(rng), s * noise(rng));
            }
        }
    }

    float2* d_v = nullptr;
    kotekan::TransientSubframeMetrics* d_sub = nullptr;
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&d_v, h_v.size() * sizeof(float2)));
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&d_sub, sizeof(kotekan::TransientSubframeMetrics)));
    CHECK_CUDA_ERROR_NON_OO(
        cudaMemcpy(d_v, h_v.data(), h_v.size() * sizeof(float2), cudaMemcpyHostToDevice));

    cudaStream_t stream = nullptr;
    CHECK_CUDA_ERROR_NON_OO(cudaStreamCreate(&stream));
    kotekan::launch_transient_subframe_coincidence(d_v, d_sub, n_time, n_freq, num_beams,
                                                  max_beams_stride, 256, stream);
    CHECK_CUDA_ERROR_NON_OO(cudaStreamSynchronize(stream));

    kotekan::TransientSubframeMetrics h_sub;
    CHECK_CUDA_ERROR_NON_OO(cudaMemcpy(&h_sub, d_sub, sizeof(kotekan::TransientSubframeMetrics),
                                       cudaMemcpyDeviceToHost));

    // Also validate full evaluate_transient_decision path with synthetic flagged channels & low R01
    kotekan::TransientTriggerConfig cfg;
    cfg.enable_multibeam_veto = true;
    cfg.min_sidelobe_confidence = 0.5f;
    cfg.max_sidelobe_ratio = 0.70f;
    cfg.min_flagged_channels = 8;
    std::vector<float2> h_metrics(n_freq, make_float2(1.4f, 0.05f));
    const auto decision =
        kotekan::evaluate_transient_decision(h_metrics.data(), n_freq, cfg, 1, 0, &h_sub);

    std::cout << "  - peak_beam_id=" << h_sub.peak_beam_id
              << ", min_sidelobe_ratio=" << h_sub.min_sidelobe_ratio
              << ", max_sidelobe_confidence=" << h_sub.max_sidelobe_confidence
              << ", trigger_fired=" << decision.trigger_fired << "\n";

    cudaFree(d_v);
    cudaFree(d_sub);
    cudaStreamDestroy(stream);

    return (h_sub.peak_beam_id == target_beam) && (h_sub.min_sidelobe_ratio < 0.2f)
           && (h_sub.max_sidelobe_confidence > 5.0f) && decision.trigger_fired;
}

bool test_common_mode_rfi_veto() {
    std::cout << "\n[Test 2/3] Common-Mode RFI Across B=8 Beams Must Be Vetoed\n";

    constexpr std::size_t n_time = 2048;
    constexpr std::size_t n_freq = 16;
    constexpr std::size_t num_beams = 8;
    constexpr std::size_t max_beams_stride = 8;

    std::mt19937 rng(3030);
    std::normal_distribution<float> noise(0.0f, 1.0f);

    std::vector<float2> h_v(n_time * n_freq * max_beams_stride);
    for (std::size_t t = 0; t < n_time; ++t) {
        const bool in_rfi = (t >= 768 && t < 1024);
        for (std::size_t f = 0; f < n_freq; ++f) {
            const float2 common =
                in_rfi ? make_float2(5.0f, -5.0f) : make_float2(0.0f, 0.0f);
            for (std::size_t b = 0; b < num_beams; ++b) {
                h_v[(t * n_freq + f) * max_beams_stride + b] =
                    make_float2(common.x + 0.25f * noise(rng), common.y + 0.25f * noise(rng));
            }
        }
    }

    float2* d_v = nullptr;
    kotekan::TransientSubframeMetrics* d_sub = nullptr;
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&d_v, h_v.size() * sizeof(float2)));
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&d_sub, sizeof(kotekan::TransientSubframeMetrics)));
    CHECK_CUDA_ERROR_NON_OO(
        cudaMemcpy(d_v, h_v.data(), h_v.size() * sizeof(float2), cudaMemcpyHostToDevice));

    cudaStream_t stream = nullptr;
    CHECK_CUDA_ERROR_NON_OO(cudaStreamCreate(&stream));
    kotekan::launch_transient_subframe_coincidence(d_v, d_sub, n_time, n_freq, num_beams,
                                                  max_beams_stride, 256, stream);
    CHECK_CUDA_ERROR_NON_OO(cudaStreamSynchronize(stream));

    kotekan::TransientSubframeMetrics h_sub;
    CHECK_CUDA_ERROR_NON_OO(cudaMemcpy(&h_sub, d_sub, sizeof(kotekan::TransientSubframeMetrics),
                                       cudaMemcpyDeviceToHost));

    kotekan::TransientTriggerConfig cfg;
    cfg.enable_multibeam_veto = true;
    cfg.min_sidelobe_confidence = 0.5f;
    cfg.max_sidelobe_ratio = 0.70f;
    cfg.min_flagged_channels = 8;
    // Even if channel SK triggers and R01 is somehow bypassed, multi-beam veto must block it
    std::vector<float2> h_metrics(n_freq, make_float2(1.8f, 0.05f));
    const auto decision =
        kotekan::evaluate_transient_decision(h_metrics.data(), n_freq, cfg, 2, 0, &h_sub);

    std::cout << "  - min_sidelobe_ratio=" << h_sub.min_sidelobe_ratio
              << ", max_sidelobe_confidence=" << h_sub.max_sidelobe_confidence
              << ", trigger_fired=" << decision.trigger_fired << "\n";

    cudaFree(d_v);
    cudaFree(d_sub);
    cudaStreamDestroy(stream);

    return (h_sub.min_sidelobe_ratio > 0.85f) && !decision.trigger_fired;
}

bool test_legacy_sk_r01_and_rfi_mask_exclusion() {
    std::cout << "\n[Test 3/3] Legacy 2-Beam SK/R01 & RFI-Mask Chunk Exclusion\n";

    constexpr std::size_t n_time = 1024;
    constexpr std::size_t n_freq = 16;
    constexpr std::size_t num_beams = 2;
    constexpr std::size_t chunk_size = 256;

    std::mt19937 rng(4040);
    std::normal_distribution<float> noise(0.0f, 1.0f);

    // Inject huge impulse in chunk 2 [512..768) that is RFI-masked -> SK should remain ~1.0
    std::vector<float2> h_v(n_time * n_freq * num_beams);
    for (std::size_t t = 0; t < n_time; ++t) {
        const float amp = (t >= 512 && t < 768) ? 20.0f : 1.0f;
        for (std::size_t f = 0; f < n_freq; ++f) {
            for (std::size_t b = 0; b < num_beams; ++b) {
                h_v[(t * n_freq + f) * num_beams + b] =
                    make_float2(amp * noise(rng), amp * noise(rng));
            }
        }
    }
    const std::size_t n_chunks = n_time / chunk_size;
    std::vector<std::uint8_t> h_mask(n_chunks * n_freq, 0);
    for (std::size_t f = 0; f < n_freq; ++f) {
        h_mask[2 * n_freq + f] = 1;
    }

    float2* d_v = nullptr;
    float2* d_m = nullptr;
    std::uint8_t* d_mask = nullptr;
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&d_v, h_v.size() * sizeof(float2)));
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&d_m, n_freq * sizeof(float2)));
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&d_mask, h_mask.size() * sizeof(std::uint8_t)));
    CHECK_CUDA_ERROR_NON_OO(
        cudaMemcpy(d_v, h_v.data(), h_v.size() * sizeof(float2), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR_NON_OO(
        cudaMemcpy(d_mask, h_mask.data(), h_mask.size() * sizeof(std::uint8_t),
                   cudaMemcpyHostToDevice));

    cudaStream_t stream = nullptr;
    CHECK_CUDA_ERROR_NON_OO(cudaStreamCreate(&stream));
    kotekan::launch_transient_detection_metrics(d_v, d_m, n_time, n_freq, num_beams, num_beams,
                                                stream, d_mask, chunk_size);
    CHECK_CUDA_ERROR_NON_OO(cudaStreamSynchronize(stream));

    std::vector<float2> h_m(n_freq);
    CHECK_CUDA_ERROR_NON_OO(
        cudaMemcpy(h_m.data(), d_m, n_freq * sizeof(float2), cudaMemcpyDeviceToHost));

    double mean_sk = 0.0;
    for (std::size_t f = 0; f < n_freq; ++f) {
        mean_sk += h_m[f].x;
    }
    mean_sk /= static_cast<double>(n_freq);
    std::cout << "  - Masked-chunk mean_sk=" << mean_sk << " (expected ~1.0)\n";

    cudaFree(d_v);
    cudaFree(d_m);
    cudaFree(d_mask);
    cudaStreamDestroy(stream);

    return std::abs(mean_sk - 1.0) < 0.25;
}

} // namespace

int main() {
    const bool t1 = test_single_beam_localized_burst();
    const bool t2 = test_common_mode_rfi_veto();
    const bool t3 = test_legacy_sk_r01_and_rfi_mask_exclusion();
    std::cout << "\nSummary: SingleBeam=" << (t1 ? "PASS" : "FAIL")
              << ", CommonModeVeto=" << (t2 ? "PASS" : "FAIL")
              << ", LegacySKMask=" << (t3 ? "PASS" : "FAIL") << "\n";
    return (t1 && t2 && t3) ? 0 : 1;
}
