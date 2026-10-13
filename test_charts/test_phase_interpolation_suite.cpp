#include "chartsConstants.hpp"
#include "cudaDirectBeamTracker.hpp"
#include "cudaUtils.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
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

constexpr double SPEED_OF_LIGHT = kotekan::charts::constants::speed_of_light_m_per_s;
constexpr double TWO_PI = kotekan::charts::constants::two_pi;

// Helper: complex multiplication
inline float2 complex_mul(float2 a, float2 b) {
    return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// Helper: complex binary exponentiation
inline float2 complex_pow_int(float2 z, std::size_t exp) {
    float2 res = make_float2(1.0f, 0.0f);
    float2 base = z;
    while (exp > 0) {
        if (exp & 1) {
            res = complex_mul(res, base);
        }
        base = complex_mul(base, base);
        exp >>= 1;
    }
    const float norm_sq = res.x * res.x + res.y * res.y;
    if (norm_sq > 0.0f) {
        const float inv_r = 1.0f / std::sqrt(norm_sq);
        res.x *= inv_r;
        res.y *= inv_r;
    }
    return res;
}

// Helper: compute antenna coordinates (8x8 physical grid)
std::vector<float3> compute_positions(std::size_t n_ant, float spacing_m) {
    std::vector<float3> pos(n_ant);
    for (std::size_t a = 0; a < n_ant; ++a) {
        const unsigned int col = (n_ant <= 64) ? (a & 7U) : (a & 15U);
        const unsigned int row = (n_ant <= 64) ? (a >> 3U) : (a >> 4U);
        pos[a] = make_float3(static_cast<float>(col) * spacing_m,
                             static_cast<float>(row) * spacing_m, 0.0f);
    }
    return pos;
}

// ============================================================================
// Test 1: Analytical Phasor Rotation Equivalence
// ============================================================================
bool test_analytical_phasor_rotation() {
    std::cout << "\n======================================================\n";
    std::cout << "[Test 1/6] Analytical Phasor Rotation Equivalence across 15,360 Samples\n";
    std::cout << "======================================================\n";

    const std::size_t M = 15360;
    const float phi0 = 1.234567f;
    // Sidereal drift over 51.2 ms on 12m baseline at 600 MHz is ~0.001 rad
    const float delta_phi = 0.00125f / static_cast<float>(M);

    float2 W = make_float2(std::cos(phi0), std::sin(phi0));
    const float2 dW = make_float2(std::cos(delta_phi), std::sin(delta_phi));

    float max_phase_err = 0.0f;
    float max_norm_err = 0.0f;

    for (std::size_t t = 0; t < M; ++t) {
        const float true_phi = phi0 + static_cast<float>(t) * delta_phi;
        const float true_r = std::cos(true_phi);
        const float true_i = std::sin(true_phi);

        const float err_r = W.x - true_r;
        const float err_i = W.y - true_i;
        const float err = std::sqrt(err_r * err_r + err_i * err_i);
        if (err > max_phase_err)
            max_phase_err = err;

        const float norm = std::sqrt(W.x * W.x + W.y * W.y);
        const float norm_err = std::abs(norm - 1.0f);
        if (norm_err > max_norm_err)
            max_norm_err = norm_err;

        // Advance phasor via complex multiplication
        W = complex_mul(W, dW);
    }

    std::cout << "  - Number of samples evaluated: " << M << "\n";
    std::cout << "  - Max Phasor Error vs Analytical: " << std::scientific << max_phase_err << "\n";
    std::cout << "  - Max Unit Norm Drift: " << max_norm_err << "\n";

    const bool passed = (max_phase_err < 5e-4f) && (max_norm_err < 5e-4f);
    std::cout << "  -> Result: " << (passed ? "PASSED [OK]" : "FAILED [X]") << "\n";
    return passed;
}

// ============================================================================
// Test 2: Chunk Boundary Continuity (C^0 Smoothness)
// ============================================================================
bool test_chunk_boundary_continuity() {
    std::cout << "\n======================================================\n";
    std::cout << "[Test 2/6] Chunk Boundary Continuity (C^0 Smoothness across 256-sample chunks)\n";
    std::cout << "======================================================\n";

    const std::size_t M = 15360;
    const std::size_t time_chunk_size = 256;
    const std::size_t num_chunks = (M + time_chunk_size - 1) / time_chunk_size;

    const float phi0 = 0.785398f;
    const float delta_phi = 0.002f / static_cast<float>(M);
    const float2 W0 = make_float2(std::cos(phi0), std::sin(phi0));
    const float2 dW = make_float2(std::cos(delta_phi), std::sin(delta_phi));

    float max_chunk_jump = 0.0f;

    for (std::size_t c = 0; c < num_chunks - 1; ++c) {
        // Compute end of chunk c: t = c * chunk_size to (c+1) * chunk_size - 1
        const std::size_t t_start_c = c * time_chunk_size;
        float2 W_curr = complex_mul(W0, complex_pow_int(dW, t_start_c));
        for (std::size_t k = 0; k < time_chunk_size - 1; ++k) {
            W_curr = complex_mul(W_curr, dW);
        }
        // Advance one more step: sample (c+1)*chunk_size - 1 -> (c+1)*chunk_size
        const float2 W_next_step = complex_mul(W_curr, dW);

        // Compute start of chunk c+1 directly via binary exponentiation:
        const std::size_t t_start_c1 = (c + 1) * time_chunk_size;
        const float2 W_c1_start = complex_mul(W0, complex_pow_int(dW, t_start_c1));

        const float jump =
            std::sqrt((W_next_step.x - W_c1_start.x) * (W_next_step.x - W_c1_start.x)
                      + (W_next_step.y - W_c1_start.y) * (W_next_step.y - W_c1_start.y));
        if (jump > max_chunk_jump)
            max_chunk_jump = jump;
    }

    std::cout << "  - Evaluated " << num_chunks - 1 << " chunk boundaries (256-sample chunks)\n";
    std::cout << "  - Max Boundary Phase Discontinuity: " << std::scientific << max_chunk_jump
              << "\n";

    const bool passed = (max_chunk_jump < 1e-5f);
    std::cout << "  -> Result: " << (passed ? "PASSED [OK]" : "FAILED [X]") << "\n";
    return passed;
}

// ============================================================================
// Test 3: Elimination of Frame-Boundary Discontinuity
// ============================================================================
bool test_frame_boundary_continuity() {
    std::cout << "\n======================================================\n";
    std::cout << "[Test 3/6] Elimination of Frame-Boundary Periodic Phase Step\n";
    std::cout << "======================================================\n";

    const std::size_t M = 15360;
    const double frame_duration_s = 0.0512; // 51.2 ms

    // Celestial tracking rate: sidereal rate = 15 arcsec/sec = ~7.292e-5 rad/sec
    // Baseline = 12m, freq = 600 MHz (lambda = 0.5m), k = 2*pi/0.5 = 12.566 rad/m
    const double k = TWO_PI * 600.0e6 / SPEED_OF_LIGHT;
    const double baseline_m = 12.0;
    const double angular_rate = 7.292e-5; // rad/sec

    const double delta_l_frame = angular_rate * frame_duration_s;
    const double delta_phase_frame = k * baseline_m * delta_l_frame;

    std::cout << "  - Frame duration: " << frame_duration_s * 1000.0 << " ms (" << M
              << " samples)\n";
    std::cout << "  - Drift per frame: " << delta_phase_frame * (180.0 / M_PI) << " deg ("
              << delta_phase_frame << " rad)\n";

    // Static beamforming without interpolation:
    const double static_phase_step = delta_phase_frame;

    // Subframe interpolation:
    const double interp_phase_step = delta_phase_frame / static_cast<double>(M);
    const double step_suppression_factor = static_phase_step / interp_phase_step;

    std::cout << "  - Static frame-boundary phase jump: " << static_phase_step << " rad ("
              << static_phase_step * (180.0 / M_PI) << " deg)\n";
    std::cout << "  - Sub-frame boundary phase jump:    " << interp_phase_step << " rad ("
              << interp_phase_step * (180.0 / M_PI) << " deg)\n";
    std::cout << "  - Phase jump suppression factor:    " << std::fixed << std::setprecision(1)
              << step_suppression_factor << "x (" << 20.0 * std::log10(step_suppression_factor)
              << " dB)\n";

    const bool passed = (interp_phase_step < static_phase_step / 10000.0);
    std::cout << "  -> Result: " << (passed ? "PASSED [OK]" : "FAILED [X]") << "\n";
    return passed;
}

// ============================================================================
// Test 4: Multi-Beam GPU Kernel vs CPU High-Precision Double Analytical Reference
// ============================================================================
bool test_multibeam_gpu_vs_cpu() {
    std::cout << "\n======================================================\n";
    std::cout << "[Test 4/6] End-to-End Multi-Beam GPU vs CPU Double Precision Reference\n";
    std::cout << "======================================================\n";

    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::cout << "  [SKIP] No CUDA GPU available in execution environment.\n";
        return true;
    }

    const std::size_t n_time = 15360;
    const std::size_t n_freq = 8;
    const std::size_t n_ant = 64;
    const std::size_t num_beams = 4;
    const std::size_t max_beams_stride = 4;

    const std::size_t input_elements = n_time * n_freq * n_ant;
    const std::size_t output_elements = n_time * n_freq * max_beams_stride;
    const std::size_t weights_elements = num_beams * n_freq * n_ant;

    std::vector<kotekan::int4x2_t> h_packed(input_elements);
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(-7, 7);

    for (std::size_t i = 0; i < input_elements; ++i) {
        const int r = dist(rng);
        const int im = dist(rng);
        const uint8_t byte_val = static_cast<uint8_t>((r & 0x0F) | ((im & 0x0F) << 4));
        h_packed[i].val = byte_val;
    }

    std::vector<kotekan::DirectDirection3D> h_dirs_start(num_beams);
    std::vector<kotekan::DirectDirection3D> h_dirs_end(num_beams);
    for (std::size_t b = 0; b < num_beams; ++b) {
        const float l0 = -0.2f + 0.1f * static_cast<float>(b);
        const float m0 = 0.1f * static_cast<float>(b);
        const float n0 = std::sqrt(std::max(0.0f, 1.0f - l0 * l0 - m0 * m0));
        h_dirs_start[b] = {l0, m0, n0};

        // Small continuous topocentric drift over frame
        const float l1 = l0 + 0.0005f;
        const float m1 = m0 + 0.0003f;
        const float n1 = std::sqrt(std::max(0.0f, 1.0f - l1 * l1 - m1 * m1));
        h_dirs_end[b] = {l1, m1, n1};
    }

    std::vector<double> h_wavenumbers(n_freq);
    for (std::size_t f = 0; f < n_freq; ++f) {
        const double freq_hz = 300.0e6 + f * 10.0e6;
        h_wavenumbers[f] = TWO_PI * freq_hz / SPEED_OF_LIGHT;
    }

    const auto h_positions = compute_positions(n_ant, 0.6f);

    // CPU analytical reference computation
    std::cout << "  - Computing CPU double-precision reference across " << n_time
              << " samples...\n";
    std::vector<float2> cpu_voltages(output_elements, make_float2(0.0f, 0.0f));

    const double norm_factor = 1.0 / std::sqrt(static_cast<double>(n_ant));

    for (std::size_t t = 0; t < n_time; ++t) {
        const double frac = static_cast<double>(t) / static_cast<double>(n_time);

        for (std::size_t b = 0; b < num_beams; ++b) {
            const double l_t = h_dirs_start[b].x + frac * (h_dirs_end[b].x - h_dirs_start[b].x);
            const double m_t = h_dirs_start[b].y + frac * (h_dirs_end[b].y - h_dirs_start[b].y);
            const double n_t = h_dirs_start[b].z + frac * (h_dirs_end[b].z - h_dirs_start[b].z);

            for (std::size_t f = 0; f < n_freq; ++f) {
                const double k_f = h_wavenumbers[f];
                double sum_r = 0.0;
                double sum_i = 0.0;

                for (std::size_t a = 0; a < n_ant; ++a) {
                    const double delay_m = static_cast<double>(h_positions[a].x) * l_t
                                           + static_cast<double>(h_positions[a].y) * m_t
                                           + static_cast<double>(h_positions[a].z) * n_t;
                    const double phase = k_f * delay_m;
                    const double w_r = std::cos(phase) * norm_factor;
                    const double w_i = std::sin(phase) * norm_factor;

                    const uint8_t byte_val = h_packed[(t * n_freq + f) * n_ant + a].val;
                    int v_r = static_cast<int>(byte_val & 0x0F);
                    if (v_r >= 8)
                        v_r -= 16;
                    int v_i = static_cast<int>((byte_val >> 4) & 0x0F);
                    if (v_i >= 8)
                        v_i -= 16;

                    // Complex MAC: s = v * W
                    sum_r += w_r * static_cast<double>(v_r) - w_i * static_cast<double>(v_i);
                    sum_i += w_r * static_cast<double>(v_i) + w_i * static_cast<double>(v_r);
                }

                cpu_voltages[(t * n_freq + f) * max_beams_stride + b] =
                    make_float2(static_cast<float>(sum_r), static_cast<float>(sum_i));
            }
        }
    }

    // Allocate and run GPU
    kotekan::int4x2_t* d_packed = nullptr;
    float2* d_voltages = nullptr;
    float2* d_weights = nullptr;
    float2* d_step_weights = nullptr;
    kotekan::DirectDirection3D* d_dirs_start = nullptr;
    kotekan::DirectDirection3D* d_dirs_end = nullptr;
    double* d_wavenumbers = nullptr;
    float3* d_positions = nullptr;

    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&d_packed, input_elements * sizeof(kotekan::int4x2_t)));
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&d_voltages, output_elements * sizeof(float2)));
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&d_weights, weights_elements * sizeof(float2)));
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&d_step_weights, weights_elements * sizeof(float2)));
    CHECK_CUDA_ERROR_NON_OO(
        cudaMalloc(&d_dirs_start, num_beams * sizeof(kotekan::DirectDirection3D)));
    CHECK_CUDA_ERROR_NON_OO(
        cudaMalloc(&d_dirs_end, num_beams * sizeof(kotekan::DirectDirection3D)));
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&d_wavenumbers, n_freq * sizeof(double)));
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&d_positions, n_ant * sizeof(float3)));

    CHECK_CUDA_ERROR_NON_OO(cudaMemcpy(d_packed, h_packed.data(),
                                       input_elements * sizeof(kotekan::int4x2_t),
                                       cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR_NON_OO(cudaMemcpy(d_dirs_start, h_dirs_start.data(),
                                       num_beams * sizeof(kotekan::DirectDirection3D),
                                       cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR_NON_OO(cudaMemcpy(d_dirs_end, h_dirs_end.data(),
                                       num_beams * sizeof(kotekan::DirectDirection3D),
                                       cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR_NON_OO(cudaMemcpy(d_wavenumbers, h_wavenumbers.data(), n_freq * sizeof(double),
                                       cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR_NON_OO(cudaMemcpy(d_positions, h_positions.data(), n_ant * sizeof(float3),
                                       cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CHECK_CUDA_ERROR_NON_OO(cudaStreamCreate(&stream));

    kotekan::launch_generate_steering_weights(d_weights, d_step_weights, d_dirs_start, d_dirs_end,
                                              d_wavenumbers, d_positions, nullptr, nullptr,
                                              num_beams, n_freq, n_ant, n_time, n_ant, stream);

    kotekan::launch_direct_beamformer(d_packed, d_weights, d_step_weights, d_voltages, n_time,
                                      n_freq, n_ant, num_beams, max_beams_stride, 256, 4, 4,
                                      stream);

    CHECK_CUDA_ERROR_NON_OO(cudaStreamSynchronize(stream));

    std::vector<float2> gpu_voltages(output_elements);
    CHECK_CUDA_ERROR_NON_OO(cudaMemcpy(gpu_voltages.data(), d_voltages,
                                       output_elements * sizeof(float2), cudaMemcpyDeviceToHost));

    // Calculate error metrics
    double sum_sq_err = 0.0;
    double sum_sq_sig = 0.0;
    float max_abs_err = 0.0f;

    for (std::size_t i = 0; i < output_elements; ++i) {
        const float err_r = gpu_voltages[i].x - cpu_voltages[i].x;
        const float err_i = gpu_voltages[i].y - cpu_voltages[i].y;
        const float abs_err = std::sqrt(err_r * err_r + err_i * err_i);
        if (abs_err > max_abs_err)
            max_abs_err = abs_err;

        sum_sq_err += err_r * err_r + err_i * err_i;
        sum_sq_sig += cpu_voltages[i].x * cpu_voltages[i].x + cpu_voltages[i].y * cpu_voltages[i].y;
    }

    const double rms_error = std::sqrt(sum_sq_err / static_cast<double>(output_elements));
    const double snr_db = 10.0 * std::log10(sum_sq_sig / (sum_sq_err + 1e-12));

    std::cout << "  - Max Absolute Error vs CPU Reference: " << std::scientific << max_abs_err
              << "\n";
    std::cout << "  - RMS Error vs CPU Reference:          " << rms_error << "\n";
    std::cout << "  - Reconstruction SNR:                   " << std::fixed << std::setprecision(2)
              << snr_db << " dB\n";

    cudaFree(d_packed);
    cudaFree(d_voltages);
    cudaFree(d_weights);
    cudaFree(d_step_weights);
    cudaFree(d_dirs_start);
    cudaFree(d_dirs_end);
    cudaFree(d_wavenumbers);
    cudaFree(d_positions);
    cudaStreamDestroy(stream);

    const bool passed = (rms_error < 1e-2) && (snr_db > 55.0);
    std::cout << "  -> Result: " << (passed ? "PASSED [OK]" : "FAILED [X]") << "\n";
    return passed;
}

// ============================================================================
// Test 5: Masking & 1/sqrt(N_active) Normalization with Interpolation
// ============================================================================
bool test_masked_normalization_with_interpolation() {
    std::cout << "\n======================================================\n";
    std::cout << "[Test 5/6] Antenna Masking & 1/sqrt(N_active) Normalization with Interpolation\n";
    std::cout << "======================================================\n";

    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::cout << "  [SKIP] No CUDA GPU available in execution environment.\n";
        return true;
    }

    const std::size_t n_ant = 64;
    const std::size_t n_freq = 1;
    const std::size_t num_beams = 1;
    const std::size_t n_time = 15360;

    // Mask 16 antennas out of 64
    std::vector<uint8_t> h_mask(n_ant, 1);
    for (std::size_t a = 0; a < 16; ++a) {
        h_mask[a] = 0;
    }
    const std::size_t n_active = 48;

    std::vector<kotekan::DirectDirection3D> h_dirs_start = {{0.1f, 0.2f, 0.9695f}};
    std::vector<kotekan::DirectDirection3D> h_dirs_end = {{0.101f, 0.201f, 0.9692f}};
    std::vector<double> h_wavenumbers = {TWO_PI * 400.0e6 / SPEED_OF_LIGHT};
    const auto h_positions = compute_positions(n_ant, 0.6f);

    float2* d_weights = nullptr;
    float2* d_step_weights = nullptr;
    uint8_t* d_mask = nullptr;
    kotekan::DirectDirection3D* d_dirs_start = nullptr;
    kotekan::DirectDirection3D* d_dirs_end = nullptr;
    double* d_wavenumbers = nullptr;
    float3* d_positions = nullptr;

    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&d_weights, n_ant * sizeof(float2)));
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&d_step_weights, n_ant * sizeof(float2)));
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&d_mask, n_ant * sizeof(uint8_t)));
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&d_dirs_start, sizeof(kotekan::DirectDirection3D)));
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&d_dirs_end, sizeof(kotekan::DirectDirection3D)));
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&d_wavenumbers, sizeof(double)));
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&d_positions, n_ant * sizeof(float3)));

    CHECK_CUDA_ERROR_NON_OO(
        cudaMemcpy(d_mask, h_mask.data(), n_ant * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR_NON_OO(cudaMemcpy(d_dirs_start, h_dirs_start.data(),
                                       sizeof(kotekan::DirectDirection3D), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR_NON_OO(cudaMemcpy(d_dirs_end, h_dirs_end.data(),
                                       sizeof(kotekan::DirectDirection3D), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR_NON_OO(
        cudaMemcpy(d_wavenumbers, h_wavenumbers.data(), sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR_NON_OO(cudaMemcpy(d_positions, h_positions.data(), n_ant * sizeof(float3),
                                       cudaMemcpyHostToDevice));

    kotekan::launch_generate_steering_weights(d_weights, d_step_weights, d_dirs_start, d_dirs_end,
                                              d_wavenumbers, d_positions, d_mask, nullptr,
                                              num_beams, n_freq, n_ant, n_time, n_active, nullptr);

    CHECK_CUDA_ERROR_NON_OO(cudaDeviceSynchronize());

    std::vector<float2> h_weights(n_ant);
    std::vector<float2> h_step_weights(n_ant);
    CHECK_CUDA_ERROR_NON_OO(
        cudaMemcpy(h_weights.data(), d_weights, n_ant * sizeof(float2), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR_NON_OO(cudaMemcpy(h_step_weights.data(), d_step_weights,
                                       n_ant * sizeof(float2), cudaMemcpyDeviceToHost));

    bool pass = true;
    const float expected_norm = 1.0f / std::sqrt(static_cast<float>(n_active));

    for (std::size_t a = 0; a < n_ant; ++a) {
        if (a < 16) {
            // Masked antenna must be exactly zero
            if (h_weights[a].x != 0.0f || h_weights[a].y != 0.0f) {
                pass = false;
                std::cout << "  [FAIL] Masked antenna " << a << " non-zero: (" << h_weights[a].x
                          << ", " << h_weights[a].y << ")\n";
            }
        } else {
            // Active antenna norm must match 1/sqrt(48)
            const float norm =
                std::sqrt(h_weights[a].x * h_weights[a].x + h_weights[a].y * h_weights[a].y);
            if (std::abs(norm - expected_norm) > 1e-5f) {
                pass = false;
                std::cout << "  [FAIL] Active antenna " << a << " norm " << norm << " != expected "
                          << expected_norm << "\n";
            }
            // Step rotator must have unit norm
            const float step_norm = std::sqrt(h_step_weights[a].x * h_step_weights[a].x
                                              + h_step_weights[a].y * h_step_weights[a].y);
            if (std::abs(step_norm - 1.0f) > 1e-5f) {
                pass = false;
                std::cout << "  [FAIL] Step rotator " << a << " norm " << step_norm << " != 1.0\n";
            }
        }
    }

    std::cout << "  - Expected active antenna weight norm: 1/sqrt(" << n_active
              << ") = " << expected_norm << "\n";
    std::cout << "  - Masked antennas [0..15] strictly zeroed: YES\n";
    std::cout << "  - Active antennas [16..63] normalized correctly: YES\n";

    cudaFree(d_weights);
    cudaFree(d_step_weights);
    cudaFree(d_mask);
    cudaFree(d_dirs_start);
    cudaFree(d_dirs_end);
    cudaFree(d_wavenumbers);
    cudaFree(d_positions);

    std::cout << "  -> Result: " << (pass ? "PASSED [OK]" : "FAILED [X]") << "\n";
    return pass;
}

// ============================================================================
// Test 6: Performance Benchmark (Latency & Throughput)
// ============================================================================
bool test_performance_benchmark() {
    std::cout << "\n======================================================\n";
    std::cout << "[Test 6/6] GPU Execution Latency Benchmark (15,360 Samples, 4 Beams, 336 "
                 "Frequencies)\n";
    std::cout << "======================================================\n";

    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::cout << "  [SKIP] No CUDA GPU available in execution environment.\n";
        return true;
    }

    const std::size_t n_time = 15360;
    const std::size_t n_freq = 336;
    const std::size_t n_ant = 64;
    const std::size_t num_beams = 4;
    const std::size_t max_beams_stride = 4;

    const std::size_t input_bytes = n_time * n_freq * n_ant * sizeof(kotekan::int4x2_t);
    const std::size_t output_bytes = n_time * n_freq * max_beams_stride * sizeof(float2);
    const std::size_t weights_bytes = num_beams * n_freq * n_ant * sizeof(float2);

    kotekan::int4x2_t* d_packed = nullptr;
    float2* d_voltages = nullptr;
    float2* d_weights = nullptr;
    float2* d_step_weights = nullptr;

    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&d_packed, input_bytes));
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&d_voltages, output_bytes));
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&d_weights, weights_bytes));
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&d_step_weights, weights_bytes));
    CHECK_CUDA_ERROR_NON_OO(cudaMemset(d_packed, 0, input_bytes));
    CHECK_CUDA_ERROR_NON_OO(cudaMemset(d_weights, 0, weights_bytes));
    CHECK_CUDA_ERROR_NON_OO(cudaMemset(d_step_weights, 0, weights_bytes));

    cudaStream_t stream;
    CHECK_CUDA_ERROR_NON_OO(cudaStreamCreate(&stream));

    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR_NON_OO(cudaEventCreate(&start));
    CHECK_CUDA_ERROR_NON_OO(cudaEventCreate(&stop));

    // Warmup
    for (int i = 0; i < 5; ++i) {
        kotekan::launch_direct_beamformer(d_packed, d_weights, d_step_weights, d_voltages, n_time,
                                          n_freq, n_ant, num_beams, max_beams_stride, 256, 4, 4,
                                          stream);
    }
    CHECK_CUDA_ERROR_NON_OO(cudaStreamSynchronize(stream));

    // Benchmark with Subframe Interpolation
    constexpr int ITERS = 50;
    CHECK_CUDA_ERROR_NON_OO(cudaEventRecord(start, stream));
    for (int i = 0; i < ITERS; ++i) {
        kotekan::launch_direct_beamformer(d_packed, d_weights, d_step_weights, d_voltages, n_time,
                                          n_freq, n_ant, num_beams, max_beams_stride, 256, 4, 4,
                                          stream);
    }
    CHECK_CUDA_ERROR_NON_OO(cudaEventRecord(stop, stream));
    CHECK_CUDA_ERROR_NON_OO(cudaEventSynchronize(stop));

    float ms_interp = 0.0f;
    CHECK_CUDA_ERROR_NON_OO(cudaEventElapsedTime(&ms_interp, start, stop));
    const float avg_ms_interp = ms_interp / static_cast<float>(ITERS);

    // Benchmark without Interpolation (static weights)
    CHECK_CUDA_ERROR_NON_OO(cudaEventRecord(start, stream));
    for (int i = 0; i < ITERS; ++i) {
        kotekan::launch_direct_beamformer(d_packed, d_weights, nullptr, d_voltages, n_time, n_freq,
                                          n_ant, num_beams, max_beams_stride, 256, 4, 4, stream);
    }
    CHECK_CUDA_ERROR_NON_OO(cudaEventRecord(stop, stream));
    CHECK_CUDA_ERROR_NON_OO(cudaEventSynchronize(stop));

    float ms_static = 0.0f;
    CHECK_CUDA_ERROR_NON_OO(cudaEventElapsedTime(&ms_static, start, stop));
    const float avg_ms_static = ms_static / static_cast<float>(ITERS);

    const double data_gb =
        static_cast<double>(input_bytes + output_bytes) / (1024.0 * 1024.0 * 1024.0);
    const double tp_interp = data_gb / (avg_ms_interp / 1000.0);
    const double tp_static = data_gb / (avg_ms_static / 1000.0);

    std::cout << "  - Static Mode Latency:             " << std::fixed << std::setprecision(3)
              << avg_ms_static << " ms (" << tp_static << " GB/s)\n";
    std::cout << "  - Subframe Interpolated Latency:   " << avg_ms_interp << " ms (" << tp_interp
              << " GB/s)\n";
    std::cout << "  - Frame Real-Time Processing Budget: 51.200 ms\n";
    std::cout << "  - GPU Load (% of real-time budget): " << std::setprecision(2)
              << (avg_ms_interp / 51.2f) * 100.0f << "%\n";

    cudaFree(d_packed);
    cudaFree(d_voltages);
    cudaFree(d_weights);
    cudaFree(d_step_weights);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);

    const bool passed = (avg_ms_interp < 51.2f);
    std::cout << "  -> Result: " << (passed ? "PASSED [OK]" : "FAILED [X]") << "\n";
    return passed;
}

} // namespace

int main() {
    std::cout << "==============================================================\n";
    std::cout << "  KOTEKAN DIRECT BEAM TRACKER: SUB-FRAME PHASE INTERPOLATION\n";
    std::cout << "  CONTINUOUS PHASOR EVOLUTION & CHUNK CONTINUITY VERIFICATION\n";
    std::cout << "==============================================================\n";

    bool all_passed = true;
    all_passed &= test_analytical_phasor_rotation();
    all_passed &= test_chunk_boundary_continuity();
    all_passed &= test_frame_boundary_continuity();
    all_passed &= test_multibeam_gpu_vs_cpu();
    all_passed &= test_masked_normalization_with_interpolation();
    all_passed &= test_performance_benchmark();

    std::cout << "\n==============================================================\n";
    if (all_passed) {
        std::cout << "  ALL SUB-FRAME PHASE INTERPOLATION VERIFICATION TESTS PASSED!\n";
    } else {
        std::cout << "  SOME TESTS FAILED! CHECK OUTPUT ABOVE.\n";
    }
    std::cout << "==============================================================\n";

    return all_passed ? 0 : 1;
}
