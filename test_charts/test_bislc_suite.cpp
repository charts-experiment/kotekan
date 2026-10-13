#include "chartsConstants.hpp"
#include "cudaBiSLC.hpp"
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

// Helper: compute complex matrix product C = A * B
void matrix_mult_complex(const float2* A, const float2* B, float2* C, std::size_t N) {
    for (std::size_t i = 0; i < N; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            float sr = 0.0f;
            float si = 0.0f;
            for (std::size_t k = 0; k < N; ++k) {
                const float2 a = A[i * N + k];
                const float2 b = B[k * N + j];
                sr += a.x * b.x - a.y * b.y;
                si += a.x * b.y + a.y * b.x;
            }
            C[i * N + j] = make_float2(sr, si);
        }
    }
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

// Test 1: Analytical and Numerical Matrix Inversion Test
bool test_matrix_inversion() {
    std::cout << "\n======================================================\n";
    std::cout << "[Test 1/4] Complex Matrix Inversion Accuracy & Stability\n";
    std::cout << "======================================================\n";

    bool all_passed = true;
    const std::vector<std::size_t> beam_counts = {1, 2, 3, 4, 8};

    for (std::size_t B : beam_counts) {
        std::vector<float2> A(B * B);
        std::vector<float2> M(B * B);
        std::vector<float2> I_test(B * B);

        // Generate synthetic well-conditioned Hermitian coupling matrix A
        for (std::size_t i = 0; i < B; ++i) {
            for (std::size_t j = 0; j < B; ++j) {
                if (i == j) {
                    A[i * B + j] = make_float2(1.0f, 0.0f);
                } else if (i < j) {
                    float coupling_mag =
                        0.15f
                        / static_cast<float>(std::abs(static_cast<int>(i) - static_cast<int>(j)));
                    float phase = static_cast<float>(0.3 * (i + 1) - 0.2 * (j + 1));
                    A[i * B + j] =
                        make_float2(coupling_mag * std::cos(phase), coupling_mag * std::sin(phase));
                } else {
                    // Hermitian conjugate: A_{j, i} = A_{i, j}^*
                    A[i * B + j] = make_float2(A[j * B + i].x, -A[j * B + i].y);
                }
            }
        }

        bool success = kotekan::invert_complex_matrix(A.data(), M.data(), B, 1.0e-4f);
        if (!success) {
            std::cerr << "  FAILED: Inversion reported failure for B=" << B << "\n";
            all_passed = false;
            continue;
        }

        // Test M * (A + eps * I) ~ I
        std::vector<float2> A_reg = A;
        for (std::size_t i = 0; i < B; ++i) {
            A_reg[i * B + i].x += 1.0e-4f;
        }
        matrix_mult_complex(M.data(), A_reg.data(), I_test.data(), B);

        float max_err = 0.0f;
        for (std::size_t i = 0; i < B; ++i) {
            for (std::size_t j = 0; j < B; ++j) {
                float expected_r = (i == j) ? 1.0f : 0.0f;
                float expected_i = 0.0f;
                float err_r = std::abs(I_test[i * B + j].x - expected_r);
                float err_i = std::abs(I_test[i * B + j].y - expected_i);
                max_err = std::max(max_err, std::max(err_r, err_i));
            }
        }

        std::cout << "  B=" << B << "x" << B << " Inversion Residual Error: " << std::scientific
                  << std::setprecision(3) << max_err;
        if (max_err < 1.0e-4f) {
            std::cout << " [PASS]\n";
        } else {
            std::cout << " [FAIL]\n";
            all_passed = false;
        }
    }

    return all_passed;
}

// Test 2: Coupling Matrix Generation & Orthogonality Check
bool test_coupling_matrix_generation() {
    std::cout << "\n======================================================\n";
    std::cout << "[Test 2/4] Beam-Coupling Matrix Generation\n";
    std::cout << "======================================================\n";

    const std::size_t n_ant = 64;
    const std::size_t n_freq = 4;
    const std::size_t num_beams = 2;
    const float spacing_m = 0.6f;

    auto pos = compute_positions(n_ant, spacing_m);
    std::vector<uint8_t> mask(n_ant, 1);
    std::vector<double> freqs = {300.0e6, 350.0e6, 400.0e6, 450.0e6};

    kotekan::DirectBeamTarget targets[2];
    // Beam 0: Zenith
    targets[0].direction = kotekan::DirectDirection3D{0.0f, 0.0f, 1.0f};
    // Beam 1: Separated by 5 degrees along l
    const float l1 = 0.0871557f; // sin(5 deg)
    targets[1].direction = kotekan::DirectDirection3D{l1, 0.0f, std::sqrt(1.0f - l1 * l1)};

    std::vector<float2> M(n_freq * num_beams * num_beams);
    std::vector<float2> A(n_freq * num_beams * num_beams);

    kotekan::compute_beam_coupling_and_inverses(M.data(), A.data(), targets, freqs, pos.data(),
                                                mask.data(), num_beams, n_freq, n_ant, n_ant,
                                                1.0e-4f);

    bool passed = true;
    for (std::size_t f = 0; f < n_freq; ++f) {
        const std::size_t base = f * 4;
        const float2 a00 = A[base + 0];
        const float2 a01 = A[base + 1];
        const float2 a10 = A[base + 2];
        const float2 a11 = A[base + 3];

        // Diagonal must be (1.0, 0.0)
        if (std::abs(a00.x - 1.0f) > 1.0e-5f || std::abs(a00.y) > 1.0e-5f
            || std::abs(a11.x - 1.0f) > 1.0e-5f || std::abs(a11.y) > 1.0e-5f) {
            std::cerr << "  FAILED: Diagonal is not 1.0 at channel " << f << "\n";
            passed = false;
        }

        // Must be Hermitian: a01 == a10^*
        if (std::abs(a01.x - a10.x) > 1.0e-5f || std::abs(a01.y + a10.y) > 1.0e-5f) {
            std::cerr << "  FAILED: Matrix is not Hermitian at channel " << f << "\n";
            passed = false;
        }

        const float coupling_pwr = a01.x * a01.x + a01.y * a01.y;
        std::cout << "  Freq=" << freqs[f] / 1.0e6
                  << " MHz: Sidelobe Coupling Power A_01=" << std::fixed << std::setprecision(4)
                  << coupling_pwr << " (" << 10.0 * std::log10(std::max(coupling_pwr, 1.0e-8f))
                  << " dB)\n";
    }

    std::cout << "  Result: " << (passed ? "[PASS]" : "[FAIL]") << "\n";
    return passed;
}

// Test 3: Coherent Sidelobe Cancellation Verification (Interference Suppression)
bool test_sidelobe_cancellation() {
    std::cout << "\n======================================================\n";
    std::cout << "[Test 3/4] Coherent Sidelobe Cancellation Suppression\n";
    std::cout << "======================================================\n";

    const std::size_t n_time = 1024;
    const std::size_t n_freq = 1;
    const std::size_t num_beams = 2;
    const std::size_t n_ant = 64;
    const float spacing_m = 0.6f;

    auto pos = compute_positions(n_ant, spacing_m);
    std::vector<uint8_t> mask(n_ant, 1);
    std::vector<double> freqs = {350.0e6};

    kotekan::DirectBeamTarget targets[2];
    targets[0].direction = kotekan::DirectDirection3D{0.0f, 0.0f, 1.0f}; // Target 0 (Zenith)
    const float l1 = 0.06f; // Sidelobe location (~3.4 degrees)
    targets[1].direction =
        kotekan::DirectDirection3D{l1, 0.0f, std::sqrt(1.0f - l1 * l1)}; // Target 1 (Interferer)

    std::vector<float2> M(num_beams * num_beams);
    std::vector<float2> A(num_beams * num_beams);

    kotekan::compute_beam_coupling_and_inverses(M.data(), A.data(), targets, freqs, pos.data(),
                                                mask.data(), num_beams, n_freq, n_ant, n_ant,
                                                1.0e-5f);

    const float2 alpha = A[1]; // Sidelobe leakage coefficient from beam 1 into beam 0
    std::cout << "  Coupling Alpha (Beam 1 -> Beam 0): (" << alpha.x << " + j*" << alpha.y << ")\n";

    // Simulate Ground Truth signals:
    // s_0(t) is a weak target signal (amplitude = 1.0)
    // s_1(t) is a strong interfering signal (amplitude = 25.0)
    std::vector<float2> s_true_0(n_time);
    std::vector<float2> s_true_1(n_time);
    std::vector<float2> y_formed(n_time * 2);

    double true_power_0 = 0.0;
    double interf_power_1 = 0.0;

    for (std::size_t t = 0; t < n_time; ++t) {
        const double phase_target = TWO_PI * 0.03 * t;
        const double phase_interf = TWO_PI * 0.17 * t;

        const float s0_r = static_cast<float>(std::cos(phase_target));
        const float s0_i = static_cast<float>(std::sin(phase_target));
        const float s1_r = static_cast<float>(25.0 * std::cos(phase_interf));
        const float s1_i = static_cast<float>(25.0 * std::sin(phase_interf));

        s_true_0[t] = make_float2(s0_r, s0_i);
        s_true_1[t] = make_float2(s1_r, s1_i);

        true_power_0 += s0_r * s0_r + s0_i * s0_i;
        interf_power_1 += s1_r * s1_r + s1_i * s1_i;

        // Observed formed beams before unmixing: y = A * s
        // y_0 = s_0 + alpha * s_1
        // y_1 = alpha^* * s_0 + s_1
        const float y0_r = s0_r + (alpha.x * s1_r - alpha.y * s1_i);
        const float y0_i = s0_i + (alpha.x * s1_i + alpha.y * s1_r);

        const float y1_r = s1_r + (alpha.x * s0_r - (-alpha.y) * s0_i);
        const float y1_i = s1_i + (alpha.x * s0_i + (-alpha.y) * s0_r);

        y_formed[t * 2 + 0] = make_float2(y0_r, y0_i);
        y_formed[t * 2 + 1] = make_float2(y1_r, y1_i);
    }

    true_power_0 /= n_time;
    interf_power_1 /= n_time;

    // Measure error before unmixing: Beam 0 corrupted by Beam 1's sidelobe
    double err_before = 0.0;
    for (std::size_t t = 0; t < n_time; ++t) {
        const float dr = y_formed[t * 2 + 0].x - s_true_0[t].x;
        const float di = y_formed[t * 2 + 0].y - s_true_0[t].y;
        err_before += dr * dr + di * di;
    }
    err_before /= n_time;

    // Apply BiSLC Unmixing: s_clean = M * y
    std::vector<float2> s_clean(n_time * 2);
    for (std::size_t t = 0; t < n_time; ++t) {
        const float2 y0 = y_formed[t * 2 + 0];
        const float2 y1 = y_formed[t * 2 + 1];

        // s0 = M00 * y0 + M01 * y1
        const float s0_clean_r = M[0].x * y0.x - M[0].y * y0.y + M[1].x * y1.x - M[1].y * y1.y;
        const float s0_clean_i = M[0].x * y0.y + M[0].y * y0.x + M[1].x * y1.y + M[1].y * y1.x;

        // s1 = M10 * y0 + M11 * y1
        const float s1_clean_r = M[2].x * y0.x - M[2].y * y0.y + M[3].x * y1.x - M[3].y * y1.y;
        const float s1_clean_i = M[2].x * y0.y + M[2].y * y0.x + M[3].x * y1.y + M[3].y * y1.x;

        s_clean[t * 2 + 0] = make_float2(s0_clean_r, s0_clean_i);
        s_clean[t * 2 + 1] = make_float2(s1_clean_r, s1_clean_i);
    }

    double err_after = 0.0;
    for (std::size_t t = 0; t < n_time; ++t) {
        const float dr = s_clean[t * 2 + 0].x - s_true_0[t].x;
        const float di = s_clean[t * 2 + 0].y - s_true_0[t].y;
        err_after += dr * dr + di * di;
    }
    err_after /= n_time;

    const double suppression_ratio_db = 10.0 * std::log10(err_before / err_after);

    std::cout << "  Target 0 True Power:              " << true_power_0 << "\n";
    std::cout << "  Interferer 1 Power:               " << interf_power_1 << " ("
              << 10.0 * std::log10(interf_power_1 / true_power_0) << " dB higher)\n";
    std::cout << "  Residual Error Before BiSLC:       " << err_before << " ("
              << 10.0 * std::log10(err_before) << " dB)\n";
    std::cout << "  Residual Error After BiSLC:        " << err_after << " ("
              << 10.0 * std::log10(err_after) << " dB)\n";
    std::cout << "  Sidelobe Interference Suppression: " << std::fixed << std::setprecision(1)
              << suppression_ratio_db << " dB\n";

    bool passed = (suppression_ratio_db >= 30.0);
    std::cout << "  Result: " << (passed ? "[PASS]" : "[FAIL]")
              << " (Requirement: >= 30 dB suppression)\n";
    return passed;
}

// Test 4: GPU Kernel Accuracy & Throughput Benchmark
bool test_gpu_kernel_benchmark() {
    std::cout << "\n======================================================\n";
    std::cout << "[Test 4/4] GPU BiSLC Kernel Parity & Throughput Benchmark\n";
    std::cout << "======================================================\n";

    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count == 0) {
        std::cout << "  No CUDA-capable GPU detected in this test environment. Skipping GPU kernel "
                     "launch.\n";
        return true;
    }

    const std::size_t n_time = 15360; // Production frame size (51.2 ms)
    const std::size_t n_freq = 336;   // Subband channel count
    const std::size_t num_beams = 2;  // 2 active beams
    const std::size_t max_stride = 2;

    const std::size_t buffer_bytes = n_time * n_freq * max_stride * sizeof(float2);
    const std::size_t matrix_bytes = n_freq * num_beams * num_beams * sizeof(float2);

    std::vector<float2> h_formed(n_time * n_freq * max_stride);
    std::vector<float2> h_matrices(n_freq * num_beams * num_beams);
    std::vector<float2> h_expected(n_time * n_freq * max_stride);

    // Populate synthetic input data and matrices
    for (std::size_t i = 0; i < h_formed.size(); ++i) {
        h_formed[i] = make_float2(static_cast<float>(static_cast<int>(i % 17) - 8),
                                  static_cast<float>(static_cast<int>(i % 13) - 6));
    }
    for (std::size_t f = 0; f < n_freq; ++f) {
        h_matrices[f * 4 + 0] = make_float2(1.02f, 0.01f);
        h_matrices[f * 4 + 1] = make_float2(-0.10f, 0.05f);
        h_matrices[f * 4 + 2] = make_float2(-0.10f, -0.05f);
        h_matrices[f * 4 + 3] = make_float2(1.02f, -0.01f);
    }

    // CPU Reference computation
    for (std::size_t t = 0; t < n_time; ++t) {
        for (std::size_t f = 0; f < n_freq; ++f) {
            const std::size_t in_idx = (t * n_freq + f) * max_stride;
            const std::size_t m_idx = f * 4;

            const float2 y0 = h_formed[in_idx + 0];
            const float2 y1 = h_formed[in_idx + 1];

            const float2 m00 = h_matrices[m_idx + 0];
            const float2 m01 = h_matrices[m_idx + 1];
            const float2 m10 = h_matrices[m_idx + 2];
            const float2 m11 = h_matrices[m_idx + 3];

            h_expected[in_idx + 0] =
                make_float2(m00.x * y0.x - m00.y * y0.y + m01.x * y1.x - m01.y * y1.y,
                            m00.x * y0.y + m00.y * y0.x + m01.x * y1.y + m01.y * y1.x);

            h_expected[in_idx + 1] =
                make_float2(m10.x * y0.x - m10.y * y0.y + m11.x * y1.x - m11.y * y1.y,
                            m10.x * y0.y + m10.y * y0.x + m11.x * y1.y + m11.y * y1.x);
        }
    }

    float2* d_in = nullptr;
    float2* d_out = nullptr;
    float2* d_mat = nullptr;

    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&d_in, buffer_bytes));
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&d_out, buffer_bytes));
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&d_mat, matrix_bytes));

    cudaStream_t stream;
    CHECK_CUDA_ERROR_NON_OO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    CHECK_CUDA_ERROR_NON_OO(
        cudaMemcpyAsync(d_in, h_formed.data(), buffer_bytes, cudaMemcpyHostToDevice, stream));
    CHECK_CUDA_ERROR_NON_OO(
        cudaMemcpyAsync(d_mat, h_matrices.data(), matrix_bytes, cudaMemcpyHostToDevice, stream));

    // Warm-up
    kotekan::launch_bislc_unmixing(d_out, d_in, d_mat, n_time, n_freq, num_beams, max_stride,
                                   stream);
    CHECK_CUDA_ERROR_NON_OO(cudaStreamSynchronize(stream));

    // Verify numerical equivalence
    std::vector<float2> h_gpu_out(n_time * n_freq * max_stride);
    CHECK_CUDA_ERROR_NON_OO(
        cudaMemcpy(h_gpu_out.data(), d_out, buffer_bytes, cudaMemcpyDeviceToHost));

    float max_diff = 0.0f;
    for (std::size_t i = 0; i < h_gpu_out.size(); ++i) {
        float dr = std::abs(h_gpu_out[i].x - h_expected[i].x);
        float di = std::abs(h_gpu_out[i].y - h_expected[i].y);
        max_diff = std::max(max_diff, std::max(dr, di));
    }
    std::cout << "  GPU vs CPU Parity Max Discrepancy: " << std::scientific << std::setprecision(3)
              << max_diff;
    bool parity_pass = (max_diff < 1.0e-4f);
    std::cout << " " << (parity_pass ? "[PASS]" : "[FAIL]") << "\n";

    // Benchmark timing
    cudaEvent_t ev_start, ev_stop;
    CHECK_CUDA_ERROR_NON_OO(cudaEventCreate(&ev_start));
    CHECK_CUDA_ERROR_NON_OO(cudaEventCreate(&ev_stop));

    constexpr int ITERS = 100;
    CHECK_CUDA_ERROR_NON_OO(cudaEventRecord(ev_start, stream));
    for (int it = 0; it < ITERS; ++it) {
        kotekan::launch_bislc_unmixing(d_out, d_in, d_mat, n_time, n_freq, num_beams, max_stride,
                                       stream);
    }
    CHECK_CUDA_ERROR_NON_OO(cudaEventRecord(ev_stop, stream));
    CHECK_CUDA_ERROR_NON_OO(cudaEventSynchronize(ev_stop));

    float total_ms = 0.0f;
    CHECK_CUDA_ERROR_NON_OO(cudaEventElapsedTime(&total_ms, ev_start, ev_stop));
    const float kernel_ms = total_ms / ITERS;

    const double data_gb =
        static_cast<double>(buffer_bytes * 2) / (1024.0 * 1024.0 * 1024.0); // Read + Write
    const double eff_gb_s = data_gb / (kernel_ms / 1000.0);
    const double budget_pct = (kernel_ms / 51.2) * 100.0;

    std::cout << "  Kernel Execution Time:  " << std::fixed << std::setprecision(3) << kernel_ms
              << " ms per frame (51.2 ms budget)\n";
    std::cout << "  Real-Time Budget Used:  " << std::fixed << std::setprecision(2) << budget_pct
              << "%\n";
    std::cout << "  Effective DRAM Bandwidth: " << std::fixed << std::setprecision(1) << eff_gb_s
              << " GB/s\n";
    std::cout << "  Real-Time Factor:       " << std::fixed << std::setprecision(1)
              << (51.2 / kernel_ms) << "x faster than real-time\n";

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_mat);
    cudaStreamDestroy(stream);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    return parity_pass;
}

} // namespace

int main() {
    std::cout << "==============================================================\n";
    std::cout << "  KOTEKAN BiSLC (Joint Matrix Unmixing) TEST SUITE            \n";
    std::cout << "==============================================================\n";

    bool pass1 = test_matrix_inversion();
    bool pass2 = test_coupling_matrix_generation();
    bool pass3 = test_sidelobe_cancellation();
    bool pass4 = test_gpu_kernel_benchmark();

    std::cout << "\n==============================================================\n";
    if (pass1 && pass2 && pass3 && pass4) {
        std::cout << "  OVERALL SUITE RESULT: ALL TESTS PASSED [SUCCESS]          \n";
        std::cout << "==============================================================\n";
        return 0;
    } else {
        std::cout << "  OVERALL SUITE RESULT: SOME TESTS FAILED [FAILURE]         \n";
        std::cout << "==============================================================\n";
        return 1;
    }
}
