/**
 * @file test_chained_mask_tracker_mvdr_suite.cpp
 * @brief End-to-End Chained Verification Suite: cudaAntennaMask -> cudaDirectBeamTracker -> MVDR/BiSLC
 *
 * Validates 5 mathematical and systems invariants against a full FP64 CPU reference chain:
 *  1. Format & Zero-Blanking Invariance (int4x2 vs int4x2_swapped_withoffset, 1/sqrt(N_alive), zero DC offset)
 *  2. Spatial Covariance & MVDR / BiSLC Conditioning (Hermitian symmetry, rank-deficiency resilience, Cholesky parity)
 *  3. Distortionless Target Constraint & Synthetic Dispersed FRB Pulse Recovery (DM=250 pc/cm^3 on +40 dB INR RFI)
 *  4. Spatial Null Depth & Sidelobe Suppression (75 dB to 90 dB attenuation, optimal SINR parity)
 *  5. Chained Real-Time Telemetry & Sustained Stress Loop (1,000-frame execution, <15% budget, 0 VRAM leaks)
 */

#include "cudaAntennaMask.hpp"
#include "cudaDirectBeamTracker.hpp"
#include "cudaBiSLC.hpp"
#include "chartsConstants.hpp"
#include "chartsMvdr.hpp"

#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <complex>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <cassert>

#define CHECK_CUDA(call)                                                                      \
    do {                                                                                      \
        cudaError_t err = call;                                                               \
        if (err != cudaSuccess) {                                                             \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ << " : "            \
                      << cudaGetErrorString(err) << " (" #call ")" << std::endl;              \
            std::exit(EXIT_FAILURE);                                                          \
        }                                                                                     \
    } while (0)

using cplx64 = std::complex<double>;

// Helper: Antenna coordinates on a regular grid
static std::vector<float3> generate_positions(std::size_t n_ant, float spacing_m) {
    std::vector<float3> pos(n_ant);
    const unsigned int grid_side = (n_ant <= 64) ? 8U : 16U;
    for (std::size_t i = 0; i < n_ant; ++i) {
        const float x = static_cast<float>(i % grid_side) * spacing_m;
        const float y = static_cast<float>(i / grid_side) * spacing_m;
        pos[i] = make_float3(x, y, 0.0f);
    }
    return pos;
}

// ============================================================================
// INVARIANT 1: Format & Zero-Blanking Invariance
// ============================================================================
bool test_format_and_zero_blanking_invariance() {
    std::cout << "\n======================================================\n";
    std::cout << "[Invariant 1/5] Format & Zero-Blanking Invariance\n";
    std::cout << "  (int4x2 vs int4x2_swapped_withoffset, 1/sqrt(N_alive), zero DC)\n";
    std::cout << "======================================================\n";

    const std::size_t n_ant = 64;
    const std::size_t n_time = 3840;
    const std::size_t n_freq = 16;
    const std::size_t total_spectra = n_time * n_freq;
    const float spacing_m = kotekan::charts::constants::charts_default_spacing_m;

    auto positions = generate_positions(n_ant, spacing_m);
    std::vector<double> freqs(n_freq);
    for (std::size_t f = 0; f < n_freq; ++f) {
        freqs[f] = 300.0e6 + f * 1.0e6;
    }

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    bool all_passed = true;

    for (bool is_swapped_withoffset : {false, true}) {
        std::string fmt_name = is_swapped_withoffset ? "int4x2_swapped_withoffset" : "int4x2 (twos_complement)";
        std::cout << "  Testing format: " << fmt_name << "...\n";

        const std::uint8_t expected_zero_byte = is_swapped_withoffset ? 0x88U : 0x00U;

        // Allocate host input buffer
        std::vector<kotekan::int4x2_t> h_voltages(total_spectra * n_ant);

        // Inject pure tone sky source at zenith: V_a = 3 + 0j
        // In twos complement: real = 3 (0x03), imag = 0 (0x00) -> 0x03
        // In swapped_withoffset: high nibble = real + 8 = 11 (0x0B), low nibble = imag + 8 = 8 (0x08) -> 0xB8
        const std::uint8_t signal_byte = is_swapped_withoffset ? 0xB8U : 0x03U;

        for (std::size_t s = 0; s < total_spectra; ++s) {
            for (std::size_t a = 0; a < n_ant; ++a) {
                h_voltages[s * n_ant + a].val = signal_byte;
            }
        }

        // Corrupt elements:
        // Antennas 0..3: Dead (zero variance, constant 0 -> 0x00 for 2s comp, 0x88 for offset)
        // Antennas 4..7: Rail-clipped (stuck at +7/+7 rail -> clip fraction = 1.0 > 0.15)
        std::vector<int> corrupted_antennas = {0, 1, 2, 3, 4, 5, 6, 7};
        const std::uint8_t dead_byte = expected_zero_byte;
        const std::uint8_t clipped_byte = is_swapped_withoffset ? 0xFFU : 0x77U;

        for (std::size_t s = 0; s < total_spectra; ++s) {
            for (int a = 0; a < 4; ++a) {
                h_voltages[s * n_ant + a].val = dead_byte;
            }
            for (int a = 4; a < 8; ++a) {
                h_voltages[s * n_ant + a].val = clipped_byte;
            }
        }

        // Device allocations
        kotekan::int4x2_t* d_voltages = nullptr;
        int* d_bad_antennas = nullptr;
        float* d_powers = nullptr;
        std::uint32_t* d_clips = nullptr;

        const std::size_t volt_bytes = total_spectra * n_ant * sizeof(kotekan::int4x2_t);
        CHECK_CUDA(cudaMalloc(&d_voltages, volt_bytes));
        CHECK_CUDA(cudaMalloc(&d_bad_antennas, corrupted_antennas.size() * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_powers, n_ant * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_clips, n_ant * sizeof(std::uint32_t)));

        CHECK_CUDA(cudaMemcpyAsync(d_voltages, h_voltages.data(), volt_bytes, cudaMemcpyHostToDevice, stream));
        CHECK_CUDA(cudaMemcpyAsync(d_bad_antennas, corrupted_antennas.data(),
                                   corrupted_antennas.size() * sizeof(int), cudaMemcpyHostToDevice, stream));

        // 1. Inspect moments
        kotekan::launch_inspect_antenna_moments(
            d_voltages, nullptr, d_powers, d_clips, n_time, n_freq, n_ant, 1, stream, is_swapped_withoffset);
        CHECK_CUDA(cudaStreamSynchronize(stream));

        std::vector<float> h_powers(n_ant);
        std::vector<std::uint32_t> h_clips(n_ant);
        CHECK_CUDA(cudaMemcpy(h_powers.data(), d_powers, n_ant * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_clips.data(), d_clips, n_ant * sizeof(std::uint32_t), cudaMemcpyDeviceToHost));

        for (int a = 0; a < 4; ++a) {
            if (h_powers[a] > 1e-4f) {
                std::cerr << "  FAILED: Dead antenna " << a << " power not zero: " << h_powers[a] << "\n";
                all_passed = false;
            }
        }
        for (int a = 4; a < 8; ++a) {
            const float clip_frac = static_cast<float>(h_clips[a]) / static_cast<float>(total_spectra);
            if (clip_frac < 0.15f) {
                std::cerr << "  FAILED: Clipped antenna " << a << " clip fraction not > 0.15: " << clip_frac << "\n";
                all_passed = false;
            }
        }

        // 2. Blank corrupted antennas in-place
        kotekan::launch_zero_bad_antennas(
            d_voltages, d_bad_antennas, static_cast<int>(corrupted_antennas.size()),
            total_spectra, n_ant, stream, is_swapped_withoffset);
        CHECK_CUDA(cudaStreamSynchronize(stream));

        // 3. Verify blanked memory bytes and zero DC offset
        CHECK_CUDA(cudaMemcpy(h_voltages.data(), d_voltages, volt_bytes, cudaMemcpyDeviceToHost));

        double sum_masked_real = 0.0;
        double sum_masked_imag = 0.0;
        std::size_t masked_samples = 0;

        for (std::size_t s = 0; s < total_spectra; ++s) {
            for (int bad_a : corrupted_antennas) {
                const std::uint8_t byte_val = h_voltages[s * n_ant + bad_a].val;
                if (byte_val != expected_zero_byte) {
                    std::cerr << "  FAILED: Masked byte mismatch: expected 0x" << std::hex
                              << static_cast<int>(expected_zero_byte) << ", got 0x"
                              << static_cast<int>(byte_val) << std::dec << " at ant " << bad_a << "\n";
                    all_passed = false;
                    break;
                }

                // Unpack byte according to format
                double r = 0.0, im = 0.0;
                if (is_swapped_withoffset) {
                    r = static_cast<double>((byte_val >> 4) & 0x0F) - 8.0;
                    im = static_cast<double>(byte_val & 0x0F) - 8.0;
                } else {
                    int8_t ir = static_cast<int8_t>((byte_val & 0x0F) << 4) >> 4;
                    int8_t ii = static_cast<int8_t>(byte_val & 0xF0) >> 4;
                    r = static_cast<double>(ir);
                    im = static_cast<double>(ii);
                }
                sum_masked_real += r;
                sum_masked_imag += im;
                masked_samples++;
            }
        }

        const double mean_dc_real = std::abs(sum_masked_real / std::max(std::size_t(1), masked_samples));
        const double mean_dc_imag = std::abs(sum_masked_imag / std::max(std::size_t(1), masked_samples));
        std::cout << "    Masked Antenna DC Offset: |E[Re]| = " << std::scientific << mean_dc_real
                  << ", |E[Im]| = " << mean_dc_imag << " (Threshold: < 1.0e-7)\n";

        if (mean_dc_real > 1.0e-7 || mean_dc_imag > 1.0e-7) {
            std::cerr << "  FAILED: Artificial DC offset detected in masked voltages!\n";
            all_passed = false;
        }

        // 4. Formed Beam Amplitude Invariance under dropout
        // N_alive = 64 - 8 = 56 antennas.
        const std::size_t n_alive = n_ant - corrupted_antennas.size();
        kotekan::DirectDirection3D dir_zenith{0.0f, 0.0f, 1.0f};

        std::vector<std::uint8_t> h_mask(n_ant, 1);
        for (int bad_a : corrupted_antennas) {
            h_mask[bad_a] = 0;
        }

        std::vector<double> h_wavenumbers(n_freq);
        for (std::size_t f = 0; f < n_freq; ++f) {
            h_wavenumbers[f] = 2.0 * M_PI * freqs[f] / kotekan::charts::constants::speed_of_light_m_per_s;
        }

        float2* d_weights = nullptr;
        float2* d_formed = nullptr;
        kotekan::DirectDirection3D* d_dirs = nullptr;
        double* d_wavenumbers = nullptr;
        float3* d_positions = nullptr;
        std::uint8_t* d_mask = nullptr;

        CHECK_CUDA(cudaMalloc(&d_weights, 1 * n_freq * n_ant * sizeof(float2)));
        CHECK_CUDA(cudaMalloc(&d_formed, total_spectra * 1 * sizeof(float2)));
        CHECK_CUDA(cudaMalloc(&d_dirs, 1 * sizeof(kotekan::DirectDirection3D)));
        CHECK_CUDA(cudaMalloc(&d_wavenumbers, n_freq * sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_positions, n_ant * sizeof(float3)));
        CHECK_CUDA(cudaMalloc(&d_mask, n_ant * sizeof(std::uint8_t)));

        CHECK_CUDA(cudaMemcpy(d_dirs, &dir_zenith, sizeof(kotekan::DirectDirection3D), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_wavenumbers, h_wavenumbers.data(), n_freq * sizeof(double), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_positions, positions.data(), n_ant * sizeof(float3), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_mask, h_mask.data(), n_ant * sizeof(std::uint8_t), cudaMemcpyHostToDevice));

        // Generate normalized steering weights (norm = 1/sqrt(N_alive))
        kotekan::launch_generate_steering_weights(
            d_weights, nullptr, d_dirs, nullptr, d_wavenumbers, d_positions, d_mask, nullptr,
            1, n_freq, n_ant, n_time, n_alive, stream, 1);

        // Run Direct Beamformer with format specified
        const kotekan::DirectInputFormat fmt = is_swapped_withoffset
            ? kotekan::DirectInputFormat::INT4X2_SWAPPED_WITHOFFSET
            : kotekan::DirectInputFormat::INT4X2;

        kotekan::launch_direct_beamformer(
            d_voltages, d_weights, d_formed, n_time, n_freq, n_ant, 1, 1, 256, 4, 4, stream,
            1, kotekan::DirectPolLayout::INTERLEAVED, fmt);
        CHECK_CUDA(cudaStreamSynchronize(stream));

        std::vector<float2> h_formed(total_spectra);
        CHECK_CUDA(cudaMemcpy(h_formed.data(), d_formed, total_spectra * sizeof(float2), cudaMemcpyDeviceToHost));

        // Expected formed beam voltage: E = sum_{a in alive} (1/sqrt(N_alive)) * V_signal
        // V_signal = 3.0. E = N_alive * (3.0 / sqrt(N_alive)) = 3.0 * sqrt(N_alive).
        // Expected formed power: |E|^2 = 9.0 * N_alive.
        const double expected_power = 9.0 * static_cast<double>(n_alive);
        double meas_power = 0.0;
        for (const auto& v : h_formed) {
            meas_power += static_cast<double>(v.x * v.x + v.y * v.y);
        }
        meas_power /= total_spectra;

        const double pwr_ratio_err = std::abs((meas_power / expected_power) - 1.0);
        std::cout << "    Formed Beam Power: Meas = " << std::fixed << std::setprecision(4) << meas_power
                  << ", Exp = " << expected_power << ", Ratio Error = " << std::scientific
                  << std::setprecision(2) << pwr_ratio_err << "\n";

        if (pwr_ratio_err > 1.0e-5) {
            std::cerr << "  FAILED: Formed beam power ratio error exceeds 1.0e-5!\n";
            all_passed = false;
        }

        cudaFree(d_voltages);
        cudaFree(d_bad_antennas);
        cudaFree(d_powers);
        cudaFree(d_clips);
        cudaFree(d_weights);
        cudaFree(d_formed);
        cudaFree(d_dirs);
        cudaFree(d_wavenumbers);
        cudaFree(d_positions);
        cudaFree(d_mask);
    }

    cudaStreamDestroy(stream);
    std::cout << "  -> Result: " << (all_passed ? "PASSED [OK]" : "FAILED [X]") << "\n";
    return all_passed;
}

// ============================================================================
// INVARIANT 2: Spatial Covariance & MVDR / BiSLC Conditioning
// ============================================================================
bool test_spatial_covariance_and_mvdr_conditioning() {
    std::cout << "\n======================================================\n";
    std::cout << "[Invariant 2/5] Spatial Covariance & MVDR / BiSLC Conditioning\n";
    std::cout << "  (Hermitian symmetry, rank-deficiency resilience, Cholesky parity)\n";
    std::cout << "======================================================\n";

    const std::size_t num_beams = 4;
    const std::size_t n_freq = 4;
    const std::size_t n_ant = 64;
    const float spacing_m = 0.6f;

    auto pos = generate_positions(n_ant, spacing_m);
    std::vector<double> freqs = {300.0e6, 350.0e6, 400.0e6, 450.0e6};

    // 4 Beams: Beam 0 zenith, Beam 1 at 3 deg, Beam 2 at 6 deg, Beam 3 at 9 deg
    kotekan::DirectBeamTarget targets[4];
    targets[0].direction = kotekan::DirectDirection3D{0.0f, 0.0f, 1.0f};
    const float l1 = std::sin(3.0f * M_PI / 180.0f);
    targets[1].direction = kotekan::DirectDirection3D{l1, 0.0f, std::sqrt(1.0f - l1 * l1)};
    const float l2 = std::sin(6.0f * M_PI / 180.0f);
    targets[2].direction = kotekan::DirectDirection3D{l2, 0.0f, std::sqrt(1.0f - l2 * l2)};
    const float l3 = std::sin(9.0f * M_PI / 180.0f);
    targets[3].direction = kotekan::DirectDirection3D{l3, 0.0f, std::sqrt(1.0f - l3 * l3)};

    // Mask 25% of antennas (16 out of 64 elements)
    std::vector<std::uint8_t> mask(n_ant, 1);
    for (std::size_t a = 0; a < 16; ++a) {
        mask[a] = 0;
    }
    const std::size_t n_active = 48;

    std::vector<float2> M(n_freq * num_beams * num_beams);
    std::vector<float2> A(n_freq * num_beams * num_beams);
    const float diag_loading = 1.0e-3f;

    kotekan::compute_beam_coupling_and_inverses(
        M.data(), A.data(), targets, freqs, pos.data(), mask.data(), num_beams, n_freq, n_ant, n_active, diag_loading);

    bool passed = true;

    // Solve exact FP64 Cholesky reference solution for w = R^-1 a / (a^H R^-1 a)
    for (std::size_t f = 0; f < n_freq; ++f) {
        const std::size_t base = f * num_beams * num_beams;

        // 1. Check Hermitian symmetry of A(f): ||A - A^H||_F / ||A||_F <= 1e-7
        double frob_diff_sq = 0.0;
        double frob_a_sq = 0.0;
        for (std::size_t i = 0; i < num_beams; ++i) {
            for (std::size_t j = 0; j < num_beams; ++j) {
                const float2 a_ij = A[base + i * num_beams + j];
                const float2 a_ji = A[base + j * num_beams + i];
                const double diff_r = static_cast<double>(a_ij.x) - static_cast<double>(a_ji.x);
                const double diff_i = static_cast<double>(a_ij.y) + static_cast<double>(a_ji.y); // Conj
                frob_diff_sq += diff_r * diff_r + diff_i * diff_i;
                frob_a_sq += static_cast<double>(a_ij.x * a_ij.x + a_ij.y * a_ij.y);
            }
        }
        const double hermitian_rel_err = std::sqrt(frob_diff_sq / std::max(1e-12, frob_a_sq));
        std::cout << "  Freq " << freqs[f] / 1e6 << " MHz: Hermitian Symmetry Rel Error = "
                  << std::scientific << hermitian_rel_err;

        if (hermitian_rel_err > 1.0e-7) {
            std::cout << " [FAIL]\n";
            passed = false;
        } else {
            std::cout << " [PASS]\n";
        }

        // 2. Rank-Deficiency Resilience: Verify Condition Number kappa(A + delta I) <= 10^4
        // With delta = 1e-3 * tr(A)/B and tr(A)=B, lambda_min >= delta = 1e-3, lambda_max <= B
        // kappa <= B / (1e-3) = 4000 <= 10000.
        double tr_A = 0.0;
        for (std::size_t i = 0; i < num_beams; ++i) tr_A += static_cast<double>(A[base + i * num_beams + i].x);
        const double kappa_bound = tr_A / static_cast<double>(diag_loading);
        std::cout << "    Condition Number Bound kappa <= " << std::fixed << std::setprecision(1) << kappa_bound;
        if (kappa_bound <= 10000.0) {
            std::cout << " [PASS] (<= 1e4)\n";
        } else {
            std::cout << " [FAIL]\n";
            passed = false;
        }

        // 3. FP64 Cholesky Solver Parity for MVDR Steering Vector: ||w_GPU - w_FP64|| / ||w_FP64|| <= 1e-5
        // Form FP64 matrix R_fp64 = A + delta * I
        cplx64 R_fp64[num_beams][num_beams];
        for (std::size_t i = 0; i < num_beams; ++i) {
            for (std::size_t j = 0; j < num_beams; ++j) {
                const float2 a_ij = A[base + i * num_beams + j];
                R_fp64[i][j] = cplx64(a_ij.x, a_ij.y);
                if (i == j) R_fp64[i][j] += static_cast<double>(diag_loading);
            }
        }

        // FP64 Cholesky Factorization R = L * L^H
        cplx64 L[num_beams][num_beams];
        for (std::size_t i = 0; i < num_beams; ++i) {
            for (std::size_t j = 0; j <= i; ++j) {
                cplx64 sum = 0.0;
                for (std::size_t k = 0; k < j; ++k) sum += L[i][k] * std::conj(L[j][k]);
                if (i == j) {
                    L[i][j] = std::sqrt(std::max(1e-12, (R_fp64[i][i] - sum).real()));
                } else {
                    L[i][j] = (R_fp64[i][j] - sum) / L[j][j];
                }
            }
        }

        // Solve R * z = a_0, where a_0 = [1, 0, 0, 0]^T (first column of inverse)
        // Forward: L * y = a_0
        cplx64 y[num_beams];
        for (std::size_t i = 0; i < num_beams; ++i) {
            cplx64 sum = 0.0;
            for (std::size_t k = 0; k < i; ++k) sum += L[i][k] * y[k];
            cplx64 rhs = (i == 0) ? cplx64(1.0, 0.0) : cplx64(0.0, 0.0);
            y[i] = (rhs - sum) / L[i][i];
        }
        // Backward: L^H * z = y
        cplx64 z_fp64[num_beams];
        for (int i = static_cast<int>(num_beams) - 1; i >= 0; --i) {
            cplx64 sum = 0.0;
            for (std::size_t k = i + 1; k < num_beams; ++k) sum += std::conj(L[k][i]) * z_fp64[k];
            z_fp64[i] = (y[i] - sum) / L[i][i];
        }

        // Compare column 0 of M (computed on GPU/host in BiSLC) with Cholesky z_fp64
        double norm_z_fp64_sq = 0.0;
        double diff_z_sq = 0.0;
        for (std::size_t i = 0; i < num_beams; ++i) {
            norm_z_fp64_sq += std::norm(z_fp64[i]);

            // Column 0 of M is M[i * num_beams + 0]
            const float2 m_i0 = M[base + i * num_beams + 0];
            cplx64 z_gpu = cplx64(m_i0.x, m_i0.y);
            diff_z_sq += std::norm(z_gpu - z_fp64[i]);
        }
        const double w_rel_err = std::sqrt(diff_z_sq / std::max(1e-12, norm_z_fp64_sq));
        std::cout << "    MVDR Cholesky Solver Parity Error = " << std::scientific << w_rel_err;
        if (w_rel_err <= 1.0e-5) {
            std::cout << " [PASS] (<= 1e-5)\n";
        } else {
            std::cout << " [FAIL]\n";
            passed = false;
        }
    }

    std::cout << "  -> Result: " << (passed ? "PASSED [OK]" : "FAILED [X]") << "\n";
    return passed;
}

// ============================================================================
// INVARIANT 3: Distortionless Target Constraint & Synthetic Dispersed FRB
// ============================================================================
bool test_distortionless_constraint_and_frb_recovery() {
    std::cout << "\n======================================================\n";
    std::cout << "[Invariant 3/5] Distortionless Target Constraint & Dispersed FRB Recovery\n";
    std::cout << "  (|w^H a - 1| <= 1e-5, DM=250 pc/cm^3 on +40 dB INR RFI, SNR loss <= 0.05 dB)\n";
    std::cout << "======================================================\n";

    const std::size_t num_beams = 2; // Beam 0 = Target (FRB), Beam 1 = Reference (RFI)
    const std::size_t n_freq = 32;
    const std::size_t n_time = 15360; // 51.2 ms frame
    const double dt_s = 51.2e-3 / 15360.0; // 3.3333 us sampling

    // L-band subband channels (1.400 GHz to 1.431 GHz) where DM=250 sweep is ~23 ms (fits in 51.2 ms)
    std::vector<double> freqs(n_freq);
    const double f_low = 1.400e9;
    const double df = 1.0e6;
    for (std::size_t f = 0; f < n_freq; ++f) {
        freqs[f] = f_low + f * df;
    }

    // 1. Distortionless Constraint Check: w_0^H a_0 = 1.0
    const float alpha_r = 0.15f, alpha_i = -0.10f;
    const float alpha_sq = alpha_r * alpha_r + alpha_i * alpha_i;

    float2 A[4] = {make_float2(1.0f, 0.0f), make_float2(alpha_r, alpha_i),
                   make_float2(alpha_r, -alpha_i), make_float2(1.0f, 0.0f)};
    float2 M[4];
    // Invert exact A without diagonal loading to verify the pure algebraic distortionless property w^H a = 1
    kotekan::invert_complex_matrix(A, M, 2, 0.0f);

    cplx64 w00(M[0].x, -M[0].y); // w^H
    cplx64 w01(M[1].x, -M[1].y);
    cplx64 a00(1.0, 0.0);
    cplx64 a01(alpha_r, alpha_i);

    cplx64 target_response = w00 * a00 + w01 * a01;
    const double distortion_err = std::abs(target_response - 1.0);
    std::cout << "  Distortionless Target Response |w^H a - 1| = " << std::scientific
              << distortion_err << " (Threshold <= 1.0e-5)\n";

    bool passed = (distortion_err <= 1.0e-5);

    // 2. Inject Dispersed FRB pulse (DM = 250 pc cm^-3)
    const double DM = 250.0;
    const double f_ref_ghz = freqs[n_freq - 1] / 1.0e9; // 1.431 GHz (highest freq arrives first)
    std::vector<int> frb_sample_peak(n_freq);

    for (std::size_t f = 0; f < n_freq; ++f) {
        const double f_ghz = freqs[f] / 1.0e9;
        const double delay_s = 4.148808e-3 * DM * (1.0 / (f_ghz * f_ghz) - 1.0 / (f_ref_ghz * f_ref_ghz));
        const double t_arrival = 8.0e-3 + delay_s; // Start at 8.0 ms, extends to ~31.0 ms
        frb_sample_peak[f] = static_cast<int>(std::round(t_arrival / dt_s));
    }

    std::vector<float2> h_formed(n_time * n_freq * num_beams);
    std::vector<float2> h_unmixed(n_time * n_freq * num_beams);

    const float frb_amplitude = 10.0f;
    const float rfi_amplitude = 1000.0f; // +40 dB power relative to FRB
    const double pulse_sigma_samples = (0.15e-3) / dt_s;

    for (std::size_t t = 0; t < n_time; ++t) {
        for (std::size_t f = 0; f < n_freq; ++f) {
            const std::size_t idx = (t * n_freq + f) * num_beams;

            // RFI tone
            const float rfi_phase = static_cast<float>(2.0 * M_PI * 0.05 * t);
            const float2 v_rfi = make_float2(rfi_amplitude * std::cos(rfi_phase), rfi_amplitude * std::sin(rfi_phase));

            // FRB pulse
            const double dt_pulse = static_cast<double>(t) - static_cast<double>(frb_sample_peak[f]);
            const float frb_env = frb_amplitude * std::exp(-0.5 * (dt_pulse * dt_pulse) / (pulse_sigma_samples * pulse_sigma_samples));
            const float2 v_frb = make_float2(frb_env, 0.0f);

            // Beam 0 receives FRB + coupled RFI
            const float b0_r = v_frb.x + (alpha_r * v_rfi.x - alpha_i * v_rfi.y);
            const float b0_i = v_frb.y + (alpha_r * v_rfi.y + alpha_i * v_rfi.x);
            h_formed[idx + 0] = make_float2(b0_r, b0_i);

            // Beam 1 receives direct RFI + coupled FRB
            const float b1_r = v_rfi.x + (alpha_r * v_frb.x - (-alpha_i) * v_frb.y);
            const float b1_i = v_rfi.y + (alpha_r * v_frb.y + (-alpha_i) * v_frb.x);
            h_formed[idx + 1] = make_float2(b1_r, b1_i);
        }
    }

    // Run BiSLC unmixing on GPU
    float2* d_in = nullptr;
    float2* d_out = nullptr;
    float2* d_mat = nullptr;
    const std::size_t buf_bytes = h_formed.size() * sizeof(float2);

    std::vector<float2> h_mat_all(n_freq * 4);
    for (std::size_t f = 0; f < n_freq; ++f) {
        std::copy(M, M + 4, h_mat_all.data() + f * 4);
    }

    CHECK_CUDA(cudaMalloc(&d_in, buf_bytes));
    CHECK_CUDA(cudaMalloc(&d_out, buf_bytes));
    CHECK_CUDA(cudaMalloc(&d_mat, h_mat_all.size() * sizeof(float2)));

    CHECK_CUDA(cudaMemcpy(d_in, h_formed.data(), buf_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_mat, h_mat_all.data(), h_mat_all.size() * sizeof(float2), cudaMemcpyHostToDevice));

    kotekan::launch_bislc_unmixing(d_out, d_in, d_mat, n_time, n_freq, num_beams, num_beams, nullptr);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_unmixed.data(), d_out, buf_bytes, cudaMemcpyDeviceToHost));

    // Dedisperse and measure recovered peak profile
    std::vector<double> profile_unmixed(n_time, 0.0);
    std::vector<double> profile_pure_frb(n_time, 0.0);

    for (std::size_t f = 0; f < n_freq; ++f) {
        const int shift = frb_sample_peak[f] - frb_sample_peak[n_freq - 1];
        for (std::size_t t = 0; t < n_time; ++t) {
            const int t_src = static_cast<int>(t) + shift;
            if (t_src >= 0 && t_src < static_cast<int>(n_time)) {
                const float2 v_clean = h_unmixed[(t_src * n_freq + f) * num_beams + 0];
                profile_unmixed[t] += static_cast<double>(v_clean.x * v_clean.x + v_clean.y * v_clean.y);

                const double dt_p = static_cast<double>(t_src) - static_cast<double>(frb_sample_peak[f]);
                const double pure_env = frb_amplitude * std::exp(-0.5 * (dt_p * dt_p) / (pulse_sigma_samples * pulse_sigma_samples));
                profile_pure_frb[t] += pure_env * pure_env;
            }
        }
    }

    // Find peak locations and amplitudes
    auto max_it_clean = std::max_element(profile_unmixed.begin(), profile_unmixed.end());
    auto max_it_pure = std::max_element(profile_pure_frb.begin(), profile_pure_frb.end());

    const std::size_t peak_idx_clean = std::distance(profile_unmixed.begin(), max_it_clean);
    const std::size_t peak_idx_pure = std::distance(profile_pure_frb.begin(), max_it_pure);

    const int delay_discrepancy = std::abs(static_cast<int>(peak_idx_clean) - static_cast<int>(peak_idx_pure));
    const double snr_loss_db = std::abs(10.0 * std::log10(*max_it_clean / *max_it_pure));

    std::cout << "  Dispersed FRB Recovery:\n";
    std::cout << "    Pure Peak Power:     " << *max_it_pure << " at sample " << peak_idx_pure << "\n";
    std::cout << "    Recovered Peak Power: " << *max_it_clean << " at sample " << peak_idx_clean << "\n";
    std::cout << "    SNR Loss:            " << std::fixed << std::setprecision(4) << snr_loss_db << " dB (Threshold <= 0.05 dB)\n";
    std::cout << "    Delay Discrepancy:   " << delay_discrepancy << " samples (Threshold = 0)\n";

    if (snr_loss_db > 0.05) {
        std::cerr << "  FAILED: FRB pulse SNR loss exceeds 0.05 dB!\n";
        passed = false;
    }
    if (delay_discrepancy != 0) {
        std::cerr << "  FAILED: FRB pulse delay discrepancy is not 0 samples!\n";
        passed = false;
    }

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_mat);

    std::cout << "  -> Result: " << (passed ? "PASSED [OK]" : "FAILED [X]") << "\n";
    return passed;
}

// ============================================================================
// INVARIANT 4: Spatial Null Depth & Sidelobe Suppression
// ============================================================================
bool test_spatial_null_depth_and_suppression() {
    std::cout << "\n======================================================\n";
    std::cout << "[Invariant 4/5] Spatial Null Depth & Sidelobe Suppression\n";
    std::cout << "  (Demonstrating 75 dB to 90 dB suppression across 64, 128, 256 antennas)\n";
    std::cout << "======================================================\n";

    const std::vector<std::size_t> ant_scales = {64, 128, 256};
    const std::size_t num_beams = 2;
    const std::size_t n_freq = 4;
    const float spacing_m = 0.6f;

    std::vector<double> freqs = {300.0e6, 350.0e6, 400.0e6, 450.0e6};

    // Target beam: Zenith.
    // Reference/RFI beam: Directional sidelobe offset l1 = 0.25 (outside main lobe).
    kotekan::DirectBeamTarget targets[2];
    targets[0].direction = kotekan::DirectDirection3D{0.0f, 0.0f, 1.0f};
    const float l1 = 0.25f;
    targets[1].direction = kotekan::DirectDirection3D{l1, 0.0f, std::sqrt(1.0f - l1 * l1)};

    bool all_passed = true;

    for (std::size_t n_ant : ant_scales) {
        auto pos = generate_positions(n_ant, spacing_m);
        std::vector<std::uint8_t> mask(n_ant, 1);

        std::vector<float2> M(n_freq * 4);
        std::vector<float2> A(n_freq * 4);

        kotekan::compute_beam_coupling_and_inverses(
            M.data(), A.data(), targets, freqs, pos.data(), mask.data(), num_beams, n_freq, n_ant, n_ant, 1.0e-5f);

        const float alpha_r = A[1].x;
        const float alpha_i = A[1].y;
        const double coupling_pwr = static_cast<double>(alpha_r * alpha_r + alpha_i * alpha_i);

        const double p_rfi = 10000.0; // +40 dB INR
        const double p_target = 1.0;  // 0 dB

        cplx64 b0 = cplx64(1.0, 0.0) + cplx64(alpha_r, alpha_i) * std::sqrt(p_rfi);
        cplx64 b1 = std::sqrt(p_rfi) + cplx64(alpha_r, -alpha_i) * cplx64(1.0, 0.0);

        cplx64 m00(M[0].x, M[0].y);
        cplx64 m01(M[1].x, M[1].y);
        cplx64 b0_clean = m00 * b0 + m01 * b1;

        const double residual_error_before = std::abs(b0 - cplx64(1.0, 0.0));
        const double residual_error_after = std::abs(b0_clean - cplx64(1.0, 0.0));

        const double suppression_db =
            10.0 * std::log10(std::max(1e-12, (residual_error_before * residual_error_before) /
                                              std::max(1e-14, residual_error_after * residual_error_after)));

        std::cout << "  N_ANT = " << std::setw(3) << n_ant
                  << " | Coupling = " << std::fixed << std::setprecision(4) << coupling_pwr
                  << " (" << 10.0 * std::log10(std::max(1e-12, coupling_pwr)) << " dB)"
                  << " | Suppression = " << std::setprecision(1) << suppression_db << " dB";

        // Requirement: 75 dB to 90 dB attenuation
        if (suppression_db < 75.0) {
            std::cout << " [FAIL] (< 75 dB)\n";
            all_passed = false;
        } else {
            std::cout << " [PASS]\n";
        }
    }

    std::cout << "  -> Result: " << (all_passed ? "PASSED [OK]" : "FAILED [X]") << "\n";
    return all_passed;
}

// ============================================================================
// INVARIANT 5: Chained Real-Time Telemetry & Sustained Stress Loop
// ============================================================================
bool test_chained_realtime_stress_loop() {
    std::cout << "\n======================================================\n";
    std::cout << "[Invariant 5/5] Chained Real-Time Telemetry & Sustained Stress Loop\n";
    std::cout << "  (1,000 iterations, <15% of 51.2 ms real-time budget, 0 VRAM leaks)\n";
    std::cout << "======================================================\n";

    const std::size_t n_ant = 128;
    const std::size_t num_beams = 4;
    const std::size_t n_freq = 336;
    const std::size_t n_time = 15360; // 51.2 ms frame
    const std::size_t total_spectra = n_time * n_freq;
    const float spacing_m = 0.6f;

    auto pos = generate_positions(n_ant, spacing_m);
    std::vector<double> freqs(n_freq);
    for (std::size_t f = 0; f < n_freq; ++f) {
        freqs[f] = 300.0e6 + f * 300.0e3;
    }

    kotekan::DirectBeamTarget targets[4];
    for (std::size_t b = 0; b < num_beams; ++b) {
        const float lb = static_cast<float>(b) * 0.05f;
        targets[b].direction = kotekan::DirectDirection3D{lb, 0.0f, std::sqrt(1.0f - lb * lb)};
    }

    std::size_t vram_free_initial = 0, vram_total = 0;
    CHECK_CUDA(cudaMemGetInfo(&vram_free_initial, &vram_total));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // Device memory setup
    kotekan::int4x2_t* d_voltages = nullptr;
    int* d_bad_antennas = nullptr;
    float2* d_weights = nullptr;
    float2* d_formed = nullptr;
    float2* d_unmixed = nullptr;
    float2* d_matrices = nullptr;
    kotekan::DirectDirection3D* d_dirs = nullptr;
    double* d_wavenumbers = nullptr;
    float3* d_positions = nullptr;
    std::uint8_t* d_mask = nullptr;

    const std::size_t raw_bytes = total_spectra * n_ant * sizeof(kotekan::int4x2_t);
    const std::size_t formed_bytes = total_spectra * num_beams * sizeof(float2);
    const std::size_t weights_bytes = num_beams * n_freq * n_ant * sizeof(float2);
    const std::size_t matrix_bytes = n_freq * num_beams * num_beams * sizeof(float2);

    CHECK_CUDA(cudaMalloc(&d_voltages, raw_bytes));
    CHECK_CUDA(cudaMalloc(&d_bad_antennas, 16 * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_weights, weights_bytes));
    CHECK_CUDA(cudaMalloc(&d_formed, formed_bytes));
    CHECK_CUDA(cudaMalloc(&d_unmixed, formed_bytes));
    CHECK_CUDA(cudaMalloc(&d_matrices, matrix_bytes));
    CHECK_CUDA(cudaMalloc(&d_dirs, num_beams * sizeof(kotekan::DirectDirection3D)));
    CHECK_CUDA(cudaMalloc(&d_wavenumbers, n_freq * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_positions, n_ant * sizeof(float3)));
    CHECK_CUDA(cudaMalloc(&d_mask, n_ant * sizeof(std::uint8_t)));

    CHECK_CUDA(cudaMemsetAsync(d_voltages, 0x88, raw_bytes, stream));
    CHECK_CUDA(cudaMemsetAsync(d_mask, 1, n_ant * sizeof(std::uint8_t), stream));

    std::vector<double> h_wavenumbers(n_freq);
    for (std::size_t f = 0; f < n_freq; ++f) {
        h_wavenumbers[f] = 2.0 * M_PI * freqs[f] / kotekan::charts::constants::speed_of_light_m_per_s;
    }
    CHECK_CUDA(cudaMemcpyAsync(d_wavenumbers, h_wavenumbers.data(), n_freq * sizeof(double), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(d_positions, pos.data(), n_ant * sizeof(float3), cudaMemcpyHostToDevice, stream));

    std::vector<kotekan::DirectDirection3D> h_dirs(num_beams);
    for (std::size_t b = 0; b < num_beams; ++b) h_dirs[b] = targets[b].direction;
    CHECK_CUDA(cudaMemcpyAsync(d_dirs, h_dirs.data(), num_beams * sizeof(kotekan::DirectDirection3D), cudaMemcpyHostToDevice, stream));

    // Warm-up
    kotekan::launch_generate_steering_weights(
        d_weights, nullptr, d_dirs, nullptr, d_wavenumbers, d_positions, d_mask, nullptr,
        num_beams, n_freq, n_ant, n_time, n_ant, stream, 1);
    kotekan::launch_direct_beamformer(
        d_voltages, d_weights, d_formed, n_time, n_freq, n_ant, num_beams, num_beams, 256, 4, 4, stream,
        1, kotekan::DirectPolLayout::INTERLEAVED, kotekan::DirectInputFormat::INT4X2_SWAPPED_WITHOFFSET);
    kotekan::launch_compute_bislc_matrices_gpu(
        d_matrices, nullptr, d_dirs, d_wavenumbers, d_positions, d_mask, num_beams, n_freq, n_ant, 1.0e-4f, stream);
    kotekan::launch_bislc_unmixing(
        d_unmixed, d_formed, d_matrices, n_time, n_freq, num_beams, num_beams, stream);
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // 1,000 sustained iterations
    constexpr int ITERS = 1000;
    cudaEvent_t ev_start, ev_stop;
    CHECK_CUDA(cudaEventCreate(&ev_start));
    CHECK_CUDA(cudaEventCreate(&ev_stop));

    std::cout << "  Executing " << ITERS << " sustained chained frames (128 Ant, 4 Beams, 336 Freq, 15,360 Time)...\n";
    auto t0_wall = std::chrono::high_resolution_clock::now();

    CHECK_CUDA(cudaEventRecord(ev_start, stream));
    for (int it = 0; it < ITERS; ++it) {
        // Step 1: Blank 4 bad antennas
        kotekan::launch_zero_bad_antennas(d_voltages, d_bad_antennas, 4, total_spectra, n_ant, stream, true);

        // Step 2: Direct Beam Tracker
        kotekan::launch_direct_beamformer(
            d_voltages, d_weights, d_formed, n_time, n_freq, n_ant, num_beams, num_beams, 256, 4, 4, stream,
            1, kotekan::DirectPolLayout::INTERLEAVED, kotekan::DirectInputFormat::INT4X2_SWAPPED_WITHOFFSET);

        // Step 3: BiSLC Unmixing
        kotekan::launch_bislc_unmixing(
            d_unmixed, d_formed, d_matrices, n_time, n_freq, num_beams, num_beams, stream);
    }
    CHECK_CUDA(cudaEventRecord(ev_stop, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    auto t1_wall = std::chrono::high_resolution_clock::now();
    float total_gpu_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&total_gpu_ms, ev_start, ev_stop));

    const float per_frame_ms = total_gpu_ms / static_cast<float>(ITERS);
    const double wall_sec = std::chrono::duration<double>(t1_wall - t0_wall).count();
    const double budget_pct = (per_frame_ms / 51.2f) * 100.0f;

    std::cout << "  Telemetry & Timing Summary:\n";
    std::cout << "    Total GPU Kernel Time: " << total_gpu_ms << " ms (" << ITERS << " frames in "
              << std::fixed << std::setprecision(2) << wall_sec << " s wall-clock)\n";
    std::cout << "    Per-Frame GPU Latency: " << std::fixed << std::setprecision(3) << per_frame_ms
              << " ms (Budget: 51.2 ms)\n";
    std::cout << "    Real-Time Budget Used: " << std::fixed << std::setprecision(2) << budget_pct
              << "% (Requirement: < 15.0%)\n";

    bool passed = (budget_pct < 15.0f);

    // VRAM Leak Check
    cudaFree(d_voltages);
    cudaFree(d_bad_antennas);
    cudaFree(d_weights);
    cudaFree(d_formed);
    cudaFree(d_unmixed);
    cudaFree(d_matrices);
    cudaFree(d_dirs);
    cudaFree(d_wavenumbers);
    cudaFree(d_positions);
    cudaFree(d_mask);
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    cudaStreamDestroy(stream);

    std::size_t vram_free_final = 0;
    CHECK_CUDA(cudaMemGetInfo(&vram_free_final, &vram_total));
    const long long leak_bytes = static_cast<long long>(vram_free_initial) - static_cast<long long>(vram_free_final);
    std::cout << "    VRAM Leak Check: Initial Free = " << (vram_free_initial / (1024 * 1024))
              << " MB, Final Free = " << (vram_free_final / (1024 * 1024))
              << " MB (Leaked = " << leak_bytes << " bytes)\n";

    if (leak_bytes > 0) {
        std::cerr << "  FAILED: VRAM leak detected!\n";
        passed = false;
    }

    std::cout << "  -> Result: " << (passed ? "PASSED [OK]" : "FAILED [X]") << "\n";
    return passed;
}

// ============================================================================
// MAIN ENTRY
// ============================================================================
int main() {
    std::cout << "================================================================================\n";
    std::cout << "  CHARTS KOTEKAN CHAINED GPU PIPELINE TEST SUITE\n";
    std::cout << "  cudaAntennaMask -> cudaDirectBeamTracker -> MVDR / BiSLC\n";
    std::cout << "================================================================================\n";

    int dev_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&dev_count));
    if (dev_count == 0) {
        std::cerr << "No CUDA devices found!\n";
        return 1;
    }

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Device: " << prop.name << " (" << (prop.totalGlobalMem / (1024 * 1024))
              << " MB VRAM, Compute " << prop.major << "." << prop.minor << ")\n";

    bool inv1 = test_format_and_zero_blanking_invariance();
    bool inv2 = test_spatial_covariance_and_mvdr_conditioning();
    bool inv3 = test_distortionless_constraint_and_frb_recovery();
    bool inv4 = test_spatial_null_depth_and_suppression();
    bool inv5 = test_chained_realtime_stress_loop();

    const bool all_passed = inv1 && inv2 && inv3 && inv4 && inv5;

    std::cout << "\n================================================================================\n";
    std::cout << "  OVERALL CHAINED PIPELINE SUITE RESULT: "
              << (all_passed ? "ALL 5 INVARIANTS VERIFIED [SUCCESS]" : "INVARIANT FAILURE [FAILED]") << "\n";
    std::cout << "================================================================================\n";

    return all_passed ? 0 : 1;
}
