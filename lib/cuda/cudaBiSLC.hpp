#ifndef CUDA_BISLC_HPP
#define CUDA_BISLC_HPP

#include "chartsConstants.hpp"
#include "cudaDirectBeamTracker.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <string>
#include <vector>

namespace kotekan {

constexpr std::size_t MAX_BISLC_BEAMS = 32;

/**
 * @brief Configuration for BiSLC (Bi-directional Sidelobe Cancellation / Joint Matrix Unmixing).
 */
struct BiSLCConfig {
    bool enabled = true;
    std::size_t num_antennas = 64;
    std::size_t num_pols = 1;
    DirectPolLayout pol_layout = DirectPolLayout::INTERLEAVED;
    std::string antenna_order = "descending";

    std::size_t num_active_beams = 1;
    std::size_t max_beams_stride = 16;
    float diagonal_loading = 1.0e-4f;

    std::vector<DirectBeamTarget> targets{MAX_DIRECT_BEAMS};
    std::array<std::uint8_t, MAX_DIRECT_ANTENNAS> antenna_mask;
    std::array<float3, MAX_DIRECT_ANTENNAS> antenna_positions;
    std::size_t num_active_antennas = 64;

    std::uint64_t config_generation = 1;
    std::uint64_t tracker_generation_synced = 0;

    BiSLCConfig() {
        antenna_mask.fill(1);
        num_active_antennas = 64;
        for (std::size_t i = 0; i < MAX_DIRECT_ANTENNAS; ++i) {
            const unsigned int col = (i < 64) ? (i & 7U) : (i & 15U);
            const unsigned int row = (i < 64) ? (i >> 3U) : (i >> 4U);
            antenna_positions[i] = make_float3(
                static_cast<float>(col) * charts::constants::charts_default_spacing_m,
                static_cast<float>(row) * charts::constants::charts_default_spacing_m, 0.0f);
        }
    }

    std::size_t total_elements() const {
        return num_antennas * num_pols;
    }
};

/**
 * @brief Invert a B x B complex matrix using Gauss-Jordan elimination with partial pivoting in double precision.
 *
 * @param A_in          Input B x B complex matrix [B * B] (row-major: A[i * B + j])
 * @param M_out         Output inverted B x B complex matrix [B * B] (row-major)
 * @param B             Dimension of matrix (B <= MAX_BISLC_BEAMS)
 * @param diag_loading  Epsilon added to diagonal (A_reg = A + diag_loading * I)
 * @return true if successfully inverted, false if singular
 */
inline bool invert_complex_matrix(const float2* A_in, float2* M_out, std::size_t B,
                                  float diag_loading = 1.0e-4f) {
    if (B == 0) return true;
    if (B == 1) {
        const float ar = A_in[0].x + diag_loading;
        const float ai = A_in[0].y;
        const float den = ar * ar + ai * ai;
        if (den < 1.0e-12f) {
            M_out[0] = make_float2(1.0f, 0.0f);
            return false;
        }
        M_out[0] = make_float2(ar / den, -ai / den);
        return true;
    }

    if (B > MAX_BISLC_BEAMS) return false;

    std::complex<double> aug[MAX_BISLC_BEAMS][2 * MAX_BISLC_BEAMS];

    for (std::size_t i = 0; i < B; ++i) {
        for (std::size_t j = 0; j < B; ++j) {
            aug[i][j] = std::complex<double>(A_in[i * B + j].x, A_in[i * B + j].y);
            if (i == j) {
                aug[i][j] += std::complex<double>(diag_loading, 0.0);
            }
            aug[i][B + j] = (i == j) ? std::complex<double>(1.0, 0.0)
                                     : std::complex<double>(0.0, 0.0);
        }
    }

    for (std::size_t col = 0; col < B; ++col) {
        std::size_t pivot_row = col;
        double max_norm_sq = std::norm(aug[col][col]);

        for (std::size_t row = col + 1; row < B; ++row) {
            const double n_sq = std::norm(aug[row][col]);
            if (n_sq > max_norm_sq) {
                max_norm_sq = n_sq;
                pivot_row = row;
            }
        }

        if (max_norm_sq < 1.0e-18) {
            for (std::size_t i = 0; i < B; ++i) {
                for (std::size_t j = 0; j < B; ++j) {
                    M_out[i * B + j] = (i == j) ? make_float2(1.0f, 0.0f)
                                                : make_float2(0.0f, 0.0f);
                }
            }
            return false;
        }

        if (pivot_row != col) {
            for (std::size_t j = 0; j < 2 * B; ++j) {
                std::swap(aug[col][j], aug[pivot_row][j]);
            }
        }

        const std::complex<double> pivot_val = aug[col][col];
        for (std::size_t j = col; j < 2 * B; ++j) {
            aug[col][j] /= pivot_val;
        }

        for (std::size_t row = 0; row < B; ++row) {
            if (row != col) {
                const std::complex<double> factor = aug[row][col];
                for (std::size_t j = col; j < 2 * B; ++j) {
                    aug[row][j] -= factor * aug[col][j];
                }
            }
        }
    }

    for (std::size_t i = 0; i < B; ++i) {
        for (std::size_t j = 0; j < B; ++j) {
            M_out[i * B + j] = make_float2(
                static_cast<float>(aug[i][B + j].real()),
                static_cast<float>(aug[i][B + j].imag()));
        }
    }
    return true;
}

/**
 * @brief Host helper: Compute the B x B complex beam coupling matrix A(f) and its inverse M(f).
 */
void compute_beam_coupling_and_inverses(
    float2* M_out,
    float2* A_out,
    const DirectBeamTarget* targets,
    const std::vector<double>& frequencies_hz,
    const float3* antenna_positions,
    const std::uint8_t* antenna_mask,
    std::size_t num_beams,
    std::size_t n_freq,
    std::size_t n_elem,
    std::size_t n_active,
    float diag_loading = 1.0e-4f);

/**
 * @brief GPU-accelerated computation of coupling matrices A(f) and unmixing inverses M(f) = (A(f) + eps*I)^-1.
 *        Executes in-stream on device without stalling host CPU or PCIe bus.
 */
void launch_compute_bislc_matrices_gpu(
    float2* d_unmix_matrices,
    float2* d_coupling_matrices,
    const DirectDirection3D* d_directions,
    const double* d_wavenumbers,
    const float3* d_antenna_positions,
    const std::uint8_t* d_antenna_mask,
    std::size_t num_beams,
    std::size_t n_freq,
    std::size_t n_elem,
    float diag_loading,
    cudaStream_t stream = nullptr);

/**
 * @brief Launch bank-conflict-free Blackwell BiSLC (Joint Matrix Unmixing) kernel.
 *        Supports n_pol = 1 or 2 and arbitrary active beams up to MAX_BISLC_BEAMS.
 */
void launch_bislc_unmixing(
    float2* d_cleaned_voltages,
    const float2* d_formed_voltages,
    const float2* d_unmix_matrices,
    std::size_t n_time,
    std::size_t n_freq,
    std::size_t num_active_beams,
    std::size_t max_beams_stride,
    cudaStream_t stream = nullptr,
    std::size_t n_pol = 1);

/**
 * @brief Optional Weight-Space Sidelobe Cancellation:
 *        Pre-multiplies steering weights W_clean(f) = M(f) * W(f) directly on GPU.
 */
void launch_apply_bislc_to_weights(
    float2* d_weights_inout,
    const float2* d_unmix_matrices,
    std::size_t num_active_beams,
    std::size_t n_freq,
    std::size_t n_elem,
    cudaStream_t stream = nullptr);

} // namespace kotekan

#endif // CUDA_BISLC_HPP
