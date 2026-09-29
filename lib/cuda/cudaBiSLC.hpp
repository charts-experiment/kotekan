#ifndef CUDA_BISLC_HPP
#define CUDA_BISLC_HPP

#include "cudaDirectBeamTracker.hpp"
#include "chartsConstants.hpp"

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>

namespace kotekan {

constexpr std::size_t MAX_BISLC_BEAMS = MAX_DIRECT_BEAMS; // up to 8 concurrent beams

/**
 * @brief Configuration for BiSLC (Bi-directional Sidelobe Cancellation / Joint Matrix Unmixing).
 */
struct BiSLCConfig {
    bool enabled = true;
    std::size_t num_active_beams = 1;
    std::size_t max_beams_stride = MAX_BISLC_BEAMS;
    float diagonal_loading = 1.0e-4f; // Regularizer epsilon added to diagonal: A_reg = A + eps * I
    std::array<DirectBeamTarget, MAX_BISLC_BEAMS> targets;
    std::array<uint8_t, MAX_DIRECT_ANTENNAS> antenna_mask;
    std::array<float3, MAX_DIRECT_ANTENNAS> antenna_positions;
    std::size_t num_active_antennas = MAX_DIRECT_ANTENNAS;

    BiSLCConfig() {
        antenna_mask.fill(1);
        num_active_antennas = MAX_DIRECT_ANTENNAS;
        for (std::size_t i = 0; i < MAX_DIRECT_ANTENNAS; ++i) {
            const unsigned int col = (i < 64) ? (i & 7U) : (i & 15U);
            const unsigned int row = (i < 64) ? (i >> 3U) : (i >> 4U);
            antenna_positions[i] = make_float3(
                static_cast<float>(col) * charts::constants::charts_default_spacing_m,
                static_cast<float>(row) * charts::constants::charts_default_spacing_m,
                0.0f);
        }
    }
};

/**
 * @brief Invert a B x B complex matrix using Gauss-Jordan elimination with partial pivoting.
 *
 * @param A_in          Input B x B complex matrix [B * B] (row-major: A[i * B + j])
 * @param M_out         Output inverted B x B complex matrix [B * B] (row-major)
 * @param B             Dimension of matrix (B <= 8)
 * @param diag_loading  Epsilon added to diagonal (A_reg = A + diag_loading * I)
 * @return true if successfully inverted, false if singular
 */
inline bool invert_complex_matrix(
    const float2* A_in,
    float2* M_out,
    std::size_t B,
    float diag_loading = 1.0e-4f) {

    if (B == 0) return true;
    if (B == 1) {
        float ar = A_in[0].x + diag_loading;
        float ai = A_in[0].y;
        float den = ar * ar + ai * ai;
        if (den < 1.0e-12f) {
            M_out[0] = make_float2(1.0f, 0.0f);
            return false;
        }
        M_out[0] = make_float2(ar / den, -ai / den);
        return true;
    }

    if (B == 2) {
        // Fast analytical 2x2 complex inversion
        // A = [a, b; c, d]
        float2 a = make_float2(A_in[0].x + diag_loading, A_in[0].y);
        float2 b = A_in[1];
        float2 c = A_in[2];
        float2 d = make_float2(A_in[3].x + diag_loading, A_in[3].y);

        // det = a*d - b*c
        float ad_r = a.x * d.x - a.y * d.y;
        float ad_i = a.x * d.y + a.y * d.x;
        float bc_r = b.x * c.x - b.y * c.y;
        float bc_i = b.x * c.y + b.y * c.x;

        float det_r = ad_r - bc_r;
        float det_i = ad_i - bc_i;
        float det_sq = det_r * det_r + det_i * det_i;

        if (det_sq < 1.0e-14f) {
            // Degenerate: fallback to identity
            M_out[0] = make_float2(1.0f, 0.0f);
            M_out[1] = make_float2(0.0f, 0.0f);
            M_out[2] = make_float2(0.0f, 0.0f);
            M_out[3] = make_float2(1.0f, 0.0f);
            return false;
        }

        // inv_det = det^* / |det|^2
        float idet_r = det_r / det_sq;
        float idet_i = -det_i / det_sq;

        // M = inv_det * [d, -b; -c, a]
        M_out[0] = make_float2(d.x * idet_r - d.y * idet_i, d.x * idet_i + d.y * idet_r);
        M_out[1] = make_float2(-b.x * idet_r - (-b.y) * idet_i, -b.x * idet_i + (-b.y) * idet_r);
        M_out[2] = make_float2(-c.x * idet_r - (-c.y) * idet_i, -c.x * idet_i + (-c.y) * idet_r);
        M_out[3] = make_float2(a.x * idet_r - a.y * idet_i, a.x * idet_i + a.y * idet_r);
        return true;
    }

    // General Gauss-Jordan elimination with partial pivoting for B <= 8
    constexpr std::size_t MAX_B = 8;
    if (B > MAX_B) return false;

    std::complex<float> aug[MAX_B][2 * MAX_B];

    for (std::size_t i = 0; i < B; ++i) {
        for (std::size_t j = 0; j < B; ++j) {
            aug[i][j] = std::complex<float>(A_in[i * B + j].x, A_in[i * B + j].y);
            if (i == j) {
                aug[i][j] += std::complex<float>(diag_loading, 0.0f);
            }
        }
        for (std::size_t j = 0; j < B; ++j) {
            aug[i][B + j] = (i == j) ? std::complex<float>(1.0f, 0.0f) : std::complex<float>(0.0f, 0.0f);
        }
    }

    for (std::size_t col = 0; col < B; ++col) {
        // Find pivot with maximum norm squared
        std::size_t pivot_row = col;
        float max_norm_sq = std::norm(aug[col][col]);

        for (std::size_t row = col + 1; row < B; ++row) {
            float n_sq = std::norm(aug[row][col]);
            if (n_sq > max_norm_sq) {
                max_norm_sq = n_sq;
                pivot_row = row;
            }
        }

        if (max_norm_sq < 1.0e-14f) {
            // Singular: return identity
            for (std::size_t i = 0; i < B; ++i) {
                for (std::size_t j = 0; j < B; ++j) {
                    M_out[i * B + j] = (i == j) ? make_float2(1.0f, 0.0f) : make_float2(0.0f, 0.0f);
                }
            }
            return false;
        }

        // Swap pivot row if needed
        if (pivot_row != col) {
            for (std::size_t j = 0; j < 2 * B; ++j) {
                std::swap(aug[col][j], aug[pivot_row][j]);
            }
        }

        // Scale pivot row so diagonal element is 1.0
        std::complex<float> pivot_val = aug[col][col];
        for (std::size_t j = 0; j < 2 * B; ++j) {
            aug[col][j] /= pivot_val;
        }

        // Eliminate column entries in all other rows
        for (std::size_t row = 0; row < B; ++row) {
            if (row != col) {
                std::complex<float> factor = aug[row][col];
                for (std::size_t j = col; j < 2 * B; ++j) {
                    aug[row][j] -= factor * aug[col][j];
                }
            }
        }
    }

    // Extract inverted matrix from right half of augmented matrix
    for (std::size_t i = 0; i < B; ++i) {
        for (std::size_t j = 0; j < B; ++j) {
            M_out[i * B + j] = make_float2(aug[i][B + j].real(), aug[i][B + j].imag());
        }
    }

    return true;
}

/**
 * @brief Compute the B x B complex beam coupling matrix A(f) and its inverse M(f) for all frequency channels.
 *
 * A_{i, j}(f) = (i == j) ? 1.0 : (1.0 / N_active) * sum_{a in active} exp(+j * k(f) * x_a . (s_j - s_i))
 *
 * @param M_out                 Output inverse matrices [n_freq][B][B] of float2
 * @param A_out                 Optional output coupling matrices [n_freq][B][B] of float2 (can be nullptr)
 * @param targets               Active beam targets (directions) [num_beams]
 * @param frequencies_hz        Physical frequency channel values [n_freq]
 * @param antenna_positions     Physical antenna positions [n_ant]
 * @param antenna_mask          Antenna active/masked flags [n_ant]
 * @param num_beams             Number of concurrent virtual beams (B <= 8)
 * @param n_freq                Number of frequency channels
 * @param n_ant                 Number of physical antennas
 * @param n_active              Number of active (unmasked) antennas
 * @param diag_loading          Diagonal loading regularizer epsilon
 */
void compute_beam_coupling_and_inverses(
    float2* M_out,
    float2* A_out,
    const DirectBeamTarget* targets,
    const std::vector<double>& frequencies_hz,
    const float3* antenna_positions,
    const uint8_t* antenna_mask,
    std::size_t num_beams,
    std::size_t n_freq,
    std::size_t n_ant,
    std::size_t n_active,
    float diag_loading = 1.0e-4f);

/**
 * @brief Launch CUDA BiSLC (Joint Matrix Unmixing) kernel across complex baseband voltages.
 *
 * s(t, f) = M(f) . y(t, f)
 *
 * @param d_cleaned_voltages    Device output buffer [time][freq][max_beams_stride] (float2)
 * @param d_formed_voltages     Device input buffer [time][freq][max_beams_stride] (float2)
 * @param d_unmix_matrices      Device pointer to inverted coupling matrices [n_freq][B][B] (float2)
 * @param n_time                Number of time samples per frame (e.g. 15,360)
 * @param n_freq                Number of frequency channels (e.g. 336)
 * @param num_active_beams      Active beams to unmix (1..8)
 * @param max_beams_stride      Total stride of beam dimension in buffers (e.g. 2, 4, or 8)
 * @param stream                CUDA stream
 */
void launch_bislc_unmixing(
    float2* d_cleaned_voltages,
    const float2* d_formed_voltages,
    const float2* d_unmix_matrices,
    std::size_t n_time,
    std::size_t n_freq,
    std::size_t num_active_beams,
    std::size_t max_beams_stride,
    cudaStream_t stream = nullptr);

} // namespace kotekan

#endif // CUDA_BISLC_HPP
