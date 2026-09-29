#include "cudaBiSLC.hpp"
#include "cudaUtils.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace kotekan {

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
    float diag_loading) {

    if (num_beams == 0 || n_freq == 0 || n_ant == 0) return;

    const std::size_t B = std::min(num_beams, MAX_BISLC_BEAMS);
    const double inv_n_active = (n_active > 0) ? (1.0 / static_cast<double>(n_active)) : 0.0;
    constexpr double TWO_PI = charts::constants::two_pi;
    constexpr double C = charts::constants::speed_of_light_m_per_s;

    // Allocate temporary B x B buffer for matrix inversion per frequency
    std::vector<float2> A_local(B * B);
    std::vector<float2> M_local(B * B);

    for (std::size_t f = 0; f < n_freq; ++f) {
        const double k = TWO_PI * frequencies_hz[f] / C;

        for (std::size_t i = 0; i < B; ++i) {
            for (std::size_t j = 0; j < B; ++j) {
                if (i == j) {
                    A_local[i * B + j] = make_float2(1.0f, 0.0f);
                    continue;
                }

                // Delta direction cosines: s_j - s_i
                const double dl = static_cast<double>(targets[j].direction.x - targets[i].direction.x);
                const double dm = static_cast<double>(targets[j].direction.y - targets[i].direction.y);
                const double dn = static_cast<double>(targets[j].direction.z - targets[i].direction.z);

                double sum_r = 0.0;
                double sum_i = 0.0;

                for (std::size_t a = 0; a < n_ant; ++a) {
                    if (antenna_mask != nullptr && antenna_mask[a] == 0) {
                        continue;
                    }

                    const float3 pos = antenna_positions[a];
                    const double delay_m = static_cast<double>(pos.x) * dl +
                                           static_cast<double>(pos.y) * dm +
                                           static_cast<double>(pos.z) * dn;
                    const double phase = k * delay_m;
                    sum_r += std::cos(phase);
                    sum_i += std::sin(phase);
                }

                A_local[i * B + j] = make_float2(
                    static_cast<float>(sum_r * inv_n_active),
                    static_cast<float>(sum_i * inv_n_active));
            }
        }

        // Invert A_local into M_local with diagonal loading
        invert_complex_matrix(A_local.data(), M_local.data(), B, diag_loading);

        // Copy into output buffer
        const std::size_t f_offset = f * B * B;
        for (std::size_t idx = 0; idx < B * B; ++idx) {
            M_out[f_offset + idx] = M_local[idx];
            if (A_out != nullptr) {
                A_out[f_offset + idx] = A_local[idx];
            }
        }
    }
}

namespace {

constexpr int WARPS_PER_BLOCK = 4;
constexpr int THREADS_PER_BLOCK = 32 * WARPS_PER_BLOCK; // 128 threads
constexpr int TIME_CHUNK_SIZE = 256;

template <int B>
__global__ void __launch_bounds__(128, 4)
bislc_unmixing_specialized_kernel(
    float2* __restrict__ cleaned,
    const float2* __restrict__ formed,
    const float2* __restrict__ unmix_matrices,
    const std::size_t n_time,
    const std::size_t n_freq,
    const std::size_t active_beams,
    const std::size_t max_beams_stride) {

    // Shared memory: 32 frequency channels per block, each holding a B x B complex matrix
    __shared__ float2 s_M[32][B][B];

    const unsigned int lane = threadIdx.x; // 0..31 (frequency lane in block)
    const unsigned int warp_id = threadIdx.y; // 0..3 (warp index in block)
    const unsigned int tid = warp_id * 32U + lane;

    const std::size_t f_base = static_cast<std::size_t>(blockIdx.x) * 32U;
    const std::size_t f = f_base + lane;
    const std::size_t t_base = static_cast<std::size_t>(blockIdx.y) * TIME_CHUNK_SIZE;

    // 1. Cooperative load of M(f) matrices for all 32 frequency channels into Shared Memory
    constexpr std::size_t ELEMS_PER_MAT = B * B;
    constexpr std::size_t TOTAL_MAT_ELEMS = 32 * ELEMS_PER_MAT;

    #pragma unroll
    for (std::size_t idx = tid; idx < TOTAL_MAT_ELEMS; idx += THREADS_PER_BLOCK) {
        const std::size_t f_slot = idx / ELEMS_PER_MAT;
        const std::size_t m_elem = idx % ELEMS_PER_MAT;
        const std::size_t f_curr = f_base + f_slot;

        if (f_curr < n_freq) {
            s_M[f_slot][m_elem / B][m_elem % B] = __ldg(&unmix_matrices[f_curr * ELEMS_PER_MAT + m_elem]);
        } else {
            s_M[f_slot][m_elem / B][m_elem % B] = make_float2(0.0f, 0.0f);
        }
    }
    __syncthreads();

    if (f >= n_freq) return;

    // 2. Each warp handles a slice of time samples within this block's TIME_CHUNK_SIZE
    constexpr std::size_t SAMPLES_PER_WARP = TIME_CHUNK_SIZE / WARPS_PER_BLOCK; // 64
    const std::size_t t_start = t_base + warp_id * SAMPLES_PER_WARP;
    const std::size_t t_end = (t_start + SAMPLES_PER_WARP < n_time) ? (t_start + SAMPLES_PER_WARP) : n_time;

    if (t_start >= n_time) return;

    const std::size_t stride_time = n_freq * max_beams_stride;
    const std::size_t in_base_offset = (t_start * n_freq + f) * max_beams_stride;

    const float2* in_ptr = formed + in_base_offset;
    float2* out_ptr = cleaned + in_base_offset;

    // 3. Process time samples with 100% coalesced global memory I/O and fused fmaf unmixing
    std::size_t t = t_start;

    // Main unrolled loop (4 samples per iteration)
    constexpr int UNROLL = 4;
    for (; t + (UNROLL - 1) < t_end; t += UNROLL) {
        #pragma unroll
        for (int k = 0; k < UNROLL; ++k) {
            const float2* curr_in = in_ptr + k * stride_time;
            float2* curr_out = out_ptr + k * stride_time;

            float2 y[B];
            #pragma unroll
            for (int b = 0; b < B; ++b) {
                y[b] = (b < active_beams) ? __ldg(&curr_in[b]) : make_float2(0.0f, 0.0f);
            }

            #pragma unroll
            for (int i = 0; i < B; ++i) {
                if (i < active_beams) {
                    float sr = 0.0f;
                    float si = 0.0f;

                    #pragma unroll
                    for (int j = 0; j < B; ++j) {
                        const float2 mij = s_M[lane][i][j];
                        const float2 yj = y[j];

                        // Complex multiplication: (mij.x + j*mij.y) * (yj.x + j*yj.y)
                        sr = fmaf(mij.x, yj.x, fmaf(-mij.y, yj.y, sr));
                        si = fmaf(mij.x, yj.y, fmaf(mij.y, yj.x, si));
                    }
                    curr_out[i] = make_float2(sr, si);
                } else if (i < max_beams_stride) {
                    curr_out[i] = make_float2(0.0f, 0.0f);
                }
            }
        }

        in_ptr += UNROLL * stride_time;
        out_ptr += UNROLL * stride_time;
    }

    // Remainder loop
    for (; t < t_end; ++t) {
        float2 y[B];
        #pragma unroll
        for (int b = 0; b < B; ++b) {
            y[b] = (b < active_beams) ? __ldg(&in_ptr[b]) : make_float2(0.0f, 0.0f);
        }

        #pragma unroll
        for (int i = 0; i < B; ++i) {
            if (i < active_beams) {
                float sr = 0.0f;
                float si = 0.0f;

                #pragma unroll
                for (int j = 0; j < B; ++j) {
                    const float2 mij = s_M[lane][i][j];
                    const float2 yj = y[j];

                    sr = fmaf(mij.x, yj.x, fmaf(-mij.y, yj.y, sr));
                    si = fmaf(mij.x, yj.y, fmaf(mij.y, yj.x, si));
                }
                out_ptr[i] = make_float2(sr, si);
            } else if (i < max_beams_stride) {
                out_ptr[i] = make_float2(0.0f, 0.0f);
            }
        }

        in_ptr += stride_time;
        out_ptr += stride_time;
    }
}

} // namespace

void launch_bislc_unmixing(
    float2* d_cleaned_voltages,
    const float2* d_formed_voltages,
    const float2* d_unmix_matrices,
    std::size_t n_time,
    std::size_t n_freq,
    std::size_t num_active_beams,
    std::size_t max_beams_stride,
    cudaStream_t stream) {

    if (num_active_beams == 0 || n_time == 0 || n_freq == 0) return;

    const std::size_t active = std::min(num_active_beams, MAX_BISLC_BEAMS);
    const std::size_t stride = std::max(max_beams_stride, active);

    const unsigned int grid_x = static_cast<unsigned int>((n_freq + 31U) / 32U);
    const unsigned int grid_y = static_cast<unsigned int>((n_time + TIME_CHUNK_SIZE - 1U) / TIME_CHUNK_SIZE);
    const dim3 grid_dim(grid_x, grid_y);
    const dim3 block_dim(32, WARPS_PER_BLOCK); // 128 threads

    // Dispatch specialized template kernel for optimal register allocation
    if (active <= 2) {
        bislc_unmixing_specialized_kernel<2><<<grid_dim, block_dim, 0, stream>>>(
            d_cleaned_voltages, d_formed_voltages, d_unmix_matrices,
            n_time, n_freq, active, stride);
    } else if (active <= 4) {
        bislc_unmixing_specialized_kernel<4><<<grid_dim, block_dim, 0, stream>>>(
            d_cleaned_voltages, d_formed_voltages, d_unmix_matrices,
            n_time, n_freq, active, stride);
    } else {
        bislc_unmixing_specialized_kernel<8><<<grid_dim, block_dim, 0, stream>>>(
            d_cleaned_voltages, d_formed_voltages, d_unmix_matrices,
            n_time, n_freq, active, stride);
    }

    CHECK_CUDA_ERROR_NON_OO(cudaGetLastError());
}

} // namespace kotekan
