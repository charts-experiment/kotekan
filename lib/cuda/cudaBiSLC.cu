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
    const std::uint8_t* antenna_mask,
    std::size_t num_beams,
    std::size_t n_freq,
    std::size_t n_elem,
    std::size_t n_active,
    float diag_loading) {

    if (num_beams == 0 || n_freq == 0 || n_elem == 0) return;

    const std::size_t B = std::min(num_beams, MAX_BISLC_BEAMS);
    const double inv_n_active = (n_active > 0) ? (1.0 / static_cast<double>(n_active)) : 0.0;
    constexpr double TWO_PI = charts::constants::two_pi;
    constexpr double C = charts::constants::speed_of_light_m_per_s;

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

                // Phase sign matching DirectBeamformer response to source j steered at i: s_i - s_j
                const double dl = static_cast<double>(targets[i].direction.x - targets[j].direction.x);
                const double dm = static_cast<double>(targets[i].direction.y - targets[j].direction.y);
                const double dn = static_cast<double>(targets[i].direction.z - targets[j].direction.z);

                double sum_r = 0.0;
                double sum_i = 0.0;

                for (std::size_t a = 0; a < n_elem; ++a) {
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

        invert_complex_matrix(A_local.data(), M_local.data(), B, diag_loading);

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

// ============================================================================
// GPU Kernel 1: Compute A(f) and Invert M(f) = (A(f) + eps*I)^-1 on Device
// ============================================================================
template <int MAX_B>
__global__ void __launch_bounds__(256)
compute_and_invert_bislc_matrices_kernel(
    float2* __restrict__ d_M,
    float2* __restrict__ d_A,
    const DirectDirection3D* __restrict__ d_dirs,
    const double* __restrict__ d_wavenumbers,
    const float3* __restrict__ d_positions,
    const std::uint8_t* __restrict__ d_mask,
    const std::size_t B,
    const std::size_t n_freq,
    const std::size_t n_elem,
    const float diag_loading) {

    const std::size_t f = blockIdx.x;
    if (f >= n_freq || B == 0 || B > MAX_B) return;

    const unsigned int tid = threadIdx.x;
    const double k = __ldg(&d_wavenumbers[f]);

    __shared__ float2 s_aug[MAX_B][2 * MAX_B];
    __shared__ float2 s_factors[MAX_B];
    __shared__ unsigned int s_active_count;

    if (tid == 0) {
        unsigned int cnt = 0;
        for (std::size_t a = 0; a < n_elem; ++a) {
            if (d_mask == nullptr || __ldg(&d_mask[a]) != 0) {
                cnt++;
            }
        }
        s_active_count = cnt;
    }
    __syncthreads();

    const float inv_n_active = (s_active_count > 0) ? (1.0f / static_cast<float>(s_active_count)) : 0.0f;
    const std::size_t total_pairs = B * B;

    // Step 1: Compute A_{i, j}(f) entries in parallel across threads
    for (std::size_t idx = tid; idx < total_pairs; idx += blockDim.x) {
        const std::size_t i = idx / B;
        const std::size_t j = idx % B;

        float2 a_val = make_float2(0.0f, 0.0f);
        if (i == j) {
            a_val = make_float2(1.0f, 0.0f);
        } else {
            const DirectDirection3D di = d_dirs[i];
            const DirectDirection3D dj = d_dirs[j];
            const double dl = static_cast<double>(di.x - dj.x);
            const double dm = static_cast<double>(di.y - dj.y);
            const double dn = static_cast<double>(di.z - dj.z);

            float sum_r = 0.0f;
            float sum_i = 0.0f;
            for (std::size_t a = 0; a < n_elem; ++a) {
                if (d_mask != nullptr && __ldg(&d_mask[a]) == 0) continue;
                const float3 pos = d_positions[a];
                const double delay_m = static_cast<double>(pos.x) * dl +
                                       static_cast<double>(pos.y) * dm +
                                       static_cast<double>(pos.z) * dn;
                const float phase = static_cast<float>(k * delay_m);
                float s, c;
                __sincosf(phase, &s, &c);
                sum_r += c;
                sum_i += s;
            }
            a_val = make_float2(sum_r * inv_n_active, sum_i * inv_n_active);
        }

        if (d_A != nullptr) {
            d_A[f * total_pairs + idx] = a_val;
        }

        s_aug[i][j] = (i == j) ? make_float2(a_val.x + diag_loading, a_val.y) : a_val;
        s_aug[i][B + j] = (i == j) ? make_float2(1.0f, 0.0f) : make_float2(0.0f, 0.0f);
    }
    __syncthreads();

    // Step 2: Parallel Gauss-Jordan elimination in Shared Memory
    __shared__ std::size_t s_pivot_row;
    __shared__ float2 s_pivot_val;
    __shared__ bool s_singular;

    if (tid == 0) s_singular = false;
    __syncthreads();

    for (std::size_t col = 0; col < B; ++col) {
        if (tid == 0) {
            std::size_t best_r = col;
            float max_norm = s_aug[col][col].x * s_aug[col][col].x +
                             s_aug[col][col].y * s_aug[col][col].y;
            for (std::size_t r = col + 1; r < B; ++r) {
                const float2 v = s_aug[r][col];
                const float n_sq = v.x * v.x + v.y * v.y;
                if (n_sq > max_norm) {
                    max_norm = n_sq;
                    best_r = r;
                }
            }
            s_pivot_row = best_r;
            if (max_norm < 1.0e-14f) {
                s_singular = true;
            }
        }
        __syncthreads();

        if (s_singular) break;

        // Swap pivot row across 2*B columns in parallel
        if (s_pivot_row != col && tid < 2 * B) {
            const float2 tmp = s_aug[col][tid];
            s_aug[col][tid] = s_aug[s_pivot_row][tid];
            s_aug[s_pivot_row][tid] = tmp;
        }
        __syncthreads();

        if (tid == 0) {
            s_pivot_val = s_aug[col][col];
        }
        __syncthreads();

        // Normalize pivot row
        if (tid < 2 * B) {
            const float2 num = s_aug[col][tid];
            const float den = s_pivot_val.x * s_pivot_val.x + s_pivot_val.y * s_pivot_val.y;
            const float inv_den = 1.0f / den;
            s_aug[col][tid] = make_float2(
                (num.x * s_pivot_val.x + num.y * s_pivot_val.y) * inv_den,
                (num.y * s_pivot_val.x - num.x * s_pivot_val.y) * inv_den);
        }
        __syncthreads();

        // Cache elimination factors in shared memory to avoid data race
        if (tid < B) {
            s_factors[tid] = s_aug[tid][col];
        }
        __syncthreads();

        // Eliminate other rows in parallel
        for (std::size_t idx = tid; idx < B * (2 * B); idx += blockDim.x) {
            const std::size_t r = idx / (2 * B);
            const std::size_t c = idx % (2 * B);
            if (r != col && c >= col) {
                const float2 factor = s_factors[r];
                const float2 p_row_val = s_aug[col][c];
                s_aug[r][c].x -= (factor.x * p_row_val.x - factor.y * p_row_val.y);
                s_aug[r][c].y -= (factor.x * p_row_val.y + factor.y * p_row_val.x);
            }
        }
        __syncthreads();
    }

    // Step 3: Write inverted matrix M(f) to global memory with stride B * B
    for (std::size_t idx = tid; idx < total_pairs; idx += blockDim.x) {
        const std::size_t i = idx / B;
        const std::size_t j = idx % B;
        if (s_singular) {
            d_M[f * total_pairs + idx] = (i == j) ? make_float2(1.0f, 0.0f) : make_float2(0.0f, 0.0f);
        } else {
            d_M[f * total_pairs + idx] = s_aug[i][B + j];
        }
    }
}

// ============================================================================
// GPU Kernel 2: Bank-Conflict-Free Blackwell BiSLC Unmixing Kernel
// ============================================================================
constexpr int WARPS_PER_BLOCK = 4;
constexpr int TIME_CHUNK_SIZE = 256;

template <int B_MAX, int N_POL, int FREQ_LANES = 32>
__global__ void __launch_bounds__(FREQ_LANES * WARPS_PER_BLOCK, 4)
bislc_unmixing_blackwell_kernel(
    float2* __restrict__ cleaned,
    const float2* __restrict__ formed,
    const float2* __restrict__ unmix_matrices,
    const std::size_t n_time,
    const std::size_t n_freq,
    const std::size_t active_beams,
    const std::size_t max_beams_stride) {

    // Transposed layout [B_MAX][B_MAX][FREQ_LANES]: lane is innermost!
    // Guarantees 100% bank-conflict-free shared memory loads across all threads in a warp.
    __shared__ float2 s_M[B_MAX][B_MAX][FREQ_LANES];

    const unsigned int lane = threadIdx.x;    // 0..FREQ_LANES - 1
    const unsigned int warp_id = threadIdx.y; // 0..WARPS_PER_BLOCK - 1
    const unsigned int tid = warp_id * FREQ_LANES + lane;
    const unsigned int block_threads = WARPS_PER_BLOCK * FREQ_LANES;

    const std::size_t f_base = static_cast<std::size_t>(blockIdx.x) * FREQ_LANES;
    const std::size_t f = f_base + lane;
    const std::size_t t_base = static_cast<std::size_t>(blockIdx.y) * TIME_CHUNK_SIZE;

    // 1. Cooperative load of M(f) using active_beams * active_beams stride
    const std::size_t elems_per_mat = active_beams * active_beams;
    const std::size_t total_mat_elems = FREQ_LANES * elems_per_mat;

    for (std::size_t idx = tid; idx < total_mat_elems; idx += block_threads) {
        const std::size_t f_slot = idx / elems_per_mat;
        const std::size_t m_elem = idx % elems_per_mat;
        const std::size_t row = m_elem / active_beams;
        const std::size_t col = m_elem % active_beams;
        const std::size_t f_curr = f_base + f_slot;

        if (f_curr < n_freq && row < B_MAX && col < B_MAX) {
            s_M[row][col][f_slot] = __ldg(&unmix_matrices[f_curr * elems_per_mat + m_elem]);
        }
    }
    __syncthreads();

    if (f >= n_freq) return;

    constexpr std::size_t SAMPLES_PER_WARP = TIME_CHUNK_SIZE / WARPS_PER_BLOCK; // 64
    const std::size_t t_start = t_base + warp_id * SAMPLES_PER_WARP;
    const std::size_t t_end =
        (t_start + SAMPLES_PER_WARP < n_time) ? (t_start + SAMPLES_PER_WARP) : n_time;

    if (t_start >= n_time) return;

    const std::size_t beam_pol_stride = max_beams_stride * N_POL;
    const std::size_t stride_time = n_freq * beam_pol_stride;
    const std::size_t base_offset = (t_start * n_freq + f) * beam_pol_stride;

    const float2* in_ptr = formed + base_offset;
    float2* out_ptr = cleaned + base_offset;

    for (std::size_t t = t_start; t < t_end; ++t) {
        float2 y[B_MAX][N_POL];
        #pragma unroll
        for (int b = 0; b < B_MAX; ++b) {
            if (b < active_beams) {
                #pragma unroll
                for (int p = 0; p < N_POL; ++p) {
                    y[b][p] = __ldg(&in_ptr[b * N_POL + p]);
                }
            }
        }

        #pragma unroll
        for (int i = 0; i < B_MAX; ++i) {
            if (i < active_beams) {
                float sr[N_POL] = {0.0f};
                float si[N_POL] = {0.0f};

                #pragma unroll
                for (int j = 0; j < B_MAX; ++j) {
                    if (j < active_beams) {
                        const float2 mij = s_M[i][j][lane];
                        #pragma unroll
                        for (int p = 0; p < N_POL; ++p) {
                            const float2 yj = y[j][p];
                            sr[p] = fmaf(mij.x, yj.x, fmaf(-mij.y, yj.y, sr[p]));
                            si[p] = fmaf(mij.x, yj.y, fmaf(mij.y, yj.x, si[p]));
                        }
                    }
                }

                #pragma unroll
                for (int p = 0; p < N_POL; ++p) {
                    out_ptr[i * N_POL + p] = make_float2(sr[p], si[p]);
                }
            }
        }

        // Explicitly zero inactive beams up to max_beams_stride
        for (std::size_t i = active_beams; i < max_beams_stride; ++i) {
            #pragma unroll
            for (int p = 0; p < N_POL; ++p) {
                out_ptr[i * N_POL + p] = make_float2(0.0f, 0.0f);
            }
        }

        in_ptr += stride_time;
        out_ptr += stride_time;
    }
}

// ============================================================================
// GPU Kernel 3: Weight-Space Sidelobe Pre-Cancellation W_clean(f) = M(f) * W(f)
// ============================================================================
template <int B_MAX>
__global__ void __launch_bounds__(256)
apply_bislc_to_weights_kernel(
    float2* __restrict__ weights,
    const float2* __restrict__ unmix_matrices,
    const std::size_t active_beams,
    const std::size_t n_freq,
    const std::size_t n_elem) {

    const std::size_t fe_idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total_fe = n_freq * n_elem;
    if (fe_idx >= total_fe || active_beams <= 1 || active_beams > B_MAX) return;

    const std::size_t freq = fe_idx / n_elem;
    const std::size_t elem = fe_idx % n_elem;
    const std::size_t fe_stride = total_fe; // [beam][freq][elem]

    float2 w_in[B_MAX];
    #pragma unroll
    for (int b = 0; b < B_MAX; ++b) {
        if (b < active_beams) {
            w_in[b] = __ldg(&weights[b * fe_stride + freq * n_elem + elem]);
        }
    }

    const float2* M_f = unmix_matrices + freq * active_beams * active_beams;

    #pragma unroll
    for (int i = 0; i < B_MAX; ++i) {
        if (i < active_beams) {
            float sr = 0.0f;
            float si = 0.0f;
            #pragma unroll
            for (int j = 0; j < B_MAX; ++j) {
                if (j < active_beams) {
                    const float2 mij = __ldg(&M_f[i * active_beams + j]);
                    const float2 wj = w_in[j];
                    sr = fmaf(mij.x, wj.x, fmaf(-mij.y, wj.y, sr));
                    si = fmaf(mij.x, wj.y, fmaf(mij.y, wj.x, si));
                }
            }
            weights[i * fe_stride + freq * n_elem + elem] = make_float2(sr, si);
        }
    }
}

} // namespace

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
    cudaStream_t stream) {

    if (num_beams == 0 || n_freq == 0 || n_elem == 0) return;
    const std::size_t B = std::min(num_beams, MAX_BISLC_BEAMS);

    const unsigned int grid = static_cast<unsigned int>(n_freq);
    constexpr unsigned int BLOCK = 256;

    if (B <= 8) {
        compute_and_invert_bislc_matrices_kernel<8><<<grid, BLOCK, 0, stream>>>(
            d_unmix_matrices, d_coupling_matrices, d_directions, d_wavenumbers,
            d_antenna_positions, d_antenna_mask, B, n_freq, n_elem, diag_loading);
    } else if (B <= 16) {
        compute_and_invert_bislc_matrices_kernel<16><<<grid, BLOCK, 0, stream>>>(
            d_unmix_matrices, d_coupling_matrices, d_directions, d_wavenumbers,
            d_antenna_positions, d_antenna_mask, B, n_freq, n_elem, diag_loading);
    } else {
        compute_and_invert_bislc_matrices_kernel<32><<<grid, BLOCK, 0, stream>>>(
            d_unmix_matrices, d_coupling_matrices, d_directions, d_wavenumbers,
            d_antenna_positions, d_antenna_mask, B, n_freq, n_elem, diag_loading);
    }
    CHECK_CUDA_ERROR_NON_OO(cudaGetLastError());
}

void launch_bislc_unmixing(
    float2* d_cleaned_voltages,
    const float2* d_formed_voltages,
    const float2* d_unmix_matrices,
    std::size_t n_time,
    std::size_t n_freq,
    std::size_t num_active_beams,
    std::size_t max_beams_stride,
    cudaStream_t stream,
    std::size_t n_pol) {

    if (num_active_beams == 0 || n_time == 0 || n_freq == 0) return;

    const std::size_t active = std::min(num_active_beams, MAX_BISLC_BEAMS);
    const std::size_t stride = std::max(max_beams_stride, active);

    auto dispatch_b = [&](auto pol_tag) {
        constexpr int NP = decltype(pol_tag)::value;
        if (active <= 2) {
            const unsigned int grid_x = static_cast<unsigned int>((n_freq + 31U) / 32U);
            const unsigned int grid_y =
                static_cast<unsigned int>((n_time + TIME_CHUNK_SIZE - 1U) / TIME_CHUNK_SIZE);
            bislc_unmixing_blackwell_kernel<2, NP, 32>
                <<<dim3(grid_x, grid_y), dim3(32, WARPS_PER_BLOCK), 0, stream>>>(
                    d_cleaned_voltages, d_formed_voltages, d_unmix_matrices, n_time, n_freq, active, stride);
        } else if (active <= 4) {
            const unsigned int grid_x = static_cast<unsigned int>((n_freq + 31U) / 32U);
            const unsigned int grid_y =
                static_cast<unsigned int>((n_time + TIME_CHUNK_SIZE - 1U) / TIME_CHUNK_SIZE);
            bislc_unmixing_blackwell_kernel<4, NP, 32>
                <<<dim3(grid_x, grid_y), dim3(32, WARPS_PER_BLOCK), 0, stream>>>(
                    d_cleaned_voltages, d_formed_voltages, d_unmix_matrices, n_time, n_freq, active, stride);
        } else if (active <= 8) {
            const unsigned int grid_x = static_cast<unsigned int>((n_freq + 31U) / 32U);
            const unsigned int grid_y =
                static_cast<unsigned int>((n_time + TIME_CHUNK_SIZE - 1U) / TIME_CHUNK_SIZE);
            bislc_unmixing_blackwell_kernel<8, NP, 32>
                <<<dim3(grid_x, grid_y), dim3(32, WARPS_PER_BLOCK), 0, stream>>>(
                    d_cleaned_voltages, d_formed_voltages, d_unmix_matrices, n_time, n_freq, active, stride);
        } else if (active <= 16) {
            const unsigned int grid_x = static_cast<unsigned int>((n_freq + 15U) / 16U);
            const unsigned int grid_y =
                static_cast<unsigned int>((n_time + TIME_CHUNK_SIZE - 1U) / TIME_CHUNK_SIZE);
            bislc_unmixing_blackwell_kernel<16, NP, 16>
                <<<dim3(grid_x, grid_y), dim3(16, WARPS_PER_BLOCK), 0, stream>>>(
                    d_cleaned_voltages, d_formed_voltages, d_unmix_matrices, n_time, n_freq, active, stride);
        } else {
            const unsigned int grid_x = static_cast<unsigned int>((n_freq + 3U) / 4U);
            const unsigned int grid_y =
                static_cast<unsigned int>((n_time + TIME_CHUNK_SIZE - 1U) / TIME_CHUNK_SIZE);
            bislc_unmixing_blackwell_kernel<32, NP, 4>
                <<<dim3(grid_x, grid_y), dim3(4, WARPS_PER_BLOCK), 0, stream>>>(
                    d_cleaned_voltages, d_formed_voltages, d_unmix_matrices, n_time, n_freq, active, stride);
        }
    };

    if (n_pol >= 2) {
        dispatch_b(std::integral_constant<int, 2>{});
    } else {
        dispatch_b(std::integral_constant<int, 1>{});
    }

    CHECK_CUDA_ERROR_NON_OO(cudaGetLastError());
}

void launch_apply_bislc_to_weights(
    float2* d_weights_inout,
    const float2* d_unmix_matrices,
    std::size_t num_active_beams,
    std::size_t n_freq,
    std::size_t n_elem,
    cudaStream_t stream) {

    const std::size_t active = std::min(num_active_beams, MAX_BISLC_BEAMS);
    const std::size_t total_fe = n_freq * n_elem;
    if (active <= 1 || total_fe == 0) return;

    constexpr unsigned int BLOCK = 256;
    const unsigned int grid = static_cast<unsigned int>((total_fe + BLOCK - 1U) / BLOCK);

    if (active <= 4) {
        apply_bislc_to_weights_kernel<4><<<grid, BLOCK, 0, stream>>>(
            d_weights_inout, d_unmix_matrices, active, n_freq, n_elem);
    } else if (active <= 8) {
        apply_bislc_to_weights_kernel<8><<<grid, BLOCK, 0, stream>>>(
            d_weights_inout, d_unmix_matrices, active, n_freq, n_elem);
    } else if (active <= 16) {
        apply_bislc_to_weights_kernel<16><<<grid, BLOCK, 0, stream>>>(
            d_weights_inout, d_unmix_matrices, active, n_freq, n_elem);
    } else {
        apply_bislc_to_weights_kernel<32><<<grid, BLOCK, 0, stream>>>(
            d_weights_inout, d_unmix_matrices, active, n_freq, n_elem);
    }
    CHECK_CUDA_ERROR_NON_OO(cudaGetLastError());
}

} // namespace kotekan
