#include "cudaDirectBeamTracker.hpp"
#include "cudaDirectBeamTracker.cuh"
#include "cudaUtils.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace kotekan {

std::vector<float2> generate_sky_grid_directions(float step) {
    std::vector<float2> grid;
    if (step <= 0.0f) step = 0.02f;

    const int n_steps = static_cast<int>(std::ceil(2.0f / step));
    for (int i = 0; i <= n_steps; ++i) {
        const float l = -1.0f + i * step;
        for (int j = 0; j <= n_steps; ++j) {
            const float m = -1.0f + j * step;
            if (l * l + m * m <= 1.0f) {
                grid.push_back(make_float2(l, m));
            }
        }
    }
    return grid;
}

int lookup_nearest_sky_grid(float l, float m, const std::vector<float2>& grid_directions) {
    if (grid_directions.empty()) return -1;

    float min_dist_sq = 1e9f;
    int best_idx = 0;

    for (std::size_t i = 0; i < grid_directions.size(); ++i) {
        const float dl = grid_directions[i].x - l;
        const float dm = grid_directions[i].y - m;
        const float dist_sq = dl * dl + dm * dm;
        if (dist_sq < min_dist_sq) {
            min_dist_sq = dist_sq;
            best_idx = static_cast<int>(i);
        }
    }
    return best_idx;
}

namespace {

__device__ __forceinline__ float2 complex_mul(float2 a, float2 b) {
    return make_float2(fmaf(-a.y, b.y, a.x * b.x), fmaf(a.x, b.y, a.y * b.x));
}

[[maybe_unused]] __device__ __forceinline__ float2 complex_pow_int(float2 z, std::size_t exp) {
    float2 res = make_float2(1.0f, 0.0f);
    float2 base = z;
    while (exp > 0) {
        if (exp & 1) {
            res = complex_mul(res, base);
        }
        base = complex_mul(base, base);
        exp >>= 1;
    }
    const float norm_sq = fmaf(res.x, res.x, res.y * res.y);
    if (norm_sq > 0.0f) {
        const float inv_r = rsqrtf(norm_sq);
        res.x *= inv_r;
        res.y *= inv_r;
    }
    return res;
}

template <bool SWAPPED_WITH_OFFSET = false>
__device__ __forceinline__ float2 unpack_int4_byte_fmt(std::uint32_t byte_val) {
    if constexpr (SWAPPED_WITH_OFFSET) {
#if defined(__CUDA_ARCH__)
        int r, i;
        asm("bfe.u32 %0, %1, 4, 4;" : "=r"(r) : "r"(byte_val));
        asm("bfe.u32 %0, %1, 0, 4;" : "=r"(i) : "r"(byte_val));
        return make_float2(__int2float_rn(r - 8), __int2float_rn(i - 8));
#else
        const int r = static_cast<int>((byte_val >> 4) & 0x0Fu) - 8;
        const int i = static_cast<int>(byte_val & 0x0Fu) - 8;
        return make_float2(static_cast<float>(r), static_cast<float>(i));
#endif
    } else {
#if defined(__CUDA_ARCH__)
        int r, i;
        asm("bfe.s32 %0, %1, 0, 4;" : "=r"(r) : "r"(byte_val));
        asm("bfe.s32 %0, %1, 4, 4;" : "=r"(i) : "r"(byte_val));
        return make_float2(__int2float_rn(r), __int2float_rn(i));
#else
        const int r = (static_cast<int>(byte_val) << 28) >> 28;
        const int i = (static_cast<int>(byte_val) << 24) >> 28;
        return make_float2(static_cast<float>(r), static_cast<float>(i));
#endif
    }
}

[[maybe_unused]] __device__ __forceinline__ float2 unpack_int4_byte(std::uint32_t byte_val) {
    return unpack_int4_byte_fmt<false>(byte_val);
}

template <int ELEM_PER_LANE, bool SWAPPED_WITH_OFFSET = false>
__device__ __forceinline__ void load_and_unpack_contiguous(
    const std::uint8_t* __restrict__ row_ptr,
    unsigned int lane,
    float2 (&out)[ELEM_PER_LANE]) {

    if constexpr (ELEM_PER_LANE == 1) {
        const std::uint32_t b = __ldg(&row_ptr[lane]);
        out[0] = unpack_int4_byte_fmt<SWAPPED_WITH_OFFSET>(b);
    } else if constexpr (ELEM_PER_LANE == 2) {
        const std::uint16_t packed2 = __ldg(reinterpret_cast<const std::uint16_t*>(row_ptr) + lane);
        out[0] = unpack_int4_byte_fmt<SWAPPED_WITH_OFFSET>(packed2 & 0xFFu);
        out[1] = unpack_int4_byte_fmt<SWAPPED_WITH_OFFSET>((packed2 >> 8) & 0xFFu);
    } else if constexpr (ELEM_PER_LANE == 4) {
        const std::uint32_t packed4 = __ldg(reinterpret_cast<const std::uint32_t*>(row_ptr) + lane);
        #pragma unroll
        for (int a = 0; a < 4; ++a) {
            out[a] = unpack_int4_byte_fmt<SWAPPED_WITH_OFFSET>((packed4 >> (a * 8)) & 0xFFu);
        }
    } else if constexpr (ELEM_PER_LANE == 8) {
        const uint2 packed8 = __ldg(reinterpret_cast<const uint2*>(row_ptr) + lane);
        #pragma unroll
        for (int a = 0; a < 4; ++a) {
            out[a]     = unpack_int4_byte_fmt<SWAPPED_WITH_OFFSET>((packed8.x >> (a * 8)) & 0xFFu);
            out[a + 4] = unpack_int4_byte_fmt<SWAPPED_WITH_OFFSET>((packed8.y >> (a * 8)) & 0xFFu);
        }
    } else if constexpr (ELEM_PER_LANE == 16) {
        const uint4 packed16 = __ldg(reinterpret_cast<const uint4*>(row_ptr) + lane);
        #pragma unroll
        for (int a = 0; a < 4; ++a) {
            out[a]      = unpack_int4_byte_fmt<SWAPPED_WITH_OFFSET>((packed16.x >> (a * 8)) & 0xFFu);
            out[a + 4]  = unpack_int4_byte_fmt<SWAPPED_WITH_OFFSET>((packed16.y >> (a * 8)) & 0xFFu);
            out[a + 8]  = unpack_int4_byte_fmt<SWAPPED_WITH_OFFSET>((packed16.z >> (a * 8)) & 0xFFu);
            out[a + 12] = unpack_int4_byte_fmt<SWAPPED_WITH_OFFSET>((packed16.w >> (a * 8)) & 0xFFu);
        }
    }
}

// ============================================================================
// Tapered & Mask-Normalized Weight Generation Kernels
// ============================================================================

__global__ void generate_steering_weights_tapered_kernel(
    float2* __restrict__ weights,
    float2* __restrict__ step_weights,
    const DirectDirection3D* __restrict__ directions_start,
    const DirectDirection3D* __restrict__ directions_end,
    const double* __restrict__ wavenumbers,
    const float3* __restrict__ antenna_positions,
    const float* __restrict__ norm_taper,
    const float2* __restrict__ calibration_gains,
    const std::size_t num_beams,
    const std::size_t n_freq,
    const std::size_t n_elem,
    const std::size_t n_time) {

    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total = num_beams * n_freq * n_elem;
    if (idx >= total) return;

    const std::size_t elem = idx % n_elem;
    const std::size_t rest = idx / n_elem;
    const std::size_t freq = rest % n_freq;
    const std::size_t beam = rest / n_freq;

    // norm_taper[elem] is 0.0f if masked, or (t_a / sqrt(sum_{k in pol} t_k^2)) if active
    const float amp = (norm_taper != nullptr) ? __ldg(&norm_taper[elem]) : 1.0f;
    if (amp <= 0.0f) {
        weights[idx] = make_float2(0.0f, 0.0f);
        if (step_weights != nullptr) {
            step_weights[idx] = make_float2(1.0f, 0.0f);
        }
        return;
    }

    const DirectDirection3D dir0 = directions_start[beam];
    const float3 pos = antenna_positions[elem];
    const double wave_num = __ldg(&wavenumbers[freq]);

    const double delay_m0 = static_cast<double>(pos.x) * dir0.x +
                            static_cast<double>(pos.y) * dir0.y +
                            static_cast<double>(pos.z) * dir0.z;
    const float phase0 = static_cast<float>(wave_num * delay_m0);
    float s0, c0;
    __sincosf(phase0, &s0, &c0);

    float wr = c0 * amp;
    float wi = s0 * amp;

    if (calibration_gains != nullptr) {
        const float2 g = __ldg(&calibration_gains[freq * n_elem + elem]);
        const float cal_r = fmaf(-wi, g.y, wr * g.x);
        const float cal_i = fmaf(wr, g.y, wi * g.x);
        wr = cal_r;
        wi = cal_i;
    }

    weights[idx] = make_float2(wr, wi);

    if (step_weights != nullptr) {
        if (directions_end != nullptr && n_time > 0) {
            const DirectDirection3D dir1 = directions_end[beam];
            const double delay_m1 = static_cast<double>(pos.x) * dir1.x +
                                    static_cast<double>(pos.y) * dir1.y +
                                    static_cast<double>(pos.z) * dir1.z;
            const double delta_delay = delay_m1 - delay_m0;
            const float delta_phase =
                static_cast<float>((wave_num * delta_delay) / static_cast<double>(n_time));
            float sd, cd;
            __sincosf(delta_phase, &sd, &cd);
            step_weights[idx] = make_float2(cd, sd);
        } else {
            step_weights[idx] = make_float2(1.0f, 0.0f);
        }
    }
}

/**
 * @brief Fused Weight-Space BiSLC Kernel:
 *        Computes A_{i,j}(f) = sum_a (norm_taper[a]^2 * exp(+j * k(f) * x_a . (s_i - s_j))),
 *        inverts M(f) = (A(f) + eps*I)^-1 in shared memory, and updates W(f) <- M(f) * W(f).
 *        Eliminates intra-block race conditions via s_factors staging.
 */
template <int MAX_B>
__global__ void __launch_bounds__(256)
fuse_weight_space_bislc_kernel(
    float2* __restrict__ weights,
    const DirectDirection3D* __restrict__ directions,
    const double* __restrict__ wavenumbers,
    const float3* __restrict__ antenna_positions,
    const float* __restrict__ norm_taper,
    const std::size_t B,
    const std::size_t n_freq,
    const std::size_t n_elem,
    const float diag_loading) {

    const std::size_t f = blockIdx.x;
    if (f >= n_freq || B <= 1 || B > MAX_B) return;

    const unsigned int tid = threadIdx.x;
    const double k = __ldg(&wavenumbers[f]);

    __shared__ float2 s_aug[MAX_B][2 * MAX_B];
    __shared__ std::size_t s_pivot_row;
    __shared__ float2 s_pivot_val;
    __shared__ bool s_singular;
    __shared__ float2 s_factors[MAX_B];

    if (tid == 0) s_singular = false;
    __syncthreads();

    const std::size_t total_pairs = B * B;
    for (std::size_t idx = tid; idx < total_pairs; idx += blockDim.x) {
        const std::size_t i = idx / B;
        const std::size_t j = idx % B;

        float2 a_val = make_float2(1.0f, 0.0f);
        if (i != j) {
            const DirectDirection3D di = directions[i];
            const DirectDirection3D dj = directions[j];
            const double dl = static_cast<double>(di.x - dj.x);
            const double dm = static_cast<double>(di.y - dj.y);
            const double dn = static_cast<double>(di.z - dj.z);

            float sum_r = 0.0f;
            float sum_i = 0.0f;
            for (std::size_t a = 0; a < n_elem; ++a) {
                const float t_a = (norm_taper != nullptr) ? __ldg(&norm_taper[a]) : 0.0f;
                if (t_a <= 0.0f) continue;
                const float w_pow = t_a * t_a;
                const float3 pos = antenna_positions[a];
                const double delay_m = static_cast<double>(pos.x) * dl +
                                       static_cast<double>(pos.y) * dm +
                                       static_cast<double>(pos.z) * dn;
                const float phase = static_cast<float>(k * delay_m);
                float s, c;
                __sincosf(phase, &s, &c);
                sum_r = fmaf(w_pow, c, sum_r);
                sum_i = fmaf(w_pow, s, sum_i);
            }
            a_val = make_float2(sum_r, sum_i);
        }
        s_aug[i][j] = (i == j) ? make_float2(a_val.x + diag_loading, a_val.y) : a_val;
        s_aug[i][B + j] = (i == j) ? make_float2(1.0f, 0.0f) : make_float2(0.0f, 0.0f);
    }
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
            if (max_norm < 1.0e-14f) s_singular = true;
        }
        __syncthreads();
        if (s_singular) return;

        if (s_pivot_row != col && tid < 2 * B) {
            const float2 tmp = s_aug[col][tid];
            s_aug[col][tid] = s_aug[s_pivot_row][tid];
            s_aug[s_pivot_row][tid] = tmp;
        }
        __syncthreads();

        if (tid == 0) s_pivot_val = s_aug[col][col];
        __syncthreads();

        if (tid < 2 * B) {
            const float2 num = s_aug[col][tid];
            const float inv_den =
                1.0f / (s_pivot_val.x * s_pivot_val.x + s_pivot_val.y * s_pivot_val.y);
            s_aug[col][tid] = make_float2(
                (num.x * s_pivot_val.x + num.y * s_pivot_val.y) * inv_den,
                (num.y * s_pivot_val.x - num.x * s_pivot_val.y) * inv_den);
        }
        __syncthreads();

        if (tid < B) {
            s_factors[tid] = s_aug[tid][col];
        }
        __syncthreads();

        for (std::size_t idx = tid; idx < B * (2 * B); idx += blockDim.x) {
            const std::size_t r = idx / (2 * B);
            const std::size_t c = idx % (2 * B);
            if (r != col && c >= col) {
                const float2 factor = s_factors[r];
                const float2 p_val = s_aug[col][c];
                s_aug[r][c].x -= (factor.x * p_val.x - factor.y * p_val.y);
                s_aug[r][c].y -= (factor.x * p_val.y + factor.y * p_val.x);
            }
        }
        __syncthreads();
    }

    const std::size_t fe_stride = n_freq * n_elem;
    for (std::size_t a = tid; a < n_elem; a += blockDim.x) {
        float2 w_orig[MAX_B];
        for (std::size_t b = 0; b < B; ++b) {
            w_orig[b] = weights[b * fe_stride + f * n_elem + a];
        }
        for (std::size_t i = 0; i < B; ++i) {
            float sr = 0.0f;
            float si = 0.0f;
            for (std::size_t j = 0; j < B; ++j) {
                const float2 mij = s_aug[i][B + j];
                const float2 wj = w_orig[j];
                sr = fmaf(mij.x, wj.x, fmaf(-mij.y, wj.y, sr));
                si = fmaf(mij.x, wj.y, fmaf(mij.y, wj.x, si));
            }
            weights[i * fe_stride + f * n_elem + a] = make_float2(sr, si);
        }
    }
}

__global__ void precompute_sky_grid_tapered_kernel(
    float2* __restrict__ grid_weights,
    const float2* __restrict__ grid_lms,
    const double* __restrict__ wavenumbers,
    const float3* __restrict__ antenna_positions,
    const float* __restrict__ norm_taper,
    const float2* __restrict__ calibration_gains,
    const std::size_t num_grid_points,
    const std::size_t n_freq,
    const std::size_t n_elem) {

    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total = num_grid_points * n_freq * n_elem;
    if (idx >= total) return;

    const std::size_t elem = idx % n_elem;
    const std::size_t rest = idx / n_elem;
    const std::size_t freq = rest % n_freq;
    const std::size_t grid_pt = rest / n_freq;

    const float amp = (norm_taper != nullptr) ? __ldg(&norm_taper[elem]) : 1.0f;
    if (amp <= 0.0f) {
        grid_weights[idx] = make_float2(0.0f, 0.0f);
        return;
    }

    const float2 lm = __ldg(&grid_lms[grid_pt]);
    const float l = lm.x;
    const float m = lm.y;
    const float trans_sq = l * l + m * m;
    const float n = (trans_sq <= 1.0f) ? sqrtf(1.0f - trans_sq) : 0.0f;

    const float3 pos = antenna_positions[elem];
    const double wave_num = __ldg(&wavenumbers[freq]);

    const double delay_m = static_cast<double>(pos.x) * l +
                           static_cast<double>(pos.y) * m +
                           static_cast<double>(pos.z) * n;
    const float phase = static_cast<float>(wave_num * delay_m);
    float s, c;
    __sincosf(phase, &s, &c);

    float wr = c * amp;
    float wi = s * amp;

    if (calibration_gains != nullptr) {
        const float2 g = __ldg(&calibration_gains[freq * n_elem + elem]);
        const float cal_r = fmaf(-wi, g.y, wr * g.x);
        const float cal_i = fmaf(wr, g.y, wi * g.x);
        wr = cal_r;
        wi = cal_i;
    }

    grid_weights[idx] = make_float2(wr, wi);
}

// Legacy untapered kernels for backwards compatibility
__global__ void generate_steering_weights_legacy_kernel(
    float2* __restrict__ weights,
    float2* __restrict__ step_weights,
    const DirectDirection3D* __restrict__ directions_start,
    const DirectDirection3D* __restrict__ directions_end,
    const double* __restrict__ wavenumbers,
    const float3* __restrict__ antenna_positions,
    const std::uint8_t* __restrict__ antenna_mask,
    const float2* __restrict__ calibration_gains,
    const std::size_t num_beams,
    const std::size_t n_freq,
    const std::size_t n_elem,
    const std::size_t n_time,
    const std::size_t n_active_per_pol) {

    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total = num_beams * n_freq * n_elem;
    if (idx >= total) return;

    const std::size_t elem = idx % n_elem;
    const std::size_t rest = idx / n_elem;
    const std::size_t freq = rest % n_freq;
    const std::size_t beam = rest / n_freq;

    if (antenna_mask != nullptr && __ldg(&antenna_mask[elem]) == 0) {
        weights[idx] = make_float2(0.0f, 0.0f);
        if (step_weights != nullptr) step_weights[idx] = make_float2(1.0f, 0.0f);
        return;
    }

    const DirectDirection3D dir0 = directions_start[beam];
    const float3 pos = antenna_positions[elem];
    const double wave_num = __ldg(&wavenumbers[freq]);

    const double delay_m0 = static_cast<double>(pos.x) * dir0.x +
                            static_cast<double>(pos.y) * dir0.y +
                            static_cast<double>(pos.z) * dir0.z;
    const float phase0 = static_cast<float>(wave_num * delay_m0);
    float s0, c0;
    __sincosf(phase0, &s0, &c0);

    const float norm = (n_active_per_pol > 0) ? rsqrtf(static_cast<float>(n_active_per_pol)) : 0.0f;
    float wr = c0 * norm;
    float wi = s0 * norm;

    if (calibration_gains != nullptr) {
        const float2 g = __ldg(&calibration_gains[freq * n_elem + elem]);
        const float cal_r = fmaf(-wi, g.y, wr * g.x);
        const float cal_i = fmaf(wr, g.y, wi * g.x);
        wr = cal_r;
        wi = cal_i;
    }
    weights[idx] = make_float2(wr, wi);

    if (step_weights != nullptr) {
        if (directions_end != nullptr && n_time > 0) {
            const DirectDirection3D dir1 = directions_end[beam];
            const double delay_m1 = static_cast<double>(pos.x) * dir1.x +
                                    static_cast<double>(pos.y) * dir1.y +
                                    static_cast<double>(pos.z) * dir1.z;
            const float delta_phase =
                static_cast<float>((wave_num * (delay_m1 - delay_m0)) / static_cast<double>(n_time));
            float sd, cd;
            __sincosf(delta_phase, &sd, &cd);
            step_weights[idx] = make_float2(cd, sd);
        } else {
            step_weights[idx] = make_float2(1.0f, 0.0f);
        }
    }
}

// ============================================================================
// Blackwell Multi-Beam & Dual-Pol Voltage Beamformer Kernel
// ============================================================================

template <
    int N_ELEM,
    int N_POL,
    int POL_LAYOUT,
    int B_TILE,
    int TIME_UNROLL,
    bool INTERPOLATE,
    bool COOPERATIVE_SMEM,
    bool SWAPPED_WITH_OFFSET = false>
__global__ void __launch_bounds__(128, 4)
direct_beamformer_blackwell_kernel(
    float2* __restrict__ voltages,
    const float2* __restrict__ weights,
    const float2* __restrict__ step_weights,
    const std::uint8_t* __restrict__ packed,
    const std::size_t n_time,
    const std::size_t n_freq,
    const std::size_t time_chunk_size,
    const std::size_t num_active_beams,
    const std::size_t max_beams_stride,
    const std::size_t total_work_units) {

    constexpr int WARPS_PER_BLOCK = 4;
    constexpr unsigned int ELEM_PER_LANE = static_cast<unsigned int>(N_ELEM / 32);
    constexpr unsigned int full_mask = 0xFFFFFFFFu;

    __shared__ float2 smem_voltages[COOPERATIVE_SMEM ? (TIME_UNROLL * N_ELEM) : 1];

    const unsigned int lane = threadIdx.x;
    const unsigned int warp_in_block = threadIdx.y;
    const unsigned int tid = warp_in_block * 32U + lane;

    std::size_t tile_idx = 0;
    std::size_t freq = 0;
    std::size_t chunk_idx = 0;

    if constexpr (COOPERATIVE_SMEM) {
        const std::size_t block_work_id = blockIdx.x;
        if (block_work_id >= total_work_units) return;

        constexpr std::size_t B_BLOCK = WARPS_PER_BLOCK * B_TILE;
        const std::size_t num_block_beam_tiles = (num_active_beams + B_BLOCK - 1) / B_BLOCK;
        const std::size_t block_beam_tile = block_work_id % num_block_beam_tiles;
        const std::size_t rest = block_work_id / num_block_beam_tiles;
        freq = rest % n_freq;
        chunk_idx = rest / n_freq;
        tile_idx = block_beam_tile * WARPS_PER_BLOCK + warp_in_block;
    } else {
        const std::size_t warp_id =
            static_cast<std::size_t>(blockIdx.x) * WARPS_PER_BLOCK + warp_in_block;
        if (warp_id >= total_work_units) return;

        const std::size_t num_beam_tiles = (num_active_beams + B_TILE - 1) / B_TILE;
        tile_idx = warp_id % num_beam_tiles;
        const std::size_t rest = warp_id / num_beam_tiles;
        freq = rest % n_freq;
        chunk_idx = rest / n_freq;
    }

    const std::size_t t_start = chunk_idx * time_chunk_size;
    if (t_start >= n_time) return;
    const std::size_t t_end = (t_start + time_chunk_size < n_time)
                                  ? (t_start + time_chunk_size)
                                  : n_time;

    const std::size_t b_base = tile_idx * B_TILE;
    const unsigned int active_in_tile =
        (b_base >= num_active_beams)
            ? 0U
            : ((b_base + B_TILE <= num_active_beams)
                   ? B_TILE
                   : static_cast<unsigned int>(num_active_beams - b_base));

    if (!COOPERATIVE_SMEM && active_in_tile == 0U) {
        return;
    }

    const unsigned int elem_base = lane * ELEM_PER_LANE;

    float w_r[B_TILE][ELEM_PER_LANE];
    float w_i[B_TILE][ELEM_PER_LANE];
    float dw_r[B_TILE][ELEM_PER_LANE];
    float dw_i[B_TILE][ELEM_PER_LANE];
    float w_amp[B_TILE][ELEM_PER_LANE];

    #pragma unroll
    for (int b = 0; b < B_TILE; ++b) {
        const std::size_t b_curr = b_base + b;
        const std::size_t w_offset = (b_curr < num_active_beams)
                                         ? ((b_curr * n_freq + freq) * N_ELEM + elem_base)
                                         : 0;
        #pragma unroll
        for (unsigned int a = 0; a < ELEM_PER_LANE; ++a) {
            if (b < active_in_tile) {
                const float2 w = __ldg(&weights[w_offset + a]);
                const float amp = hypotf(w.x, w.y);
                w_amp[b][a] = amp;
                if constexpr (INTERPOLATE) {
                    const float2 dw = __ldg(&step_weights[w_offset + a]);
                    if (amp > 1.0e-12f) {
                        const float phi0 = atan2f(w.y, w.x);
                        const float dphi = atan2f(dw.y, dw.x);
                        const double phi_exact_d = static_cast<double>(phi0) +
                                                  static_cast<double>(t_start) * static_cast<double>(dphi);
                        const double cycles = round(phi_exact_d * (1.0 / (2.0 * 3.14159265358979323846)));
                        const float phi_red = static_cast<float>(phi_exact_d - cycles * (2.0 * 3.14159265358979323846));
                        float s0, c0;
                        __sincosf(phi_red, &s0, &c0);
                        w_r[b][a] = c0 * amp;
                        w_i[b][a] = s0 * amp;
                        float sd, cd;
                        __sincosf(dphi, &sd, &cd);
                        dw_r[b][a] = cd;
                        dw_i[b][a] = sd;
                    } else {
                        w_r[b][a] = 0.0f;
                        w_i[b][a] = 0.0f;
                        dw_r[b][a] = 1.0f;
                        dw_i[b][a] = 0.0f;
                    }
                } else {
                    w_r[b][a] = w.x;
                    w_i[b][a] = w.y;
                }
            } else {
                w_amp[b][a] = 0.0f;
                w_r[b][a] = 0.0f;
                w_i[b][a] = 0.0f;
                if constexpr (INTERPOLATE) {
                    dw_r[b][a] = 1.0f;
                    dw_i[b][a] = 0.0f;
                }
            }
        }
    }

    const std::size_t t_stride = n_freq * N_ELEM;
    const std::size_t out_beam_pol_stride = max_beams_stride * N_POL;
    const std::size_t voltage_stride = n_freq * out_beam_pol_stride;

    const std::uint8_t* packed_ptr = packed + (t_start * n_freq + freq) * N_ELEM;
    float2* voltages_ptr = voltages + (t_start * n_freq + freq) * out_beam_pol_stride + b_base * N_POL;

    std::size_t t = t_start;

    for (; t + (TIME_UNROLL - 1) < t_end; t += TIME_UNROLL) {
        float s_r[B_TILE][TIME_UNROLL][N_POL] = {0.0f};
        float s_i[B_TILE][TIME_UNROLL][N_POL] = {0.0f};

        if constexpr (COOPERATIVE_SMEM) {
            constexpr int TOTAL_UNROLLED_ELEMS = TIME_UNROLL * N_ELEM;
            constexpr int BYTES_PER_THREAD = (TOTAL_UNROLLED_ELEMS >= 128) ? (TOTAL_UNROLLED_ELEMS / 128) : 1;

            #pragma unroll
            for (int c = 0; c < BYTES_PER_THREAD; ++c) {
                const int flat_idx = tid * BYTES_PER_THREAD + c;
                const int k_idx = flat_idx / N_ELEM;
                const int e_idx = flat_idx % N_ELEM;
                const std::uint32_t raw_b = __ldg(&packed_ptr[k_idx * t_stride + e_idx]);
                smem_voltages[flat_idx] = unpack_int4_byte_fmt<SWAPPED_WITH_OFFSET>(raw_b);
            }
            __syncthreads();
        }

        if (active_in_tile > 0U) {
            #pragma unroll
            for (int k = 0; k < TIME_UNROLL; ++k) {
                float2 p_vec[ELEM_PER_LANE];
                if constexpr (COOPERATIVE_SMEM) {
                    #pragma unroll
                    for (unsigned int a = 0; a < ELEM_PER_LANE; ++a) {
                        p_vec[a] = smem_voltages[k * N_ELEM + elem_base + a];
                    }
                } else {
                    load_and_unpack_contiguous<ELEM_PER_LANE, SWAPPED_WITH_OFFSET>(packed_ptr + k * t_stride, lane, p_vec);
                }

                #pragma unroll
                for (unsigned int a = 0; a < ELEM_PER_LANE; ++a) {
                    const float2 p = p_vec[a];
                    constexpr bool IS_INTERLEAVED_DUAL = (N_POL == 2 && POL_LAYOUT == 0);
                    constexpr bool IS_PLANAR_DUAL = (N_POL == 2 && POL_LAYOUT == 1);
                    const int pol = IS_INTERLEAVED_DUAL
                                        ? static_cast<int>(a & 1U)
                                        : (IS_PLANAR_DUAL ? static_cast<int>(lane >> 4U) : 0);

                    #pragma unroll
                    for (int b = 0; b < B_TILE; ++b) {
                        const float wr = w_r[b][a];
                        const float wi = w_i[b][a];

                        s_r[b][k][pol] = fmaf(wr, p.x, fmaf(-wi, p.y, s_r[b][k][pol]));
                        s_i[b][k][pol] = fmaf(wr, p.y, fmaf(wi, p.x, s_i[b][k][pol]));

                        if constexpr (INTERPOLATE) {
                            const float dwr = dw_r[b][a];
                            const float dwi = dw_i[b][a];
                            w_r[b][a] = fmaf(-wi, dwi, wr * dwr);
                            w_i[b][a] = fmaf(wr, dwi, wi * dwr);
                        }
                    }
                }
            }

            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                #pragma unroll
                for (int b = 0; b < B_TILE; ++b) {
                    #pragma unroll
                    for (int k = 0; k < TIME_UNROLL; ++k) {
                        #pragma unroll
                        for (int p = 0; p < N_POL; ++p) {
                            s_r[b][k][p] += __shfl_down_sync(full_mask, s_r[b][k][p], offset);
                            s_i[b][k][p] += __shfl_down_sync(full_mask, s_i[b][k][p], offset);
                        }
                    }
                }
            }

            if (lane == 0) {
                #pragma unroll
                for (int k = 0; k < TIME_UNROLL; ++k) {
                    #pragma unroll
                    for (int b = 0; b < B_TILE; ++b) {
                        if (b < active_in_tile) {
                            #pragma unroll
                            for (int p = 0; p < N_POL; ++p) {
                                voltages_ptr[k * voltage_stride + b * N_POL + p] =
                                    make_float2(s_r[b][k][p], s_i[b][k][p]);
                            }
                        }
                    }
                }
            }

            if constexpr (INTERPOLATE) {
                #pragma unroll
                for (int b = 0; b < B_TILE; ++b) {
                    #pragma unroll
                    for (unsigned int a = 0; a < ELEM_PER_LANE; ++a) {
                        const float amp = w_amp[b][a];
                        if (amp > 1.0e-12f) {
                            const float norm_sq = fmaf(w_r[b][a], w_r[b][a], w_i[b][a] * w_i[b][a]);
                            if (norm_sq > 0.0f) {
                                const float renorm = amp * rsqrtf(norm_sq);
                                w_r[b][a] *= renorm;
                                w_i[b][a] *= renorm;
                            }
                        }
                    }
                }
            }
        }

        if constexpr (COOPERATIVE_SMEM) {
            __syncthreads();
        }

        packed_ptr += TIME_UNROLL * t_stride;
        voltages_ptr += TIME_UNROLL * voltage_stride;
    }

    for (; t < t_end; ++t) {
        if (active_in_tile > 0U) {
            float s_r[B_TILE][N_POL] = {0.0f};
            float s_i[B_TILE][N_POL] = {0.0f};

            float2 p_vec[ELEM_PER_LANE];
            load_and_unpack_contiguous<ELEM_PER_LANE, SWAPPED_WITH_OFFSET>(packed_ptr, lane, p_vec);

            #pragma unroll
            for (unsigned int a = 0; a < ELEM_PER_LANE; ++a) {
                const float2 p = p_vec[a];
                constexpr bool IS_INTERLEAVED_DUAL = (N_POL == 2 && POL_LAYOUT == 0);
                constexpr bool IS_PLANAR_DUAL = (N_POL == 2 && POL_LAYOUT == 1);
                const int pol = IS_INTERLEAVED_DUAL
                                    ? static_cast<int>(a & 1U)
                                    : (IS_PLANAR_DUAL ? static_cast<int>(lane >> 4U) : 0);

                #pragma unroll
                for (int b = 0; b < B_TILE; ++b) {
                    const float wr = w_r[b][a];
                    const float wi = w_i[b][a];

                    s_r[b][pol] = fmaf(wr, p.x, fmaf(-wi, p.y, s_r[b][pol]));
                    s_i[b][pol] = fmaf(wr, p.y, fmaf(wi, p.x, s_i[b][pol]));

                    if constexpr (INTERPOLATE) {
                        const float dwr = dw_r[b][a];
                        const float dwi = dw_i[b][a];
                        w_r[b][a] = fmaf(-wi, dwi, wr * dwr);
                        w_i[b][a] = fmaf(wr, dwi, wi * dwr);
                    }
                }
            }

            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                #pragma unroll
                for (int b = 0; b < B_TILE; ++b) {
                    #pragma unroll
                    for (int p = 0; p < N_POL; ++p) {
                        s_r[b][p] += __shfl_down_sync(full_mask, s_r[b][p], offset);
                        s_i[b][p] += __shfl_down_sync(full_mask, s_i[b][p], offset);
                    }
                }
            }

            if (lane == 0) {
                #pragma unroll
                for (int b = 0; b < B_TILE; ++b) {
                    if (b < active_in_tile) {
                        #pragma unroll
                        for (int p = 0; p < N_POL; ++p) {
                            voltages_ptr[b * N_POL + p] = make_float2(s_r[b][p], s_i[b][p]);
                        }
                    }
                }
            }
        }

        packed_ptr += t_stride;
        voltages_ptr += voltage_stride;
    }
}

} // namespace

void set_l2_persisting_weights_policy(cudaStream_t stream, const void* ptr, std::size_t bytes) {
#if CUDART_VERSION >= 11000
    if (ptr == nullptr || bytes == 0) return;
    if (std::getenv("KOTEKAN_DISABLE_L2_POLICY") != nullptr) return;

    int dev_id = 0;
    if (cudaGetDevice(&dev_id) != cudaSuccess) {
        cudaGetLastError();
        return;
    }

    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, dev_id) != cudaSuccess) {
        cudaGetLastError();
        return;
    }

    if (prop.persistingL2CacheMaxSize > 0) {
        const std::size_t aligned_bytes = (bytes + 127) & ~static_cast<std::size_t>(127);
        const std::size_t window_size =
            std::min(aligned_bytes, static_cast<std::size_t>(prop.persistingL2CacheMaxSize));
        if (cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, window_size) != cudaSuccess) {
            cudaGetLastError();
            return;
        }

        cudaStreamAttrValue attr;
        std::memset(&attr, 0, sizeof(attr));
        attr.accessPolicyWindow.base_ptr = const_cast<void*>(ptr);
        attr.accessPolicyWindow.num_bytes = window_size;
        attr.accessPolicyWindow.hitRatio = 1.0f;
        attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
        attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;

        if (cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr) != cudaSuccess) {
            cudaGetLastError();
        }
    }
#else
    (void)stream; (void)ptr; (void)bytes;
#endif
}

void launch_direct_beamformer(
    const int4x2_t* d_packed,
    const float2* d_weights,
    const float2* d_step_weights,
    float2* d_voltages,
    std::size_t n_time,
    std::size_t n_freq,
    std::size_t n_ant,
    std::size_t num_active_beams,
    std::size_t max_beams_stride,
    std::size_t time_chunk_size,
    std::size_t time_unroll,
    std::size_t beam_tile_size,
    cudaStream_t stream,
    std::size_t n_pol,
    DirectPolLayout pol_layout,
    DirectInputFormat input_format) {

    if (num_active_beams == 0) return;

    const std::size_t n_elem = n_ant * std::max<std::size_t>(1, n_pol);
    const std::size_t active_beams = num_active_beams;
    const std::size_t stride_beams = std::max(max_beams_stride, active_beams);

    std::size_t b_tile = 1;
    if (beam_tile_size >= 4 && active_beams >= 4) {
        b_tile = (n_elem <= 128) ? 4 : 2;
    } else if (beam_tile_size >= 2 && active_beams >= 2) {
        b_tile = (n_elem <= 256) ? 2 : 1;
    }

    const bool use_coop_smem = (active_beams >= 2 * b_tile) && (n_elem >= 128);

    constexpr int WARPS_PER_BLOCK = 4;
    const std::size_t num_chunks = (n_time + time_chunk_size - 1) / time_chunk_size;

    std::size_t total_work_units = 0;
    unsigned int grid_dim = 0;

    if (use_coop_smem) {
        const std::size_t beams_per_block = WARPS_PER_BLOCK * b_tile;
        const std::size_t num_block_beam_tiles = (active_beams + beams_per_block - 1) / beams_per_block;
        total_work_units = num_chunks * n_freq * num_block_beam_tiles;
        grid_dim = static_cast<unsigned int>(total_work_units);
    } else {
        const std::size_t num_beam_tiles = (active_beams + b_tile - 1) / b_tile;
        total_work_units = num_chunks * n_freq * num_beam_tiles;
        grid_dim = static_cast<unsigned int>((total_work_units + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    }

    const dim3 block_dim(32, WARPS_PER_BLOCK);
    const auto* packed_bytes = reinterpret_cast<const std::uint8_t*>(d_packed);
    const bool interpolate = (d_step_weights != nullptr);

    auto launch_configured = [&](auto elem_tag, auto pol_tag, auto layout_tag,
                                 auto b_tile_tag, auto interp_tag, auto smem_tag, auto fmt_tag) {
        constexpr int N_E = decltype(elem_tag)::value;
        constexpr int N_P = decltype(pol_tag)::value;
        constexpr int L_O = decltype(layout_tag)::value;
        constexpr int B_T = decltype(b_tile_tag)::value;
        constexpr bool I_P = decltype(interp_tag)::value;
        constexpr bool C_S = decltype(smem_tag)::value;
        constexpr bool F_O = decltype(fmt_tag)::value;
        constexpr int T_U = (B_T == 4) ? 4 : 8;

        const int t_unroll_sel = (time_unroll >= 8 && B_T <= 2) ? 8 : 4;
        if (t_unroll_sel == 8 && T_U == 8) {
            direct_beamformer_blackwell_kernel<N_E, N_P, L_O, B_T, 8, I_P, C_S, F_O>
                <<<grid_dim, block_dim, 0, stream>>>(
                    d_voltages, d_weights, d_step_weights, packed_bytes, n_time, n_freq,
                    time_chunk_size, active_beams, stride_beams, total_work_units);
        } else {
            direct_beamformer_blackwell_kernel<N_E, N_P, L_O, B_T, 4, I_P, C_S, F_O>
                <<<grid_dim, block_dim, 0, stream>>>(
                    d_voltages, d_weights, d_step_weights, packed_bytes, n_time, n_freq,
                    time_chunk_size, active_beams, stride_beams, total_work_units);
        }
    };

    auto dispatch_fmt = [&](auto elem_tag, auto pol_tag, auto layout_tag, auto b_tile_tag, auto interp_tag, auto smem_tag) {
        if (input_format == DirectInputFormat::INT4X2_SWAPPED_WITHOFFSET) {
            launch_configured(elem_tag, pol_tag, layout_tag, b_tile_tag, interp_tag, smem_tag, std::true_type{});
        } else {
            launch_configured(elem_tag, pol_tag, layout_tag, b_tile_tag, interp_tag, smem_tag, std::false_type{});
        }
    };

    auto dispatch_smem = [&](auto elem_tag, auto pol_tag, auto layout_tag, auto b_tile_tag, auto interp_tag) {
        if (use_coop_smem) {
            dispatch_fmt(elem_tag, pol_tag, layout_tag, b_tile_tag, interp_tag, std::true_type{});
        } else {
            dispatch_fmt(elem_tag, pol_tag, layout_tag, b_tile_tag, interp_tag, std::false_type{});
        }
    };

    auto dispatch_interp = [&](auto elem_tag, auto pol_tag, auto layout_tag, auto b_tile_tag) {
        if (interpolate) {
            dispatch_smem(elem_tag, pol_tag, layout_tag, b_tile_tag, std::true_type{});
        } else {
            dispatch_smem(elem_tag, pol_tag, layout_tag, b_tile_tag, std::false_type{});
        }
    };

    auto dispatch_b_tile = [&](auto elem_tag, auto pol_tag, auto layout_tag) {
        if (b_tile == 4) dispatch_interp(elem_tag, pol_tag, layout_tag, std::integral_constant<int, 4>{});
        else if (b_tile == 2) dispatch_interp(elem_tag, pol_tag, layout_tag, std::integral_constant<int, 2>{});
        else dispatch_interp(elem_tag, pol_tag, layout_tag, std::integral_constant<int, 1>{});
    };

    auto dispatch_pol = [&](auto elem_tag) {
        if (n_pol == 2) {
            if (pol_layout == DirectPolLayout::PLANAR) {
                dispatch_b_tile(elem_tag, std::integral_constant<int, 2>{}, std::integral_constant<int, 1>{});
            } else {
                dispatch_b_tile(elem_tag, std::integral_constant<int, 2>{}, std::integral_constant<int, 0>{});
            }
        } else {
            dispatch_b_tile(elem_tag, std::integral_constant<int, 1>{}, std::integral_constant<int, 0>{});
        }
    };

    switch (n_elem) {
        case 32:  dispatch_pol(std::integral_constant<int, 32>{}); break;
        case 64:  dispatch_pol(std::integral_constant<int, 64>{}); break;
        case 128: dispatch_pol(std::integral_constant<int, 128>{}); break;
        case 256: dispatch_pol(std::integral_constant<int, 256>{}); break;
        case 512: dispatch_pol(std::integral_constant<int, 512>{}); break;
        default:
            throw std::invalid_argument(
                "Unsupported total elements (n_ant * n_pol): must be 32, 64, 128, 256, or 512");
    }

    CHECK_CUDA_ERROR_NON_OO(cudaGetLastError());
}

void launch_generate_steering_weights_tapered(
    float2* d_weights,
    float2* d_step_weights,
    const DirectDirection3D* d_directions_start,
    const DirectDirection3D* d_directions_end,
    const double* d_wavenumbers,
    const float3* d_antenna_positions,
    const float* d_norm_taper,
    const float2* d_calibration_gains,
    std::size_t num_beams,
    std::size_t n_freq,
    std::size_t n_elem,
    std::size_t n_time,
    bool enable_weight_bislc,
    float bislc_diag_loading,
    cudaStream_t stream) {

    const std::size_t total_weights = num_beams * n_freq * n_elem;
    if (total_weights == 0) return;

    constexpr int BLOCK_SIZE = 256;
    const unsigned int grid_size =
        static_cast<unsigned int>((total_weights + BLOCK_SIZE - 1) / BLOCK_SIZE);

    generate_steering_weights_tapered_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
        d_weights,
        d_step_weights,
        d_directions_start,
        d_directions_end,
        d_wavenumbers,
        d_antenna_positions,
        d_norm_taper,
        d_calibration_gains,
        num_beams,
        n_freq,
        n_elem,
        n_time);

    // Optional Fused Weight-Space BiSLC (zero-cost cross-beam sidelobe nulling)
    if (enable_weight_bislc && num_beams > 1 && num_beams <= 32) {
        const unsigned int f_grid = static_cast<unsigned int>(n_freq);
        if (num_beams <= 8) {
            fuse_weight_space_bislc_kernel<8><<<f_grid, 256, 0, stream>>>(
                d_weights, d_directions_start, d_wavenumbers,
                d_antenna_positions, d_norm_taper, num_beams, n_freq, n_elem, bislc_diag_loading);
        } else if (num_beams <= 16) {
            fuse_weight_space_bislc_kernel<16><<<f_grid, 256, 0, stream>>>(
                d_weights, d_directions_start, d_wavenumbers,
                d_antenna_positions, d_norm_taper, num_beams, n_freq, n_elem, bislc_diag_loading);
        } else {
            fuse_weight_space_bislc_kernel<32><<<f_grid, 256, 0, stream>>>(
                d_weights, d_directions_start, d_wavenumbers,
                d_antenna_positions, d_norm_taper, num_beams, n_freq, n_elem, bislc_diag_loading);
        }
    }
}

void launch_generate_steering_weights(
    float2* d_weights,
    float2* d_step_weights,
    const DirectDirection3D* d_directions_start,
    const DirectDirection3D* d_directions_end,
    const double* d_wavenumbers,
    const float3* d_antenna_positions,
    const std::uint8_t* d_antenna_mask,
    const float2* d_calibration_gains,
    std::size_t num_beams,
    std::size_t n_freq,
    std::size_t n_elem,
    std::size_t n_time,
    std::size_t n_active,
    cudaStream_t stream,
    std::size_t n_pol) {

    const std::size_t total_weights = num_beams * n_freq * n_elem;
    if (total_weights == 0) return;

    if (n_active == 0) n_active = n_elem;
    const std::size_t n_active_per_pol =
        std::max<std::size_t>(1, n_active / std::max<std::size_t>(1, n_pol));

    constexpr int BLOCK_SIZE = 256;
    const unsigned int grid_size =
        static_cast<unsigned int>((total_weights + BLOCK_SIZE - 1) / BLOCK_SIZE);

    generate_steering_weights_legacy_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
        d_weights, d_step_weights, d_directions_start, d_directions_end,
        d_wavenumbers, d_antenna_positions, d_antenna_mask, d_calibration_gains,
        num_beams, n_freq, n_elem, n_time, n_active_per_pol);
}

void launch_precompute_sky_grid_tapered(
    float2* d_grid_weights,
    const float2* d_grid_lms,
    const double* d_wavenumbers,
    const float3* d_antenna_positions,
    const float* d_norm_taper,
    const float2* d_calibration_gains,
    std::size_t num_grid_points,
    std::size_t n_freq,
    std::size_t n_elem,
    cudaStream_t stream) {

    const std::size_t total_weights = num_grid_points * n_freq * n_elem;
    if (total_weights == 0) return;

    constexpr int BLOCK_SIZE = 256;
    const unsigned int grid_size =
        static_cast<unsigned int>((total_weights + BLOCK_SIZE - 1) / BLOCK_SIZE);

    precompute_sky_grid_tapered_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
        d_grid_weights, d_grid_lms, d_wavenumbers, d_antenna_positions,
        d_norm_taper, d_calibration_gains, num_grid_points, n_freq, n_elem);
}

void launch_precompute_sky_grid(
    float2* d_grid_weights,
    const float2* d_grid_lms,
    const double* d_wavenumbers,
    const float3* d_antenna_positions,
    const std::uint8_t* /*d_antenna_mask*/,
    const float2* d_calibration_gains,
    std::size_t num_grid_points,
    std::size_t n_freq,
    std::size_t n_elem,
    std::size_t /*n_active*/,
    cudaStream_t stream,
    std::size_t /*n_pol*/) {

    launch_precompute_sky_grid_tapered(
        d_grid_weights, d_grid_lms, d_wavenumbers, d_antenna_positions,
        nullptr, d_calibration_gains, num_grid_points, n_freq, n_elem, stream);
}

template <int BLOCK_THREADS = 256>
__global__ void reduce_beam_power_kernel(
    const float2* __restrict__ voltages,
    float* __restrict__ power,
    std::size_t n_time,
    std::size_t n_freq,
    std::size_t active_beams,
    std::size_t max_beams_stride,
    std::size_t n_pol)
{
    const std::size_t b = blockIdx.x; // Beam index: 0 .. max_beams_stride - 1
    const std::size_t f = blockIdx.y; // Frequency index: 0 .. n_freq - 1
    if (b >= max_beams_stride || f >= n_freq) return;

    if (b >= active_beams) {
        if (threadIdx.x == 0) {
            power[b * n_freq + f] = 0.0f;
        }
        return;
    }

    const std::size_t beam_pol_stride = max_beams_stride * n_pol;
    const std::size_t tf_base = f * beam_pol_stride + b * n_pol;
    const std::size_t t_stride = n_freq * beam_pol_stride;

    float thread_sum = 0.0f;
    for (std::size_t t = threadIdx.x; t < n_time; t += blockDim.x) {
        const float2* v_ptr = voltages + t * t_stride + tf_base;
        #pragma unroll
        for (std::size_t p = 0; p < n_pol; ++p) {
            float2 v = v_ptr[p];
            thread_sum = fmaf(v.x, v.x, fmaf(v.y, v.y, thread_sum));
        }
    }

    // 1. Warp shuffle down reduction across 32 threads in warp
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }

    // 2. Shared memory reduction across warps (256 threads / 32 = 8 warps)
    constexpr int NUM_WARPS = BLOCK_THREADS / 32;
    __shared__ float s_warp_sums[NUM_WARPS];

    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;

    if (lane == 0) {
        s_warp_sums[warp_id] = thread_sum;
    }
    __syncthreads();

    // 3. First warp sums the 8 warp totals
    if (warp_id == 0) {
        float block_sum = (lane < NUM_WARPS) ? s_warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int offset = NUM_WARPS / 2; offset > 0; offset >>= 1) {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
        }
        if (lane == 0) {
            const float inv_time = (n_time > 0) ? (1.0f / static_cast<float>(n_time)) : 0.0f;
            power[b * n_freq + f] = block_sum * inv_time;
        }
    }
}

void launch_reduce_beam_power(
    const float2* d_voltages,
    float* d_power,
    std::size_t n_time,
    std::size_t n_freq,
    std::size_t active_beams,
    std::size_t max_beams_stride,
    cudaStream_t stream,
    std::size_t n_pol)
{
    if (max_beams_stride == 0 || n_freq == 0 || d_voltages == nullptr || d_power == nullptr) return;
    const dim3 grid(static_cast<unsigned int>(max_beams_stride), static_cast<unsigned int>(n_freq));
    constexpr int BLOCK_THREADS = 256;
    reduce_beam_power_kernel<BLOCK_THREADS><<<grid, BLOCK_THREADS, 0, stream>>>(
        d_voltages, d_power, n_time, n_freq, active_beams, max_beams_stride, n_pol);
}

} // namespace kotekan
