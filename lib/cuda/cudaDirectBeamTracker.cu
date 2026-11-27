#include "cudaDirectBeamTracker.hpp"
#include "cudaDirectBeamTracker.cuh"
#include "cudaUtils.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
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
    return build_sky_grid_index(step).directions;
}

SkyGridIndex build_sky_grid_index(float step) {
    SkyGridIndex idx;
    if (step <= 0.0f) step = 0.02f;
    idx.step = step;
    idx.n_steps = static_cast<int>(std::ceil(2.0f / step));
    const int side = idx.n_steps + 1;
    idx.offset_lut.assign(static_cast<std::size_t>(side) * side, -1);

    for (int i = 0; i <= idx.n_steps; ++i) {
        const float l = -1.0f + i * step;
        for (int j = 0; j <= idx.n_steps; ++j) {
            const float m = -1.0f + j * step;
            if (l * l + m * m <= 1.0f) {
                idx.offset_lut[static_cast<std::size_t>(i) * side + j] =
                    static_cast<int>(idx.directions.size());
                idx.directions.push_back(make_float2(l, m));
            }
        }
    }
    return idx;
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

int lookup_nearest_sky_grid_o1(float l, float m, const SkyGridIndex& grid_index) {
    if (grid_index.directions.empty() || grid_index.step <= 0.0f) return -1;

    const int n_steps = grid_index.n_steps;
    const int side = n_steps + 1;
    const int i0 = std::max(0, std::min(n_steps, static_cast<int>(std::lround((l + 1.0f) / grid_index.step))));
    const int j0 = std::max(0, std::min(n_steps, static_cast<int>(std::lround((m + 1.0f) / grid_index.step))));

    // When (l, m) is inside the unit disk and its rounded cell is valid, the global nearest
    // grid point is strictly within the 3x3 neighborhood (dist <= step/sqrt(2) < 1.5*step).
    if (l * l + m * m <= 1.0f && grid_index.offset_lut[static_cast<std::size_t>(i0) * side + j0] >= 0) {
        float min_dist_sq = 1e9f;
        int best_idx = -1;

        for (int di = -1; di <= 1; ++di) {
            const int ii = i0 + di;
            if (ii < 0 || ii > n_steps) continue;
            for (int dj = -1; dj <= 1; ++dj) {
                const int jj = j0 + dj;
                if (jj < 0 || jj > n_steps) continue;
                const int g_idx = grid_index.offset_lut[static_cast<std::size_t>(ii) * side + jj];
                if (g_idx < 0) continue;
                const float2 pt = grid_index.directions[static_cast<std::size_t>(g_idx)];
                const float dl = pt.x - l;
                const float dm = pt.y - m;
                const float d2 = dl * dl + dm * dm;
                if (d2 < min_dist_sq || (d2 == min_dist_sq && g_idx < best_idx)) {
                    min_dist_sq = d2;
                    best_idx = g_idx;
                }
            }
        }
        if (best_idx >= 0) {
            return best_idx;
        }
    }

    return lookup_nearest_sky_grid(l, m, grid_index.directions);
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
    const std::size_t n_time,
    const std::uint64_t beam_calibration_mask) {

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

    const bool apply_cal =
        (calibration_gains != nullptr)
        && (beam >= 64 || ((beam_calibration_mask >> beam) & 1ULL) != 0ULL);
    if (apply_cal) {
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

template <int MAX_B>
__device__ void build_and_invert_bislc_matrix(
    float2 (&s_aug)[MAX_B][2 * MAX_B],
    float2 (&s_factors)[MAX_B],
    std::size_t& s_pivot_row,
    float2& s_pivot_val,
    bool& s_singular,
    const DirectDirection3D* __restrict__ directions,
    const double k,
    const float3* __restrict__ antenna_positions,
    const float* __restrict__ norm_taper,
    const float inv_diag,
    const std::size_t B,
    const std::size_t n_elem,
    const float diag_loading,
    const std::uint64_t beam_calibration_mask,
    const unsigned int tid) {

    if (tid == 0) {
        s_singular = false;
    }
    __syncthreads();

    const std::size_t total_pairs = B * B;
    for (std::size_t idx = tid; idx < total_pairs; idx += blockDim.x) {
        const std::size_t i = idx / B;
        const std::size_t j = idx % B;
        const bool cal_i = (i >= 64) || (((beam_calibration_mask >> i) & 1ULL) != 0ULL);
        const bool cal_j = (j >= 64) || (((beam_calibration_mask >> j) & 1ULL) != 0ULL);

        float2 a_val = make_float2(1.0f, 0.0f);
        if (i != j) {
            if (!cal_i || !cal_j) {
                a_val = make_float2(0.0f, 0.0f);
            } else {
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
                    const float3 pos = antenna_positions[a];
                    const double delay_m = static_cast<double>(pos.x) * dl +
                                           static_cast<double>(pos.y) * dm +
                                           static_cast<double>(pos.z) * dn;
                    const float phase = static_cast<float>(k * delay_m);
                    float s, c;
                    __sincosf(phase, &s, &c);
                    sum_r = fmaf(t_a, c, sum_r);
                    sum_i = fmaf(t_a, s, sum_i);
                }
                a_val = make_float2(sum_r * inv_diag, sum_i * inv_diag);
            }
        }
        s_aug[i][j] = (i == j && cal_i) ? make_float2(a_val.x + diag_loading, a_val.y) : a_val;
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
}

/**
 * @brief Fused Weight-Space BiSLC Kernel:
 *        Computes unit-diagonal normalized A_{i,j}(f), inverts M(f) = (A(f) + eps*I)^-1
 *        excluding uncalibrated control beams, updates W(f) <- M(f) * W(f), and when
 *        sub-frame motion is active updates step_weights from (W_clean_1 / W_clean_0)^(1/n_time).
 */
template <int MAX_B>
__global__ void __launch_bounds__(256)
fuse_weight_space_bislc_kernel(
    float2* __restrict__ weights,
    float2* __restrict__ step_weights,
    const DirectDirection3D* __restrict__ directions_start,
    const DirectDirection3D* __restrict__ directions_end,
    const double* __restrict__ wavenumbers,
    const float3* __restrict__ antenna_positions,
    const float* __restrict__ norm_taper,
    const float2* __restrict__ calibration_gains,
    const std::size_t B,
    const std::size_t n_freq,
    const std::size_t n_elem,
    const std::size_t n_time,
    const float diag_loading,
    const std::uint64_t beam_calibration_mask) {

    const std::size_t f = blockIdx.x;
    if (f >= n_freq || B <= 1 || B > MAX_B) return;

    const unsigned int tid = threadIdx.x;
    const double k = __ldg(&wavenumbers[f]);

    __shared__ float2 s_aug[MAX_B][2 * MAX_B];
    __shared__ std::size_t s_pivot_row;
    __shared__ float2 s_pivot_val;
    __shared__ bool s_singular;
    __shared__ float2 s_factors[MAX_B];
    __shared__ float s_diag_pow;
    __shared__ bool s_any_moving;

    if (tid == 0) {
        if (norm_taper != nullptr) {
            float diag_sum = 0.0f;
            for (std::size_t a = 0; a < n_elem; ++a) {
                const float t_a = __ldg(&norm_taper[a]);
                if (t_a > 0.0f) {
                    diag_sum += t_a;
                }
            }
            s_diag_pow = (diag_sum > 0.0f) ? diag_sum : 1.0f;
        } else {
            s_diag_pow = 1.0f;
        }

        bool moving = false;
        if (step_weights != nullptr && directions_end != nullptr && n_time > 0) {
            for (std::size_t b = 0; b < B; ++b) {
                const DirectDirection3D d0 = directions_start[b];
                const DirectDirection3D d1 = directions_end[b];
                if (fabsf(d1.x - d0.x) > 1.0e-7f || fabsf(d1.y - d0.y) > 1.0e-7f
                    || fabsf(d1.z - d0.z) > 1.0e-7f) {
                    moving = true;
                    break;
                }
            }
        }
        s_any_moving = moving;
    }
    __syncthreads();

    const float inv_diag = (s_diag_pow > 0.0f) ? (1.0f / s_diag_pow) : 0.0f;

    build_and_invert_bislc_matrix<MAX_B>(
        s_aug, s_factors, s_pivot_row, s_pivot_val, s_singular,
        directions_start, k, antenna_positions, norm_taper, inv_diag,
        B, n_elem, diag_loading, beam_calibration_mask, tid);
    if (s_singular) return;

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
    __syncthreads();

    if (!s_any_moving) return;

    // Recompute end-of-frame M_1(f) and W_clean_1 = M_1 * W_1 so BiSLC nulls stay stationary
    build_and_invert_bislc_matrix<MAX_B>(
        s_aug, s_factors, s_pivot_row, s_pivot_val, s_singular,
        directions_end, k, antenna_positions, norm_taper, inv_diag,
        B, n_elem, diag_loading, beam_calibration_mask, tid);

    const float inv_n_time = 1.0f / static_cast<float>(n_time);
    for (std::size_t a = tid; a < n_elem; a += blockDim.x) {
        if (s_singular) {
            for (std::size_t i = 0; i < B; ++i) {
                step_weights[i * fe_stride + f * n_elem + a] = make_float2(1.0f, 0.0f);
            }
            continue;
        }
        const float amp = (norm_taper != nullptr) ? __ldg(&norm_taper[a]) : 1.0f;
        const float3 pos = antenna_positions[a];
        const float2 cal_g = (calibration_gains != nullptr)
                                 ? __ldg(&calibration_gains[f * n_elem + a])
                                 : make_float2(1.0f, 0.0f);

        float2 w_end[MAX_B];
        for (std::size_t b = 0; b < B; ++b) {
            if (amp <= 0.0f) {
                w_end[b] = make_float2(0.0f, 0.0f);
                continue;
            }
            const DirectDirection3D dir1 = directions_end[b];
            const double delay_m1 = static_cast<double>(pos.x) * dir1.x +
                                    static_cast<double>(pos.y) * dir1.y +
                                    static_cast<double>(pos.z) * dir1.z;
            const float phase1 = static_cast<float>(k * delay_m1);
            float s1, c1;
            __sincosf(phase1, &s1, &c1);
            float wr = c1 * amp;
            float wi = s1 * amp;
            const bool apply_cal =
                (calibration_gains != nullptr)
                && (b >= 64 || ((beam_calibration_mask >> b) & 1ULL) != 0ULL);
            if (apply_cal) {
                const float cr = fmaf(-wi, cal_g.y, wr * cal_g.x);
                const float ci = fmaf(wr, cal_g.y, wi * cal_g.x);
                wr = cr;
                wi = ci;
            }
            w_end[b] = make_float2(wr, wi);
        }

        for (std::size_t i = 0; i < B; ++i) {
            float sr = 0.0f;
            float si = 0.0f;
            for (std::size_t j = 0; j < B; ++j) {
                const float2 mij = s_aug[i][B + j];
                const float2 wj = w_end[j];
                sr = fmaf(mij.x, wj.x, fmaf(-mij.y, wj.y, sr));
                si = fmaf(mij.x, wj.y, fmaf(mij.y, wj.x, si));
            }
            const float2 w0 = weights[i * fe_stride + f * n_elem + a];
            const float a0 = hypotf(w0.x, w0.y);
            const float a1 = hypotf(sr, si);
            if (a0 > 1.0e-6f && a1 > 1.0e-6f && fabsf(a1 - a0) <= 0.25f * fmaxf(a0, a1)) {
                const float cross_re = fmaf(sr, w0.x, si * w0.y);
                const float cross_im = fmaf(si, w0.x, -sr * w0.y);
                const float dphi = atan2f(cross_im, cross_re) * inv_n_time;
                float sd, cd;
                __sincosf(dphi, &sd, &cd);
                step_weights[i * fe_stride + f * n_elem + a] = make_float2(cd, sd);
            } else {
                step_weights[i * fe_stride + f * n_elem + a] = make_float2(1.0f, 0.0f);
            }
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

template <
    int N_ELEM,
    int N_POL,
    int POL_LAYOUT,
    int B_TILE,
    bool INTERPOLATE,
    bool SWAPPED_WITH_OFFSET = false>
__global__ void __launch_bounds__(128, 4)
direct_beamformer_power_only_kernel(
    float* __restrict__ power,
    const float2* __restrict__ weights,
    const float2* __restrict__ step_weights,
    const std::uint8_t* __restrict__ packed,
    const std::uint8_t* __restrict__ d_rfi_mask,
    const std::size_t n_time,
    const std::size_t n_freq,
    const std::size_t time_chunk_size,
    const std::size_t n_chunks,
    const std::size_t num_active_beams,
    const std::size_t max_beams_stride) {

    constexpr int WARPS_PER_BLOCK = 4;
    constexpr int TIME_UNROLL = 4;
    constexpr unsigned int ELEM_PER_LANE = static_cast<unsigned int>(N_ELEM / 32);
    constexpr unsigned int full_mask = 0xFFFFFFFFu;

    const unsigned int lane = threadIdx.x;
    const unsigned int warp_in_block = threadIdx.y;

    const std::size_t tile_idx = blockIdx.x;
    const std::size_t freq = blockIdx.y;
    const std::size_t chunk_idx = blockIdx.z;
    if (freq >= n_freq) return;

    const bool use_mask = (d_rfi_mask != nullptr && n_chunks > 0 && time_chunk_size > 0);
    if (use_mask && d_rfi_mask[chunk_idx * n_freq + freq] != 0) {
        return;
    }

    const std::size_t c_start = chunk_idx * time_chunk_size;
    if (c_start >= n_time) return;
    const std::size_t c_end = (c_start + time_chunk_size < n_time)
                                  ? (c_start + time_chunk_size)
                                  : n_time;

    const std::size_t c_len = c_end - c_start;
    const std::size_t per_warp = (c_len + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const std::size_t t_start = c_start + static_cast<std::size_t>(warp_in_block) * per_warp;
    if (t_start >= c_end) return;
    const std::size_t t_end = (t_start + per_warp < c_end) ? (t_start + per_warp) : c_end;

    const std::size_t b_base = tile_idx * B_TILE;
    const unsigned int active_in_tile =
        (b_base >= num_active_beams)
            ? 0U
            : ((b_base + B_TILE <= num_active_beams)
                   ? B_TILE
                   : static_cast<unsigned int>(num_active_beams - b_base));
    if (active_in_tile == 0U) return;

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
    const std::uint8_t* packed_ptr = packed + (t_start * n_freq + freq) * N_ELEM;
    float p_acc[B_TILE] = {0.0f};
    std::size_t t = t_start;

    for (; t + (TIME_UNROLL - 1) < t_end; t += TIME_UNROLL) {
        float s_r[B_TILE][TIME_UNROLL][N_POL] = {0.0f};
        float s_i[B_TILE][TIME_UNROLL][N_POL] = {0.0f};

        #pragma unroll
        for (int k = 0; k < TIME_UNROLL; ++k) {
            float2 p_vec[ELEM_PER_LANE];
            load_and_unpack_contiguous<ELEM_PER_LANE, SWAPPED_WITH_OFFSET>(
                packed_ptr + k * t_stride, lane, p_vec);

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
                            p_acc[b] = fmaf(s_r[b][k][p], s_r[b][k][p],
                                            fmaf(s_i[b][k][p], s_i[b][k][p], p_acc[b]));
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

        packed_ptr += TIME_UNROLL * t_stride;
    }

    for (; t < t_end; ++t) {
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
                        p_acc[b] = fmaf(s_r[b][p], s_r[b][p],
                                        fmaf(s_i[b][p], s_i[b][p], p_acc[b]));
                    }
                }
            }
        }

        packed_ptr += t_stride;
    }

    if (lane == 0) {
        float inv_valid = 0.0f;
        if (use_mask) {
            std::size_t valid_samples = 0;
            for (std::size_t c = 0; c < n_chunks; ++c) {
                if (d_rfi_mask[c * n_freq + freq] == 0) {
                    const std::size_t cs = c * time_chunk_size;
                    const std::size_t ce = (cs + time_chunk_size < n_time) ? (cs + time_chunk_size) : n_time;
                    valid_samples += (ce > cs) ? (ce - cs) : 0;
                }
            }
            inv_valid = (valid_samples > 0) ? (1.0f / static_cast<float>(valid_samples)) : 0.0f;
        } else {
            inv_valid = (n_time > 0) ? (1.0f / static_cast<float>(n_time)) : 0.0f;
        }
        if (inv_valid > 0.0f) {
            #pragma unroll
            for (int b = 0; b < B_TILE; ++b) {
                if (b < active_in_tile && p_acc[b] > 0.0f) {
                    atomicAdd(&power[(b_base + b) * n_freq + freq], p_acc[b] * inv_valid);
                }
            }
        }
    }
}

} // namespace

// -----------------------------------------------------------------------------
// GPU-4.2: Blackwell Tensor Core FP16->FP32 WMMA Beamformer (N_ELEM=64, B>=16)
// -----------------------------------------------------------------------------
namespace {

template <bool SWAPPED_WITH_OFFSET>
__global__ void __launch_bounds__(256, 2)
direct_beamformer_wmma_kernel(
    float2* __restrict__ voltages,
    const float2* __restrict__ weights,
    const std::uint8_t* __restrict__ packed,
    const std::size_t n_time,
    const std::size_t n_freq,
    const std::size_t active_beams,
    const std::size_t max_beams_stride) {

    constexpr int T_CHUNK = 256;
    constexpr int T_TILE = 32;
    constexpr int B_TILE = 64;
    constexpr int N_ELEM = 64;
    constexpr int LD_K = 72; // 64 + 8 padding avoids bank conflicts in 16x16 WMMA loads

    const int tid = static_cast<int>(threadIdx.x);
    const int warp_id = tid >> 5;
    const int w_t = warp_id >> 2; // 0..1 -> 32 time rows
    const int w_b = warp_id & 3;  // 0..3 -> 64 beam cols

    const std::size_t t_chunk_start = static_cast<std::size_t>(blockIdx.x) * T_CHUNK;
    const std::size_t f = static_cast<std::size_t>(blockIdx.y);
    const std::size_t b0 = static_cast<std::size_t>(blockIdx.z) * B_TILE;
    if (t_chunk_start >= n_time || f >= n_freq || b0 >= active_beams) return;

    __shared__ struct {
        __half wr[B_TILE][LD_K];
        __half wi[B_TILE][LD_K];
        __half wi_neg[B_TILE][LD_K];
    } smem_w;

    __shared__ union {
        struct {
            __half er[T_TILE][LD_K];
            __half ei[T_TILE][LD_K];
        } in;
        struct {
            float r[T_TILE][B_TILE];
            float i[T_TILE][B_TILE];
        } out;
    } smem_io;

    // 1. Load [64 x 64] complex weights once per T_CHUNK=256 block (16 items/thread)
    #pragma unroll
    for (int iter = 0; iter < 16; ++iter) {
        const int flat = tid + iter * 256;
        const int lb = flat >> 6;
        const int a = flat & 63;
        const std::size_t bg = b0 + static_cast<std::size_t>(lb);
        if (bg < active_beams) {
            const float2 w = __ldg(&weights[(bg * n_freq + f) * N_ELEM + a]);
            smem_w.wr[lb][a] = __float2half(w.x);
            smem_w.wi[lb][a] = __float2half(w.y);
            smem_w.wi_neg[lb][a] = __float2half(-w.y);
        } else {
            smem_w.wr[lb][a] = __float2half(0.0f);
            smem_w.wi[lb][a] = __float2half(0.0f);
            smem_w.wi_neg[lb][a] = __float2half(0.0f);
        }
    }
    __syncthreads();

    const std::size_t t_stride = n_freq * N_ELEM;
    const std::size_t out_row_stride = n_freq * max_beams_stride;
    const std::size_t t_chunk_end =
        (t_chunk_start + T_CHUNK < n_time) ? (t_chunk_start + T_CHUNK) : n_time;

    for (std::size_t t0 = t_chunk_start; t0 < t_chunk_end; t0 += T_TILE) {
        // 2. Coalesced unpack of [32 x 64] int4 voltages into FP16 shared buffers (8 items/thread)
        #pragma unroll
        for (int iter = 0; iter < 8; ++iter) {
            const int flat = tid + iter * 256;
            const int lt = flat >> 6;
            const int a = flat & 63;
            const std::size_t tg = t0 + static_cast<std::size_t>(lt);
            if (tg < n_time) {
                const float2 pv = unpack_int4_byte_fmt<SWAPPED_WITH_OFFSET>(
                    __ldg(&packed[tg * t_stride + f * N_ELEM + a]));
                smem_io.in.er[lt][a] = __float2half(pv.x);
                smem_io.in.ei[lt][a] = __float2half(pv.y);
            } else {
                smem_io.in.er[lt][a] = __float2half(0.0f);
                smem_io.in.ei[lt][a] = __float2half(0.0f);
            }
        }
        __syncthreads();

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
        using namespace nvcuda;
        const int t_off = w_t * 16;
        const int b_off = w_b * 16;

        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_r;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_i;
        wmma::fill_fragment(acc_r, 0.0f);
        wmma::fill_fragment(acc_i, 0.0f);

        #pragma unroll
        for (int k = 0; k < N_ELEM; k += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_er;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_ei;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_wr;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_wi;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b_wi_neg;

            wmma::load_matrix_sync(a_er, &smem_io.in.er[t_off][k], LD_K);
            wmma::load_matrix_sync(a_ei, &smem_io.in.ei[t_off][k], LD_K);
            wmma::load_matrix_sync(b_wr, &smem_w.wr[b_off][k], LD_K);
            wmma::load_matrix_sync(b_wi, &smem_w.wi[b_off][k], LD_K);
            wmma::load_matrix_sync(b_wi_neg, &smem_w.wi_neg[b_off][k], LD_K);

            wmma::mma_sync(acc_r, a_er, b_wr, acc_r);
            wmma::mma_sync(acc_r, a_ei, b_wi_neg, acc_r);
            wmma::mma_sync(acc_i, a_er, b_wi, acc_i);
            wmma::mma_sync(acc_i, a_ei, b_wr, acc_i);
        }

        __syncthreads();

        wmma::store_matrix_sync(&smem_io.out.r[t_off][b_off], acc_r, B_TILE, wmma::mem_row_major);
        wmma::store_matrix_sync(&smem_io.out.i[t_off][b_off], acc_i, B_TILE, wmma::mem_row_major);
        __syncthreads();

        // 3. Coalesced store of formed float2 beam voltages
        #pragma unroll
        for (int iter = 0; iter < 8; ++iter) {
            const int flat = tid + iter * 256;
            const int lt = flat >> 6;
            const int lb = flat & 63;
            const std::size_t tg = t0 + static_cast<std::size_t>(lt);
            const std::size_t bg = b0 + static_cast<std::size_t>(lb);
            if (tg < n_time && bg < active_beams) {
                voltages[(tg * out_row_stride) + f * max_beams_stride + bg] =
                    make_float2(smem_io.out.r[lt][lb], smem_io.out.i[lt][lb]);
            }
        }
#else
        // Pre-Volta scalar fallback (writes directly to voltages without aliasing smem_io.out)
        if (tid < T_TILE) {
            const int lt = tid;
            const std::size_t tg = t0 + static_cast<std::size_t>(lt);
            if (tg < n_time) {
                for (int lb = 0; lb < B_TILE; ++lb) {
                    const std::size_t bg = b0 + static_cast<std::size_t>(lb);
                    if (bg < active_beams) {
                        float sr = 0.0f, si = 0.0f;
                        for (int a = 0; a < N_ELEM; ++a) {
                            const float ex = __half2float(smem_io.in.er[lt][a]);
                            const float ey = __half2float(smem_io.in.ei[lt][a]);
                            const float wx = __half2float(smem_w.wr[lb][a]);
                            const float wy = __half2float(smem_w.wi[lb][a]);
                            sr = fmaf(wx, ex, fmaf(-wy, ey, sr));
                            si = fmaf(wx, ey, fmaf(wy, ex, si));
                        }
                        voltages[(tg * out_row_stride) + f * max_beams_stride + bg] =
                            make_float2(sr, si);
                    }
                }
            }
        }
#endif
        __syncthreads();
    }
}

// -----------------------------------------------------------------------------
// GPU-4.3: Separable 2D All-Sky FFT (8x8 / 16x16) & Factorized 2D DFT Kernels
// -----------------------------------------------------------------------------

template <int N_FFT, bool INVERSE>
__device__ __forceinline__ void fft_1d_radix2_in_reg(float2 x[N_FFT]) {
    constexpr float TWO_PI_F = 6.2831853071795864769f;
    // Bit-reversal permutation
    if constexpr (N_FFT == 8) {
        constexpr int perm[8] = {0, 4, 2, 6, 1, 5, 3, 7};
        float2 tmp[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) tmp[i] = x[perm[i]];
        #pragma unroll
        for (int i = 0; i < 8; ++i) x[i] = tmp[i];
    } else {
        constexpr int perm[16] = {0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15};
        float2 tmp[16];
        #pragma unroll
        for (int i = 0; i < 16; ++i) tmp[i] = x[perm[i]];
        #pragma unroll
        for (int i = 0; i < 16; ++i) x[i] = tmp[i];
    }

    #pragma unroll
    for (int len = 2; len <= N_FFT; len <<= 1) {
        const int half_len = len >> 1;
        const float sign = INVERSE ? 1.0f : -1.0f;
        const float base_ang = sign * TWO_PI_F / static_cast<float>(len);
        for (int i = 0; i < N_FFT; i += len) {
            #pragma unroll
            for (int j = 0; j < half_len; ++j) {
                float s = 0.0f, c = 1.0f;
                __sincosf(base_ang * static_cast<float>(j), &s, &c);
                const float2 u = x[i + j];
                const float2 v = x[i + j + half_len];
                const float vr = fmaf(c, v.x, -s * v.y);
                const float vi = fmaf(c, v.y, s * v.x);
                x[i + j] = make_float2(u.x + vr, u.y + vi);
                x[i + j + half_len] = make_float2(u.x - vr, u.y - vi);
            }
        }
    }
}

template <int N_FFT, bool SWAPPED_WITH_OFFSET, bool WRITE_VOLTAGES, bool WRITE_POWER>
__global__ void __launch_bounds__(128, 4)
factorized_2d_fft_beamformer_kernel(
    const std::uint8_t* __restrict__ packed,
    float2* __restrict__ voltages,
    float* __restrict__ power,
    const float2* __restrict__ d_calibration_gains,
    const float* __restrict__ d_norm_taper,
    const std::uint8_t* __restrict__ d_rfi_mask,
    const std::size_t n_time,
    const std::size_t n_freq,
    const std::size_t max_beams_stride,
    const std::size_t time_chunk_size,
    const std::size_t n_chunks,
    const bool descending_order) {

    constexpr int WARPS_PER_BLOCK = 4;
    constexpr int N_BEAMS_2D = N_FFT * N_FFT;
    constexpr int ELEMS_PER_LANE_OUT = N_BEAMS_2D / 32; // 2 (for 8x8) or 8 (for 16x16)

    const int lane = static_cast<int>(threadIdx.x);
    const int warp = static_cast<int>(threadIdx.y);
    const std::size_t chunk_idx = static_cast<std::size_t>(blockIdx.x);
    const std::size_t freq = static_cast<std::size_t>(blockIdx.y);
    if (freq >= n_freq || chunk_idx >= n_chunks) return;

    const bool use_mask = (d_rfi_mask != nullptr && n_chunks > 0 && time_chunk_size > 0);
    const bool chunk_masked = use_mask && (d_rfi_mask[chunk_idx * n_freq + freq] != 0);
    if (!WRITE_VOLTAGES && chunk_masked) return;

    const std::size_t t_start = chunk_idx * time_chunk_size;
    if (t_start >= n_time) return;
    const std::size_t t_end = (t_start + time_chunk_size < n_time)
                                  ? (t_start + time_chunk_size)
                                  : n_time;

    __shared__ float2 s_tile[WARPS_PER_BLOCK][N_FFT][N_FFT + 1];

    // Preload per-antenna complex weights w[r] = cal_gain[f, r] * norm_taper[r] (2 elems/lane)
    float2 w_ant[2];
    int grid_row[2], grid_col[2];
    #pragma unroll
    for (int m = 0; m < 2; ++m) {
        const int r = lane + m * 32;
        const float tap = (d_norm_taper != nullptr) ? __ldg(&d_norm_taper[r]) : 1.0f;
        const float2 g = (d_calibration_gains != nullptr)
                             ? __ldg(&d_calibration_gains[freq * 64 + r])
                             : make_float2(1.0f, 0.0f);
        w_ant[m] = make_float2(g.x * tap, g.y * tap);
        const int phys = descending_order ? (63 - r) : r;
        grid_row[m] = phys >> 3;
        grid_col[m] = phys & 7;
    }

    float p_acc[ELEMS_PER_LANE_OUT];
    #pragma unroll
    for (int m = 0; m < ELEMS_PER_LANE_OUT; ++m) p_acc[m] = 0.0f;

    const std::size_t t_stride = n_freq * 64;
    const std::size_t out_row_stride = n_freq * max_beams_stride;
    constexpr int HALF_N = N_FFT >> 1;

    for (std::size_t t = t_start + static_cast<std::size_t>(warp); t < t_end; t += WARPS_PER_BLOCK) {
        // Zero tile (8 floats/lane for 16x16, 2 floats/lane for 8x8)
        #pragma unroll
        for (int m = 0; m < ELEMS_PER_LANE_OUT; ++m) {
            const int idx = lane + m * 32;
            const int rr = idx / N_FFT;
            const int cc = idx % N_FFT;
            s_tile[warp][rr][cc] = make_float2(0.0f, 0.0f);
        }
        __syncwarp();

        // Populate 8x8 physical subgrid
        const std::uint8_t* p_base = packed + t * t_stride + freq * 64;
        #pragma unroll
        for (int m = 0; m < 2; ++m) {
            const int r = lane + m * 32;
            const float2 pv = unpack_int4_byte_fmt<SWAPPED_WITH_OFFSET>(__ldg(&p_base[r]));
            const float2 w = w_ant[m];
            const float2 vw = make_float2(fmaf(w.x, pv.x, -w.y * pv.y),
                                          fmaf(w.x, pv.y, w.y * pv.x));
            s_tile[warp][grid_row[m]][grid_col[m]] = vw;
        }
        __syncwarp();

        // 1D row FFT (East-West, exp(-j 2pi p col / N), +p => +l)
        if (lane < 8) {
            float2 row_vals[N_FFT];
            #pragma unroll
            for (int c = 0; c < N_FFT; ++c) row_vals[c] = s_tile[warp][lane][c];
            fft_1d_radix2_in_reg<N_FFT, false>(row_vals);
            #pragma unroll
            for (int c = 0; c < N_FFT; ++c) {
                s_tile[warp][lane][(c + HALF_N) & (N_FFT - 1)] = row_vals[c];
            }
        }
        __syncwarp();

        // 1D column FFT (North-South, exp(+j 2pi q row / N), +q => +m)
        if (lane < N_FFT) {
            float2 col_vals[N_FFT];
            #pragma unroll
            for (int r = 0; r < N_FFT; ++r) col_vals[r] = s_tile[warp][r][lane];
            fft_1d_radix2_in_reg<N_FFT, true>(col_vals);
            #pragma unroll
            for (int r = 0; r < N_FFT; ++r) {
                s_tile[warp][(r + HALF_N) & (N_FFT - 1)][lane] = col_vals[r];
            }
        }
        __syncwarp();

        // Coalesced store / power accumulate across all 32 lanes
        #pragma unroll
        for (int m = 0; m < ELEMS_PER_LANE_OUT; ++m) {
            const int b = lane + m * 32;
            const int q = b / N_FFT;
            const int p = b % N_FFT;
            const float2 v = s_tile[warp][q][p];
            if constexpr (WRITE_VOLTAGES) {
                if (static_cast<std::size_t>(b) < max_beams_stride) {
                    voltages[t * out_row_stride + freq * max_beams_stride + b] = v;
                }
            }
            if constexpr (WRITE_POWER) {
                p_acc[m] = fmaf(v.x, v.x, fmaf(v.y, v.y, p_acc[m]));
            }
        }
    }

    if constexpr (WRITE_POWER) {
        if (!chunk_masked) {
            float inv_valid = 0.0f;
            if (use_mask) {
                std::size_t valid_samples = 0;
                for (std::size_t c = 0; c < n_chunks; ++c) {
                    if (d_rfi_mask[c * n_freq + freq] == 0) {
                        const std::size_t cs = c * time_chunk_size;
                        const std::size_t ce = (cs + time_chunk_size < n_time) ? (cs + time_chunk_size) : n_time;
                        valid_samples += (ce > cs) ? (ce - cs) : 0;
                    }
                }
                inv_valid = (valid_samples > 0) ? (1.0f / static_cast<float>(valid_samples)) : 0.0f;
            } else {
                inv_valid = (n_time > 0) ? (1.0f / static_cast<float>(n_time)) : 0.0f;
            }
            if (inv_valid > 0.0f) {
                #pragma unroll
                for (int m = 0; m < ELEMS_PER_LANE_OUT; ++m) {
                    const std::size_t b = static_cast<std::size_t>(lane + m * 32);
                    if (b < max_beams_stride && p_acc[m] > 0.0f) {
                        atomicAdd(&power[b * n_freq + freq], p_acc[m] * inv_valid);
                    }
                }
            }
        }
    }
}

template <bool SWAPPED_WITH_OFFSET>
__global__ void factorized_2d_dft_beamformer_kernel(
    const std::uint8_t* __restrict__ packed,
    const float2* __restrict__ d_bx_weights,
    const float2* __restrict__ d_by_weights,
    float2* __restrict__ voltages,
    const std::size_t n_time,
    const std::size_t n_freq,
    const std::size_t nx_beams,
    const std::size_t ny_beams,
    const std::size_t max_beams_stride,
    const bool descending_order) {
    // 1 thread per (p, q) output beam at (t = blockIdx.x, f = blockIdx.y) via shared 8x8 staging
    const std::size_t t = static_cast<std::size_t>(blockIdx.x);
    const std::size_t f = static_cast<std::size_t>(blockIdx.y);
    const std::size_t b = static_cast<std::size_t>(threadIdx.x);
    if (t >= n_time || f >= n_freq) return;

    __shared__ float2 s_v[8][8];
    const std::uint8_t* p_base = packed + (t * n_freq + f) * 64;
    if (b < 64) {
        const std::size_t phys = descending_order ? (63 - b) : b;
        s_v[phys >> 3][phys & 7] = unpack_int4_byte_fmt<SWAPPED_WITH_OFFSET>(__ldg(&p_base[b]));
    }
    __syncthreads();

    const std::size_t total_beams = nx_beams * ny_beams;
    if (b < total_beams && b < max_beams_stride) {
        const std::size_t q = b / nx_beams;
        const std::size_t p = b % nx_beams;
        float out_r = 0.0f, out_i = 0.0f;
        for (int row = 0; row < 8; ++row) {
            float xr = 0.0f, xi = 0.0f;
            for (int col = 0; col < 8; ++col) {
                const float2 v = s_v[row][col];
                const float2 bx = __ldg(&d_bx_weights[(p * n_freq + f) * 8 + col]);
                xr = fmaf(bx.x, v.x, fmaf(-bx.y, v.y, xr));
                xi = fmaf(bx.x, v.y, fmaf(bx.y, v.x, xi));
            }
            const float2 by = __ldg(&d_by_weights[(q * n_freq + f) * 8 + row]);
            out_r = fmaf(by.x, xr, fmaf(-by.y, xi, out_r));
            out_i = fmaf(by.x, xi, fmaf(by.y, xr, out_i));
        }
        voltages[(t * n_freq + f) * max_beams_stride + b] = make_float2(out_r, out_i);
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
    DirectInputFormat input_format,
    bool enable_tensor_core) {

    if (num_active_beams == 0) return;

    const std::size_t n_elem = n_ant * std::max<std::size_t>(1, n_pol);
    const std::size_t active_beams = num_active_beams;
    const std::size_t stride_beams = std::max(max_beams_stride, active_beams);
    const auto* packed_bytes = reinterpret_cast<const std::uint8_t*>(d_packed);
    const bool interpolate = (d_step_weights != nullptr);

    // Dispatch Blackwell/Volta+ Tensor Core WMMA kernel for stationary/large-beam regimes (GPU-4.2)
    if (enable_tensor_core && !interpolate && n_pol == 1 && n_elem == 64
        && active_beams >= 16 && n_time >= 16 && (n_time % 16 == 0)) {
        const dim3 grid_wmma(static_cast<unsigned int>((n_time + 255) / 256),
                             static_cast<unsigned int>(n_freq),
                             static_cast<unsigned int>((active_beams + 63) / 64));
        if (input_format == DirectInputFormat::INT4X2_SWAPPED_WITHOFFSET) {
            direct_beamformer_wmma_kernel<true><<<grid_wmma, 256, 0, stream>>>(
                d_voltages, d_weights, packed_bytes, n_time, n_freq, active_beams, stride_beams);
        } else {
            direct_beamformer_wmma_kernel<false><<<grid_wmma, 256, 0, stream>>>(
                d_voltages, d_weights, packed_bytes, n_time, n_freq, active_beams, stride_beams);
        }
        CHECK_CUDA_ERROR_NON_OO(cudaGetLastError());
        return;
    }

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
    cudaStream_t stream,
    std::uint64_t beam_calibration_mask) {

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
        n_time,
        beam_calibration_mask);

    // Optional Fused Weight-Space BiSLC (zero-cost cross-beam sidelobe nulling)
    if (enable_weight_bislc && num_beams > 1 && num_beams <= 32) {
        const unsigned int f_grid = static_cast<unsigned int>(n_freq);
        if (num_beams <= 8) {
            fuse_weight_space_bislc_kernel<8><<<f_grid, 256, 0, stream>>>(
                d_weights, d_step_weights, d_directions_start, d_directions_end, d_wavenumbers,
                d_antenna_positions, d_norm_taper, d_calibration_gains, num_beams, n_freq, n_elem,
                n_time, bislc_diag_loading, beam_calibration_mask);
        } else if (num_beams <= 16) {
            fuse_weight_space_bislc_kernel<16><<<f_grid, 256, 0, stream>>>(
                d_weights, d_step_weights, d_directions_start, d_directions_end, d_wavenumbers,
                d_antenna_positions, d_norm_taper, d_calibration_gains, num_beams, n_freq, n_elem,
                n_time, bislc_diag_loading, beam_calibration_mask);
        } else {
            fuse_weight_space_bislc_kernel<32><<<f_grid, 256, 0, stream>>>(
                d_weights, d_step_weights, d_directions_start, d_directions_end, d_wavenumbers,
                d_antenna_positions, d_norm_taper, d_calibration_gains, num_beams, n_freq, n_elem,
                n_time, bislc_diag_loading, beam_calibration_mask);
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
    std::size_t n_pol,
    const std::uint8_t* __restrict__ d_rfi_mask = nullptr,
    std::size_t chunk_size = 256,
    std::size_t n_chunks = 0)
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
    const bool use_mask = (d_rfi_mask != nullptr && n_chunks > 0 && chunk_size > 0);

    float thread_sum = 0.0f;
    for (std::size_t t = threadIdx.x; t < n_time; t += blockDim.x) {
        if (use_mask && d_rfi_mask[(t / chunk_size) * n_freq + f] != 0) {
            continue;
        }
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
            float inv_time = 0.0f;
            if (use_mask) {
                std::size_t valid_samples = 0;
                for (std::size_t c = 0; c < n_chunks; ++c) {
                    if (d_rfi_mask[c * n_freq + f] == 0) {
                        const std::size_t c_start = c * chunk_size;
                        const std::size_t c_end = (c_start + chunk_size < n_time) ? (c_start + chunk_size) : n_time;
                        valid_samples += (c_end > c_start) ? (c_end - c_start) : 0;
                    }
                }
                inv_time = (valid_samples > 0) ? (1.0f / static_cast<float>(valid_samples)) : 0.0f;
            } else {
                inv_time = (n_time > 0) ? (1.0f / static_cast<float>(n_time)) : 0.0f;
            }
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
    std::size_t n_pol,
    const std::uint8_t* d_rfi_mask,
    std::size_t chunk_size)
{
    if (max_beams_stride == 0 || n_freq == 0 || d_voltages == nullptr || d_power == nullptr) return;
    const dim3 grid(static_cast<unsigned int>(max_beams_stride), static_cast<unsigned int>(n_freq));
    constexpr int BLOCK_THREADS = 256;
    const std::size_t n_chunks = (chunk_size > 0) ? ((n_time + chunk_size - 1) / chunk_size) : 0;
    reduce_beam_power_kernel<BLOCK_THREADS><<<grid, BLOCK_THREADS, 0, stream>>>(
        d_voltages, d_power, n_time, n_freq, active_beams, max_beams_stride, n_pol,
        d_rfi_mask, chunk_size, n_chunks);
}

void launch_direct_beamformer_power_only(
    const int4x2_t* d_packed,
    const float2* d_weights,
    const float2* d_step_weights,
    float* d_power,
    std::size_t n_time,
    std::size_t n_freq,
    std::size_t n_ant,
    std::size_t num_active_beams,
    std::size_t max_beams_stride,
    std::size_t time_chunk_size,
    std::size_t /*time_unroll*/,
    std::size_t beam_tile_size,
    cudaStream_t stream,
    std::size_t n_pol,
    DirectPolLayout pol_layout,
    DirectInputFormat input_format,
    const std::uint8_t* d_rfi_mask) {

    const std::size_t stride_beams = std::max(max_beams_stride, num_active_beams);
    if (stride_beams == 0 || n_freq == 0 || d_power == nullptr) return;

    CHECK_CUDA_ERROR_NON_OO(
        cudaMemsetAsync(d_power, 0, stride_beams * n_freq * sizeof(float), stream));

    if (num_active_beams == 0 || n_time == 0 || d_packed == nullptr || d_weights == nullptr) {
        return;
    }

    const std::size_t n_elem = n_ant * std::max<std::size_t>(1, n_pol);
    std::size_t b_tile = 1;
    if (beam_tile_size >= 4 && num_active_beams >= 4) {
        b_tile = (n_elem <= 128) ? 4 : 2;
    } else if (beam_tile_size >= 2 && num_active_beams >= 2) {
        b_tile = (n_elem <= 256) ? 2 : 1;
    }

    constexpr int WARPS_PER_BLOCK = 4;
    const std::size_t chunk_sz = (time_chunk_size > 0) ? time_chunk_size : 256;
    const std::size_t num_chunks = (n_time + chunk_sz - 1) / chunk_sz;
    const std::size_t num_beam_tiles = (num_active_beams + b_tile - 1) / b_tile;
    const dim3 grid_dim(static_cast<unsigned int>(num_beam_tiles),
                        static_cast<unsigned int>(n_freq),
                        static_cast<unsigned int>(num_chunks));
    const dim3 block_dim(32, WARPS_PER_BLOCK);

    auto launch_configured = [&](auto elem_tag, auto pol_tag, auto layout_tag,
                                 auto b_tile_tag, auto interp_tag, auto fmt_tag) {
        constexpr int N_E = decltype(elem_tag)::value;
        constexpr int N_P = decltype(pol_tag)::value;
        constexpr int L_O = decltype(layout_tag)::value;
        constexpr int B_T = decltype(b_tile_tag)::value;
        constexpr bool I_P = decltype(interp_tag)::value;
        constexpr bool F_O = decltype(fmt_tag)::value;
        direct_beamformer_power_only_kernel<N_E, N_P, L_O, B_T, I_P, F_O>
            <<<grid_dim, block_dim, 0, stream>>>(
                d_power, d_weights, d_step_weights,
                reinterpret_cast<const std::uint8_t*>(d_packed), d_rfi_mask,
                n_time, n_freq, chunk_sz, num_chunks, num_active_beams, stride_beams);
    };

    auto dispatch_fmt = [&](auto elem_tag, auto pol_tag, auto layout_tag, auto b_tile_tag, auto interp_tag) {
        if (input_format == DirectInputFormat::INT4X2_SWAPPED_WITHOFFSET) {
            launch_configured(elem_tag, pol_tag, layout_tag, b_tile_tag, interp_tag, std::true_type{});
        } else {
            launch_configured(elem_tag, pol_tag, layout_tag, b_tile_tag, interp_tag, std::false_type{});
        }
    };

    auto dispatch_interp = [&](auto elem_tag, auto pol_tag, auto layout_tag, auto b_tile_tag) {
        if (d_step_weights != nullptr) {
            dispatch_fmt(elem_tag, pol_tag, layout_tag, b_tile_tag, std::true_type{});
        } else {
            dispatch_fmt(elem_tag, pol_tag, layout_tag, b_tile_tag, std::false_type{});
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

void launch_factorized_2d_fft_beamformer(
    const int4x2_t* d_packed,
    float2* d_voltages,
    float* d_power,
    const float2* d_calibration_gains,
    const float* d_norm_taper,
    const std::uint8_t* d_rfi_mask,
    std::size_t n_time,
    std::size_t n_freq,
    std::size_t fft_size,
    std::size_t max_beams_stride,
    std::size_t time_chunk_size,
    bool descending_order,
    cudaStream_t stream,
    DirectInputFormat input_format) {

    if (d_packed == nullptr || n_time == 0 || n_freq == 0) return;
    if (fft_size != 8 && fft_size != 16) {
        throw std::invalid_argument("launch_factorized_2d_fft_beamformer: fft_size must be 8 or 16");
    }
    const std::size_t stride_beams = std::max(max_beams_stride, fft_size * fft_size);
    const bool write_v = (d_voltages != nullptr);
    const bool write_p = (d_power != nullptr);
    if (!write_v && !write_p) return;

    if (write_p) {
        CHECK_CUDA_ERROR_NON_OO(
            cudaMemsetAsync(d_power, 0, stride_beams * n_freq * sizeof(float), stream));
    }

    const std::size_t chunk_sz = (time_chunk_size > 0) ? time_chunk_size : 256;
    const std::size_t n_chunks = (n_time + chunk_sz - 1) / chunk_sz;
    const dim3 grid_dim(static_cast<unsigned int>(n_chunks), static_cast<unsigned int>(n_freq));
    const dim3 block_dim(32, 4);
    const auto* packed_bytes = reinterpret_cast<const std::uint8_t*>(d_packed);

    auto dispatch = [&](auto fft_tag, auto fmt_tag, auto v_tag, auto p_tag) {
        constexpr int NF = decltype(fft_tag)::value;
        constexpr bool SO = decltype(fmt_tag)::value;
        constexpr bool WV = decltype(v_tag)::value;
        constexpr bool WP = decltype(p_tag)::value;
        factorized_2d_fft_beamformer_kernel<NF, SO, WV, WP><<<grid_dim, block_dim, 0, stream>>>(
            packed_bytes, d_voltages, d_power, d_calibration_gains, d_norm_taper, d_rfi_mask,
            n_time, n_freq, stride_beams, chunk_sz, n_chunks, descending_order);
    };

    auto dispatch_vp = [&](auto fft_tag, auto fmt_tag) {
        if (write_v && write_p) dispatch(fft_tag, fmt_tag, std::true_type{}, std::true_type{});
        else if (write_v) dispatch(fft_tag, fmt_tag, std::true_type{}, std::false_type{});
        else dispatch(fft_tag, fmt_tag, std::false_type{}, std::true_type{});
    };

    auto dispatch_fmt = [&](auto fft_tag) {
        if (input_format == DirectInputFormat::INT4X2_SWAPPED_WITHOFFSET) {
            dispatch_vp(fft_tag, std::true_type{});
        } else {
            dispatch_vp(fft_tag, std::false_type{});
        }
    };

    if (fft_size == 8) dispatch_fmt(std::integral_constant<int, 8>{});
    else dispatch_fmt(std::integral_constant<int, 16>{});

    CHECK_CUDA_ERROR_NON_OO(cudaGetLastError());
}

void launch_factorized_2d_dft_beamformer(
    const int4x2_t* d_packed,
    const float2* d_bx_weights,
    const float2* d_by_weights,
    float2* d_voltages,
    std::size_t n_time,
    std::size_t n_freq,
    std::size_t nx_beams,
    std::size_t ny_beams,
    std::size_t max_beams_stride,
    std::size_t /*time_chunk_size*/,
    bool descending_order,
    cudaStream_t stream,
    DirectInputFormat input_format) {

    if (!d_packed || !d_bx_weights || !d_by_weights || !d_voltages) return;
    if (n_time == 0 || n_freq == 0 || nx_beams == 0 || ny_beams == 0) return;

    const std::size_t total_beams = nx_beams * ny_beams;
    const std::size_t stride_beams = std::max(max_beams_stride, total_beams);
    const unsigned int block_threads = static_cast<unsigned int>(std::max<std::size_t>(64, std::min<std::size_t>(1024, total_beams)));
    const dim3 grid_dim(static_cast<unsigned int>(n_time), static_cast<unsigned int>(n_freq));
    const auto* packed_bytes = reinterpret_cast<const std::uint8_t*>(d_packed);

    if (input_format == DirectInputFormat::INT4X2_SWAPPED_WITHOFFSET) {
        factorized_2d_dft_beamformer_kernel<true><<<grid_dim, block_threads, 0, stream>>>(
            packed_bytes, d_bx_weights, d_by_weights, d_voltages, n_time, n_freq,
            nx_beams, ny_beams, stride_beams, descending_order);
    } else {
        factorized_2d_dft_beamformer_kernel<false><<<grid_dim, block_threads, 0, stream>>>(
            packed_bytes, d_bx_weights, d_by_weights, d_voltages, n_time, n_freq,
            nx_beams, ny_beams, stride_beams, descending_order);
    }
    CHECK_CUDA_ERROR_NON_OO(cudaGetLastError());
}

} // namespace kotekan
