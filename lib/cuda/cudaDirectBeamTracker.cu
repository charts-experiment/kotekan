#include "cudaDirectBeamTracker.hpp"
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

__device__ __forceinline__ float2 complex_pow_int(float2 z, std::size_t exp) {
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

__device__ __forceinline__ float2 unpack_int4_byte(std::uint32_t byte_val) {
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

template <int ELEM_PER_LANE>
__device__ __forceinline__ void load_and_unpack_contiguous(
    const std::uint8_t* __restrict__ row_ptr,
    unsigned int lane,
    float2 (&out)[ELEM_PER_LANE]) {

    if constexpr (ELEM_PER_LANE == 1) {
        const std::uint32_t b = __ldg(&row_ptr[lane]);
        out[0] = unpack_int4_byte(b);
    } else if constexpr (ELEM_PER_LANE == 2) {
        const std::uint16_t packed2 = __ldg(reinterpret_cast<const std::uint16_t*>(row_ptr) + lane);
        out[0] = unpack_int4_byte(packed2 & 0xFFu);
        out[1] = unpack_int4_byte((packed2 >> 8) & 0xFFu);
    } else if constexpr (ELEM_PER_LANE == 4) {
        const std::uint32_t packed4 = __ldg(reinterpret_cast<const std::uint32_t*>(row_ptr) + lane);
        #pragma unroll
        for (int a = 0; a < 4; ++a) {
            out[a] = unpack_int4_byte((packed4 >> (a * 8)) & 0xFFu);
        }
    } else if constexpr (ELEM_PER_LANE == 8) {
        const uint2 packed8 = __ldg(reinterpret_cast<const uint2*>(row_ptr) + lane);
        #pragma unroll
        for (int a = 0; a < 4; ++a) {
            out[a]     = unpack_int4_byte((packed8.x >> (a * 8)) & 0xFFu);
            out[a + 4] = unpack_int4_byte((packed8.y >> (a * 8)) & 0xFFu);
        }
    } else if constexpr (ELEM_PER_LANE == 16) {
        const uint4 packed16 = __ldg(reinterpret_cast<const uint4*>(row_ptr) + lane);
        #pragma unroll
        for (int a = 0; a < 4; ++a) {
            out[a]      = unpack_int4_byte((packed16.x >> (a * 8)) & 0xFFu);
            out[a + 4]  = unpack_int4_byte((packed16.y >> (a * 8)) & 0xFFu);
            out[a + 8]  = unpack_int4_byte((packed16.z >> (a * 8)) & 0xFFu);
            out[a + 12] = unpack_int4_byte((packed16.w >> (a * 8)) & 0xFFu);
        }
    } else {
        #pragma unroll
        for (int a = 0; a < ELEM_PER_LANE; ++a) {
            out[a] = unpack_int4_byte(__ldg(&row_ptr[lane * ELEM_PER_LANE + a]));
        }
    }
}

__global__ void generate_steering_weights_kernel(
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

__global__ void precompute_sky_grid_kernel(
    float2* __restrict__ grid_weights,
    const float2* __restrict__ grid_lms,
    const double* __restrict__ wavenumbers,
    const float3* __restrict__ antenna_positions,
    const std::uint8_t* __restrict__ antenna_mask,
    const float2* __restrict__ calibration_gains,
    const std::size_t num_grid_points,
    const std::size_t n_freq,
    const std::size_t n_elem,
    const std::size_t n_active_per_pol) {

    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total = num_grid_points * n_freq * n_elem;
    if (idx >= total) return;

    const std::size_t elem = idx % n_elem;
    const std::size_t rest = idx / n_elem;
    const std::size_t freq = rest % n_freq;
    const std::size_t grid_pt = rest / n_freq;

    if (antenna_mask != nullptr && __ldg(&antenna_mask[elem]) == 0) {
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

    const float norm = (n_active_per_pol > 0) ? rsqrtf(static_cast<float>(n_active_per_pol)) : 0.0f;
    float wr = c * norm;
    float wi = s * norm;

    if (calibration_gains != nullptr) {
        const float2 g = __ldg(&calibration_gains[freq * n_elem + elem]);
        const float cal_r = fmaf(-wi, g.y, wr * g.x);
        const float cal_i = fmaf(wr, g.y, wi * g.x);
        wr = cal_r;
        wi = cal_i;
    }

    grid_weights[idx] = make_float2(wr, wi);
}

template <
    int N_ELEM,
    int N_POL,
    int POL_LAYOUT,
    int B_TILE,
    int TIME_UNROLL,
    bool INTERPOLATE,
    bool COOPERATIVE_SMEM>
__global__ void __launch_bounds__(128)
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
                if constexpr (INTERPOLATE) {
                    const float2 dw = __ldg(&step_weights[w_offset + a]);
                    dw_r[b][a] = dw.x;
                    dw_i[b][a] = dw.y;
                    if (t_start > 0) {
                        const float2 pow_dw = complex_pow_int(dw, t_start);
                        const float2 w_rot = complex_mul(w, pow_dw);
                        w_r[b][a] = w_rot.x;
                        w_i[b][a] = w_rot.y;
                    } else {
                        w_r[b][a] = w.x;
                        w_i[b][a] = w.y;
                    }
                } else {
                    w_r[b][a] = w.x;
                    w_i[b][a] = w.y;
                }
            } else {
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
                smem_voltages[flat_idx] = unpack_int4_byte(raw_b);
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
                    load_and_unpack_contiguous<ELEM_PER_LANE>(packed_ptr + k * t_stride, lane, p_vec);
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
            load_and_unpack_contiguous<ELEM_PER_LANE>(packed_ptr, lane, p_vec);

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
    DirectPolLayout pol_layout) {

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
                                 auto b_tile_tag, auto interp_tag, auto smem_tag) {
        constexpr int N_E = decltype(elem_tag)::value;
        constexpr int N_P = decltype(pol_tag)::value;
        constexpr int L_O = decltype(layout_tag)::value;
        constexpr int B_T = decltype(b_tile_tag)::value;
        constexpr bool I_P = decltype(interp_tag)::value;
        constexpr bool C_S = decltype(smem_tag)::value;
        constexpr int T_U = (B_T == 4) ? 4 : 8;

        const int t_unroll_sel = (time_unroll >= 8 && B_T <= 2) ? 8 : 4;
        if (t_unroll_sel == 8 && T_U == 8) {
            direct_beamformer_blackwell_kernel<N_E, N_P, L_O, B_T, 8, I_P, C_S>
                <<<grid_dim, block_dim, 0, stream>>>(
                    d_voltages, d_weights, d_step_weights, packed_bytes, n_time, n_freq,
                    time_chunk_size, active_beams, stride_beams, total_work_units);
        } else {
            direct_beamformer_blackwell_kernel<N_E, N_P, L_O, B_T, 4, I_P, C_S>
                <<<grid_dim, block_dim, 0, stream>>>(
                    d_voltages, d_weights, d_step_weights, packed_bytes, n_time, n_freq,
                    time_chunk_size, active_beams, stride_beams, total_work_units);
        }
    };

    auto dispatch_smem = [&](auto elem_tag, auto pol_tag, auto layout_tag, auto b_tile_tag, auto interp_tag) {
        if (use_coop_smem) {
            launch_configured(elem_tag, pol_tag, layout_tag, b_tile_tag, interp_tag, std::true_type{});
        } else {
            launch_configured(elem_tag, pol_tag, layout_tag, b_tile_tag, interp_tag, std::false_type{});
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

    if (n_active == 0) {
        n_active = n_elem;
    }
    const std::size_t n_active_per_pol =
        std::max<std::size_t>(1, n_active / std::max<std::size_t>(1, n_pol));

    constexpr int BLOCK_SIZE = 256;
    const unsigned int grid_size =
        static_cast<unsigned int>((total_weights + BLOCK_SIZE - 1) / BLOCK_SIZE);

    generate_steering_weights_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
        d_weights,
        d_step_weights,
        d_directions_start,
        d_directions_end,
        d_wavenumbers,
        d_antenna_positions,
        d_antenna_mask,
        d_calibration_gains,
        num_beams,
        n_freq,
        n_elem,
        n_time,
        n_active_per_pol);
}

void launch_precompute_sky_grid(
    float2* d_grid_weights,
    const float2* d_grid_lms,
    const double* d_wavenumbers,
    const float3* d_antenna_positions,
    const std::uint8_t* d_antenna_mask,
    const float2* d_calibration_gains,
    std::size_t num_grid_points,
    std::size_t n_freq,
    std::size_t n_elem,
    std::size_t n_active,
    cudaStream_t stream,
    std::size_t n_pol) {

    const std::size_t total_weights = num_grid_points * n_freq * n_elem;
    if (total_weights == 0) return;

    if (n_active == 0) {
        n_active = n_elem;
    }
    const std::size_t n_active_per_pol =
        std::max<std::size_t>(1, n_active / std::max<std::size_t>(1, n_pol));

    constexpr int BLOCK_SIZE = 256;
    const unsigned int grid_size =
        static_cast<unsigned int>((total_weights + BLOCK_SIZE - 1) / BLOCK_SIZE);

    precompute_sky_grid_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
        d_grid_weights,
        d_grid_lms,
        d_wavenumbers,
        d_antenna_positions,
        d_antenna_mask,
        d_calibration_gains,
        num_grid_points,
        n_freq,
        n_elem,
        n_active_per_pol);
}

} // namespace kotekan
