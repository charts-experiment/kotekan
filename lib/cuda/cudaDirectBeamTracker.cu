#include "cudaDirectBeamTracker.hpp"
#include "cudaUtils.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
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
            const float r2 = l * l + m * m;
            if (r2 <= 1.0f) {
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
    return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
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
    const float norm_sq = res.x * res.x + res.y * res.y;
    if (norm_sq > 0.0f) {
        const float inv_r = rsqrtf(norm_sq);
        res.x *= inv_r;
        res.y *= inv_r;
    }
    return res;
}

__device__ __forceinline__ float2 unpack_int4_direct(const std::uint8_t* ptr) {
#if defined(__CUDA_ARCH__)
    const std::uint32_t byte_val = static_cast<std::uint32_t>(__ldg(ptr));
    int r, i;
    asm("bfe.s32 %0, %1, 0, 4;" : "=r"(r) : "r"(byte_val));
    asm("bfe.s32 %0, %1, 4, 4;" : "=r"(i) : "r"(byte_val));
    return make_float2(__int2float_rn(r), __int2float_rn(i));
#else
    const std::uint32_t byte_val = *ptr;
    const int r = (static_cast<int>(byte_val) << 28) >> 28;
    const int i = ((static_cast<int>(byte_val) << 24) >> 28);
    return make_float2(static_cast<float>(r), static_cast<float>(i));
#endif
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
    const std::size_t n_ant,
    const std::size_t n_time,
    const std::size_t n_active) {

    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total = num_beams * n_freq * n_ant;
    if (idx >= total) return;

    const std::size_t ant = idx % n_ant;
    const std::size_t rest = idx / n_ant;
    const std::size_t freq = rest % n_freq;
    const std::size_t beam = rest / n_freq;

    if (antenna_mask != nullptr && __ldg(&antenna_mask[ant]) == 0) {
        weights[idx] = make_float2(0.0f, 0.0f);
        if (step_weights != nullptr) {
            step_weights[idx] = make_float2(1.0f, 0.0f);
        }
        return;
    }

    const DirectDirection3D dir0 = directions_start[beam];
    const float3 pos = antenna_positions[ant];
    const double wave_num = __ldg(&wavenumbers[freq]);

    const double delay_m0 = static_cast<double>(pos.x) * dir0.x +
                            static_cast<double>(pos.y) * dir0.y +
                            static_cast<double>(pos.z) * dir0.z;
    const float phase0 = static_cast<float>(wave_num * delay_m0);
    float s0, c0;
    __sincosf(phase0, &s0, &c0);

    const float norm = (n_active > 0) ? (1.0f / sqrtf(static_cast<float>(n_active))) : 0.0f;
    float wr = c0 * norm;
    float wi = s0 * norm;

    if (calibration_gains != nullptr) {
        const float2 g = __ldg(&calibration_gains[freq * n_ant + ant]);
        // w_cal = w * g
        const float cal_r = wr * g.x - wi * g.y;
        const float cal_i = wr * g.y + wi * g.x;
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
            const float phase1 = static_cast<float>(wave_num * delay_m1);
            const float delta_phase = (phase1 - phase0) / static_cast<float>(n_time);
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
    const std::size_t n_ant,
    const std::size_t n_active) {

    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total = num_grid_points * n_freq * n_ant;
    if (idx >= total) return;

    const std::size_t ant = idx % n_ant;
    const std::size_t rest = idx / n_ant;
    const std::size_t freq = rest % n_freq;
    const std::size_t grid_pt = rest / n_freq;

    if (antenna_mask != nullptr && __ldg(&antenna_mask[ant]) == 0) {
        grid_weights[idx] = make_float2(0.0f, 0.0f);
        return;
    }

    const float2 lm = __ldg(&grid_lms[grid_pt]);
    const float l = lm.x;
    const float m = lm.y;
    const float trans_sq = l * l + m * m;
    const float n = (trans_sq <= 1.0f) ? sqrtf(1.0f - trans_sq) : 0.0f;

    const float3 pos = antenna_positions[ant];
    const double wave_num = __ldg(&wavenumbers[freq]);

    const double delay_m = static_cast<double>(pos.x) * l +
                           static_cast<double>(pos.y) * m +
                           static_cast<double>(pos.z) * n;
    const float phase = static_cast<float>(wave_num * delay_m);
    float s, c;
    __sincosf(phase, &s, &c);

    const float norm = (n_active > 0) ? (1.0f / sqrtf(static_cast<float>(n_active))) : 0.0f;
    float wr = c * norm;
    float wi = s * norm;

    if (calibration_gains != nullptr) {
        const float2 g = __ldg(&calibration_gains[freq * n_ant + ant]);
        const float cal_r = wr * g.x - wi * g.y;
        const float cal_i = wr * g.y + wi * g.x;
        wr = cal_r;
        wi = cal_i;
    }

    grid_weights[idx] = make_float2(wr, wi);
}

template <int N_ANT, int B_TILE, int TIME_UNROLL, bool INTERPOLATE = false>
__global__ void __launch_bounds__(128)
direct_beamformer_fused_multibeam_kernel(
    float2* __restrict__ voltages,
    const float2* __restrict__ weights,
    const float2* __restrict__ step_weights,
    const std::uint8_t* __restrict__ packed,
    const std::size_t n_time,
    const std::size_t n_freq,
    const std::size_t time_chunk_size,
    const std::size_t num_active_beams,
    const std::size_t max_beams_stride,
    const std::size_t total_warps) {

    constexpr unsigned int ANT_PER_LANE = static_cast<unsigned int>(N_ANT / 32);
    constexpr unsigned int full_mask = 0xFFFFFFFFu;

    const unsigned int lane = threadIdx.x;
    const unsigned int warp_in_block = threadIdx.y;
    const std::size_t warp_id =
        static_cast<std::size_t>(blockIdx.x) * blockDim.y + warp_in_block;

    if (warp_id >= total_warps) {
        return;
    }

    const std::size_t num_beam_tiles = (num_active_beams + B_TILE - 1) / B_TILE;
    const std::size_t tile_idx = warp_id % num_beam_tiles;
    const std::size_t rest = warp_id / num_beam_tiles;
    const std::size_t freq = rest % n_freq;
    const std::size_t chunk_idx = rest / n_freq;

    const std::size_t t_start = chunk_idx * time_chunk_size;
    if (t_start >= n_time) {
        return;
    }
    const std::size_t t_end = (t_start + time_chunk_size < n_time)
                                  ? (t_start + time_chunk_size)
                                  : n_time;

    const std::size_t b_base = tile_idx * B_TILE;
    const unsigned int active_in_tile = (b_base + B_TILE <= num_active_beams)
                                            ? B_TILE
                                            : static_cast<unsigned int>(num_active_beams - b_base);

    // 1. Load weights (and optional step rotators) into registers for this warp
    float w_r[B_TILE][ANT_PER_LANE];
    float w_i[B_TILE][ANT_PER_LANE];
    float nw_i[B_TILE][ANT_PER_LANE];
    float dw_r[B_TILE][ANT_PER_LANE];
    float dw_i[B_TILE][ANT_PER_LANE];

    #pragma unroll
    for (int b = 0; b < B_TILE; ++b) {
        const std::size_t b_curr = b_base + b;
        const std::size_t w_base = (b_curr < num_active_beams)
                                       ? ((b_curr * n_freq + freq) * N_ANT)
                                       : 0;
        #pragma unroll
        for (unsigned int a = 0; a < ANT_PER_LANE; ++a) {
            if (b < active_in_tile) {
                const unsigned int elem = lane + a * 32U;
                const float2 w = __ldg(&weights[w_base + elem]);

                if constexpr (INTERPOLATE) {
                    const float2 dw = __ldg(&step_weights[w_base + elem]);
                    dw_r[b][a] = dw.x;
                    dw_i[b][a] = dw.y;

                    // Re-anchor chunk start weights at t_start using binary exponentiation
                    if (t_start > 0) {
                        const float2 pow_dw = complex_pow_int(dw, t_start);
                        const float cur_r = w.x * pow_dw.x - w.y * pow_dw.y;
                        const float cur_i = w.x * pow_dw.y + w.y * pow_dw.x;
                        w_r[b][a] = cur_r;
                        w_i[b][a] = cur_i;
                        nw_i[b][a] = -cur_i;
                    } else {
                        w_r[b][a] = w.x;
                        w_i[b][a] = w.y;
                        nw_i[b][a] = -w.y;
                    }
                } else {
                    w_r[b][a] = w.x;
                    w_i[b][a] = w.y;
                    nw_i[b][a] = -w.y;
                }
            } else {
                w_r[b][a] = 0.0F;
                w_i[b][a] = 0.0F;
                nw_i[b][a] = 0.0F;
                if constexpr (INTERPOLATE) {
                    dw_r[b][a] = 1.0F;
                    dw_i[b][a] = 0.0F;
                }
            }
        }
    }

    const std::size_t t_stride = n_freq * N_ANT;
    const std::size_t voltage_stride = n_freq * max_beams_stride;

    const std::uint8_t* packed_ptr = packed + (t_start * n_freq + freq) * N_ANT + lane;
    float2* voltages_ptr = voltages + (t_start * n_freq + freq) * max_beams_stride + b_base;

    std::size_t t = t_start;

    for (; t + (TIME_UNROLL - 1) < t_end; t += TIME_UNROLL) {
        float s_r[B_TILE][TIME_UNROLL] = {0.0F};
        float s_i[B_TILE][TIME_UNROLL] = {0.0F};

        #pragma unroll
        for (unsigned int a = 0; a < ANT_PER_LANE; ++a) {
            const unsigned int a_offset = a * 32U;

            if constexpr (INTERPOLATE) {
                float wra[B_TILE];
                float wia[B_TILE];
                float nwi[B_TILE];
                #pragma unroll
                for (int b = 0; b < B_TILE; ++b) {
                    wra[b] = w_r[b][a];
                    wia[b] = w_i[b][a];
                    nwi[b] = nw_i[b][a];
                }

                #pragma unroll
                for (int k = 0; k < TIME_UNROLL; ++k) {
                    const float2 p = unpack_int4_direct(&packed_ptr[k * t_stride + a_offset]);

                    #pragma unroll
                    for (int b = 0; b < B_TILE; ++b) {
                        s_r[b][k] = fmaf(wra[b], p.x, fmaf(nwi[b], p.y, s_r[b][k]));
                        s_i[b][k] = fmaf(wra[b], p.y, fmaf(wia[b], p.x, s_i[b][k]));

                        // Rotate phasor for next time sample: W(t+k+1) = W(t+k) * dW
                        const float dwr = dw_r[b][a];
                        const float dwi = dw_i[b][a];
                        const float next_r = wra[b] * dwr - wia[b] * dwi;
                        const float next_i = wra[b] * dwi + wia[b] * dwr;
                        wra[b] = next_r;
                        wia[b] = next_i;
                        nwi[b] = -next_i;
                    }
                }

                #pragma unroll
                for (int b = 0; b < B_TILE; ++b) {
                    w_r[b][a] = wra[b];
                    w_i[b][a] = wia[b];
                    nw_i[b][a] = nwi[b];
                }
            } else {
                #pragma unroll
                for (int k = 0; k < TIME_UNROLL; ++k) {
                    const float2 p = unpack_int4_direct(&packed_ptr[k * t_stride + a_offset]);

                    #pragma unroll
                    for (int b = 0; b < B_TILE; ++b) {
                        const float wra = w_r[b][a];
                        const float wia = w_i[b][a];
                        const float nwi = nw_i[b][a];

                        s_r[b][k] = fmaf(wra, p.x, fmaf(nwi, p.y, s_r[b][k]));
                        s_i[b][k] = fmaf(wra, p.y, fmaf(wia, p.x, s_i[b][k]));
                    }
                }
            }
        }

        // Intra-warp shuffle reduction across all B_TILE beams
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            #pragma unroll
            for (int b = 0; b < B_TILE; ++b) {
                #pragma unroll
                for (int k = 0; k < TIME_UNROLL; ++k) {
                    s_r[b][k] += __shfl_down_sync(full_mask, s_r[b][k], offset);
                    s_i[b][k] += __shfl_down_sync(full_mask, s_i[b][k], offset);
                }
            }
        }

        // Write across beams: Lane 0 holds the warp-reduced sum for all B_TILE beams
        if (lane == 0) {
            #pragma unroll
            for (int b = 0; b < active_in_tile; ++b) {
                #pragma unroll
                for (int k = 0; k < TIME_UNROLL; ++k) {
                    voltages_ptr[k * voltage_stride + b] = make_float2(s_r[b][k], s_i[b][k]);
                }
            }
        }

        packed_ptr += TIME_UNROLL * t_stride;
        voltages_ptr += TIME_UNROLL * voltage_stride;
    }

    // Remainder loop
    for (; t < t_end; ++t) {
        float s_r[B_TILE] = {0.0F};
        float s_i[B_TILE] = {0.0F};

        #pragma unroll
        for (unsigned int a = 0; a < ANT_PER_LANE; ++a) {
            const float2 p = unpack_int4_direct(&packed_ptr[a * 32U]);

            #pragma unroll
            for (int b = 0; b < B_TILE; ++b) {
                const float wra = w_r[b][a];
                const float wia = w_i[b][a];
                const float nwi = nw_i[b][a];

                s_r[b] = fmaf(wra, p.x, fmaf(nwi, p.y, s_r[b]));
                s_i[b] = fmaf(wra, p.y, fmaf(wia, p.x, s_i[b]));

                if constexpr (INTERPOLATE) {
                    const float dwr = dw_r[b][a];
                    const float dwi = dw_i[b][a];
                    const float next_r = wra * dwr - wia * dwi;
                    const float next_i = wra * dwi + wia * dwr;
                    w_r[b][a] = next_r;
                    w_i[b][a] = next_i;
                    nw_i[b][a] = -next_i;
                }
            }
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            #pragma unroll
            for (int b = 0; b < B_TILE; ++b) {
                s_r[b] += __shfl_down_sync(full_mask, s_r[b], offset);
                s_i[b] += __shfl_down_sync(full_mask, s_i[b], offset);
            }
        }

        if (lane == 0) {
            #pragma unroll
            for (int b = 0; b < active_in_tile; ++b) {
                voltages_ptr[b] = make_float2(s_r[b], s_i[b]);
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
        const std::size_t window_size = std::min(aligned_bytes, static_cast<std::size_t>(prop.persistingL2CacheMaxSize));
        if (cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, window_size) != cudaSuccess) {
            cudaGetLastError();
            return;
        }

        cudaStreamAttrValue attr;
        std::memset(&attr, 0, sizeof(attr));
        attr.accessPolicyWindow.base_ptr = const_cast<void*>(ptr);
        attr.accessPolicyWindow.num_bytes = window_size;
        attr.accessPolicyWindow.hitRatio = 1.0F;
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
    cudaStream_t stream) {

    if (num_active_beams == 0) {
        return; // Zero active beams: instantaneous bypass
    }

    const std::size_t active_beams = std::min(num_active_beams, MAX_DIRECT_BEAMS);
    const std::size_t stride_beams = std::max(max_beams_stride, active_beams);

    // Auto-select optimal B_TILE (4, 2, or 1) based on active beam count and antenna count.
    std::size_t b_tile = 1;
    if (beam_tile_size >= 4 && active_beams >= 4) {
        if (n_ant <= 64) {
            b_tile = 4;
        } else if (n_ant <= 128) {
            b_tile = 2;
        } else {
            b_tile = 1;
        }
    } else if (beam_tile_size >= 2 && active_beams >= 2) {
        if (n_ant <= 128) {
            b_tile = 2;
        } else {
            b_tile = 1;
        }
    }

    const std::size_t num_chunks = (n_time + time_chunk_size - 1) / time_chunk_size;
    const std::size_t num_beam_tiles = (active_beams + b_tile - 1) / b_tile;
    const std::size_t total_warps = num_chunks * n_freq * num_beam_tiles;

    // 128 threads per block (4 warps) matching __launch_bounds__(128, 2)
    constexpr int WARPS_PER_BLOCK = 4;
    const dim3 block_dim(32, WARPS_PER_BLOCK);
    const unsigned int grid_dim =
        static_cast<unsigned int>((total_warps + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

    const auto* packed_bytes = reinterpret_cast<const std::uint8_t*>(d_packed);
    const bool interpolate = (d_step_weights != nullptr);

    auto dispatch_kernel_combo = [&](auto ant_tag, auto b_tile_tag, auto interp_tag) {
        constexpr int N_A = decltype(ant_tag)::value;
        constexpr int B_T = decltype(b_tile_tag)::value;
        constexpr bool I_P = decltype(interp_tag)::value;

        if constexpr (B_T == 4) {
            direct_beamformer_fused_multibeam_kernel<N_A, 4, 4, I_P><<<grid_dim, block_dim, 0, stream>>>(
                d_voltages, d_weights, d_step_weights, packed_bytes, n_time, n_freq,
                time_chunk_size, active_beams, stride_beams, total_warps);
        } else if constexpr (B_T == 2) {
            if (time_unroll >= 8) {
                direct_beamformer_fused_multibeam_kernel<N_A, 2, 8, I_P><<<grid_dim, block_dim, 0, stream>>>(
                    d_voltages, d_weights, d_step_weights, packed_bytes, n_time, n_freq,
                    time_chunk_size, active_beams, stride_beams, total_warps);
            } else {
                direct_beamformer_fused_multibeam_kernel<N_A, 2, 4, I_P><<<grid_dim, block_dim, 0, stream>>>(
                    d_voltages, d_weights, d_step_weights, packed_bytes, n_time, n_freq,
                    time_chunk_size, active_beams, stride_beams, total_warps);
            }
        } else {
            if (time_unroll >= 8) {
                direct_beamformer_fused_multibeam_kernel<N_A, 1, 8, I_P><<<grid_dim, block_dim, 0, stream>>>(
                    d_voltages, d_weights, d_step_weights, packed_bytes, n_time, n_freq,
                    time_chunk_size, active_beams, stride_beams, total_warps);
            } else {
                direct_beamformer_fused_multibeam_kernel<N_A, 1, 4, I_P><<<grid_dim, block_dim, 0, stream>>>(
                    d_voltages, d_weights, d_step_weights, packed_bytes, n_time, n_freq,
                    time_chunk_size, active_beams, stride_beams, total_warps);
            }
        }
    };

    auto dispatch_interp = [&](auto ant_tag, auto b_tile_tag) {
        if (interpolate) {
            dispatch_kernel_combo(ant_tag, b_tile_tag, std::true_type{});
        } else {
            dispatch_kernel_combo(ant_tag, b_tile_tag, std::false_type{});
        }
    };

    auto dispatch_b_tile = [&](auto ant_tag) {
        if (b_tile == 4) dispatch_interp(ant_tag, std::integral_constant<int, 4>{});
        else if (b_tile == 2) dispatch_interp(ant_tag, std::integral_constant<int, 2>{});
        else dispatch_interp(ant_tag, std::integral_constant<int, 1>{});
    };

    switch (n_ant) {
        case 32:  dispatch_b_tile(std::integral_constant<int, 32>{}); break;
        case 64:  dispatch_b_tile(std::integral_constant<int, 64>{}); break;
        case 128: dispatch_b_tile(std::integral_constant<int, 128>{}); break;
        case 256: dispatch_b_tile(std::integral_constant<int, 256>{}); break;
        default:
            throw std::invalid_argument("Unsupported n_ant for Direct Beamformer: must be 32, 64, 128, or 256");
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
    std::size_t n_ant,
    std::size_t n_time,
    std::size_t n_active,
    cudaStream_t stream) {

    const std::size_t total_weights = num_beams * n_freq * n_ant;
    if (total_weights == 0) return;

    if (n_active == 0) {
        n_active = n_ant;
    }

    constexpr int BLOCK_SIZE = 256;
    const unsigned int grid_size = static_cast<unsigned int>((total_weights + BLOCK_SIZE - 1) / BLOCK_SIZE);

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
        n_ant,
        n_time,
        n_active);
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
    std::size_t n_ant,
    std::size_t n_active,
    cudaStream_t stream) {

    const std::size_t total_weights = num_grid_points * n_freq * n_ant;
    if (total_weights == 0) return;

    if (n_active == 0) {
        n_active = n_ant;
    }

    constexpr int BLOCK_SIZE = 256;
    const unsigned int grid_size = static_cast<unsigned int>((total_weights + BLOCK_SIZE - 1) / BLOCK_SIZE);

    precompute_sky_grid_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
        d_grid_weights,
        d_grid_lms,
        d_wavenumbers,
        d_antenna_positions,
        d_antenna_mask,
        d_calibration_gains,
        num_grid_points,
        n_freq,
        n_ant,
        n_active);
}

} // namespace kotekan
