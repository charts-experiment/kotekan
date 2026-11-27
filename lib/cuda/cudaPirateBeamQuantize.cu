#include "cudaPirateBeamQuantize.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <limits.h>

namespace kotekan {
namespace {

constexpr int QUANT_MIN_CODE = -7;
constexpr int QUANT_MAX_CODE = +7;
constexpr int QUANT_MASK_CODE = -8;
constexpr float MAD_TO_SIGMA = 1.4826022185f;

__device__ __forceinline__ int sane_lrint_dev(float x) {
    if (isnan(x))
        return 0;
    if (x < static_cast<float>(INT_MIN))
        return INT_MIN;
    if (x >= static_cast<float>(INT_MAX))
        return INT_MAX;
    return __float2int_rn(x);
}

__device__ __forceinline__ void bitonic_sort_256(float* s_data, int tid) {
    for (int k = 2; k <= 256; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            const int ixj = tid ^ j;
            if (ixj > tid) {
                const bool ascending = ((tid & k) == 0);
                const float a = s_data[tid];
                const float b = s_data[ixj];
                if ((ascending && a > b) || (!ascending && a < b)) {
                    s_data[tid] = b;
                    s_data[ixj] = a;
                }
            }
            __syncthreads();
        }
    }
}

__global__ void pirate_decimate_intensity_kernel(
    const float2* __restrict__ d_formed_beams,
    const std::uint8_t* __restrict__ d_rfi_mask,
    float* __restrict__ d_intensity,
    std::uint8_t* __restrict__ d_valid,
    int num_beams,
    int max_beams,
    int num_freq,
    int num_pols,
    int n_time,
    int n_frb,
    int decimation,
    int rfi_chunk_size,
    int n_rfi_chunks,
    float rfi_mask_fraction_threshold,
    bool enable_incoherent_subtraction) {
    const int tau = static_cast<int>(blockIdx.x);
    const int f = static_cast<int>(blockIdx.y);
    if (tau >= n_frb || f >= num_freq)
        return;

    const int tid = static_cast<int>(threadIdx.x);
    const int t0 = tau * decimation;
    const int t1 = min(n_time, t0 + decimation);
    const int n_src = max(0, t1 - t0);

    // Check RFI mask once via lane 0 and broadcast via shared memory.
    __shared__ std::uint8_t s_rfi_valid;
    __shared__ float s_inc_mean;
    __shared__ float s_warp_reduce[32];

    if (tid == 0) {
        std::uint8_t ok = 1;
        if (d_rfi_mask != nullptr && n_rfi_chunks > 0 && rfi_chunk_size > 0 && n_src > 0) {
            int masked = 0;
            for (int t = t0; t < t1; ++t) {
                const int c = min(t / rfi_chunk_size, n_rfi_chunks - 1);
                if (d_rfi_mask[c * num_freq + f] != 0) {
                    ++masked;
                }
            }
            const float frac = static_cast<float>(masked) / static_cast<float>(n_src);
            if (frac >= rfi_mask_fraction_threshold || masked >= n_src) {
                ok = 0;
            }
        }
        s_rfi_valid = ok;
        d_valid[f * n_frb + tau] = ok;
        s_inc_mean = 0.0f;
    }
    __syncthreads();

    const bool rfi_ok = (s_rfi_valid != 0);

    // First pass: accumulate per-beam intensity and incoherent sum across beams.
    float beam_sum = 0.0f;
    int beam_cnt = 0;

    for (int b = tid; b < num_beams; b += blockDim.x) {
        float val = NAN;
        if (rfi_ok && n_src > 0) {
            double accum = 0.0;
            int valid_samples = 0;
            for (int t = t0; t < t1; ++t) {
                if (d_rfi_mask != nullptr && n_rfi_chunks > 0 && rfi_chunk_size > 0) {
                    const int c = min(t / rfi_chunk_size, n_rfi_chunks - 1);
                    if (d_rfi_mask[c * num_freq + f] != 0)
                        continue;
                }
                const std::size_t base =
                    ((static_cast<std::size_t>(t) * num_freq + f) * max_beams + b) * num_pols;
                double pwr = 0.0;
                bool sample_finite = true;
                for (int p = 0; p < num_pols; ++p) {
                    const float2 v = d_formed_beams[base + p];
                    if (!isfinite(v.x) || !isfinite(v.y)) {
                        sample_finite = false;
                        break;
                    }
                    pwr += static_cast<double>(v.x) * v.x + static_cast<double>(v.y) * v.y;
                }
                if (sample_finite) {
                    accum += pwr;
                    ++valid_samples;
                }
            }
            if (valid_samples > 0) {
                val = static_cast<float>(accum / static_cast<double>(valid_samples));
                beam_sum += val;
                ++beam_cnt;
            }
        }
        d_intensity[(static_cast<std::size_t>(b) * num_freq + f) * n_frb + tau] = val;
    }

    if (enable_incoherent_subtraction && num_beams >= 2 && rfi_ok) {
        // Block-wide reduction of beam_sum and beam_cnt (blockDim.x is up to 64).
        const unsigned mask = 0xffffffffU;
        for (int off = 16; off > 0; off >>= 1) {
            beam_sum += __shfl_down_sync(mask, beam_sum, off);
            beam_cnt += __shfl_down_sync(mask, beam_cnt, off);
        }
        const int warp = tid >> 5;
        const int lane = tid & 31;
        __shared__ int s_warp_cnt[32];
        if (lane == 0) {
            s_warp_reduce[warp] = beam_sum;
            s_warp_cnt[warp] = beam_cnt;
        }
        __syncthreads();
        if (tid == 0) {
            float total_sum = 0.0f;
            int total_cnt = 0;
            const int n_warps = (blockDim.x + 31) >> 5;
            for (int w = 0; w < n_warps; ++w) {
                total_sum += s_warp_reduce[w];
                total_cnt += s_warp_cnt[w];
            }
            s_inc_mean = (total_cnt > 0) ? (total_sum / static_cast<float>(total_cnt)) : 0.0f;
        }
        __syncthreads();

        const float mean_b = s_inc_mean;
        for (int b = tid; b < num_beams; b += blockDim.x) {
            const std::size_t idx = (static_cast<std::size_t>(b) * num_freq + f) * n_frb + tau;
            const float cur = d_intensity[idx];
            if (isfinite(cur)) {
                d_intensity[idx] = cur - mean_b;
            }
        }
    }
}

__global__ void pirate_detrend_and_quantize_kernel(
    const float* __restrict__ d_intensity,
    const std::uint8_t* __restrict__ d_valid,
    std::uint8_t* __restrict__ d_pirate_out,
    int num_beams,
    int num_freq,
    int n_frb,
    bool enable_highpass_detrend,
    int half_win,
    bool use_robust_mad_scale,
    float stddev_cutoff) {
    const int f = static_cast<int>(blockIdx.x);
    const int b = static_cast<int>(blockIdx.y);
    const int mc = static_cast<int>(blockIdx.z);
    const int tau = static_cast<int>(threadIdx.x); // 0..255

    if (f >= num_freq || b >= num_beams)
        return;

    const int tau_global = mc * static_cast<int>(PIRATE_SAMPLES_PER_MINICHUNK) + tau;
    if (tau_global >= n_frb)
        return;

    __shared__ float s_raw[PIRATE_SAMPLES_PER_MINICHUNK];
    __shared__ float s_sort[PIRATE_SAMPLES_PER_MINICHUNK];
    __shared__ std::uint8_t s_valid[PIRATE_SAMPLES_PER_MINICHUNK];
    __shared__ int8_t s_q[PIRATE_SAMPLES_PER_MINICHUNK];
    __shared__ float s_offset;
    __shared__ float s_scale;

    const float raw_val =
        d_intensity[(static_cast<std::size_t>(b) * num_freq + f) * n_frb + tau_global];
    const bool is_valid = (d_valid[f * n_frb + tau_global] != 0) && isfinite(raw_val);

    s_raw[tau] = is_valid ? raw_val : 0.0f;
    s_valid[tau] = is_valid ? 1 : 0;
    __syncthreads();

    float val = raw_val;
    if (is_valid && enable_highpass_detrend && half_win >= 1) {
        const int lo = max(0, tau - half_win);
        const int hi = min(static_cast<int>(PIRATE_SAMPLES_PER_MINICHUNK) - 1, tau + half_win);
        double wsum = 0.0;
        int wcnt = 0;
        for (int k = lo; k <= hi; ++k) {
            if (s_valid[k] != 0) {
                wsum += static_cast<double>(s_raw[k]);
                ++wcnt;
            }
        }
        if (wcnt > 0) {
            val = raw_val - static_cast<float>(wsum / static_cast<double>(wcnt));
        }
    }

    const bool final_valid = is_valid && isfinite(val);
    s_valid[tau] = final_valid ? 1 : 0;
    s_sort[tau] = final_valid ? val : FLT_MAX;
    __syncthreads();

    // Count valid samples across the block via shared reductions in s_det-independent warp ops,
    // plus mean and stddev (used directly or as fallback when MAD == 0).
    int warp_cnt = final_valid ? 1 : 0;
    float warp_sum = final_valid ? val : 0.0f;
    float warp_sum2 = final_valid ? (val * val) : 0.0f;
    for (int off = 16; off > 0; off >>= 1) {
        warp_cnt += __shfl_down_sync(0xffffffffU, warp_cnt, off);
        warp_sum += __shfl_down_sync(0xffffffffU, warp_sum, off);
        warp_sum2 += __shfl_down_sync(0xffffffffU, warp_sum2, off);
    }

    __shared__ int s_warp_cnt[8];
    __shared__ float s_warp_sum[8];
    __shared__ float s_warp_sum2[8];
    __shared__ int s_valid_count;
    __shared__ float s_mean;
    __shared__ float s_stddev;

    const int warp = tau >> 5;
    const int lane = tau & 31;
    if (lane == 0) {
        s_warp_cnt[warp] = warp_cnt;
        s_warp_sum[warp] = warp_sum;
        s_warp_sum2[warp] = warp_sum2;
    }
    __syncthreads();

    if (tau == 0) {
        int cnt = 0;
        float sum = 0.0f;
        float sum2 = 0.0f;
        for (int w = 0; w < 8; ++w) {
            cnt += s_warp_cnt[w];
            sum += s_warp_sum[w];
            sum2 += s_warp_sum2[w];
        }
        s_valid_count = cnt;
        if (cnt > 0) {
            const float mean = sum / static_cast<float>(cnt);
            s_mean = mean;
            s_stddev = sqrtf(fmaxf(0.0f, sum2 / static_cast<float>(cnt) - mean * mean));
        } else {
            s_mean = 0.0f;
            s_stddev = 0.0f;
        }
    }
    __syncthreads();

    if (use_robust_mad_scale) {
        bitonic_sort_256(s_sort, tau);
        if (tau == 0) {
            if (s_valid_count > 0) {
                const int mid = s_valid_count >> 1;
                s_offset = (s_valid_count & 1) ? s_sort[mid] : 0.5f * (s_sort[mid - 1] + s_sort[mid]);
            } else {
                s_offset = 0.0f;
            }
        }
        __syncthreads();

        const float med = s_offset;
        s_sort[tau] = final_valid ? fabsf(val - med) : FLT_MAX;
        __syncthreads();
        bitonic_sort_256(s_sort, tau);

        if (tau == 0) {
            float offset = med;
            float sigma = 0.0f;
            if (s_valid_count > 0) {
                const int mid = s_valid_count >> 1;
                const float mad =
                    (s_valid_count & 1) ? s_sort[mid] : 0.5f * (s_sort[mid - 1] + s_sort[mid]);
                sigma = MAD_TO_SIGMA * mad;
                if (sigma <= 1.0e-12f && s_stddev > 1.0e-12f) {
                    offset = s_mean;
                    sigma = s_stddev;
                }
            } else {
                offset = 0.0f;
            }
            s_offset = offset;
            s_scale = (2.0f * stddev_cutoff * sigma) / 15.0f;
        }
    } else if (tau == 0) {
        s_offset = s_mean;
        s_scale = (s_valid_count > 0) ? ((2.0f * stddev_cutoff * s_stddev) / 15.0f) : 0.0f;
    }
    __syncthreads();

    // Pointer math matching pirateNetworkSend:
    //   [minichunk][float16 scales[num_beams][num_freq][2] (scale, offset)][int4 data[num_beams][num_freq][128]]
    const std::size_t mc_bytes =
        static_cast<std::size_t>(num_beams) * num_freq * PIRATE_BYTES_PER_CHAN;
    std::uint8_t* mc_base = d_pirate_out + static_cast<std::size_t>(mc) * mc_bytes;
    __half* scales_base = reinterpret_cast<__half*>(mc_base);
    std::uint8_t* data_base =
        mc_base + static_cast<std::size_t>(num_beams) * num_freq * PIRATE_SCALES_BYTES_PER_CHAN;

    if (tau == 0) {
        __half scaleh = __float2half(s_scale);
        __half offseth = __float2half(s_offset);
        if (s_valid_count == 0 || !isfinite(__half2float(scaleh))
            || !isfinite(__half2float(offseth))) {
            scaleh = __float2half(0.0f);
            offseth = __float2half(0.0f);
        }
        const std::size_t s_idx = (static_cast<std::size_t>(b) * num_freq + f) * 2;
        scales_base[s_idx + 0] = scaleh; // scale first
        scales_base[s_idx + 1] = offseth; // offset second
        s_scale = __half2float(scaleh);
        s_offset = __half2float(offseth);
    }
    __syncthreads();

    const float scale = s_scale;
    const float offset = s_offset;
    int q = QUANT_MASK_CODE;
    if (final_valid) {
        if (scale <= 0.0f) {
            q = 0;
        } else {
            const int j = sane_lrint_dev((val - offset) / scale);
            q = max(QUANT_MIN_CODE, min(QUANT_MAX_CODE, j));
        }
    }
    s_q[tau] = static_cast<int8_t>(q);
    __syncthreads();

    if ((tau & 1) == 0) {
        const int i = tau >> 1;
        const int q0 = static_cast<int>(s_q[tau]);
        const int q1 = static_cast<int>(s_q[tau + 1]);
        const std::uint8_t packed =
            static_cast<std::uint8_t>((q0 & 0x0f) | ((q1 & 0x0f) << 4));
        const std::size_t d_idx =
            (static_cast<std::size_t>(b) * num_freq + f) * PIRATE_DATA_BYTES_PER_CHAN + i;
        data_base[d_idx] = packed;
    }
}

} // namespace

void launch_pirate_beam_quantize(const float2* d_formed_beams,
                                 const std::uint8_t* d_rfi_mask,
                                 float* d_scratch_intensity,
                                 std::uint8_t* d_scratch_valid,
                                 std::uint8_t* d_pirate_out,
                                 const PirateBeamQuantizeConfig& cfg,
                                 cudaStream_t stream) {
    if (!d_formed_beams || !d_scratch_intensity || !d_scratch_valid || !d_pirate_out)
        return;
    if (cfg.num_beams == 0 || cfg.max_beams == 0 || cfg.num_freq == 0 || cfg.seq_per_frb_time_sample == 0)
        return;

    const std::size_t n_frb = cfg.samples_per_data_set / cfg.seq_per_frb_time_sample;
    const std::size_t num_minichunks = n_frb / PIRATE_SAMPLES_PER_MINICHUNK;
    if (n_frb == 0 || num_minichunks == 0)
        return;

    const int n_rfi_chunks = (cfg.rfi_chunk_size > 0)
                                 ? static_cast<int>((cfg.samples_per_data_set + cfg.rfi_chunk_size - 1)
                                                    / cfg.rfi_chunk_size)
                                 : 0;

    // 1. Time decimate voltages -> intensity + optional incoherent-beam subtraction
    {
        const dim3 grid1(static_cast<unsigned>(n_frb), static_cast<unsigned>(cfg.num_freq));
        const int block_b = (cfg.num_beams <= 32) ? 32 : 64;
        pirate_decimate_intensity_kernel<<<grid1, block_b, 0, stream>>>(
            d_formed_beams, d_rfi_mask, d_scratch_intensity, d_scratch_valid,
            static_cast<int>(cfg.num_beams), static_cast<int>(cfg.max_beams),
            static_cast<int>(cfg.num_freq), static_cast<int>(cfg.num_pols),
            static_cast<int>(cfg.samples_per_data_set), static_cast<int>(n_frb),
            static_cast<int>(cfg.seq_per_frb_time_sample), static_cast<int>(cfg.rfi_chunk_size),
            n_rfi_chunks, cfg.rfi_mask_fraction_threshold, cfg.enable_incoherent_subtraction);
    }

    // 2. High-pass detrend + robust MAD / stddev quantization -> pirate v2 wire format
    {
        int win = std::max(3, std::min(255, cfg.highpass_window_samples));
        if ((win & 1) == 0)
            win = std::min(255, win + 1);
        const int half_win = win / 2;

        const dim3 grid2(static_cast<unsigned>(cfg.num_freq), static_cast<unsigned>(cfg.num_beams),
                         static_cast<unsigned>(num_minichunks));
        pirate_detrend_and_quantize_kernel<<<grid2, PIRATE_SAMPLES_PER_MINICHUNK, 0, stream>>>(
            d_scratch_intensity, d_scratch_valid, d_pirate_out, static_cast<int>(cfg.num_beams),
            static_cast<int>(cfg.num_freq), static_cast<int>(n_frb),
            cfg.enable_highpass_detrend, half_win, cfg.use_robust_mad_scale, cfg.stddev_cutoff);
    }
}

} // namespace kotekan
