#include "cudaTransientTrigger.hpp"
#include "cudaUtils.hpp"

#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace kotekan {

namespace {

/**
 * @brief Warp-level reduction sum for double-precision float using shuffle instructions.
 */
__device__ inline double warp_reduce_sum_double(double val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        int low = __double2loint(val);
        int high = __double2hiint(val);
        low = __shfl_down_sync(0xffffffff, low, offset);
        high = __shfl_down_sync(0xffffffff, high, offset);
        val += __hiloint2double(high, low);
    }
    return val;
}

/**
 * @brief Single-pass 256-thread reduction kernel across all M time samples for frequency channel f.
 *
 * Evaluates:
 * S_1 = sum |y_0|^2
 * S_2 = sum |y_0|^4
 * SK(f) = (M+1)/(M-1) * (M * S_2 / (S_1^2) - 1.0)
 * R_01(f) = |sum y_0 * conj(y_1)| / sqrt(S_1(0) * S_1(1))
 */
__global__ void __launch_bounds__(256, 4)
transient_detection_metrics_kernel(
    const float2* __restrict__ d_voltages,
    float2* __restrict__ d_metrics,
    std::size_t n_time,
    std::size_t n_freq,
    std::size_t num_beams,
    std::size_t max_beams_stride,
    const std::uint8_t* __restrict__ d_rfi_mask,
    std::size_t chunk_size,
    std::size_t n_chunks) {

    const std::size_t f = blockIdx.x;
    if (f >= n_freq) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const bool use_mask = (d_rfi_mask != nullptr && n_chunks > 0 && chunk_size > 0);

    double acc_s1_0 = 0.0;
    double acc_s2_0 = 0.0;
    double acc_s1_1 = 0.0;
    double acc_cr   = 0.0;
    double acc_ci   = 0.0;

    // Strided accumulation across unmasked time samples
    for (std::size_t t = tid; t < n_time; t += 256) {
        if (use_mask && d_rfi_mask[(t / chunk_size) * n_freq + f] != 0) {
            continue;
        }
        const std::size_t idx = (t * n_freq + f) * max_beams_stride;
        const float2 y0 = d_voltages[idx];
        const double p0 = static_cast<double>(y0.x) * static_cast<double>(y0.x) +
                          static_cast<double>(y0.y) * static_cast<double>(y0.y);
        acc_s1_0 += p0;
        acc_s2_0 += p0 * p0;

        if (num_beams >= 2) {
            const float2 y1 = d_voltages[idx + 1];
            const double p1 = static_cast<double>(y1.x) * static_cast<double>(y1.x) +
                              static_cast<double>(y1.y) * static_cast<double>(y1.y);
            acc_s1_1 += p1;
            acc_cr += static_cast<double>(y0.x) * static_cast<double>(y1.x) +
                      static_cast<double>(y0.y) * static_cast<double>(y1.y);
            acc_ci += static_cast<double>(y0.y) * static_cast<double>(y1.x) -
                      static_cast<double>(y0.x) * static_cast<double>(y1.y);
        }
    }

    // Intra-warp reduction across 32 lanes
    acc_s1_0 = warp_reduce_sum_double(acc_s1_0);
    acc_s2_0 = warp_reduce_sum_double(acc_s2_0);
    if (num_beams >= 2) {
        acc_s1_1 = warp_reduce_sum_double(acc_s1_1);
        acc_cr   = warp_reduce_sum_double(acc_cr);
        acc_ci   = warp_reduce_sum_double(acc_ci);
    }

    // Shared memory staging for 8 warps
    __shared__ double smem_s1_0[8];
    __shared__ double smem_s2_0[8];
    __shared__ double smem_s1_1[8];
    __shared__ double smem_cr[8];
    __shared__ double smem_ci[8];

    if (lane_id == 0) {
        smem_s1_0[warp_id] = acc_s1_0;
        smem_s2_0[warp_id] = acc_s2_0;
        if (num_beams >= 2) {
            smem_s1_1[warp_id] = acc_s1_1;
            smem_cr[warp_id]   = acc_cr;
            smem_ci[warp_id]   = acc_ci;
        }
    }

    __syncthreads();

    // Inter-warp reduction performed by warp 0
    if (warp_id == 0) {
        double s1_0 = (lane_id < 8) ? smem_s1_0[lane_id] : 0.0;
        double s2_0 = (lane_id < 8) ? smem_s2_0[lane_id] : 0.0;
        double s1_1 = (num_beams >= 2 && lane_id < 8) ? smem_s1_1[lane_id] : 0.0;
        double cr   = (num_beams >= 2 && lane_id < 8) ? smem_cr[lane_id]   : 0.0;
        double ci   = (num_beams >= 2 && lane_id < 8) ? smem_ci[lane_id]   : 0.0;

        s1_0 = warp_reduce_sum_double(s1_0);
        s2_0 = warp_reduce_sum_double(s2_0);
        if (num_beams >= 2) {
            s1_1 = warp_reduce_sum_double(s1_1);
            cr   = warp_reduce_sum_double(cr);
            ci   = warp_reduce_sum_double(ci);
        }

        // Thread 0 computes final channel metrics over unmasked sample count
        if (lane_id == 0) {
            std::size_t valid_samples = n_time;
            if (use_mask) {
                valid_samples = 0;
                for (std::size_t c = 0; c < n_chunks; ++c) {
                    if (d_rfi_mask[c * n_freq + f] == 0) {
                        const std::size_t cs = c * chunk_size;
                        const std::size_t ce =
                            (cs + chunk_size < n_time) ? (cs + chunk_size) : n_time;
                        valid_samples += (ce > cs) ? (ce - cs) : 0;
                    }
                }
            }

            const double M = static_cast<double>(valid_samples);
            constexpr double EPS = 1.0e-12;

            double sk = 1.0;
            double r01 = 0.0;
            if (valid_samples >= 32 && s1_0 > EPS) {
                const double factor = (M + 1.0) / (M - 1.0);
                sk = factor * ((M * s2_0) / (s1_0 * s1_0 + EPS) - 1.0);

                if (num_beams >= 2 && s1_1 > EPS) {
                    const double cross_mag = sqrt(cr * cr + ci * ci);
                    const double denom = sqrt(s1_0 * s1_1) + EPS;
                    r01 = cross_mag / denom;
                    if (r01 > 1.0) r01 = 1.0;
                }
            }

            d_metrics[f] = make_float2(static_cast<float>(sk), static_cast<float>(r01));
        }
    }
}

} // namespace

void launch_transient_detection_metrics(
    const float2* d_voltages,
    float2* d_metrics,
    std::size_t n_time,
    std::size_t n_freq,
    std::size_t num_beams,
    std::size_t max_beams_stride,
    cudaStream_t stream,
    const std::uint8_t* d_rfi_mask,
    std::size_t chunk_size) {

    if (n_freq == 0 || n_time == 0 || d_voltages == nullptr || d_metrics == nullptr) {
        return;
    }

    const std::size_t n_chunks = (chunk_size > 0) ? ((n_time + chunk_size - 1) / chunk_size) : 0;
    dim3 grid(static_cast<unsigned int>(n_freq), 1, 1);
    dim3 block(256, 1, 1);

    transient_detection_metrics_kernel<<<grid, block, 0, stream>>>(
        d_voltages,
        d_metrics,
        n_time,
        n_freq,
        num_beams,
        max_beams_stride,
        d_rfi_mask,
        chunk_size,
        n_chunks);
}

namespace {

constexpr std::size_t kMaxCoincidenceBeams = 64;
constexpr std::size_t kMaxCoincidenceChunks = 256;

__global__ void __launch_bounds__(256, 4)
transient_subframe_chunk_power_kernel(
    const float2* __restrict__ d_voltages,
    float* __restrict__ d_chunk_power,
    std::size_t n_time,
    std::size_t n_freq,
    std::size_t active_beams,
    std::size_t max_beams_stride,
    std::size_t chunk_samples,
    std::size_t n_chunks,
    const std::uint8_t* __restrict__ d_rfi_mask,
    std::size_t rfi_chunk_size,
    std::size_t n_rfi_chunks) {

    const std::size_t c = blockIdx.x;
    const std::size_t b = blockIdx.y;
    if (c >= n_chunks || b >= active_beams) {
        return;
    }

    const std::size_t t0 = c * chunk_samples;
    const std::size_t t1 = (t0 + chunk_samples < n_time) ? (t0 + chunk_samples) : n_time;
    const bool use_mask = (d_rfi_mask != nullptr && rfi_chunk_size > 0 && n_rfi_chunks > 0);

    double sum_p = 0.0;
    std::size_t count = 0;
    const std::size_t total_tf = (t1 - t0) * n_freq;
    for (std::size_t idx = threadIdx.x; idx < total_tf; idx += blockDim.x) {
        const std::size_t dt = idx / n_freq;
        const std::size_t f = idx - dt * n_freq;
        const std::size_t t = t0 + dt;
        if (use_mask) {
            const std::size_t rc = t / rfi_chunk_size;
            if (rc < n_rfi_chunks && d_rfi_mask[rc * n_freq + f] != 0) {
                continue;
            }
        }
        const float2 v = d_voltages[(t * n_freq + f) * max_beams_stride + b];
        sum_p += static_cast<double>(v.x) * v.x + static_cast<double>(v.y) * v.y;
        ++count;
    }

    sum_p = warp_reduce_sum_double(sum_p);
    // Reduce uint32 count within warp
    unsigned int c_local = static_cast<unsigned int>(count);
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        c_local += __shfl_down_sync(0xffffffff, c_local, offset);
    }

    __shared__ double s_p[8];
    __shared__ unsigned int s_c[8];
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    if (lane_id == 0) {
        s_p[warp_id] = sum_p;
        s_c[warp_id] = c_local;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        double tot_p = 0.0;
        unsigned int tot_c = 0;
        for (int w = 0; w < 8; ++w) {
            tot_p += s_p[w];
            tot_c += s_c[w];
        }
        d_chunk_power[c * active_beams + b] =
            (tot_c > 0) ? static_cast<float>(tot_p / static_cast<double>(tot_c)) : -1.0f;
    }
}

__device__ float median_small(float* vals, int n) {
    for (int i = 1; i < n; ++i) {
        const float key = vals[i];
        int j = i - 1;
        while (j >= 0 && vals[j] > key) {
            vals[j + 1] = vals[j];
            --j;
        }
        vals[j + 1] = key;
    }
    if ((n & 1) != 0) {
        return vals[n >> 1];
    }
    return 0.5f * (vals[(n >> 1) - 1] + vals[n >> 1]);
}

__global__ void transient_subframe_coincidence_reduce_kernel(
    const float* __restrict__ d_chunk_power,
    TransientSubframeMetrics* __restrict__ d_out,
    std::size_t n_chunks,
    std::size_t active_beams) {

    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }

    TransientSubframeMetrics res;
    if (n_chunks == 0 || active_beams == 0) {
        *d_out = res;
        return;
    }

    const int n_c = static_cast<int>((n_chunks < kMaxCoincidenceChunks) ? n_chunks : kMaxCoincidenceChunks);
    const int n_b = static_cast<int>((active_beams < kMaxCoincidenceBeams) ? active_beams : kMaxCoincidenceBeams);

    float med[kMaxCoincidenceBeams];
    float sig[kMaxCoincidenceBeams];
    float scratch[kMaxCoincidenceChunks];

    for (int b = 0; b < n_b; ++b) {
        int vc = 0;
        for (int c = 0; c < n_c; ++c) {
            const float p = d_chunk_power[c * n_b + b];
            if (p >= 0.0f) {
                scratch[vc++] = p;
            }
        }
        if (vc == 0) {
            med[b] = 0.0f;
            sig[b] = 1.0e-6f;
            continue;
        }
        const float m = median_small(scratch, vc);
        med[b] = m;
        for (int i = 0; i < vc; ++i) {
            scratch[i] = fabsf(scratch[i] - m);
        }
        const float mad = median_small(scratch, vc);
        const float floor_sig = fmaxf(1.0e-6f * fmaxf(m, 1.0f), 1.0e-6f);
        sig[b] = (mad > 0.0f) ? fmaxf(1.4826f * mad, floor_sig) : floor_sig;
    }

    float best_conf = -1.0f;
    float best_ratio = 1.0f;
    uint32_t best_beam = 0;
    uint32_t best_chunk = 0;
    double ratio_sum = 0.0;
    uint32_t valid_chunks = 0;

    for (int c = 0; c < n_c; ++c) {
        bool chunk_valid = false;
        float e_peak = -1.0f;
        float snr_peak = 0.0f;
        int peak_b = 0;

        for (int b = 0; b < n_b; ++b) {
            const float p = d_chunk_power[c * n_b + b];
            if (p < 0.0f) {
                continue;
            }
            chunk_valid = true;
            const float ex = fmaxf(0.0f, p - med[b]);
            if (ex > e_peak) {
                e_peak = ex;
                snr_peak = ex / sig[b];
                peak_b = b;
            }
        }
        if (!chunk_valid) {
            continue;
        }
        ++valid_chunks;

        float e_side = 0.0f;
        if (n_b >= 2) {
            for (int b = 0; b < n_b; ++b) {
                if (b == peak_b) {
                    continue;
                }
                const float p = d_chunk_power[c * n_b + b];
                if (p >= 0.0f) {
                    const float ex = fmaxf(0.0f, p - med[b]);
                    if (ex > e_side) {
                        e_side = ex;
                    }
                }
            }
        }

        const float ratio = (e_peak > 1.0e-12f && n_b >= 2)
                                ? fminf(1.0f, fmaxf(0.0f, e_side / e_peak))
                                : ((n_b >= 2) ? 1.0f : 0.0f);
        const float conf = snr_peak * fmaxf(0.0f, 1.0f - ratio);
        ratio_sum += static_cast<double>(ratio);

        if (conf > best_conf) {
            best_conf = conf;
            best_ratio = ratio;
            best_beam = static_cast<uint32_t>(peak_b);
            best_chunk = static_cast<uint32_t>(c);
        }
    }

    res.valid_chunks = valid_chunks;
    if (valid_chunks > 0) {
        res.max_sidelobe_confidence = fmaxf(0.0f, best_conf);
        res.min_sidelobe_ratio = best_ratio;
        res.mean_sidelobe_ratio = static_cast<float>(ratio_sum / static_cast<double>(valid_chunks));
        res.peak_beam_id = best_beam;
        res.peak_chunk_id = best_chunk;
    }
    *d_out = res;
}

} // namespace

void launch_transient_subframe_coincidence(
    const float2* d_voltages, TransientSubframeMetrics* d_out, std::size_t n_time,
    std::size_t n_freq, std::size_t num_beams, std::size_t max_beams_stride,
    std::size_t chunk_samples, cudaStream_t stream, const std::uint8_t* d_rfi_mask,
    std::size_t rfi_chunk_size, float* d_scratch_chunk_power) {

    if (d_out == nullptr) {
        return;
    }
    if (d_voltages == nullptr || n_time == 0 || n_freq == 0 || num_beams == 0) {
        CHECK_CUDA_ERROR_NON_OO(cudaMemsetAsync(d_out, 0, sizeof(TransientSubframeMetrics), stream));
        return;
    }

    const std::size_t chunk_sz = (chunk_samples > 0) ? chunk_samples : 256;
    const std::size_t raw_chunks = (n_time + chunk_sz - 1) / chunk_sz;
    const std::size_t n_chunks =
        (raw_chunks < kMaxCoincidenceChunks) ? raw_chunks : kMaxCoincidenceChunks;
    const std::size_t active_beams =
        (num_beams < kMaxCoincidenceBeams) ? num_beams : kMaxCoincidenceBeams;
    const std::size_t n_rfi_chunks =
        (rfi_chunk_size > 0) ? ((n_time + rfi_chunk_size - 1) / rfi_chunk_size) : 0;

    float* scratch_ptr = d_scratch_chunk_power;
    bool owned_scratch = false;
    if (scratch_ptr == nullptr) {
        CHECK_CUDA_ERROR_NON_OO(
            cudaMallocAsync(&scratch_ptr, n_chunks * active_beams * sizeof(float), stream));
        owned_scratch = true;
    }

    const dim3 pwr_grid(static_cast<unsigned int>(n_chunks),
                        static_cast<unsigned int>(active_beams), 1);
    transient_subframe_chunk_power_kernel<<<pwr_grid, 256, 0, stream>>>(
        d_voltages, scratch_ptr, n_time, n_freq, active_beams,
        (max_beams_stride >= active_beams) ? max_beams_stride : active_beams, chunk_sz, n_chunks,
        d_rfi_mask, rfi_chunk_size, n_rfi_chunks);

    transient_subframe_coincidence_reduce_kernel<<<1, 1, 0, stream>>>(
        scratch_ptr, d_out, n_chunks, active_beams);

    if (owned_scratch) {
        CHECK_CUDA_ERROR_NON_OO(cudaFreeAsync(scratch_ptr, stream));
    }
    CHECK_CUDA_ERROR_NON_OO(cudaGetLastError());
}

} // namespace kotekan
