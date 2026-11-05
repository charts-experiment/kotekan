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
    std::size_t max_beams_stride) {

    const std::size_t f = blockIdx.x;
    if (f >= n_freq) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    double acc_s1_0 = 0.0;
    double acc_s2_0 = 0.0;
    double acc_s1_1 = 0.0;
    double acc_cr   = 0.0;
    double acc_ci   = 0.0;

    // Strided accumulation across all M time samples
    for (std::size_t t = tid; t < n_time; t += 256) {
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

        // Thread 0 computes final channel metrics
        if (lane_id == 0) {
            const double M = static_cast<double>(n_time);
            constexpr double EPS = 1.0e-12;

            // Spectral Kurtosis on Beam 0
            double sk = 1.0;
            if (s1_0 > EPS && M > 1.0) {
                const double factor = (M + 1.0) / (M - 1.0);
                sk = factor * ((M * s2_0) / (s1_0 * s1_0 + EPS) - 1.0);
            }

            // Normalized Inter-Beam Cross-Correlation
            double r01 = 0.0;
            if (num_beams >= 2 && s1_0 > EPS && s1_1 > EPS) {
                const double cross_mag = sqrt(cr * cr + ci * ci);
                const double denom = sqrt(s1_0 * s1_1) + EPS;
                r01 = cross_mag / denom;
                if (r01 > 1.0) r01 = 1.0;
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
    cudaStream_t stream) {

    if (n_freq == 0 || n_time == 0 || d_voltages == nullptr || d_metrics == nullptr) {
        return;
    }

    dim3 grid(static_cast<unsigned int>(n_freq), 1, 1);
    dim3 block(256, 1, 1);

    transient_detection_metrics_kernel<<<grid, block, 0, stream>>>(
        d_voltages,
        d_metrics,
        n_time,
        n_freq,
        num_beams,
        max_beams_stride);
}

} // namespace kotekan
