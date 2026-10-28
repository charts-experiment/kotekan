#include "cudaVisibilityCalibrator.hpp"

#include <cuda_runtime.h>

namespace {

__device__ __forceinline__ int32_t round_saturate(float x) {
    // Clamp in float before converting; 2147483520 is the largest float below 2^31.
    x = fminf(fmaxf(x, -2147483648.0f), 2147483520.0f);
    return __float2int_rn(x);
}

__global__ void apply_visibility_calibration_kernel(int2* __restrict__ vis,
                                                    const float2* __restrict__ cal, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    int2 v = vis[i];
    float2 c = cal[i];
    float vr = (float)v.x;
    float vi = (float)v.y;
    vis[i] = make_int2(round_saturate(fmaf(vr, c.x, -vi * c.y)),
                       round_saturate(fmaf(vr, c.y, vi * c.x)));
}

} // namespace

void launch_apply_visibility_calibration(int32_t* d_correlation, const float* d_calibration,
                                         size_t num_visibilities, cudaStream_t stream) {
    const unsigned int threads = 256;
    const unsigned int blocks = (unsigned int)((num_visibilities + threads - 1) / threads);
    apply_visibility_calibration_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<int2*>(d_correlation), reinterpret_cast<const float2*>(d_calibration),
        num_visibilities);
}
