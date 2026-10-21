#include "cudaChartsUpchannelizer.hpp"

#include <array>
#include <cmath>
#include <mutex>
#include <stdexcept>
#include <string>

namespace beamformer {
namespace {

void check_cuda(const cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
    }
}

__constant__ float c_pfb_weights[128];
__constant__ float2 c_half_bin_twiddles[32];
__constant__ float2 c_w16[8];
__constant__ float2 c_w32[16];

constexpr double pi = 3.141592653589793238462643383279502884;
constexpr float c_sqrt2_2 = 0.7071067811865475244F;

void init_pfb_constant_tables_once() {
    static std::once_flag once;
    std::call_once(once, []() {
        std::array<float, 128> signed_weights{};
        for (std::size_t tap = 0; tap < 128; ++tap) {
            const std::size_t j = tap / upchannelization_factor;
            const float sign = (j % 2 == 1) ? -1.0F : 1.0F;
            signed_weights[tap] = sign * sinc_hanning_pfb_weight(tap);
        }
        check_cuda(cudaMemcpyToSymbol(c_pfb_weights, signed_weights.data(), sizeof(signed_weights)),
                   "cudaMemcpyToSymbol c_pfb_weights");

        std::array<float2, 32> twiddles{};
        for (std::size_t p = 0; p < 32; ++p) {
            const double angle = 2.0 * pi * 15.5 * static_cast<double>(p) / 32.0;
            twiddles[p] = make_float2(static_cast<float>(std::cos(angle)),
                                      static_cast<float>(std::sin(angle)));
        }
        check_cuda(cudaMemcpyToSymbol(c_half_bin_twiddles, twiddles.data(), sizeof(twiddles)),
                   "cudaMemcpyToSymbol c_half_bin_twiddles");

        std::array<float2, 8> w16{};
        for (std::size_t j = 0; j < 8; ++j) {
            const double angle = -2.0 * pi * static_cast<double>(j) / 16.0;
            w16[j] = make_float2(static_cast<float>(std::cos(angle)),
                                 static_cast<float>(std::sin(angle)));
        }
        check_cuda(cudaMemcpyToSymbol(c_w16, w16.data(), sizeof(w16)), "cudaMemcpyToSymbol c_w16");

        std::array<float2, 16> w32{};
        for (std::size_t j = 0; j < 16; ++j) {
            const double angle = -2.0 * pi * static_cast<double>(j) / 32.0;
            w32[j] = make_float2(static_cast<float>(std::cos(angle)),
                                 static_cast<float>(std::sin(angle)));
        }
        check_cuda(cudaMemcpyToSymbol(c_w32, w32.data(), sizeof(w32)), "cudaMemcpyToSymbol c_w32");
    });
}

__device__ __forceinline__ unsigned int bit_reverse_5bit(const unsigned int x) {
    return ((x & 1U) << 4) | ((x & 2U) << 2) | (x & 4U) | ((x & 8U) >> 2) | ((x & 16U) >> 4);
}

__device__ __forceinline__ void fft32_in_place(float* __restrict__ r_real,
                                               float* __restrict__ r_imag) {
    #pragma unroll
    for (int k = 0; k < 32; k += 2) {
        const float u_r = r_real[k];
        const float u_i = r_imag[k];
        const float v_r = r_real[k + 1];
        const float v_i = r_imag[k + 1];
        r_real[k] = u_r + v_r;
        r_imag[k] = u_i + v_i;
        r_real[k + 1] = u_r - v_r;
        r_imag[k + 1] = u_i - v_i;
    }

    #pragma unroll
    for (int k = 0; k < 32; k += 4) {
        const float u0_r = r_real[k];
        const float u0_i = r_imag[k];
        const float v0_r = r_real[k + 2];
        const float v0_i = r_imag[k + 2];
        r_real[k] = u0_r + v0_r;
        r_imag[k] = u0_i + v0_i;
        r_real[k + 2] = u0_r - v0_r;
        r_imag[k + 2] = u0_i - v0_i;

        const float u1_r = r_real[k + 1];
        const float u1_i = r_imag[k + 1];
        const float v1_r = r_imag[k + 3];
        const float v1_i = -r_real[k + 3];
        r_real[k + 1] = u1_r + v1_r;
        r_imag[k + 1] = u1_i + v1_i;
        r_real[k + 3] = u1_r - v1_r;
        r_imag[k + 3] = u1_i - v1_i;
    }

    #pragma unroll
    for (int k = 0; k < 32; k += 8) {
        {
            const float u_r = r_real[k];
            const float u_i = r_imag[k];
            const float v_r = r_real[k + 4];
            const float v_i = r_imag[k + 4];
            r_real[k] = u_r + v_r;
            r_imag[k] = u_i + v_i;
            r_real[k + 4] = u_r - v_r;
            r_imag[k + 4] = u_i - v_i;
        }
        {
            const float u_r = r_real[k + 1];
            const float u_i = r_imag[k + 1];
            const float t_r = r_real[k + 5];
            const float t_i = r_imag[k + 5];
            const float v_r = c_sqrt2_2 * (t_r + t_i);
            const float v_i = c_sqrt2_2 * (t_i - t_r);
            r_real[k + 1] = u_r + v_r;
            r_imag[k + 1] = u_i + v_i;
            r_real[k + 5] = u_r - v_r;
            r_imag[k + 5] = u_i - v_i;
        }
        {
            const float u_r = r_real[k + 2];
            const float u_i = r_imag[k + 2];
            const float v_r = r_imag[k + 6];
            const float v_i = -r_real[k + 6];
            r_real[k + 2] = u_r + v_r;
            r_imag[k + 2] = u_i + v_i;
            r_real[k + 6] = u_r - v_r;
            r_imag[k + 6] = u_i - v_i;
        }
        {
            const float u_r = r_real[k + 3];
            const float u_i = r_imag[k + 3];
            const float t_r = r_real[k + 7];
            const float t_i = r_imag[k + 7];
            const float v_r = c_sqrt2_2 * (-t_r + t_i);
            const float v_i = c_sqrt2_2 * (-t_r - t_i);
            r_real[k + 3] = u_r + v_r;
            r_imag[k + 3] = u_i + v_i;
            r_real[k + 7] = u_r - v_r;
            r_imag[k + 7] = u_i - v_i;
        }
    }

    #pragma unroll
    for (int k = 0; k < 32; k += 16) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float t_r = r_real[k + 8 + j];
            const float t_i = r_imag[k + 8 + j];
            const float w_r = c_w16[j].x;
            const float w_i = c_w16[j].y;
            const float v_r = t_r * w_r - t_i * w_i;
            const float v_i = t_r * w_i + t_i * w_r;
            const float u_r = r_real[k + j];
            const float u_i = r_imag[k + j];
            r_real[k + j] = u_r + v_r;
            r_imag[k + j] = u_i + v_i;
            r_real[k + 8 + j] = u_r - v_r;
            r_imag[k + 8 + j] = u_i - v_i;
        }
    }

    #pragma unroll
    for (int j = 0; j < 16; ++j) {
        const float t_r = r_real[16 + j];
        const float t_i = r_imag[16 + j];
        const float w_r = c_w32[j].x;
        const float w_i = c_w32[j].y;
        const float v_r = t_r * w_r - t_i * w_i;
        const float v_i = t_r * w_i + t_i * w_r;
        const float u_r = r_real[j];
        const float u_i = r_imag[j];
        r_real[j] = u_r + v_r;
        r_imag[j] = u_i + v_i;
        r_real[16 + j] = u_r - v_r;
        r_imag[16 + j] = u_i - v_i;
    }
}

__global__ void fused_upchannelizer_kernel(
    const std::uint8_t* __restrict__ raw_payload,
    const std::uint8_t* __restrict__ history_samples,
    const bool history_initialized,
    std::uint8_t* __restrict__ channelized_packed,
    const std::size_t n_time,
    const std::size_t n_coarse,
    const std::size_t n_element,
    const float quantize_clamp,
    const std::size_t total_transforms) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total_transforms) {
        return;
    }

    const std::size_t e = idx % n_element;
    const std::size_t tf = idx / n_element;
    const std::size_t f = tf % n_coarse;
    const std::size_t t_out = tf / n_coarse;

    float r_real[32];
    float r_imag[32];

    #pragma unroll
    for (std::size_t p = 0; p < 32; ++p) {
        float z_real = 0.0F;
        float z_imag = 0.0F;

        #pragma unroll
        for (std::size_t j = 0; j < pfb_taps_per_phase; ++j) {
            const std::size_t tap = j * upchannelization_factor + p;
            const std::int64_t raw_t =
                (static_cast<std::int64_t>(t_out) + static_cast<std::int64_t>(j) - 3) * 32
                + static_cast<std::int64_t>(p);
            ComplexInt4 raw_sample{0, 0};

            if (raw_t < 0) {
                if (history_initialized && history_samples != nullptr) {
                    const std::size_t hist_idx = static_cast<std::size_t>(raw_t + pfb_history_samples);
                    const std::size_t offset = (f * n_element + e) * pfb_history_samples + hist_idx;
                    raw_sample = unpack_complex_int4(history_samples[offset]);
                }
            } else if (static_cast<std::size_t>(raw_t) < n_time) {
                const std::size_t raw_idx =
                    (static_cast<std::size_t>(raw_t) * n_coarse + f) * n_element + e;
                raw_sample = unpack_complex_int4(raw_payload[raw_idx]);
            }

            const float w = c_pfb_weights[tap];
            z_real += w * static_cast<float>(raw_sample.real);
            z_imag += w * static_cast<float>(raw_sample.imag);
        }

        const float2 tw = c_half_bin_twiddles[p];
        const float zr = z_real * tw.x - z_imag * tw.y;
        const float zi = z_real * tw.y + z_imag * tw.x;

        const unsigned int br = bit_reverse_5bit(static_cast<unsigned int>(p));
        r_real[br] = zr;
        r_imag[br] = zi;
    }

    fft32_in_place(r_real, r_imag);

    // Quantize and pack the results into 4-bit signed integers
    const std::size_t base_out_idx = (t_out * n_coarse + f) * 32;
    #pragma unroll
    for (std::size_t u = 0; u < 32; ++u) {
        const std::size_t out_idx = (base_out_idx + u) * n_element + e;
        const float clamped_r = fminf(fmaxf(r_real[u], -quantize_clamp), quantize_clamp);
        const float clamped_i = fminf(fmaxf(r_imag[u], -quantize_clamp), quantize_clamp);
        const int r = __float2int_rn(clamped_r);
        const int i = __float2int_rn(clamped_i);
        const auto byte = static_cast<std::uint8_t>(
            (static_cast<std::uint8_t>(r) & 0x0F) | ((static_cast<std::uint8_t>(i) & 0x0F) << 4));
        channelized_packed[out_idx] = byte;
    }
}

__global__ void history_update_kernel(
    const std::uint8_t* __restrict__ raw_payload,
    std::uint8_t* __restrict__ history_samples,
    const std::size_t n_time,
    const std::size_t n_coarse,
    const std::size_t n_element,
    const std::size_t total_samples) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total_samples) {
        return;
    }

    const std::size_t s = idx % pfb_history_samples;
    const std::size_t fe_idx = idx / pfb_history_samples;
    const std::size_t e = fe_idx % n_element;
    const std::size_t f = fe_idx / n_element;

    const std::int64_t raw_t = static_cast<std::int64_t>(n_time)
                               - static_cast<std::int64_t>(pfb_history_samples)
                               + static_cast<std::int64_t>(s);

    if (raw_t >= 0 && static_cast<std::size_t>(raw_t) < n_time) {
        const std::size_t raw_idx =
            (static_cast<std::size_t>(raw_t) * n_coarse + f) * n_element + e;
        history_samples[idx] = raw_payload[raw_idx];
    }
}

__global__ void expand_missing_to_validity_kernel(
    const std::uint8_t* __restrict__ raw_missing_flags,
    std::uint8_t* __restrict__ expanded_validity,
    const std::size_t n_coarse,
    const std::size_t total_entries) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total_entries) {
        return;
    }
    const std::size_t t = idx / n_coarse;
    expanded_validity[idx] = (raw_missing_flags[t] == 0) ? 1U : 0U;
}

} // namespace

void launch_fused_upchannelizer(const cudaStream_t stream, const std::uint8_t* raw_payload,
                                const std::uint8_t* history_samples, const bool history_initialized,
                                std::uint8_t* channelized_packed, const std::size_t raw_n_time,
                                const std::size_t n_coarse, const std::size_t n_element,
                                const float quantize_clamp) {
    if (!raw_payload || !channelized_packed) {
        throw std::invalid_argument("upchannelizer device pointers must not be null");
    }
    if (raw_n_time == 0 || raw_n_time % upchannelization_factor != 0) {
        throw std::invalid_argument("raw_n_time must be a positive multiple of 32");
    }
    init_pfb_constant_tables_once();
    (void)cudaGetLastError();

    const std::size_t n_time_out = raw_n_time / upchannelization_factor;
    const std::size_t total_transforms = n_time_out * n_coarse * n_element;
    constexpr std::size_t threads = 128;
    const std::size_t blocks = (total_transforms + threads - 1) / threads;
    fused_upchannelizer_kernel<<<static_cast<unsigned int>(blocks),
                                 static_cast<unsigned int>(threads), 0, stream>>>(
        raw_payload, history_samples, history_initialized, channelized_packed, raw_n_time, n_coarse,
        n_element, quantize_clamp, total_transforms);
    check_cuda(cudaGetLastError(), "fused_upchannelizer_kernel launch");
}

void launch_pfb_history_update(const cudaStream_t stream, const std::uint8_t* raw_payload,
                               std::uint8_t* history_samples, const std::size_t raw_n_time,
                               const std::size_t n_coarse, const std::size_t n_element) {
    if (!raw_payload || !history_samples) {
        throw std::invalid_argument("PFB history update pointers must not be null");
    }
    (void)cudaGetLastError();

    const std::size_t total_samples = n_coarse * n_element * pfb_history_samples;
    constexpr std::size_t threads = 256;
    const std::size_t blocks = (total_samples + threads - 1) / threads;
    history_update_kernel<<<static_cast<unsigned int>(blocks), static_cast<unsigned int>(threads),
                            0, stream>>>(
        raw_payload, history_samples, raw_n_time, n_coarse, n_element, total_samples);
    check_cuda(cudaGetLastError(), "history_update_kernel launch");
}

void launch_expand_missing_to_validity(const cudaStream_t stream,
                                       const std::uint8_t* raw_missing_flags,
                                       std::uint8_t* expanded_validity,
                                       const std::size_t raw_n_time,
                                       const std::size_t n_coarse) {
    if (!raw_missing_flags || !expanded_validity) {
        throw std::invalid_argument("mask expansion pointers must not be null");
    }
    (void)cudaGetLastError();

    const std::size_t total_entries = raw_n_time * n_coarse;
    constexpr std::size_t threads = 256;
    const std::size_t blocks = (total_entries + threads - 1) / threads;
    expand_missing_to_validity_kernel<<<static_cast<unsigned int>(blocks),
                                        static_cast<unsigned int>(threads), 0, stream>>>(
        raw_missing_flags, expanded_validity, n_coarse, total_entries);
    check_cuda(cudaGetLastError(), "expand_missing_to_validity_kernel launch");
}

} // namespace beamformer
