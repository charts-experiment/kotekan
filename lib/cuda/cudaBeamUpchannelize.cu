#include "cudaBeamUpchannelize.hpp"

#include "cudaUtils.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>

namespace kotekan {

BeamUpchanWindowType parse_beam_upchan_window(const std::string& name) {
    std::string lower;
    lower.reserve(name.size());
    for (char ch : name) {
        lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
    }
    if (lower == "rectangular" || lower == "rect" || lower == "boxcar") {
        return BeamUpchanWindowType::RECTANGULAR;
    }
    if (lower == "hann" || lower == "hanning" || lower.empty()) {
        return BeamUpchanWindowType::HANN;
    }
    throw std::invalid_argument("Unsupported beam upchannelize window_type: " + name);
}

const char* beam_upchan_window_to_string(BeamUpchanWindowType w) {
    switch (w) {
        case BeamUpchanWindowType::RECTANGULAR:
            return "rectangular";
        case BeamUpchanWindowType::HANN:
        default:
            return "hann";
    }
}

namespace {

// Periodic 16-pt Hann weights normalized by 1/sqrt(mean(h^2)) = sqrt(8/3) * (1/sqrt(16))
// so mean power (Parseval energy per block) is preserved for flat-spectrum signals.
// h[n] = 0.5 * (1 - cos(2*pi*n/16)), sum_{n=0}^{15} h[n]^2 = 6.0, scale = 1/sqrt(6) = 0.4082482904638630
__constant__ float k_hann16_unitary[16] = {
    0.0000000000f, 0.0155380254f, 0.0597865779f, 0.1260092167f,
    0.2041241452f, 0.2822390738f, 0.3484617125f, 0.3927102651f,
    0.4082482905f, 0.3927102651f, 0.3484617125f, 0.2822390738f,
    0.2041241452f, 0.1260092167f, 0.0597865779f, 0.0155380254f,
};

__device__ __forceinline__ float2 c_mul(float2 a, float2 b) {
    return make_float2(fmaf(a.x, b.x, -a.y * b.y), fmaf(a.x, b.y, a.y * b.x));
}

__device__ __forceinline__ void fft16_forward_in_reg(float2 x[16]) {
    constexpr int perm[16] = {0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15};
    float2 t[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        t[i] = x[perm[i]];
    }
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        x[i] = t[i];
    }

    constexpr float TWO_PI_F = 6.2831853071795864769f;
    #pragma unroll
    for (int len = 2; len <= 16; len <<= 1) {
        const int half_len = len >> 1;
        const float base_ang = -TWO_PI_F / static_cast<float>(len);
        for (int i = 0; i < 16; i += len) {
            for (int j = 0; j < half_len; ++j) {
                float s, c;
                __sincosf(base_ang * static_cast<float>(j), &s, &c);
                const float2 w = make_float2(c, s);
                const float2 u = x[i + j];
                const float2 v = c_mul(x[i + j + half_len], w);
                x[i + j] = make_float2(u.x + v.x, u.y + v.y);
                x[i + j + half_len] = make_float2(u.x - v.x, u.y - v.y);
            }
        }
    }
}

template <bool HANN_WINDOW, bool WRITE_VOLTAGES, bool WRITE_POWER>
__global__ void beam_upchannelize_16pt_kernel(const float2* __restrict__ d_formed_beams,
                                              float2* __restrict__ d_upchan_beams,
                                              float* __restrict__ d_upchan_power,
                                              const std::size_t n_time_out,
                                              const std::size_t n_freq,
                                              const std::size_t num_beams,
                                              const std::size_t max_beams_stride,
                                              const std::size_t n_pol,
                                              const std::size_t tb_chunk) {
    const std::size_t beam_block = static_cast<std::size_t>(blockIdx.x) * blockDim.x;
    const std::size_t b = beam_block + threadIdx.x;
    const std::size_t f = static_cast<std::size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (b >= num_beams || f >= n_freq) {
        return;
    }

    const std::size_t tb_start = static_cast<std::size_t>(blockIdx.z) * tb_chunk;
    if (tb_start >= n_time_out) {
        return;
    }
    const std::size_t tb_end = (tb_start + tb_chunk < n_time_out) ? (tb_start + tb_chunk) : n_time_out;

    const std::size_t in_t_stride = n_freq * max_beams_stride * n_pol;
    const std::size_t n_freq_out = n_freq * 16;
    const std::size_t out_tb_stride = n_freq_out * max_beams_stride * n_pol;
    const float inv_nt_out = 1.0f / static_cast<float>(n_time_out);

    for (std::size_t pol = 0; pol < n_pol; ++pol) {
        float p_acc[16];
        if constexpr (WRITE_POWER) {
            #pragma unroll
            for (int k = 0; k < 16; ++k) {
                p_acc[k] = 0.0f;
            }
        }

        const std::size_t in_fbp = (f * max_beams_stride + b) * n_pol + pol;
        const std::size_t out_fbp_base = ((f * 16) * max_beams_stride + b) * n_pol + pol;
        const std::size_t out_k_step = max_beams_stride * n_pol;

        for (std::size_t tb = tb_start; tb < tb_end; ++tb) {
            float2 x[16];
            const std::size_t t0 = tb * 16;
            #pragma unroll
            for (int n = 0; n < 16; ++n) {
                const float2 v = __ldg(&d_formed_beams[(t0 + n) * in_t_stride + in_fbp]);
                const float w = HANN_WINDOW ? k_hann16_unitary[n] : 0.25f;
                x[n] = make_float2(v.x * w, v.y * w);
            }

            fft16_forward_in_reg(x);

            const std::size_t out_base = tb * out_tb_stride + out_fbp_base;
            #pragma unroll
            for (int out_k = 0; out_k < 16; ++out_k) {
                const float2 val = x[(out_k + 8) & 15];
                if constexpr (WRITE_VOLTAGES) {
                    d_upchan_beams[out_base + static_cast<std::size_t>(out_k) * out_k_step] = val;
                }
                if constexpr (WRITE_POWER) {
                    p_acc[out_k] = fmaf(val.x, val.x, fmaf(val.y, val.y, p_acc[out_k]));
                }
            }
        }

        if constexpr (WRITE_POWER) {
            const std::size_t p_base = b * n_freq_out + f * 16;
            #pragma unroll
            for (int out_k = 0; out_k < 16; ++out_k) {
                if (p_acc[out_k] > 0.0f) {
                    atomicAdd(&d_upchan_power[p_base + out_k], p_acc[out_k] * inv_nt_out);
                }
            }
        }
    }
}

} // namespace

void launch_beam_upchannelize_16pt(const float2* d_formed_beams, float2* d_upchan_beams,
                                   float* d_upchan_power, const BeamUpchannelizeConfig& cfg,
                                   cudaStream_t stream) {
    if (!d_formed_beams || cfg.n_time == 0 || cfg.n_freq == 0 || cfg.num_beams == 0) {
        return;
    }
    if (cfg.upchan_factor != 16) {
        throw std::invalid_argument("launch_beam_upchannelize_16pt: upchan_factor must be 16");
    }
    if (cfg.n_time % 16 != 0) {
        throw std::invalid_argument("launch_beam_upchannelize_16pt: n_time must be a multiple of 16");
    }

    const bool write_v = (d_upchan_beams != nullptr);
    const bool write_p = (d_upchan_power != nullptr);
    if (!write_v && !write_p) {
        return;
    }

    const std::size_t stride_beams = std::max(cfg.max_beams_stride, cfg.num_beams);
    const std::size_t n_pol = std::max<std::size_t>(1, cfg.n_pol);
    const std::size_t n_time_out = cfg.n_time / 16;
    const std::size_t n_freq_out = cfg.n_freq * 16;

    if (write_p) {
        CHECK_CUDA_ERROR_NON_OO(cudaMemsetAsync(
            d_upchan_power, 0, cfg.num_beams * n_freq_out * sizeof(float), stream));
    }

    // Map consecutive beams across blockDim.x so warp loads/stores are 100% coalesced
    unsigned int bx = 128;
    unsigned int by = 1;
    if (cfg.num_beams <= 16) {
        bx = 16;
        by = 8;
    } else if (cfg.num_beams <= 32) {
        bx = 32;
        by = 4;
    } else if (cfg.num_beams <= 64) {
        bx = 64;
        by = 2;
    }

    constexpr std::size_t TB_CHUNK = 16;
    const std::size_t num_tb_chunks = (n_time_out + TB_CHUNK - 1) / TB_CHUNK;
    const dim3 block_dim(bx, by);
    const dim3 grid_dim(static_cast<unsigned int>((cfg.num_beams + bx - 1) / bx),
                        static_cast<unsigned int>((cfg.n_freq + by - 1) / by),
                        static_cast<unsigned int>(num_tb_chunks));

    auto launch = [&](auto hann_tag, auto v_tag, auto p_tag) {
        constexpr bool H = decltype(hann_tag)::value;
        constexpr bool WV = decltype(v_tag)::value;
        constexpr bool WP = decltype(p_tag)::value;
        beam_upchannelize_16pt_kernel<H, WV, WP><<<grid_dim, block_dim, 0, stream>>>(
            d_formed_beams, d_upchan_beams, d_upchan_power, n_time_out, cfg.n_freq, cfg.num_beams,
            stride_beams, n_pol, TB_CHUNK);
    };

    auto dispatch_vp = [&](auto hann_tag) {
        if (write_v && write_p) {
            launch(hann_tag, std::true_type{}, std::true_type{});
        } else if (write_v) {
            launch(hann_tag, std::true_type{}, std::false_type{});
        } else {
            launch(hann_tag, std::false_type{}, std::true_type{});
        }
    };

    if (cfg.window_type == BeamUpchanWindowType::HANN) {
        dispatch_vp(std::true_type{});
    } else {
        dispatch_vp(std::false_type{});
    }
    CHECK_CUDA_ERROR_NON_OO(cudaGetLastError());
}

} // namespace kotekan
