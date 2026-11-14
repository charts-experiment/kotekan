#include "cudaRfiMitigation.hpp"
#include "cudaUtils.hpp"

#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace kotekan {

namespace {

template <bool SWAPPED_WITH_OFFSET = false>
__device__ __forceinline__ void decode_int4x2_rfi(std::uint32_t byte_val, int& re, int& im) {
    if constexpr (SWAPPED_WITH_OFFSET) {
#if defined(__CUDA_ARCH__)
        asm("bfe.u32 %0, %1, 4, 4;" : "=r"(re) : "r"(byte_val));
        asm("bfe.u32 %0, %1, 0, 4;" : "=r"(im) : "r"(byte_val));
        re -= 8;
        im -= 8;
#else
        re = static_cast<int>((byte_val >> 4) & 0x0Fu) - 8;
        im = static_cast<int>(byte_val & 0x0Fu) - 8;
#endif
    } else {
#if defined(__CUDA_ARCH__)
        asm("bfe.s32 %0, %1, 0, 4;" : "=r"(re) : "r"(byte_val));
        asm("bfe.s32 %0, %1, 4, 4;" : "=r"(im) : "r"(byte_val));
#else
        re = (static_cast<int>(byte_val) << 28) >> 28;
        im = (static_cast<int>(byte_val) << 24) >> 28;
#endif
    }
}

__device__ __forceinline__ float device_median_small(float* buf, int n) {
    if (n <= 0)
        return 0.0f;
    for (int i = 1; i < n; ++i) {
        float key = buf[i];
        int j = i - 1;
        while (j >= 0 && buf[j] > key) {
            buf[j + 1] = buf[j];
            --j;
        }
        buf[j + 1] = key;
    }
    if (n & 1) {
        return buf[n / 2];
    }
    return 0.5f * (buf[n / 2 - 1] + buf[n / 2]);
}

// ============================================================================
// Pass 1: Scale-Invariant Per-(Chunk, Freq) Spectral Kurtosis, Power & Saturation
// ============================================================================

template <int BLOCK_ANT, int BLOCK_TIME, bool SWAPPED_WITH_OFFSET = false>
__global__ void __launch_bounds__(256, 4) inspect_rfi_chunks_kernel(
    const std::uint8_t* __restrict__ raw_bytes, const int* __restrict__ active_antennas,
    int num_active_antennas, float* __restrict__ d_chunk_power, float* __restrict__ d_chunk_sk,
    float* __restrict__ d_chunk_sat_fraction, std::size_t n_time, std::size_t n_freq,
    std::size_t n_elem, std::size_t chunk_size, std::size_t sample_stride) {

    const std::size_t f = blockIdx.x;
    const std::size_t c = blockIdx.y;
    if (f >= n_freq)
        return;

    const std::size_t t_start = c * chunk_size;
    if (t_start >= n_time)
        return;
    const std::size_t t_end = (t_start + chunk_size < n_time) ? (t_start + chunk_size) : n_time;
    const std::size_t eff_m = (t_end - t_start + sample_stride - 1) / sample_stride;

    __shared__ float s_ant_power[BLOCK_ANT];
    __shared__ float s_ant_sk[BLOCK_ANT];
    __shared__ float s_ant_sat[BLOCK_ANT];
    __shared__ unsigned int s_p1[BLOCK_TIME][BLOCK_ANT];
    __shared__ unsigned long long s_p2[BLOCK_TIME][BLOCK_ANT];
    __shared__ unsigned int s_sat[BLOCK_TIME][BLOCK_ANT];

    const unsigned int ax = threadIdx.x;
    const unsigned int ty = threadIdx.y;

    float total_power_sum = 0.0f;
    float total_sk_sum = 0.0f;
    float total_sat_sum = 0.0f;
    int valid_ant_count = 0;

    for (int a_base = 0; a_base < num_active_antennas; a_base += BLOCK_ANT) {
        const int a_idx = a_base + static_cast<int>(ax);
        const int raw_elem = (a_idx < num_active_antennas) ? active_antennas[a_idx] : -1;

        unsigned int local_p1 = 0U;
        unsigned long long local_p2 = 0ULL;
        unsigned int local_sat = 0U;

        if (raw_elem >= 0 && static_cast<std::size_t>(raw_elem) < n_elem) {
            for (std::size_t t = t_start + ty * sample_stride; t < t_end;
                 t += BLOCK_TIME * sample_stride) {
                const std::size_t byte_idx =
                    (t * n_freq + f) * n_elem + static_cast<std::size_t>(raw_elem);
                const std::uint32_t byte_val = __ldg(&raw_bytes[byte_idx]);
                int re, im;
                decode_int4x2_rfi<SWAPPED_WITH_OFFSET>(byte_val, re, im);
                const unsigned int p = static_cast<unsigned int>(re * re + im * im);
                local_p1 += p;
                local_p2 += static_cast<unsigned long long>(p) * static_cast<unsigned long long>(p);

                // Quantizer symmetric saturation check: [-7, +7] bounds
                if (re <= -7 || re >= 7 || im <= -7 || im >= 7) {
                    local_sat++;
                }
            }
        }

        s_p1[ty][ax] = local_p1;
        s_p2[ty][ax] = local_p2;
        s_sat[ty][ax] = local_sat;
        __syncthreads();

        if (ty == 0 && a_idx < num_active_antennas) {
            unsigned long long sum_p1 = 0ULL;
            unsigned long long sum_p2 = 0ULL;
            unsigned int sum_sat = 0U;
            #pragma unroll
            for (int r = 0; r < BLOCK_TIME; ++r) {
                sum_p1 += s_p1[r][ax];
                sum_p2 += s_p2[r][ax];
                sum_sat += s_sat[r][ax];
            }

            const float m_f = static_cast<float>(eff_m);
            const float mean_p = (m_f > 0.0f) ? (static_cast<float>(sum_p1) / m_f) : 0.0f;
            float sk_a = 1.0f;
            if (eff_m >= 2 && sum_p1 > 0ULL) {
                const double s1_d = static_cast<double>(sum_p1);
                const double s2_d = static_cast<double>(sum_p2);
                const double m_d = static_cast<double>(eff_m);
                sk_a = static_cast<float>(((m_d + 1.0) / (m_d - 1.0))
                                          * ((m_d * s2_d) / (s1_d * s1_d) - 1.0));
            }
            s_ant_power[ax] = mean_p;
            s_ant_sk[ax] = sk_a;
            s_ant_sat[ax] = (eff_m > 0) ? (static_cast<float>(sum_sat) / m_f) : 0.0f;
        }
        __syncthreads();

        if (ty == 0 && ax == 0) {
            const int tile_ants = (num_active_antennas - a_base < BLOCK_ANT)
                                      ? (num_active_antennas - a_base)
                                      : BLOCK_ANT;
            for (int k = 0; k < tile_ants; ++k) {
                if (s_ant_power[k] > 0.0f) {
                    total_power_sum += s_ant_power[k];
                    total_sk_sum += s_ant_sk[k];
                    total_sat_sum += s_ant_sat[k];
                    valid_ant_count++;
                }
            }
        }
        __syncthreads();
    }

    if (ty == 0 && ax == 0) {
        const std::size_t out_idx = c * n_freq + f;
        if (valid_ant_count > 0) {
            const float inv_ants = 1.0f / static_cast<float>(valid_ant_count);
            d_chunk_power[out_idx] = total_power_sum * inv_ants;
            d_chunk_sk[out_idx] = total_sk_sum * inv_ants;
            if (d_chunk_sat_fraction != nullptr) {
                d_chunk_sat_fraction[out_idx] = total_sat_sum * inv_ants;
            }
        } else {
            d_chunk_power[out_idx] = 0.0f;
            d_chunk_sk[out_idx] = 1.0f;
            if (d_chunk_sat_fraction != nullptr) {
                d_chunk_sat_fraction[out_idx] = 0.0f;
            }
        }
    }
}

// ============================================================================
// Pass 2: Fused Temporal Baseline + Iterative Outlier-Rejected Passband MAD +
//         Empirical SK Variance + Morphological SIR Extension
// ============================================================================

constexpr int MAX_STACK_CHUNKS = 128;
constexpr int MAX_SHARED_FREQ = 1024;
constexpr int MAX_FREQ_WINDOW = 65;

__global__ void __launch_bounds__(512, 2) evaluate_rfi_flags_fused_kernel(
    const float* __restrict__ d_chunk_power, const float* __restrict__ d_chunk_sk,
    const float* __restrict__ d_chunk_sat_fraction, const std::uint8_t* __restrict__ d_static_notch,
    const float* __restrict__ d_channel_gains, std::uint8_t* __restrict__ d_chunk_rfi_mask,
    RfiChannelStats* __restrict__ d_channel_stats, std::size_t n_chunks, std::size_t n_freq,
    float sk_sigma_unit, bool enable_sk, float sk_sigma_thresh, bool enable_burst,
    float burst_sigma_thresh, bool enable_freq_mad, int freq_win_half, float mad_sigma_thresh,
    bool enable_saturation_check, float sat_fraction_thresh, bool enable_morphology_sir,
    int edge_channels_to_ignore, const std::uint8_t* __restrict__ d_exempt_channels,
    bool enable_persistent_ema, float ema_alpha, float* __restrict__ d_ema_power,
    float* __restrict__ d_ema_mad) {

    __shared__ float s_chan_med[MAX_SHARED_FREQ];
    __shared__ float s_chan_mad[MAX_SHARED_FREQ];
    __shared__ float s_chan_med_sk[MAX_SHARED_FREQ];
    __shared__ float s_chan_mad_sk[MAX_SHARED_FREQ];

    const int n_f = static_cast<int>((n_freq < MAX_SHARED_FREQ) ? n_freq : MAX_SHARED_FREQ);
    const int nc = static_cast<int>((n_chunks < MAX_STACK_CHUNKS) ? n_chunks : MAX_STACK_CHUNKS);

    // Phase A: Temporal statistics per frequency channel
    for (int f = static_cast<int>(threadIdx.x); f < n_f; f += static_cast<int>(blockDim.x)) {
        float buf[MAX_STACK_CHUNKS];
        for (int c = 0; c < nc; ++c) {
            buf[c] = d_chunk_power[static_cast<std::size_t>(c) * n_freq + f];
        }
        const float med_p = device_median_small(buf, nc);

        for (int c = 0; c < nc; ++c) {
            const float p = d_chunk_power[static_cast<std::size_t>(c) * n_freq + f];
            buf[c] = fabsf(p - med_p);
        }
        const float mad_p = 1.4826f * device_median_small(buf, nc);

        for (int c = 0; c < nc; ++c) {
            buf[c] = d_chunk_sk[static_cast<std::size_t>(c) * n_freq + f];
        }
        const float med_sk = device_median_small(buf, nc);

        // Compute empirical SK MAD to account for spatial correlation across antennas
        for (int c = 0; c < nc; ++c) {
            const float sk = d_chunk_sk[static_cast<std::size_t>(c) * n_freq + f];
            buf[c] = fabsf(sk - med_sk);
        }
        const float mad_sk = 1.4826f * device_median_small(buf, nc);

        s_chan_med[f] = med_p;
        s_chan_mad[f] = mad_p;
        s_chan_med_sk[f] = med_sk;
        s_chan_mad_sk[f] = mad_sk;
    }
    __syncthreads();

    // Phase B: Frequency baseline and chunk flag evaluation
    for (int f = static_cast<int>(threadIdx.x); f < n_f; f += static_cast<int>(blockDim.x)) {
        const float chan_med_p = s_chan_med[f];
        const float chan_mad_p = s_chan_mad[f];

        float freq_z = 0.0f;
        bool is_freq_mad_outlier = false;
        float local_sk_center = 1.0f;

        // Skip edge channels near PFB roll-off from triggering passband false alarms
        const bool is_edge = (f < edge_channels_to_ignore) || (f >= n_f - edge_channels_to_ignore);

        if (n_f >= 5 && !is_edge) {
            float win_buf[MAX_FREQ_WINDOW];
            float sk_win_buf[MAX_FREQ_WINDOW];
            int half_w = freq_win_half;
            if (2 * half_w + 1 > MAX_FREQ_WINDOW)
                half_w = (MAX_FREQ_WINDOW - 1) / 2;

            int f_lo = f - half_w;
            int f_hi = f + half_w;
            if (f_lo < 0) {
                f_hi = min(n_f - 1, f_hi - f_lo);
                f_lo = 0;
            }
            if (f_hi >= n_f) {
                f_lo = max(0, f_lo - (f_hi - (n_f - 1)));
                f_hi = n_f - 1;
            }

            int count = 0;
            for (int idx = f_lo; idx <= f_hi && count < MAX_FREQ_WINDOW; ++idx) {
                // If digital gains provided, evaluate pre-quantizer spectra P_pre = P / g^2
                float p_val = s_chan_med[idx];
                if (d_channel_gains != nullptr) {
                    const float g = d_channel_gains[idx];
                    if (g > 0.0f) {
                        p_val /= (g * g);
                    }
                }
                win_buf[count] = p_val;
                sk_win_buf[count] = s_chan_med_sk[idx];
                count++;
            }
            const float local_med = device_median_small(win_buf, count);
            const float raw_sk_med = device_median_small(sk_win_buf, count);
            if (raw_sk_med > 0.70f && raw_sk_med < 1.30f) {
                local_sk_center = raw_sk_med;
            }

            if (enable_freq_mad) {
                int count_dev = 0;
                for (int idx = f_lo; idx <= f_hi && count_dev < MAX_FREQ_WINDOW; ++idx) {
                    float p_val = s_chan_med[idx];
                    if (d_channel_gains != nullptr) {
                        const float g = d_channel_gains[idx];
                        if (g > 0.0f) {
                            p_val /= (g * g);
                        }
                    }
                    win_buf[count_dev++] = fabsf(p_val - local_med);
                }
                const float prelim_mad = 1.4826f * device_median_small(win_buf, count_dev);

                // Iterative outlier-rejection (f-engine smooth_bandpass):
                // exclude channels with dev >= 4.0 * max(prelim_mad, 0.05 * local_med)
                const float mad_cutoff = 4.0f * fmaxf(prelim_mad, 0.05f * local_med);
                int inlier_count = 0;
                for (int idx = f_lo; idx <= f_hi && inlier_count < MAX_FREQ_WINDOW; ++idx) {
                    float p_val = s_chan_med[idx];
                    if (d_channel_gains != nullptr) {
                        const float g = d_channel_gains[idx];
                        if (g > 0.0f) {
                            p_val /= (g * g);
                        }
                    }
                    const float dev = fabsf(p_val - local_med);
                    if (dev < mad_cutoff) {
                        win_buf[inlier_count++] = dev;
                    }
                }
                const float local_mad = (inlier_count >= 3)
                                            ? (1.4826f * device_median_small(win_buf, inlier_count))
                                            : prelim_mad;
                const float eff_sigma = fmaxf(local_mad, 0.05f * fmaxf(local_med, 1.0e-3f));

                float p_target = chan_med_p;
                if (d_channel_gains != nullptr) {
                    const float g = d_channel_gains[f];
                    if (g > 0.0f) {
                        p_target /= (g * g);
                    }
                }
                freq_z = (p_target - local_med) / eff_sigma;
                if (freq_z >= mad_sigma_thresh) {
                    is_freq_mad_outlier = true;
                }
            }
        }

        const bool is_static_notch = (d_static_notch != nullptr && d_static_notch[f] != 0);
        const bool is_exempt = (d_exempt_channels != nullptr && d_exempt_channels[f] != 0);

        // Effective SK standard error bounded by both theoretical and empirical temporal MAD
        const float eff_sk_unit = fmaxf(sk_sigma_unit, fmaxf(s_chan_mad_sk[f], 0.025f));
        const float sk_half_width = sk_sigma_thresh * eff_sk_unit;
        const float sk_hi = local_sk_center + sk_half_width;
        const float sk_lo = fmaxf(0.05f, local_sk_center - sk_half_width);

        // Bound temporal burst baseline by persistent cross-frame EMA when available so
        // multi-second steady bursts cannot hide inside a single 51.2 ms frame median.
        const float ema_p = (d_ema_power != nullptr) ? d_ema_power[f] : 0.0f;
        const float ema_mad = (d_ema_mad != nullptr) ? d_ema_mad[f] : 0.0f;
        const bool use_ema =
            enable_persistent_ema && (d_ema_power != nullptr) && !is_exempt && (ema_p > 0.0f);
        const float ref_med_p = use_ema ? fminf(chan_med_p, ema_p) : chan_med_p;
        const float ref_mad_p =
            (use_ema && ema_mad > 0.0f) ? fminf(chan_mad_p, ema_mad) : chan_mad_p;
        const float burst_eff_sigma = fmaxf(ref_mad_p, 0.025f * fmaxf(ref_med_p, 1.0e-3f));

        float sum_p = 0.0f;
        float sum_sk = 0.0f;
        float max_sk = 0.0f;
        std::uint32_t flagged_chunks = 0U;
        std::uint32_t chan_flag_bits = 0U;

        for (std::size_t c = 0; c < n_chunks; ++c) {
            const std::size_t idx = c * n_freq + static_cast<std::size_t>(f);
            const float p = d_chunk_power[idx];
            const float sk = d_chunk_sk[idx];

            sum_p += p;
            sum_sk += sk;
            if (c == 0 || sk > max_sk)
                max_sk = sk;

            std::uint8_t mask_bits = RFI_CLEAN;
            if (is_static_notch) {
                mask_bits |= RFI_STATIC_NOTCH;
            }
            if (is_freq_mad_outlier) {
                mask_bits |= RFI_FREQ_MAD;
            }
            if (enable_sk && p > 0.0f) {
                if (sk > sk_hi) {
                    mask_bits |= RFI_SK_IMPULSIVE;
                } else if (sk < sk_lo) {
                    mask_bits |= RFI_SK_CW_TONE;
                }
            }
            if (enable_burst && n_chunks >= 4) {
                const float burst_z = (p - ref_med_p) / burst_eff_sigma;
                if (burst_z >= burst_sigma_thresh) {
                    mask_bits |= RFI_TIME_BURST;
                }
            }
            if (enable_saturation_check && d_chunk_sat_fraction != nullptr) {
                const float sat_frac = d_chunk_sat_fraction[idx];
                if (sat_frac >= sat_fraction_thresh) {
                    mask_bits |= RFI_SATURATION_CLIP;
                }
            }

            d_chunk_rfi_mask[idx] = is_exempt ? RFI_CLEAN : mask_bits;
            if (mask_bits != RFI_CLEAN) {
                flagged_chunks++;
                chan_flag_bits |= static_cast<std::uint32_t>(mask_bits);
            }
        }

        // 1D Temporal SIR / Morphology Dilation: expand flags to neighboring chunks with elevated power
        if (!is_exempt && enable_morphology_sir && n_chunks >= 3) {
            for (std::size_t c = 0; c < n_chunks; ++c) {
                const std::size_t idx = c * n_freq + static_cast<std::size_t>(f);
                if (d_chunk_rfi_mask[idx] & (RFI_TIME_BURST | RFI_SK_IMPULSIVE | RFI_SATURATION_CLIP)) {
                    if (c > 0) {
                        const std::size_t prev_idx = (c - 1) * n_freq + static_cast<std::size_t>(f);
                        if (d_chunk_rfi_mask[prev_idx] == RFI_CLEAN) {
                            const float prev_p = d_chunk_power[prev_idx];
                            const float prev_z = (prev_p - ref_med_p) / burst_eff_sigma;
                            if (prev_z >= 0.5f * burst_sigma_thresh) {
                                d_chunk_rfi_mask[prev_idx] |= RFI_MORPHOLOGY_SIR;
                                flagged_chunks++;
                                chan_flag_bits |= static_cast<std::uint32_t>(RFI_MORPHOLOGY_SIR);
                            }
                        }
                    }
                    if (c + 1 < n_chunks) {
                        const std::size_t next_idx = (c + 1) * n_freq + static_cast<std::size_t>(f);
                        if (d_chunk_rfi_mask[next_idx] == RFI_CLEAN) {
                            const float next_p = d_chunk_power[next_idx];
                            const float next_z = (next_p - ref_med_p) / burst_eff_sigma;
                            if (next_z >= 0.5f * burst_sigma_thresh) {
                                d_chunk_rfi_mask[next_idx] |= RFI_MORPHOLOGY_SIR;
                                flagged_chunks++;
                                chan_flag_bits |= static_cast<std::uint32_t>(RFI_MORPHOLOGY_SIR);
                            }
                        }
                    }
                }
            }
        }

        if (enable_persistent_ema && d_ema_power != nullptr && !is_exempt && chan_med_p > 0.0f) {
            const float alpha = fminf(fmaxf(ema_alpha, 1.0e-4f), 1.0f);
            if (ema_p <= 0.0f) {
                if (flagged_chunks == 0U) {
                    d_ema_power[f] = chan_med_p;
                    if (d_ema_mad != nullptr) {
                        d_ema_mad[f] = chan_mad_p;
                    }
                }
            } else if (flagged_chunks == 0U) {
                d_ema_power[f] = (1.0f - alpha) * ema_p + alpha * chan_med_p;
                if (d_ema_mad != nullptr) {
                    d_ema_mad[f] = (1.0f - alpha) * ema_mad + alpha * chan_mad_p;
                }
            }
        }

        if (d_channel_stats != nullptr) {
            const float inv_nc = (n_chunks > 0) ? (1.0f / static_cast<float>(n_chunks)) : 0.0f;
            RfiChannelStats st;
            st.mean_power = sum_p * inv_nc;
            st.mean_sk = sum_sk * inv_nc;
            st.max_sk = max_sk;
            st.freq_mad_z = freq_z;
            st.flagged_chunk_fraction = static_cast<float>(flagged_chunks) * inv_nc;
            st.flag_bits = chan_flag_bits;
            d_channel_stats[f] = st;
        }
    }
}

// ============================================================================
// Pass 3: Coalesced In-Place Voltage Blanking of Flagged (Chunk, Freq) Cells
// ============================================================================

__global__ void __launch_bounds__(256, 4) blank_rfi_chunks_kernel(
    std::uint8_t* __restrict__ raw_bytes, const std::uint8_t* __restrict__ d_chunk_rfi_mask,
    std::size_t n_time, std::size_t n_freq, std::size_t n_elem, std::size_t chunk_size,
    std::uint8_t zero_byte) {

    const std::size_t f = blockIdx.x;
    const std::size_t c = blockIdx.y;
    if (f >= n_freq)
        return;

    if (d_chunk_rfi_mask[c * n_freq + f] == 0)
        return;

    const std::size_t t_start = c * chunk_size;
    if (t_start >= n_time)
        return;
    const std::size_t t_end = (t_start + chunk_size < n_time) ? (t_start + chunk_size) : n_time;

    const unsigned int ex = threadIdx.x;
    const unsigned int ty = threadIdx.y;

    for (std::size_t t = t_start + ty; t < t_end; t += blockDim.y) {
        const std::size_t row_offset = (t * n_freq + f) * n_elem;
        for (std::size_t e = ex; e < n_elem; e += blockDim.x) {
            raw_bytes[row_offset + e] = zero_byte;
        }
    }
}

} // anonymous namespace

// ============================================================================
// Host Launch Wrappers
// ============================================================================

void launch_inspect_rfi_chunks(const int4x2_t* __restrict__ d_voltages,
                               const int* __restrict__ d_active_antennas, int num_active_antennas,
                               float* __restrict__ d_chunk_power, float* __restrict__ d_chunk_sk,
                               float* __restrict__ d_chunk_sat_fraction,
                               std::size_t n_time, std::size_t n_freq, std::size_t n_elem,
                               std::size_t chunk_size, std::size_t sample_stride,
                               cudaStream_t stream, bool is_swapped_withoffset) {
    if (d_voltages == nullptr || d_active_antennas == nullptr || num_active_antennas <= 0
        || n_time == 0 || n_freq == 0 || n_elem == 0 || chunk_size == 0) {
        return;
    }

    const std::size_t stride = std::max<std::size_t>(1, sample_stride);
    const std::size_t n_chunks = (n_time + chunk_size - 1) / chunk_size;
    const auto* raw_bytes = reinterpret_cast<const std::uint8_t*>(d_voltages);

    constexpr int BLOCK_ANT = 16;
    constexpr int BLOCK_TIME = 16;
    dim3 block(BLOCK_ANT, BLOCK_TIME);
    dim3 grid(static_cast<unsigned int>(n_freq), static_cast<unsigned int>(n_chunks));

    if (is_swapped_withoffset) {
        inspect_rfi_chunks_kernel<BLOCK_ANT, BLOCK_TIME, true><<<grid, block, 0, stream>>>(
            raw_bytes, d_active_antennas, num_active_antennas, d_chunk_power, d_chunk_sk,
            d_chunk_sat_fraction, n_time, n_freq, n_elem, chunk_size, stride);
    } else {
        inspect_rfi_chunks_kernel<BLOCK_ANT, BLOCK_TIME, false><<<grid, block, 0, stream>>>(
            raw_bytes, d_active_antennas, num_active_antennas, d_chunk_power, d_chunk_sk,
            d_chunk_sat_fraction, n_time, n_freq, n_elem, chunk_size, stride);
    }
}

void launch_evaluate_rfi_flags(const float* __restrict__ d_chunk_power,
                               const float* __restrict__ d_chunk_sk,
                               const float* __restrict__ d_chunk_sat_fraction,
                               const std::uint8_t* __restrict__ d_static_notch,
                               const float* __restrict__ d_channel_gains,
                               std::uint8_t* __restrict__ d_chunk_rfi_mask,
                               RfiChannelStats* __restrict__ d_channel_stats, std::size_t n_chunks,
                               std::size_t n_freq, int num_active_antennas,
                               std::size_t eff_samples_per_chunk, const RfiMitigationConfig& cfg,
                               cudaStream_t stream,
                               const std::uint8_t* __restrict__ d_exempt_channels,
                               float* __restrict__ d_ema_power, float* __restrict__ d_ema_mad) {
    if (d_chunk_power == nullptr || d_chunk_sk == nullptr || d_chunk_rfi_mask == nullptr
        || n_chunks == 0 || n_freq == 0) {
        return;
    }

    const float total_dof =
        static_cast<float>(std::max<std::size_t>(1, eff_samples_per_chunk)
                           * static_cast<std::size_t>(std::max(1, num_active_antennas)));
    const float sk_sigma_unit = 2.0f / std::sqrt(total_dof);

    evaluate_rfi_flags_fused_kernel<<<1, 512, 0, stream>>>(
        d_chunk_power, d_chunk_sk, d_chunk_sat_fraction, d_static_notch, d_channel_gains,
        d_chunk_rfi_mask, d_channel_stats, n_chunks, n_freq, sk_sigma_unit, cfg.enable_sk_check,
        cfg.sk_sigma_threshold, cfg.enable_time_burst_check, cfg.burst_sigma_threshold,
        cfg.enable_freq_mad_check, cfg.freq_window_half_width, cfg.mad_sigma_threshold,
        cfg.enable_saturation_check, cfg.saturation_fraction_threshold, cfg.enable_morphology_sir,
        cfg.edge_channels_to_ignore, d_exempt_channels, cfg.enable_persistent_ema, cfg.ema_alpha,
        d_ema_power, d_ema_mad);
}

void launch_blank_rfi_chunks(int4x2_t* __restrict__ d_voltages,
                             const std::uint8_t* __restrict__ d_chunk_rfi_mask, std::size_t n_time,
                             std::size_t n_freq, std::size_t n_elem, std::size_t chunk_size,
                             cudaStream_t stream, bool is_swapped_withoffset) {
    if (d_voltages == nullptr || d_chunk_rfi_mask == nullptr || n_time == 0 || n_freq == 0
        || n_elem == 0 || chunk_size == 0) {
        return;
    }

    const std::size_t n_chunks = (n_time + chunk_size - 1) / chunk_size;
    auto* raw_bytes = reinterpret_cast<std::uint8_t*>(d_voltages);
    const std::uint8_t zero_byte = is_swapped_withoffset ? 0x88u : 0x00u;

    dim3 block(64, 4);
    dim3 grid(static_cast<unsigned int>(n_freq), static_cast<unsigned int>(n_chunks));

    blank_rfi_chunks_kernel<<<grid, block, 0, stream>>>(raw_bytes, d_chunk_rfi_mask, n_time, n_freq,
                                                        n_elem, chunk_size, zero_byte);
}

} // namespace kotekan
