#ifndef CUDA_RFI_MITIGATION_HPP
#define CUDA_RFI_MITIGATION_HPP

#include "DataType.hpp"
#include "chartsConstants.hpp"
#include "cudaDirectBeamTracker.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <string>
#include <utility>
#include <vector>

namespace kotekan {

constexpr std::size_t MAX_RFI_ANTENNAS = MAX_DIRECT_ANTENNAS;

/**
 * @enum RfiFlagReason
 * @brief Bitmask flags classifying why a (time_chunk, freq_channel) cell was flagged.
 *
 * Grounded in Mena-Parra et al. (2018) 4+4 bit quantization bias theory and
 * CHARTS F-Engine requantizer specifications (simulink_models/rtl/quantizer.v):
 * 15-level mid-tread quantizer with symmetric [-7, +7] saturation, operating at
 * target q-RMS = 3.5 (sigma = 1.70 LSB, sigma_c = 1.20 LSB per component).
 */
enum RfiFlagReason : std::uint8_t {
    RFI_CLEAN = 0,
    RFI_SK_IMPULSIVE = 1 << 0,    ///< Spectral Kurtosis above upper Gaussian threshold
    RFI_SK_CW_TONE = 1 << 1,      ///< Spectral Kurtosis below lower Gaussian threshold (CW lock)
    RFI_TIME_BURST = 1 << 2,      ///< Temporal chunk power spike relative to frame median
    RFI_FREQ_MAD = 1 << 3,        ///< Narrowband frequency power outlier relative to sliding passband
    RFI_STATIC_NOTCH = 1 << 4,    ///< Inside a configured static notch band (e.g. MUOS)
    RFI_SATURATION_CLIP = 1 << 5, ///< Excessive fraction of samples saturated at [-7, +7] limits
    RFI_MORPHOLOGY_SIR = 1 << 6   ///< Flagged by Scale-Invariant Rank / morphological burst expansion
};

/**
 * @struct RfiChannelStats
 * @brief Per-channel RFI summary populated on the GPU for each frame (24 bytes, alignas(8)).
 */
struct alignas(8) RfiChannelStats {
    float mean_power = 0.0f;             ///< Mean sample power across active antennas and time
    float mean_sk = 1.0f;                ///< Mean scale-invariant Spectral Kurtosis across chunks
    float max_sk = 1.0f;                 ///< Max chunk Spectral Kurtosis in frame
    float freq_mad_z = 0.0f;             ///< Robust Z-score against local sliding frequency window
    float flagged_chunk_fraction = 0.0f; ///< Fraction of time chunks [0..1] flagged in this frame
    std::uint32_t flag_bits = 0;         ///< Bitwise OR of RfiFlagReason across chunks in this channel
};

/**
 * @struct RfiMitigationConfig
 * @brief Runtime-configurable thresholds and parameters for lightweight GPU RFI mitigation.
 */
struct RfiMitigationConfig {
    bool enabled = true;
    bool blank_voltages_enabled = true;

    std::size_t time_chunk_size = 256; ///< Samples per SK time chunk (256 samples = 853 us)
    std::size_t sample_stride = 1;     ///< Time subsampling stride inside each chunk (1 = exact)

    bool enable_sk_check = true;
    float sk_sigma_threshold = 4.5f; ///< Number of SK std-devs (sigma_SK bounded by empirical temporal MAD)

    bool enable_time_burst_check = true;
    float burst_sigma_threshold = 5.0f; ///< Temporal chunk MAD Z-score threshold within a channel

    bool enable_freq_mad_check = true;
    int freq_window_half_width = 8;   ///< Sliding window half-width (8 -> 17-channel = 5.1 MHz window)
    float mad_sigma_threshold = 6.0f; ///< Local passband MAD Z-score threshold

    bool enable_saturation_check = true;          ///< Check for quantizer +-7 saturation fraction
    float saturation_fraction_threshold = 0.02f;  ///< Fraction of samples saturated at +-7 to trigger flag (2%)

    bool enable_morphology_sir = true;            ///< Morphological dilation to cover pulse edges & skirts
    int edge_channels_to_ignore = 0;              ///< Exclude PFB roll-off channels from passband baseline
    bool enable_persistent_ema = true;            ///< Track slow cross-frame temporal median baseline (EMA)
    float ema_alpha = 0.02f;                      ///< Exponential moving average update factor per clean frame

    /// Optional static frequency notch intervals [start_mhz, end_mhz]
    std::vector<std::pair<double, double>> static_notch_bands_mhz;
};

/**
 * @brief Pass 1: Coalesced per-(chunk, freq) scale-invariant Spectral Kurtosis, mean power,
 *        and quantizer saturation inspection across active antennas.
 *
 * @param d_voltages             Raw packed 4+4 bit complex voltages [n_time][n_freq][n_elem]
 * @param d_active_antennas      Device array of active raw element indices [num_active_antennas]
 * @param num_active_antennas    Number of active elements (> 0)
 * @param d_chunk_power          Output [n_chunks * n_freq] mean power per chunk
 * @param d_chunk_sk             Output [n_chunks * n_freq] scale-invariant Spectral Kurtosis per chunk
 * @param d_chunk_sat_fraction   Optional output [n_chunks * n_freq] fraction of samples saturating at [-7, +7]
 * @param n_time                 Total time samples per frame (e.g. 15360)
 * @param n_freq                 Number of local frequency channels (e.g. 336)
 * @param n_elem                 Total raw elements in voltage buffer (e.g. 64)
 * @param chunk_size             Time samples per chunk (e.g. 256)
 * @param sample_stride          Time stride inside chunk (1 = all samples)
 * @param stream                 CUDA stream
 * @param is_swapped_withoffset  True if input is INT4X2_SWAPPED_WITH_OFFSET
 */
void launch_inspect_rfi_chunks(const int4x2_t* __restrict__ d_voltages,
                               const int* __restrict__ d_active_antennas, int num_active_antennas,
                               float* __restrict__ d_chunk_power, float* __restrict__ d_chunk_sk,
                               float* __restrict__ d_chunk_sat_fraction,
                               std::size_t n_time, std::size_t n_freq, std::size_t n_elem,
                               std::size_t chunk_size, std::size_t sample_stride,
                               cudaStream_t stream, bool is_swapped_withoffset = false);

inline void launch_inspect_rfi_chunks(const int4x2_t* __restrict__ d_voltages,
                                      const int* __restrict__ d_active_antennas, int num_active_antennas,
                                      float* __restrict__ d_chunk_power, float* __restrict__ d_chunk_sk,
                                      std::size_t n_time, std::size_t n_freq, std::size_t n_elem,
                                      std::size_t chunk_size, std::size_t sample_stride,
                                      cudaStream_t stream, bool is_swapped_withoffset = false) {
    launch_inspect_rfi_chunks(d_voltages, d_active_antennas, num_active_antennas, d_chunk_power,
                              d_chunk_sk, nullptr, n_time, n_freq, n_elem, chunk_size,
                              sample_stride, stream, is_swapped_withoffset);
}

/**
 * @brief Pass 2: Evaluate per-channel and per-chunk RFI flags (SK bounds, temporal burst MAD,
 *        iterative outlier-rejected frequency passband MAD, saturation detection, and SIR morphology).
 *
 * @param d_chunk_power          Input [n_chunks * n_freq] mean power per chunk
 * @param d_chunk_sk             Input [n_chunks * n_freq] Spectral Kurtosis per chunk
 * @param d_chunk_sat_fraction   Optional input [n_chunks * n_freq] saturation fraction per chunk
 * @param d_static_notch         Input [n_freq] uint8 static notch mask (1 = notched, 0 = unnotched)
 * @param d_channel_gains        Optional input [n_freq] digital gains for pre-quantizer spectral unweighting
 * @param d_chunk_rfi_mask       Output [n_chunks * n_freq] uint8 bitmask of RfiFlagReason
 * @param d_channel_stats        Output [n_freq] RfiChannelStats summary
 * @param n_chunks               Number of time chunks per frame
 * @param n_freq                 Number of local frequency channels
 * @param num_active_antennas    Number of active antennas used in Pass 1
 * @param eff_samples_per_chunk  Effective time samples inspected per chunk (chunk_size / stride)
 * @param cfg                    RFI mitigation thresholds
 * @param stream                 CUDA stream
 */
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
                               const std::uint8_t* __restrict__ d_exempt_channels = nullptr,
                               float* __restrict__ d_ema_power = nullptr,
                               float* __restrict__ d_ema_mad = nullptr);

inline void launch_evaluate_rfi_flags(const float* __restrict__ d_chunk_power,
                                      const float* __restrict__ d_chunk_sk,
                                      const std::uint8_t* __restrict__ d_static_notch,
                                      std::uint8_t* __restrict__ d_chunk_rfi_mask,
                                      RfiChannelStats* __restrict__ d_channel_stats, std::size_t n_chunks,
                                      std::size_t n_freq, int num_active_antennas,
                                      std::size_t eff_samples_per_chunk, const RfiMitigationConfig& cfg,
                                      cudaStream_t stream) {
    launch_evaluate_rfi_flags(d_chunk_power, d_chunk_sk, nullptr, d_static_notch, nullptr,
                              d_chunk_rfi_mask, d_channel_stats, n_chunks, n_freq,
                              num_active_antennas, eff_samples_per_chunk, cfg, stream, nullptr);
}

/**
 * @brief Pass 3: Coalesced in-place voltage blanking of flagged (chunk, freq) cells.
 *        Unflagged cells are untouched.
 */
void launch_blank_rfi_chunks(int4x2_t* __restrict__ d_voltages,
                             const std::uint8_t* __restrict__ d_chunk_rfi_mask, std::size_t n_time,
                             std::size_t n_freq, std::size_t n_elem, std::size_t chunk_size,
                             cudaStream_t stream, bool is_swapped_withoffset = false);

} // namespace kotekan

#endif // CUDA_RFI_MITIGATION_HPP
