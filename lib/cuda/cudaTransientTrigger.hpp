#ifndef CUDA_TRANSIENT_TRIGGER_HPP
#define CUDA_TRANSIENT_TRIGGER_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <string>
#include <vector>

namespace kotekan {

/**
 * @struct TransientTriggerConfig
 * @brief Configuration parameters for real-time transient detection and VRAM ring buffer dumping.
 */
struct TransientTriggerConfig {
    bool enabled = true;
    float sk_threshold = 0.08f;        ///< Threshold |SK(f) - 1.0| > sk_threshold to flag channel
    float rfi_threshold = 0.30f;       ///< Threshold mean(R01) < rfi_threshold to reject common RFI
    uint32_t min_flagged_channels = 8; ///< Minimum flagged channels to trigger candidate event
    uint32_t ring_buffer_depth = 32;   ///< Circular ring buffer depth in frames
    uint32_t pre_trigger_frames = 4;   ///< Leading frames prior to trigger to extract
    uint32_t post_trigger_frames = 4;  ///< Trailing frames after trigger to extract
    std::string dump_directory = "./transient_dumps"; ///< Output directory for candidate disk dumps
    bool auto_dump_enabled = true;  ///< Automatically dump candidate frames on trigger fire
    uint32_t cooldown_frames = 200; ///< Refractory cooldown period in frames (~10.2s) between dumps
    uint32_t min_free_disk_gb = 20; ///< Minimum free disk space in GB required to write dumps
    uint32_t max_auto_dumps = 20;   ///< Maximum automated candidate dumps per session
    bool enable_multibeam_veto = true; ///< Reject common-mode transients across multiple beams
    float min_sidelobe_confidence = 0.5f; ///< Minimum SPOTLIGHT C = SNR * (1 - P_side/P_peak)
    float max_sidelobe_ratio = 0.70f;     ///< Maximum allowed non-peak/peak excess power ratio
    uint32_t subframe_chunk_samples = 256; ///< Sub-frame integration chunk size in samples
};

/**
 * @struct TransientSubframeMetrics
 * @brief Sub-frame multi-beam coincidence & sidelobe discrimination summary.
 */
struct TransientSubframeMetrics {
    float max_sidelobe_confidence = 0.0f;
    float min_sidelobe_ratio = 1.0f;
    float mean_sidelobe_ratio = 1.0f;
    uint32_t peak_beam_id = 0;
    uint32_t peak_chunk_id = 0;
    uint32_t valid_chunks = 0;
};

/**
 * @struct TransientFrameMetrics
 * @brief Detection metrics and trigger decision for a single data frame.
 */
struct TransientFrameMetrics {
    uint32_t frame_id = 0;
    uint64_t timestamp_ns = 0;
    uint32_t flagged_channels = 0;
    float mean_sk = 1.0f;
    float min_sk = 1.0f;
    float max_sk = 1.0f;
    float median_sk = 1.0f;
    float mean_r01 = 0.0f;
    float max_sidelobe_confidence = 0.0f;
    float sidelobe_ratio = 0.0f;
    uint32_t peak_beam_id = 0;
    bool trigger_fired = false;
};

/**
 * @brief Launch the high-performance single-pass transient detection metrics kernel.
 *
 * Evaluates Spectral Kurtosis (SK) on beam 0 and normalized Inter-Beam Cross-Correlation (R01)
 * between beam 0 and beam 1 across all time samples for every frequency channel.
 *
 * @param d_voltages        Input baseband voltages [time][freq][beams] of float2
 * @param d_metrics         Output metrics array [freq] of float2 (.x = SK(f), .y = R01(f))
 * @param n_time            Number of time samples per frame (e.g. 15,360)
 * @param n_freq            Number of coarse frequency channels (e.g. 336)
 * @param num_beams         Number of active virtual beams (e.g. 2)
 * @param max_beams_stride  Stride across beam dimension
 * @param stream            CUDA stream for asynchronous execution
 */
void launch_transient_detection_metrics(const float2* d_voltages, float2* d_metrics,
                                        std::size_t n_time, std::size_t n_freq,
                                        std::size_t num_beams, std::size_t max_beams_stride,
                                        cudaStream_t stream = 0,
                                        const std::uint8_t* d_rfi_mask = nullptr,
                                        std::size_t chunk_size = 256);

/**
 * @brief Evaluate sub-frame multi-beam peak-to-sidelobe ratio and SPOTLIGHT confidence C.
 */
void launch_transient_subframe_coincidence(
    const float2* d_voltages, TransientSubframeMetrics* d_out, std::size_t n_time,
    std::size_t n_freq, std::size_t num_beams, std::size_t max_beams_stride,
    std::size_t chunk_samples = 256, cudaStream_t stream = 0,
    const std::uint8_t* d_rfi_mask = nullptr, std::size_t rfi_chunk_size = 256,
    float* d_scratch_chunk_power = nullptr);

/**
 * @brief Host evaluation of the combined detection logic over channel metrics.
 *
 * Condition:
 * Trigger Fire = (Sum II[|SK(f) - 1.0| > theta_SK] >= N_min) AND (mean(R01) < theta_RFI)
 *                AND (multi-beam sidelobe veto when subframe != nullptr && num_beams >= 2)
 */
inline TransientFrameMetrics evaluate_transient_decision(
    const float2* h_metrics, std::size_t n_freq, const TransientTriggerConfig& config,
    uint32_t frame_id, uint64_t timestamp_ns,
    const TransientSubframeMetrics* subframe = nullptr) {

    TransientFrameMetrics out;
    out.frame_id = frame_id;
    out.timestamp_ns = timestamp_ns;

    if (!config.enabled || n_freq == 0) {
        return out;
    }

    uint32_t flagged = 0;
    double sum_sk = 0.0;
    double sum_r01 = 0.0;
    double sum_flagged_r01 = 0.0;

    std::vector<float> sk_values(n_freq);
    float min_sk = 1e9f;
    float max_sk = -1e9f;

    for (std::size_t f = 0; f < n_freq; ++f) {
        const float sk = h_metrics[f].x;
        const float r01 = h_metrics[f].y;

        sk_values[f] = sk;
        if (sk < min_sk)
            min_sk = sk;
        if (sk > max_sk)
            max_sk = sk;

        sum_sk += static_cast<double>(sk);
        sum_r01 += static_cast<double>(r01);

        if (std::abs(sk - 1.0f) > config.sk_threshold) {
            flagged++;
            sum_flagged_r01 += static_cast<double>(r01);
        }
    }

    std::size_t mid = n_freq / 2;
    std::nth_element(sk_values.begin(), sk_values.begin() + mid, sk_values.end());

    out.flagged_channels = flagged;
    out.mean_sk = static_cast<float>(sum_sk / static_cast<double>(n_freq));
    out.min_sk = min_sk;
    out.max_sk = max_sk;
    out.median_sk = sk_values[mid];
    out.mean_r01 = static_cast<float>(flagged > 0 ? (sum_flagged_r01 / static_cast<double>(flagged))
                                                  : (sum_r01 / static_cast<double>(n_freq)));

    const bool sk_condition = (flagged >= config.min_flagged_channels);
    const bool rfi_condition = (out.mean_r01 < config.rfi_threshold);

    bool multibeam_ok = true;
    if (subframe != nullptr && config.enable_multibeam_veto && subframe->valid_chunks > 0) {
        out.max_sidelobe_confidence = subframe->max_sidelobe_confidence;
        out.sidelobe_ratio = subframe->min_sidelobe_ratio;
        out.peak_beam_id = subframe->peak_beam_id;
        multibeam_ok = (subframe->min_sidelobe_ratio <= config.max_sidelobe_ratio)
                       && (subframe->max_sidelobe_confidence >= config.min_sidelobe_confidence);
    }

    out.trigger_fired = sk_condition && rfi_condition && multibeam_ok;
    return out;
}

} // namespace kotekan

#endif // CUDA_TRANSIENT_TRIGGER_HPP
