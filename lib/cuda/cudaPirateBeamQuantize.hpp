#ifndef CUDA_PIRATE_BEAM_QUANTIZE_HPP
#define CUDA_PIRATE_BEAM_QUANTIZE_HPP

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace kotekan {

constexpr std::size_t PIRATE_SAMPLES_PER_MINICHUNK = 256;
constexpr std::size_t PIRATE_SCALES_BYTES_PER_CHAN = 4; // 2 * float16 (scale, offset)
constexpr std::size_t PIRATE_DATA_BYTES_PER_CHAN = PIRATE_SAMPLES_PER_MINICHUNK / 2;
constexpr std::size_t PIRATE_BYTES_PER_CHAN =
    PIRATE_SCALES_BYTES_PER_CHAN + PIRATE_DATA_BYTES_PER_CHAN; // 132

struct PirateBeamQuantizeConfig {
    std::size_t num_beams = 1;
    std::size_t max_beams = 1;
    std::size_t num_freq = 336;
    std::size_t num_pols = 1;
    std::size_t samples_per_data_set = 15360;
    std::size_t seq_per_frb_time_sample = 60;
    std::size_t rfi_chunk_size = 256;
    float rfi_mask_fraction_threshold = 0.5f;
    bool enable_incoherent_subtraction = true;
    bool enable_highpass_detrend = true;
    int highpass_window_samples = 155; // ~31 ms at 0.2 ms cadence
    bool use_robust_mad_scale = true;
    float stddev_cutoff = 3.0f;
};

/**
 * @brief Launches fused GPU FRB intensity decimation, RFI mask propagation, optional
 *        incoherent-beam subtraction, sliding high-pass detrending, and pirate v2
 *        float16 (scale, offset) + int4 minichunk packing.
 *
 * Output buffer `d_pirate_out` holds `num_minichunks` consecutive minichunks of size
 * `num_beams * num_freq * 132` bytes each, formatted for `pirateNetworkSend`:
 *   - float16 `[num_beams][num_freq][2]`: `(scale, offset)`, reconstructed as `scale * q + offset`
 *   - int4    `[num_beams][num_freq][256]`: two's-complement codes [-7, +7], `-8` = masked/NaN
 */
void launch_pirate_beam_quantize(const float2* d_formed_beams,
                                 const std::uint8_t* d_rfi_mask,
                                 float* d_scratch_intensity,
                                 std::uint8_t* d_scratch_valid,
                                 std::uint8_t* d_pirate_out,
                                 const PirateBeamQuantizeConfig& cfg,
                                 cudaStream_t stream);

} // namespace kotekan

#endif // CUDA_PIRATE_BEAM_QUANTIZE_HPP
