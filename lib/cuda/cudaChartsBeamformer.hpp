#ifndef CUDA_CHARTS_BEAMFORMER_HPP
#define CUDA_CHARTS_BEAMFORMER_HPP

#include "cudaChartsUpchannelizer.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace beamformer {

// -----------------------------------------------------------------------------
// Core CHARTS Tiled Beamformer, Integration & Quantization Constants
// -----------------------------------------------------------------------------
inline constexpr std::size_t frequency_shard_count = 2;
inline constexpr std::size_t full_band_frequency_channels =
    local_frequency_channels * frequency_shard_count;

inline constexpr std::size_t maximum_beams = 128;
inline constexpr std::size_t tiled_weight_beam_tile = 32;

inline constexpr double speed_of_light_m_per_s = 299792458.0;
inline constexpr double two_pi = 6.283185307179586476925286766559;

// 1 x 16 x 16 CHIME-style int8 chunk quantization constants
inline constexpr std::size_t quantization_chunk_time = 1;
inline constexpr std::size_t quantization_chunk_frequency = 16;
inline constexpr std::size_t quantization_chunk_beam = 16;
inline constexpr int quantized_min_code = -127;
inline constexpr int quantized_max_code = 127;
inline constexpr int quantized_invalid_code = -128;

// -----------------------------------------------------------------------------
// Core Types & Enums
// -----------------------------------------------------------------------------
struct ComplexFloat {
    float real = 0.0F;
    float imag = 0.0F;
};

using Vec3 = std::array<float, 3>;
using Weights = std::vector<ComplexFloat>;
using TiledWeights = std::vector<ComplexFloat>;

struct Dimensions {
    std::size_t n_time = 15360;
    std::size_t n_freq = local_frequency_channels;
    std::size_t n_ant = 64;
    std::size_t n_beams = 64;
};

enum class BeamformerInputMode {
    RawCoarse,
    ChannelizedFine,
};

struct Int8QuantizationParameters {
    float offset = 0.0F;
    float scale = 0.0F;
};

// -----------------------------------------------------------------------------
// Tiled Weight & Integration Indexing Helpers
// -----------------------------------------------------------------------------
constexpr std::size_t weight_index(const std::size_t b, const std::size_t f,
                                   const std::size_t a, const Dimensions& dims) {
    return (b * dims.n_freq + f) * dims.n_ant + a;
}

constexpr std::size_t tiled_weight_beam_tiles(const Dimensions& dims) {
    return (dims.n_beams + tiled_weight_beam_tile - 1) / tiled_weight_beam_tile;
}

constexpr std::size_t tiled_weight_count(const Dimensions& dims) {
    return dims.n_freq * tiled_weight_beam_tiles(dims) * dims.n_ant * tiled_weight_beam_tile;
}

constexpr std::size_t tiled_weight_index(const std::size_t f, const std::size_t beam_tile,
                                         const std::size_t a, const std::size_t local_beam,
                                         const Dimensions& dims) {
    return (((f * tiled_weight_beam_tiles(dims) + beam_tile) * dims.n_ant + a)
            * tiled_weight_beam_tile + local_beam);
}

constexpr std::size_t integrated_time_count(const std::size_t n_time,
                                            const std::size_t integration_spectra) {
    return (n_time + integration_spectra - 1) / integration_spectra;
}

constexpr std::size_t quantization_parameter_count(const Dimensions& integrated_dims) {
    const std::size_t t_tiles =
        (integrated_dims.n_time + quantization_chunk_time - 1) / quantization_chunk_time;
    const std::size_t f_tiles =
        (integrated_dims.n_freq + quantization_chunk_frequency - 1) / quantization_chunk_frequency;
    const std::size_t b_tiles =
        (integrated_dims.n_beams + quantization_chunk_beam - 1) / quantization_chunk_beam;
    return t_tiles * f_tiles * b_tiles;
}

inline double channel_frequency_hz(const BeamformerInputMode mode, const std::size_t shard_id,
                                   const std::size_t local_channel,
                                   const double first_coarse_center_hz = 300e6,
                                   const double coarse_channel_width_hz = 300e3) {
    const std::size_t coarse_origin = shard_id * local_frequency_channels;
    if (mode == BeamformerInputMode::RawCoarse) {
        return first_coarse_center_hz
               + static_cast<double>(coarse_origin + local_channel) * coarse_channel_width_hz;
    }
    const std::size_t coarse_idx = local_channel / upchannelization_factor;
    const std::size_t fine_bin = local_channel % upchannelization_factor;
    return first_coarse_center_hz
           + (static_cast<double>(coarse_origin + coarse_idx) + upchannelized_bin_fraction(fine_bin))
                 * coarse_channel_width_hz;
}

// -----------------------------------------------------------------------------
// Host Geometry & Tiled Steering Weight Generation
// -----------------------------------------------------------------------------
std::vector<Vec3> default_positions(std::size_t n_ant = 64, float spacing_m = 0.6F);
std::vector<Vec3> default_beam_grid(std::size_t n_beams = 64, float l_step = 0.01F, float m = 0.0F);
std::vector<double> shard_frequencies_hz(BeamformerInputMode mode, std::size_t shard_id,
                                         double first_coarse_center_hz = 300e6,
                                         double coarse_channel_width_hz = 300e3);

TiledWeights generate_tiled_weights(const Dimensions& dims, const std::vector<Vec3>& positions_m,
                                    const std::vector<double>& frequencies_hz,
                                    const std::vector<Vec3>& beam_directions);

TiledWeights repack_weights_to_tiled(const Weights& canonical_weights, const Dimensions& dims);

// -----------------------------------------------------------------------------
// Tiled Beamforming, Temporal Integration & Quantization Device Launchers
// -----------------------------------------------------------------------------
void launch_tiled_packed_beamformer(cudaStream_t stream, const std::uint8_t* packed_voltage,
                                    const ComplexFloat* tiled_weights, float* intensity,
                                    const Dimensions& dims);

void launch_tiled_packed_integrated_beamformer(cudaStream_t stream,
                                               const std::uint8_t* packed_voltage,
                                               const ComplexFloat* tiled_weights,
                                               float* integrated_intensity,
                                               const Dimensions& dims,
                                               std::size_t integration_spectra);

void launch_quantize_integrated_intensity(cudaStream_t stream, const float* integrated_intensity,
                                          std::int8_t* quantized_intensity,
                                          Int8QuantizationParameters* parameters,
                                          const Dimensions& integrated_dims);

} // namespace beamformer

#endif // CUDA_CHARTS_BEAMFORMER_HPP
