#ifndef CUDA_CHARTS_BEAMFORMER_HPP
#define CUDA_CHARTS_BEAMFORMER_HPP

#include "cudaChartsUpchannelizer.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace beamformer {

// -----------------------------------------------------------------------------
// Core CHARTS Tiled Beamformer, Integration & Quantization Constants
// -----------------------------------------------------------------------------
inline constexpr std::size_t frequency_shard_count = 2;
inline constexpr std::size_t full_band_frequency_channels =
    local_frequency_channels * frequency_shard_count;

// Beam count B: 128 (default) or 256, configured by num_beams (BEAM_256_PLAN.md WP15, user
// decision 2026-11-26). Weight files, the fp32 scratch and the products follow it.
inline constexpr std::size_t beam_count = 128; // default B
inline constexpr std::size_t max_beam_count = 256;
constexpr bool is_supported_beam_count(const std::size_t n_beams) {
    return n_beams == 128 || n_beams == 256;
}

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
// Canonical combined weights, complex64 [B][F][E] (the .npy interchange order).
using Weights = std::vector<ComplexFloat>;

struct Dimensions {
    std::size_t n_time = 15360;
    std::size_t n_freq = local_frequency_channels;
    std::size_t n_ant = 64;
    std::size_t n_beams = beam_count;
};

struct Int8QuantizationParameters {
    float offset = 0.0F;
    float scale = 0.0F;
};

// -----------------------------------------------------------------------------
// Weight & Integration Indexing Helpers
// -----------------------------------------------------------------------------
// Canonical complex64 weights [B][F][E].
constexpr std::size_t weight_index(const std::size_t b, const std::size_t f,
                                   const std::size_t a, const Dimensions& dims) {
    return (b * dims.n_freq + f) * dims.n_ant + a;
}

// Device fp16 weights (FP16_TC_ARCHITECTURE.md WP2): per fine channel f, two row-major
// [B][E] planes (0 = real, 1 = imaginary) of W16 = fp16_rn(W / s[f][b]), i.e. the
// [f][plane][b][e] layout. One channel is the 128 x 64 A-matrix pair of the tensor-core
// GEMM (leading dimension E, 32 KiB per channel at E = 64). Scales s are float32 [f][b].
constexpr std::size_t fp16_weight_plane_count = 2;

constexpr std::size_t fp16_weight_count(const Dimensions& dims) {
    return dims.n_freq * fp16_weight_plane_count * dims.n_beams * dims.n_ant;
}

constexpr std::size_t fp16_weight_index(const std::size_t f, const std::size_t plane,
                                        const std::size_t b, const std::size_t a,
                                        const Dimensions& dims) {
    return ((f * fp16_weight_plane_count + plane) * dims.n_beams + b) * dims.n_ant + a;
}

constexpr std::size_t weight_scale_index(const std::size_t f, const std::size_t b,
                                         const Dimensions& dims) {
    return f * dims.n_beams + b;
}

struct Fp16Weights {
    std::vector<__half> values;   // fp16_weight_count, layout fp16_weight_index
    std::vector<float> scale;     // n_freq * n_beams, layout weight_scale_index
    std::size_t zero_beam_channels = 0;     // (b, f) with all-zero weights: s = 0
    std::size_t subnormal_components = 0;   // nonzero W/s components stored as fp16 subnormals
    std::size_t flushed_components = 0;     // nonzero W/s components rounded to fp16 zero
    // |s*W16 - W| / max_e |W[b,f,e]| over the elements of nonzero (b, f).
    double max_relative_error = 0.0;
    double rms_relative_error = 0.0;
};

// Power-of-two scale per (b, f): s = 2^k with k the smallest integer such that
// s^2 >= max_e (Re W^2 + Im W^2) (squares of float32 values summed in double; the squares are
// exact, so the result does not depend on FMA contraction). W16 = fp16 round-to-nearest-even of
// W / s (exact division), so |W16| <= 1 and dyadic weights stay exact; s = 0 and W16 = 0 when
// all E weights of (b, f) are zero, which makes that beam's power exactly 0.
Fp16Weights convert_weights_fp16(const Weights& canonical_weights, const Dimensions& dims);

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

// int8 output beam selection (output_beam_groups): whole quantization beam groups of 16, so
// every 1x16x16 chunk keeps its content and its codes and (offset, scale) are bit-identical
// to the full product; only their destination changes. Output group g holds the computed
// beams 16*source[g] ... 16*source[g] + 15 (all B beams are always computed; B/16 groups).
inline constexpr std::size_t max_beam_group_count = max_beam_count / quantization_chunk_beam;
static_assert(max_beam_count % quantization_chunk_beam == 0, "B must be a multiple of 16");

struct BeamGroupSelection {
    std::uint32_t count = 0;                         // G = number of written groups, 1..B/16
    std::uint32_t source[max_beam_group_count] = {}; // computed group of output group g, increasing

    std::size_t written_beams() const {
        return count * quantization_chunk_beam;
    }
    // Computed (weight-file) beam index of every written beam, in output order.
    std::vector<std::size_t> written_beam_indices() const {
        std::vector<std::size_t> beams;
        beams.reserve(written_beams());
        for (std::uint32_t g = 0; g < count; ++g)
            for (std::size_t b = 0; b < quantization_chunk_beam; ++b)
                beams.push_back(source[g] * quantization_chunk_beam + b);
        return beams;
    }
    std::vector<std::size_t> groups() const {
        return std::vector<std::size_t>(source, source + count);
    }
};

inline std::size_t beam_group_count(const std::size_t n_beams) {
    if (n_beams == 0 || n_beams > max_beam_count || n_beams % quantization_chunk_beam != 0)
        throw std::invalid_argument("beam count must be a positive multiple of 16, at most 256");
    return n_beams / quantization_chunk_beam;
}

// Every group of B computed beams, in order (the default when output_beam_groups is absent).
inline BeamGroupSelection all_beam_groups(const std::size_t n_beams = beam_count) {
    BeamGroupSelection selection;
    selection.count = static_cast<std::uint32_t>(beam_group_count(n_beams));
    for (std::uint32_t g = 0; g < selection.count; ++g)
        selection.source[g] = g;
    return selection;
}

// Validates a configured list against B computed beams: non-empty, entries in 0..B/16 - 1,
// strictly increasing.
inline BeamGroupSelection make_beam_group_selection(const std::vector<std::int64_t>& groups,
                                                    const std::size_t n_beams = beam_count) {
    const std::size_t groups_max = beam_group_count(n_beams);
    if (groups.empty())
        throw std::invalid_argument("output_beam_groups must not be empty");
    if (groups.size() > groups_max)
        throw std::invalid_argument("output_beam_groups has more than "
                                    + std::to_string(groups_max) + " groups (B/16)");
    BeamGroupSelection selection;
    for (std::size_t i = 0; i < groups.size(); ++i) {
        if (groups[i] < 0 || groups[i] >= static_cast<std::int64_t>(groups_max))
            throw std::invalid_argument("output_beam_groups entries must be in 0.."
                                        + std::to_string(groups_max - 1)
                                        + " (beam group b // 16 of B = "
                                        + std::to_string(n_beams) + ")");
        if (i > 0 && groups[i] <= groups[i - 1])
            throw std::invalid_argument("output_beam_groups must be strictly increasing "
                                        "(no duplicates)");
        selection.source[i] = static_cast<std::uint32_t>(groups[i]);
    }
    selection.count = static_cast<std::uint32_t>(groups.size());
    return selection;
}

// Centre of local fine channel f*32 + u (U = 32 always).
inline double channel_frequency_hz(const std::size_t shard_id, const std::size_t local_channel,
                                   const double first_coarse_center_hz = 300e6,
                                   const double coarse_channel_width_hz = 300e3) {
    const std::size_t coarse_origin = shard_id * local_frequency_channels;
    const std::size_t coarse_idx = local_channel / upchannelization_factor;
    const std::size_t fine_bin = local_channel % upchannelization_factor;
    return first_coarse_center_hz
           + (static_cast<double>(coarse_origin + coarse_idx) + upchannelized_bin_fraction(fine_bin))
                 * coarse_channel_width_hz;
}

// -----------------------------------------------------------------------------
// Host Geometry & Steering Weight Generation
// -----------------------------------------------------------------------------
std::vector<Vec3> default_positions(std::size_t n_ant = 64, float spacing_m = 0.6F);
std::vector<Vec3> default_beam_grid(std::size_t n_beams = beam_count, float l_step = 0.01F, float m = 0.0F);
std::vector<double> shard_frequencies_hz(std::size_t shard_id,
                                         double first_coarse_center_hz = 300e6,
                                         double coarse_channel_width_hz = 300e3);

// Diagnostic steering weights exp(+i k n.p), canonical [B][F][E].
Weights generate_weights(const Dimensions& dims, const std::vector<Vec3>& positions_m,
                         const std::vector<double>& frequencies_hz,
                         const std::vector<Vec3>& beam_directions);

// -----------------------------------------------------------------------------
// Tiled Beamforming, Temporal Integration & Quantization Device Launchers
// -----------------------------------------------------------------------------
// Scalar fp32 reference beamformer (K = 10) on fp16 fine voltages [t][f][e] and fp16 weights
// with power-of-two scales (fp16_weight_index / weight_scale_index):
// P = sum over the window of (s*Yr)^2 + (s*Yi)^2, Y = sum_e W16[b,f,e] x[t,f,e] in fp32.
void launch_tiled_fp16_integrated_beamformer(cudaStream_t stream, const __half2* fine_voltage,
                                             const __half* weights_fp16, const float* weight_scale,
                                             float* integrated_intensity, const Dimensions& dims,
                                             std::size_t integration_spectra);

// Production beamformer (FP16_TC_ARCHITECTURE.md WP3): same inputs, output and power
// convention as the scalar reference, computed with fp16 tensor-core MMA (WMMA m16n16k16)
// and fp32 accumulation; one block per fine channel, 16 consecutive spectra per WMMA tile
// (windows may straddle tiles), staging and window sums pipelined with the MMAs.
// Requires E = 32 or 64, B a multiple of 16 up to 256, K = 10, sm_70 or newer, 32-byte
// aligned weights and 16-byte aligned fine voltages. Accumulation order and rounding inside
// the MMA are hardware-defined: not bit-identical to the scalar reference.
void launch_tensor_core_fp16_integrated_beamformer(cudaStream_t stream,
                                                   const __half2* fine_voltage,
                                                   const __half* weights_fp16,
                                                   const float* weight_scale,
                                                   float* integrated_intensity,
                                                   const Dimensions& dims,
                                                   std::size_t integration_spectra);

// Compiled resources of the tensor-core kernel on the current device, for the startup log.
struct TensorCoreKernelResources {
    int registers_per_thread = 0;
    std::size_t static_shared_bytes = 0;
    int threads_per_block = 0;
    int blocks_per_sm = 0; // resident blocks per SM (cudaOccupancyMaxActiveBlocksPerMultiprocessor)
};
TensorCoreKernelResources tensor_core_beamformer_resources(std::size_t n_ant);

// integrated_dims describes the full fp32 input [n_time][n_freq][n_beams = 128]. Only the
// selected beam groups are quantized: codes [n_time][n_freq][16G] and parameters
// [(t_tile * f_tiles + f_tile) * G + g].
void launch_quantize_integrated_intensity(cudaStream_t stream, const float* integrated_intensity,
                                          std::int8_t* quantized_intensity,
                                          Int8QuantizationParameters* parameters,
                                          const Dimensions& integrated_dims,
                                          const BeamGroupSelection& beam_groups);

} // namespace beamformer

#endif // CUDA_CHARTS_BEAMFORMER_HPP
