#ifndef CUDA_CHARTS_UPCHANNELIZER_HPP
#define CUDA_CHARTS_UPCHANNELIZER_HPP

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace beamformer {

// -----------------------------------------------------------------------------
// CHARTS U=32 4-Tap PFB / FFT Upchannelizer Constants
// -----------------------------------------------------------------------------
inline constexpr std::size_t local_frequency_channels = 336;
inline constexpr std::size_t upchannelization_factor = 32;
inline constexpr std::size_t fine_frequency_channels =
    local_frequency_channels * upchannelization_factor; // 10752
inline constexpr std::size_t pfb_taps_per_phase = 4;
inline constexpr std::size_t pfb_prototype_length =
    upchannelization_factor * pfb_taps_per_phase; // 128
inline constexpr std::size_t pfb_history_samples =
    (pfb_taps_per_phase - 1) * upchannelization_factor; // 96
inline constexpr std::size_t pfb_history_spectra = pfb_taps_per_phase - 1; // 3

// -----------------------------------------------------------------------------
// Signed int4x2 Codec (Real in low nibble, Imaginary in high nibble, [-8..7])
// -----------------------------------------------------------------------------
struct ComplexInt4 {
    std::int8_t real = 0;
    std::int8_t imag = 0;
};

#if defined(__CUDACC__)
#define CHARTS_UPCHAN_HOST_DEVICE __host__ __device__
#else
#define CHARTS_UPCHAN_HOST_DEVICE
#endif

CHARTS_UPCHAN_HOST_DEVICE constexpr std::int8_t decode_signed_nibble(const std::uint8_t nibble) {
    const auto bits = static_cast<std::uint8_t>(nibble & 0x0F);
    return bits < 8 ? static_cast<std::int8_t>(bits)
                    : static_cast<std::int8_t>(static_cast<int>(bits) - 16);
}

CHARTS_UPCHAN_HOST_DEVICE constexpr ComplexInt4 unpack_complex_int4(const std::uint8_t packed) {
    return {decode_signed_nibble(packed),
            decode_signed_nibble(static_cast<std::uint8_t>(packed >> 4))};
}

inline std::uint8_t pack_complex_int4(const std::int8_t real, const std::int8_t imag) {
    if (real < -8 || real > 7 || imag < -8 || imag > 7) {
        throw std::out_of_range("int4 components must be in [-8, 7]");
    }
    const auto real_bits = static_cast<std::uint8_t>(real) & 0x0F;
    const auto imag_bits = static_cast<std::uint8_t>(imag) & 0x0F;
    return static_cast<std::uint8_t>(real_bits | (imag_bits << 4));
}

#undef CHARTS_UPCHAN_HOST_DEVICE

// -----------------------------------------------------------------------------
// PFB Prototype Filter & Fine-Bin Frequency Offset Helpers
// -----------------------------------------------------------------------------
inline float sinc_hanning_pfb_weight(const std::size_t tap) {
    if (tap >= pfb_prototype_length) {
        throw std::out_of_range("PFB tap is outside the 128-coefficient prototype");
    }
    constexpr double pi = 3.141592653589793238462643383279502884;
    constexpr double length = static_cast<double>(pfb_prototype_length);
    const double coordinate =
        (2.0 * static_cast<double>(tap) - (length - 1.0)) / (2.0 * (length + 1.0));
    const double argument = static_cast<double>(pfb_taps_per_phase) * coordinate;
    const double sinc = argument == 0.0 ? 1.0 : std::sin(pi * argument) / (pi * argument);
    const double hanning = std::cos(pi * coordinate) * std::cos(pi * coordinate);
    return static_cast<float>(hanning * sinc / static_cast<double>(upchannelization_factor));
}

inline double upchannelized_bin_fraction(const std::size_t fine_bin) {
    return (static_cast<double>(fine_bin) - static_cast<double>(upchannelization_factor - 1) * 0.5)
           / static_cast<double>(upchannelization_factor);
}

// -----------------------------------------------------------------------------
// Device Stage Launchers (Caller-Owned Buffers & Stream, Non-Blocking)
// -----------------------------------------------------------------------------
void launch_fused_upchannelizer(cudaStream_t stream, const std::uint8_t* raw_payload,
                                const std::uint8_t* history_samples, bool history_initialized,
                                std::uint8_t* channelized_packed, std::size_t raw_n_time,
                                std::size_t n_coarse = local_frequency_channels,
                                std::size_t n_element = 64, float quantize_clamp = 7.0F);

void launch_pfb_history_update(cudaStream_t stream, const std::uint8_t* raw_payload,
                               std::uint8_t* history_samples, std::size_t raw_n_time,
                               std::size_t n_coarse = local_frequency_channels,
                               std::size_t n_element = 64);

void launch_expand_missing_to_validity(cudaStream_t stream, const std::uint8_t* raw_missing_flags,
                                       std::uint8_t* expanded_validity, std::size_t raw_n_time,
                                       std::size_t n_coarse = local_frequency_channels);

// Compact time-only validity: handler flags apply to every coarse channel.
void launch_pfb_validity(cudaStream_t stream, const std::uint8_t* raw_validity,
                         const std::uint8_t* history_validity, bool history_initialized,
                         std::uint8_t* fine_validity, std::size_t raw_n_time);

// Counts [ceil(n_time/K)][336]. Fine K=10 uses uint8; coarse K=10/320 uses uint16.
// Tail windows count only supplied spectra. This never gates or rescales intensity.
void launch_integrated_exposure(cudaStream_t stream, const std::uint8_t* validity,
                                void* exposure, std::size_t n_time,
                                std::size_t integration_spectra, bool fine_mode);

} // namespace beamformer

#endif // CUDA_CHARTS_UPCHANNELIZER_HPP
