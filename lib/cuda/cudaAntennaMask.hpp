#ifndef CUDA_ANTENNA_MASK_HPP
#define CUDA_ANTENNA_MASK_HPP

#include "DataType.hpp"
#include "chartsConstants.hpp"
#include "cudaDirectBeamTracker.hpp" // for MAX_DIRECT_ANTENNAS, DirectPolLayout

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <string>
#include <vector>

namespace kotekan {

constexpr std::size_t MAX_MASK_ANTENNAS = MAX_DIRECT_ANTENNAS; // 512 elements

/**
 * @enum AntennaHealthStatus
 * @brief Discrete classification of antenna element operational status.
 */
enum class AntennaHealthStatus : std::uint8_t {
    HEALTHY = 0,
    DEAD = 1,
    SATURATED = 2,
    MANUAL_MASK = 3,
    DC_LOCKED = 4,    ///< Stuck bit / excessive DC offset (low AC variance)
    RFI_KURTOSIS = 5, ///< Non-Gaussian impulsive RFI (high kurtosis) or CW tone (low kurtosis)
    IQ_IMBALANCE = 6, ///< Broken Real (I) or Imaginary (Q) 4-bit channel
    OUTLIER_POWER = 7 ///< Robust Median/MAD Z-score power outlier relative to array
};

inline const char* antenna_health_status_to_string(AntennaHealthStatus s) {
    switch (s) {
        case AntennaHealthStatus::HEALTHY:
            return "HEALTHY";
        case AntennaHealthStatus::DEAD:
            return "DEAD";
        case AntennaHealthStatus::SATURATED:
            return "SATURATED";
        case AntennaHealthStatus::MANUAL_MASK:
            return "MANUAL_MASK";
        case AntennaHealthStatus::DC_LOCKED:
            return "DC_LOCKED";
        case AntennaHealthStatus::RFI_KURTOSIS:
            return "RFI_KURTOSIS";
        case AntennaHealthStatus::IQ_IMBALANCE:
            return "IQ_IMBALANCE";
        case AntennaHealthStatus::OUTLIER_POWER:
            return "OUTLIER_POWER";
        default:
            return "UNKNOWN";
    }
}

/**
 * @struct AntennaRawStats
 * @brief Raw per-element moment accumulators populated on the GPU in a single pass.
 *        48 bytes per element (alignas(16)), 24 KB total for 512 elements.
 */
struct alignas(16) AntennaRawStats {
    unsigned long long sum_re2; ///< Sum(re^2)
    unsigned long long sum_im2; ///< Sum(im^2)
    unsigned long long sum_p2;  ///< Sum((re^2 + im^2)^2) for kurtosis
    long long sum_re;           ///< Sum(re) for DC offset
    long long sum_im;           ///< Sum(im) for DC offset
    std::uint32_t clips;        ///< Count of samples hitting -8 or +7
    std::uint32_t pad;
};

/**
 * @struct AntennaHealthMetrics
 * @brief Rich telemetry and health state per antenna element.
 */
struct AntennaHealthMetrics {
    float power = 0.0f;             ///< Mean total power: E[re^2 + im^2]
    float ac_variance = 0.0f;       ///< True AC variance: E[|z|^2] - |E[z]|^2
    float dc_fraction = 0.0f;       ///< Fraction of power in DC: |E[z]|^2 / E[|z|^2]
    float clipping_fraction = 0.0f; ///< Fraction of nibbles hitting 4-bit rails (+7 or -8)
    float kurtosis = 2.0f;       ///< Complex kurtosis: E[|z|^4] / (E[|z|^2])^2 (~2.0 for Gaussian)
    float iq_imbalance = 0.0f;   ///< |E[re^2] - E[im^2]| / E[re^2 + im^2]
    float robust_z_score = 0.0f; ///< |P - Median(P)| / (1.4826 * MAD)
    std::uint32_t clipped_count = 0; ///< Absolute count of clipped nibbles in frame
    AntennaHealthStatus status = AntennaHealthStatus::HEALTHY;
    std::uint16_t consecutive_healthy = 0;
};

/**
 * @struct AntennaMaskConfig
 * @brief Tunable thresholds and array/polarization configuration for antenna masking.
 */
struct AntennaMaskConfig {
    std::size_t num_antennas = 64;
    std::size_t num_pols = 1;
    DirectPolLayout pol_layout = DirectPolLayout::INTERLEAVED;
    std::string antenna_order = "descending";
    bool couple_pol_masks = false; ///< If true, masking one pol of an antenna masks both pols

    bool auto_detect_enabled = true;
    bool blank_voltages_enabled = false;

    // Power & Saturation thresholds
    float dead_power_threshold = 5.0f;     ///< Mean AC variance or power <= this -> DEAD
    float sat_power_threshold = 80.0f;     ///< Mean power >= this -> SATURATED
    float clip_fraction_threshold = 0.15f; ///< Rail clip fraction >= this -> SATURATED

    // Advanced signal quality thresholds
    bool enable_dc_lock_check = true;
    float max_dc_fraction = 0.35f; ///< |E[z]|^2 / P >= this -> DC_LOCKED (stuck bits)

    bool enable_iq_imbalance_check = true;
    float max_iq_imbalance = 0.50f; ///< |P_re - P_im| / P >= this -> IQ_IMBALANCE

    bool enable_kurtosis_check = true;
    float min_kurtosis_threshold = 1.15f; ///< Kurtosis <= this -> CW tone / oscillator lock
    float max_kurtosis_threshold = 4.50f; ///< Kurtosis >= this -> Impulsive RFI / arcing

    bool enable_mad_outlier_check = true;
    float mad_sigma_threshold = 6.0f; ///< Robust Z-score >= this -> OUTLIER_POWER

    std::uint16_t revival_frames = 5;
    std::size_t sample_stride = 1;

    std::array<std::uint8_t, MAX_MASK_ANTENNAS> manual_mask;
    std::uint64_t mask_generation = 1;

    AntennaMaskConfig() {
        manual_mask.fill(1);
    }

    std::size_t total_elements() const {
        return num_antennas * num_pols;
    }

    std::size_t raw_elem_to_physical_ant(std::size_t raw_elem) const {
        const std::size_t n_elem = total_elements();
        if (n_elem == 0)
            return 0;
        const bool desc = (antenna_order == "descending");
        const std::size_t ordered_elem = desc ? ((n_elem - 1) - raw_elem) : raw_elem;
        if (num_pols <= 1)
            return ordered_elem;
        return (pol_layout == DirectPolLayout::INTERLEAVED) ? (ordered_elem / num_pols)
                                                            : (ordered_elem % num_antennas);
    }

    std::size_t raw_elem_to_pol(std::size_t raw_elem) const {
        if (num_pols <= 1)
            return 0;
        const std::size_t n_elem = total_elements();
        const bool desc = (antenna_order == "descending");
        const std::size_t ordered_elem = desc ? ((n_elem - 1) - raw_elem) : raw_elem;
        return (pol_layout == DirectPolLayout::INTERLEAVED) ? (ordered_elem % num_pols)
                                                            : (ordered_elem / num_antennas);
    }
};

// ============================================================================
// CUDA Kernel Launches
// ============================================================================

/**
 * @brief Full multi-moment inspection kernel: accumulates exact integer moments
 *        (sum_re, sum_im, sum_re2, sum_im2, sum_p2, clips) in a single pass.
 */
void launch_inspect_antenna_moments(const int4x2_t* __restrict__ d_voltages,
                                    AntennaRawStats* __restrict__ d_raw_stats,
                                    float* __restrict__ d_antenna_powers,
                                    std::uint32_t* __restrict__ d_antenna_clips, std::size_t n_time,
                                    std::size_t n_freq, std::size_t n_elem,
                                    std::size_t sample_stride, cudaStream_t stream);

/**
 * @brief Legacy-compatible wrapper for launch_inspect_antenna_health.
 */
void launch_inspect_antenna_health(const int4x2_t* __restrict__ d_voltages,
                                   float* __restrict__ d_antenna_powers,
                                   std::uint32_t* __restrict__ d_antenna_clips, std::size_t n_time,
                                   std::size_t n_freq, std::size_t n_ant, std::size_t sample_stride,
                                   cudaStream_t stream);

/**
 * @brief Coalesced in-place voltage blanking for bad antenna elements.
 */
void launch_zero_bad_antennas(int4x2_t* __restrict__ d_voltages,
                              const int* __restrict__ d_bad_antennas, int num_bad_antennas,
                              std::size_t total_spectra, std::size_t n_ant, cudaStream_t stream);

/**
 * @brief Coalesced warp-per-spectrum extraction of alive antennas into a compact buffer.
 */
void launch_extract_alive_antennas(const int4x2_t* __restrict__ d_voltages,
                                   int4x2_t* __restrict__ d_alive_voltages,
                                   const int* __restrict__ d_alive_antennas, int num_to_extract,
                                   int max_alive_antennas, std::size_t total_spectra,
                                   std::size_t n_ant, cudaStream_t stream);

/**
 * @brief Chaos-engineering fault injection kernel (dead=1, rail_sat=2, dc_stuck=3, rfi_spike=4).
 */
void launch_inject_faults(int4x2_t* __restrict__ d_voltages,
                          const std::uint8_t* __restrict__ d_fault_types, std::size_t total_spectra,
                          std::size_t n_ant, unsigned int seed, cudaStream_t stream);

} // namespace kotekan

#endif // CUDA_ANTENNA_MASK_HPP
