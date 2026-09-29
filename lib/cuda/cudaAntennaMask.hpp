#ifndef CUDA_ANTENNA_MASK_HPP
#define CUDA_ANTENNA_MASK_HPP

#include "DataType.hpp"
#include "chartsConstants.hpp"

#include <cuda_runtime.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace kotekan {

constexpr std::size_t MAX_MASK_ANTENNAS = charts::constants::charts_total_antennas;

/**
 * @enum AntennaHealthStatus
 * @brief Discrete classification of antenna operational status.
 */
enum class AntennaHealthStatus : std::uint8_t {
    HEALTHY = 0,
    DEAD = 1,
    SATURATED = 2,
    MANUAL_MASK = 3
};

inline const char* antenna_health_status_to_string(AntennaHealthStatus s) {
    switch (s) {
        case AntennaHealthStatus::HEALTHY: return "HEALTHY";
        case AntennaHealthStatus::DEAD: return "DEAD";
        case AntennaHealthStatus::SATURATED: return "SATURATED";
        case AntennaHealthStatus::MANUAL_MASK: return "MANUAL_MASK";
        default: return "UNKNOWN";
    }
}

/**
 * @struct AntennaHealthMetrics
 * @brief Telemetry and health state per antenna.
 */
struct AntennaHealthMetrics {
    float power = 0.0f;              ///< Mean sample power: E[re^2 + im^2]
    float clipping_fraction = 0.0f;  ///< Fraction of samples hitting 4-bit ADC rails (+7 or -8)
    std::uint32_t clipped_count = 0; ///< Absolute count of clipped samples in frame
    AntennaHealthStatus status = AntennaHealthStatus::HEALTHY;
    std::uint16_t consecutive_healthy = 0;
};

/**
 * @struct AntennaMaskConfig
 * @brief Tunable thresholds and operational modes for antenna masking.
 */
struct AntennaMaskConfig {
    bool auto_detect_enabled = true;
    bool blank_voltages_enabled = true;   ///< Level 1: In-place zeroing of bad antennas in GPU memory
    float dead_power_threshold = 5.0f;    ///< Mean power <= this value classified as DEAD (open ADC noise floor ~0.25)
    float sat_power_threshold = 80.0f;    ///< Mean power >= this value classified as SATURATED
    float clip_fraction_threshold = 0.15f;///< Clipping fraction >= this value classified as SATURATED (15%)
    std::uint16_t revival_frames = 5;     ///< Required consecutive healthy frames to revive a masked antenna
    std::size_t sample_stride = 1;        ///< Stride over spectra (1 = inspect 100% of data, 4 = inspect 25%)

    std::array<std::uint8_t, MAX_MASK_ANTENNAS> manual_mask; ///< 1 = enabled, 0 = masked

    AntennaMaskConfig() {
        manual_mask.fill(1);
    }
};

// ============================================================================
// CUDA Kernel Launches
// ============================================================================

/**
 * @brief Inspects incoming voltages tensor [time][freq][antenna] and accumulates
 *        per-antenna total power and ADC rail clipping counts.
 *
 * @param d_voltages         Input complex voltages in int4x2_t format
 * @param d_antenna_powers   Output device array of accumulated powers [n_ant] (float)
 * @param d_antenna_clips    Output device array of accumulated clipping counts [n_ant] (uint32)
 * @param n_time             Time samples per frame
 * @param n_freq             Frequency channels
 * @param n_ant              Number of antenna elements
 * @param sample_stride      Subsampling stride across spectra
 * @param stream             CUDA stream
 */
void launch_inspect_antenna_health(
    const int4x2_t* __restrict__ d_voltages,
    float* __restrict__ d_antenna_powers,
    std::uint32_t* __restrict__ d_antenna_clips,
    std::size_t n_time,
    std::size_t n_freq,
    std::size_t n_ant,
    std::size_t sample_stride,
    cudaStream_t stream);

/**
 * @brief Blanks (zeroes) voltages in-place for specified bad antenna indices.
 *        Executes only for bad antennas, preserving 100% memory bandwidth on healthy data.
 *
 * @param d_voltages         Voltages buffer to blank in-place
 * @param d_bad_antennas     Device array containing indices of bad antennas
 * @param num_bad_antennas   Number of bad antennas
 * @param total_spectra      n_time * n_freq
 * @param n_ant              Number of antenna elements
 * @param stream             CUDA stream
 */
void launch_zero_bad_antennas(
    int4x2_t* __restrict__ d_voltages,
    const int* __restrict__ d_bad_antennas,
    int num_bad_antennas,
    std::size_t total_spectra,
    std::size_t n_ant,
    cudaStream_t stream);

/**
 * @brief Extracts the alive antennas from the full voltage buffer into a compact
 *        contiguous baseband buffer for disk streaming and recording.
 */
void launch_extract_alive_antennas(
    const int4x2_t* __restrict__ d_voltages,
    int4x2_t* __restrict__ d_alive_voltages,
    const int* __restrict__ d_alive_antennas,
    int num_to_extract,
    int max_alive_antennas,
    std::size_t total_spectra,
    std::size_t n_ant,
    cudaStream_t stream);

/**
 * @brief Injects simulated antenna faults (dead=0, saturated=+/-7, -8) directly into
 *        device voltage buffer in microseconds for testing & chaos engineering.
 */
void launch_inject_faults(
    int4x2_t* __restrict__ d_voltages,
    const std::uint8_t* __restrict__ d_fault_types,
    std::size_t total_spectra,
    std::size_t n_ant,
    unsigned int seed,
    cudaStream_t stream);

} // namespace kotekan

#endif // CUDA_ANTENNA_MASK_HPP
