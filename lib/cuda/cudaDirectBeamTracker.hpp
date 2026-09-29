#ifndef CUDA_DIRECT_BEAM_TRACKER_HPP
#define CUDA_DIRECT_BEAM_TRACKER_HPP

#include "DataType.hpp" // for kotekan::int4x2_t
#include "chartsConstants.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include <cuda_runtime.h>

namespace kotekan {

constexpr std::size_t MAX_DIRECT_BEAMS = 8;
constexpr std::size_t MAX_DIRECT_ANTENNAS = charts::constants::charts_total_antennas;

// 3D vector for directions
struct DirectDirection3D {
    float x = 0.0f;
    float y = 0.0f;
    float z = 1.0f;
};

// Celestial Equatorial Coordinates (RA, Dec)
struct DirectCelestialTarget {
    double ra_deg = 0.0;
    double dec_deg = 0.0;
    bool is_set = false;
};

// Site location
struct DirectSiteLocation {
    double lat_deg = charts::constants::charts_caren_lat_deg;
    double lon_deg = charts::constants::charts_caren_lon_deg;
    double alt_m = charts::constants::charts_caren_alt_m;
};

// Target specification for each beam slot
struct DirectBeamTarget {
    DirectDirection3D direction{0.0f, 0.0f, 1.0f};
    DirectDirection3D direction_end{0.0f, 0.0f, 1.0f};
    DirectCelestialTarget celestial;
    int grid_index = -1; // -1 if continuous direction, >= 0 if locked to precomputed grid
};

// Direct Beam Tracker configuration
struct DirectBeamTrackerConfig {
    std::size_t num_active_beams = 1;
    std::array<DirectBeamTarget, MAX_DIRECT_BEAMS> targets;
    float spacing_m = charts::constants::charts_default_spacing_m;
    std::size_t time_chunk_size = 256;
    std::size_t time_unroll = 4; // 2, 4, or 8 (4 recommended with beam_tile_size=4 for optimal register pressure on SM120)
    std::size_t beam_tile_size = 4; // 1, 2, or 4 (4 fuses 4 beams into 1 warp, 4x DRAM bandwidth reduction)
    bool enable_grid_mode = false;
    bool enable_cuda_graph = false;
    std::array<uint8_t, MAX_DIRECT_ANTENNAS> antenna_mask;
    std::array<float3, MAX_DIRECT_ANTENNAS> antenna_positions;
    DirectSiteLocation site;

    std::size_t num_active_antennas = MAX_DIRECT_ANTENNAS;

    DirectBeamTrackerConfig() {
        antenna_mask.fill(1);
        num_active_antennas = MAX_DIRECT_ANTENNAS;
        for (std::size_t i = 0; i < MAX_DIRECT_ANTENNAS; ++i) {
            const unsigned int col = (i < 64) ? (i & 7U) : (i & 15U);
            const unsigned int row = (i < 64) ? (i >> 3U) : (i >> 4U);
            antenna_positions[i] = make_float3(static_cast<float>(col) * spacing_m,
                                              static_cast<float>(row) * spacing_m,
                                              0.0f);
        }
    }
    void set_antenna_grid(std::size_t n_ant, float spacing) {
        spacing_m = spacing;
        num_active_antennas = n_ant;
        for (std::size_t i = 0; i < MAX_DIRECT_ANTENNAS; ++i) {
            const unsigned int col = (n_ant <= 64) ? (i & 7U) : (i & 15U);
            const unsigned int row = (n_ant <= 64) ? (i >> 3U) : (i >> 4U);
            antenna_positions[i] = make_float3(static_cast<float>(col) * spacing_m,
                                              static_cast<float>(row) * spacing_m,
                                              0.0f);
        }
    }
};

/**
 * @brief Generate a list of (l, m) direction cosine grid points covering the visible hemisphere (l^2 + m^2 <= 1.0).
 *
 * @param step  Spacing between grid points in direction cosine units (e.g. 0.02 ~ 1.15 degrees).
 * @return Vector of float2 where x = l, y = m.
 */
std::vector<float2> generate_sky_grid_directions(float step = 0.02f);

/**
 * @brief Look up the nearest grid index in a direction cosine grid table.
 *
 * @param l                 Target l direction cosine
 * @param m                 Target m direction cosine
 * @param grid_directions   Vector of grid directions
 * @return Index of closest grid direction
 */
int lookup_nearest_sky_grid(float l, float m, const std::vector<float2>& grid_directions);

/**
 * @brief Astrometry helper: Converts Celestial (RA, Dec) + Local Sidereal Time to Topocentric Direction Cosines (l, m, n).
 */
inline void compute_celestial_direction(
    double ra_deg,
    double dec_deg,
    double lst_hours,
    double lat_deg,
    DirectDirection3D& out_dir) {

    constexpr double DEG_TO_RAD = M_PI / 180.0;
    constexpr double HOURS_TO_RAD = M_PI / 12.0;

    const double ha_rad = lst_hours * HOURS_TO_RAD - ra_deg * DEG_TO_RAD;
    const double dec_rad = dec_deg * DEG_TO_RAD;
    const double lat_rad = lat_deg * DEG_TO_RAD;

    const double sin_dec = std::sin(dec_rad);
    const double cos_dec = std::cos(dec_rad);
    const double sin_lat = std::sin(lat_rad);
    const double cos_lat = std::cos(lat_rad);
    const double cos_ha = std::cos(ha_rad);
    const double sin_ha = std::sin(ha_rad);

    const double l = -cos_dec * sin_ha;
    const double m = cos_lat * sin_dec - sin_lat * cos_dec * cos_ha;
    const double trans_sq = l * l + m * m;
    const double n = (trans_sq <= 1.0) ? std::sqrt(1.0 - trans_sq) : 0.0;

    out_dir.x = static_cast<float>(l);
    out_dir.y = static_cast<float>(m);
    out_dir.z = static_cast<float>(n);
}

/**
 * @brief Launch the Direct Beamformer kernel applying precalculated weights directly.
 *
 * @param d_packed              Device pointer to input voltages [time][freq][antenna] (int4x2_t)
 * @param d_weights             Device pointer to precalculated weights [beam][freq][antenna] (float2)
 * @param d_voltages            Device pointer to output formed beams [time][freq][max_beams_stride] (float2)
 * @param n_time                Number of time samples (e.g. 153,600)
 * @param n_freq                Number of frequency channels (e.g. 672)
 * @param n_ant                 Number of antennas (32, 64, 128, or 256)
 * @param num_active_beams      Number of active beams to process (1..8)
 * @param max_beams_stride      Total allocated beam stride in output buffer (e.g. 4 or 8)
 * @param time_chunk_size       Time chunk tile size per warp (default 256)
 * @param time_unroll           Unroll factor (default 4)
 * @param beam_tile_size        Number of concurrent beams fused per warp (1, 2, or 4; default 4 for SM120)
 * @param stream                CUDA stream
 */
/**
 * @brief Launch the Direct Beamformer kernel applying precalculated weights directly with optional sub-frame phase interpolation.
 *
 * @param d_packed              Device pointer to input voltages [time][freq][antenna] (int4x2_t)
 * @param d_weights             Device pointer to precalculated base weights [beam][freq][antenna] (float2)
 * @param d_step_weights        Device pointer to per-sample delta phasors [beam][freq][antenna] (float2, optional)
 * @param d_voltages            Device pointer to output formed beams [time][freq][max_beams_stride] (float2)
 * @param n_time                Number of time samples (e.g. 15,360)
 * @param n_freq                Number of frequency channels (e.g. 672)
 * @param n_ant                 Number of antennas (32, 64, 128, or 256)
 * @param num_active_beams      Number of active beams to process (1..8)
 * @param max_beams_stride      Total allocated beam stride in output buffer (e.g. 4 or 8)
 * @param time_chunk_size       Time chunk tile size per warp (default 256)
 * @param time_unroll           Unroll factor (default 4)
 * @param beam_tile_size        Number of concurrent beams fused per warp (1, 2, or 4; default 4 for SM120)
 * @param stream                CUDA stream
 */
void launch_direct_beamformer(
    const int4x2_t* d_packed,
    const float2* d_weights,
    const float2* d_step_weights,
    float2* d_voltages,
    std::size_t n_time,
    std::size_t n_freq,
    std::size_t n_ant,
    std::size_t num_active_beams,
    std::size_t max_beams_stride,
    std::size_t time_chunk_size = 256,
    std::size_t time_unroll = 4,
    std::size_t beam_tile_size = 4,
    cudaStream_t stream = nullptr);

inline void launch_direct_beamformer(
    const int4x2_t* d_packed,
    const float2* d_weights,
    float2* d_voltages,
    std::size_t n_time,
    std::size_t n_freq,
    std::size_t n_ant,
    std::size_t num_active_beams,
    std::size_t max_beams_stride,
    std::size_t time_chunk_size = 256,
    std::size_t time_unroll = 4,
    std::size_t beam_tile_size = 4,
    cudaStream_t stream = nullptr) {
    launch_direct_beamformer(
        d_packed, d_weights, nullptr, d_voltages,
        n_time, n_freq, n_ant, num_active_beams, max_beams_stride,
        time_chunk_size, time_unroll, beam_tile_size, stream);
}

/**
 * @brief Configure CUDA Stream L2 Access Policy Window to persist steering weights in Blackwell L2 cache.
 *
 * @param stream    CUDA stream
 * @param ptr       Pointer to device memory (e.g. _d_weights)
 * @param bytes     Size of the buffer in bytes
 */
void set_l2_persisting_weights_policy(cudaStream_t stream, const void* ptr, std::size_t bytes);

/**
 * @brief Compute / update steering weights on GPU for active beam targets with optional phase rotator generation.
 *
 * @param d_weights             Device pointer to destination base weights buffer [num_beams][n_freq][n_ant]
 * @param d_step_weights        Device pointer to destination step phasors buffer [num_beams][n_freq][n_ant] (optional)
 * @param d_directions_start    Device pointer to start beam directions [num_beams] (DirectDirection3D)
 * @param d_directions_end      Device pointer to end beam directions [num_beams] (DirectDirection3D, optional)
 * @param d_wavenumbers         Device pointer to wavenumbers [n_freq] (double)
 * @param d_antenna_positions   Device pointer to antenna positions [n_ant] (float3)
 * @param d_antenna_mask        Device pointer to antenna mask [n_ant] (uint8_t, optional)
 * @param d_calibration_gains   Device pointer to per-channel complex gains [n_freq][n_ant] (float2, optional)
 * @param num_beams             Number of beams
 * @param n_freq                Number of frequency channels
 * @param n_ant                 Number of antennas
 * @param n_time                Number of time samples in frame (e.g. 15,360)
 * @param n_active              Number of active unmasked antennas (for 1/sqrt(n_active) scaling)
 * @param stream                CUDA stream
 */
void launch_generate_steering_weights(
    float2* d_weights,
    float2* d_step_weights,
    const DirectDirection3D* d_directions_start,
    const DirectDirection3D* d_directions_end,
    const double* d_wavenumbers,
    const float3* d_antenna_positions,
    const std::uint8_t* d_antenna_mask,
    const float2* d_calibration_gains,
    std::size_t num_beams,
    std::size_t n_freq,
    std::size_t n_ant,
    std::size_t n_time,
    std::size_t n_active = 0,
    cudaStream_t stream = nullptr);

inline void launch_generate_steering_weights(
    float2* d_weights,
    const DirectDirection3D* d_directions,
    const double* d_wavenumbers,
    const float3* d_antenna_positions,
    const std::uint8_t* d_antenna_mask,
    const float2* d_calibration_gains,
    std::size_t num_beams,
    std::size_t n_freq,
    std::size_t n_ant,
    std::size_t n_active = 0,
    cudaStream_t stream = nullptr) {
    launch_generate_steering_weights(
        d_weights, nullptr, d_directions, nullptr,
        d_wavenumbers, d_antenna_positions, d_antenna_mask,
        d_calibration_gains, num_beams, n_freq, n_ant, 0,
        n_active, stream);
}

inline void launch_generate_steering_weights(
    float2* d_weights,
    const DirectDirection3D* d_directions,
    const double* d_wavenumbers,
    const float3* d_antenna_positions,
    const std::uint8_t* d_antenna_mask,
    const float2* d_calibration_gains,
    std::size_t num_beams,
    std::size_t n_freq,
    std::size_t n_ant,
    cudaStream_t stream) {
    launch_generate_steering_weights(
        d_weights, nullptr, d_directions, nullptr,
        d_wavenumbers, d_antenna_positions, d_antenna_mask,
        d_calibration_gains, num_beams, n_freq, n_ant, 0,
        0, stream);
}

/**
 * @brief Precompute steering weights for an entire sky grid in device memory.
 *
 * @param d_grid_weights        Device pointer to grid weights [num_grid_points][n_freq][n_ant]
 * @param d_grid_lms            Device pointer to grid (l, m) direction cosines [num_grid_points] (float2)
 * @param d_wavenumbers         Device pointer to wavenumbers [n_freq] (double)
 * @param d_antenna_positions   Device pointer to antenna positions [n_ant] (float3)
 * @param d_antenna_mask        Device pointer to antenna mask [n_ant] (uint8_t, optional)
 * @param d_calibration_gains   Device pointer to per-channel complex gains [n_freq][n_ant] (float2, optional)
 * @param num_grid_points       Number of discrete grid directions
 * @param n_freq                Number of frequency channels
 * @param n_ant                 Number of antennas
 * @param n_active              Number of active unmasked antennas (for 1/sqrt(n_active) scaling)
 * @param stream                CUDA stream
 */
void launch_precompute_sky_grid(
    float2* d_grid_weights,
    const float2* d_grid_lms,
    const double* d_wavenumbers,
    const float3* d_antenna_positions,
    const std::uint8_t* d_antenna_mask,
    const float2* d_calibration_gains,
    std::size_t num_grid_points,
    std::size_t n_freq,
    std::size_t n_ant,
    std::size_t n_active = 0,
    cudaStream_t stream = nullptr);

inline void launch_precompute_sky_grid(
    float2* d_grid_weights,
    const float2* d_grid_lms,
    const double* d_wavenumbers,
    const float3* d_antenna_positions,
    const std::uint8_t* d_antenna_mask,
    const float2* d_calibration_gains,
    std::size_t num_grid_points,
    std::size_t n_freq,
    std::size_t n_ant,
    cudaStream_t stream) {
    launch_precompute_sky_grid(
        d_grid_weights, d_grid_lms, d_wavenumbers, d_antenna_positions,
        d_antenna_mask, d_calibration_gains, num_grid_points, n_freq, n_ant,
        0, stream);
}

} // namespace kotekan

#endif // CUDA_DIRECT_BEAM_TRACKER_HPP
