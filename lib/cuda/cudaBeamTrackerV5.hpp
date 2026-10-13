#ifndef CUDA_BEAM_TRACKER_V5_HPP
#define CUDA_BEAM_TRACKER_V5_HPP

#include "DataType.hpp" // for kotekan::int4x2_t
#include "chartsConstants.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <memory>
#include <vector>

namespace kotekan {

constexpr std::size_t MAX_TRACKER_BEAMS = 8;
constexpr std::size_t MAX_TRACKER_ANTENNAS = charts::constants::charts_total_antennas;

// 3D vector for directions
struct Direction3D {
    float x = 0.0f;
    float y = 0.0f;
    float z = 1.0f;
};

// 2D vector for direction rates (dl/dt, dm/dt in direction cosine space)
struct DirectionRate2D {
    float dl = 0.0f;
    float dm = 0.0f;
};

// Celestial Equatorial Coordinates (RA, Dec)
struct CelestialTarget {
    double ra_deg = 0.0;
    double dec_deg = 0.0;
    bool is_set = false;
};

// Telescope geographic site location (defaults to CHARTS Carén site)
struct SiteLocation {
    double lat_deg = charts::constants::charts_caren_lat_deg; // Latitude (degrees)
    double lon_deg = charts::constants::charts_caren_lon_deg; // Longitude (degrees)
    double alt_m = charts::constants::charts_caren_alt_m;     // Altitude (meters)
};

// Trajectory definition
struct BeamTrackerTrajectory {
    Direction3D direction_start{0.0f, 0.0f, 1.0f};
    DirectionRate2D direction_rate_per_sample{0.0f, 0.0f};
    CelestialTarget celestial_target;
};

// Single-beam configuration for backward compatibility
struct BeamTrackerConfig {
    BeamTrackerTrajectory trajectory;
    std::size_t integration_spectra = 320;
    float spacing_m = charts::constants::charts_default_spacing_m;
    std::size_t time_chunk_size = 80;
    std::size_t time_unroll = 8; // 2, 4, or 8
    bool enable_cuda_graph = false;
    std::array<uint8_t, MAX_TRACKER_ANTENNAS> antenna_mask;
    std::array<float3, MAX_TRACKER_ANTENNAS> antenna_positions;
    SiteLocation site;
    BeamTrackerConfig() {
        antenna_mask.fill(1);
        for (std::size_t i = 0; i < MAX_TRACKER_ANTENNAS; ++i) {
            const unsigned int col = (i < 64) ? (i & 7U) : (i & 15U);
            const unsigned int row = (i < 64) ? (i >> 3U) : (i >> 4U);
            antenna_positions[i] = make_float3(static_cast<float>(col) * spacing_m,
                                               static_cast<float>(row) * spacing_m, 0.0f);
        }
    }
    void set_antenna_grid(std::size_t n_ant, float spacing) {
        spacing_m = spacing;
        for (std::size_t i = 0; i < MAX_TRACKER_ANTENNAS; ++i) {
            const unsigned int col = (n_ant <= 64) ? (i & 7U) : (i & 15U);
            const unsigned int row = (n_ant <= 64) ? (i >> 3U) : (i >> 4U);
            antenna_positions[i] = make_float3(static_cast<float>(col) * spacing_m,
                                               static_cast<float>(row) * spacing_m, 0.0f);
        }
    }
};

// Multi-beam configuration supporting dynamic active slot counts
struct MultiBeamTrackerConfig {
    std::size_t num_active_beams = 1;
    std::array<BeamTrackerTrajectory, MAX_TRACKER_BEAMS> trajectories;
    std::size_t integration_spectra = 320;
    float spacing_m = charts::constants::charts_default_spacing_m;
    std::size_t time_chunk_size = 80;
    std::size_t time_unroll = 8;
    bool enable_cuda_graph = false;
    std::array<uint8_t, MAX_TRACKER_ANTENNAS> antenna_mask;
    std::array<float3, MAX_TRACKER_ANTENNAS> antenna_positions;
    SiteLocation site;
    MultiBeamTrackerConfig() {
        antenna_mask.fill(1);
        for (std::size_t i = 0; i < MAX_TRACKER_ANTENNAS; ++i) {
            const unsigned int col = (i < 64) ? (i & 7U) : (i & 15U);
            const unsigned int row = (i < 64) ? (i >> 3U) : (i >> 4U);
            antenna_positions[i] = make_float3(static_cast<float>(col) * spacing_m,
                                               static_cast<float>(row) * spacing_m, 0.0f);
        }
    }
    void set_antenna_grid(std::size_t n_ant, float spacing) {
        spacing_m = spacing;
        for (std::size_t i = 0; i < MAX_TRACKER_ANTENNAS; ++i) {
            const unsigned int col = (n_ant <= 64) ? (i & 7U) : (i & 15U);
            const unsigned int row = (n_ant <= 64) ? (i >> 3U) : (i >> 4U);
            antenna_positions[i] = make_float3(static_cast<float>(col) * spacing_m,
                                               static_cast<float>(row) * spacing_m, 0.0f);
        }
    }
};

/**
 * @brief Astrometry helper: Converts Celestial (RA, Dec) + Local Sidereal Time to Topocentric
 * Direction Cosines (l, m, n) and rates (dl, dm).
 *
 * @param ra_deg            Right ascension in degrees
 * @param dec_deg           Declination in degrees
 * @param lst_hours         Local Sidereal Time in hours
 * @param lat_deg           Observatory latitude in degrees
 * @param sample_period_s   Time sample duration (dt in seconds)
 * @param out_traj          Trajectory struct to populate
 */
inline void compute_celestial_trajectory(double ra_deg, double dec_deg, double lst_hours,
                                         double lat_deg, double sample_period_s,
                                         BeamTrackerTrajectory& out_traj) {

    constexpr double DEG_TO_RAD = M_PI / 180.0;
    constexpr double HOURS_TO_RAD = M_PI / 12.0;
    constexpr double EARTH_ROT_RATE_RAD_S = 7.292115e-5; // rad / s

    const double ha_rad = lst_hours * HOURS_TO_RAD - ra_deg * DEG_TO_RAD;
    const double dec_rad = dec_deg * DEG_TO_RAD;
    const double lat_rad = lat_deg * DEG_TO_RAD;

    const double sin_dec = std::sin(dec_rad);
    const double cos_dec = std::cos(dec_rad);
    const double sin_lat = std::sin(lat_rad);
    const double cos_lat = std::cos(lat_rad);
    const double cos_ha = std::cos(ha_rad);
    const double sin_ha = std::sin(ha_rad);

    // Direction cosines in topocentric coordinate system (x = East, y = North, z = Up/Zenith)
    const double l = -cos_dec * sin_ha;
    const double m = cos_lat * sin_dec - sin_lat * cos_dec * cos_ha;
    const double trans_sq = l * l + m * m;
    const double n = (trans_sq <= 1.0) ? std::sqrt(1.0 - trans_sq) : 0.0;

    // Time derivatives (d(ha)/dt = omega_earth)
    const double dl_dt = -cos_dec * cos_ha * EARTH_ROT_RATE_RAD_S;
    const double dm_dt = sin_lat * cos_dec * sin_ha * EARTH_ROT_RATE_RAD_S;

    out_traj.direction_start =
        Direction3D{static_cast<float>(l), static_cast<float>(m), static_cast<float>(n)};
    out_traj.direction_rate_per_sample = DirectionRate2D{
        static_cast<float>(dl_dt * sample_period_s), static_cast<float>(dm_dt * sample_period_s)};
    out_traj.celestial_target.ra_deg = ra_deg;
    out_traj.celestial_target.dec_deg = dec_deg;
    out_traj.celestial_target.is_set = true;
}

/**
 * @brief Launch the V5 Beam Tracker kernel for a single beam with complex voltage output.
 */
void launch_beam_tracker_v5(const int4x2_t* d_packed, float2* d_voltages, std::size_t n_time,
                            std::size_t n_freq, std::size_t n_ant,
                            const std::vector<double>& frequencies_hz,
                            const BeamTrackerConfig& config, cudaStream_t stream = nullptr,
                            std::size_t window_offset = 0);

/**
 * @brief Launch the V5 Beam Tracker kernel for multiple dynamic beams with pre-allocated complex
 * voltage output slots.
 *
 * @param d_packed              Device pointer to packed input voltages [time][freq][antenna]
 * (int4x2_t)
 * @param d_voltages            Device pointer to output complex formed beams
 * [time][freq][max_beams_allocated] (float2: real, imag)
 * @param n_time                Number of time samples
 * @param n_freq                Number of frequency channels
 * @param n_ant                 Number of antennas (32, 64, 128, or 256)
 * @param max_beams_allocated   Total allocated beam stride in output buffer (e.g. 4 or 8)
 * @param frequencies_hz        Vector of physical frequencies in Hz
 * @param config                Multi-beam configuration with num_active_beams and trajectories
 * @param stream                CUDA stream
 * @param window_offset         Time window index offset
 */
void launch_beam_tracker_v5_multibeam(const int4x2_t* d_packed, float2* d_voltages,
                                      std::size_t n_time, std::size_t n_freq, std::size_t n_ant,
                                      std::size_t max_beams_allocated,
                                      const std::vector<double>& frequencies_hz,
                                      const MultiBeamTrackerConfig& config,
                                      cudaStream_t stream = nullptr, std::size_t window_offset = 0);

/**
 * @brief Persistent Batched Beam Tracker Stream with optional CUDA graph execution.
 */
class CudaBeamTrackerV5Stream {
public:
    CudaBeamTrackerV5Stream(std::size_t n_time_per_batch, std::size_t n_freq, std::size_t n_ant,
                            const std::vector<double>& frequencies_hz,
                            const BeamTrackerConfig& config);
    ~CudaBeamTrackerV5Stream();

    CudaBeamTrackerV5Stream(const CudaBeamTrackerV5Stream&) = delete;
    CudaBeamTrackerV5Stream& operator=(const CudaBeamTrackerV5Stream&) = delete;

    void process_batch(std::size_t window_offset, const int4x2_t* host_packed,
                       float2* host_voltages);

    void process_batch_device(std::size_t window_offset, const int4x2_t* d_packed,
                              float2* d_voltages);

    float last_kernel_time_ms() const;

    cudaStream_t get_stream() const;
    int4x2_t* device_packed_buffer();
    float2* device_voltages_buffer();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace kotekan

#endif // CUDA_BEAM_TRACKER_V5_HPP
