#ifndef CUDA_DIRECT_BEAM_TRACKER_HPP
#define CUDA_DIRECT_BEAM_TRACKER_HPP

#include "DataType.hpp" // for kotekan::int4x2_t
#include "chartsConstants.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>

namespace kotekan {

constexpr std::size_t MAX_DIRECT_BEAMS = 64;
constexpr std::size_t MAX_DIRECT_ANTENNAS = 512;

enum class DirectPolLayout : int {
    INTERLEAVED = 0, // [time][freq][ant][pol] -> elem = ant * n_pol + pol
    PLANAR = 1       // [time][freq][pol][ant] -> elem = pol * n_ant + ant
};

enum class DirectTaperType : int {
    NONE = 0, // Uniform weights (first sidelobe ~ -13.2 dB)
    COSINE_PEDESTAL =
        1,       // Raised cosine on pedestal p (p=0.25 -> ~ -25 dB sidelobes, 86% efficiency)
    HAMMING = 2, // Classic 2D Hamming (p=0.08 -> ~ -40 dB sidelobes)
    BLACKMAN_PEDESTAL = 3, // 3-term Blackman with pedestal floor
    GAUSSIAN = 4           // Truncated Gaussian (edge value controlled by taper_pedestal)
};

inline DirectTaperType parse_taper_type(const std::string& s) {
    if (s == "none" || s == "uniform")
        return DirectTaperType::NONE;
    if (s == "hamming")
        return DirectTaperType::HAMMING;
    if (s == "blackman" || s == "blackman_pedestal")
        return DirectTaperType::BLACKMAN_PEDESTAL;
    if (s == "gaussian")
        return DirectTaperType::GAUSSIAN;
    return DirectTaperType::COSINE_PEDESTAL;
}

inline const char* taper_type_to_string(DirectTaperType t) {
    switch (t) {
        case DirectTaperType::NONE:
            return "none";
        case DirectTaperType::COSINE_PEDESTAL:
            return "cosine_pedestal";
        case DirectTaperType::HAMMING:
            return "hamming";
        case DirectTaperType::BLACKMAN_PEDESTAL:
            return "blackman_pedestal";
        case DirectTaperType::GAUSSIAN:
            return "gaussian";
        default:
            return "cosine_pedestal";
    }
}

inline float evaluate_1d_taper(float u, DirectTaperType type, float pedestal) {
    const float abs_u = std::min(1.0f, std::abs(u));
    const float p = std::clamp(pedestal, 0.05f, 1.0f);
    constexpr float PI_F = static_cast<float>(M_PI);

    switch (type) {
        case DirectTaperType::COSINE_PEDESTAL:
            return p + (1.0f - p) * 0.5f * (1.0f + std::cos(PI_F * abs_u));
        case DirectTaperType::HAMMING:
            return 0.54f + 0.46f * std::cos(PI_F * abs_u);
        case DirectTaperType::BLACKMAN_PEDESTAL: {
            const float b =
                0.42f + 0.50f * std::cos(PI_F * abs_u) + 0.08f * std::cos(2.0f * PI_F * abs_u);
            return p + (1.0f - p) * std::max(0.0f, b);
        }
        case DirectTaperType::GAUSSIAN:
            return std::pow(p, 0.5f * abs_u * abs_u);
        case DirectTaperType::NONE:
        default:
            return 1.0f;
    }
}

struct DirectDirection3D {
    float x = 0.0f;
    float y = 0.0f;
    float z = 1.0f;
};

struct DirectCelestialTarget {
    double ra_deg = 0.0;
    double dec_deg = 0.0;
    bool is_set = false;
};

struct DirectSiteLocation {
    double lat_deg = charts::constants::charts_caren_lat_deg;
    double lon_deg = charts::constants::charts_caren_lon_deg;
    double alt_m = charts::constants::charts_caren_alt_m;
};

struct DirectBeamTarget {
    DirectDirection3D direction{0.0f, 0.0f, 1.0f};
    DirectDirection3D direction_end{0.0f, 0.0f, 1.0f};
    DirectCelestialTarget celestial;
    int grid_index = -1;
};

struct DirectBeamTrackerConfig {
    std::size_t num_antennas = 64;
    std::size_t num_pols = 1;
    DirectPolLayout pol_layout = DirectPolLayout::INTERLEAVED;
    std::string antenna_order = "descending";
    std::size_t samples_per_data_set = 15360;

    std::size_t max_beams = MAX_DIRECT_BEAMS;
    std::size_t num_active_beams = 1;
    std::vector<DirectBeamTarget> targets{MAX_DIRECT_BEAMS};

    float spacing_m = charts::constants::charts_default_spacing_m;
    std::size_t grid_cols = 8;
    std::size_t time_chunk_size = 256;
    std::size_t time_unroll = 4;
    std::size_t beam_tile_size = 4;
    bool enable_grid_mode = false;
    float grid_step = 0.02f;
    bool enable_cuda_graph = false;
    bool enable_subframe_interpolation = true;

    // Fused 2D Spatial Tapering & Optional Weight-Space BiSLC
    DirectTaperType taper_type = DirectTaperType::COSINE_PEDESTAL;
    float taper_pedestal = 0.25f;
    bool enable_weight_bislc = false;
    float bislc_diagonal_loading = 1.0e-4f;

    std::array<std::uint8_t, MAX_DIRECT_ANTENNAS> antenna_mask;
    std::array<float3, MAX_DIRECT_ANTENNAS> antenna_positions;
    std::array<float, MAX_DIRECT_ANTENNAS> normalized_taper_weights;
    DirectSiteLocation site;

    std::size_t num_active_antennas = 64;
    std::size_t num_local_freq = 1;
    std::size_t num_grid_points = 0;
    std::uint64_t config_generation = 1;
    std::uint64_t mask_generation = 1;

    DirectBeamTrackerConfig() {
        antenna_mask.fill(1);
        normalized_taper_weights.fill(0.0f);
        set_antenna_grid(64, 1, spacing_m, DirectPolLayout::INTERLEAVED, true, 8);
    }

    std::size_t total_elements() const {
        return num_antennas * num_pols;
    }

    void ensure_beam_capacity(std::size_t n_beams) {
        max_beams = std::max<std::size_t>(1, n_beams);
        if (targets.size() < max_beams) {
            targets.resize(max_beams);
        }
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

    void recompute_normalized_taper() {
        normalized_taper_weights.fill(0.0f);
        const std::size_t elem_limit = std::min(total_elements(), MAX_DIRECT_ANTENNAS);
        if (elem_limit == 0)
            return;

        // 1. Compute physical bounding box across all configured elements (ignoring mask so
        //    the geometric center of the aperture never shifts when edge antennas drop out)
        float min_x = 1e30f, max_x = -1e30f;
        float min_y = 1e30f, max_y = -1e30f;
        for (std::size_t r = 0; r < elem_limit; ++r) {
            min_x = std::min(min_x, antenna_positions[r].x);
            max_x = std::max(max_x, antenna_positions[r].x);
            min_y = std::min(min_y, antenna_positions[r].y);
            max_y = std::max(max_y, antenna_positions[r].y);
        }

        const float cx = 0.5f * (min_x + max_x);
        const float cy = 0.5f * (min_y + max_y);
        const float rx = std::max(1e-6f, 0.5f * (max_x - min_x));
        const float ry = std::max(1e-6f, 0.5f * (max_y - min_y));

        // 2. Evaluate unnormalized 2D window t_a and accumulate sum(t_a^2) per polarization
        double sum_sq_pol[2] = {0.0, 0.0};
        std::size_t active_cnt = 0;

        for (std::size_t r = 0; r < elem_limit; ++r) {
            if (antenna_mask[r] == 0) {
                normalized_taper_weights[r] = 0.0f;
                continue;
            }
            active_cnt++;
            const float ux = (antenna_positions[r].x - cx) / rx;
            const float uy = (antenna_positions[r].y - cy) / ry;
            const float wx = evaluate_1d_taper(ux, taper_type, taper_pedestal);
            const float wy = evaluate_1d_taper(uy, taper_type, taper_pedestal);
            const float ta = wx * wy;

            normalized_taper_weights[r] = ta;
            const std::size_t pol = raw_elem_to_pol(r) & 1U;
            sum_sq_pol[pol] += static_cast<double>(ta) * static_cast<double>(ta);
        }
        num_active_antennas = active_cnt;

        // 3. L2 normalization per polarization: guarantees sum_{a in pol} norm_taper[a]^2 == 1.0
        const float inv_norm_pol[2] = {
            (sum_sq_pol[0] > 0.0) ? static_cast<float>(1.0 / std::sqrt(sum_sq_pol[0])) : 0.0f,
            (sum_sq_pol[1] > 0.0) ? static_cast<float>(1.0 / std::sqrt(sum_sq_pol[1])) : 0.0f};

        for (std::size_t r = 0; r < elem_limit; ++r) {
            if (antenna_mask[r] != 0) {
                const std::size_t pol = raw_elem_to_pol(r) & 1U;
                normalized_taper_weights[r] *= inv_norm_pol[pol];
            }
        }
    }

    void set_antenna_grid(std::size_t n_ant, std::size_t n_pol = 1, float spacing = 0.5f,
                          DirectPolLayout layout = DirectPolLayout::INTERLEAVED,
                          bool descending = true, std::size_t custom_cols = 0) {
        num_antennas = std::max<std::size_t>(1, n_ant);
        num_pols = std::max<std::size_t>(1, n_pol);
        pol_layout = layout;
        antenna_order = descending ? "descending" : "ascending";
        spacing_m = spacing;

        if (custom_cols > 0) {
            grid_cols = custom_cols;
        } else if (num_antennas <= 64) {
            grid_cols = 8; // 8x4 (32) or 8x8 (64)
        } else {
            grid_cols = 16; // 16x8 (128) or 16x16 (256)
        }

        const std::size_t n_elem = std::min(total_elements(), MAX_DIRECT_ANTENNAS);
        for (std::size_t r = 0; r < MAX_DIRECT_ANTENNAS; ++r) {
            if (r < n_elem) {
                const std::size_t phys_ant = raw_elem_to_physical_ant(r);
                const std::size_t col = phys_ant % grid_cols;
                const std::size_t row = phys_ant / grid_cols;
                antenna_positions[r] = make_float3(static_cast<float>(col) * spacing_m,
                                                   static_cast<float>(row) * spacing_m, 0.0f);
            } else {
                antenna_positions[r] = make_float3(0.0f, 0.0f, 0.0f);
            }
        }
        recompute_normalized_taper();
    }

    void set_antenna_grid(std::size_t n_ant, float spacing) {
        set_antenna_grid(n_ant, 1, spacing, DirectPolLayout::INTERLEAVED, true, 0);
    }
};

std::vector<float2> generate_sky_grid_directions(float step = 0.02f);
int lookup_nearest_sky_grid(float l, float m, const std::vector<float2>& grid_directions);

inline void compute_celestial_direction(double ra_deg, double dec_deg, double lst_hours,
                                        double lat_deg, DirectDirection3D& out_dir) {
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

void launch_direct_beamformer(const int4x2_t* d_packed, const float2* d_weights,
                              const float2* d_step_weights, float2* d_voltages, std::size_t n_time,
                              std::size_t n_freq, std::size_t n_ant, std::size_t num_active_beams,
                              std::size_t max_beams_stride, std::size_t time_chunk_size = 256,
                              std::size_t time_unroll = 4, std::size_t beam_tile_size = 4,
                              cudaStream_t stream = nullptr, std::size_t n_pol = 1,
                              DirectPolLayout pol_layout = DirectPolLayout::INTERLEAVED);

inline void launch_direct_beamformer(const int4x2_t* d_packed, const float2* d_weights,
                                     float2* d_voltages, std::size_t n_time, std::size_t n_freq,
                                     std::size_t n_ant, std::size_t num_active_beams,
                                     std::size_t max_beams_stride,
                                     std::size_t time_chunk_size = 256, std::size_t time_unroll = 4,
                                     std::size_t beam_tile_size = 4, cudaStream_t stream = nullptr,
                                     std::size_t n_pol = 1,
                                     DirectPolLayout pol_layout = DirectPolLayout::INTERLEAVED) {
    launch_direct_beamformer(d_packed, d_weights, nullptr, d_voltages, n_time, n_freq, n_ant,
                             num_active_beams, max_beams_stride, time_chunk_size, time_unroll,
                             beam_tile_size, stream, n_pol, pol_layout);
}

void set_l2_persisting_weights_policy(cudaStream_t stream, const void* ptr, std::size_t bytes);

/**
 * @brief Generate tapered, mask-normalized steering weights and optional step rotators,
 *        plus optional fused Weight-Space BiSLC cross-beam nulling.
 */
void launch_generate_steering_weights_tapered(
    float2* d_weights, float2* d_step_weights, const DirectDirection3D* d_directions_start,
    const DirectDirection3D* d_directions_end, const double* d_wavenumbers,
    const float3* d_antenna_positions, const float* d_norm_taper, const float2* d_calibration_gains,
    std::size_t num_beams, std::size_t n_freq, std::size_t n_elem, std::size_t n_time,
    bool enable_weight_bislc = false, float bislc_diag_loading = 1.0e-4f,
    cudaStream_t stream = nullptr);

inline void launch_generate_steering_weights_tapered(
    float2* d_weights, const DirectDirection3D* d_directions, const double* d_wavenumbers,
    const float3* d_antenna_positions, const float* d_norm_taper, const float2* d_calibration_gains,
    std::size_t num_beams, std::size_t n_freq, std::size_t n_elem, bool enable_weight_bislc = false,
    float bislc_diag_loading = 1.0e-4f, cudaStream_t stream = nullptr) {
    launch_generate_steering_weights_tapered(d_weights, nullptr, d_directions, nullptr,
                                             d_wavenumbers, d_antenna_positions, d_norm_taper,
                                             d_calibration_gains, num_beams, n_freq, n_elem, 0,
                                             enable_weight_bislc, bislc_diag_loading, stream);
}

/**
 * @brief Legacy-compatible overload using uint8_t antenna_mask.
 */
void launch_generate_steering_weights(
    float2* d_weights, float2* d_step_weights, const DirectDirection3D* d_directions_start,
    const DirectDirection3D* d_directions_end, const double* d_wavenumbers,
    const float3* d_antenna_positions, const std::uint8_t* d_antenna_mask,
    const float2* d_calibration_gains, std::size_t num_beams, std::size_t n_freq,
    std::size_t n_elem, std::size_t n_time, std::size_t n_active = 0, cudaStream_t stream = nullptr,
    std::size_t n_pol = 1);

inline void
launch_generate_steering_weights(float2* d_weights, const DirectDirection3D* d_directions,
                                 const double* d_wavenumbers, const float3* d_antenna_positions,
                                 const std::uint8_t* d_antenna_mask,
                                 const float2* d_calibration_gains, std::size_t num_beams,
                                 std::size_t n_freq, std::size_t n_elem, std::size_t n_active = 0,
                                 cudaStream_t stream = nullptr, std::size_t n_pol = 1) {
    launch_generate_steering_weights(d_weights, nullptr, d_directions, nullptr, d_wavenumbers,
                                     d_antenna_positions, d_antenna_mask, d_calibration_gains,
                                     num_beams, n_freq, n_elem, 0, n_active, stream, n_pol);
}

inline void
launch_generate_steering_weights(float2* d_weights, const DirectDirection3D* d_directions,
                                 const double* d_wavenumbers, const float3* d_antenna_positions,
                                 const std::uint8_t* d_antenna_mask,
                                 const float2* d_calibration_gains, std::size_t num_beams,
                                 std::size_t n_freq, std::size_t n_elem, cudaStream_t stream) {
    launch_generate_steering_weights(d_weights, nullptr, d_directions, nullptr, d_wavenumbers,
                                     d_antenna_positions, d_antenna_mask, d_calibration_gains,
                                     num_beams, n_freq, n_elem, 0, 0, stream, 1);
}

void launch_precompute_sky_grid_tapered(float2* d_grid_weights, const float2* d_grid_lms,
                                        const double* d_wavenumbers,
                                        const float3* d_antenna_positions,
                                        const float* d_norm_taper,
                                        const float2* d_calibration_gains,
                                        std::size_t num_grid_points, std::size_t n_freq,
                                        std::size_t n_elem, cudaStream_t stream = nullptr);

void launch_precompute_sky_grid(float2* d_grid_weights, const float2* d_grid_lms,
                                const double* d_wavenumbers, const float3* d_antenna_positions,
                                const std::uint8_t* d_antenna_mask,
                                const float2* d_calibration_gains, std::size_t num_grid_points,
                                std::size_t n_freq, std::size_t n_elem, std::size_t n_active = 0,
                                cudaStream_t stream = nullptr, std::size_t n_pol = 1);

inline void launch_precompute_sky_grid(float2* d_grid_weights, const float2* d_grid_lms,
                                       const double* d_wavenumbers,
                                       const float3* d_antenna_positions,
                                       const std::uint8_t* d_antenna_mask,
                                       const float2* d_calibration_gains,
                                       std::size_t num_grid_points, std::size_t n_freq,
                                       std::size_t n_elem, cudaStream_t stream) {
    launch_precompute_sky_grid(d_grid_weights, d_grid_lms, d_wavenumbers, d_antenna_positions,
                               d_antenna_mask, d_calibration_gains, num_grid_points, n_freq, n_elem,
                               0, stream, 1);
}

void launch_reduce_beam_power(const float2* d_voltages, float* d_power, std::size_t n_time,
                              std::size_t n_freq, std::size_t active_beams,
                              std::size_t max_beams_stride, cudaStream_t stream = nullptr,
                              std::size_t n_pol = 1);

} // namespace kotekan

#endif // CUDA_DIRECT_BEAM_TRACKER_HPP
