#include "cudaDirectBeamTrackerCommand.hpp"

#include "chartsConstants.hpp"
#include "chartsMetadata.hpp"
#include "cudaDirectBeamTracker.cuh"
#include "cudaDirectBeamTracker.hpp"
#include "cudaUtils.hpp"
#include "gpuCommand.hpp"
#include "kotekanLogging.hpp"
#include "restServer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <tuple>
#include <vector>

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::connectionInstance;
using kotekan::cudaDirectBeamTrackerCommand;
using kotekan::HTTP_RESPONSE;
using kotekan::restServer;

REGISTER_CUDA_COMMAND(cudaDirectBeamTrackerCommand);

namespace kotekan {

std::mutex cudaDirectBeamTrackerCommand::_registry_mutex;
std::map<std::string, std::shared_ptr<TrackerGroup>> cudaDirectBeamTrackerCommand::_tracker_groups;
DirectBeamTrackerConfig cudaDirectBeamTrackerCommand::_default_shared_config;

void cudaDirectBeamTrackerCommand::set_shared_antenna_mask(
    const std::array<std::uint8_t, MAX_DIRECT_ANTENNAS>& mask, int /*n_active*/) {
    std::lock_guard<std::mutex> lock(_registry_mutex);
    _default_shared_config.antenna_mask = mask;
    _default_shared_config.recompute_normalized_taper();
    _default_shared_config.mask_generation++;
    _default_shared_config.config_generation++;

    for (auto& [prefix, group] : _tracker_groups) {
        std::lock_guard<std::mutex> glock(group->group_mutex);
        if (group->config.antenna_mask != mask) {
            group->config.antenna_mask = mask;
            group->config.recompute_normalized_taper();
            group->config.mask_generation++;
            group->config.config_generation++;
        }
    }
}

void cudaDirectBeamTrackerCommand::set_shared_antenna_mask(
    const std::array<std::uint8_t, charts::constants::charts_total_antennas>& mask,
    int /*n_active*/) {
    std::lock_guard<std::mutex> lock(_registry_mutex);
    const std::size_t n_elem =
        std::min(_default_shared_config.total_elements(), MAX_DIRECT_ANTENNAS);
    std::array<std::uint8_t, MAX_DIRECT_ANTENNAS> new_mask = _default_shared_config.antenna_mask;
    for (std::size_t i = 0; i < n_elem; ++i) {
        const std::size_t phys_ant = _default_shared_config.raw_elem_to_physical_ant(i);
        new_mask[i] = (phys_ant < charts::constants::charts_total_antennas) ? mask[phys_ant] : 0;
    }
    _default_shared_config.antenna_mask = new_mask;
    _default_shared_config.recompute_normalized_taper();
    _default_shared_config.mask_generation++;
    _default_shared_config.config_generation++;

    for (auto& [prefix, group] : _tracker_groups) {
        std::lock_guard<std::mutex> glock(group->group_mutex);
        const std::size_t gn_elem = std::min(group->config.total_elements(), MAX_DIRECT_ANTENNAS);
        std::array<std::uint8_t, MAX_DIRECT_ANTENNAS> gnew_mask = group->config.antenna_mask;
        for (std::size_t i = 0; i < gn_elem; ++i) {
            const std::size_t phys_ant = group->config.raw_elem_to_physical_ant(i);
            gnew_mask[i] =
                (phys_ant < charts::constants::charts_total_antennas) ? mask[phys_ant] : 0;
        }
        if (group->config.antenna_mask != gnew_mask) {
            group->config.antenna_mask = gnew_mask;
            group->config.recompute_normalized_taper();
            group->config.mask_generation++;
            group->config.config_generation++;
        }
    }
}

std::array<std::uint8_t, MAX_DIRECT_ANTENNAS>
cudaDirectBeamTrackerCommand::get_shared_antenna_mask() {
    return get_shared_config().antenna_mask;
}

DirectBeamTrackerConfig cudaDirectBeamTrackerCommand::get_shared_config() {
    std::lock_guard<std::mutex> lock(_registry_mutex);
    auto it = _tracker_groups.find("/direct_tracker");
    if (it != _tracker_groups.end()) {
        std::lock_guard<std::mutex> glock(it->second->group_mutex);
        return it->second->config;
    }
    if (!_tracker_groups.empty()) {
        std::lock_guard<std::mutex> glock(_tracker_groups.begin()->second->group_mutex);
        return _tracker_groups.begin()->second->config;
    }
    return _default_shared_config;
}

cudaDirectBeamTrackerCommand::cudaDirectBeamTrackerCommand(Config& config,
                                                           const std::string& unique_name,
                                                           bufferContainer& host_buffers,
                                                           cudaDeviceInterface& device, int inst) :
    cudaCommand(config, unique_name, host_buffers, device, inst, no_cuda_command_state,
                "cudaDirectBeamTrackerCommand") {

    _num_pols = config.get_default<int>(unique_name, "num_pols", 1);
    if (config.exists(unique_name, "num_antennas")) {
        _num_antennas = config.get<int>(unique_name, "num_antennas");
        _num_elements =
            config.get_default<int>(unique_name, "num_elements", _num_antennas * _num_pols);
    } else {
        _num_elements = config.get<int>(unique_name, "num_elements");
        _num_antennas = _num_elements / std::max(1, _num_pols);
    }

    const std::string pol_layout_str =
        config.get_default<std::string>(unique_name, "pol_layout", "interleaved");
    _pol_layout =
        (pol_layout_str == "planar") ? DirectPolLayout::PLANAR : DirectPolLayout::INTERLEAVED;
    _grid_cols = config.get_default<int>(unique_name, "grid_cols", 0);

    _num_local_freq = config.get<int>(unique_name, "num_local_freq");
    _samples_per_data_set = config.get<int>(unique_name, "samples_per_data_set");
    _buffer_depth = config.get<int>(unique_name, "buffer_depth");

    _max_beams = std::max(1, config.get_default<int>(unique_name, "max_beams", 1));
    _spacing_m = config.get_default<float>(unique_name, "spacing_m",
                                           charts::constants::charts_default_spacing_m);
    _time_chunk_size =
        static_cast<std::size_t>(config.get_default<int>(unique_name, "time_chunk_size", 256));
    _time_unroll = static_cast<std::size_t>(config.get_default<int>(unique_name, "time_unroll", 4));
    _beam_tile_size =
        static_cast<std::size_t>(config.get_default<int>(unique_name, "beam_tile_size", 4));
    _enable_cuda_graph = config.get_default<bool>(unique_name, "enable_cuda_graph", false);

    const std::size_t ring_slots =
        static_cast<std::size_t>(std::max(_buffer_depth, _gpu_buffer_depth));
    _cuda_graphs.resize(ring_slots, nullptr);
    _cuda_graph_execs.resize(ring_slots, nullptr);

    _grid_mode_enabled = config.get_default<bool>(unique_name, "enable_grid_mode", false);
    _grid_step = config.get_default<float>(unique_name, "grid_step", 0.02f);
    const bool enable_subframe_interp =
        config.get_default<bool>(unique_name, "enable_subframe_interpolation", true);

    const std::string taper_str =
        config.get_default<std::string>(unique_name, "taper_type", "cosine_pedestal");
    const float taper_pedestal = config.get_default<float>(unique_name, "taper_pedestal", 0.25f);
    const bool enable_weight_bislc =
        config.get_default<bool>(unique_name, "enable_weight_bislc", false);
    const float bislc_diag =
        config.get_default<float>(unique_name, "bislc_diagonal_loading", 1.0e-4f);

    _gpu_mem_voltage = config.get_default<std::string>(unique_name, "gpu_mem_voltage", "voltage");
    _gpu_mem_formed_beams =
        config.get_default<std::string>(unique_name, "gpu_mem_formed_beams", "formed_beams");

    const double site_lat_deg = config.get_default<double>(unique_name, "site_lat_deg",
                                                           charts::constants::charts_caren_lat_deg);
    const double site_lon_deg = config.get_default<double>(unique_name, "site_lon_deg",
                                                           charts::constants::charts_caren_lon_deg);
    const double site_alt_m = config.get_default<double>(unique_name, "site_alt_m",
                                                         charts::constants::charts_caren_alt_m);

    _antenna_order = config.get_default<std::string>(unique_name, "antenna_order", "descending");

    _rest_prefix = config.get_default<std::string>(
        unique_name, "rest_prefix", _grid_mode_enabled ? "/grid_tracker" : "/direct_tracker");
    _freq_start_hz = config.get_default<double>(unique_name, "freq_start_hz", 300.0e6);
    _freq_step_hz = config.get_default<double>(unique_name, "freq_step_hz", 300.0e3);

    std::shared_ptr<TrackerGroup> group;
    {
        std::lock_guard<std::mutex> reg_lock(_registry_mutex);
        auto it = _tracker_groups.find(_rest_prefix);
        if (it == _tracker_groups.end()) {
            group = std::make_shared<TrackerGroup>();
            group->rest_prefix = _rest_prefix;
            _tracker_groups[_rest_prefix] = group;
        } else {
            group = it->second;
        }
        group->instances.push_back(this);
    }

    {
        std::lock_guard<std::mutex> lock(group->group_mutex);
        group->config.ensure_beam_capacity(static_cast<std::size_t>(_max_beams));
        group->config.site = DirectSiteLocation{site_lat_deg, site_lon_deg, site_alt_m};

        const std::size_t init_beams = static_cast<std::size_t>(
            std::max(0, config.get_default<int>(unique_name, "initial_active_beams", 1)));
        group->config.num_active_beams = std::min(init_beams, group->config.max_beams);

        group->config.enable_grid_mode = _grid_mode_enabled;
        group->config.grid_step = _grid_step;
        group->config.enable_cuda_graph = _enable_cuda_graph;
        group->config.enable_subframe_interpolation = enable_subframe_interp;
        group->config.taper_type = parse_taper_type(taper_str);
        group->config.taper_pedestal = taper_pedestal;
        group->config.enable_weight_bislc = enable_weight_bislc;
        group->config.bislc_diagonal_loading = bislc_diag;
        group->config.spacing_m = _spacing_m;
        group->config.time_chunk_size = _time_chunk_size;
        group->config.time_unroll = _time_unroll;
        group->config.beam_tile_size = _beam_tile_size;
        group->config.samples_per_data_set = static_cast<std::size_t>(_samples_per_data_set);
        group->config.num_local_freq = static_cast<std::size_t>(_num_local_freq);

        for (std::size_t b = 0; b < group->config.max_beams; ++b) {
            std::string l0_key = (b == 0) ? "source_l0" : fmt::format("source_l0_{:d}", b);
            std::string m0_key = (b == 0) ? "source_m0" : fmt::format("source_m0_{:d}", b);

            const float b_l0 = config.get_default<float>(unique_name, l0_key, 0.0f);
            const float b_m0 = config.get_default<float>(unique_name, m0_key, 0.0f);
            const float b_trans_sq = b_l0 * b_l0 + b_m0 * b_m0;
            const float b_n0 = (b_trans_sq <= 1.0f) ? std::sqrt(1.0f - b_trans_sq) : 0.0f;

            group->config.targets[b].direction = DirectDirection3D{b_l0, b_m0, b_n0};
            group->config.targets[b].direction_end = DirectDirection3D{b_l0, b_m0, b_n0};

            std::string ra_key = (b == 0) ? "source_ra_deg" : fmt::format("source_ra_deg_{:d}", b);
            std::string dec_key =
                (b == 0) ? "source_dec_deg" : fmt::format("source_dec_deg_{:d}", b);
            std::string lst_key =
                (b == 0) ? "initial_lst_hours" : fmt::format("initial_lst_hours_{:d}", b);

            if (config.exists(unique_name, ra_key) && config.exists(unique_name, dec_key)) {
                const double ra_deg = config.get<double>(unique_name, ra_key);
                const double dec_deg = config.get<double>(unique_name, dec_key);
                double lst_hours = config.get_default<double>(unique_name, lst_key, -1.0);
                if (lst_hours < 0.0) {
                    using namespace std::chrono;
                    double unix_s =
                        duration_cast<duration<double>>(system_clock::now().time_since_epoch())
                            .count();
                    const double d = unix_s / 86400.0 - 10957.5;
                    double gmst_hours = std::fmod(18.697374558 + 24.06570982441908 * d, 24.0);
                    if (gmst_hours < 0.0)
                        gmst_hours += 24.0;
                    lst_hours = std::fmod(gmst_hours + site_lon_deg / 15.0, 24.0);
                    if (lst_hours < 0.0)
                        lst_hours += 24.0;
                }
                compute_celestial_direction(ra_deg, dec_deg, lst_hours, site_lat_deg,
                                            group->config.targets[b].direction);
                group->config.targets[b].celestial.ra_deg = ra_deg;
                group->config.targets[b].celestial.dec_deg = dec_deg;
                group->config.targets[b].celestial.is_set = true;
            }
        }

        group->config.antenna_mask.fill(1);
        group->config.set_antenna_grid(static_cast<std::size_t>(_num_antennas),
                                       static_cast<std::size_t>(_num_pols), _spacing_m, _pol_layout,
                                       (_antenna_order == "descending"),
                                       static_cast<std::size_t>(std::max(0, _grid_cols)));

        auto set_phys_ant_mask = [&](int phys_ant, std::uint8_t state) {
            if (phys_ant < 0 || phys_ant >= _num_antennas)
                return;
            for (int r = 0; r < _num_elements && r < static_cast<int>(MAX_DIRECT_ANTENNAS); ++r) {
                if (static_cast<int>(group->config.raw_elem_to_physical_ant(r)) == phys_ant) {
                    group->config.antenna_mask[r] = state;
                }
            }
        };

        auto bad_antennas = config.get_default<std::vector<int>>(unique_name, "bad_antennas", {});
        for (int ant : bad_antennas)
            set_phys_ant_mask(ant, 0);

        auto bad_elements = config.get_default<std::vector<int>>(unique_name, "bad_elements", {});
        for (int elem : bad_elements) {
            if (elem >= 0 && elem < _num_elements && elem < static_cast<int>(MAX_DIRECT_ANTENNAS)) {
                group->config.antenna_mask[elem] = 0;
            }
        }

        auto active_antennas =
            config.get_default<std::vector<int>>(unique_name, "active_antennas", {});
        if (!active_antennas.empty()) {
            group->config.antenna_mask.fill(0);
            for (int ant : active_antennas)
                set_phys_ant_mask(ant, 1);
        }

        auto active_raw =
            config.get_default<std::vector<int>>(unique_name, "active_raw_elements", {});
        if (!active_raw.empty()) {
            group->config.antenna_mask.fill(0);
            for (int r : active_raw) {
                if (r >= 0 && r < _num_elements && r < static_cast<int>(MAX_DIRECT_ANTENNAS)) {
                    group->config.antenna_mask[r] = 1;
                }
            }
        }

        if (config.exists(unique_name, "custom_antenna_positions")) {
            auto custom_pos = config.get<std::vector<std::vector<float>>>(
                unique_name, "custom_antenna_positions");
            for (int r = 0; r < _num_elements && r < static_cast<int>(MAX_DIRECT_ANTENNAS); ++r) {
                const std::size_t idx =
                    (custom_pos.size() == static_cast<std::size_t>(_num_antennas))
                        ? group->config.raw_elem_to_physical_ant(r)
                        : static_cast<std::size_t>(r);
                if (idx < custom_pos.size() && custom_pos[idx].size() >= 2) {
                    const float z = (custom_pos[idx].size() >= 3) ? custom_pos[idx][2] : 0.0f;
                    group->config.antenna_positions[r] =
                        make_float3(custom_pos[idx][0], custom_pos[idx][1], z);
                }
            }
        }

        group->config.recompute_normalized_taper();
        group->config.config_generation++;
        group->config.mask_generation++;

        if (_rest_prefix == "/direct_tracker") {
            _default_shared_config = group->config;
        }

        if (!group->endpoints_registered) {
            auto& rest = restServer::instance();

            // 1. Direct Target Steering (l0, m0)
            auto set_target_cb = [group](connectionInstance& conn, nlohmann::json& j) {
                try {
                    std::lock_guard<std::mutex> lk(group->group_mutex);
                    std::size_t beam_id = j.value("beam_id", 0);
                    if (beam_id >= group->config.max_beams) {
                        conn.send_error("beam_id exceeds configured max_beams",
                                        HTTP_RESPONSE::BAD_REQUEST);
                        return;
                    }

                    float lx = group->config.targets[beam_id].direction.x;
                    float my = group->config.targets[beam_id].direction.y;

                    if (j.contains("l0"))
                        lx = j["l0"];
                    else if (j.contains("l"))
                        lx = j["l"];

                    if (j.contains("m0"))
                        my = j["m0"];
                    else if (j.contains("m"))
                        my = j["m"];

                    float lx1 = lx;
                    float my1 = my;
                    if (j.contains("l1"))
                        lx1 = j["l1"];
                    if (j.contains("m1"))
                        my1 = j["m1"];
                    if (j.contains("dl")) {
                        lx1 = lx + static_cast<float>(j["dl"]) * group->config.samples_per_data_set;
                    }
                    if (j.contains("dm")) {
                        my1 = my + static_cast<float>(j["dm"]) * group->config.samples_per_data_set;
                    }

                    const float tsq = lx * lx + my * my;
                    const float nz = (tsq <= 1.0f) ? std::sqrt(1.0f - tsq) : 0.0f;
                    const float tsq1 = lx1 * lx1 + my1 * my1;
                    const float nz1 = (tsq1 <= 1.0f) ? std::sqrt(1.0f - tsq1) : 0.0f;

                    group->config.targets[beam_id].direction = DirectDirection3D{lx, my, nz};
                    group->config.targets[beam_id].direction_end = DirectDirection3D{lx1, my1, nz1};
                    group->config.targets[beam_id].celestial.is_set = false;
                    group->config.targets[beam_id].grid_index = -1;
                    group->config.config_generation++;

                    conn.send_text_reply(fmt::format("Target updated for beam {:d}\n", beam_id));
                } catch (const std::exception& e) {
                    conn.send_error(e.what(), HTTP_RESPONSE::BAD_REQUEST);
                }
            };
            rest.register_post_callback(group->rest_prefix + "/set_target", set_target_cb);
            if (group->rest_prefix == "/direct_tracker") {
                rest.register_post_callback("/beam_tracker/set_trajectory", set_target_cb);
                rest.register_post_callback("/beam_tracker/set_target", set_target_cb);
            }

            // 2. Direct Celestial Target (RA, Dec)
            auto set_celestial_cb = [group](connectionInstance& conn, nlohmann::json& j) {
                try {
                    std::lock_guard<std::mutex> lk(group->group_mutex);
                    std::size_t beam_id = j.value("beam_id", 0);
                    if (beam_id >= group->config.max_beams) {
                        conn.send_error("beam_id exceeds configured max_beams",
                                        HTTP_RESPONSE::BAD_REQUEST);
                        return;
                    }
                    if (!j.contains("ra_deg") || !j.contains("dec_deg")) {
                        conn.send_error("Missing ra_deg or dec_deg", HTTP_RESPONSE::BAD_REQUEST);
                        return;
                    }

                    const double ra_deg = j["ra_deg"];
                    const double dec_deg = j["dec_deg"];
                    double lst_hours = 0.0;
                    if (j.contains("lst_hours")) {
                        lst_hours = j["lst_hours"];
                    } else {
                        double unix_s = 0.0;
                        if (j.contains("unix_timestamp_s")) {
                            unix_s = j["unix_timestamp_s"];
                        } else {
                            using namespace std::chrono;
                            unix_s = duration_cast<duration<double>>(
                                         system_clock::now().time_since_epoch())
                                         .count();
                        }
                        const double d = unix_s / 86400.0 - 10957.5;
                        double gmst = std::fmod(18.697374558 + 24.06570982441908 * d, 24.0);
                        if (gmst < 0.0)
                            gmst += 24.0;
                        lst_hours = std::fmod(gmst + group->config.site.lon_deg / 15.0, 24.0);
                        if (lst_hours < 0.0)
                            lst_hours += 24.0;
                    }

                    const double frame_duration_s =
                        static_cast<double>(group->config.samples_per_data_set)
                        * charts::constants::fpga_time_resolution_s;
                    const double lst_end_hours = std::fmod(
                        lst_hours + (frame_duration_s / 3600.0) * (24.0 / 23.9344696), 24.0);

                    compute_celestial_direction(ra_deg, dec_deg, lst_hours,
                                                group->config.site.lat_deg,
                                                group->config.targets[beam_id].direction);
                    compute_celestial_direction(ra_deg, dec_deg, lst_end_hours,
                                                group->config.site.lat_deg,
                                                group->config.targets[beam_id].direction_end);
                    group->config.targets[beam_id].celestial.ra_deg = ra_deg;
                    group->config.targets[beam_id].celestial.dec_deg = dec_deg;
                    group->config.targets[beam_id].celestial.is_set = true;
                    group->config.targets[beam_id].grid_index = -1;
                    group->config.config_generation++;

                    conn.send_text_reply(
                        fmt::format("Celestial target set for beam {:d}\n", beam_id));
                } catch (const std::exception& e) {
                    conn.send_error(e.what(), HTTP_RESPONSE::BAD_REQUEST);
                }
            };
            rest.register_post_callback(group->rest_prefix + "/set_celestial_target",
                                        set_celestial_cb);
            if (group->rest_prefix == "/direct_tracker") {
                rest.register_post_callback("/beam_tracker/set_celestial_target", set_celestial_cb);
            }

            // 3. Enable Beams
            auto enable_beams_cb = [group](connectionInstance& conn, nlohmann::json& j) {
                try {
                    std::size_t count = j.value("num_active_beams", 1);
                    std::lock_guard<std::mutex> lk(group->group_mutex);
                    group->config.num_active_beams = std::min(count, group->config.max_beams);
                    group->config.config_generation++;
                    conn.send_text_reply(
                        fmt::format("Active beams set to {:d}\n", group->config.num_active_beams));
                } catch (const std::exception& e) {
                    conn.send_error(e.what(), HTTP_RESPONSE::BAD_REQUEST);
                }
            };
            rest.register_post_callback(group->rest_prefix + "/enable_beam", enable_beams_cb);
            if (group->rest_prefix == "/direct_tracker") {
                rest.register_post_callback("/beam_tracker/enable_beam", enable_beams_cb);
            }

            // 4. Mask Antenna / Raw Element
            auto mask_ant_cb = [group](connectionInstance& conn, nlohmann::json& j) {
                try {
                    std::lock_guard<std::mutex> lk(group->group_mutex);
                    const std::size_t n_elem =
                        std::min(group->config.total_elements(), MAX_DIRECT_ANTENNAS);
                    bool is_raw = j.value("is_raw", false);
                    bool enabled = j.value("enabled", false);

                    if (j.contains("raw_id") || (j.contains("antenna_id") && is_raw)) {
                        std::size_t raw_id = j.contains("raw_id") ? j["raw_id"] : j["antenna_id"];
                        if (raw_id >= n_elem) {
                            conn.send_error("raw_id exceeds total configured elements",
                                            HTTP_RESPONSE::BAD_REQUEST);
                            return;
                        }
                        group->config.antenna_mask[raw_id] = enabled ? 1 : 0;
                    } else if (j.contains("physical_id") || j.contains("antenna_id")) {
                        std::size_t phys_id =
                            j.contains("physical_id") ? j["physical_id"] : j["antenna_id"];
                        if (phys_id >= group->config.num_antennas) {
                            conn.send_error("physical_id exceeds num_antennas",
                                            HTTP_RESPONSE::BAD_REQUEST);
                            return;
                        }
                        for (std::size_t r = 0; r < n_elem; ++r) {
                            if (group->config.raw_elem_to_physical_ant(r) == phys_id) {
                                group->config.antenna_mask[r] = enabled ? 1 : 0;
                            }
                        }
                    } else {
                        conn.send_error("Missing antenna_id, physical_id, or raw_id",
                                        HTTP_RESPONSE::BAD_REQUEST);
                        return;
                    }

                    group->config.recompute_normalized_taper();
                    group->config.mask_generation++;
                    group->config.config_generation++;

                    conn.send_text_reply(fmt::format("Mask updated: {:d}/{:d} active elements\n",
                                                     group->config.num_active_antennas, n_elem));
                } catch (const std::exception& e) {
                    conn.send_error(e.what(), HTTP_RESPONSE::BAD_REQUEST);
                }
            };
            rest.register_post_callback(group->rest_prefix + "/mask_antenna", mask_ant_cb);
            if (group->rest_prefix == "/direct_tracker") {
                rest.register_post_callback("/beam_tracker/mask_antenna", mask_ant_cb);
            }

            // 5. Subframe Phase Interpolation Toggle
            auto set_interp_cb = [group](connectionInstance& conn, nlohmann::json& j) {
                try {
                    bool enabled = j.value("enabled", true);
                    std::lock_guard<std::mutex> lk(group->group_mutex);
                    group->config.enable_subframe_interpolation = enabled;
                    group->config.config_generation++;
                    conn.send_text_reply(fmt::format("Subframe phase interpolation set to {:s}\n",
                                                     enabled ? "ENABLED" : "DISABLED"));
                } catch (const std::exception& e) {
                    conn.send_error(e.what(), HTTP_RESPONSE::BAD_REQUEST);
                }
            };
            rest.register_post_callback(group->rest_prefix + "/set_interpolation", set_interp_cb);
            if (group->rest_prefix == "/direct_tracker") {
                rest.register_post_callback("/beam_tracker/set_interpolation", set_interp_cb);
            }

            // 6. Live Spatial Taper & Fused Weight-BiSLC Control
            auto set_taper_cb = [group](connectionInstance& conn, nlohmann::json& j) {
                try {
                    std::lock_guard<std::mutex> lk(group->group_mutex);
                    if (j.contains("taper_type")) {
                        group->config.taper_type =
                            parse_taper_type(j["taper_type"].get<std::string>());
                    }
                    if (j.contains("taper_pedestal")) {
                        group->config.taper_pedestal = j["taper_pedestal"];
                    }
                    if (j.contains("enable_weight_bislc")) {
                        group->config.enable_weight_bislc = j["enable_weight_bislc"];
                    }
                    if (j.contains("bislc_diagonal_loading")) {
                        group->config.bislc_diagonal_loading = j["bislc_diagonal_loading"];
                    }
                    group->config.recompute_normalized_taper();
                    group->config.mask_generation++;
                    group->config.config_generation++;
                    conn.send_text_reply(fmt::format(
                        "Taper updated: type={:s}, pedestal={:.2f}, weight_bislc={:s}\n",
                        taper_type_to_string(group->config.taper_type),
                        group->config.taper_pedestal,
                        group->config.enable_weight_bislc ? "ENABLED" : "DISABLED"));
                } catch (const std::exception& e) {
                    conn.send_error(e.what(), HTTP_RESPONSE::BAD_REQUEST);
                }
            };
            rest.register_post_callback(group->rest_prefix + "/set_taper", set_taper_cb);
            if (group->rest_prefix == "/direct_tracker") {
                rest.register_post_callback("/beam_tracker/set_taper", set_taper_cb);
            }

            // 7. Status Telemetry
            auto status_cb = [group](connectionInstance& conn) {
                nlohmann::json reply;
                std::lock_guard<std::mutex> lk(group->group_mutex);

                reply["version"] =
                    "Direct Beam Tracker v2.5 (Blackwell Fused 2D Taper + Weight-BiSLC)";
                reply["rest_prefix"] = group->rest_prefix;
                reply["num_active_beams"] = group->config.num_active_beams;
                reply["max_beams"] = group->config.max_beams;
                reply["max_beams_capacity"] = group->config.max_beams;
                reply["num_antennas"] = group->config.num_antennas;
                reply["num_pols"] = group->config.num_pols;
                reply["pol_layout"] = (group->config.pol_layout == DirectPolLayout::PLANAR)
                                          ? "planar"
                                          : "interleaved";
                reply["num_elements"] = group->config.total_elements();
                reply["total_elements"] = group->config.total_elements();
                reply["antenna_order"] = group->config.antenna_order;
                reply["num_local_freq"] = group->config.num_local_freq;
                reply["samples_per_data_set"] = group->config.samples_per_data_set;
                reply["subframe_interpolation_enabled"] =
                    group->config.enable_subframe_interpolation;
                reply["taper_type"] = taper_type_to_string(group->config.taper_type);
                reply["taper_pedestal"] = group->config.taper_pedestal;
                reply["enable_weight_bislc"] = group->config.enable_weight_bislc;
                reply["bislc_diagonal_loading"] = group->config.bislc_diagonal_loading;
                reply["grid_mode_enabled"] = group->config.enable_grid_mode;
                reply["grid_points"] = group->config.num_grid_points;
                reply["spacing_m"] = group->config.spacing_m;
                reply["active_elements"] = group->config.num_active_antennas;
                reply["active_antennas"] = group->config.num_active_antennas;

                reply["site"] = {{"lat_deg", group->config.site.lat_deg},
                                 {"lon_deg", group->config.site.lon_deg},
                                 {"alt_m", group->config.site.alt_m}};

                using namespace std::chrono;
                double unix_s =
                    duration_cast<duration<double>>(system_clock::now().time_since_epoch()).count();
                const double d = unix_s / 86400.0 - 10957.5;
                double gmst = std::fmod(18.697374558 + 24.06570982441908 * d, 24.0);
                if (gmst < 0.0)
                    gmst += 24.0;
                double cur_lst = std::fmod(gmst + group->config.site.lon_deg / 15.0, 24.0);
                if (cur_lst < 0.0)
                    cur_lst += 24.0;
                reply["current_utc_timestamp"] = unix_s;
                reply["current_lst_hours"] = cur_lst;

                const std::size_t n_elem =
                    std::min(group->config.total_elements(), MAX_DIRECT_ANTENNAS);
                nlohmann::json active_raw = nlohmann::json::array();
                nlohmann::json active_phys = nlohmann::json::array();
                nlohmann::json bad_elems = nlohmann::json::array();

                for (std::size_t a = 0; a < n_elem; ++a) {
                    if (group->config.antenna_mask[a] != 0) {
                        active_raw.push_back(a);
                        active_phys.push_back(group->config.raw_elem_to_physical_ant(a));
                    } else {
                        bad_elems.push_back(a);
                    }
                }
                reply["masked_antennas"] = n_elem - group->config.num_active_antennas;
                reply["active_raw_elements"] = active_raw;
                reply["active_physical_antennas"] = active_phys;
                reply["bad_elements"] = bad_elems;

                reply["beams"] = nlohmann::json::array();
                for (std::size_t b = 0; b < group->config.num_active_beams; ++b) {
                    nlohmann::json b_info;
                    b_info["beam_id"] = b;
                    b_info["l0"] = group->config.targets[b].direction.x;
                    b_info["m0"] = group->config.targets[b].direction.y;
                    b_info["n0"] = group->config.targets[b].direction.z;
                    b_info["l1"] = group->config.targets[b].direction_end.x;
                    b_info["m1"] = group->config.targets[b].direction_end.y;
                    b_info["dl"] =
                        (group->config.samples_per_data_set > 0)
                            ? static_cast<double>(group->config.targets[b].direction_end.x
                                                  - group->config.targets[b].direction.x)
                                  / group->config.samples_per_data_set
                            : 0.0;
                    b_info["dm"] =
                        (group->config.samples_per_data_set > 0)
                            ? static_cast<double>(group->config.targets[b].direction_end.y
                                                  - group->config.targets[b].direction.y)
                                  / group->config.samples_per_data_set
                            : 0.0;
                    b_info["grid_index"] = group->config.targets[b].grid_index;
                    if (group->config.targets[b].celestial.is_set) {
                        b_info["celestial_target"] = {
                            {"is_set", true},
                            {"ra_deg", group->config.targets[b].celestial.ra_deg},
                            {"dec_deg", group->config.targets[b].celestial.dec_deg}};
                    } else {
                        b_info["celestial_target"] = {{"is_set", false}};
                    }
                    reply["beams"].push_back(b_info);
                }
                reply["trajectories"] = reply["beams"];
                conn.send_json_reply(reply);
            };
            rest.register_get_callback(group->rest_prefix + "/status", status_cb);
            if (group->rest_prefix == "/direct_tracker") {
                rest.register_get_callback("/beam_tracker/status", status_cb);
            }

            // 8. Unified Stitched Beam Power Telemetry (Locked Contract 1)
            auto beam_power_cb = [group](connectionInstance& conn) {
                nlohmann::json reply;
                std::lock_guard<std::mutex> lk(group->group_mutex);

                std::vector<cudaDirectBeamTrackerCommand*> sorted_insts = group->instances;
                std::sort(sorted_insts.begin(), sorted_insts.end(),
                          [](const cudaDirectBeamTrackerCommand* a,
                             const cudaDirectBeamTrackerCommand* b) {
                              return a->get_freq_start_hz() < b->get_freq_start_hz();
                          });

                const int max_b = static_cast<int>(group->config.max_beams);
                const int active_b = static_cast<int>(group->config.num_active_beams);
                const int num_sub = static_cast<int>(sorted_insts.size());

                int total_channels = 0;
                double freq_start = 300.0e6;
                double freq_step = 300.0e3;
                std::uint64_t latest_frame = 0;
                double latest_timestamp = 0.0;

                if (!sorted_insts.empty()) {
                    freq_start = sorted_insts[0]->get_freq_start_hz();
                    freq_step = sorted_insts[0]->get_freq_step_hz();
                    for (auto* inst : sorted_insts) {
                        total_channels += inst->get_num_local_freq();
                        latest_frame = std::max(latest_frame, inst->get_last_frame_id());
                        latest_timestamp =
                            std::max(latest_timestamp, inst->get_last_timestamp_utc());
                    }
                } else {
                    total_channels = static_cast<int>(group->config.num_local_freq);
                }

                if (latest_timestamp <= 0.0) {
                    using namespace std::chrono;
                    latest_timestamp =
                        duration_cast<duration<double>>(system_clock::now().time_since_epoch())
                            .count();
                }

                reply["rest_prefix"] = group->rest_prefix;
                reply["enable_grid_mode"] = group->config.enable_grid_mode;
                reply["max_beams"] = max_b;
                reply["active_beams"] = active_b;
                reply["num_subbands"] = (num_sub > 0) ? num_sub : 1;
                reply["total_freq_channels"] = total_channels;
                reply["freq_start_hz"] = freq_start;
                reply["freq_step_hz"] = freq_step;
                reply["grid_step_rad"] = group->config.grid_step;
                reply["frame_id"] = latest_frame;
                reply["timestamp_utc"] = latest_timestamp;

                nlohmann::json beam_power_array = nlohmann::json::array();
                for (int b = 0; b < max_b; ++b) {
                    std::vector<float> spectrum;
                    spectrum.reserve(total_channels);
                    for (auto* inst : sorted_insts) {
                        const float* h_ptr = inst->get_host_beam_power();
                        const int n_f = inst->get_num_local_freq();
                        if (h_ptr != nullptr && b < active_b) {
                            const float* beam_f_ptr = h_ptr + b * n_f;
                            spectrum.insert(spectrum.end(), beam_f_ptr, beam_f_ptr + n_f);
                        } else {
                            spectrum.insert(spectrum.end(), n_f, 0.0f);
                        }
                    }
                    if (sorted_insts.empty()) {
                        spectrum.resize(total_channels, 0.0f);
                    }
                    beam_power_array.push_back(spectrum);
                }
                reply["beam_power"] = beam_power_array;

                conn.send_json_reply(reply);
            };
            rest.register_get_callback(group->rest_prefix + "/beam_power", beam_power_cb);
            if (group->rest_prefix == "/direct_tracker") {
                rest.register_get_callback("/beam_tracker/beam_power", beam_power_cb);
            }

            group->endpoints_registered = true;
        }
    }

    _frequencies_hz.resize(static_cast<std::size_t>(_num_local_freq));
    for (int f = 0; f < _num_local_freq; ++f) {
        _frequencies_hz[f] = _freq_start_hz + f * _freq_step_hz;
    }

    set_command_type(gpuCommandType::KERNEL);
    set_name("cudaDirectBeamTrackerCommand");

    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_voltage, true, false, true));
    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_formed_beams, false, true, true));

    allocate_device_buffers();
}

cudaDirectBeamTrackerCommand::~cudaDirectBeamTrackerCommand() {
    {
        std::lock_guard<std::mutex> reg_lock(_registry_mutex);
        auto it = _tracker_groups.find(_rest_prefix);
        if (it != _tracker_groups.end()) {
            std::lock_guard<std::mutex> glock(it->second->group_mutex);
            auto& insts = it->second->instances;
            insts.erase(std::remove(insts.begin(), insts.end(), this), insts.end());
        }
    }
    invalidate_cuda_graphs();
    free_device_buffers();
}

void cudaDirectBeamTrackerCommand::invalidate_cuda_graphs() {
    for (auto& exec : _cuda_graph_execs) {
        if (exec != nullptr) {
            cudaGraphExecDestroy(exec);
            exec = nullptr;
        }
    }
    for (auto& graph : _cuda_graphs) {
        if (graph != nullptr) {
            cudaGraphDestroy(graph);
            graph = nullptr;
        }
    }
}

void cudaDirectBeamTrackerCommand::allocate_device_buffers() {
    const std::size_t weights_elems = static_cast<std::size_t>(_max_beams)
                                      * static_cast<std::size_t>(_num_local_freq)
                                      * static_cast<std::size_t>(_num_elements);
    const std::size_t weights_bytes = weights_elems * sizeof(float2);

    // Single contiguous allocation for _d_weights + _d_step_weights in Blackwell L2 cache
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&_d_weights, 2 * weights_bytes));
    _d_step_weights = _d_weights + weights_elems;

    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&_d_directions, _max_beams * sizeof(DirectDirection3D)));
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&_d_directions_end, _max_beams * sizeof(DirectDirection3D)));
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&_d_wavenumbers, _num_local_freq * sizeof(double)));
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&_d_antenna_positions, _num_elements * sizeof(float3)));
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&_d_norm_taper, _num_elements * sizeof(float)));

    // Allocate device and pinned host buffers for beam power reduction
    const std::size_t power_bytes = static_cast<std::size_t>(_max_beams)
                                    * static_cast<std::size_t>(_num_local_freq) * sizeof(float);
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&_d_beam_power, power_bytes));
    CHECK_CUDA_ERROR_NON_OO(cudaMemset(_d_beam_power, 0, power_bytes));
    CHECK_CUDA_ERROR_NON_OO(cudaMallocHost(&_h_beam_power, power_bytes));
    std::memset(_h_beam_power, 0, power_bytes);

    cudaStream_t stream = device.getStream(cuda_stream_id);
    set_l2_persisting_weights_policy(stream, _d_weights, 2 * weights_bytes);

    std::vector<double> h_wavenumbers(_num_local_freq);
    for (int f = 0; f < _num_local_freq; ++f) {
        h_wavenumbers[f] = charts::constants::two_pi * _frequencies_hz[f]
                           / charts::constants::speed_of_light_m_per_s;
    }
    CHECK_CUDA_ERROR_NON_OO(cudaMemcpy(_d_wavenumbers, h_wavenumbers.data(),
                                       _num_local_freq * sizeof(double), cudaMemcpyHostToDevice));

    DirectBeamTrackerConfig cfg;
    {
        std::lock_guard<std::mutex> reg_lock(_registry_mutex);
        auto it = _tracker_groups.find(_rest_prefix);
        if (it != _tracker_groups.end()) {
            std::lock_guard<std::mutex> glock(it->second->group_mutex);
            cfg = it->second->config;
        } else {
            cfg = get_shared_config();
        }
    }

    CHECK_CUDA_ERROR_NON_OO(cudaMemcpy(_d_antenna_positions, cfg.antenna_positions.data(),
                                       _num_elements * sizeof(float3), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR_NON_OO(cudaMemcpy(_d_norm_taper, cfg.normalized_taper_weights.data(),
                                       _num_elements * sizeof(float), cudaMemcpyHostToDevice));

    if (_grid_mode_enabled) {
        _h_grid_lms = generate_sky_grid_directions(_grid_step);
        _num_grid_points = _h_grid_lms.size();

        {
            std::lock_guard<std::mutex> reg_lock(_registry_mutex);
            auto it = _tracker_groups.find(_rest_prefix);
            if (it != _tracker_groups.end()) {
                std::lock_guard<std::mutex> glock(it->second->group_mutex);
                it->second->config.num_grid_points = _num_grid_points;
            }
        }

        const std::size_t grid_bytes =
            _num_grid_points * _num_local_freq * _num_elements * sizeof(float2);
        CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&_d_grid_lms, _num_grid_points * sizeof(float2)));
        CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&_d_grid_weights, grid_bytes));

        CHECK_CUDA_ERROR_NON_OO(cudaMemcpy(_d_grid_lms, _h_grid_lms.data(),
                                           _num_grid_points * sizeof(float2),
                                           cudaMemcpyHostToDevice));

        launch_precompute_sky_grid_tapered(
            _d_grid_weights, _d_grid_lms, _d_wavenumbers, _d_antenna_positions, _d_norm_taper,
            _d_calibration_gains, _num_grid_points, _num_local_freq, _num_elements, stream);

        CHECK_CUDA_ERROR_NON_OO(cudaStreamSynchronize(stream));
        INFO_NON_OO(
            "Direct Beam Tracker: Precomputed Sky Grid with {:d} directions ({:.2f} MB VRAM)",
            _num_grid_points, grid_bytes / (1024.0 * 1024.0));
    }
}

void cudaDirectBeamTrackerCommand::free_device_buffers() {
    if (_d_beam_power) {
        cudaFree(_d_beam_power);
        _d_beam_power = nullptr;
    }
    if (_h_beam_power) {
        cudaFreeHost(_h_beam_power);
        _h_beam_power = nullptr;
    }
    if (_d_weights) {
        cudaFree(_d_weights);
        _d_weights = nullptr;
        _d_step_weights = nullptr;
    }
    if (_d_directions) {
        cudaFree(_d_directions);
        _d_directions = nullptr;
    }
    if (_d_directions_end) {
        cudaFree(_d_directions_end);
        _d_directions_end = nullptr;
    }
    if (_d_wavenumbers) {
        cudaFree(_d_wavenumbers);
        _d_wavenumbers = nullptr;
    }
    if (_d_antenna_positions) {
        cudaFree(_d_antenna_positions);
        _d_antenna_positions = nullptr;
    }
    if (_d_norm_taper) {
        cudaFree(_d_norm_taper);
        _d_norm_taper = nullptr;
    }
    if (_d_calibration_gains) {
        cudaFree(_d_calibration_gains);
        _d_calibration_gains = nullptr;
    }
    if (_d_grid_weights) {
        cudaFree(_d_grid_weights);
        _d_grid_weights = nullptr;
    }
    if (_d_grid_lms) {
        cudaFree(_d_grid_lms);
        _d_grid_lms = nullptr;
    }
}

void cudaDirectBeamTrackerCommand::update_weights_if_needed(const DirectBeamTrackerConfig& in_cfg,
                                                            cudaStream_t stream) {
    std::shared_ptr<TrackerGroup> group;
    {
        std::lock_guard<std::mutex> reg_lock(_registry_mutex);
        auto it = _tracker_groups.find(_rest_prefix);
        if (it != _tracker_groups.end()) {
            group = it->second;
        }
    }

    DirectBeamTrackerConfig cfg = in_cfg;
    bool needs_update = false;
    bool mask_changed = false;

    {
        using namespace std::chrono;
        double unix_s =
            duration_cast<duration<double>>(system_clock::now().time_since_epoch()).count();
        const double frame_duration_s =
            static_cast<double>(_samples_per_data_set) * charts::constants::fpga_time_resolution_s;
        const double unix_s_end = unix_s + frame_duration_s;

        const double d_start = unix_s / 86400.0 - 10957.5;
        double gmst_start = std::fmod(18.697374558 + 24.06570982441908 * d_start, 24.0);
        if (gmst_start < 0.0)
            gmst_start += 24.0;

        const double d_end = unix_s_end / 86400.0 - 10957.5;
        double gmst_end = std::fmod(18.697374558 + 24.06570982441908 * d_end, 24.0);
        if (gmst_end < 0.0)
            gmst_end += 24.0;

        std::unique_lock<std::mutex> lk(group ? group->group_mutex : _registry_mutex);
        DirectBeamTrackerConfig& grp_cfg = group ? group->config : _default_shared_config;

        double lst_start = std::fmod(gmst_start + grp_cfg.site.lon_deg / 15.0, 24.0);
        if (lst_start < 0.0)
            lst_start += 24.0;

        double lst_end = std::fmod(gmst_end + grp_cfg.site.lon_deg / 15.0, 24.0);
        if (lst_end < 0.0)
            lst_end += 24.0;

        const std::size_t active_beams =
            std::min(grp_cfg.num_active_beams, static_cast<std::size_t>(_max_beams));
        for (std::size_t b = 0; b < active_beams; ++b) {
            if (grp_cfg.targets[b].celestial.is_set) {
                DirectDirection3D dir_start, dir_end;
                compute_celestial_direction(grp_cfg.targets[b].celestial.ra_deg,
                                            grp_cfg.targets[b].celestial.dec_deg, lst_start,
                                            grp_cfg.site.lat_deg, dir_start);
                compute_celestial_direction(grp_cfg.targets[b].celestial.ra_deg,
                                            grp_cfg.targets[b].celestial.dec_deg, lst_end,
                                            grp_cfg.site.lat_deg, dir_end);

                if (std::abs(dir_start.x - grp_cfg.targets[b].direction.x) > 1.0e-6f
                    || std::abs(dir_start.y - grp_cfg.targets[b].direction.y) > 1.0e-6f
                    || std::abs(dir_end.x - grp_cfg.targets[b].direction_end.x) > 1.0e-6f
                    || std::abs(dir_end.y - grp_cfg.targets[b].direction_end.y) > 1.0e-6f) {
                    grp_cfg.targets[b].direction = dir_start;
                    grp_cfg.targets[b].direction_end = dir_end;
                    grp_cfg.config_generation++;
                }
            }
        }

        if (_applied_config_generation != grp_cfg.config_generation) {
            needs_update = true;
            mask_changed = (_applied_mask_generation != grp_cfg.mask_generation);
            _applied_config_generation = grp_cfg.config_generation;
            _applied_mask_generation = grp_cfg.mask_generation;
            cfg = grp_cfg;
        }
    }

    if (!needs_update)
        return;

    const std::size_t active_beams =
        std::min(cfg.num_active_beams, static_cast<std::size_t>(_max_beams));

    // Only copy antenna taper weights to device when the mask / taper generation actually changed
    if (mask_changed) {
        CHECK_CUDA_ERROR_NON_OO(cudaMemcpyAsync(_d_norm_taper, cfg.normalized_taper_weights.data(),
                                                _num_elements * sizeof(float),
                                                cudaMemcpyHostToDevice, stream));
        if (_grid_mode_enabled && _d_grid_weights != nullptr) {
            launch_precompute_sky_grid_tapered(
                _d_grid_weights, _d_grid_lms, _d_wavenumbers, _d_antenna_positions, _d_norm_taper,
                _d_calibration_gains, _num_grid_points, _num_local_freq, _num_elements, stream);
        }
    }

    if (_grid_mode_enabled && !_h_grid_lms.empty()) {
        const std::size_t beam_stride = static_cast<std::size_t>(_num_local_freq) * _num_elements;
        for (std::size_t b = 0; b < active_beams; ++b) {
            int g_idx = cfg.targets[b].grid_index;
            if (g_idx < 0 || g_idx >= static_cast<int>(_num_grid_points)) {
                g_idx = lookup_nearest_sky_grid(cfg.targets[b].direction.x,
                                                cfg.targets[b].direction.y, _h_grid_lms);
            }
            CHECK_CUDA_ERROR_NON_OO(
                cudaMemcpyAsync(_d_weights + b * beam_stride, _d_grid_weights + g_idx * beam_stride,
                                beam_stride * sizeof(float2), cudaMemcpyDeviceToDevice, stream));
        }
    } else if (active_beams > 0) {
        std::vector<DirectDirection3D> h_dirs(active_beams);
        std::vector<DirectDirection3D> h_dirs_end(active_beams);
        for (std::size_t b = 0; b < active_beams; ++b) {
            h_dirs[b] = cfg.targets[b].direction;
            h_dirs_end[b] = cfg.targets[b].direction_end;
        }

        CHECK_CUDA_ERROR_NON_OO(cudaMemcpyAsync(_d_directions, h_dirs.data(),
                                                active_beams * sizeof(DirectDirection3D),
                                                cudaMemcpyHostToDevice, stream));
        CHECK_CUDA_ERROR_NON_OO(cudaMemcpyAsync(_d_directions_end, h_dirs_end.data(),
                                                active_beams * sizeof(DirectDirection3D),
                                                cudaMemcpyHostToDevice, stream));

        launch_generate_steering_weights_tapered(
            _d_weights, _d_step_weights, _d_directions, _d_directions_end, _d_wavenumbers,
            _d_antenna_positions, _d_norm_taper, _d_calibration_gains, active_beams,
            _num_local_freq, _num_elements, static_cast<std::size_t>(_samples_per_data_set),
            cfg.enable_weight_bislc, cfg.bislc_diagonal_loading, stream);
    }
}

cudaEvent_t cudaDirectBeamTrackerCommand::execute(cudaPipelineState& /*pipestate*/,
                                                  const std::vector<cudaEvent_t>&) {
    pre_execute();

    std::shared_ptr<TrackerGroup> group;
    {
        std::lock_guard<std::mutex> reg_lock(_registry_mutex);
        auto it = _tracker_groups.find(_rest_prefix);
        if (it != _tracker_groups.end()) {
            group = it->second;
        }
    }

    DirectBeamTrackerConfig current_config;
    if (group) {
        std::lock_guard<std::mutex> glock(group->group_mutex);
        current_config = group->config;
    } else {
        current_config = get_shared_config();
    }

    const std::size_t active_beams =
        std::min(current_config.num_active_beams, static_cast<std::size_t>(_max_beams));

    if (active_beams == 0) {
        return record_end_event();
    }

    const std::size_t input_bytes =
        static_cast<std::size_t>(_num_elements) * static_cast<std::size_t>(_num_local_freq)
        * static_cast<std::size_t>(_samples_per_data_set) * sizeof(int4x2_t);
    void* input_memory =
        device.get_gpu_memory_array(_gpu_mem_voltage, gpu_frame_id, _gpu_buffer_depth, input_bytes);

    const std::size_t output_bytes = static_cast<std::size_t>(_num_local_freq)
                                     * static_cast<std::size_t>(_samples_per_data_set)
                                     * static_cast<std::size_t>(_max_beams)
                                     * static_cast<std::size_t>(_num_pols) * sizeof(float2);
    void* output_memory = device.get_gpu_memory_array(_gpu_mem_formed_beams, gpu_frame_id,
                                                      _gpu_buffer_depth, output_bytes);
    if (!input_memory || !output_memory) {
        return record_end_event();
    }

    // Metadata & Safe Fallback Timestamp Computation
    std::int64_t current_frame = static_cast<std::int64_t>(_frame_counter++);
    double frame_timestamp_utc = 0.0;

    std::shared_ptr<metadataObject> meta =
        device.get_gpu_memory_array_metadata(_gpu_mem_voltage, gpu_frame_id);
    if (meta) {
        device.claim_gpu_memory_array_metadata(_gpu_mem_formed_beams, gpu_frame_id, meta);

        auto charts_meta = std::dynamic_pointer_cast<chartsMetadata>(meta);
        if (charts_meta) {
            if (charts_meta->has_frame_counter()) {
                current_frame = charts_meta->get_frame_counter();
            } else if (charts_meta->has_fpga_seq_num()) {
                current_frame = charts_meta->get_fpga_seq_num() / _samples_per_data_set;
            }
            if (_initial_utc_start_time <= 0.0 && charts_meta->has_first_packet_recv_time()) {
                timeval tv = charts_meta->get_first_packet_recv_time();
                _initial_utc_start_time =
                    static_cast<double>(tv.tv_sec) + static_cast<double>(tv.tv_usec) * 1e-6;
            }
        }
    }

    if (_initial_utc_start_time <= 0.0) {
        using namespace std::chrono;
        _initial_utc_start_time =
            duration_cast<duration<double>>(system_clock::now().time_since_epoch()).count();
    }

    frame_timestamp_utc = _initial_utc_start_time
                          + static_cast<double>(current_frame)
                                * static_cast<double>(_samples_per_data_set) * _sample_period_s;

    _last_frame_id.store(static_cast<std::uint64_t>(current_frame), std::memory_order_relaxed);
    _last_timestamp_utc.store(frame_timestamp_utc, std::memory_order_relaxed);

    record_start_event();

    cudaStream_t stream = device.getStream(cuda_stream_id);
    update_weights_if_needed(current_config, stream);

    const bool use_interp = current_config.enable_subframe_interpolation && !_grid_mode_enabled;
    const float2* step_weights_ptr = use_interp ? _d_step_weights : nullptr;

    // Invalidate CUDA graphs if either active beam count or interpolation mode changed
    if (_last_graph_beams != active_beams || _last_graph_interp != use_interp) {
        invalidate_cuda_graphs();
        _last_graph_beams = active_beams;
        _last_graph_interp = use_interp;
    }

    if (_enable_cuda_graph) {
        const std::size_t slot = static_cast<std::size_t>(gpu_frame_id) % _cuda_graph_execs.size();
        if (_cuda_graph_execs[slot] == nullptr) {
            if (_cuda_graphs[slot] != nullptr) {
                cudaGraphDestroy(_cuda_graphs[slot]);
                _cuda_graphs[slot] = nullptr;
            }
            CHECK_CUDA_ERROR_NON_OO(
                cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
            launch_direct_beamformer(
                reinterpret_cast<const int4x2_t*>(input_memory), _d_weights, step_weights_ptr,
                reinterpret_cast<float2*>(output_memory),
                static_cast<std::size_t>(_samples_per_data_set),
                static_cast<std::size_t>(_num_local_freq), static_cast<std::size_t>(_num_antennas),
                active_beams, static_cast<std::size_t>(_max_beams), _time_chunk_size, _time_unroll,
                _beam_tile_size, stream, static_cast<std::size_t>(_num_pols), _pol_layout);
            CHECK_CUDA_ERROR_NON_OO(cudaStreamEndCapture(stream, &_cuda_graphs[slot]));
            CHECK_CUDA_ERROR_NON_OO(cudaGraphInstantiate(&_cuda_graph_execs[slot],
                                                         _cuda_graphs[slot], nullptr, nullptr, 0));
        }
        CHECK_CUDA_ERROR_NON_OO(cudaGraphLaunch(_cuda_graph_execs[slot], stream));
    } else {
        launch_direct_beamformer(
            reinterpret_cast<const int4x2_t*>(input_memory), _d_weights, step_weights_ptr,
            reinterpret_cast<float2*>(output_memory),
            static_cast<std::size_t>(_samples_per_data_set),
            static_cast<std::size_t>(_num_local_freq), static_cast<std::size_t>(_num_antennas),
            active_beams, static_cast<std::size_t>(_max_beams), _time_chunk_size, _time_unroll,
            _beam_tile_size, stream, static_cast<std::size_t>(_num_pols), _pol_layout);
    }

    // Launch reduction kernel & copy to pinned host buffer
    if (_d_beam_power && _h_beam_power) {
        launch_reduce_beam_power(reinterpret_cast<const float2*>(output_memory), _d_beam_power,
                                 static_cast<std::size_t>(_samples_per_data_set),
                                 static_cast<std::size_t>(_num_local_freq), active_beams,
                                 static_cast<std::size_t>(_max_beams), stream,
                                 static_cast<std::size_t>(_num_pols));

        CHECK_CUDA_ERROR_NON_OO(cudaMemcpyAsync(_h_beam_power, _d_beam_power,
                                                static_cast<std::size_t>(_max_beams)
                                                    * static_cast<std::size_t>(_num_local_freq)
                                                    * sizeof(float),
                                                cudaMemcpyDeviceToHost, stream));
    }

    return record_end_event();
}

} // namespace kotekan
