#include "cudaBeamTrackerCommand.hpp"

#include "cudaBeamTrackerV5.hpp"
#include "gpuCommand.hpp"
#include "kotekanLogging.hpp"
#include "restServer.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <tuple>
#include <vector>

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::connectionInstance;
using kotekan::cudaBeamTrackerCommand;
using kotekan::HTTP_RESPONSE;
using kotekan::restServer;

REGISTER_CUDA_COMMAND(cudaBeamTrackerCommand);

namespace kotekan {

std::mutex cudaBeamTrackerCommand::_global_mutex;
MultiBeamTrackerConfig cudaBeamTrackerCommand::_shared_config;
bool cudaBeamTrackerCommand::_endpoints_registered = false;

cudaBeamTrackerCommand::cudaBeamTrackerCommand(Config& config, const std::string& unique_name,
                                               bufferContainer& host_buffers,
                                               cudaDeviceInterface& device, int inst) :
    cudaCommand(config, unique_name, host_buffers, device, inst, no_cuda_command_state,
                "cudaBeamTrackerCommand") {

    _num_elements = config.get<int>(unique_name, "num_elements");
    _num_local_freq = config.get<int>(unique_name, "num_local_freq");
    _samples_per_data_set = config.get<int>(unique_name, "samples_per_data_set");
    _buffer_depth = config.get<int>(unique_name, "buffer_depth");

    _max_beams = config.get_default<int>(unique_name, "max_beams", 1);
    _integration_spectra = config.get_default<int>(unique_name, "integration_spectra", 320);
    _spacing_m = config.get_default<float>(unique_name, "spacing_m",
                                           charts::constants::charts_default_spacing_m);

    _sample_period_s = config.get_default<double>(unique_name, "sample_period_s",
                                                  charts::constants::fpga_time_resolution_s);

    _gpu_mem_voltage = config.get_default<std::string>(unique_name, "gpu_mem_voltage", "voltage");
    // Support either "formed_beams" or backward-compatible "intensity" name
    _gpu_mem_formed_beams = config.get_default<std::string>(
        unique_name, "gpu_mem_formed_beams",
        config.get_default<std::string>(unique_name, "gpu_mem_intensity", "formed_beams"));

    const double site_lat_deg = config.get_default<double>(unique_name, "site_lat_deg",
                                                           charts::constants::charts_caren_lat_deg);
    const double site_lon_deg = config.get_default<double>(unique_name, "site_lon_deg",
                                                           charts::constants::charts_caren_lon_deg);
    const double site_alt_m = config.get_default<double>(unique_name, "site_alt_m",
                                                         charts::constants::charts_caren_alt_m);

    {
        std::lock_guard<std::mutex> lock(_global_mutex);
        _shared_config.site = SiteLocation{site_lat_deg, site_lon_deg, site_alt_m};
        _shared_config.num_active_beams =
            config.get_default<int>(unique_name, "initial_active_beams", 1);
        // Initialize trajectories for each beam slot (0..MAX_TRACKER_BEAMS-1)
        for (std::size_t b = 0; b < MAX_TRACKER_BEAMS; ++b) {
            std::string l0_key = (b == 0) ? "source_l0" : fmt::format("source_l0_{:d}", b);
            std::string m0_key = (b == 0) ? "source_m0" : fmt::format("source_m0_{:d}", b);
            std::string dl_key = (b == 0) ? "source_dl" : fmt::format("source_dl_{:d}", b);
            std::string dm_key = (b == 0) ? "source_dm" : fmt::format("source_dm_{:d}", b);

            const float b_l0 = config.get_default<float>(unique_name, l0_key, 0.0f);
            const float b_m0 = config.get_default<float>(unique_name, m0_key, 0.0f);
            const float b_dl = config.get_default<float>(unique_name, dl_key, 0.0f);
            const float b_dm = config.get_default<float>(unique_name, dm_key, 0.0f);

            const float b_trans_sq = b_l0 * b_l0 + b_m0 * b_m0;
            const float b_n0 = (b_trans_sq <= 1.0f) ? std::sqrt(1.0f - b_trans_sq) : 0.0f;

            _shared_config.trajectories[b].direction_start = Direction3D{b_l0, b_m0, b_n0};
            _shared_config.trajectories[b].direction_rate_per_sample = DirectionRate2D{b_dl, b_dm};

            // Optional celestial target for beam b
            std::string ra_key = (b == 0) ? "source_ra_deg" : fmt::format("source_ra_deg_{:d}", b);
            std::string dec_key =
                (b == 0) ? "source_dec_deg" : fmt::format("source_dec_deg_{:d}", b);
            std::string lst_key =
                (b == 0) ? "initial_lst_hours" : fmt::format("initial_lst_hours_{:d}", b);

            if (config.exists(unique_name, ra_key) && config.exists(unique_name, dec_key)) {
                const double ra_deg = config.get<double>(unique_name, ra_key);
                const double dec_deg = config.get<double>(unique_name, dec_key);
                const double lst_hours = config.get_default<double>(unique_name, lst_key, 0.0);
                compute_celestial_trajectory(ra_deg, dec_deg, lst_hours, site_lat_deg,
                                             _sample_period_s, _shared_config.trajectories[b]);
                INFO_NON_OO(
                    "Beam Tracker: Initialized beam {:d} celestial target RA={:.4f}°, Dec={:.4f}°",
                    b, ra_deg, dec_deg);
            }
        }

        // Load pre-configured bad / dead antenna elements if specified in YAML
        auto bad_elements = config.get_default<std::vector<int>>(unique_name, "bad_elements", {});
        for (int elem : bad_elements) {
            if (elem >= 0 && elem < _num_elements
                && elem < static_cast<int>(MAX_TRACKER_ANTENNAS)) {
                _shared_config.antenna_mask[elem] = 0;
                INFO_NON_OO(
                    "Beam Tracker: Initialized bad raw antenna element {:d} (masked/disabled).",
                    elem);
            }
        }

        // Support explicit active raw elements list (e.g. [24, 25, 26, 27, 28, 29, 30, 31])
        auto active_raw =
            config.get_default<std::vector<int>>(unique_name, "active_raw_elements", {});
        if (!active_raw.empty()) {
            _shared_config.antenna_mask.fill(0);
            for (int r : active_raw) {
                if (r >= 0 && r < _num_elements && r < static_cast<int>(MAX_TRACKER_ANTENNAS)) {
                    _shared_config.antenna_mask[r] = 1;
                    const int phys_p = (_num_elements - 1) - r;
                    INFO_NON_OO("Beam Tracker: Active raw element {:d} -> physical antenna {:d}", r,
                                phys_p);
                }
            }
        }

        // Support explicit active physical antennas list (e.g. [0, 1, 2, 3, 4, 5, 6, 7])
        auto active_phys =
            config.get_default<std::vector<int>>(unique_name, "active_physical_antennas", {});
        if (!active_phys.empty()) {
            _shared_config.antenna_mask.fill(0);
            for (int p : active_phys) {
                const int r = (_num_elements - 1) - p;
                if (r >= 0 && r < _num_elements && r < static_cast<int>(MAX_TRACKER_ANTENNAS)) {
                    _shared_config.antenna_mask[r] = 1;
                    INFO_NON_OO("Beam Tracker: Active physical antenna {:d} -> raw element {:d}", p,
                                r);
                }
            }
        }

        // 3D Antenna Physical Positions: Calculated outside the CUDA kernel
        // Supports "descending" (CHARTS convention: raw 31 -> physical 0), "ascending", or custom
        // 3D coordinates
        std::string antenna_order =
            config.get_default<std::string>(unique_name, "antenna_order", "descending");
        _shared_config.spacing_m = _spacing_m;

        for (int r = 0; r < _num_elements && r < static_cast<int>(MAX_TRACKER_ANTENNAS); ++r) {
            int phys_elem = (antenna_order == "descending") ? ((_num_elements - 1) - r) : r;
            unsigned int col, row;
            if (_num_elements <= 64) {
                col = phys_elem & 7U;
                row = phys_elem >> 3U;
            } else {
                col = phys_elem & 15U;
                row = phys_elem >> 4U;
            }
            _shared_config.antenna_positions[r] = make_float3(
                static_cast<float>(col) * _spacing_m, static_cast<float>(row) * _spacing_m, 0.0f);
        }
        INFO_NON_OO("Beam Tracker: Antenna physical positions precomputed outside kernel "
                    "(order='{:s}', spacing={:.3f}m)",
                    antenna_order, _spacing_m);

        // Optional custom 3D coordinates from YAML (e.g. [[x0, y0, z0], [x1, y1, z1], ...])
        if (config.exists(unique_name, "custom_antenna_positions")) {
            auto custom_pos = config.get<std::vector<std::vector<float>>>(
                unique_name, "custom_antenna_positions");
            for (std::size_t i = 0;
                 i < custom_pos.size() && i < static_cast<std::size_t>(_num_elements)
                 && i < MAX_TRACKER_ANTENNAS;
                 ++i) {
                if (custom_pos[i].size() >= 2) {
                    float z = (custom_pos[i].size() >= 3) ? custom_pos[i][2] : 0.0f;
                    _shared_config.antenna_positions[i] =
                        make_float3(custom_pos[i][0], custom_pos[i][1], z);
                    INFO_NON_OO("Beam Tracker: Custom survey coordinate for raw element {:d}: "
                                "({:.3f}, {:.3f}, {:.3f}) m",
                                i, custom_pos[i][0], custom_pos[i][1], z);
                }
            }
        }

        // Register Dynamic REST Endpoints once
        if (!_endpoints_registered) {
            auto& rest = restServer::instance();

            // 1. Update Trajectory in direction cosines (l, m, dl, dm)
            rest.register_post_callback("/beam_tracker/set_trajectory", [](connectionInstance& conn,
                                                                           nlohmann::json& j) {
                try {
                    std::size_t beam_id = j.value("beam_id", 0);
                    if (beam_id >= MAX_TRACKER_BEAMS) {
                        conn.send_error("beam_id exceeds MAX_TRACKER_BEAMS",
                                        HTTP_RESPONSE::BAD_REQUEST);
                        return;
                    }

                    std::lock_guard<std::mutex> lk(_global_mutex);
                    if (j.contains("l0"))
                        _shared_config.trajectories[beam_id].direction_start.x = j["l0"];
                    else if (j.contains("source_l0"))
                        _shared_config.trajectories[beam_id].direction_start.x = j["source_l0"];

                    if (j.contains("m0"))
                        _shared_config.trajectories[beam_id].direction_start.y = j["m0"];
                    else if (j.contains("source_m0"))
                        _shared_config.trajectories[beam_id].direction_start.y = j["source_m0"];

                    if (j.contains("dl"))
                        _shared_config.trajectories[beam_id].direction_rate_per_sample.dl = j["dl"];
                    else if (j.contains("source_dl"))
                        _shared_config.trajectories[beam_id].direction_rate_per_sample.dl =
                            j["source_dl"];

                    if (j.contains("dm"))
                        _shared_config.trajectories[beam_id].direction_rate_per_sample.dm = j["dm"];
                    else if (j.contains("source_dm"))
                        _shared_config.trajectories[beam_id].direction_rate_per_sample.dm =
                            j["source_dm"];

                    const float lx = _shared_config.trajectories[beam_id].direction_start.x;
                    const float my = _shared_config.trajectories[beam_id].direction_start.y;
                    const float tsq = lx * lx + my * my;
                    _shared_config.trajectories[beam_id].direction_start.z =
                        (tsq <= 1.0f) ? std::sqrt(1.0f - tsq) : 0.0f;
                    _shared_config.trajectories[beam_id].celestial_target.is_set = false;

                    INFO_NON_OO("Beam Tracker: Updated direction trajectory for beam {:d} "
                                "(l0={:.5f}, m0={:.5f})",
                                beam_id, lx, my);
                    conn.send_text_reply(
                        fmt::format("Trajectory updated for beam slot {:d}\n", beam_id));
                } catch (const std::exception& e) {
                    conn.send_error(e.what(), HTTP_RESPONSE::BAD_REQUEST);
                }
            });

            // 2. Update Celestial Target directly using (RA, Dec) with Astrometry conversion
            rest.register_post_callback(
                "/beam_tracker/set_celestial_target",
                [this](connectionInstance& conn, nlohmann::json& j) {
                    try {
                        std::size_t beam_id = j.value("beam_id", 0);
                        if (beam_id >= MAX_TRACKER_BEAMS) {
                            conn.send_error("beam_id exceeds MAX_TRACKER_BEAMS",
                                            HTTP_RESPONSE::BAD_REQUEST);
                            return;
                        }
                        if (!j.contains("ra_deg") || !j.contains("dec_deg")) {
                            conn.send_error("Missing ra_deg or dec_deg in request",
                                            HTTP_RESPONSE::BAD_REQUEST);
                            return;
                        }

                        const double ra_deg = j["ra_deg"];
                        const double dec_deg = j["dec_deg"];
                        double lst_hours = j.value("lst_hours", 0.0);

                        if (!j.contains("lst_hours") && j.contains("unix_timestamp_s")) {
                            const double unix_s = j["unix_timestamp_s"];
                            const double d = unix_s / 86400.0 - 10957.5;
                            double gmst_hours =
                                std::fmod(18.697374558 + 24.06570982441908 * d, 24.0);
                            if (gmst_hours < 0.0)
                                gmst_hours += 24.0;
                            lst_hours =
                                std::fmod(gmst_hours + _shared_config.site.lon_deg / 15.0, 24.0);
                            if (lst_hours < 0.0)
                                lst_hours += 24.0;
                        }

                        std::lock_guard<std::mutex> lk(_global_mutex);
                        compute_celestial_trajectory(ra_deg, dec_deg, lst_hours,
                                                     _shared_config.site.lat_deg, _sample_period_s,
                                                     _shared_config.trajectories[beam_id]);

                        const auto& traj = _shared_config.trajectories[beam_id];
                        INFO_NON_OO("Beam Tracker: Set celestial target beam {:d} (RA={:.4f}°, "
                                    "Dec={:.4f}°) -> l0={:.5f}, m0={:.5f}",
                                    beam_id, ra_deg, dec_deg, traj.direction_start.x,
                                    traj.direction_start.y);

                        conn.send_text_reply(fmt::format(
                            "Celestial target set for beam {:d}: RA={:.4f}°, Dec={:.4f}°\n",
                            beam_id, ra_deg, dec_deg));
                    } catch (const std::exception& e) {
                        conn.send_error(e.what(), HTTP_RESPONSE::BAD_REQUEST);
                    }
                });

            // 3. Enable / set number of active beams
            rest.register_post_callback(
                "/beam_tracker/enable_beam", [](connectionInstance& conn, nlohmann::json& j) {
                    try {
                        std::size_t count = j.value("num_active_beams", 1);
                        std::lock_guard<std::mutex> lk(_global_mutex);
                        _shared_config.num_active_beams = std::min(count, MAX_TRACKER_BEAMS);
                        INFO_NON_OO("Beam Tracker: Active beams set to {:d}",
                                    _shared_config.num_active_beams);
                        conn.send_text_reply(fmt::format("Active beams set to {:d}\n",
                                                         _shared_config.num_active_beams));
                    } catch (const std::exception& e) {
                        conn.send_error(e.what(), HTTP_RESPONSE::BAD_REQUEST);
                    }
                });

            // 4. Mask / Unmask Antenna (Fault-tolerance for dead/failing antennas)
            rest.register_post_callback(
                "/beam_tracker/mask_antenna", [](connectionInstance& conn, nlohmann::json& j) {
                    try {
                        if (!j.contains("antenna_id")) {
                            conn.send_error("Missing antenna_id in request",
                                            HTTP_RESPONSE::BAD_REQUEST);
                            return;
                        }
                        std::size_t ant_id = j["antenna_id"];
                        if (ant_id >= MAX_TRACKER_ANTENNAS) {
                            conn.send_error("antenna_id exceeds MAX_TRACKER_ANTENNAS",
                                            HTTP_RESPONSE::BAD_REQUEST);
                            return;
                        }
                        bool enabled = j.value("enabled", false);

                        std::lock_guard<std::mutex> lk(_global_mutex);
                        _shared_config.antenna_mask[ant_id] = enabled ? 1 : 0;
                        INFO_NON_OO("Beam Tracker: Antenna {:d} set to {:s}", ant_id,
                                    enabled ? "ACTIVE" : "DEAD/MASKED");
                        conn.send_text_reply(fmt::format("Antenna {:d} set to {:s}\n", ant_id,
                                                         enabled ? "ACTIVE" : "DEAD/MASKED"));
                    } catch (const std::exception& e) {
                        conn.send_error(e.what(), HTTP_RESPONSE::BAD_REQUEST);
                    }
                });

            // 5. Batch Mask Antennas
            rest.register_post_callback(
                "/beam_tracker/set_antenna_mask", [](connectionInstance& conn, nlohmann::json& j) {
                    try {
                        std::lock_guard<std::mutex> lk(_global_mutex);
                        if (j.contains("bad_elements")) {
                            for (int elem : j["bad_elements"]) {
                                if (elem >= 0 && elem < static_cast<int>(MAX_TRACKER_ANTENNAS)) {
                                    _shared_config.antenna_mask[elem] = 0;
                                }
                            }
                        }
                        if (j.contains("active_elements")) {
                            _shared_config.antenna_mask.fill(0);
                            for (int elem : j["active_elements"]) {
                                if (elem >= 0 && elem < static_cast<int>(MAX_TRACKER_ANTENNAS)) {
                                    _shared_config.antenna_mask[elem] = 1;
                                }
                            }
                        }
                        INFO_NON_OO("Beam Tracker: Updated antenna mask configuration.");
                        conn.send_text_reply("Antenna mask configuration updated\n");
                    } catch (const std::exception& e) {
                        conn.send_error(e.what(), HTTP_RESPONSE::BAD_REQUEST);
                    }
                });

            // 6. Auto-Mask Antennas (Automatic detection & masking)
            rest.register_post_callback("/beam_tracker/auto_mask", [](connectionInstance& conn,
                                                                      nlohmann::json& j) {
                try {
                    std::lock_guard<std::mutex> lk(_global_mutex);
                    std::size_t masked_count = 0;
                    if (j.contains("bad_elements")) {
                        for (int elem : j["bad_elements"]) {
                            if (elem >= 0 && elem < static_cast<int>(MAX_TRACKER_ANTENNAS)) {
                                _shared_config.antenna_mask[elem] = 0;
                                masked_count++;
                            }
                        }
                    }
                    if (j.contains("active_elements")) {
                        _shared_config.antenna_mask.fill(0);
                        for (int elem : j["active_elements"]) {
                            if (elem >= 0 && elem < static_cast<int>(MAX_TRACKER_ANTENNAS)) {
                                _shared_config.antenna_mask[elem] = 1;
                            }
                        }
                    }
                    INFO_NON_OO(
                        "Beam Tracker: Auto-mask applied. {:d} bad antenna elements masked.",
                        masked_count);
                    conn.send_text_reply(fmt::format(
                        "Auto-mask applied successfully ({:d} elements masked)\n", masked_count));
                } catch (const std::exception& e) {
                    conn.send_error(e.what(), HTTP_RESPONSE::BAD_REQUEST);
                }
            });

            // 7. Status Query
            rest.register_get_callback("/beam_tracker/status", [this](connectionInstance& conn) {
                nlohmann::json reply;
                std::lock_guard<std::mutex> lk(_global_mutex);
                reply["output_format"] = "complex64 (float2: real, imag)";
                reply["num_active_beams"] = _shared_config.num_active_beams;
                reply["max_beams_capacity"] = MAX_TRACKER_BEAMS;
                reply["integration_spectra"] = _shared_config.integration_spectra;
                reply["spacing_m"] = _shared_config.spacing_m;
                reply["total_elements"] = _num_elements;
                reply["site"] = {{"lat_deg", _shared_config.site.lat_deg},
                                 {"lon_deg", _shared_config.site.lon_deg},
                                 {"alt_m", _shared_config.site.alt_m}};

                std::size_t active_ant_count = 0;
                nlohmann::json bad_elems = nlohmann::json::array();
                nlohmann::json active_raw = nlohmann::json::array();
                nlohmann::json active_phys = nlohmann::json::array();
                for (int a = 0; a < _num_elements; ++a) {
                    if (_shared_config.antenna_mask[a] != 0) {
                        active_ant_count++;
                        active_raw.push_back(a);
                        active_phys.push_back((_num_elements - 1) - a);
                    } else {
                        bad_elems.push_back(a);
                    }
                }
                reply["num_active_antennas"] = active_ant_count;
                reply["active_antennas"] = active_ant_count;
                reply["masked_antennas"] = bad_elems.size();
                reply["bad_elements"] = bad_elems;
                reply["active_raw_elements"] = active_raw;
                reply["active_physical_antennas"] = active_phys;

                nlohmann::json ant_pos_json = nlohmann::json::array();
                for (int a = 0; a < _num_elements; ++a) {
                    ant_pos_json.push_back({{"raw_element", a},
                                            {"x", _shared_config.antenna_positions[a].x},
                                            {"y", _shared_config.antenna_positions[a].y},
                                            {"z", _shared_config.antenna_positions[a].z}});
                }
                reply["antenna_positions"] = ant_pos_json;

                reply["beams"] = nlohmann::json::array();
                reply["trajectories"] = nlohmann::json::array();
                for (std::size_t b = 0; b < MAX_TRACKER_BEAMS; ++b) {
                    nlohmann::json b_info;
                    b_info["beam_id"] = b;
                    b_info["l0"] = _shared_config.trajectories[b].direction_start.x;
                    b_info["m0"] = _shared_config.trajectories[b].direction_start.y;
                    b_info["n0"] = _shared_config.trajectories[b].direction_start.z;
                    b_info["dl"] = _shared_config.trajectories[b].direction_rate_per_sample.dl;
                    b_info["dm"] = _shared_config.trajectories[b].direction_rate_per_sample.dm;
                    b_info["source_l0"] = b_info["l0"];
                    b_info["source_m0"] = b_info["m0"];
                    b_info["source_n0"] = b_info["n0"];
                    b_info["source_dl"] = b_info["dl"];
                    b_info["source_dm"] = b_info["dm"];
                    if (_shared_config.trajectories[b].celestial_target.is_set) {
                        b_info["celestial_target"] = {
                            {"is_set", true},
                            {"ra_deg", _shared_config.trajectories[b].celestial_target.ra_deg},
                            {"dec_deg", _shared_config.trajectories[b].celestial_target.dec_deg}};
                    } else {
                        b_info["celestial_target"] = {
                            {"is_set", false}, {"ra_deg", 0.0}, {"dec_deg", 0.0}};
                    }
                    reply["trajectories"].push_back(b_info);
                    if (b < _shared_config.num_active_beams) {
                        reply["beams"].push_back(b_info);
                    }
                }
                conn.send_json_reply(reply);
            });

            // 7. Dynamic Antenna Positions Update (e.g. site survey coordinates or online
            // calibration)
            rest.register_post_callback(
                "/beam_tracker/set_antenna_positions",
                [](connectionInstance& conn, nlohmann::json& j) {
                    try {
                        std::lock_guard<std::mutex> lk(_global_mutex);
                        if (j.contains("positions") && j["positions"].is_array()) {
                            for (const auto& item : j["positions"]) {
                                int elem = item.value("raw_element", item.value("element", -1));
                                if (elem >= 0 && elem < static_cast<int>(MAX_TRACKER_ANTENNAS)) {
                                    float x =
                                        item.value("x", _shared_config.antenna_positions[elem].x);
                                    float y =
                                        item.value("y", _shared_config.antenna_positions[elem].y);
                                    float z =
                                        item.value("z", _shared_config.antenna_positions[elem].z);
                                    _shared_config.antenna_positions[elem] = make_float3(x, y, z);
                                }
                            }
                        }
                        INFO_NON_OO(
                            "Beam Tracker: Dynamically updated antenna physical coordinates.");
                        conn.send_text_reply("Antenna positions updated\n");
                    } catch (const std::exception& e) {
                        conn.send_error(e.what(), HTTP_RESPONSE::BAD_REQUEST);
                    }
                });

            _endpoints_registered = true;
        }
    }

    const double freq_start_hz = config.get_default<double>(unique_name, "freq_start_hz", 300.0e6);
    const double freq_step_hz = config.get_default<double>(unique_name, "freq_step_hz", 300.0e3);

    _frequencies_hz.resize(static_cast<std::size_t>(_num_local_freq));
    for (int f = 0; f < _num_local_freq; ++f) {
        _frequencies_hz[f] = freq_start_hz + f * freq_step_hz;
    }

    set_command_type(gpuCommandType::KERNEL);
    set_name("cudaBeamTrackerCommand");

    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_voltage, true, false, true));
    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_formed_beams, false, true, true));
}

cudaEvent_t cudaBeamTrackerCommand::execute(cudaPipelineState& pipestate,
                                            const std::vector<cudaEvent_t>&) {

    pre_execute();

    MultiBeamTrackerConfig current_config;
    {
        std::lock_guard<std::mutex> lock(_global_mutex);
        current_config = _shared_config;
    }

    // Instantaneous bypass when 0 active beams
    if (current_config.num_active_beams == 0) {
        return record_end_event();
    }

    const std::size_t input_bytes =
        static_cast<std::size_t>(_num_elements) * static_cast<std::size_t>(_num_local_freq)
        * static_cast<std::size_t>(_samples_per_data_set) * sizeof(int4x2_t);
    void* input_memory =
        device.get_gpu_memory_array(_gpu_mem_voltage, gpu_frame_id, _gpu_buffer_depth, input_bytes);

    const std::size_t output_bytes = static_cast<std::size_t>(_num_local_freq)
                                     * static_cast<std::size_t>(_samples_per_data_set)
                                     * static_cast<std::size_t>(_max_beams) * sizeof(float2);
    void* output_memory = device.get_gpu_memory_array(_gpu_mem_formed_beams, gpu_frame_id,
                                                      _gpu_buffer_depth, output_bytes);

    // Propagate metadata if available
    std::shared_ptr<metadataObject> meta =
        device.get_gpu_memory_array_metadata(_gpu_mem_voltage, gpu_frame_id);
    if (meta) {
        device.claim_gpu_memory_array_metadata(_gpu_mem_formed_beams, gpu_frame_id, meta);
    }

    record_start_event();

    const std::size_t windows_per_frame =
        static_cast<std::size_t>((_samples_per_data_set + current_config.integration_spectra - 1)
                                 / current_config.integration_spectra);
    const std::size_t window_offset =
        static_cast<std::size_t>(pipestate.gpu_frame_id) * windows_per_frame;

    launch_beam_tracker_v5_multibeam(
        reinterpret_cast<const int4x2_t*>(input_memory), reinterpret_cast<float2*>(output_memory),
        static_cast<std::size_t>(_samples_per_data_set), static_cast<std::size_t>(_num_local_freq),
        static_cast<std::size_t>(_num_elements), static_cast<std::size_t>(_max_beams),
        _frequencies_hz, current_config, device.getStream(cuda_stream_id), window_offset);

    return record_end_event();
}

} // namespace kotekan
