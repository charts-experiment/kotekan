#include "cudaDirectBeamTrackerCommand.hpp"
#include "cudaDirectBeamTracker.hpp"
#include "cudaUtils.hpp"
#include "gpuCommand.hpp"
#include "kotekanLogging.hpp"
#include "restServer.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <tuple>
#include <vector>

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::connectionInstance;
using kotekan::HTTP_RESPONSE;
using kotekan::restServer;
using kotekan::cudaDirectBeamTrackerCommand;

REGISTER_CUDA_COMMAND(cudaDirectBeamTrackerCommand);

namespace kotekan {

std::mutex cudaDirectBeamTrackerCommand::_global_mutex;
DirectBeamTrackerConfig cudaDirectBeamTrackerCommand::_shared_config;
bool cudaDirectBeamTrackerCommand::_endpoints_registered = false;
bool cudaDirectBeamTrackerCommand::_weights_dirty = true;

void cudaDirectBeamTrackerCommand::set_shared_antenna_mask(const std::array<std::uint8_t, MAX_DIRECT_ANTENNAS>& mask, int n_active) {
    std::lock_guard<std::mutex> lock(_global_mutex);
    if (n_active < 0) {
        n_active = 0;
        for (std::size_t i = 0; i < MAX_DIRECT_ANTENNAS; ++i) {
            if (mask[i] != 0) n_active++;
        }
    }
    if (_shared_config.antenna_mask != mask || _shared_config.num_active_antennas != static_cast<std::size_t>(n_active)) {
        _shared_config.antenna_mask = mask;
        _shared_config.num_active_antennas = static_cast<std::size_t>(n_active);
        _weights_dirty = true;
    }
}

std::array<std::uint8_t, MAX_DIRECT_ANTENNAS> cudaDirectBeamTrackerCommand::get_shared_antenna_mask() {
    std::lock_guard<std::mutex> lock(_global_mutex);
    return _shared_config.antenna_mask;
}

DirectBeamTrackerConfig cudaDirectBeamTrackerCommand::get_shared_config() {
    std::lock_guard<std::mutex> lock(_global_mutex);
    return _shared_config;
}

cudaDirectBeamTrackerCommand::cudaDirectBeamTrackerCommand(
    Config& config, const std::string& unique_name,
    bufferContainer& host_buffers, cudaDeviceInterface& device, int inst)
    : cudaCommand(config, unique_name, host_buffers, device, inst,
                  no_cuda_command_state, "cudaDirectBeamTrackerCommand") {

    _num_elements = config.get<int>(unique_name, "num_elements");
    _num_local_freq = config.get<int>(unique_name, "num_local_freq");
    _samples_per_data_set = config.get<int>(unique_name, "samples_per_data_set");
    _buffer_depth = config.get<int>(unique_name, "buffer_depth");

    _max_beams = config.get_default<int>(unique_name, "max_beams", 1);
    _spacing_m = config.get_default<float>(unique_name, "spacing_m", charts::constants::charts_default_spacing_m);
    _time_chunk_size = static_cast<std::size_t>(config.get_default<int>(unique_name, "time_chunk_size", 256));
    _time_unroll = static_cast<std::size_t>(config.get_default<int>(unique_name, "time_unroll", 4));
    _beam_tile_size = static_cast<std::size_t>(config.get_default<int>(unique_name, "beam_tile_size", 4));
    _enable_cuda_graph = config.get_default<bool>(unique_name, "enable_cuda_graph", false);

    _cuda_graphs.resize(static_cast<std::size_t>(_buffer_depth), nullptr);
    _cuda_graph_execs.resize(static_cast<std::size_t>(_buffer_depth), nullptr);

    _grid_mode_enabled = config.get_default<bool>(unique_name, "enable_grid_mode", false);
    _grid_step = config.get_default<float>(unique_name, "grid_step", 0.02f);
    _enable_subframe_interpolation = config.get_default<bool>(unique_name, "enable_subframe_interpolation", true);

    _gpu_mem_voltage = config.get_default<std::string>(unique_name, "gpu_mem_voltage", "voltage");
    _gpu_mem_formed_beams = config.get_default<std::string>(unique_name, "gpu_mem_formed_beams", "formed_beams");

    const double site_lat_deg = config.get_default<double>(unique_name, "site_lat_deg", charts::constants::charts_caren_lat_deg);
    const double site_lon_deg = config.get_default<double>(unique_name, "site_lon_deg", charts::constants::charts_caren_lon_deg);
    const double site_alt_m = config.get_default<double>(unique_name, "site_alt_m", charts::constants::charts_caren_alt_m);

    {
        std::lock_guard<std::mutex> lock(_global_mutex);
        _shared_config.site = DirectSiteLocation{site_lat_deg, site_lon_deg, site_alt_m};
        _shared_config.num_active_beams = config.get_default<int>(unique_name, "initial_active_beams", 1);
        _shared_config.enable_grid_mode = _grid_mode_enabled;
        _shared_config.spacing_m = _spacing_m;
        _shared_config.time_chunk_size = _time_chunk_size;
        _shared_config.time_unroll = _time_unroll;
        _shared_config.beam_tile_size = _beam_tile_size;
        _shared_config.enable_cuda_graph = _enable_cuda_graph;

        // Initialize beam targets
        for (std::size_t b = 0; b < MAX_DIRECT_BEAMS; ++b) {
            std::string l0_key = (b == 0) ? "source_l0" : fmt::format("source_l0_{:d}", b);
            std::string m0_key = (b == 0) ? "source_m0" : fmt::format("source_m0_{:d}", b);

            const float b_l0 = config.get_default<float>(unique_name, l0_key, 0.0f);
            const float b_m0 = config.get_default<float>(unique_name, m0_key, 0.0f);
            const float b_trans_sq = b_l0 * b_l0 + b_m0 * b_m0;
            const float b_n0 = (b_trans_sq <= 1.0f) ? std::sqrt(1.0f - b_trans_sq) : 0.0f;

            _shared_config.targets[b].direction = DirectDirection3D{b_l0, b_m0, b_n0};
            _shared_config.targets[b].direction_end = DirectDirection3D{b_l0, b_m0, b_n0};

            std::string ra_key = (b == 0) ? "source_ra_deg" : fmt::format("source_ra_deg_{:d}", b);
            std::string dec_key = (b == 0) ? "source_dec_deg" : fmt::format("source_dec_deg_{:d}", b);
            std::string lst_key = (b == 0) ? "initial_lst_hours" : fmt::format("initial_lst_hours_{:d}", b);

            if (config.exists(unique_name, ra_key) && config.exists(unique_name, dec_key)) {
                const double ra_deg = config.get<double>(unique_name, ra_key);
                const double dec_deg = config.get<double>(unique_name, dec_key);
                double lst_hours = config.get_default<double>(unique_name, lst_key, -1.0);
                if (lst_hours < 0.0) {
                    using namespace std::chrono;
                    double unix_s = duration_cast<duration<double>>(system_clock::now().time_since_epoch()).count();
                    const double d = unix_s / 86400.0 - 10957.5;
                    double gmst_hours = std::fmod(18.697374558 + 24.06570982441908 * d, 24.0);
                    if (gmst_hours < 0.0) gmst_hours += 24.0;
                    lst_hours = std::fmod(gmst_hours + site_lon_deg / 15.0, 24.0);
                    if (lst_hours < 0.0) lst_hours += 24.0;
                }
                compute_celestial_direction(ra_deg, dec_deg, lst_hours, site_lat_deg, _shared_config.targets[b].direction);
                _shared_config.targets[b].celestial.ra_deg = ra_deg;
                _shared_config.targets[b].celestial.dec_deg = dec_deg;
                _shared_config.targets[b].celestial.is_set = true;
                INFO_NON_OO("Direct Beam Tracker: Initialized beam {:d} celestial target RA={:.4f}°, Dec={:.4f}°, LST={:.4f}h -> (l0={:.5f}, m0={:.5f})",
                            b, ra_deg, dec_deg, lst_hours,
                            _shared_config.targets[b].direction.x,
                            _shared_config.targets[b].direction.y);
            }
        }

        // Antenna active/bad element masks
        auto bad_elements = config.get_default<std::vector<int>>(unique_name, "bad_elements", {});
        for (int elem : bad_elements) {
            if (elem >= 0 && elem < _num_elements && elem < static_cast<int>(MAX_DIRECT_ANTENNAS)) {
                _shared_config.antenna_mask[elem] = 0;
            }
        }

        auto active_raw = config.get_default<std::vector<int>>(unique_name, "active_raw_elements", {});
        if (!active_raw.empty()) {
            _shared_config.antenna_mask.fill(0);
            for (int r : active_raw) {
                if (r >= 0 && r < _num_elements && r < static_cast<int>(MAX_DIRECT_ANTENNAS)) {
                    _shared_config.antenna_mask[r] = 1;
                }
            }
        }

        int n_active_init = 0;
        for (int i = 0; i < _num_elements; ++i) {
            if (_shared_config.antenna_mask[i] != 0) n_active_init++;
        }
        _shared_config.num_active_antennas = n_active_init;

        // Antenna positions: descending (CHARTS raw 31 -> physical 0) or ascending
        std::string antenna_order = config.get_default<std::string>(unique_name, "antenna_order", "descending");
        for (int r = 0; r < _num_elements && r < static_cast<int>(MAX_DIRECT_ANTENNAS); ++r) {
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
                static_cast<float>(col) * _spacing_m,
                static_cast<float>(row) * _spacing_m,
                0.0f
            );
        }

        if (config.exists(unique_name, "custom_antenna_positions")) {
            auto custom_pos = config.get<std::vector<std::vector<float>>>(unique_name, "custom_antenna_positions");
            for (std::size_t i = 0; i < custom_pos.size() && i < static_cast<std::size_t>(_num_elements) && i < MAX_DIRECT_ANTENNAS; ++i) {
                if (custom_pos[i].size() >= 2) {
                    float z = (custom_pos[i].size() >= 3) ? custom_pos[i][2] : 0.0f;
                    _shared_config.antenna_positions[i] = make_float3(custom_pos[i][0], custom_pos[i][1], z);
                }
            }
        }

        _weights_dirty = true;

        // Register REST endpoints once
        if (!_endpoints_registered) {
            auto& rest = restServer::instance();

            // 1. Direct Target Steering (l0, m0)
            auto set_target_cb = [this](connectionInstance& conn, nlohmann::json& j) {
                try {
                    std::size_t beam_id = j.value("beam_id", 0);
                    if (beam_id >= MAX_DIRECT_BEAMS) {
                        conn.send_error("beam_id exceeds MAX_DIRECT_BEAMS", HTTP_RESPONSE::BAD_REQUEST);
                        return;
                    }

                    std::lock_guard<std::mutex> lk(_global_mutex);
                    float lx = _shared_config.targets[beam_id].direction.x;
                    float my = _shared_config.targets[beam_id].direction.y;

                    if (j.contains("l0")) lx = j["l0"];
                    else if (j.contains("l")) lx = j["l"];

                    if (j.contains("m0")) my = j["m0"];
                    else if (j.contains("m")) my = j["m"];

                    float lx1 = lx;
                    float my1 = my;
                    if (j.contains("l1")) lx1 = j["l1"];
                    if (j.contains("m1")) my1 = j["m1"];
                    if (j.contains("dl")) lx1 = lx + static_cast<float>(j["dl"]) * _samples_per_data_set;
                    if (j.contains("dm")) my1 = my + static_cast<float>(j["dm"]) * _samples_per_data_set;

                    const float tsq = lx * lx + my * my;
                    const float nz = (tsq <= 1.0f) ? std::sqrt(1.0f - tsq) : 0.0f;
                    const float tsq1 = lx1 * lx1 + my1 * my1;
                    const float nz1 = (tsq1 <= 1.0f) ? std::sqrt(1.0f - tsq1) : 0.0f;

                    _shared_config.targets[beam_id].direction = DirectDirection3D{lx, my, nz};
                    _shared_config.targets[beam_id].direction_end = DirectDirection3D{lx1, my1, nz1};
                    _shared_config.targets[beam_id].celestial.is_set = false;
                    _shared_config.targets[beam_id].grid_index = -1;
                    _weights_dirty = true;

                    INFO_NON_OO("Direct Beam Tracker: Steered beam {:d} -> (l0={:.5f}, m0={:.5f}) to (l1={:.5f}, m1={:.5f})",
                                beam_id, lx, my, lx1, my1);
                    conn.send_text_reply(fmt::format("Target updated for beam {:d}\n", beam_id));
                } catch (const std::exception& e) {
                    conn.send_error(e.what(), HTTP_RESPONSE::BAD_REQUEST);
                }
            };
            rest.register_post_callback("/direct_tracker/set_target", set_target_cb);
            rest.register_post_callback("/beam_tracker/set_trajectory", set_target_cb);

            // 2. Direct Celestial Target (RA, Dec)
            auto set_celestial_cb = [this](connectionInstance& conn, nlohmann::json& j) {
                try {
                    std::size_t beam_id = j.value("beam_id", 0);
                    if (beam_id >= MAX_DIRECT_BEAMS) {
                        conn.send_error("beam_id exceeds MAX_DIRECT_BEAMS", HTTP_RESPONSE::BAD_REQUEST);
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
                            unix_s = duration_cast<duration<double>>(system_clock::now().time_since_epoch()).count();
                        }
                        const double d = unix_s / 86400.0 - 10957.5;
                        double gmst = std::fmod(18.697374558 + 24.06570982441908 * d, 24.0);
                        if (gmst < 0.0) gmst += 24.0;
                        lst_hours = std::fmod(gmst + _shared_config.site.lon_deg / 15.0, 24.0);
                        if (lst_hours < 0.0) lst_hours += 24.0;
                    }

                    const double frame_duration_s = static_cast<double>(_samples_per_data_set) * charts::constants::fpga_time_resolution_s;
                    const double lst_end_hours = std::fmod(lst_hours + (frame_duration_s / 3600.0) * (24.0 / 23.9344696), 24.0);

                    std::lock_guard<std::mutex> lk(_global_mutex);
                    compute_celestial_direction(ra_deg, dec_deg, lst_hours, _shared_config.site.lat_deg,
                                                _shared_config.targets[beam_id].direction);
                    compute_celestial_direction(ra_deg, dec_deg, lst_end_hours, _shared_config.site.lat_deg,
                                                _shared_config.targets[beam_id].direction_end);
                    _shared_config.targets[beam_id].celestial.ra_deg = ra_deg;
                    _shared_config.targets[beam_id].celestial.dec_deg = dec_deg;
                    _shared_config.targets[beam_id].celestial.is_set = true;
                    _shared_config.targets[beam_id].grid_index = -1;
                    _weights_dirty = true;

                    INFO_NON_OO("Direct Beam Tracker: Celestial target beam {:d} (RA={:.4f}°, Dec={:.4f}°) -> l={:.5f}, m={:.5f}",
                                beam_id, ra_deg, dec_deg, _shared_config.targets[beam_id].direction.x, _shared_config.targets[beam_id].direction.y);
                    conn.send_text_reply(fmt::format("Celestial target set for beam {:d}\n", beam_id));
                } catch (const std::exception& e) {
                    conn.send_error(e.what(), HTTP_RESPONSE::BAD_REQUEST);
                }
            };
            rest.register_post_callback("/direct_tracker/set_celestial_target", set_celestial_cb);
            rest.register_post_callback("/beam_tracker/set_celestial_target", set_celestial_cb);

            // 3. Enable Beams
            auto enable_beams_cb = [](connectionInstance& conn, nlohmann::json& j) {
                try {
                    std::size_t count = j.value("num_active_beams", 1);
                    std::lock_guard<std::mutex> lk(_global_mutex);
                    _shared_config.num_active_beams = std::min(count, MAX_DIRECT_BEAMS);
                    _weights_dirty = true;
                    INFO_NON_OO("Direct Beam Tracker: Active beams set to {:d}", _shared_config.num_active_beams);
                    conn.send_text_reply(fmt::format("Active beams set to {:d}\n", _shared_config.num_active_beams));
                } catch (const std::exception& e) {
                    conn.send_error(e.what(), HTTP_RESPONSE::BAD_REQUEST);
                }
            };
            rest.register_post_callback("/direct_tracker/enable_beam", enable_beams_cb);
            rest.register_post_callback("/beam_tracker/enable_beam", enable_beams_cb);

            // 4. Mask Antenna
            auto mask_ant_cb = [](connectionInstance& conn, nlohmann::json& j) {
                try {
                    if (!j.contains("antenna_id")) {
                        conn.send_error("Missing antenna_id", HTTP_RESPONSE::BAD_REQUEST);
                        return;
                    }
                    std::size_t ant_id = j["antenna_id"];
                    if (ant_id >= MAX_DIRECT_ANTENNAS) {
                        conn.send_error("antenna_id exceeds MAX_DIRECT_ANTENNAS", HTTP_RESPONSE::BAD_REQUEST);
                        return;
                    }
                    bool enabled = j.value("enabled", false);
                    std::lock_guard<std::mutex> lk(_global_mutex);
                    _shared_config.antenna_mask[ant_id] = enabled ? 1 : 0;
                    int n_active = 0;
                    for (std::size_t i = 0; i < MAX_DIRECT_ANTENNAS; ++i) {
                        if (_shared_config.antenna_mask[i] != 0) n_active++;
                    }
                    _shared_config.num_active_antennas = static_cast<std::size_t>(n_active);
                    _weights_dirty = true;
                    INFO_NON_OO("Direct Beam Tracker: Antenna {:d} set to {:s}", ant_id, enabled ? "ACTIVE" : "MASKED");
                    conn.send_text_reply(fmt::format("Antenna {:d} set to {:s}\n", ant_id, enabled ? "ACTIVE" : "MASKED"));
                } catch (const std::exception& e) {
                    conn.send_error(e.what(), HTTP_RESPONSE::BAD_REQUEST);
                }
            };
            rest.register_post_callback("/direct_tracker/mask_antenna", mask_ant_cb);
            rest.register_post_callback("/beam_tracker/mask_antenna", mask_ant_cb);

            // 5. Subframe Phase Interpolation Toggle
            auto set_interp_cb = [this](connectionInstance& conn, nlohmann::json& j) {
                try {
                    bool enabled = j.value("enabled", true);
                    _enable_subframe_interpolation = enabled;
                    INFO_NON_OO("Direct Beam Tracker: Subframe phase interpolation set to {:s}", enabled ? "ENABLED" : "DISABLED");
                    conn.send_text_reply(fmt::format("Subframe phase interpolation set to {:s}\n", enabled ? "ENABLED" : "DISABLED"));
                } catch (const std::exception& e) {
                    conn.send_error(e.what(), HTTP_RESPONSE::BAD_REQUEST);
                }
            };
            rest.register_post_callback("/direct_tracker/set_interpolation", set_interp_cb);
            rest.register_post_callback("/beam_tracker/set_interpolation", set_interp_cb);

            // 6. Status Telemetry
            auto status_cb = [this](connectionInstance& conn) {
                nlohmann::json reply;
                std::lock_guard<std::mutex> lk(_global_mutex);

                reply["version"] = "Direct Beam Tracker v1.0 (Sub-frame Phase Interpolation)";
                reply["num_active_beams"] = _shared_config.num_active_beams;
                reply["max_beams"] = _max_beams;
                reply["num_elements"] = _num_elements;
                reply["num_local_freq"] = _num_local_freq;
                reply["samples_per_data_set"] = _samples_per_data_set;
                reply["subframe_interpolation_enabled"] = _enable_subframe_interpolation;
                reply["grid_mode_enabled"] = _grid_mode_enabled;
                reply["grid_points"] = _num_grid_points;

                using namespace std::chrono;
                double unix_s = duration_cast<duration<double>>(system_clock::now().time_since_epoch()).count();
                const double d = unix_s / 86400.0 - 10957.5;
                double gmst = std::fmod(18.697374558 + 24.06570982441908 * d, 24.0);
                if (gmst < 0.0) gmst += 24.0;
                double cur_lst = std::fmod(gmst + _shared_config.site.lon_deg / 15.0, 24.0);
                if (cur_lst < 0.0) cur_lst += 24.0;
                reply["current_utc_timestamp"] = unix_s;
                reply["current_lst_hours"] = cur_lst;

                int active_count = 0;
                nlohmann::json active_raw = nlohmann::json::array();
                for (int a = 0; a < _num_elements; ++a) {
                    if (_shared_config.antenna_mask[a] != 0) {
                        active_count++;
                        active_raw.push_back(a);
                    }
                }
                reply["active_antennas"] = active_count;
                reply["active_raw_elements"] = active_raw;

                reply["beams"] = nlohmann::json::array();
                for (std::size_t b = 0; b < _shared_config.num_active_beams; ++b) {
                    nlohmann::json b_info;
                    b_info["beam_id"] = b;
                    b_info["l0"] = _shared_config.targets[b].direction.x;
                    b_info["m0"] = _shared_config.targets[b].direction.y;
                    b_info["n0"] = _shared_config.targets[b].direction.z;
                    b_info["grid_index"] = _shared_config.targets[b].grid_index;
                    if (_shared_config.targets[b].celestial.is_set) {
                        b_info["celestial_target"] = {
                            {"is_set", true},
                            {"ra_deg", _shared_config.targets[b].celestial.ra_deg},
                            {"dec_deg", _shared_config.targets[b].celestial.dec_deg}
                        };
                    } else {
                        b_info["celestial_target"] = {{"is_set", false}};
                    }
                    reply["beams"].push_back(b_info);
                }
                conn.send_json_reply(reply);
            };
            rest.register_get_callback("/direct_tracker/status", status_cb);
            rest.register_get_callback("/beam_tracker/status", status_cb);

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
    set_name("cudaDirectBeamTrackerCommand");

    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_voltage, true, false, true));
    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_formed_beams, false, true, true));

    allocate_device_buffers();
}

cudaDirectBeamTrackerCommand::~cudaDirectBeamTrackerCommand() {
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
    free_device_buffers();
}

void cudaDirectBeamTrackerCommand::allocate_device_buffers() {
    const std::size_t weights_bytes = static_cast<std::size_t>(_max_beams) *
                                      static_cast<std::size_t>(_num_local_freq) *
                                      static_cast<std::size_t>(_num_elements) * sizeof(float2);

    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&_d_weights, weights_bytes));
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&_d_step_weights, weights_bytes));
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&_d_directions, _max_beams * sizeof(DirectDirection3D)));
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&_d_directions_end, _max_beams * sizeof(DirectDirection3D)));
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&_d_wavenumbers, _num_local_freq * sizeof(double)));
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&_d_antenna_positions, _num_elements * sizeof(float3)));
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&_d_antenna_mask, _num_elements * sizeof(std::uint8_t)));

    // Configure L2 persisting access policy window for steering weights on Blackwell
    cudaStream_t stream = device.getStream(cuda_stream_id);
    set_l2_persisting_weights_policy(stream, _d_weights, weights_bytes);

    // Upload wavenumbers
    std::vector<double> h_wavenumbers(_num_local_freq);
    for (int f = 0; f < _num_local_freq; ++f) {
        h_wavenumbers[f] = charts::constants::two_pi * _frequencies_hz[f] / charts::constants::speed_of_light_m_per_s;
    }
    CHECK_CUDA_ERROR_NON_OO(cudaMemcpy(_d_wavenumbers, h_wavenumbers.data(),
                                       _num_local_freq * sizeof(double), cudaMemcpyHostToDevice));

    // Upload antenna positions
    std::vector<float3> h_positions(_num_elements);
    for (int a = 0; a < _num_elements; ++a) {
        h_positions[a] = _shared_config.antenna_positions[a];
    }
    CHECK_CUDA_ERROR_NON_OO(cudaMemcpy(_d_antenna_positions, h_positions.data(),
                                       _num_elements * sizeof(float3), cudaMemcpyHostToDevice));

    // Precompute sky grid if enabled
    if (_grid_mode_enabled) {
        _h_grid_lms = generate_sky_grid_directions(_grid_step);
        _num_grid_points = _h_grid_lms.size();

        const std::size_t grid_bytes = _num_grid_points * _num_local_freq * _num_elements * sizeof(float2);
        CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&_d_grid_lms, _num_grid_points * sizeof(float2)));
        CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&_d_grid_weights, grid_bytes));

        CHECK_CUDA_ERROR_NON_OO(cudaMemcpy(_d_grid_lms, _h_grid_lms.data(),
                                           _num_grid_points * sizeof(float2), cudaMemcpyHostToDevice));

        launch_precompute_sky_grid(
            _d_grid_weights, _d_grid_lms, _d_wavenumbers, _d_antenna_positions,
            nullptr, nullptr, _num_grid_points, _num_local_freq, _num_elements,
            _shared_config.num_active_antennas, nullptr);

        CHECK_CUDA_ERROR_NON_OO(cudaDeviceSynchronize());
        INFO_NON_OO("Direct Beam Tracker: Precomputed Sky Grid with {:d} directions ({:.2f} MB VRAM)",
                    _num_grid_points, grid_bytes / (1024.0 * 1024.0));
    }
}

void cudaDirectBeamTrackerCommand::free_device_buffers() {
    if (_d_weights) { cudaFree(_d_weights); _d_weights = nullptr; }
    if (_d_step_weights) { cudaFree(_d_step_weights); _d_step_weights = nullptr; }
    if (_d_directions) { cudaFree(_d_directions); _d_directions = nullptr; }
    if (_d_directions_end) { cudaFree(_d_directions_end); _d_directions_end = nullptr; }
    if (_d_wavenumbers) { cudaFree(_d_wavenumbers); _d_wavenumbers = nullptr; }
    if (_d_antenna_positions) { cudaFree(_d_antenna_positions); _d_antenna_positions = nullptr; }
    if (_d_antenna_mask) { cudaFree(_d_antenna_mask); _d_antenna_mask = nullptr; }
    if (_d_calibration_gains) { cudaFree(_d_calibration_gains); _d_calibration_gains = nullptr; }
    if (_d_grid_weights) { cudaFree(_d_grid_weights); _d_grid_weights = nullptr; }
    if (_d_grid_lms) { cudaFree(_d_grid_lms); _d_grid_lms = nullptr; }
}

void cudaDirectBeamTrackerCommand::update_weights_if_needed(cudaStream_t stream) {
    // Dynamic real-time celestial tracking: compute topocentric pointing from UTC time
    {
        using namespace std::chrono;
        double unix_s = duration_cast<duration<double>>(system_clock::now().time_since_epoch()).count();
        const double frame_duration_s = static_cast<double>(_samples_per_data_set) * charts::constants::fpga_time_resolution_s;
        const double unix_s_end = unix_s + frame_duration_s;

        const double d_start = unix_s / 86400.0 - 10957.5;
        double gmst_start = std::fmod(18.697374558 + 24.06570982441908 * d_start, 24.0);
        if (gmst_start < 0.0) gmst_start += 24.0;

        const double d_end = unix_s_end / 86400.0 - 10957.5;
        double gmst_end = std::fmod(18.697374558 + 24.06570982441908 * d_end, 24.0);
        if (gmst_end < 0.0) gmst_end += 24.0;

        std::lock_guard<std::mutex> lk(_global_mutex);
        double lst_start = std::fmod(gmst_start + _shared_config.site.lon_deg / 15.0, 24.0);
        if (lst_start < 0.0) lst_start += 24.0;

        double lst_end = std::fmod(gmst_end + _shared_config.site.lon_deg / 15.0, 24.0);
        if (lst_end < 0.0) lst_end += 24.0;

        for (std::size_t b = 0; b < _shared_config.num_active_beams; ++b) {
            if (_shared_config.targets[b].celestial.is_set) {
                DirectDirection3D dir_start, dir_end;
                compute_celestial_direction(
                    _shared_config.targets[b].celestial.ra_deg,
                    _shared_config.targets[b].celestial.dec_deg,
                    lst_start,
                    _shared_config.site.lat_deg,
                    dir_start);
                compute_celestial_direction(
                    _shared_config.targets[b].celestial.ra_deg,
                    _shared_config.targets[b].celestial.dec_deg,
                    lst_end,
                    _shared_config.site.lat_deg,
                    dir_end);

                if (std::abs(dir_start.x - _shared_config.targets[b].direction.x) > 1.0e-6f ||
                    std::abs(dir_start.y - _shared_config.targets[b].direction.y) > 1.0e-6f ||
                    std::abs(dir_end.x - _shared_config.targets[b].direction_end.x) > 1.0e-6f ||
                    std::abs(dir_end.y - _shared_config.targets[b].direction_end.y) > 1.0e-6f) {
                    _shared_config.targets[b].direction = dir_start;
                    _shared_config.targets[b].direction_end = dir_end;
                    _weights_dirty = true;
                }
            }
        }
    }

    if (!_weights_dirty) return;

    DirectBeamTrackerConfig cfg;
    {
        std::lock_guard<std::mutex> lk(_global_mutex);
        cfg = _shared_config;
        _weights_dirty = false;
    }

    // Upload antenna mask
    CHECK_CUDA_ERROR_NON_OO(cudaMemcpyAsync(
        _d_antenna_mask, cfg.antenna_mask.data(),
        _num_elements * sizeof(std::uint8_t), cudaMemcpyHostToDevice, stream));

    if (_grid_mode_enabled && !_h_grid_lms.empty()) {
        // Grid mode: copy precomputed weights slice for each active beam
        const std::size_t beam_stride = static_cast<std::size_t>(_num_local_freq) * _num_elements;
        for (std::size_t b = 0; b < cfg.num_active_beams; ++b) {
            int g_idx = cfg.targets[b].grid_index;
            if (g_idx < 0 || g_idx >= static_cast<int>(_num_grid_points)) {
                g_idx = lookup_nearest_sky_grid(cfg.targets[b].direction.x, cfg.targets[b].direction.y, _h_grid_lms);
            }
            CHECK_CUDA_ERROR_NON_OO(cudaMemcpyAsync(
                _d_weights + b * beam_stride,
                _d_grid_weights + g_idx * beam_stride,
                beam_stride * sizeof(float2),
                cudaMemcpyDeviceToDevice, stream));
        }
    } else {
        // Direct continuous mode: calculate weights for active directions
        std::vector<DirectDirection3D> h_dirs(cfg.num_active_beams);
        std::vector<DirectDirection3D> h_dirs_end(cfg.num_active_beams);
        for (std::size_t b = 0; b < cfg.num_active_beams; ++b) {
            h_dirs[b] = cfg.targets[b].direction;
            h_dirs_end[b] = cfg.targets[b].direction_end;
        }

        CHECK_CUDA_ERROR_NON_OO(cudaMemcpyAsync(
            _d_directions, h_dirs.data(),
            cfg.num_active_beams * sizeof(DirectDirection3D), cudaMemcpyHostToDevice, stream));
        CHECK_CUDA_ERROR_NON_OO(cudaMemcpyAsync(
            _d_directions_end, h_dirs_end.data(),
            cfg.num_active_beams * sizeof(DirectDirection3D), cudaMemcpyHostToDevice, stream));

        if (cfg.num_active_antennas == 0) {
            int count = 0;
            for (int a = 0; a < _num_elements; ++a) {
                if (cfg.antenna_mask[a] != 0) count++;
            }
            cfg.num_active_antennas = count;
        }

        launch_generate_steering_weights(
            _d_weights, _d_step_weights,
            _d_directions, _d_directions_end,
            _d_wavenumbers, _d_antenna_positions,
            _d_antenna_mask, _d_calibration_gains,
            cfg.num_active_beams, _num_local_freq, _num_elements,
            static_cast<std::size_t>(_samples_per_data_set),
            cfg.num_active_antennas, stream);
    }
}

cudaEvent_t cudaDirectBeamTrackerCommand::execute(
    cudaPipelineState& /*pipestate*/, const std::vector<cudaEvent_t>&) {

    pre_execute();

    DirectBeamTrackerConfig current_config;
    {
        std::lock_guard<std::mutex> lock(_global_mutex);
        current_config = _shared_config;
    }

    if (current_config.num_active_beams == 0) {
        return record_end_event();
    }

    const std::size_t input_bytes = static_cast<std::size_t>(_num_elements) *
                                    static_cast<std::size_t>(_num_local_freq) *
                                    static_cast<std::size_t>(_samples_per_data_set) *
                                    sizeof(int4x2_t);
    void* input_memory = device.get_gpu_memory_array(_gpu_mem_voltage, gpu_frame_id,
                                                     _gpu_buffer_depth, input_bytes);

    const std::size_t output_bytes = static_cast<std::size_t>(_num_local_freq) *
                                     static_cast<std::size_t>(_samples_per_data_set) *
                                     static_cast<std::size_t>(_max_beams) *
                                     sizeof(float2);
    void* output_memory = device.get_gpu_memory_array(_gpu_mem_formed_beams, gpu_frame_id,
                                                      _gpu_buffer_depth, output_bytes);
    if (!input_memory || !output_memory) {
        return record_end_event();
    }

    std::shared_ptr<metadataObject> meta = device.get_gpu_memory_array_metadata(_gpu_mem_voltage, gpu_frame_id);
    if (meta) {
        device.claim_gpu_memory_array_metadata(_gpu_mem_formed_beams, gpu_frame_id, meta);
    }

    record_start_event();

    cudaStream_t stream = device.getStream(cuda_stream_id);
    update_weights_if_needed(stream);

    // Invalidate CUDA graphs if active beam count changed
    if (_last_graph_beams != current_config.num_active_beams) {
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
        _last_graph_beams = current_config.num_active_beams;
    }

    const float2* step_weights_ptr = (_enable_subframe_interpolation && !_grid_mode_enabled) ? _d_step_weights : nullptr;

    if (_enable_cuda_graph) {
        const std::size_t slot = static_cast<std::size_t>(gpu_frame_id) % static_cast<std::size_t>(_gpu_buffer_depth);
        if (_cuda_graph_execs[slot] == nullptr) {
            if (_cuda_graphs[slot] != nullptr) {
                cudaGraphDestroy(_cuda_graphs[slot]);
                _cuda_graphs[slot] = nullptr;
            }
            CHECK_CUDA_ERROR_NON_OO(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
            launch_direct_beamformer(
                reinterpret_cast<const int4x2_t*>(input_memory),
                _d_weights,
                step_weights_ptr,
                reinterpret_cast<float2*>(output_memory),
                static_cast<std::size_t>(_samples_per_data_set),
                static_cast<std::size_t>(_num_local_freq),
                static_cast<std::size_t>(_num_elements),
                current_config.num_active_beams,
                static_cast<std::size_t>(_max_beams),
                _time_chunk_size,
                _time_unroll,
                _beam_tile_size,
                stream);
            CHECK_CUDA_ERROR_NON_OO(cudaStreamEndCapture(stream, &_cuda_graphs[slot]));
            CHECK_CUDA_ERROR_NON_OO(cudaGraphInstantiate(&_cuda_graph_execs[slot], _cuda_graphs[slot], nullptr, nullptr, 0));
        }
        CHECK_CUDA_ERROR_NON_OO(cudaGraphLaunch(_cuda_graph_execs[slot], stream));
    } else {
        launch_direct_beamformer(
            reinterpret_cast<const int4x2_t*>(input_memory),
            _d_weights,
            step_weights_ptr,
            reinterpret_cast<float2*>(output_memory),
            static_cast<std::size_t>(_samples_per_data_set),
            static_cast<std::size_t>(_num_local_freq),
            static_cast<std::size_t>(_num_elements),
            current_config.num_active_beams,
            static_cast<std::size_t>(_max_beams),
            _time_chunk_size,
            _time_unroll,
            _beam_tile_size,
            stream);
    }

    return record_end_event();
}

} // namespace kotekan
