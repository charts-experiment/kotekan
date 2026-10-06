#include "cudaAntennaMaskCommand.hpp"
#include "cudaDirectBeamTrackerCommand.hpp"
#include "cudaUtils.hpp"
#include "gpuCommand.hpp"
#include "kotekanLogging.hpp"
#include "restServer.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fmt/format.h>
#include <vector>

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::connectionInstance;
using kotekan::HTTP_RESPONSE;
using kotekan::restServer;
using kotekan::cudaAntennaMaskCommand;

REGISTER_CUDA_COMMAND(cudaAntennaMaskCommand);

namespace kotekan {

std::mutex cudaAntennaMaskCommand::_global_mutex;
AntennaMaskConfig cudaAntennaMaskCommand::_shared_config;
std::array<AntennaHealthMetrics, MAX_MASK_ANTENNAS> cudaAntennaMaskCommand::_shared_metrics{};
std::array<std::uint8_t, MAX_MASK_ANTENNAS> cudaAntennaMaskCommand::_shared_mask{};
bool cudaAntennaMaskCommand::_endpoints_registered = false;
bool cudaAntennaMaskCommand::_mask_dirty = true;

cudaAntennaMaskCommand::cudaAntennaMaskCommand(
    Config& config, const std::string& unique_name,
    bufferContainer& host_buffers, cudaDeviceInterface& device, int inst)
    : cudaCommand(config, unique_name, host_buffers, device, inst,
                  no_cuda_command_state, "cudaAntennaMaskCommand") {

    _num_elements = config.get<int>(unique_name, "num_elements");
    _num_local_freq = config.get<int>(unique_name, "num_local_freq");
    _samples_per_data_set = config.get<int>(unique_name, "samples_per_data_set");
    _buffer_depth = config.get<int>(unique_name, "buffer_depth");

    _gpu_mem_voltage = config.get_default<std::string>(unique_name, "gpu_mem_voltage", "voltage");
    _gpu_mem_alive_voltages = config.get_default<std::string>(unique_name, "gpu_mem_alive_voltages", "");
    _max_alive_antennas = config.get_default<int>(unique_name, "max_alive_antennas", _num_elements);
    _antenna_order = config.get_default<std::string>(unique_name, "antenna_order", "descending");

    {
        std::lock_guard<std::mutex> lock(_global_mutex);
        _shared_config.auto_detect_enabled = config.get_default<bool>(unique_name, "auto_detect_enabled", true);
        _shared_config.blank_voltages_enabled = config.get_default<bool>(unique_name, "blank_voltages_enabled", false);
        _shared_config.dead_power_threshold = config.get_default<float>(unique_name, "dead_power_threshold", 5.0f);
        _shared_config.sat_power_threshold = config.get_default<float>(unique_name, "sat_power_threshold", 80.0f);
        _shared_config.clip_fraction_threshold = config.get_default<float>(unique_name, "clip_fraction_threshold", 0.15f);
        _shared_config.revival_frames = static_cast<std::uint16_t>(config.get_default<int>(unique_name, "revival_frames", 5));
        _shared_config.sample_stride = static_cast<std::size_t>(config.get_default<int>(unique_name, "sample_stride", 1));

        // Initialize masks
        _shared_mask.fill(1);
        _shared_config.manual_mask.fill(1);

        auto to_raw = [&](int ant) {
            return (_antenna_order == "descending") ? ((_num_elements - 1) - ant) : ant;
        };

        auto bad_antennas = config.get_default<std::vector<int>>(unique_name, "bad_antennas", {});
        for (int ant : bad_antennas) {
            int elem = to_raw(ant);
            if (elem >= 0 && elem < _num_elements && elem < static_cast<int>(MAX_MASK_ANTENNAS)) {
                _shared_config.manual_mask[elem] = 0;
                _shared_mask[elem] = 0;
                _shared_metrics[elem].status = AntennaHealthStatus::MANUAL_MASK;
            }
        }

        auto bad_elements = config.get_default<std::vector<int>>(unique_name, "bad_elements", {});
        for (int elem : bad_elements) {
            if (elem >= 0 && elem < _num_elements && elem < static_cast<int>(MAX_MASK_ANTENNAS)) {
                _shared_config.manual_mask[elem] = 0;
                _shared_mask[elem] = 0;
                _shared_metrics[elem].status = AntennaHealthStatus::MANUAL_MASK;
            }
        }

        auto active_antennas = config.get_default<std::vector<int>>(unique_name, "active_antennas", {});
        if (!active_antennas.empty()) {
            _shared_config.manual_mask.fill(0);
            _shared_mask.fill(0);
            for (std::size_t i = 0; i < MAX_MASK_ANTENNAS; ++i) {
                _shared_metrics[i].status = AntennaHealthStatus::MANUAL_MASK;
                _shared_metrics[i].consecutive_healthy = 0;
            }
            for (int ant : active_antennas) {
                int r = to_raw(ant);
                if (r >= 0 && r < _num_elements && r < static_cast<int>(MAX_MASK_ANTENNAS)) {
                    _shared_config.manual_mask[r] = 1;
                    _shared_mask[r] = 1;
                }
            }
        }

        auto active_raw = config.get_default<std::vector<int>>(unique_name, "active_raw_elements", {});
        if (!active_raw.empty()) {
            _shared_config.manual_mask.fill(0);
            _shared_mask.fill(0);
            for (std::size_t i = 0; i < MAX_MASK_ANTENNAS; ++i) {
                _shared_metrics[i].status = AntennaHealthStatus::MANUAL_MASK;
                _shared_metrics[i].consecutive_healthy = 0;
            }
            for (int r : active_raw) {
                if (r >= 0 && r < _num_elements && r < static_cast<int>(MAX_MASK_ANTENNAS)) {
                    _shared_config.manual_mask[r] = 1;
                    _shared_mask[r] = 1;
                }
            }
        }

        // Initialize metrics
        for (std::size_t i = 0; i < MAX_MASK_ANTENNAS; ++i) {
            if (_shared_mask[i] != 0) {
                _shared_metrics[i].status = AntennaHealthStatus::HEALTHY;
                _shared_metrics[i].consecutive_healthy = _shared_config.revival_frames;
            }
        }

        _mask_dirty = true;

        // Register REST endpoints once
        if (!_endpoints_registered) {
            auto& rest = restServer::instance();

            // 1. GET /antenna_mask/status
            int num_elements = _num_elements;
            auto status_cb = [this, num_elements](connectionInstance& conn) {
                nlohmann::json reply;
                std::lock_guard<std::mutex> lk(_global_mutex);

                reply["version"] = "Kotekan Antenna Masking Stage v1.0";
                reply["num_elements"] = num_elements;
                reply["antenna_order"] = _antenna_order;
                reply["auto_detect_enabled"] = _shared_config.auto_detect_enabled;
                reply["blank_voltages_enabled"] = _shared_config.blank_voltages_enabled;
                reply["thresholds"] = {
                    {"dead_power_threshold", _shared_config.dead_power_threshold},
                    {"sat_power_threshold", _shared_config.sat_power_threshold},
                    {"clip_fraction_threshold", _shared_config.clip_fraction_threshold},
                    {"revival_frames", _shared_config.revival_frames},
                    {"sample_stride", _shared_config.sample_stride}
                };

                int active_count = 0;
                int dead_count = 0;
                int sat_count = 0;
                int manual_count = 0;
                nlohmann::json ant_array = nlohmann::json::array();

                auto to_phys = [&](int raw_id) {
                    return (_antenna_order == "descending") ? ((num_elements - 1) - raw_id) : raw_id;
                };

                for (int a = 0; a < num_elements; ++a) {
                    const auto& m = _shared_metrics[a];
                    const bool active = (_shared_mask[a] != 0);
                    if (active) active_count++;
                    if (m.status == AntennaHealthStatus::DEAD) dead_count++;
                    else if (m.status == AntennaHealthStatus::SATURATED) sat_count++;
                    else if (m.status == AntennaHealthStatus::MANUAL_MASK) manual_count++;

                    ant_array.push_back({
                        {"id", a},
                        {"raw_id", a},
                        {"physical_id", to_phys(a)},
                        {"active", active},
                        {"status", antenna_health_status_to_string(m.status)},
                        {"power", m.power},
                        {"clipping_fraction", m.clipping_fraction},
                        {"clipped_count", m.clipped_count},
                        {"consecutive_healthy", m.consecutive_healthy}
                    });
                }

                reply["active_antennas"] = active_count;
                reply["masked_antennas"] = num_elements - active_count;
                reply["dead_antennas"] = dead_count;
                reply["saturated_antennas"] = sat_count;
                reply["manual_masked_antennas"] = manual_count;

                nlohmann::json active_indices = nlohmann::json::array();
                nlohmann::json active_phys = nlohmann::json::array();
                for (int a = 0; a < num_elements; ++a) {
                    if (_shared_mask[a] != 0) {
                        active_indices.push_back(a);
                        active_phys.push_back(to_phys(a));
                    }
                }
                reply["active_raw_elements"] = active_indices;
                reply["active_physical_antennas"] = active_phys;
                reply["antennas"] = ant_array;

                conn.send_json_reply(reply);
            };
            rest.register_get_callback("/antenna_mask/status", status_cb);

            // 2. POST /antenna_mask/mask
            auto mask_cb = [this, num_elements](connectionInstance& conn, nlohmann::json& j) {
                try {
                    bool is_raw = j.value("is_raw", false);
                    std::size_t raw_id = 0;
                    std::size_t phys_id = 0;
                    if (j.contains("physical_id")) {
                        phys_id = j["physical_id"];
                        raw_id = (_antenna_order == "descending") ? ((num_elements - 1) - phys_id) : phys_id;
                    } else if (j.contains("raw_id")) {
                        raw_id = j["raw_id"];
                        phys_id = (_antenna_order == "descending") ? ((num_elements - 1) - raw_id) : raw_id;
                    } else if (j.contains("antenna_id")) {
                        std::size_t ant_id = j["antenna_id"];
                        if (is_raw) {
                            raw_id = ant_id;
                            phys_id = (_antenna_order == "descending") ? ((num_elements - 1) - raw_id) : raw_id;
                        } else {
                            phys_id = ant_id;
                            raw_id = (_antenna_order == "descending") ? ((num_elements - 1) - phys_id) : phys_id;
                        }
                    } else {
                        conn.send_error("Missing antenna_id, physical_id, or raw_id", HTTP_RESPONSE::BAD_REQUEST);
                        return;
                    }

                    if (raw_id >= MAX_MASK_ANTENNAS) {
                        conn.send_error("antenna index exceeds MAX_MASK_ANTENNAS", HTTP_RESPONSE::BAD_REQUEST);
                        return;
                    }
                    std::lock_guard<std::mutex> lk(_global_mutex);
                    _shared_config.manual_mask[raw_id] = 0;
                    _shared_mask[raw_id] = 0;
                    _shared_metrics[raw_id].status = AntennaHealthStatus::MANUAL_MASK;
                    _shared_metrics[raw_id].consecutive_healthy = 0;
                    _mask_dirty = true;
                    int n_active = 0;
                    for (int i = 0; i < num_elements; ++i) {
                        if (_shared_mask[i] != 0) n_active++;
                    }
                    cudaDirectBeamTrackerCommand::set_shared_antenna_mask(_shared_mask, n_active);
                    INFO_NON_OO("AntennaMask: Physical antenna {:d} (raw element {:d}) MANUAL MASK applied", phys_id, raw_id);
                    conn.send_text_reply(fmt::format("Physical antenna {:d} (raw element {:d}) masked successfully\n", phys_id, raw_id));
                } catch (const std::exception& e) {
                    conn.send_error(e.what(), HTTP_RESPONSE::BAD_REQUEST);
                }
            };
            rest.register_post_callback("/antenna_mask/mask", mask_cb);

            // 3. POST /antenna_mask/unmask
            auto unmask_cb = [this, num_elements](connectionInstance& conn, nlohmann::json& j) {
                try {
                    bool is_raw = j.value("is_raw", false);
                    std::size_t raw_id = 0;
                    std::size_t phys_id = 0;
                    if (j.contains("physical_id")) {
                        phys_id = j["physical_id"];
                        raw_id = (_antenna_order == "descending") ? ((num_elements - 1) - phys_id) : phys_id;
                    } else if (j.contains("raw_id")) {
                        raw_id = j["raw_id"];
                        phys_id = (_antenna_order == "descending") ? ((num_elements - 1) - raw_id) : raw_id;
                    } else if (j.contains("antenna_id")) {
                        std::size_t ant_id = j["antenna_id"];
                        if (is_raw) {
                            raw_id = ant_id;
                            phys_id = (_antenna_order == "descending") ? ((num_elements - 1) - raw_id) : raw_id;
                        } else {
                            phys_id = ant_id;
                            raw_id = (_antenna_order == "descending") ? ((num_elements - 1) - phys_id) : phys_id;
                        }
                    } else {
                        conn.send_error("Missing antenna_id, physical_id, or raw_id", HTTP_RESPONSE::BAD_REQUEST);
                        return;
                    }

                    if (raw_id >= MAX_MASK_ANTENNAS) {
                        conn.send_error("antenna index exceeds MAX_MASK_ANTENNAS", HTTP_RESPONSE::BAD_REQUEST);
                        return;
                    }
                    std::lock_guard<std::mutex> lk(_global_mutex);
                    _shared_config.manual_mask[raw_id] = 1;
                    _shared_mask[raw_id] = 1;
                    _shared_metrics[raw_id].status = AntennaHealthStatus::HEALTHY;
                    _shared_metrics[raw_id].consecutive_healthy = _shared_config.revival_frames;
                    _mask_dirty = true;
                    int n_active = 0;
                    for (int i = 0; i < num_elements; ++i) {
                        if (_shared_mask[i] != 0) n_active++;
                    }
                    cudaDirectBeamTrackerCommand::set_shared_antenna_mask(_shared_mask, n_active);
                    INFO_NON_OO("AntennaMask: Physical antenna {:d} (raw element {:d}) UNMASK applied", phys_id, raw_id);
                    conn.send_text_reply(fmt::format("Physical antenna {:d} (raw element {:d}) unmasked successfully\n", phys_id, raw_id));
                } catch (const std::exception& e) {
                    conn.send_error(e.what(), HTTP_RESPONSE::BAD_REQUEST);
                }
            };
            rest.register_post_callback("/antenna_mask/unmask", unmask_cb);

            // 4. POST /antenna_mask/reset
            auto reset_cb = [num_elements](connectionInstance& conn, nlohmann::json& /*j*/) {
                std::lock_guard<std::mutex> lk(_global_mutex);
                _shared_config.manual_mask.fill(1);
                _shared_mask.fill(1);
                for (std::size_t i = 0; i < MAX_MASK_ANTENNAS; ++i) {
                    _shared_metrics[i].status = AntennaHealthStatus::HEALTHY;
                    _shared_metrics[i].consecutive_healthy = _shared_config.revival_frames;
                }
                _mask_dirty = true;
                int n_active = 0;
                for (int i = 0; i < num_elements; ++i) {
                    if (_shared_mask[i] != 0) n_active++;
                }
                cudaDirectBeamTrackerCommand::set_shared_antenna_mask(_shared_mask, n_active);
                INFO_NON_OO("AntennaMask: Reset all antenna masks to ACTIVE");
                conn.send_text_reply("All antenna masks reset to active\n");
            };
            rest.register_post_callback("/antenna_mask/reset", reset_cb);

            // 5. POST /antenna_mask/configure
            auto config_cb = [](connectionInstance& conn, nlohmann::json& j) {
                try {
                    std::lock_guard<std::mutex> lk(_global_mutex);
                    if (j.contains("auto_detect_enabled")) _shared_config.auto_detect_enabled = j["auto_detect_enabled"];
                    if (j.contains("blank_voltages_enabled")) _shared_config.blank_voltages_enabled = j["blank_voltages_enabled"];
                    if (j.contains("dead_power_threshold")) _shared_config.dead_power_threshold = j["dead_power_threshold"];
                    if (j.contains("sat_power_threshold")) _shared_config.sat_power_threshold = j["sat_power_threshold"];
                    if (j.contains("clip_fraction_threshold")) _shared_config.clip_fraction_threshold = j["clip_fraction_threshold"];
                    if (j.contains("revival_frames")) _shared_config.revival_frames = j["revival_frames"];
                    if (j.contains("sample_stride")) _shared_config.sample_stride = j["sample_stride"];
                    conn.send_text_reply("Antenna mask configuration updated successfully\n");
                } catch (const std::exception& e) {
                    conn.send_error(e.what(), HTTP_RESPONSE::BAD_REQUEST);
                }
            };
            rest.register_post_callback("/antenna_mask/configure", config_cb);

            _endpoints_registered = true;
        }
    }

    set_command_type(gpuCommandType::KERNEL);
    set_name("cudaAntennaMaskCommand");
    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_voltage, true, true, true));
    if (!_gpu_mem_alive_voltages.empty()) {
        gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_alive_voltages, true, false, true));
    }

    allocate_device_buffers();
    INFO_NON_OO("Kotekan Antenna Masking Stage initialized for {:d} elements ({:s})",
                _num_elements, unique_name.c_str());
}

cudaAntennaMaskCommand::~cudaAntennaMaskCommand() {
    free_device_buffers();
}

void cudaAntennaMaskCommand::allocate_device_buffers() {
    device.set_thread_device();
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&_d_antenna_powers, _num_elements * sizeof(float)));
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&_d_antenna_clips, _num_elements * sizeof(std::uint32_t)));
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&_d_bad_antennas, _num_elements * sizeof(int)));
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&_d_alive_antennas, _num_elements * sizeof(int)));

    CHECK_CUDA_ERROR_NON_OO(cudaMallocHost(&_h_antenna_powers, _num_elements * sizeof(float)));
    CHECK_CUDA_ERROR_NON_OO(cudaMallocHost(&_h_antenna_clips, _num_elements * sizeof(std::uint32_t)));
}

void cudaAntennaMaskCommand::free_device_buffers() {
    device.set_thread_device();
    if (_d_antenna_powers) { cudaFree(_d_antenna_powers); _d_antenna_powers = nullptr; }
    if (_d_antenna_clips) { cudaFree(_d_antenna_clips); _d_antenna_clips = nullptr; }
    if (_d_bad_antennas) { cudaFree(_d_bad_antennas); _d_bad_antennas = nullptr; }
    if (_d_alive_antennas) { cudaFree(_d_alive_antennas); _d_alive_antennas = nullptr; }

    if (_h_antenna_powers) { cudaFreeHost(_h_antenna_powers); _h_antenna_powers = nullptr; }
    if (_h_antenna_clips) { cudaFreeHost(_h_antenna_clips); _h_antenna_clips = nullptr; }
}

std::array<std::uint8_t, MAX_MASK_ANTENNAS> cudaAntennaMaskCommand::get_current_mask() {
    std::lock_guard<std::mutex> lk(_global_mutex);
    return _shared_mask;
}

AntennaHealthMetrics cudaAntennaMaskCommand::get_antenna_metrics(std::size_t ant_id) {
    std::lock_guard<std::mutex> lk(_global_mutex);
    if (ant_id < MAX_MASK_ANTENNAS) return _shared_metrics[ant_id];
    return AntennaHealthMetrics{};
}

void cudaAntennaMaskCommand::set_manual_mask(std::size_t ant_id, bool enable) {
    std::lock_guard<std::mutex> lk(_global_mutex);
    if (ant_id < MAX_MASK_ANTENNAS) {
        _shared_config.manual_mask[ant_id] = enable ? 1 : 0;
        _shared_mask[ant_id] = enable ? 1 : 0;
        _shared_metrics[ant_id].status = enable ? AntennaHealthStatus::HEALTHY : AntennaHealthStatus::MANUAL_MASK;
        _shared_metrics[ant_id].consecutive_healthy = enable ? _shared_config.revival_frames : 0;
        _mask_dirty = true;
        cudaDirectBeamTrackerCommand::set_shared_antenna_mask(_shared_mask);
    }
}

void cudaAntennaMaskCommand::reset_all_masks() {
    std::lock_guard<std::mutex> lk(_global_mutex);
    _shared_config.manual_mask.fill(1);
    _shared_mask.fill(1);
    for (std::size_t i = 0; i < MAX_MASK_ANTENNAS; ++i) {
        _shared_metrics[i].status = AntennaHealthStatus::HEALTHY;
        _shared_metrics[i].consecutive_healthy = _shared_config.revival_frames;
    }
    _mask_dirty = true;
    cudaDirectBeamTrackerCommand::set_shared_antenna_mask(_shared_mask);
}

void cudaAntennaMaskCommand::update_config(const AntennaMaskConfig& cfg) {
    std::lock_guard<std::mutex> lk(_global_mutex);
    _shared_config = cfg;
    _mask_dirty = true;
}

AntennaMaskConfig cudaAntennaMaskCommand::get_config() {
    std::lock_guard<std::mutex> lk(_global_mutex);
    return _shared_config;
}

cudaEvent_t cudaAntennaMaskCommand::execute(
    cudaPipelineState& /*pipestate*/, const std::vector<cudaEvent_t>&) {

    pre_execute();

    const std::size_t input_bytes = static_cast<std::size_t>(_num_elements) *
                                    static_cast<std::size_t>(_num_local_freq) *
                                    static_cast<std::size_t>(_samples_per_data_set) *
                                    sizeof(int4x2_t);

    void* input_memory = device.get_gpu_memory_array(_gpu_mem_voltage, gpu_frame_id,
                                                     _gpu_buffer_depth, input_bytes);
    if (!input_memory) {
        return record_end_event();
    }

    record_start_event();
    cudaStream_t stream = device.getStream(cuda_stream_id);
    int4x2_t* d_voltages = reinterpret_cast<int4x2_t*>(input_memory);

    AntennaMaskConfig current_cfg;
    {
        std::lock_guard<std::mutex> lk(_global_mutex);
        current_cfg = _shared_config;
    }

    const std::size_t total_spectra = static_cast<std::size_t>(_samples_per_data_set) * _num_local_freq;
    const std::size_t inspected_spectra = (total_spectra + current_cfg.sample_stride - 1) / current_cfg.sample_stride;

    if (current_cfg.auto_detect_enabled) {
        // 1. Inspect on GPU
        launch_inspect_antenna_health(
            d_voltages, _d_antenna_powers, _d_antenna_clips,
            _samples_per_data_set, _num_local_freq, _num_elements,
            current_cfg.sample_stride, stream);

        // 2. Low-latency transfer of metrics (2 KB)
        CHECK_CUDA_ERROR_NON_OO(cudaMemcpyAsync(
            _h_antenna_powers, _d_antenna_powers,
            _num_elements * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA_ERROR_NON_OO(cudaMemcpyAsync(
            _h_antenna_clips, _d_antenna_clips,
            _num_elements * sizeof(std::uint32_t), cudaMemcpyDeviceToHost, stream));

        cudaStreamSynchronize(stream);

        // 3. Evaluate health with hysteresis debounce
        std::lock_guard<std::mutex> lk(_global_mutex);
        bool mask_changed = false;
        _current_bad_indices.clear();
        _current_alive_indices.clear();

        for (int a = 0; a < _num_elements; ++a) {
            float p = _h_antenna_powers[a];
            std::uint32_t clips = _h_antenna_clips[a];
            float clip_frac = (inspected_spectra > 0) ? (static_cast<float>(clips) / (2.0f * inspected_spectra)) : 0.0f;

            auto& m = _shared_metrics[a];
            m.power = p;
            m.clipping_fraction = clip_frac;
            m.clipped_count = clips;

            uint8_t old_mask_val = _shared_mask[a];
            uint8_t new_mask_val = 1;

            if (_shared_config.manual_mask[a] == 0) {
                m.status = AntennaHealthStatus::MANUAL_MASK;
                m.consecutive_healthy = 0;
                new_mask_val = 0;
            } else if (p <= _shared_config.dead_power_threshold) {
                m.status = AntennaHealthStatus::DEAD;
                m.consecutive_healthy = 0;
                new_mask_val = 0;
            } else if (clip_frac >= _shared_config.clip_fraction_threshold || p >= _shared_config.sat_power_threshold) {
                m.status = AntennaHealthStatus::SATURATED;
                m.consecutive_healthy = 0;
                new_mask_val = 0;
            } else {
                // Healthy candidate
                m.consecutive_healthy++;
                if (old_mask_val == 0) {
                    // Requires consecutive healthy frames to revive
                    if (m.consecutive_healthy >= _shared_config.revival_frames) {
                        m.status = AntennaHealthStatus::HEALTHY;
                        new_mask_val = 1;
                    } else {
                        // Keep masked during recovery debounce
                        new_mask_val = 0;
                    }
                } else {
                    m.status = AntennaHealthStatus::HEALTHY;
                    new_mask_val = 1;
                }
            }

            if (new_mask_val != old_mask_val) {
                _shared_mask[a] = new_mask_val;
                mask_changed = true;
            }

            if (new_mask_val == 0) {
                _current_bad_indices.push_back(a);
            } else {
                _current_alive_indices.push_back(a);
            }
        }

        if (mask_changed || _mask_dirty) {
            int n_active = 0;
            for (int i = 0; i < _num_elements; ++i) {
                if (_shared_mask[i] != 0) n_active++;
            }
            cudaDirectBeamTrackerCommand::set_shared_antenna_mask(_shared_mask, n_active);
            _mask_dirty = false;
        }
    } else {
        std::lock_guard<std::mutex> lk(_global_mutex);
        _current_bad_indices.clear();
        _current_alive_indices.clear();
        for (int a = 0; a < _num_elements; ++a) {
            if (_shared_mask[a] == 0) {
                _current_bad_indices.push_back(a);
            } else {
                _current_alive_indices.push_back(a);
            }
        }
    }

    // 4. Level 1: In-Place Voltage Blanking for all bad antennas
    if (current_cfg.blank_voltages_enabled && !_current_bad_indices.empty()) {
        CHECK_CUDA_ERROR_NON_OO(cudaMemcpyAsync(
            _d_bad_antennas, _current_bad_indices.data(),
            _current_bad_indices.size() * sizeof(int), cudaMemcpyHostToDevice, stream));

        launch_zero_bad_antennas(
            d_voltages, _d_bad_antennas, static_cast<int>(_current_bad_indices.size()),
            total_spectra, _num_elements, stream);
    }

    // 5. Extract Alive Antennas baseband into compact buffer if requested
    if (!_gpu_mem_alive_voltages.empty()) {
        const std::size_t alive_bytes = static_cast<std::size_t>(_max_alive_antennas) *
                                        static_cast<std::size_t>(_num_local_freq) *
                                        static_cast<std::size_t>(_samples_per_data_set) *
                                        sizeof(int4x2_t);
        void* alive_memory = device.get_gpu_memory_array(_gpu_mem_alive_voltages, gpu_frame_id,
                                                         _gpu_buffer_depth, alive_bytes);
        if (alive_memory) {
            int num_to_extract = static_cast<int>(std::min(static_cast<std::size_t>(_max_alive_antennas),
                                                           _current_alive_indices.size()));
            if (num_to_extract > 0) {
                CHECK_CUDA_ERROR_NON_OO(cudaMemcpyAsync(
                    _d_alive_antennas, _current_alive_indices.data(),
                    num_to_extract * sizeof(int), cudaMemcpyHostToDevice, stream));
            }

            int4x2_t* d_alive_voltages = reinterpret_cast<int4x2_t*>(alive_memory);
            launch_extract_alive_antennas(
                d_voltages, d_alive_voltages, _d_alive_antennas,
                num_to_extract, _max_alive_antennas, total_spectra, _num_elements, stream);

            std::shared_ptr<metadataObject> meta =
                device.get_gpu_memory_array_metadata(_gpu_mem_voltage, gpu_frame_id);
            if (meta) {
                device.claim_gpu_memory_array_metadata(_gpu_mem_alive_voltages, gpu_frame_id, meta);
            }
        }
    }

    return record_end_event();
}

} // namespace kotekan
