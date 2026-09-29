#include "cudaBiSLCCommand.hpp"
#include "cudaBiSLC.hpp"
#include "cudaDirectBeamTrackerCommand.hpp"
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
using kotekan::cudaBiSLCCommand;

REGISTER_CUDA_COMMAND(cudaBiSLCCommand);

namespace kotekan {

std::mutex cudaBiSLCCommand::_global_mutex;
BiSLCConfig cudaBiSLCCommand::_shared_config;
bool cudaBiSLCCommand::_endpoints_registered = false;
bool cudaBiSLCCommand::_matrices_dirty = true;

void cudaBiSLCCommand::set_shared_config(const BiSLCConfig& config) {
    std::lock_guard<std::mutex> lock(_global_mutex);
    _shared_config = config;
    _matrices_dirty = true;
}

BiSLCConfig cudaBiSLCCommand::get_shared_config() {
    std::lock_guard<std::mutex> lock(_global_mutex);
    return _shared_config;
}

cudaBiSLCCommand::cudaBiSLCCommand(
    Config& config, const std::string& unique_name,
    bufferContainer& host_buffers, cudaDeviceInterface& device, int inst)
    : cudaCommand(config, unique_name, host_buffers, device, inst,
                  no_cuda_command_state, "cudaBiSLCCommand") {

    _num_elements = config.get<int>(unique_name, "num_elements");
    _num_local_freq = config.get<int>(unique_name, "num_local_freq");
    _samples_per_data_set = config.get<int>(unique_name, "samples_per_data_set");
    _buffer_depth = config.get<int>(unique_name, "buffer_depth");

    _max_beams = config.get_default<int>(unique_name, "max_beams", 2);
    _spacing_m = config.get_default<float>(unique_name, "spacing_m", charts::constants::charts_default_spacing_m);
    _diagonal_loading = config.get_default<float>(unique_name, "diagonal_loading", 1.0e-4f);
    _enabled = config.get_default<bool>(unique_name, "enabled", true);

    _gpu_mem_formed_beams = config.get_default<std::string>(unique_name, "gpu_mem_formed_beams", "formed_beams");
    _gpu_mem_cleaned_beams = config.get_default<std::string>(unique_name, "gpu_mem_cleaned_beams", "cleaned_beams");

    const double freq_start_hz = config.get_default<double>(unique_name, "freq_start_hz", 300.0e6);
    const double freq_step_hz = config.get_default<double>(unique_name, "freq_step_hz", 300.0e3);

    _frequencies_hz.resize(static_cast<std::size_t>(_num_local_freq));
    for (int f = 0; f < _num_local_freq; ++f) {
        _frequencies_hz[f] = freq_start_hz + f * freq_step_hz;
    }

    {
        std::lock_guard<std::mutex> lock(_global_mutex);
        _shared_config.enabled = _enabled;
        _shared_config.diagonal_loading = _diagonal_loading;
        _shared_config.max_beams_stride = static_cast<std::size_t>(_max_beams);
        _shared_config.num_active_beams = config.get_default<int>(unique_name, "initial_active_beams", 2);
        _matrices_dirty = true;

        if (!_endpoints_registered) {
            auto& rest = restServer::instance();

            // 1. POST /bislc/enable
            auto enable_cb = [](connectionInstance& conn, nlohmann::json& j) {
                try {
                    bool en = j.value("enabled", true);
                    std::lock_guard<std::mutex> lk(_global_mutex);
                    _shared_config.enabled = en;
                    _matrices_dirty = true;
                    INFO_NON_OO("BiSLC: Unmixing stage {:s}", en ? "ENABLED" : "DISABLED");
                    conn.send_text_reply(fmt::format("BiSLC unmixing {:s}\n", en ? "enabled" : "disabled"));
                } catch (const std::exception& e) {
                    conn.send_error(e.what(), HTTP_RESPONSE::BAD_REQUEST);
                }
            };
            rest.register_post_callback("/bislc/enable", enable_cb);

            // 2. POST /bislc/set_diagonal_loading
            auto set_diag_cb = [](connectionInstance& conn, nlohmann::json& j) {
                try {
                    if (!j.contains("diagonal_loading")) {
                        conn.send_error("Missing diagonal_loading", HTTP_RESPONSE::BAD_REQUEST);
                        return;
                    }
                    float diag = j["diagonal_loading"];
                    std::lock_guard<std::mutex> lk(_global_mutex);
                    _shared_config.diagonal_loading = diag;
                    _matrices_dirty = true;
                    INFO_NON_OO("BiSLC: Diagonal loading set to {:.2e}", diag);
                    conn.send_text_reply(fmt::format("Diagonal loading set to {:.2e}\n", diag));
                } catch (const std::exception& e) {
                    conn.send_error(e.what(), HTTP_RESPONSE::BAD_REQUEST);
                }
            };
            rest.register_post_callback("/bislc/set_diagonal_loading", set_diag_cb);

            // 3. GET /bislc/status
            auto status_cb = [this](connectionInstance& conn) {
                nlohmann::json reply;
                std::lock_guard<std::mutex> lk(_global_mutex);
                reply["version"] = "cudaBiSLCCommand v1.0 (Joint Matrix Unmixing)";
                reply["enabled"] = _shared_config.enabled;
                reply["diagonal_loading"] = _shared_config.diagonal_loading;
                reply["num_active_beams"] = _shared_config.num_active_beams;
                reply["max_beams"] = _max_beams;
                reply["num_local_freq"] = _num_local_freq;
                reply["samples_per_data_set"] = _samples_per_data_set;

                // Compute average off-diagonal coupling magnitude across channels
                double avg_sidelobe_mag = 0.0;
                std::size_t off_diag_count = 0;
                const std::size_t B = _shared_config.num_active_beams;

                if (B > 1 && !_h_coupling_matrices.empty()) {
                    for (int f = 0; f < _num_local_freq; ++f) {
                        const std::size_t f_off = f * B * B;
                        for (std::size_t i = 0; i < B; ++i) {
                            for (std::size_t j = 0; j < B; ++j) {
                                if (i != j) {
                                    const float2 c = _h_coupling_matrices[f_off + i * B + j];
                                    avg_sidelobe_mag += std::sqrt(c.x * c.x + c.y * c.y);
                                    off_diag_count++;
                                }
                            }
                        }
                    }
                }
                if (off_diag_count > 0) {
                    reply["average_coupling_magnitude"] = avg_sidelobe_mag / off_diag_count;
                } else {
                    reply["average_coupling_magnitude"] = 0.0;
                }

                conn.send_json_reply(reply);
            };
            rest.register_get_callback("/bislc/status", status_cb);

            _endpoints_registered = true;
        }
    }

    set_command_type(gpuCommandType::KERNEL);
    set_name("cudaBiSLCCommand");

    if (_gpu_mem_cleaned_beams == _gpu_mem_formed_beams) {
        gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_formed_beams, true, true, true));
    } else {
        gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_formed_beams, true, false, true));
        gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_cleaned_beams, false, true, true));
    }

    allocate_device_buffers();
}

cudaBiSLCCommand::~cudaBiSLCCommand() {
    free_device_buffers();
}

void cudaBiSLCCommand::allocate_device_buffers() {
    const std::size_t matrix_bytes = static_cast<std::size_t>(_num_local_freq) *
                                     static_cast<std::size_t>(_max_beams) *
                                     static_cast<std::size_t>(_max_beams) * sizeof(float2);

    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&_d_unmix_matrices, matrix_bytes));
    _h_unmix_matrices.resize(static_cast<std::size_t>(_num_local_freq) * _max_beams * _max_beams);
    _h_coupling_matrices.resize(static_cast<std::size_t>(_num_local_freq) * _max_beams * _max_beams);
}

void cudaBiSLCCommand::free_device_buffers() {
    if (_d_unmix_matrices) {
        cudaFree(_d_unmix_matrices);
        _d_unmix_matrices = nullptr;
    }
}

void cudaBiSLCCommand::update_matrices_if_needed(cudaStream_t stream) {
    // Inherit latest tracking coordinates and active masks from direct beam tracker
    DirectBeamTrackerConfig tracker_cfg = cudaDirectBeamTrackerCommand::get_shared_config();

    BiSLCConfig current_cfg;
    bool needs_update = false;
    {
        std::lock_guard<std::mutex> lock(_global_mutex);
        current_cfg = _shared_config;

        if (_matrices_dirty ||
            current_cfg.num_active_beams != tracker_cfg.num_active_beams ||
            current_cfg.antenna_mask != tracker_cfg.antenna_mask) {

            current_cfg.num_active_beams = tracker_cfg.num_active_beams;
            current_cfg.antenna_mask = tracker_cfg.antenna_mask;
            current_cfg.targets = tracker_cfg.targets;
            current_cfg.antenna_positions = tracker_cfg.antenna_positions;
            current_cfg.num_active_antennas = tracker_cfg.num_active_antennas;

            _shared_config = current_cfg;
            _matrices_dirty = false;
            needs_update = true;
        }
    }

    if (!needs_update) return;

    const std::size_t B = std::min(current_cfg.num_active_beams, static_cast<std::size_t>(_max_beams));
    if (B == 0) return;

    // Compute coupling matrices A(f) and unmix matrices M(f) = A^{-1}(f) on host
    compute_beam_coupling_and_inverses(
        _h_unmix_matrices.data(),
        _h_coupling_matrices.data(),
        current_cfg.targets.data(),
        _frequencies_hz,
        current_cfg.antenna_positions.data(),
        current_cfg.antenna_mask.data(),
        B,
        static_cast<std::size_t>(_num_local_freq),
        static_cast<std::size_t>(_num_elements),
        current_cfg.num_active_antennas,
        current_cfg.diagonal_loading);

    // Asynchronously upload unmixing matrices M(f) to GPU
    const std::size_t matrix_bytes = static_cast<std::size_t>(_num_local_freq) * B * B * sizeof(float2);
    CHECK_CUDA_ERROR_NON_OO(cudaMemcpyAsync(
        _d_unmix_matrices, _h_unmix_matrices.data(),
        matrix_bytes, cudaMemcpyHostToDevice, stream));
}

cudaEvent_t cudaBiSLCCommand::execute(
    cudaPipelineState& /*pipestate*/, const std::vector<cudaEvent_t>&) {

    pre_execute();

    BiSLCConfig current_config;
    {
        std::lock_guard<std::mutex> lock(_global_mutex);
        current_config = _shared_config;
    }

    const std::size_t buffer_bytes = static_cast<std::size_t>(_num_local_freq) *
                                     static_cast<std::size_t>(_samples_per_data_set) *
                                     static_cast<std::size_t>(_max_beams) *
                                     sizeof(float2);

    void* input_memory = device.get_gpu_memory_array(_gpu_mem_formed_beams, gpu_frame_id,
                                                     _gpu_buffer_depth, buffer_bytes);
    void* output_memory = (_gpu_mem_cleaned_beams == _gpu_mem_formed_beams)
                              ? input_memory
                              : device.get_gpu_memory_array(_gpu_mem_cleaned_beams, gpu_frame_id,
                                                             _gpu_buffer_depth, buffer_bytes);

    if (!input_memory || !output_memory) {
        return record_end_event();
    }

    // Forward metadata to output buffer if separate
    if (_gpu_mem_cleaned_beams != _gpu_mem_formed_beams) {
        std::shared_ptr<metadataObject> meta = device.get_gpu_memory_array_metadata(_gpu_mem_formed_beams, gpu_frame_id);
        if (meta) {
            device.claim_gpu_memory_array_metadata(_gpu_mem_cleaned_beams, gpu_frame_id, meta);
        }
    }

    record_start_event();
    cudaStream_t stream = device.getStream(cuda_stream_id);

    // Bypass mode: if disabled or only 1 active beam, pass formed beams straight through
    if (!current_config.enabled || current_config.num_active_beams <= 1) {
        if (output_memory != input_memory) {
            CHECK_CUDA_ERROR_NON_OO(cudaMemcpyAsync(
                output_memory, input_memory, buffer_bytes,
                cudaMemcpyDeviceToDevice, stream));
        }
        return record_end_event();
    }

    // Update unmixing matrices if targets or antenna masks changed
    update_matrices_if_needed(stream);

    // Launch CUDA Joint Matrix Unmixing kernel
    launch_bislc_unmixing(
        reinterpret_cast<float2*>(output_memory),
        reinterpret_cast<const float2*>(input_memory),
        _d_unmix_matrices,
        static_cast<std::size_t>(_samples_per_data_set),
        static_cast<std::size_t>(_num_local_freq),
        current_config.num_active_beams,
        static_cast<std::size_t>(_max_beams),
        stream);

    return record_end_event();
}

} // namespace kotekan
