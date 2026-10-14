#include "cudaBiSLCCommand.hpp"

#include "cudaBiSLC.hpp"
#include "cudaDirectBeamTrackerCommand.hpp"
#include "cudaUtils.hpp"
#include "gpuCommand.hpp"
#include "kotekanLogging.hpp"
#include "restServer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <tuple>
#include <vector>

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::connectionInstance;
using kotekan::cudaBiSLCCommand;
using kotekan::HTTP_RESPONSE;
using kotekan::restServer;

REGISTER_CUDA_COMMAND(cudaBiSLCCommand);

namespace kotekan {

std::mutex cudaBiSLCCommand::_global_mutex;
BiSLCConfig cudaBiSLCCommand::_shared_config;
bool cudaBiSLCCommand::_endpoints_registered = false;

void cudaBiSLCCommand::set_shared_config(const BiSLCConfig& config) {
    std::lock_guard<std::mutex> lock(_global_mutex);
    _shared_config = config;
    _shared_config.config_generation++;
}

BiSLCConfig cudaBiSLCCommand::get_shared_config() {
    std::lock_guard<std::mutex> lock(_global_mutex);
    return _shared_config;
}

cudaBiSLCCommand::cudaBiSLCCommand(Config& config, const std::string& unique_name,
                                   bufferContainer& host_buffers, cudaDeviceInterface& device,
                                   int inst) :
    cudaCommand(config, unique_name, host_buffers, device, inst, no_cuda_command_state,
                "cudaBiSLCCommand") {

    _num_pols = config.get_default<int>(unique_name, "num_pols", 1);
    if (config.exists(unique_name, "num_antennas")) {
        _num_antennas = config.get<int>(unique_name, "num_antennas");
        _num_elements = config.get_default<int>(unique_name, "num_elements",
                                                _num_antennas * _num_pols);
    } else {
        _num_elements = config.get<int>(unique_name, "num_elements");
        _num_antennas = _num_elements / std::max(1, _num_pols);
    }

    _num_local_freq = config.get<int>(unique_name, "num_local_freq");
    _samples_per_data_set = config.get<int>(unique_name, "samples_per_data_set");
    _buffer_depth = config.get<int>(unique_name, "buffer_depth");

    _max_beams = std::max(1, config.get_default<int>(unique_name, "max_beams", 2));
    _spacing_m = config.get_default<float>(unique_name, "spacing_m",
                                           charts::constants::charts_default_spacing_m);
    _diagonal_loading = config.get_default<float>(unique_name, "diagonal_loading", 1.0e-4f);
    _enabled = config.get_default<bool>(unique_name, "enabled", true);

    _gpu_mem_formed_beams =
        config.get_default<std::string>(unique_name, "gpu_mem_formed_beams", "formed_beams");
    _gpu_mem_cleaned_beams =
        config.get_default<std::string>(unique_name, "gpu_mem_cleaned_beams", "cleaned_beams");

    const double freq_start_hz = config.get_default<double>(unique_name, "freq_start_hz", 300.0e6);
    const double freq_step_hz = config.get_default<double>(unique_name, "freq_step_hz", 300.0e3);

    _frequencies_hz.resize(static_cast<std::size_t>(_num_local_freq));
    for (int f = 0; f < _num_local_freq; ++f) {
        _frequencies_hz[f] = freq_start_hz + f * freq_step_hz;
    }

    {
        std::lock_guard<std::mutex> lock(_global_mutex);
        _shared_config.enabled = _enabled;
        _shared_config.num_antennas = static_cast<std::size_t>(_num_antennas);
        _shared_config.num_pols = static_cast<std::size_t>(_num_pols);
        _shared_config.diagonal_loading = _diagonal_loading;
        _shared_config.max_beams_stride = static_cast<std::size_t>(_max_beams);
        _shared_config.num_active_beams = static_cast<std::size_t>(
            std::max(1, config.get_default<int>(unique_name, "initial_active_beams", 2)));
        if (_shared_config.targets.size() < static_cast<std::size_t>(_max_beams)) {
            _shared_config.targets.resize(static_cast<std::size_t>(_max_beams));
        }
        _shared_config.config_generation++;

        if (!_endpoints_registered) {
            auto& rest = restServer::instance();

            // 1. POST /bislc/enable
            auto enable_cb = [](connectionInstance& conn, nlohmann::json& j) {
                try {
                    bool en = j.value("enabled", true);
                    std::lock_guard<std::mutex> lk(_global_mutex);
                    _shared_config.enabled = en;
                    _shared_config.config_generation++;
                    INFO_NON_OO("BiSLC: Unmixing stage {:s}", en ? "ENABLED" : "DISABLED");
                    conn.send_text_reply(
                        fmt::format("BiSLC unmixing {:s}\n", en ? "enabled" : "disabled"));
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
                    _shared_config.config_generation++;
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
                reply["version"] = "cudaBiSLCCommand v2.0 (Blackwell GPU-Inverted Unmixing)";
                reply["enabled"] = _shared_config.enabled;
                reply["diagonal_loading"] = _shared_config.diagonal_loading;
                reply["num_active_beams"] = _shared_config.num_active_beams;
                reply["max_beams"] = _shared_config.max_beams_stride;
                reply["num_antennas"] = _shared_config.num_antennas;
                reply["num_pols"] = _shared_config.num_pols;
                reply["num_local_freq"] = _num_local_freq;
                reply["samples_per_data_set"] = _samples_per_data_set;

                double avg_sidelobe_mag = 0.0;
                std::size_t off_diag_count = 0;
                const std::size_t B =
                    std::min(_shared_config.num_active_beams, _shared_config.max_beams_stride);

                if (B > 1 && _d_coupling_matrices != nullptr) {
                    const std::size_t copy_elems =
                        static_cast<std::size_t>(_num_local_freq) * B * B;
                    if (_h_coupling_matrices.size() >= copy_elems) {
                        cudaMemcpy(_h_coupling_matrices.data(), _d_coupling_matrices,
                                   copy_elems * sizeof(float2), cudaMemcpyDeviceToHost);
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
                }
                reply["average_coupling_magnitude"] =
                    (off_diag_count > 0) ? (avg_sidelobe_mag / off_diag_count) : 0.0;

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
    const std::size_t max_b =
        std::min(static_cast<std::size_t>(_max_beams), MAX_BISLC_BEAMS);
    const std::size_t matrix_elems =
        static_cast<std::size_t>(_num_local_freq) * max_b * max_b;
    const std::size_t matrix_bytes = matrix_elems * sizeof(float2);

    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&_d_unmix_matrices, matrix_bytes));
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&_d_coupling_matrices, matrix_bytes));
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&_d_directions, max_b * sizeof(DirectDirection3D)));
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&_d_wavenumbers, _num_local_freq * sizeof(double)));
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&_d_antenna_positions, _num_elements * sizeof(float3)));
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&_d_antenna_mask, _num_elements * sizeof(std::uint8_t)));

    // Allocate pinned host memory for safe, asynchronous DMA transfers
    CHECK_CUDA_ERROR_NON_OO(cudaMallocHost(&_h_directions, max_b * sizeof(DirectDirection3D)));
    CHECK_CUDA_ERROR_NON_OO(cudaMallocHost(&_h_antenna_positions, _num_elements * sizeof(float3)));
    CHECK_CUDA_ERROR_NON_OO(cudaMallocHost(&_h_antenna_mask, _num_elements * sizeof(std::uint8_t)));

    _h_coupling_matrices.resize(matrix_elems);

    std::vector<double> h_wavenumbers(_num_local_freq);
    for (int f = 0; f < _num_local_freq; ++f) {
        h_wavenumbers[f] = charts::constants::two_pi * _frequencies_hz[f]
                           / charts::constants::speed_of_light_m_per_s;
    }
    CHECK_CUDA_ERROR_NON_OO(cudaMemcpy(_d_wavenumbers, h_wavenumbers.data(),
                                       _num_local_freq * sizeof(double), cudaMemcpyHostToDevice));
}

void cudaBiSLCCommand::free_device_buffers() {
    if (_d_unmix_matrices) {
        cudaFree(_d_unmix_matrices);
        _d_unmix_matrices = nullptr;
    }
    if (_d_coupling_matrices) {
        cudaFree(_d_coupling_matrices);
        _d_coupling_matrices = nullptr;
    }
    if (_d_directions) {
        cudaFree(_d_directions);
        _d_directions = nullptr;
    }
    if (_d_wavenumbers) {
        cudaFree(_d_wavenumbers);
        _d_wavenumbers = nullptr;
    }
    if (_d_antenna_positions) {
        cudaFree(_d_antenna_positions);
        _d_antenna_positions = nullptr;
    }
    if (_d_antenna_mask) {
        cudaFree(_d_antenna_mask);
        _d_antenna_mask = nullptr;
    }
    if (_h_directions) {
        cudaFreeHost(_h_directions);
        _h_directions = nullptr;
    }
    if (_h_antenna_positions) {
        cudaFreeHost(_h_antenna_positions);
        _h_antenna_positions = nullptr;
    }
    if (_h_antenna_mask) {
        cudaFreeHost(_h_antenna_mask);
        _h_antenna_mask = nullptr;
    }
}

void cudaBiSLCCommand::update_matrices_if_needed(cudaStream_t stream) {
    // Poll Direct Beam Tracker state first so beam count, targets, and masks are always fresh
    const DirectBeamTrackerConfig tracker_cfg =
        cudaDirectBeamTrackerCommand::get_shared_config();

    BiSLCConfig current_cfg;
    bool needs_update = false;

    {
        std::lock_guard<std::mutex> lock(_global_mutex);
        if (_shared_config.tracker_generation_synced != tracker_cfg.config_generation
            || _shared_config.num_active_beams != tracker_cfg.num_active_beams) {

            _shared_config.num_active_beams = tracker_cfg.num_active_beams;
            _shared_config.antenna_mask = tracker_cfg.antenna_mask;
            _shared_config.antenna_positions = tracker_cfg.antenna_positions;
            _shared_config.num_active_antennas = tracker_cfg.num_active_antennas;

            if (_shared_config.targets.size() < tracker_cfg.targets.size()) {
                _shared_config.targets.resize(tracker_cfg.targets.size());
            }
            for (std::size_t b = 0; b < tracker_cfg.targets.size(); ++b) {
                _shared_config.targets[b] = tracker_cfg.targets[b];
            }

            _shared_config.tracker_generation_synced = tracker_cfg.config_generation;
            _shared_config.config_generation++;
        }

        if (_applied_bislc_generation != _shared_config.config_generation
            || _applied_tracker_generation != tracker_cfg.config_generation) {
            _applied_bislc_generation = _shared_config.config_generation;
            _applied_tracker_generation = tracker_cfg.config_generation;
            current_cfg = _shared_config;
            needs_update = true;
        }
    }

    if (!needs_update) return;

    const std::size_t B = std::min({current_cfg.num_active_beams,
                                    static_cast<std::size_t>(_max_beams),
                                    MAX_BISLC_BEAMS});
    if (B <= 1) return;

    for (std::size_t b = 0; b < B; ++b) {
        _h_directions[b] = current_cfg.targets[b].direction;
    }
    for (std::size_t a = 0; a < static_cast<std::size_t>(_num_elements); ++a) {
        _h_antenna_positions[a] = current_cfg.antenna_positions[a];
        _h_antenna_mask[a] = current_cfg.antenna_mask[a];
    }

    CHECK_CUDA_ERROR_NON_OO(cudaMemcpyAsync(_d_directions, _h_directions,
                                            B * sizeof(DirectDirection3D),
                                            cudaMemcpyHostToDevice, stream));
    CHECK_CUDA_ERROR_NON_OO(cudaMemcpyAsync(_d_antenna_positions, _h_antenna_positions,
                                            _num_elements * sizeof(float3),
                                            cudaMemcpyHostToDevice, stream));
    CHECK_CUDA_ERROR_NON_OO(cudaMemcpyAsync(_d_antenna_mask, _h_antenna_mask,
                                            _num_elements * sizeof(std::uint8_t),
                                            cudaMemcpyHostToDevice, stream));

    // Compute A(f) and M(f) = (A(f) + eps*I)^-1 directly on the GPU in-stream
    launch_compute_bislc_matrices_gpu(
        _d_unmix_matrices, _d_coupling_matrices, _d_directions, _d_wavenumbers,
        _d_antenna_positions, _d_antenna_mask, B,
        static_cast<std::size_t>(_num_local_freq),
        static_cast<std::size_t>(_num_elements),
        current_cfg.diagonal_loading, stream);
}

cudaEvent_t cudaBiSLCCommand::execute(cudaPipelineState& /*pipestate*/,
                                      const std::vector<cudaEvent_t>&) {
    pre_execute();

    const std::size_t buffer_bytes =
        static_cast<std::size_t>(_num_local_freq)
        * static_cast<std::size_t>(_samples_per_data_set)
        * static_cast<std::size_t>(_max_beams)
        * static_cast<std::size_t>(_num_pols) * sizeof(float2);

    void* input_memory = device.get_gpu_memory_array(_gpu_mem_formed_beams, gpu_frame_id,
                                                     _gpu_buffer_depth, buffer_bytes);
    void* output_memory = (_gpu_mem_cleaned_beams == _gpu_mem_formed_beams)
                              ? input_memory
                              : device.get_gpu_memory_array(_gpu_mem_cleaned_beams, gpu_frame_id,
                                                            _gpu_buffer_depth, buffer_bytes);

    if (!input_memory || !output_memory) {
        return record_end_event();
    }

    if (_gpu_mem_cleaned_beams != _gpu_mem_formed_beams) {
        std::shared_ptr<metadataObject> meta =
            device.get_gpu_memory_array_metadata(_gpu_mem_formed_beams, gpu_frame_id);
        if (meta) {
            device.claim_gpu_memory_array_metadata(_gpu_mem_cleaned_beams, gpu_frame_id, meta);
        }
    }

    record_start_event();
    cudaStream_t stream = device.getStream(cuda_stream_id);

    // Always synchronize state with DirectBeamTracker BEFORE checking for 1-beam bypass!
    update_matrices_if_needed(stream);

    const BiSLCConfig current_config = get_shared_config();
    const std::size_t active_beams = std::min({current_config.num_active_beams,
                                               static_cast<std::size_t>(_max_beams),
                                               MAX_BISLC_BEAMS});

    if (!current_config.enabled || active_beams <= 1) {
        if (output_memory != input_memory) {
            CHECK_CUDA_ERROR_NON_OO(cudaMemcpyAsync(output_memory, input_memory, buffer_bytes,
                                                    cudaMemcpyDeviceToDevice, stream));
        }
        return record_end_event();
    }

    launch_bislc_unmixing(
        reinterpret_cast<float2*>(output_memory),
        reinterpret_cast<const float2*>(input_memory),
        _d_unmix_matrices,
        static_cast<std::size_t>(_samples_per_data_set),
        static_cast<std::size_t>(_num_local_freq),
        active_beams,
        static_cast<std::size_t>(_max_beams),
        stream,
        static_cast<std::size_t>(_num_pols));

    return record_end_event();
}

} // namespace kotekan
