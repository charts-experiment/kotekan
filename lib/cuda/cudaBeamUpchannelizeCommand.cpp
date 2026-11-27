#include "cudaBeamUpchannelizeCommand.hpp"

#include "gpuCommand.hpp"
#include "kotekanLogging.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <tuple>

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::cudaBeamUpchannelizeCommand;

REGISTER_CUDA_COMMAND(cudaBeamUpchannelizeCommand);

namespace kotekan {

cudaBeamUpchannelizeCommand::cudaBeamUpchannelizeCommand(Config& config,
                                                         const std::string& unique_name,
                                                         bufferContainer& host_buffers,
                                                         cudaDeviceInterface& device, int inst) :
    cudaCommand(config, unique_name, host_buffers, device, inst, no_cuda_command_state,
                "cudaBeamUpchannelizeCommand") {

    _cfg.n_freq = static_cast<std::size_t>(
        std::max(1, config.get_default<int>(unique_name, "num_local_freq", 336)));
    _cfg.n_time = static_cast<std::size_t>(
        std::max(1, config.get_default<int>(unique_name, "samples_per_data_set", 15360)));
    _cfg.max_beams_stride =
        static_cast<std::size_t>(std::max(1, config.get_default<int>(unique_name, "max_beams", 1)));
    _cfg.num_beams = static_cast<std::size_t>(std::max(
        1, config.get_default<int>(unique_name, "num_beams",
                                   static_cast<int>(_cfg.max_beams_stride))));
    _cfg.num_beams = std::min(_cfg.num_beams, _cfg.max_beams_stride);
    _cfg.n_pol =
        static_cast<std::size_t>(std::max(1, config.get_default<int>(unique_name, "num_pols", 1)));
    _cfg.upchan_factor = static_cast<std::size_t>(
        config.get_default<int>(unique_name, "upchan_factor", 16));
    if (_cfg.upchan_factor != 16) {
        FATAL_ERROR("cudaBeamUpchannelizeCommand: upchan_factor must be 16 (got {:d})",
                    _cfg.upchan_factor);
    }
    if (_cfg.n_time % _cfg.upchan_factor != 0) {
        FATAL_ERROR("cudaBeamUpchannelizeCommand: samples_per_data_set ({:d}) must be a multiple "
                    "of upchan_factor ({:d})",
                    _cfg.n_time, _cfg.upchan_factor);
    }

    const std::string win_str =
        config.get_default<std::string>(unique_name, "window_type", "hann");
    try {
        _cfg.window_type = parse_beam_upchan_window(win_str);
    } catch (const std::exception& e) {
        FATAL_ERROR("cudaBeamUpchannelizeCommand: {:s}", e.what());
    }

    _gpu_mem_formed_beams =
        config.get_default<std::string>(unique_name, "gpu_mem_formed_beams", "formed_beams");
    _gpu_mem_upchan_beams =
        config.get_default<std::string>(unique_name, "gpu_mem_upchan_beams", "upchan_beams");
    _gpu_mem_upchan_power =
        config.get_default<std::string>(unique_name, "gpu_mem_upchan_power", "");

    set_command_type(gpuCommandType::KERNEL);
    set_name("cudaBeamUpchannelizeCommand");

    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_formed_beams, true, false, true));
    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_upchan_beams, false, true, true));
    device.register_gpu_memory_name(_gpu_mem_upchan_beams);
    if (!_gpu_mem_upchan_power.empty()) {
        gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_upchan_power, false, true, true));
        device.register_gpu_memory_name(_gpu_mem_upchan_power);
    }
}

cudaEvent_t cudaBeamUpchannelizeCommand::execute(cudaPipelineState&,
                                                 const std::vector<cudaEvent_t>&) {
    pre_execute();

    const std::size_t in_bytes = _cfg.n_time * _cfg.n_freq * _cfg.max_beams_stride * _cfg.n_pol
                                 * sizeof(float2);
    const std::size_t out_bytes = in_bytes; // (n_time/16) * (n_freq*16) == n_time * n_freq

    void* formed_mem =
        device.get_gpu_memory_array(_gpu_mem_formed_beams, gpu_frame_id, _gpu_buffer_depth, in_bytes);
    void* upchan_mem = device.get_gpu_memory_array(_gpu_mem_upchan_beams, gpu_frame_id,
                                                   _gpu_buffer_depth, out_bytes);
    if (!formed_mem || !upchan_mem) {
        return record_end_event();
    }

    float* power_ptr = nullptr;
    if (!_gpu_mem_upchan_power.empty()) {
        const std::size_t power_bytes =
            _cfg.num_beams * _cfg.n_freq * _cfg.upchan_factor * sizeof(float);
        void* power_mem = device.get_gpu_memory_array(_gpu_mem_upchan_power, gpu_frame_id,
                                                      _gpu_buffer_depth, power_bytes);
        power_ptr = static_cast<float*>(power_mem);
    }

    std::shared_ptr<metadataObject> meta =
        device.get_gpu_memory_array_metadata(_gpu_mem_formed_beams, gpu_frame_id);
    if (meta) {
        device.claim_gpu_memory_array_metadata(_gpu_mem_upchan_beams, gpu_frame_id, meta);
        if (power_ptr) {
            device.claim_gpu_memory_array_metadata(_gpu_mem_upchan_power, gpu_frame_id, meta);
        }
    }

    record_start_event();
    cudaStream_t stream = device.getStream(cuda_stream_id);
    launch_beam_upchannelize_16pt(static_cast<const float2*>(formed_mem),
                                  static_cast<float2*>(upchan_mem), power_ptr, _cfg, stream);
    return record_end_event();
}

} // namespace kotekan
