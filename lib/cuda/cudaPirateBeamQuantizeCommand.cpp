#include "cudaPirateBeamQuantizeCommand.hpp"

#include "chartsConstants.hpp"
#include "cudaUtils.hpp"
#include "gpuCommand.hpp"
#include "kotekanLogging.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <tuple>

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::cudaPirateBeamQuantizeCommand;

REGISTER_CUDA_COMMAND(cudaPirateBeamQuantizeCommand);

namespace kotekan {

cudaPirateBeamQuantizeCommand::cudaPirateBeamQuantizeCommand(Config& config,
                                                             const std::string& unique_name,
                                                             bufferContainer& host_buffers,
                                                             cudaDeviceInterface& device,
                                                             int inst) :
    cudaCommand(config, unique_name, host_buffers, device, inst, no_cuda_command_state,
                "cudaPirateBeamQuantizeCommand") {

    _cfg.num_freq = static_cast<std::size_t>(
        std::max(1, config.get_default<int>(unique_name, "num_local_freq", 336)));
    _cfg.samples_per_data_set = static_cast<std::size_t>(
        std::max(1, config.get_default<int>(unique_name, "samples_per_data_set", 15360)));
    _cfg.max_beams =
        static_cast<std::size_t>(std::max(1, config.get_default<int>(unique_name, "max_beams", 1)));
    _cfg.num_beams = static_cast<std::size_t>(std::max(
        1, config.get_default<int>(unique_name, "num_beams", static_cast<int>(_cfg.max_beams))));
    _cfg.num_beams = std::min(_cfg.num_beams, _cfg.max_beams);
    _cfg.num_pols =
        static_cast<std::size_t>(std::max(1, config.get_default<int>(unique_name, "num_pols", 1)));
    _cfg.seq_per_frb_time_sample = static_cast<std::size_t>(
        std::max(1, config.get_default<int>(unique_name, "seq_per_frb_time_sample", 60)));

    if (_cfg.samples_per_data_set % (PIRATE_SAMPLES_PER_MINICHUNK * _cfg.seq_per_frb_time_sample)
        != 0) {
        FATAL_ERROR("cudaPirateBeamQuantizeCommand: samples_per_data_set ({:d}) must be a "
                    "multiple of 256 * seq_per_frb_time_sample ({:d})",
                    _cfg.samples_per_data_set,
                    PIRATE_SAMPLES_PER_MINICHUNK * _cfg.seq_per_frb_time_sample);
    }

    _cfg.rfi_chunk_size = static_cast<std::size_t>(
        std::max(1, config.get_default<int>(unique_name, "rfi_chunk_size", 256)));
    _cfg.rfi_mask_fraction_threshold =
        config.get_default<float>(unique_name, "rfi_mask_fraction_threshold", 0.5f);
    _cfg.enable_incoherent_subtraction =
        config.get_default<bool>(unique_name, "enable_incoherent_subtraction", true);
    _cfg.enable_highpass_detrend =
        config.get_default<bool>(unique_name, "enable_highpass_detrend", true);

    if (config.exists(unique_name, "highpass_window_samples")) {
        _cfg.highpass_window_samples = config.get<int>(unique_name, "highpass_window_samples");
    } else {
        const double hp_ms = config.get_default<double>(unique_name, "highpass_window_ms", 31.0);
        const double frb_dt_ms =
            static_cast<double>(_cfg.seq_per_frb_time_sample)
            * charts::constants::fpga_time_resolution_s * 1.0e3;
        _cfg.highpass_window_samples =
            std::max(3, static_cast<int>(std::llround(hp_ms / std::max(frb_dt_ms, 1.0e-6))));
    }

    _cfg.use_robust_mad_scale =
        config.get_default<bool>(unique_name, "use_robust_mad_scale", true);
    _cfg.stddev_cutoff = config.get_default<float>(unique_name, "stddev_cutoff", 3.0f);

    _gpu_mem_formed_beams =
        config.get_default<std::string>(unique_name, "gpu_mem_formed_beams", "formed_beams");
    _gpu_mem_rfi_mask = config.get_default<std::string>(unique_name, "gpu_mem_rfi_mask", "");
    _gpu_mem_pirate = config.get_default<std::string>(unique_name, "gpu_mem_pirate", "pirate_out");

    set_command_type(gpuCommandType::KERNEL);
    set_name("cudaPirateBeamQuantizeCommand");

    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_formed_beams, true, false, true));
    if (!_gpu_mem_rfi_mask.empty()) {
        gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_rfi_mask, true, false, true));
    }
    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_pirate, false, true, true));
    device.register_gpu_memory_name(_gpu_mem_pirate);

    const std::size_t n_frb = _cfg.samples_per_data_set / _cfg.seq_per_frb_time_sample;
    const std::size_t int_elems = _cfg.num_beams * _cfg.num_freq * n_frb;
    const std::size_t valid_elems = _cfg.num_freq * n_frb;
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&_d_scratch_intensity, int_elems * sizeof(float)));
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&_d_scratch_valid, valid_elems * sizeof(std::uint8_t)));
}

cudaPirateBeamQuantizeCommand::~cudaPirateBeamQuantizeCommand() {
    if (_d_scratch_intensity) {
        cudaFree(_d_scratch_intensity);
        _d_scratch_intensity = nullptr;
    }
    if (_d_scratch_valid) {
        cudaFree(_d_scratch_valid);
        _d_scratch_valid = nullptr;
    }
}

cudaEvent_t cudaPirateBeamQuantizeCommand::execute(cudaPipelineState&,
                                                   const std::vector<cudaEvent_t>&) {
    pre_execute();

    const std::size_t in_bytes = _cfg.samples_per_data_set * _cfg.num_freq * _cfg.max_beams
                                 * _cfg.num_pols * sizeof(float2);
    const std::size_t n_frb = _cfg.samples_per_data_set / _cfg.seq_per_frb_time_sample;
    const std::size_t n_minichunks = n_frb / PIRATE_SAMPLES_PER_MINICHUNK;
    const std::size_t out_bytes =
        n_minichunks * _cfg.num_beams * _cfg.num_freq * PIRATE_BYTES_PER_CHAN;

    void* formed_mem =
        device.get_gpu_memory_array(_gpu_mem_formed_beams, gpu_frame_id, _gpu_buffer_depth, in_bytes);
    void* pirate_mem =
        device.get_gpu_memory_array(_gpu_mem_pirate, gpu_frame_id, _gpu_buffer_depth, out_bytes);
    if (!formed_mem || !pirate_mem) {
        return record_end_event();
    }

    const std::uint8_t* rfi_mask_ptr = nullptr;
    if (!_gpu_mem_rfi_mask.empty()) {
        const std::size_t n_chunks =
            (_cfg.samples_per_data_set + _cfg.rfi_chunk_size - 1) / _cfg.rfi_chunk_size;
        const std::size_t mask_bytes = n_chunks * _cfg.num_freq * sizeof(std::uint8_t);
        void* mask_mem = device.get_gpu_memory_array(_gpu_mem_rfi_mask, gpu_frame_id,
                                                     _gpu_buffer_depth, mask_bytes);
        rfi_mask_ptr = static_cast<const std::uint8_t*>(mask_mem);
    }

    std::shared_ptr<metadataObject> meta =
        device.get_gpu_memory_array_metadata(_gpu_mem_formed_beams, gpu_frame_id);
    if (meta) {
        device.claim_gpu_memory_array_metadata(_gpu_mem_pirate, gpu_frame_id, meta);
    }

    record_start_event();
    cudaStream_t stream = device.getStream(cuda_stream_id);
    launch_pirate_beam_quantize(static_cast<const float2*>(formed_mem), rfi_mask_ptr,
                                _d_scratch_intensity, _d_scratch_valid,
                                static_cast<std::uint8_t*>(pirate_mem), _cfg, stream);
    return record_end_event();
}

} // namespace kotekan
