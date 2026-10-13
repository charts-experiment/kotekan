#include "cudaCommand.hpp"
#include "chartsMetadata.hpp"
#include "chartsBeamMetadata.hpp"
#include "cudaUtils.hpp"
#include "beamformer/cuda_stage_api.hpp"
#include "beamformer/indexing.hpp"
#include "beamformer/io.hpp"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <optional>
#include <sstream>
#include <stdexcept>

using kotekan::Config;
using kotekan::bufferContainer;
using namespace beamformer;

namespace {
void require(bool ok, const char* message) {
    if (!ok) throw std::invalid_argument(message);
}
// Stable byte fingerprint for identity/corruption detection, not authentication.
std::string weight_fingerprint(const Weights& weights) {
    uint64_t hash = 14695981039346656037ULL;
    const auto* bytes = reinterpret_cast<const uint8_t*>(weights.data());
    for (size_t i = 0; i < weights.size() * sizeof(ComplexFloat); ++i)
        hash = (hash ^ bytes[i]) * 1099511628211ULL;
    std::ostringstream text;
    text << "fnv1a64:" << std::hex << std::setfill('0') << std::setw(16) << hash;
    return text.str();
}
}

class chartsBeamformerState : public cudaCommandState {
public:
    BeamformerInputContract input;
    TemporalIntegrationConfig integration;
    CudaBeamformerKernel kernel;
    nlohmann::json manifest;
    ComplexFloat* device_weights = nullptr; // owned by cudaDeviceInterface
    size_t output_bytes;
    std::optional<uint64_t> previous_end;

    chartsBeamformerState(Config& cfg, const std::string& name, bufferContainer& buffers,
                         cudaDeviceInterface& device) : cudaCommandState(cfg, name, buffers, device) {
        require(cfg.get_default<std::string>(name, "input_mode", "coarse") == "coarse",
                "Phase 2 supports coarse mode only");
        const uint16_t endian = 1;
        require(*reinterpret_cast<const uint8_t*>(&endian) == 1, "weights/output require little endian host");
        input = make_beamformer_input_contract(BeamformerInputMode::RawCoarse,
            cfg.get<size_t>(name, "samples_per_data_set"), cfg.get<size_t>(name, "shard_id"),
            cfg.get<size_t>(name, "num_elements"), cfg.get<size_t>(name, "num_beams"));
        input.frequency_grid.first_coarse_center_hz = cfg.get_default<double>(name, "frequency_zero_hz", 300e6);
        input.frequency_grid.coarse_channel_width_hz = cfg.get_default<double>(name, "frequency_step_hz", 300e3);
        const auto selector = cfg.get_default<std::string>(name, "beamformer_kernel", "tiled");
        require(selector == "tiled" || selector == "direct", "unknown beamformer kernel");
        kernel = selector == "tiled" ? CudaBeamformerKernel::Tiled : CudaBeamformerKernel::Direct;
        integration.integration_spectra = cfg.get_default<size_t>(name, "integration_spectra", 320);
        validate_cuda_beamformer_launch(input, kernel, integration);
        output_bytes = intensity_bytes(cuda_beamformer_output_dimensions(input, integration));
        std::ifstream manifest_file(cfg.get<std::string>(name, "weights_manifest"));
        require(bool(manifest_file), "cannot open weight manifest");
        manifest_file >> manifest;
        require(manifest.at("schema") == "charts.weights.v1", "unknown weight manifest");
        require(manifest.at("layout") == "BFE_complex_float32_le", "weights must use canonical BFE layout");
        require(manifest.at("shard_id") == input.raw_shard.shard_id, "weights belong to another shard");
        require(manifest.at("shape") == nlohmann::json::array({input.dimensions.n_beams, size_t(336), input.dimensions.n_ant}),
                "weight manifest shape mismatch");
        require(manifest.at("frequency_first_hz") == beamformer_frequency_hz(input, 0)
                && manifest.at("frequency_step_hz") == input.frequency_grid.coarse_channel_width_hz,
                "weight frequency grid mismatch");
        require(manifest.at("beam_order_id").is_string() && !manifest.at("beam_order_id").get<std::string>().empty(),
                "missing beam order identity");
        auto weights = read_weights(cfg.get<std::string>(name, "weights_file"), input);
        for (const auto value : weights)
            require(std::isfinite(value.real) && std::isfinite(value.imag), "nonfinite weight");
        require(manifest.at("digest") == weight_fingerprint(weights), "weight digest mismatch");
        if (kernel == CudaBeamformerKernel::Tiled) {
            TiledWeights tiled(tiled_weight_count(input.dimensions), {0, 0});
            for (size_t f = 0; f < 336; ++f)
                for (size_t b = 0; b < input.dimensions.n_beams; ++b)
                    for (size_t a = 0; a < input.dimensions.n_ant; ++a)
                        tiled[tiled_weight_index(f, b / 32, a, b % 32, input.dimensions)] =
                            weights[weight_index(b, f, a, input.dimensions)];
            weights = std::move(tiled);
        }
        device_weights = static_cast<ComplexFloat*>(device.get_gpu_memory(name + "/weights", weights.size() * sizeof(ComplexFloat)));
        // Setup-only upload. No allocation or weight preparation in execute().
        CHECK_CUDA_ERROR(cudaMemcpy(device_weights, weights.data(), weights.size() * sizeof(ComplexFloat), cudaMemcpyHostToDevice));
    }
};

class cudaChartsBeamformerCommand : public cudaCommand {
public:
    cudaChartsBeamformerCommand(Config& cfg, const std::string& name, bufferContainer& buffers,
                               cudaDeviceInterface& device, int inst,
                               const std::shared_ptr<cudaCommandState>& shared) :
        cudaCommand(cfg, name, buffers, device, inst, shared, "charts_beamformer"),
        state(std::static_pointer_cast<chartsBeamformerState>(shared)),
        voltage(buffers.get_buffer(cfg.get<std::string>(name, "in_buf"))),
        mask(buffers.get_buffer(cfg.get<std::string>(name, "mask_buf"))),
        output(buffers.get_buffer(cfg.get<std::string>(name, "out_buf"))),
        voltage_name(cfg.get<std::string>(name, "gpu_mem_voltage")),
        intensity_name(cfg.get<std::string>(name, "gpu_mem_intensity")) {
        require(bool(state) && voltage && mask && output, "missing beamformer state/buffers");
        require(_gpu_buffer_depth > 0, "invalid GPU buffer depth");
        require(voltage_name != intensity_name, "input/output GPU buffers must be distinct");
        require(voltage_name != name + "/weights" && intensity_name != name + "/weights",
                "frame buffers must not alias persistent weights");
        require(voltage->frame_size == packed_voltage_bytes(state->input.dimensions), "raw buffer capacity mismatch");
        require(mask->frame_size == state->input.raw_n_time, "mask buffer capacity mismatch");
        require(output->frame_size == state->output_bytes, "output buffer capacity mismatch");
        require(voltage->metadata_pool && voltage->metadata_pool->type_name == "chartsMetadata"
                && mask->metadata_pool && mask->metadata_pool->type_name == "chartsMetadata",
                "raw and original mask buffers require chartsMetadata");
        require(output->metadata_pool && output->metadata_pool->type_name == "chartsBeamMetadata",
                "output requires chartsBeamMetadata pool");
        require(_required_flag.empty(), "conditional execution is unsupported for paired input ownership");
        if (inst == 0) {
            voltage->register_consumer(name);
            mask->register_consumer(name);
        }
        d_voltage = static_cast<const uint8_t*>(device.get_gpu_memory_array(voltage_name, inst, _gpu_buffer_depth, voltage->frame_size));
        d_output = static_cast<float*>(device.get_gpu_memory_array(intensity_name, inst, _gpu_buffer_depth, state->output_bytes));
        register_gpu_buffer_user({voltage_name, true, true, false});
        register_gpu_buffer_user({intensity_name, true, false, true});
        set_command_type(gpuCommandType::KERNEL);
    }

    int wait_on_precondition() override {
        if (!voltage->wait_for_full_frame(unique_name, gpu_frame_id % voltage->num_frames)) return -1;
        auto* flags = mask->wait_for_full_frame(unique_name, gpu_frame_id % mask->num_frames);
        if (!flags) return -1;
        auto raw_meta = std::dynamic_pointer_cast<chartsMetadata>(voltage->get_metadata(gpu_frame_id % voltage->num_frames));
        auto mask_meta = std::dynamic_pointer_cast<chartsMetadata>(mask->get_metadata(gpu_frame_id % mask->num_frames));
        require(raw_meta && mask_meta, "missing voltage/mask identity metadata");
        require(raw_meta->has_fpga_seq_num() && mask_meta->has_fpga_seq_num()
                && raw_meta->get_fpga_seq_num() >= 0
                && raw_meta->get_fpga_seq_num() == mask_meta->get_fpga_seq_num(), "stale/mismatched voltage-mask sequence");
        require(raw_meta->has_time_downsampling_fpga() && mask_meta->has_time_downsampling_fpga()
                && raw_meta->get_time_downsampling_fpga() == 1 && mask_meta->get_time_downsampling_fpga() == 1,
                "raw inputs require sequence tick step 1");
        require(raw_meta->type == kotekan::int4x2, "raw input must use signed real-low int4x2");
        require(raw_meta->dims == -1 || raw_meta->dims == 3, "raw rank must be TFE when provided");
        const size_t extents[] = {state->input.raw_n_time, 336, state->input.dimensions.n_ant};
        const size_t strides[] = {336 * state->input.dimensions.n_ant, state->input.dimensions.n_ant, 1};
        for (size_t d = 0; d < 3; ++d) {
            require(raw_meta->dim[d] == -1 || (raw_meta->dim[d] >= 0 && size_t(raw_meta->dim[d]) == extents[d]),
                    "raw metadata extent mismatch");
            require(raw_meta->stride[d] == -1 || (raw_meta->stride[d] >= 0 && size_t(raw_meta->stride[d]) == strides[d]),
                    "raw metadata stride mismatch");
        }
        require(raw_meta->has_coarse_freq(), "missing raw frequency identity");
        auto frequencies = raw_meta->get_coarse_freq();
        require(frequencies.size() == 336, "raw frequency count mismatch");
        for (size_t f = 0; f < 336; ++f)
            require(frequencies[f] == int(state->input.raw_shard.absolute_frequency_start + f), "raw frequency shard mismatch");
        if (mask_meta->has_coarse_freq())
            require(mask_meta->get_coarse_freq() == frequencies, "mask frequency shard mismatch");
        pending = std::dynamic_pointer_cast<chartsBeamMetadata>(output->metadata_pool->request_metadata_object());
        require(bool(pending), "beam output metadata factory mismatch");
        auto input = state->input;
        input.raw_shard.timestamp_start = uint64_t(raw_meta->get_fpga_seq_num());
        input.raw_frame_id = uint64_t(gpu_frame_id);
        validate_beamformer_input_contract(input); // includes exclusive-end overflow
        const auto seq = input.raw_shard.timestamp_start;
        require(!state->previous_end || seq >= *state->previous_end, "duplicate/backwards/overlapping raw frame");
        const auto k = state->integration.integration_spectra;
        const auto dims = cuda_beamformer_output_dimensions(input, state->integration);
        std::vector<size_t> supplied(dims.n_time, 0), valid(dims.n_time, 0);
        for (size_t t = 0; t < input.raw_n_time; ++t) {
            require(flags[t] <= 1, "missing flags must be binary");
            ++supplied[t / k];
            valid[t / k] += flags[t] == 0;
        }
        pending->document = {
            {"schema", "charts.beam.intensity.v1"}, {"mode", "coarse"}, {"datatype", "float32_le"},
            {"layout", "TFB"}, {"shape", {dims.n_time, size_t(336), dims.n_beams}},
            {"strides", {336 * dims.n_beams, dims.n_beams, size_t(1)}}, {"payload_bytes", state->output_bytes},
            {"raw_n_time", input.raw_n_time}, {"n_elements", dims.n_ant}, {"integration_spectra", k},
            {"raw_frame_id", gpu_frame_id}, {"sequence_start", seq},
            {"sequence_gap_ticks", state->previous_end ? seq - *state->previous_end : 0},
            {"tick_period_us", {10, 3}}, {"timestamp_step_ticks", k},
            {"shard_id", input.raw_shard.shard_id}, {"coarse_origin", input.raw_shard.absolute_frequency_start},
            {"frequency_first_hz", beamformer_frequency_hz(input, 0)},
            {"frequency_step_hz", input.frequency_grid.coarse_channel_width_hz},
            {"weights_id", state->manifest.at("digest")}, {"weights_layout", state->manifest.at("layout")},
            {"beam_order_id", state->manifest.at("beam_order_id")},
            {"intensity_convention", "sum_power_all_supplied"}, {"history", "none"}, {"voltage_gain", 1},
            {"validity_layout", "T_integrated_broadcast_frequency_beam"},
            {"supplied_spectra", supplied}, {"valid_spectra", valid}};
        pending->validate();
        frame_input = input;
        return 0;
    }

    cudaEvent_t execute(cudaPipelineState&, const std::vector<cudaEvent_t>& pre_events) override {
        require(bool(pending), "beamformer execute requires validated paired inputs");
        require(device.get_gpu_memory_array_metadata(voltage_name, gpu_frame_id).get()
                == voltage->get_metadata(gpu_frame_id % voltage->num_frames).get(),
                "GPU voltage was not staged from the paired host frame");
        pre_execute();
        record_start_event();
        // Join supplied events explicitly, including H2D if no sync command precedes us.
        for (auto event : pre_events)
            if (event) CHECK_CUDA_ERROR(cudaStreamWaitEvent(device.getStream(cuda_stream_id), event, 0));
        launch_packed_integrated_beamformer(state->kernel, device.getStream(cuda_stream_id),
            d_voltage, state->device_weights, d_output, frame_input, state->integration);
        device.claim_gpu_memory_array_metadata(intensity_name, gpu_frame_id, pending);
        state->previous_end = pending->document.at("sequence_start").get<uint64_t>() + state->input.raw_n_time;
        return record_end_event();
    }
    void finalize_frame() override {
        cudaCommand::finalize_frame();
        voltage->mark_frame_empty(unique_name, gpu_frame_id % voltage->num_frames);
        mask->mark_frame_empty(unique_name, gpu_frame_id % mask->num_frames);
        pending.reset();
    }
private:
    std::shared_ptr<chartsBeamformerState> state;
    Buffer *voltage, *mask, *output;
    std::string voltage_name, intensity_name;
    const uint8_t* d_voltage;
    float* d_output;
    std::shared_ptr<chartsBeamMetadata> pending;
    BeamformerInputContract frame_input;
};

REGISTER_CUDA_COMMAND_WITH_STATE(cudaChartsBeamformerCommand, chartsBeamformerState);
