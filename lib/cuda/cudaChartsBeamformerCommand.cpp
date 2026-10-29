#include "cudaCommand.hpp"
#include "cudaChartsUpchannelizer.hpp"
#include "cudaChartsBeamformer.hpp"
#include "chartsMetadata.hpp"
#include "chartsPacketLoss.hpp"
#include "chartsWeightsNpy.hpp"
#include "cudaUtils.hpp"
#include "kotekanLogging.hpp"

#include <cmath>
#include <cstring>
#include <fstream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

using kotekan::bufferContainer;
using kotekan::Config;
using namespace beamformer;

namespace {
void require(bool ok, const char* message) {
    if (!ok) {
        throw std::invalid_argument(message);
    }
}
} // namespace

class chartsBeamformerState : public cudaCommandState {
public:
    BeamformerInputMode mode = BeamformerInputMode::ChannelizedFine;
    bool quantize_int8 = true;
    std::size_t shard_id = 0;
    std::size_t raw_n_time = 15360;
    std::size_t integration_spectra = 10;
    float pfb_quantize_clamp = 7.0F;
    Dimensions beamform_dims;
    Dimensions output_dims;
    std::size_t raw_voltage_bytes = 0;
    std::size_t fp32_intensity_bytes = 0;
    std::size_t output_bytes = 0;
    bool packet_loss_enabled = false;
    std::size_t exposure_bytes = 0;
    std::string weights_summary; // set for .npy weights, logged by the command
    bool legacy_bin_weights = false;

    ComplexFloat* device_weights = nullptr;
    std::uint8_t* device_history_samples = nullptr;
    std::uint8_t* device_fine_packed = nullptr;
    float* device_fp32_scratch = nullptr;
    Int8QuantizationParameters* device_quant_params = nullptr;
    std::uint8_t* device_raw_validity = nullptr;
    std::uint8_t* device_fine_validity = nullptr;
    std::uint8_t* device_history_validity = nullptr;
    // Independent of per-slot end_event, which the results thread destroys.
    cudaEvent_t workspace_ready = nullptr;
    bool workspace_submitted = false;
    bool history_initialized = false;
    std::optional<std::int64_t> expected_next_seq;

    chartsBeamformerState(Config& cfg, const std::string& name, bufferContainer& buffers,
                          cudaDeviceInterface& device) :
        cudaCommandState(cfg, name, buffers, device) {
        const auto mode_str = cfg.get_default<std::string>(name, "input_mode", "fine");
        require(mode_str == "coarse" || mode_str == "fine",
                "input_mode must be 'coarse' or 'fine'");
        mode = (mode_str == "fine") ? BeamformerInputMode::ChannelizedFine
                                    : BeamformerInputMode::RawCoarse;

        const auto fmt_str = cfg.get_default<std::string>(name, "output_format", "int8");
        require(fmt_str == "float32" || fmt_str == "int8",
                "output_format must be 'float32' or 'int8'");
        quantize_int8 = (fmt_str == "int8");

        shard_id = cfg.get_default<std::size_t>(name, "shard_id", 0);
        require(shard_id < frequency_shard_count, "shard_id must be 0 or 1");

        raw_n_time = cfg.get<std::size_t>(name, "samples_per_data_set");
        const std::size_t n_ant = cfg.get_default<std::size_t>(name, "num_elements", 64);
        const std::size_t n_beams = cfg.get_default<std::size_t>(name, "num_beams",
                                    cfg.get_default<std::size_t>(name, "max_beams", 64));
        require(n_ant == 32 || n_ant == 64, "num_elements must be 32 or 64");
        require(n_beams >= 1 && n_beams <= maximum_beams, "num_beams must be between 1 and 128");

        if (mode == BeamformerInputMode::ChannelizedFine) {
            require(raw_n_time >= pfb_history_samples && raw_n_time % upchannelization_factor == 0,
                    "fine mode requires samples_per_data_set >= 96 divisible by 32");
            integration_spectra = cfg.get_default<std::size_t>(name, "integration_spectra", 10);
            require(integration_spectra == 10, "fine mode requires integration_spectra=10");
            beamform_dims = {raw_n_time / upchannelization_factor, fine_frequency_channels, n_ant,
                             n_beams};
        } else {
            require(raw_n_time > 0, "samples_per_data_set must be positive");
            integration_spectra = cfg.get_default<std::size_t>(name, "integration_spectra", 320);
            require(integration_spectra == 10 || integration_spectra == 320,
                    "coarse mode supports integration_spectra of 10 or 320");
            beamform_dims = {raw_n_time, local_frequency_channels, n_ant, n_beams};
        }

        pfb_quantize_clamp = cfg.get_default<float>(name, "quantize_clamp", 7.0F);

        output_dims = {integrated_time_count(beamform_dims.n_time, integration_spectra),
                       beamform_dims.n_freq, n_ant, n_beams};
        raw_voltage_bytes = raw_n_time * local_frequency_channels * n_ant;
        const std::size_t output_samples =
            output_dims.n_time * output_dims.n_freq * output_dims.n_beams;
        fp32_intensity_bytes = output_samples * sizeof(float);
        output_bytes = quantize_int8 ? (output_samples * sizeof(std::int8_t)) : fp32_intensity_bytes;

        packet_loss_enabled = !cfg.get_default<std::string>(name, "gpu_mem_missing", "").empty();
        if (packet_loss_enabled) {
            require(n_ant == 64, "handler masks require 64 CHARTS elements");
            const bool fine = mode == BeamformerInputMode::ChannelizedFine;
            exposure_bytes = output_dims.n_time * local_frequency_channels
                             * (fine ? sizeof(std::uint8_t) : sizeof(std::uint16_t));
            device_raw_validity = static_cast<std::uint8_t*>(
                device.get_gpu_memory(name + "/raw_validity", raw_n_time));
            if (fine) {
                device_fine_validity = static_cast<std::uint8_t*>(
                    device.get_gpu_memory(name + "/fine_validity", beamform_dims.n_time));
                device_history_validity = static_cast<std::uint8_t*>(
                    device.get_gpu_memory(name + "/history_validity", pfb_history_samples));
                CHECK_CUDA_ERROR(cudaMemset(device_history_validity, 0, pfb_history_samples));
            }
        }
        CHECK_CUDA_ERROR(cudaEventCreateWithFlags(&workspace_ready, cudaEventDisableTiming));

        const double freq_zero_hz = cfg.get_default<double>(name, "frequency_zero_hz", 300e6);
        const double freq_step_hz = cfg.get_default<double>(name, "frequency_step_hz", 300e3);
        const float spacing_m = cfg.get_default<float>(name, "spacing_m", 0.6F);
        const float beam_l_step = cfg.get_default<float>(name, "beam_l_step", 0.01F);

        Weights host_weights;
        const std::string weights_file = cfg.get_default<std::string>(name, "weights_file", "");
        if (!weights_file.empty()) {
            const std::size_t expected_count =
                beamform_dims.n_beams * beamform_dims.n_freq * beamform_dims.n_ant;
            const bool is_npy = weights_file.size() > 4
                                && weights_file.compare(weights_file.size() - 4, 4, ".npy") == 0;
            if (is_npy) {
                // Combined gain-correction and steering coefficients from
                // charts/scripts/build_beamformer_weights.py. A descriptor is required.
                const NpyWeightsExpectation expect{
                    mode == BeamformerInputMode::ChannelizedFine ? "fine" : "coarse", shard_id,
                    beamform_dims.n_beams, beamform_dims.n_freq, beamform_dims.n_ant};
                const std::string descriptor_file = cfg.get_default<std::string>(
                    name, "weights_descriptor",
                    weights_file.substr(0, weights_file.size() - 4) + ".json");
                nlohmann::json descriptor;
                try {
                    std::ifstream din(descriptor_file);
                    require(bool(din), "cannot open weights_descriptor");
                    din >> descriptor;
                    const NpyWeightsInfo info = validate_weights_descriptor(descriptor, expect);
                    check_descriptor_frequency(
                        descriptor,
                        shard_frequencies_hz(mode, shard_id, freq_zero_hz, freq_step_hz),
                        freq_zero_hz, freq_step_hz);
                    const auto bytes = read_whole_file(weights_file);
                    host_weights = parse_npy_complex64(bytes, expect);
                    check_npy_sha256(bytes, info.sha256);
                    weights_summary = fmt::format(
                        "{} shard {} {}: calibrated={} flagged_channel_antennas={} sha256={} "
                        "note='{}'",
                        weights_file, shard_id, expect.mode, info.calibrated,
                        info.flagged_channel_antennas, info.sha256, info.calibration_note);
                } catch (const nlohmann::json::exception& e) {
                    throw std::invalid_argument(std::string("weights_descriptor: ") + e.what());
                }
            } else {
                // Legacy raw complex64 payload: the size must match exactly (a .npy renamed
                // to .bin or a file for another mode/shard is rejected).
                const auto bytes = read_whole_file(weights_file);
                require(bytes.size() == expected_count * sizeof(ComplexFloat),
                        "weights_file byte size mismatch");
                host_weights.resize(expected_count);
                std::memcpy(host_weights.data(), bytes.data(), bytes.size());
                legacy_bin_weights = true;
            }
            host_weights = repack_weights_to_tiled(host_weights, beamform_dims);
        } else {
            const auto positions = default_positions(n_ant, spacing_m);
            const auto directions = default_beam_grid(n_beams, beam_l_step, 0.0F);
            const auto freqs = shard_frequencies_hz(mode, shard_id, freq_zero_hz, freq_step_hz);
            host_weights = generate_tiled_weights(beamform_dims, positions, freqs, directions);
        }

        const std::size_t weight_bytes = host_weights.size() * sizeof(ComplexFloat);
        device_weights =
            static_cast<ComplexFloat*>(device.get_gpu_memory(name + "/weights", weight_bytes));
        (void)cudaGetLastError();
        CHECK_CUDA_ERROR(cudaMemcpy(device_weights, host_weights.data(), weight_bytes,
                                    cudaMemcpyHostToDevice));

        if (mode == BeamformerInputMode::ChannelizedFine) {
            const std::size_t history_bytes =
                local_frequency_channels * n_ant * pfb_history_samples;
            device_history_samples = static_cast<std::uint8_t*>(
                device.get_gpu_memory(name + "/pfb_history", history_bytes));
            CHECK_CUDA_ERROR(cudaMemset(device_history_samples, 0, history_bytes));

            const std::size_t fine_packed_bytes =
                beamform_dims.n_time * beamform_dims.n_freq * beamform_dims.n_ant;
            device_fine_packed = static_cast<std::uint8_t*>(
                device.get_gpu_memory(name + "/fine_packed", fine_packed_bytes));
        }

        if (quantize_int8) {
            device_fp32_scratch = static_cast<float*>(
                device.get_gpu_memory(name + "/fp32_scratch", fp32_intensity_bytes));
            const std::size_t param_bytes =
                quantization_parameter_count(output_dims) * sizeof(Int8QuantizationParameters);
            device_quant_params = static_cast<Int8QuantizationParameters*>(
                device.get_gpu_memory(name + "/quant_params", param_bytes));
        }
    }

    ~chartsBeamformerState() {
        if (workspace_ready)
            cudaEventDestroy(workspace_ready);
    }
};

class cudaChartsBeamformerCommand : public cudaCommand {
public:
    cudaChartsBeamformerCommand(Config& cfg, const std::string& name, bufferContainer& buffers,
                                cudaDeviceInterface& device, int inst,
                                const std::shared_ptr<cudaCommandState>& shared) :
        cudaCommand(cfg, name, buffers, device, inst, shared, "charts_beamformer"),
        state(std::static_pointer_cast<chartsBeamformerState>(shared)),
        voltage_name(cfg.get_default<std::string>(name, "gpu_mem_voltage", "voltage")),
        intensity_name(cfg.get_default<std::string>(name, "gpu_mem_intensity", "intensity")) {
        require(bool(state), "missing chartsBeamformerState");
        if (inst == 0 && !state->weights_summary.empty())
            INFO("CHARTS weights loaded from .npy: {:s}", state->weights_summary);
        if (inst == 0 && state->legacy_bin_weights)
            WARN("CHARTS weights loaded from a raw .bin: no descriptor, version or SHA-256 "
                 "check; prefer the .npy produced by build_beamformer_weights.py");
        require(voltage_name != intensity_name,
                "gpu_mem_voltage and gpu_mem_intensity must be distinct");

        missing_name = cfg.get_default<std::string>(name, "gpu_mem_missing", "");
        exposure_name = cfg.get_default<std::string>(name, "gpu_mem_exposure", "");
        if (!state->packet_loss_enabled) {
            require(exposure_name.empty()
                        && cfg.get_default<std::string>(name, "in_buf_missing", "").empty()
                        && cfg.get_default<std::string>(name, "out_buf_exposure", "").empty(),
                    "mask/exposure options require gpu_mem_missing; no all-valid fallback");
        }

        const std::string in_buf_name = cfg.get_default<std::string>(name, "in_buf", "");
        if (!in_buf_name.empty()) {
            voltage_buf = buffers.get_buffer(in_buf_name);
            require(voltage_buf && voltage_buf->frame_size == state->raw_voltage_bytes,
                    "input host buffer frame_size mismatch");
        }
        const std::string out_buf_name = cfg.get_default<std::string>(name, "out_buf", "");
        if (!out_buf_name.empty()) {
            output_buf = buffers.get_buffer(out_buf_name);
            require(output_buf && output_buf->frame_size == state->output_bytes,
                    "output host buffer frame_size mismatch");
        }

        if (state->packet_loss_enabled) {
            require(voltage_buf, "packet loss requires in_buf for frame association");
            mask_shard_id = cfg.get<std::size_t>(name, "mask_shard_id");
            require(mask_shard_id == state->shard_id, "mask_shard_id must match shard_id");
            missing_buf = buffers.get_buffer(cfg.get<std::string>(name, "in_buf_missing"));
            auto* exposure_buf = buffers.get_buffer(cfg.get<std::string>(name, "out_buf_exposure"));
            require(missing_buf != voltage_buf && missing_buf->frame_size == state->raw_n_time,
                    "mask buffer must be distinct and contain T_raw bytes");
            require(exposure_buf->frame_size == state->exposure_bytes,
                    "exposure host buffer frame_size mismatch");
            require(!exposure_name.empty() && missing_name != voltage_name
                        && missing_name != intensity_name && exposure_name != voltage_name
                        && exposure_name != intensity_name && exposure_name != missing_name,
                    "voltage/mask/intensity/exposure GPU names must be distinct");
            // Retain host frames for pre-H2D validation using normal consumer ownership.
            // cudaInputData independently consumes the SAME mask buffer; there is no fan-out.
            if (inst == 0) {
                voltage_buf->register_consumer(name);
                missing_buf->register_consumer(name);
            }
            d_missing = static_cast<const std::uint8_t*>(device.get_gpu_memory_array(
                missing_name, inst, _gpu_buffer_depth, state->raw_n_time));
            d_exposure = device.get_gpu_memory_array(exposure_name, inst, _gpu_buffer_depth,
                                                     state->exposure_bytes);
            register_gpu_buffer_user({missing_name, true, true, false});
            register_gpu_buffer_user({exposure_name, true, false, true});
        }

        d_voltage = static_cast<const std::uint8_t*>(device.get_gpu_memory_array(
            voltage_name, inst, _gpu_buffer_depth, state->raw_voltage_bytes));
        d_output = device.get_gpu_memory_array(intensity_name, inst, _gpu_buffer_depth,
                                               state->output_bytes);

        register_gpu_buffer_user({voltage_name, true, true, false});
        register_gpu_buffer_user({intensity_name, true, false, true});
        set_command_type(gpuCommandType::KERNEL);
        require(cuda_stream_id >= 2, "CHARTS compute requires stream >= 2 for cudaSyncOutput");
    }

    int wait_on_precondition() override {
        if (!state->packet_loss_enabled)
            return 0;
        const int raw_slot = gpu_frame_id % voltage_buf->num_frames;
        const int mask_slot = gpu_frame_id % missing_buf->num_frames;
        if (!voltage_buf->wait_for_full_frame(unique_name, raw_slot))
            return -1;
        const auto* flags = missing_buf->wait_for_full_frame(unique_name, mask_slot);
        if (!flags)
            return -1;
        const auto raw_meta = std::dynamic_pointer_cast<chartsMetadata>(
            voltage_buf->get_metadata(raw_slot));
        const auto mask_meta = std::dynamic_pointer_cast<chartsMetadata>(
            missing_buf->get_metadata(mask_slot));
        try {
            associated_seq = validate_charts_mask_pair(
                raw_meta.get(), mask_meta.get(), flags, missing_buf->frame_size,
                state->raw_n_time, state->shard_id, mask_shard_id);
        } catch (const std::invalid_argument& error) {
            FATAL_ERROR("CHARTS packet-loss input rejected: {:s}", error.what());
            return -1;
        }
        return 0;
    }

    void finalize_frame() override {
        cudaCommand::finalize_frame();
        if (state->packet_loss_enabled) {
            voltage_buf->mark_frame_empty(unique_name, gpu_frame_id % voltage_buf->num_frames);
            missing_buf->mark_frame_empty(unique_name, gpu_frame_id % missing_buf->num_frames);
        }
    }

    cudaEvent_t execute(cudaPipelineState&, const std::vector<cudaEvent_t>& pre_events) override {
        pre_execute();
        record_start_event();

        cudaStream_t stream = device.getStream(cuda_stream_id);
        for (auto event : pre_events) {
            if (event) {
                CHECK_CUDA_ERROR(cudaStreamWaitEvent(stream, event, 0));
            }
        }
        // Serialize persistent history and scratch across rotating slots, even if
        // a supported configuration selects a different compute stream.
        if (state->workspace_submitted)
            CHECK_CUDA_ERROR(cudaStreamWaitEvent(stream, state->workspace_ready, 0));

        // Check for sequence discontinuity on this shard if chartsMetadata is attached
        auto meta = std::dynamic_pointer_cast<chartsMetadata>(
            device.get_gpu_memory_array_metadata(voltage_name, gpu_frame_id));
        if (state->packet_loss_enabled) {
            const auto mask_meta = std::dynamic_pointer_cast<chartsMetadata>(
                device.get_gpu_memory_array_metadata(missing_name, gpu_frame_id));
            if (!meta || !mask_meta || !meta->has_fpga_seq_num() || !mask_meta->has_fpga_seq_num()
                || meta->get_fpga_seq_num() != associated_seq
                || mask_meta->get_fpga_seq_num() != associated_seq)
                FATAL_ERROR("CHARTS H2D voltage/mask metadata does not match validated host frame");
        }
        if (meta && meta->has_fpga_seq_num()) {
            const std::int64_t seq = meta->get_fpga_seq_num();
            if (state->mode == BeamformerInputMode::ChannelizedFine && state->expected_next_seq
                && seq != *state->expected_next_seq) {
                const std::size_t history_bytes =
                    local_frequency_channels * state->beamform_dims.n_ant * pfb_history_samples;
                CHECK_CUDA_ERROR(
                    cudaMemsetAsync(state->device_history_samples, 0, history_bytes, stream));
                if (state->packet_loss_enabled)
                    CHECK_CUDA_ERROR(cudaMemsetAsync(state->device_history_validity, 0,
                                                     pfb_history_samples, stream));
                state->history_initialized = false;
            }
            state->expected_next_seq = seq + static_cast<std::int64_t>(state->raw_n_time);
        }

        if (state->packet_loss_enabled) {
            // n_coarse=1 explicitly inverts missing flags into compact [T_raw] validity.
            launch_expand_missing_to_validity(stream, d_missing, state->device_raw_validity,
                                               state->raw_n_time, 1);
            const bool fine = state->mode == BeamformerInputMode::ChannelizedFine;
            const std::uint8_t* validity = state->device_raw_validity;
            if (fine) {
                launch_pfb_validity(stream, validity, state->device_history_validity,
                                    state->history_initialized, state->device_fine_validity,
                                    state->raw_n_time);
                validity = state->device_fine_validity;
            }
            launch_integrated_exposure(stream, validity, d_exposure, state->beamform_dims.n_time,
                                        state->integration_spectra, fine);
            if (fine)
                CHECK_CUDA_ERROR(cudaMemcpyAsync(
                    state->device_history_validity,
                    state->device_raw_validity + state->raw_n_time - pfb_history_samples,
                    pfb_history_samples, cudaMemcpyDeviceToDevice, stream));
        }

        const std::uint8_t* beamform_input = d_voltage;
        if (state->mode == BeamformerInputMode::ChannelizedFine) {
            launch_fused_upchannelizer(stream, d_voltage, state->device_history_samples,
                                       state->history_initialized, state->device_fine_packed,
                                       state->raw_n_time, local_frequency_channels,
                                       state->beamform_dims.n_ant, state->pfb_quantize_clamp);
            launch_pfb_history_update(stream, d_voltage, state->device_history_samples,
                                      state->raw_n_time, local_frequency_channels,
                                      state->beamform_dims.n_ant);
            state->history_initialized = true;
            beamform_input = state->device_fine_packed;
        }

        float* fp32_target =
            state->quantize_int8 ? state->device_fp32_scratch : static_cast<float*>(d_output);
        launch_tiled_packed_integrated_beamformer(stream, beamform_input, state->device_weights,
                                                  fp32_target, state->beamform_dims,
                                                  state->integration_spectra);

        if (state->quantize_int8) {
            launch_quantize_integrated_intensity(
                stream, fp32_target, static_cast<std::int8_t*>(d_output),
                state->device_quant_params, state->output_dims);
        }

        if (meta) {
            device.claim_gpu_memory_array_metadata(intensity_name, gpu_frame_id, meta);
            if (state->packet_loss_enabled)
                device.claim_gpu_memory_array_metadata(exposure_name, gpu_frame_id, meta);
        }
        CHECK_CUDA_ERROR(cudaEventRecord(state->workspace_ready, stream));
        state->workspace_submitted = true;
        return record_end_event();
    }

private:
    std::shared_ptr<chartsBeamformerState> state;
    std::string voltage_name;
    std::string intensity_name;
    std::string missing_name;
    std::string exposure_name;
    std::size_t mask_shard_id = 0;
    std::int64_t associated_seq = -1;
    Buffer* voltage_buf = nullptr;
    Buffer* missing_buf = nullptr;
    Buffer* output_buf = nullptr;
    const std::uint8_t* d_voltage = nullptr;
    void* d_output = nullptr;
    const std::uint8_t* d_missing = nullptr;
    void* d_exposure = nullptr;
};

REGISTER_CUDA_COMMAND_WITH_STATE(cudaChartsBeamformerCommand, chartsBeamformerState);
