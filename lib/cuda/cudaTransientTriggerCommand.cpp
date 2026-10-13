#include "cudaTransientTriggerCommand.hpp"

#include "cudaTransientTrigger.hpp"
#include "cudaUtils.hpp"
#include "gpuCommand.hpp"
#include "kotekanLogging.hpp"
#include "restServer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::connectionInstance;
using kotekan::cudaTransientTriggerCommand;
using kotekan::cudaTransientTriggerState;
using kotekan::HTTP_RESPONSE;
using kotekan::restServer;

REGISTER_CUDA_COMMAND_WITH_STATE(cudaTransientTriggerCommand, cudaTransientTriggerState);

namespace kotekan {

std::mutex cudaTransientTriggerCommand::_global_mutex;
TransientTriggerConfig cudaTransientTriggerCommand::_shared_config;
bool cudaTransientTriggerCommand::_endpoints_registered = false;
std::atomic<uint32_t> cudaTransientTriggerCommand::_manual_trigger_count{0};
uint32_t cudaTransientTriggerCommand::_total_triggers_fired = 0;
std::atomic<uint32_t> cudaTransientTriggerCommand::_auto_dumps_written{0};
TransientFrameMetrics cudaTransientTriggerCommand::_last_trigger_metrics;
std::string cudaTransientTriggerCommand::_last_dump_path = "none";

void cudaTransientTriggerCommand::set_shared_config(const TransientTriggerConfig& config) {
    std::lock_guard<std::mutex> lock(_global_mutex);
    _shared_config = config;
}

TransientTriggerConfig cudaTransientTriggerCommand::get_shared_config() {
    std::lock_guard<std::mutex> lock(_global_mutex);
    return _shared_config;
}

void cudaTransientTriggerCommand::trigger_manual() {
    _manual_trigger_count++;
}

// -----------------------------------------------------------------------------
// cudaTransientTriggerState Implementation (Shared per cudaProcess Stage)
// -----------------------------------------------------------------------------

cudaTransientTriggerState::cudaTransientTriggerState(Config& config, const std::string& unique_name,
                                                     bufferContainer& host_buffers,
                                                     cudaDeviceInterface& device) :
    cudaCommandState(config, unique_name, host_buffers, device), device(device),
    unique_name(unique_name) {

    num_local_freq = config.get<int>(unique_name, "num_local_freq");
    samples_per_data_set = config.get<int>(unique_name, "samples_per_data_set");
    max_beams = config.get_default<int>(unique_name, "max_beams", 2);

    ring_buffer_depth =
        static_cast<uint32_t>(config.get_default<int>(unique_name, "ring_buffer_depth", 16));
    pre_trigger_frames =
        static_cast<uint32_t>(config.get_default<int>(unique_name, "pre_trigger_frames", 4));
    post_trigger_frames =
        static_cast<uint32_t>(config.get_default<int>(unique_name, "post_trigger_frames", 4));
    dump_directory =
        config.get_default<std::string>(unique_name, "dump_directory", "./transient_dumps");

    frame_elements = static_cast<std::size_t>(samples_per_data_set)
                     * static_cast<std::size_t>(num_local_freq)
                     * static_cast<std::size_t>(max_beams);
    frame_bytes = frame_elements * sizeof(float2);

    allocate_device_buffers();
}

cudaTransientTriggerState::~cudaTransientTriggerState() {
    free_device_buffers();
}

void cudaTransientTriggerState::allocate_device_buffers() {
    device.set_thread_device();

    // 1. Allocate GPU VRAM Circular Ring Buffer (Shared across all pipeline instances of this
    // stage)
    const std::size_t total_ring_bytes = static_cast<std::size_t>(ring_buffer_depth) * frame_bytes;
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(reinterpret_cast<void**>(&d_ring_buffer), total_ring_bytes));

    // 2. Allocate GPU Metrics Output Buffer
    const std::size_t metrics_bytes = static_cast<std::size_t>(num_local_freq) * sizeof(float2);
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(reinterpret_cast<void**>(&d_metrics), metrics_bytes));

    // 3. Allocate Pinned Host Metrics Buffer
    CHECK_CUDA_ERROR_NON_OO(cudaHostAlloc(reinterpret_cast<void**>(&h_metrics_pinned),
                                          metrics_bytes, cudaHostAllocDefault));

    // 4. Allocate Pinned Host Dump Staging Buffer
    const std::size_t max_dump_frames =
        static_cast<std::size_t>(pre_trigger_frames + 1 + post_trigger_frames);
    const std::size_t total_dump_bytes = max_dump_frames * frame_bytes;
    CHECK_CUDA_ERROR_NON_OO(cudaHostAlloc(reinterpret_cast<void**>(&h_dump_pinned),
                                          total_dump_bytes, cudaHostAllocDefault));

    // 5. Create Dedicated Non-Blocking CUDA Dump Stream
    CHECK_CUDA_ERROR_NON_OO(cudaStreamCreateWithFlags(&dump_stream, cudaStreamNonBlocking));

    INFO_NON_OO("cudaTransientTriggerState: Allocated Shared VRAM Ring Buffer ({:d} frames, {:.2f} "
                "MB, Pinned Dump: {:.2f} MB) [{:s}]",
                ring_buffer_depth, static_cast<double>(total_ring_bytes) / (1024.0 * 1024.0),
                static_cast<double>(total_dump_bytes) / (1024.0 * 1024.0), unique_name.c_str());
}

void cudaTransientTriggerState::free_device_buffers() {
    device.set_thread_device();

    if (dump_stream) {
        cudaStreamSynchronize(dump_stream);
        cudaStreamDestroy(dump_stream);
        dump_stream = nullptr;
    }
    if (d_ring_buffer) {
        cudaFree(d_ring_buffer);
        d_ring_buffer = nullptr;
    }
    if (d_metrics) {
        cudaFree(d_metrics);
        d_metrics = nullptr;
    }
    if (h_metrics_pinned) {
        cudaFreeHost(h_metrics_pinned);
        h_metrics_pinned = nullptr;
    }
    if (h_dump_pinned) {
        cudaFreeHost(h_dump_pinned);
        h_dump_pinned = nullptr;
    }
}

void cudaTransientTriggerState::dispatch_candidate_dump(uint32_t trigger_frame,
                                                        const TransientFrameMetrics& metrics) {

    TransientTriggerConfig current_config = cudaTransientTriggerCommand::get_shared_config();
    const uint32_t pre = current_config.pre_trigger_frames;
    const uint32_t post = current_config.post_trigger_frames;

    const uint32_t start_frame = (trigger_frame > pre) ? (trigger_frame - pre) : 0;
    const uint32_t end_frame = trigger_frame + post;
    const uint32_t num_frames_to_dump = end_frame - start_frame + 1;

    DEBUG(
        "TransientTrigger: Dispatching dump for candidate window frames [{:d}..{:d}] ({:d} frames)",
        start_frame, end_frame, num_frames_to_dump);

    // Asynchronously copy candidate frames from GPU ring buffer to pinned host staging buffer
    for (uint32_t f_idx = 0; f_idx < num_frames_to_dump; ++f_idx) {
        const uint32_t target_frame = start_frame + f_idx;
        const uint32_t slot = target_frame % ring_buffer_depth;

        const float2* d_src = d_ring_buffer + slot * frame_elements;
        float2* h_dst = h_dump_pinned + f_idx * frame_elements;

        CHECK_CUDA_ERROR_NON_OO(
            cudaMemcpyAsync(h_dst, d_src, frame_bytes, cudaMemcpyDeviceToHost, dump_stream));
    }

    std::string dump_dir =
        !this->dump_directory.empty() ? this->dump_directory : current_config.dump_directory;
    std::size_t total_bytes = static_cast<std::size_t>(num_frames_to_dump) * frame_bytes;
    cudaStream_t stream_to_sync = dump_stream;
    const float2* host_data = h_dump_pinned;
    std::string stage_tag = unique_name;
    std::replace(stage_tag.begin(), stage_tag.end(), '/', '_');

    std::thread([dump_dir, trigger_frame, start_frame, end_frame, num_frames_to_dump, total_bytes,
                 stream_to_sync, host_data, metrics, stage_tag]() {
        try {
            // Wait for GPU -> Host D2H copy to complete on dump stream
            CHECK_CUDA_ERROR_NON_OO(cudaStreamSynchronize(stream_to_sync));

            std::filesystem::create_directories(dump_dir);

            // Layer 2 Guard Rail: Hard disk space safety circuit breaker
            std::error_code ec;
            auto space_info = std::filesystem::space(dump_dir, ec);
            if (!ec) {
                uint64_t free_gb = space_info.available / (1024ULL * 1024ULL * 1024ULL);
                uint32_t min_free =
                    cudaTransientTriggerCommand::get_shared_config().min_free_disk_gb;
                if (free_gb < min_free) {
                    ERROR_NON_OO("TransientTrigger: DISK SAFETY TRIP! Free disk space ({:d} GB) < "
                                 "min_free_disk_gb ({:d} GB). Dump aborted to protect filesystem!",
                                 free_gb, min_free);
                    return;
                }
            }

            auto now = std::chrono::system_clock::now();
            auto now_ms =
                std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch())
                    .count();

            std::string base_path = fmt::format("{:s}/candidate_{:s}_frame_{:08d}_{:d}", dump_dir,
                                                stage_tag, trigger_frame, now_ms);
            std::string raw_path = base_path + ".raw";
            std::string json_path = base_path + ".json";

            // 1. Write baseband raw voltages
            std::ofstream raw_file(raw_path, std::ios::binary);
            if (raw_file.is_open()) {
                raw_file.write(reinterpret_cast<const char*>(host_data), total_bytes);
                raw_file.close();
            }

            // 2. Write metadata JSON
            nlohmann::json meta_json;
            meta_json["trigger_frame"] = trigger_frame;
            meta_json["start_frame"] = start_frame;
            meta_json["end_frame"] = end_frame;
            meta_json["num_dump_frames"] = num_frames_to_dump;
            meta_json["total_bytes"] = total_bytes;
            meta_json["stage"] = stage_tag;
            meta_json["timestamp_epoch_ms"] = now_ms;
            meta_json["flagged_channels"] = metrics.flagged_channels;
            meta_json["mean_sk"] = metrics.mean_sk;
            meta_json["median_sk"] = metrics.median_sk;
            meta_json["min_sk"] = metrics.min_sk;
            meta_json["max_sk"] = metrics.max_sk;
            meta_json["mean_r01"] = metrics.mean_r01;

            std::ofstream json_file(json_path);
            if (json_file.is_open()) {
                json_file << meta_json.dump(4);
                json_file.close();
            }

            {
                std::lock_guard<std::mutex> lk(cudaTransientTriggerCommand::_global_mutex);
                cudaTransientTriggerCommand::_last_dump_path = raw_path;
            }

            INFO_NON_OO("TransientTrigger: Successfully saved candidate dump to {:s} ({:.2f} MB)",
                        raw_path, static_cast<double>(total_bytes) / (1024.0 * 1024.0));
        } catch (const std::exception& e) {
            ERROR_NON_OO("TransientTrigger: Failed to write candidate dump: {:s}", e.what());
        }
    }).detach();
}

// -----------------------------------------------------------------------------
// cudaTransientTriggerCommand Implementation
// -----------------------------------------------------------------------------

cudaTransientTriggerCommand::cudaTransientTriggerCommand(Config& config,
                                                         const std::string& unique_name,
                                                         bufferContainer& host_buffers,
                                                         cudaDeviceInterface& device, int inst,
                                                         std::shared_ptr<cudaCommandState> state) :
    cudaCommand(config, unique_name, host_buffers, device, inst, state,
                "cudaTransientTriggerCommand") {

    _num_local_freq = config.get<int>(unique_name, "num_local_freq");
    _samples_per_data_set = config.get<int>(unique_name, "samples_per_data_set");
    _buffer_depth = config.get<int>(unique_name, "buffer_depth");
    _max_beams = config.get_default<int>(unique_name, "max_beams", 2);

    _gpu_mem_cleaned_beams =
        config.get_default<std::string>(unique_name, "gpu_mem_cleaned_beams", "cleaned_beams");
    _gpu_mem_output =
        config.get_default<std::string>(unique_name, "gpu_mem_output", _gpu_mem_cleaned_beams);

    // Register GPU buffers used in Kotekan framework
    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_cleaned_beams, true, false, false));
    if (_gpu_mem_output != _gpu_mem_cleaned_beams) {
        gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_output, false, true, false));
    }

    {
        std::lock_guard<std::mutex> lock(_global_mutex);
        _shared_config.enabled = config.get_default<bool>(unique_name, "enabled", true);
        _shared_config.sk_threshold = config.get_default<float>(unique_name, "sk_threshold", 0.08f);
        _shared_config.rfi_threshold =
            config.get_default<float>(unique_name, "rfi_threshold", 0.30f);
        _shared_config.min_flagged_channels =
            static_cast<uint32_t>(config.get_default<int>(unique_name, "min_flagged_channels", 8));
        _shared_config.ring_buffer_depth =
            static_cast<uint32_t>(config.get_default<int>(unique_name, "ring_buffer_depth", 16));
        _shared_config.pre_trigger_frames =
            static_cast<uint32_t>(config.get_default<int>(unique_name, "pre_trigger_frames", 4));
        _shared_config.post_trigger_frames =
            static_cast<uint32_t>(config.get_default<int>(unique_name, "post_trigger_frames", 4));
        _shared_config.dump_directory =
            config.get_default<std::string>(unique_name, "dump_directory", "./transient_dumps");
        _shared_config.auto_dump_enabled =
            config.get_default<bool>(unique_name, "auto_dump_enabled", true);
        _shared_config.cooldown_frames =
            static_cast<uint32_t>(config.get_default<int>(unique_name, "cooldown_frames", 200));
        _shared_config.min_free_disk_gb =
            static_cast<uint32_t>(config.get_default<int>(unique_name, "min_free_disk_gb", 20));
        _shared_config.max_auto_dumps =
            static_cast<uint32_t>(config.get_default<int>(unique_name, "max_auto_dumps", 20));

        if (!_endpoints_registered) {
            auto& rest = restServer::instance();

            // 1. GET /transient_trigger/status
            auto status_cb = [](connectionInstance& conn) {
                nlohmann::json reply;
                std::lock_guard<std::mutex> lk(_global_mutex);
                reply["version"] =
                    "cudaTransientTriggerCommand v1.0 (SK & Inter-Beam Cross-Correlation)";
                reply["enabled"] = _shared_config.enabled;
                reply["sk_threshold"] = _shared_config.sk_threshold;
                reply["rfi_threshold"] = _shared_config.rfi_threshold;
                reply["min_flagged_channels"] = _shared_config.min_flagged_channels;
                reply["ring_buffer_depth"] = _shared_config.ring_buffer_depth;
                reply["pre_trigger_frames"] = _shared_config.pre_trigger_frames;
                reply["post_trigger_frames"] = _shared_config.post_trigger_frames;
                reply["dump_directory"] = _shared_config.dump_directory;
                reply["auto_dump_enabled"] = _shared_config.auto_dump_enabled;
                reply["cooldown_frames"] = _shared_config.cooldown_frames;
                reply["min_free_disk_gb"] = _shared_config.min_free_disk_gb;
                reply["max_auto_dumps"] = _shared_config.max_auto_dumps;
                reply["auto_dumps_written"] = _auto_dumps_written.load();
                reply["total_triggers_fired"] = _total_triggers_fired;
                reply["last_dump_path"] = _last_dump_path;
                reply["last_trigger_frame"] = _last_trigger_metrics.frame_id;
                reply["last_trigger_flagged_channels"] = _last_trigger_metrics.flagged_channels;
                reply["last_trigger_mean_sk"] = _last_trigger_metrics.mean_sk;
                reply["last_trigger_median_sk"] = _last_trigger_metrics.median_sk;
                reply["last_trigger_min_sk"] = _last_trigger_metrics.min_sk;
                reply["last_trigger_max_sk"] = _last_trigger_metrics.max_sk;
                reply["last_trigger_mean_r01"] = _last_trigger_metrics.mean_r01;

                conn.send_json_reply(reply);
            };
            rest.register_get_callback("/transient_trigger/status", status_cb);

            // 2. POST /transient_trigger/set_thresholds
            auto set_thresholds_cb = [](connectionInstance& conn, nlohmann::json& j) {
                try {
                    std::lock_guard<std::mutex> lk(_global_mutex);
                    if (j.contains("sk_threshold")) {
                        _shared_config.sk_threshold = j["sk_threshold"];
                    }
                    if (j.contains("rfi_threshold")) {
                        _shared_config.rfi_threshold = j["rfi_threshold"];
                    }
                    if (j.contains("min_flagged_channels")) {
                        _shared_config.min_flagged_channels = j["min_flagged_channels"];
                    }
                    if (j.contains("cooldown_frames")) {
                        _shared_config.cooldown_frames = j["cooldown_frames"];
                    }
                    if (j.contains("min_free_disk_gb")) {
                        _shared_config.min_free_disk_gb = j["min_free_disk_gb"];
                    }
                    if (j.contains("max_auto_dumps")) {
                        _shared_config.max_auto_dumps = j["max_auto_dumps"];
                    }
                    if (j.contains("auto_dump_enabled")) {
                        _shared_config.auto_dump_enabled = j["auto_dump_enabled"];
                    }
                    if (j.contains("reset_quota") && j["reset_quota"].get<bool>()) {
                        _auto_dumps_written.store(0);
                        INFO_NON_OO("TransientTrigger: Auto-dump quota reset to 0");
                    }
                    INFO_NON_OO("TransientTrigger: Thresholds updated: SK={:.3f}, RFI={:.3f}, "
                                "MinCh={:d}, Cooldown={:d}, MinFreeDisk={:d}GB, MaxDumps={:d}",
                                _shared_config.sk_threshold, _shared_config.rfi_threshold,
                                _shared_config.min_flagged_channels, _shared_config.cooldown_frames,
                                _shared_config.min_free_disk_gb, _shared_config.max_auto_dumps);
                    conn.send_text_reply("Transient trigger thresholds updated successfully\n");
                } catch (const std::exception& e) {
                    conn.send_error(e.what(), HTTP_RESPONSE::BAD_REQUEST);
                }
            };
            rest.register_post_callback("/transient_trigger/set_thresholds", set_thresholds_cb);

            // 3. POST & GET /transient_trigger/manual_trigger
            auto manual_trig_cb = [](connectionInstance& conn, nlohmann::json&) {
                _manual_trigger_count++;
                INFO_NON_OO("TransientTrigger: Manual trigger requested via REST (count={:d})",
                            _manual_trigger_count.load());
                conn.send_text_reply("Manual candidate dump trigger queued\n");
            };
            rest.register_post_callback("/transient_trigger/manual_trigger", manual_trig_cb);
            auto manual_trig_get_cb = [](connectionInstance& conn) {
                _manual_trigger_count++;
                INFO_NON_OO("TransientTrigger: Manual trigger requested via REST GET (count={:d})",
                            _manual_trigger_count.load());
                conn.send_text_reply("Manual candidate dump trigger queued\n");
            };
            rest.register_get_callback("/transient_trigger/manual_trigger", manual_trig_get_cb);

            // 4. POST /transient_trigger/enable
            auto enable_cb = [](connectionInstance& conn, nlohmann::json& j) {
                try {
                    bool en = j.value("enabled", true);
                    std::lock_guard<std::mutex> lk(_global_mutex);
                    _shared_config.enabled = en;
                    INFO_NON_OO("TransientTrigger: Monitoring {:s}", en ? "ENABLED" : "DISABLED");
                    conn.send_text_reply(fmt::format("Transient trigger monitoring {:s}\n",
                                                     en ? "enabled" : "disabled"));
                } catch (const std::exception& e) {
                    conn.send_error(e.what(), HTTP_RESPONSE::BAD_REQUEST);
                }
            };
            rest.register_post_callback("/transient_trigger/enable", enable_cb);

            _endpoints_registered = true;
        }
    }
}

cudaTransientTriggerCommand::~cudaTransientTriggerCommand() {}

cudaTransientTriggerState* cudaTransientTriggerCommand::get_state() {
    return static_cast<cudaTransientTriggerState*>(command_state.get());
}

cudaEvent_t cudaTransientTriggerCommand::execute(cudaPipelineState& /*pipestate*/,
                                                 const std::vector<cudaEvent_t>&) {

    pre_execute();

    cudaTransientTriggerState* state = get_state();
    if (!state) {
        return record_end_event();
    }

    void* input_memory = device.get_gpu_memory_array(_gpu_mem_cleaned_beams, gpu_frame_id,
                                                     _gpu_buffer_depth, state->frame_bytes);

    if (!input_memory) {
        return record_end_event();
    }

    // Forward metadata to output buffer if different
    if (_gpu_mem_output != _gpu_mem_cleaned_beams) {
        std::shared_ptr<metadataObject> meta =
            device.get_gpu_memory_array_metadata(_gpu_mem_cleaned_beams, gpu_frame_id);
        if (meta) {
            device.claim_gpu_memory_array_metadata(_gpu_mem_output, gpu_frame_id, meta);
        }
    }

    record_start_event();
    cudaStream_t stream = device.getStream(cuda_stream_id);

    // Forward cleaned beams to output memory if required
    if (_gpu_mem_output != _gpu_mem_cleaned_beams) {
        void* output_memory = device.get_gpu_memory_array(_gpu_mem_output, gpu_frame_id,
                                                          _gpu_buffer_depth, state->frame_bytes);
        if (output_memory) {
            CHECK_CUDA_ERROR_NON_OO(cudaMemcpyAsync(output_memory, input_memory, state->frame_bytes,
                                                    cudaMemcpyDeviceToDevice, stream));
        }
    }

    std::lock_guard<std::mutex> lock(state->state_mutex);

    // Ingest current frame into VRAM circular ring buffer
    const uint32_t current_slot = state->frames_processed % state->ring_buffer_depth;
    float2* ring_dest = state->d_ring_buffer + current_slot * state->frame_elements;
    CHECK_CUDA_ERROR_NON_OO(cudaMemcpyAsync(ring_dest, input_memory, state->frame_bytes,
                                            cudaMemcpyDeviceToDevice, stream));

    TransientTriggerConfig current_config = get_shared_config();

    if (current_config.enabled) {
        // 1. Launch Spectral Kurtosis and Inter-Beam Cross-Correlation Reduction Kernel
        launch_transient_detection_metrics(reinterpret_cast<const float2*>(input_memory),
                                           state->d_metrics,
                                           static_cast<std::size_t>(state->samples_per_data_set),
                                           static_cast<std::size_t>(state->num_local_freq),
                                           static_cast<std::size_t>(state->max_beams),
                                           static_cast<std::size_t>(state->max_beams), stream);

        // 2. Asynchronously copy metrics (2.68 KB) to pinned host buffer
        const std::size_t metrics_bytes =
            static_cast<std::size_t>(state->num_local_freq) * sizeof(float2);
        CHECK_CUDA_ERROR_NON_OO(cudaMemcpyAsync(state->h_metrics_pinned, state->d_metrics,
                                                metrics_bytes, cudaMemcpyDeviceToHost, stream));

        // Stream synchronization for lightweight metric evaluation
        CHECK_CUDA_ERROR_NON_OO(cudaStreamSynchronize(stream));

        // 3. Evaluate combined decision logic on host
        auto now_ns = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                                std::chrono::system_clock::now().time_since_epoch())
                                                .count());

        TransientFrameMetrics frame_metrics = evaluate_transient_decision(
            state->h_metrics_pinned, static_cast<std::size_t>(state->num_local_freq),
            current_config, state->frames_processed, now_ns);

        // Check manual trigger request
        bool is_manual_trigger = false;
        uint32_t current_manual_count = _manual_trigger_count.load();
        if (state->last_manual_trigger_id != current_manual_count) {
            state->last_manual_trigger_id = current_manual_count;
            frame_metrics.trigger_fired = true;
            is_manual_trigger = true;
            INFO_NON_OO("TransientTrigger: Manual trigger executed on frame {:d} [{:s}]",
                        state->frames_processed, state->unique_name.c_str());
        }

        // Layer 1 Guard Rail: Cooldown / Refractory holdoff for automated triggers
        if (!is_manual_trigger && state->frames_processed < state->cooldown_until_frame) {
            frame_metrics.trigger_fired = false;
        }

        // 4. Decision handling
        if (frame_metrics.trigger_fired) {
            {
                std::lock_guard<std::mutex> lk(_global_mutex);
                _total_triggers_fired++;
                _last_trigger_metrics = frame_metrics;
            }

            INFO_NON_OO("TRANSIENT DETECTED! Frame {:d} [{:s}]: Flagged Channels={:d}/{:d}, Mean "
                        "SK={:.3f}, Mean R01={:.3f}",
                        state->frames_processed, state->unique_name.c_str(),
                        frame_metrics.flagged_channels, state->num_local_freq,
                        frame_metrics.mean_sk, frame_metrics.mean_r01);

            // Layer 3 Guard Rail: Session auto-dump quota ceiling
            if (!is_manual_trigger && _auto_dumps_written.load() >= current_config.max_auto_dumps) {
                WARN_NON_OO("TransientTrigger: Auto-dump quota reached ({:d}/{:d}) on [{:s}]. Dump "
                            "skipped! Reset quota or increase max_auto_dumps via REST.",
                            _auto_dumps_written.load(), current_config.max_auto_dumps,
                            state->unique_name.c_str());
            } else if (current_config.auto_dump_enabled && !state->dump_pending) {
                state->dump_pending = true;
                state->dump_trigger_frame = state->frames_processed;
                state->dump_target_frame =
                    state->frames_processed + current_config.post_trigger_frames;
                state->dump_metrics = frame_metrics;

                // Enforce refractory cooldown starting after trigger window completes
                state->cooldown_until_frame =
                    state->dump_target_frame + current_config.cooldown_frames;
                if (!is_manual_trigger) {
                    _auto_dumps_written++;
                }
            }
        }

        // 5. Complete candidate dump once post-trigger frames arrive in circular ring buffer
        if (state->dump_pending && state->frames_processed >= state->dump_target_frame) {
            state->dispatch_candidate_dump(state->dump_trigger_frame, state->dump_metrics);
            state->dump_pending = false;
        }
    }

    state->frames_processed++;
    return record_end_event();
}

} // namespace kotekan
