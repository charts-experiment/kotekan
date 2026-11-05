#ifndef CUDA_TRANSIENT_TRIGGER_COMMAND_HPP
#define CUDA_TRANSIENT_TRIGGER_COMMAND_HPP

#include "Config.hpp"
#include "bufferContainer.hpp"
#include "cudaCommand.hpp"
#include "cudaDeviceInterface.hpp"
#include "cudaTransientTrigger.hpp"
#include "driver_types.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace kotekan {

/**
 * @class cudaTransientTriggerState
 * @brief Shared state across all pipeline instances of cudaTransientTriggerCommand in a cudaProcess
 * stage. Owns the single VRAM circular ring buffer and pinned dump staging buffers.
 */
class cudaTransientTriggerState : public cudaCommandState {
public:
    cudaTransientTriggerState(Config& config, const std::string& unique_name,
                              bufferContainer& host_buffers, cudaDeviceInterface& device);
    ~cudaTransientTriggerState();

    void allocate_device_buffers();
    void free_device_buffers();
    void dispatch_candidate_dump(uint32_t trigger_frame, const TransientFrameMetrics& metrics);

    cudaDeviceInterface& device;
    std::string unique_name;
    std::string dump_directory;

    int32_t num_local_freq = 0;
    int32_t samples_per_data_set = 0;
    int32_t max_beams = 2;

    // GPU VRAM Buffers (single instance shared across buffer_depth commands!)
    float2* d_ring_buffer = nullptr;
    float2* d_metrics = nullptr;

    // Pinned Host Buffers
    float2* h_metrics_pinned = nullptr;
    float2* h_dump_pinned = nullptr;

    // Dedicated Stream for non-blocking candidate dumps
    cudaStream_t dump_stream = nullptr;

    // Ring buffer sizing
    uint32_t ring_buffer_depth = 16;
    uint32_t pre_trigger_frames = 4;
    uint32_t post_trigger_frames = 4;
    std::size_t frame_elements = 0;
    std::size_t frame_bytes = 0;

    // Runtime state (protected by state_mutex)
    std::mutex state_mutex;
    uint32_t frames_processed = 0;
    bool dump_pending = false;
    uint32_t dump_trigger_frame = 0;
    uint32_t dump_target_frame = 0;
    TransientFrameMetrics dump_metrics;
    uint32_t last_manual_trigger_id = 0;
    uint32_t cooldown_until_frame = 0;
    uint32_t auto_dumps_written = 0;
};

/**
 * @class cudaTransientTriggerCommand
 * @brief High-performance CUDA processing and ring buffering stage that continuously monitors
 *        cleaned baseband beams for transient events using Spectral Kurtosis and Inter-Beam
 *        Cross-Correlation, triggering automated candidate disk dumps from GPU VRAM.
 */
class cudaTransientTriggerCommand : public cudaCommand {
public:
    cudaTransientTriggerCommand(Config& config, const std::string& unique_name,
                                bufferContainer& host_buffers, cudaDeviceInterface& device,
                                int inst, std::shared_ptr<cudaCommandState> state = nullptr);
    ~cudaTransientTriggerCommand() override;

    cudaEvent_t execute(cudaPipelineState& pipestate,
                        const std::vector<cudaEvent_t>& pre_events) override;

    static void set_shared_config(const TransientTriggerConfig& config);
    static TransientTriggerConfig get_shared_config();
    static void trigger_manual();

    // Friend access to static members from cudaTransientTriggerState
    friend class cudaTransientTriggerState;

protected:
    cudaTransientTriggerState* get_state();

private:
    int32_t _num_local_freq;
    int32_t _samples_per_data_set;
    int32_t _buffer_depth;
    int32_t _max_beams;

    std::string _gpu_mem_cleaned_beams;
    std::string _gpu_mem_output;

    // Thread-safe shared configuration and telemetry across all stages
    static std::mutex _global_mutex;
    static TransientTriggerConfig _shared_config;
    static bool _endpoints_registered;
    static std::atomic<uint32_t> _manual_trigger_count;
    static uint32_t _total_triggers_fired;
    static std::atomic<uint32_t> _auto_dumps_written;
    static TransientFrameMetrics _last_trigger_metrics;
    static std::string _last_dump_path;
};

} // namespace kotekan

#endif // CUDA_TRANSIENT_TRIGGER_COMMAND_HPP
