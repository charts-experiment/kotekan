#ifndef CUDA_RFI_MITIGATION_COMMAND_HPP
#define CUDA_RFI_MITIGATION_COMMAND_HPP

#include "Config.hpp"
#include "bufferContainer.hpp"
#include "cudaCommand.hpp"
#include "cudaDeviceInterface.hpp"
#include "cudaRfiMitigation.hpp"

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace kotekan {

/**
 * @class cudaRfiMitigationCommand
 * @brief Lightweight real-time GPU RFI mitigation stage for the CHARTS Direct Beam Tracker
 *        pipeline. Computes scale-invariant per-(chunk, freq) Spectral Kurtosis, temporal burst
 *        MAD, and sliding passband frequency MAD across active antennas, optionally blanking
 *        contaminated time-frequency chunks in place before beamforming.
 */
class cudaRfiMitigationCommand : public cudaCommand {
public:
    cudaRfiMitigationCommand(Config& config, const std::string& unique_name,
                             bufferContainer& host_buffers, cudaDeviceInterface& device, int inst);
    ~cudaRfiMitigationCommand() override;

    cudaEvent_t execute(cudaPipelineState& pipestate,
                        const std::vector<cudaEvent_t>& pre_events) override;

    double get_freq_start_hz() const {
        return _freq_start_hz;
    }
    double get_freq_step_hz() const {
        return _freq_step_hz;
    }
    int32_t get_num_local_freq() const {
        return _num_local_freq;
    }
    std::size_t get_num_chunks() const {
        return _num_chunks;
    }
    std::uint64_t get_frame_counter() const {
        return _frame_counter;
    }
    std::vector<RfiChannelStats> get_last_channel_stats() const {
        std::lock_guard<std::mutex> lk(_stats_mutex);
        return _last_channel_stats;
    }

    static RfiMitigationConfig get_config();
    static void update_config(const RfiMitigationConfig& cfg);

private:
    void allocate_device_buffers();
    void free_device_buffers();
    void rebuild_static_notch_mask(const RfiMitigationConfig& cfg);

    std::string _gpu_mem_voltage;
    std::string _gpu_mem_rfi_mask;
    std::string _antenna_order = "descending";

    int32_t _num_elements = 64;
    int32_t _num_local_freq = 336;
    int32_t _samples_per_data_set = 15360;
    int32_t _buffer_depth = 3;
    std::size_t _num_chunks = 60;

    double _freq_start_hz = 300.0e6;
    double _freq_step_hz = 300.0e3;

    std::vector<int> _configured_active_raw;
    std::vector<int> _current_active_raw;
    std::uint64_t _applied_mask_gen = 0;
    std::uint64_t _applied_rfi_cfg_gen = 0;
    std::uint64_t _frame_counter = 0;

    // Device buffers
    int* _d_active_antennas = nullptr;
    float* _d_chunk_power = nullptr;
    float* _d_chunk_sk = nullptr;
    std::uint8_t* _d_static_notch = nullptr;
    std::uint8_t* _d_chunk_rfi_mask = nullptr;
    RfiChannelStats* _d_channel_stats = nullptr;

    // Pinned host telemetry buffer
    RfiChannelStats* _h_channel_stats = nullptr;

    mutable std::mutex _stats_mutex;
    std::vector<RfiChannelStats> _last_channel_stats;

    static std::mutex _global_mutex;
    static RfiMitigationConfig _shared_config;
    static std::uint64_t _config_generation;
    static std::vector<cudaRfiMitigationCommand*> _instances;
    static bool _endpoints_registered;
};

} // namespace kotekan

#endif // CUDA_RFI_MITIGATION_COMMAND_HPP
