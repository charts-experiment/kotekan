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
    void finalize_frame() override;

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
    std::vector<float> get_pending_channel_gains() const;

    void set_local_channel_gains(const std::vector<float>& gains);
    void queue_ema_reset_channel(int local_channel);

    static RfiMitigationConfig get_config();
    static void update_config(const RfiMitigationConfig& cfg);
    static bool set_channel_gains_by_freq(double freq_start_hz, const std::vector<float>& gains);
    static std::pair<int, int> reset_ema_channels_by_freq_mhz(const std::vector<double>& freqs_mhz);

private:
    void allocate_device_buffers();
    void free_device_buffers();
    void rebuild_static_notch_mask(const RfiMitigationConfig& cfg, cudaStream_t stream = nullptr);

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
    std::vector<int> _excision_exempt_freq_ids;
    std::vector<std::uint8_t> _h_exempt_channels;
    bool _exempt_resolved = false;
    std::uint64_t _applied_mask_gen = 0;
    std::uint64_t _applied_rfi_cfg_gen = 0;
    std::uint64_t _applied_ema_reset_gen = 1;
    std::uint64_t _frame_counter = 0;

    // Device buffers
    int* _d_active_antennas = nullptr;
    float* _d_chunk_power = nullptr;
    float* _d_chunk_sk = nullptr;
    float* _d_chunk_sat_frac = nullptr;
    std::uint8_t* _d_static_notch = nullptr;
    float* _d_channel_gains = nullptr;
    std::uint8_t* _d_chunk_rfi_mask = nullptr;
    std::uint8_t* _d_exempt_channels = nullptr;
    float* _d_ema_power = nullptr;
    float* _d_ema_mad = nullptr;
    RfiChannelStats* _d_channel_stats = nullptr;

    // Host pending/staging buffers (async-uploaded on stage CUDA stream in execute())
    std::vector<std::uint8_t> _h_static_notch;
    std::vector<float> _pending_channel_gains;
    std::vector<float> _exec_channel_gains;
    bool _pending_gains_dirty = false;
    std::vector<int> _pending_ema_reset_indices;

    // Pinned host telemetry buffer
    RfiChannelStats* _h_channel_stats = nullptr;
    bool _last_frame_enabled = false;

    mutable std::mutex _stats_mutex;
    std::vector<RfiChannelStats> _last_channel_stats;

    static std::mutex _global_mutex;
    static RfiMitigationConfig _shared_config;
    static std::uint64_t _config_generation;
    static std::uint64_t _ema_reset_generation;
    static std::vector<cudaRfiMitigationCommand*> _instances;
    static bool _endpoints_registered;
};

} // namespace kotekan

#endif // CUDA_RFI_MITIGATION_COMMAND_HPP
