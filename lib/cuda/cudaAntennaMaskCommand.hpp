#ifndef CUDA_ANTENNA_MASK_COMMAND_HPP
#define CUDA_ANTENNA_MASK_COMMAND_HPP

#include "Config.hpp"
#include "bufferContainer.hpp"
#include "cudaAntennaMask.hpp"
#include "cudaCommand.hpp"
#include "cudaDeviceInterface.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace kotekan {

/**
 * @class cudaAntennaMaskCommand
 * @brief High-throughput GPU stage in Kotekan for dead and saturated antenna detection,
 *        debouncing, in-place voltage blanking, and cross-stage mask propagation.
 */
class cudaAntennaMaskCommand : public cudaCommand {
public:
    cudaAntennaMaskCommand(Config& config, const std::string& unique_name,
                           bufferContainer& host_buffers,
                           cudaDeviceInterface& device, int inst);
    ~cudaAntennaMaskCommand() override;

    cudaEvent_t execute(cudaPipelineState& pipestate,
                        const std::vector<cudaEvent_t>& pre_events) override;

    // Static accessors for inter-stage queries & REST integration
    static std::array<std::uint8_t, MAX_MASK_ANTENNAS> get_current_mask();
    static AntennaHealthMetrics get_antenna_metrics(std::size_t ant_id);
    static void set_manual_mask(std::size_t ant_id, bool enable);
    static void reset_all_masks();
    static void update_config(const AntennaMaskConfig& cfg);
    static AntennaMaskConfig get_config();

private:
    void allocate_device_buffers();
    void free_device_buffers();

    int32_t _num_elements = 256;
    int32_t _num_local_freq = 336;
    int32_t _samples_per_data_set = 3840;
    int32_t _buffer_depth = 3;

    std::string _gpu_mem_voltage;
    std::string _gpu_mem_alive_voltages;
    int32_t _max_alive_antennas = 64;

    // Device allocations
    float* _d_antenna_powers = nullptr;
    std::uint32_t* _d_antenna_clips = nullptr;
    int* _d_bad_antennas = nullptr;
    int* _d_alive_antennas = nullptr;

    // Pinned host memory for low-latency telemetry copy
    float* _h_antenna_powers = nullptr;
    std::uint32_t* _h_antenna_clips = nullptr;

    std::vector<int> _current_bad_indices;
    std::vector<int> _current_alive_indices;

    // Thread-safe shared configuration and metrics state
    static std::mutex _global_mutex;
    static AntennaMaskConfig _shared_config;
    static std::array<AntennaHealthMetrics, MAX_MASK_ANTENNAS> _shared_metrics;
    static std::array<std::uint8_t, MAX_MASK_ANTENNAS> _shared_mask;
    static bool _endpoints_registered;
    static bool _mask_dirty;
};

} // namespace kotekan

#endif // CUDA_ANTENNA_MASK_COMMAND_HPP
