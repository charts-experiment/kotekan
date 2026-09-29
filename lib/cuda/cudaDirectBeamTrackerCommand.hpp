#ifndef CUDA_DIRECT_BEAM_TRACKER_COMMAND_HPP
#define CUDA_DIRECT_BEAM_TRACKER_COMMAND_HPP

#include "Config.hpp"
#include "bufferContainer.hpp"
#include "cudaCommand.hpp"
#include "cudaDeviceInterface.hpp"
#include "cudaDirectBeamTracker.hpp"
#include "driver_types.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace kotekan {

/**
 * @class cudaDirectBeamTrackerCommand
 * @brief High-throughput Direct Beam Tracker stage in Kotekan with direct complex weight application,
 *        precomputed sky grid codebooks, and live REST control.
 */
class cudaDirectBeamTrackerCommand : public cudaCommand {
public:
    cudaDirectBeamTrackerCommand(Config& config, const std::string& unique_name,
                                bufferContainer& host_buffers,
                                cudaDeviceInterface& device, int inst);
    ~cudaDirectBeamTrackerCommand() override;

    cudaEvent_t execute(cudaPipelineState& pipestate,
                        const std::vector<cudaEvent_t>& pre_events) override;

    static void set_shared_antenna_mask(const std::array<std::uint8_t, MAX_DIRECT_ANTENNAS>& mask, int n_active = -1);
    static std::array<std::uint8_t, MAX_DIRECT_ANTENNAS> get_shared_antenna_mask();
    static DirectBeamTrackerConfig get_shared_config();

private:
    void allocate_device_buffers();
    void free_device_buffers();
    void update_weights_if_needed(cudaStream_t stream);

    int32_t _num_elements;
    int32_t _num_local_freq;
    int32_t _samples_per_data_set;
    float _spacing_m;
    int32_t _buffer_depth;
    int32_t _max_beams;
    std::size_t _time_chunk_size;
    std::size_t _time_unroll;
    std::size_t _beam_tile_size = 4;

    // CUDA Graph state for zero CPU dispatch latency across ring buffer slots
    bool _enable_cuda_graph = false;
    std::size_t _last_graph_beams = 0;
    std::vector<cudaGraph_t> _cuda_graphs;
    std::vector<cudaGraphExec_t> _cuda_graph_execs;

    std::string _gpu_mem_voltage;
    std::string _gpu_mem_formed_beams;

    std::vector<double> _frequencies_hz;

    // Subframe phase interpolation
    bool _enable_subframe_interpolation = true;

    // Persistent GPU buffers allocated once
    float2* _d_weights = nullptr;
    float2* _d_step_weights = nullptr;
    DirectDirection3D* _d_directions = nullptr;
    DirectDirection3D* _d_directions_end = nullptr;
    double* _d_wavenumbers = nullptr;
    float3* _d_antenna_positions = nullptr;
    std::uint8_t* _d_antenna_mask = nullptr;
    float2* _d_calibration_gains = nullptr;

    // Sky grid support
    bool _grid_mode_enabled = false;
    float _grid_step = 0.02f;
    std::size_t _num_grid_points = 0;
    std::vector<float2> _h_grid_lms;
    float2* _d_grid_weights = nullptr;
    float2* _d_grid_lms = nullptr;

    // Thread-safe shared configuration for live dynamic updates
    static std::mutex _global_mutex;
    static DirectBeamTrackerConfig _shared_config;
    static bool _endpoints_registered;
    static bool _weights_dirty;
};

} // namespace kotekan

#endif // CUDA_DIRECT_BEAM_TRACKER_COMMAND_HPP
