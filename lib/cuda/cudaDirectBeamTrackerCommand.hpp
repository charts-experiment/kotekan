#ifndef CUDA_DIRECT_BEAM_TRACKER_COMMAND_HPP
#define CUDA_DIRECT_BEAM_TRACKER_COMMAND_HPP

#include "Config.hpp"
#include "bufferContainer.hpp"
#include "chartsConstants.hpp"
#include "cudaCommand.hpp"
#include "cudaDeviceInterface.hpp"
#include "cudaDirectBeamTracker.hpp"
#include "driver_types.h"

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

class StatTracker;

namespace kotekan {

class cudaDirectBeamTrackerCommand;

struct TrackerGroup {
    std::string rest_prefix;
    std::mutex group_mutex;
    DirectBeamTrackerConfig config;
    std::vector<cudaDirectBeamTrackerCommand*> instances;
    bool endpoints_registered = false;
};

class cudaDirectBeamTrackerCommand : public cudaCommand {
public:
    cudaDirectBeamTrackerCommand(Config& config, const std::string& unique_name,
                                 bufferContainer& host_buffers, cudaDeviceInterface& device,
                                 int inst);
    ~cudaDirectBeamTrackerCommand() override;

    cudaEvent_t execute(cudaPipelineState& pipestate,
                        const std::vector<cudaEvent_t>& pre_events) override;

    static void set_shared_antenna_mask(const std::array<std::uint8_t, MAX_DIRECT_ANTENNAS>& mask,
                                        int n_active = -1);
    static void set_shared_antenna_mask(
        const std::array<std::uint8_t, charts::constants::charts_total_antennas>& mask,
        int n_active = -1);
    static std::array<std::uint8_t, MAX_DIRECT_ANTENNAS> get_shared_antenna_mask();
    static DirectBeamTrackerConfig get_shared_config();

    // Accessors for group subband stitching
    std::string get_rest_prefix() const {
        return _rest_prefix;
    }
    double get_freq_start_hz() const {
        return _freq_start_hz;
    }
    double get_freq_step_hz() const {
        return _freq_step_hz;
    }
    int32_t get_num_local_freq() const {
        return _num_local_freq;
    }
    int32_t get_max_beams() const {
        return _max_beams;
    }
    const float* get_host_beam_power() const {
        return _h_beam_power;
    }
    std::uint64_t get_last_frame_id() const {
        return _last_frame_id.load(std::memory_order_relaxed);
    }
    double get_last_timestamp_utc() const {
        return _last_timestamp_utc.load(std::memory_order_relaxed);
    }

private:
    void allocate_device_buffers();
    void free_device_buffers();
    void invalidate_cuda_graphs();
    void update_weights_if_needed(const DirectBeamTrackerConfig& cfg, cudaStream_t stream,
                                   double frame_timestamp_utc = -1.0);
    void upload_calibration_gains(const DirectBeamTrackerConfig& cfg);

    std::string _rest_prefix = "/direct_tracker";
    double _freq_start_hz = 300.0e6;
    double _freq_step_hz = 300.0e3;

    int32_t _num_elements = 64;
    int32_t _num_antennas = 64;
    int32_t _num_pols = 1;
    DirectPolLayout _pol_layout = DirectPolLayout::INTERLEAVED;
    int32_t _grid_cols = 8;

    int32_t _num_local_freq = 1;
    int32_t _samples_per_data_set = 15360;
    float _spacing_m = 0.6f;
    int32_t _buffer_depth = 3;
    int32_t _max_beams = 2;
    std::size_t _time_chunk_size = 256;
    std::size_t _time_unroll = 4;
    std::size_t _beam_tile_size = 4;
    std::string _antenna_order = "descending";

    // Timing and frame tracking
    std::atomic<std::uint64_t> _last_frame_id{0};
    std::atomic<double> _last_timestamp_utc{0.0};
    double _initial_utc_start_time = -1.0;
    double _sample_period_s = charts::constants::fpga_time_resolution_s;
    std::uint64_t _frame_counter = 0;

    // Kotekan StatTrackers for live REST telemetry (/trackers)
    std::shared_ptr<StatTracker> _tracker_kernel_time;
    std::shared_ptr<StatTracker> _tracker_realtime_budget;
    std::shared_ptr<StatTracker> _tracker_active_beams;
    std::shared_ptr<StatTracker> _tracker_vram_mb;

    // Per-instance generation trackers for multi-stream synchronization
    std::uint64_t _applied_config_generation = 0;
    std::uint64_t _applied_mask_generation = 0;

    // CUDA Graph state
    bool _enable_cuda_graph = false;
    std::size_t _last_graph_beams = 0;
    bool _last_graph_interp = false;
    std::vector<cudaGraph_t> _cuda_graphs;
    std::vector<cudaGraphExec_t> _cuda_graph_execs;

    std::string _gpu_mem_voltage;
    std::string _gpu_mem_formed_beams;

    std::vector<double> _frequencies_hz;

    // Persistent GPU buffers
    float2* _d_weights = nullptr;
    float2* _d_step_weights = nullptr;
    DirectDirection3D* _d_directions = nullptr;
    DirectDirection3D* _d_directions_end = nullptr;
    double* _d_wavenumbers = nullptr;
    float3* _d_antenna_positions = nullptr;
    float* _d_norm_taper = nullptr;
    float2* _d_calibration_gains = nullptr;

    // Complex antenna gains (charts-utils .npy, v_obs = g v); empty: no calibration
    std::string _gains_file;
    int32_t _gains_first_channel = 0;
    std::vector<int> _gains_antennas;

    // Power reduction buffers
    float* _d_beam_power = nullptr;
    float* _h_beam_power = nullptr;

    // Sky grid support
    bool _grid_mode_enabled = false;
    float _grid_step = 0.02f;
    std::size_t _num_grid_points = 0;
    std::vector<float2> _h_grid_lms;
    float2* _d_grid_weights = nullptr;
    float2* _d_grid_lms = nullptr;

    // Static TrackerGroup registry keyed by rest_prefix
    static std::mutex _registry_mutex;
    static std::map<std::string, std::shared_ptr<TrackerGroup>> _tracker_groups;
    static DirectBeamTrackerConfig _default_shared_config;
};

} // namespace kotekan

#endif // CUDA_DIRECT_BEAM_TRACKER_COMMAND_HPP
