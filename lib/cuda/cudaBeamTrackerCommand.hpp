#ifndef CUDA_BEAM_TRACKER_COMMAND_HPP
#define CUDA_BEAM_TRACKER_COMMAND_HPP

#include "Config.hpp"
#include "bufferContainer.hpp"
#include "cudaBeamTrackerV5.hpp"
#include "cudaCommand.hpp"
#include "cudaDeviceInterface.hpp"
#include "driver_types.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace kotekan {

/**
 * @class cudaBeamTrackerCommand
 * @brief cudaCommand stage plugin for running the V5 Multi-Beam Tracker in cudaProcess with live
 * REST control and Astrometry tracking.
 */
class cudaBeamTrackerCommand : public cudaCommand {
public:
    cudaBeamTrackerCommand(Config& config, const std::string& unique_name,
                           bufferContainer& host_buffers, cudaDeviceInterface& device, int inst);
    ~cudaBeamTrackerCommand() override = default;

    cudaEvent_t execute(cudaPipelineState& pipestate,
                        const std::vector<cudaEvent_t>& pre_events) override;

private:
    int32_t _num_elements;
    int32_t _num_local_freq;
    int32_t _samples_per_data_set;
    int32_t _integration_spectra;
    float _spacing_m;
    int32_t _buffer_depth;
    int32_t _max_beams;
    double _sample_period_s;

    std::string _gpu_mem_voltage;
    std::string _gpu_mem_formed_beams;

    std::vector<double> _frequencies_hz;

    // Thread-safe shared configuration for live dynamic updates
    static std::mutex _global_mutex;
    static MultiBeamTrackerConfig _shared_config;
    static bool _endpoints_registered;
};

} // namespace kotekan

#endif // CUDA_BEAM_TRACKER_COMMAND_HPP
