#ifndef CUDA_BISLC_COMMAND_HPP
#define CUDA_BISLC_COMMAND_HPP

#include "Config.hpp"
#include "bufferContainer.hpp"
#include "cudaCommand.hpp"
#include "cudaDeviceInterface.hpp"
#include "cudaBiSLC.hpp"
#include "driver_types.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace kotekan {

/**
 * @class cudaBiSLCCommand
 * @brief CUDA-accelerated processing node performing real-time bi-directional coherent
 *        sidelobe unmixing (Joint Matrix Unmixing) across concurrent virtual beams.
 */
class cudaBiSLCCommand : public cudaCommand {
public:
    cudaBiSLCCommand(Config& config, const std::string& unique_name,
                     bufferContainer& host_buffers,
                     cudaDeviceInterface& device, int inst);
    ~cudaBiSLCCommand() override;

    cudaEvent_t execute(cudaPipelineState& pipestate,
                        const std::vector<cudaEvent_t>& pre_events) override;

    static void set_shared_config(const BiSLCConfig& config);
    static BiSLCConfig get_shared_config();

private:
    void allocate_device_buffers();
    void free_device_buffers();
    void update_matrices_if_needed(cudaStream_t stream);

    int32_t _num_elements;
    int32_t _num_local_freq;
    int32_t _samples_per_data_set;
    int32_t _buffer_depth;
    int32_t _max_beams;
    float _spacing_m;
    float _diagonal_loading;
    bool _enabled;

    std::string _gpu_mem_formed_beams;
    std::string _gpu_mem_cleaned_beams;

    std::vector<double> _frequencies_hz;

    // Persistent GPU buffers
    float2* _d_unmix_matrices = nullptr;

    // Host buffers for matrix inversion staging
    std::vector<float2> _h_unmix_matrices;
    std::vector<float2> _h_coupling_matrices;

    // Thread-safe shared configuration for live dynamic updates
    static std::mutex _global_mutex;
    static BiSLCConfig _shared_config;
    static bool _endpoints_registered;
    static bool _matrices_dirty;
};

} // namespace kotekan

#endif // CUDA_BISLC_COMMAND_HPP
