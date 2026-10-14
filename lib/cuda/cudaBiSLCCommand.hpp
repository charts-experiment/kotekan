#ifndef CUDA_BISLC_COMMAND_HPP
#define CUDA_BISLC_COMMAND_HPP

#include "Config.hpp"
#include "bufferContainer.hpp"
#include "cudaBiSLC.hpp"
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
 * @class cudaBiSLCCommand
 * @brief CUDA-accelerated processing node performing real-time bi-directional coherent
 *        sidelobe unmixing (Joint Matrix Unmixing) across concurrent virtual beams.
 */
class cudaBiSLCCommand : public cudaCommand {
public:
    cudaBiSLCCommand(Config& config, const std::string& unique_name, bufferContainer& host_buffers,
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

    int32_t _num_elements = 64;
    int32_t _num_antennas = 64;
    int32_t _num_pols = 1;
    int32_t _num_local_freq = 336;
    int32_t _samples_per_data_set = 15360;
    int32_t _buffer_depth = 4;
    int32_t _max_beams = 16;
    float _spacing_m = 0.5f;
    float _diagonal_loading = 1.0e-4f;
    bool _enabled = true;

    std::uint64_t _applied_bislc_generation = 0;
    std::uint64_t _applied_tracker_generation = 0;

    std::string _gpu_mem_formed_beams;
    std::string _gpu_mem_cleaned_beams;

    std::vector<double> _frequencies_hz;

    // Persistent GPU buffers for zero-CPU-stall matrix inversion & unmixing
    float2* _d_unmix_matrices = nullptr;
    float2* _d_coupling_matrices = nullptr;
    DirectDirection3D* _d_directions = nullptr;
    double* _d_wavenumbers = nullptr;
    float3* _d_antenna_positions = nullptr;
    std::uint8_t* _d_antenna_mask = nullptr;

    // Pinned host buffers for DMA-safe, asynchronous streaming
    DirectDirection3D* _h_directions = nullptr;
    float3* _h_antenna_positions = nullptr;
    std::uint8_t* _h_antenna_mask = nullptr;

    // Host staging for telemetry inspection
    std::vector<float2> _h_coupling_matrices;

    static std::mutex _global_mutex;
    static BiSLCConfig _shared_config;
    static bool _endpoints_registered;
};

} // namespace kotekan

#endif // CUDA_BISLC_COMMAND_HPP
