#ifndef CUDA_BEAM_UPCHANNELIZE_COMMAND_HPP
#define CUDA_BEAM_UPCHANNELIZE_COMMAND_HPP

#include "Config.hpp"
#include "bufferContainer.hpp"
#include "cudaBeamUpchannelize.hpp"
#include "cudaCommand.hpp"
#include "cudaDeviceInterface.hpp"
#include "driver_types.h"

#include <string>
#include <vector>

namespace kotekan {

/**
 * @class cudaBeamUpchannelizeCommand
 * @brief GPU 16-point temporal upchannelizer for coherent formed beam voltages.
 *
 * Reads formed complex beam voltages from `gpu_mem_formed_beams`
 * (`[samples_per_data_set, num_local_freq, max_beams, num_pols]`), computes a unitary
 * 16-point windowed FFT along time with FFT-shift (`-BW/2` .. `+BW/2`), writes complex
 * fine spectra into `gpu_mem_upchan_beams` (forwarding metadata), and optionally writes
 * integrated fine-channel power `[num_beams, num_local_freq * 16]` into
 * `gpu_mem_upchan_power`.
 */
class cudaBeamUpchannelizeCommand : public cudaCommand {
public:
    cudaBeamUpchannelizeCommand(Config& config, const std::string& unique_name,
                                bufferContainer& host_buffers, cudaDeviceInterface& device,
                                int inst);
    ~cudaBeamUpchannelizeCommand() override = default;

    cudaEvent_t execute(cudaPipelineState& pipestate,
                        const std::vector<cudaEvent_t>& pre_events) override;

private:
    BeamUpchannelizeConfig _cfg;

    std::string _gpu_mem_formed_beams;
    std::string _gpu_mem_upchan_beams;
    std::string _gpu_mem_upchan_power;
};

} // namespace kotekan

#endif // CUDA_BEAM_UPCHANNELIZE_COMMAND_HPP
