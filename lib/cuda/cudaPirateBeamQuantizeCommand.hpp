#ifndef CUDA_PIRATE_BEAM_QUANTIZE_COMMAND_HPP
#define CUDA_PIRATE_BEAM_QUANTIZE_COMMAND_HPP

#include "Config.hpp"
#include "bufferContainer.hpp"
#include "cudaCommand.hpp"
#include "cudaDeviceInterface.hpp"
#include "cudaPirateBeamQuantize.hpp"
#include "driver_types.h"

#include <cstdint>
#include <string>
#include <vector>

namespace kotekan {

/**
 * @class cudaPirateBeamQuantizeCommand
 * @brief Fused CUDA stage bridging `cudaDirectBeamTrackerCommand` to `pirateNetworkSend`.
 *
 * Reads formed complex beam voltages from `gpu_mem_formed_beams`, optionally applies
 * `gpu_mem_rfi_mask`, decimates to intensity at FRB cadence, optionally subtracts the
 * cross-beam incoherent mean and a 31 ms sliding high-pass baseline, and packs
 * 256-sample minichunks in pirate v2 `float16 (scale, offset)` + `int4` wire format into
 * `gpu_mem_pirate` with forwarded metadata.
 */
class cudaPirateBeamQuantizeCommand : public cudaCommand {
public:
    cudaPirateBeamQuantizeCommand(Config& config, const std::string& unique_name,
                                  bufferContainer& host_buffers, cudaDeviceInterface& device,
                                  int inst);
    ~cudaPirateBeamQuantizeCommand() override;

    cudaEvent_t execute(cudaPipelineState& pipestate,
                        const std::vector<cudaEvent_t>& pre_events) override;

private:
    PirateBeamQuantizeConfig _cfg;

    std::string _gpu_mem_formed_beams;
    std::string _gpu_mem_rfi_mask;
    std::string _gpu_mem_pirate;

    float* _d_scratch_intensity = nullptr;
    std::uint8_t* _d_scratch_valid = nullptr;
};

} // namespace kotekan

#endif // CUDA_PIRATE_BEAM_QUANTIZE_COMMAND_HPP
