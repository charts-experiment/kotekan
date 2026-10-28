/**
 * @file
 * @brief Per-visibility gain calibration applied after the Astron N2 correlator.
 *  - cudaVisibilityCalibrator : public cudaCommand
 */

#ifndef CUDA_VISIBILITY_CALIBRATOR_HPP
#define CUDA_VISIBILITY_CALIBRATOR_HPP

#include "Config.hpp"              // for Config
#include "bufferContainer.hpp"     // for bufferContainer
#include "cudaCommand.hpp"         // for cudaCommand, cudaPipelineState
#include "cudaDeviceInterface.hpp" // for cudaDeviceInterface
#include "driver_types.h"          // for cudaEvent_t, cudaStream_t

#include <complex>  // for complex
#include <stddef.h> // for size_t
#include <stdint.h> // for int32_t
#include <string>   // for string
#include <vector>   // for vector

/// Launches the in-place calibration kernel (defined in cudaVisibilityCalibrator.cu).
void launch_apply_visibility_calibration(int32_t* d_correlation, const float* d_calibration,
                                         size_t num_visibilities, cudaStream_t stream);

/**
 * @class cudaVisibilityCalibrator
 * @brief Multiplies the int32 complex visibilities produced by cudaCorrelatorAstron, in place,
 *        by a precomputed complex64 table of 1 / (g_i * conj(g_j)).
 *
 * The table is built offline by charts-utils (scripts/kotekan_vis_calibrator.py) in the same
 * layout as the correlator output, [freq][block][pol_a][pol_b], and loaded from a .npy file at
 * start-up. Each result is rounded to the nearest int32 with saturation. Without a
 * `calibration_file` the command is a pass-through.
 */
class cudaVisibilityCalibrator : public cudaCommand {
public:
    cudaVisibilityCalibrator(kotekan::Config& config, const std::string& unique_name,
                             kotekan::bufferContainer& host_buffers, cudaDeviceInterface& device,
                             int inst);
    ~cudaVisibilityCalibrator();
    cudaEvent_t execute(cudaPipelineState& pipestate,
                        const std::vector<cudaEvent_t>& pre_events) override;

private:
    /// Frequencies in the correlation frame handled by this command.
    int32_t _num_local_freq;
    /// Triangular 2x2 blocks per frequency (528 for 64 antennas).
    int32_t _num_blocks;
    /// Fixed at 2 by the Astron kernel.
    int32_t _block_size;
    /// Must be 1; the table has no data-set axis.
    int32_t _num_data_sets;
    /// First channel of this frame inside the full-band table.
    int32_t _freq_offset;

    /// GPU side memory name for the correlated data, calibrated in place
    std::string _gpu_mem_correlation_matrix;

    /// Number of complex visibilities in a frame
    size_t _num_visibilities;
    /// Device copy of this command's slice of the table, as interleaved float (re, im)
    float* _d_calibration = nullptr;
};

#endif // CUDA_VISIBILITY_CALIBRATOR_HPP
