#ifndef CUDA_BEAM_UPCHANNELIZE_HPP
#define CUDA_BEAM_UPCHANNELIZE_HPP

#include <cstddef>
#include <cuda_runtime.h>
#include <string>

namespace kotekan {

enum class BeamUpchanWindowType : int { RECTANGULAR = 0, HANN = 1 };

BeamUpchanWindowType parse_beam_upchan_window(const std::string& name);
const char* beam_upchan_window_to_string(BeamUpchanWindowType w);

struct BeamUpchannelizeConfig {
    std::size_t n_time = 15360;
    std::size_t n_freq = 336;
    std::size_t num_beams = 1;
    std::size_t max_beams_stride = 1;
    std::size_t n_pol = 1;
    std::size_t upchan_factor = 16;
    BeamUpchanWindowType window_type = BeamUpchanWindowType::HANN;
};

/**
 * @brief Launch GPU 16-point unitary in-register temporal upchannelizer for formed beam voltages.
 *
 * Converts complex formed beam voltages of shape
 * `[n_time, n_freq, max_beams_stride, n_pol]` into fine spectra of shape
 * `[n_time / 16, n_freq * 16, max_beams_stride, n_pol]` with FFT-shifted frequency ordering
 * (`-BW/2` to `+BW/2` within each coarse channel). Optionally accumulates mean fine-channel
 * power `[num_beams, n_freq * 16]` summed across polarizations and averaged over `n_time / 16`.
 */
void launch_beam_upchannelize_16pt(const float2* d_formed_beams, float2* d_upchan_beams,
                                   float* d_upchan_power, const BeamUpchannelizeConfig& cfg,
                                   cudaStream_t stream = nullptr);

} // namespace kotekan

#endif // CUDA_BEAM_UPCHANNELIZE_HPP
