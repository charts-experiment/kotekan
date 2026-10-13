// Validate the shared library boundary without needing a CUDA device.
#include "beamformer/cuda_stage_api.hpp"
#include "beamformer/cuda_fine_channel_pipeline.hpp"
#include "beamformer/cpu_beamformer.hpp"
#include <iostream>
#include <stdexcept>

using Launch = void (*)(beamformer::CudaBeamformerKernel, cudaStream_t,
                        const std::uint8_t*, const beamformer::ComplexFloat*, float*,
                        const beamformer::BeamformerInputContract&);
// Volatile references require real linker definitions; no null-buffer launch.
Launch volatile linked_launch = static_cast<Launch>(&beamformer::launch_packed_beamformer);
auto* volatile linked_fine_validation = &beamformer::validate_fine_pipeline_buffers;

int main() {
    using namespace beamformer;
    if (!linked_launch || !linked_fine_validation)
        throw std::runtime_error("Missing CUDA stage symbols");
    // One compact CPU numerical reference, never a CPU performance benchmark.
    const Dimensions dims{1, 336, 32, 1};
    PackedVoltage packed(packed_voltage_bytes(dims), 0x01);
    Weights weights(336 * 32, ComplexFloat{1.0F, 0.0F});
    auto intensity = cpu_beamform_packed_intensity(packed, weights, dims);
    for (const auto value : intensity)
        if (value != 1024.0F) throw std::runtime_error("Reference library mismatch");
    for (const auto t : {15360u, 153600u}) {
        auto input = make_beamformer_input_contract(BeamformerInputMode::ChannelizedFine, t);
        const auto metadata = fine_pipeline_metadata(input);
        if (metadata.output_dimensions.n_time != t / 320)
            throw std::runtime_error("Fine contract mismatch");
    }
    std::cout << "PASS: shared CPU reference and CUDA stage symbols; no GPU launch\n";
}
