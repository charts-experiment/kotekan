#include "cudaCorrelatorAstron.hpp"

#include "cuda.h"             // for cuGetErrorString, cuLaunchKernel, CUresult, cudaError_enum
#include "gpuCommand.hpp"     // for gpuCommandType
#include "kotekanLogging.hpp" // for INFO

#include "fmt.hpp" // for format, format_string, compile_string_to_view

#include <map>       // for map
#include <stdexcept> // for runtime_error

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_CUDA_COMMAND(cudaCorrelatorAstron);

cudaCorrelatorAstron::cudaCorrelatorAstron(Config& config, const std::string& unique_name,
                                           bufferContainer& host_buffers,
                                           cudaDeviceInterface& device, int inst) :
    cudaCommand(config, unique_name, host_buffers, device, inst, no_cuda_command_state, "correlate",
                "TCCorrelator.cu") {
    _num_elements = config.get<int>(unique_name, "num_elements");
    _num_local_freq = config.get<int>(unique_name, "num_local_freq");
    _samples_per_data_set = config.get<int>(unique_name, "samples_per_data_set");
    _num_data_sets = config.get<int>(unique_name, "num_data_sets");
    _block_size = config.get_default<int>(unique_name, "block_size", 2);
    _elements_per_thread_block =
        config.get_default<int>(unique_name, "elements_per_thread_block", 32);
    _num_blocks = config.get<int>(unique_name, "num_blocks");
    _buffer_depth = config.get<int>(unique_name, "buffer_depth");
    _gpu_mem_voltage = config.get<std::string>(unique_name, "gpu_mem_voltage");
    _gpu_mem_correlation_matrix =
        config.get<std::string>(unique_name, "gpu_mem_correlation_matrix");

    set_command_type(gpuCommandType::KERNEL);

    if (_block_size != 2)
        throw std::runtime_error("The block size must be 2 for the Astron TC kernels");

    if (_elements_per_thread_block != 32 && _elements_per_thread_block != 48
        && _elements_per_thread_block != 64)
        throw std::runtime_error("elements_per_thread_block must be one of 32, 48, 64 for TCCorrelator");


    // Multiple cudaProcess stages may share a CUDA device (for example, one
    // pipeline per frequency block). The compiled kernel is device-wide, so
    // only the first command instance on that device should register it.
    if (inst == 0 && !device.runtime_kernels.count("correlate")) {
        std::vector<std::string> opts = {
            "-arch=compute_86",
            "-lineinfo",
            "-DNR_BITS=4",
            fmt::format("-DNR_RECEIVERS={:d}", _num_elements / 2),
            fmt::format("-DNR_CHANNELS={:d}", _num_local_freq),
            fmt::format("-DNR_SAMPLES_PER_CHANNEL={:d}", _samples_per_data_set),
            fmt::format("-DNR_RECEIVERS_PER_BLOCK={:d}", _elements_per_thread_block),
            "-DNR_POLARIZATIONS=2", // 10/03/2026 On CHARTS now we are using 1 pol, but the kernel expect 2, so this is set to 2 for now and its working fine
            "-I/usr/local/cuda/include"};
        device.build(kernel_file_name, {"correlate"}, opts);
    }
}

cudaCorrelatorAstron::~cudaCorrelatorAstron() {}

cudaEvent_t cudaCorrelatorAstron::execute(cudaPipelineState&, const std::vector<cudaEvent_t>&) {
    pre_execute();

    size_t input_frame_len = (size_t)_num_elements * _num_local_freq * _samples_per_data_set;
    void* input_memory = device.get_gpu_memory(_gpu_mem_voltage, input_frame_len);

    size_t output_len = (size_t)_num_local_freq * _num_blocks * (_block_size * _block_size) * 2
                        * _num_data_sets * sizeof(int32_t);
    void* output_memory = device.get_gpu_memory_array(_gpu_mem_correlation_matrix, gpu_frame_id,
                                                      _gpu_buffer_depth, output_len);

    record_start_event();

    CUresult err;
    void* parameters[] = {&output_memory, &input_memory};

    const int nr_receivers = _num_elements / 2;

    // n = ceil(nr_receivers / NR_RECEIVERS_PER_BLOCK)
    int n = (nr_receivers + _elements_per_thread_block - 1) / _elements_per_thread_block;
    if (n < 1) n = 1;

    // triangular blocks
    int num_thread_blocks = n * (n + 1) / 2;
    if (num_thread_blocks < 1) num_thread_blocks = 1;

    err = cuLaunchKernel(device.runtime_kernels["correlate"], num_thread_blocks, _num_local_freq, 1,
                         32, 2, 2, 0, device.getStream(cuda_stream_id), parameters, NULL);
    if (err != CUDA_SUCCESS) {
        const char* errStr;
        cuGetErrorString(err, &errStr);
        INFO("ERROR IN cuLaunchKernel: {}", errStr);
    }

    return record_end_event();
}
