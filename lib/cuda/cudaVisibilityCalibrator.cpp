#include "cudaVisibilityCalibrator.hpp"

#include "cudaUtils.hpp"      // for CHECK_CUDA_ERROR_NON_OO
#include "gpuCommand.hpp"     // for gpuCommandType
#include "kotekanLogging.hpp" // for INFO

#include <algorithm>    // for min
#include <cstring>      // for memcmp
#include <fstream>      // for ifstream
#include <regex>        // for regex
#include <stdexcept>    // for runtime_error

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_CUDA_COMMAND(cudaVisibilityCalibrator);

namespace {

/// Interleaved (re, im) float values and the array shape of a little-endian complex64 .npy.
struct NpyComplex64 {
    std::vector<size_t> shape;
    std::vector<float> data;
};

NpyComplex64 read_npy_complex64(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("cudaVisibilityCalibrator: cannot open " + path);

    char magic[8];
    in.read(magic, 8);
    if (!in || std::memcmp(magic, "\x93NUMPY", 6) != 0)
        throw std::runtime_error("cudaVisibilityCalibrator: " + path + " is not a .npy file");
    const int major = magic[6];

    uint32_t header_len = 0;
    if (major == 1) {
        uint16_t len16 = 0;
        in.read(reinterpret_cast<char*>(&len16), 2);
        header_len = len16;
    } else {
        in.read(reinterpret_cast<char*>(&header_len), 4);
    }
    std::string header(header_len, '\0');
    in.read(header.data(), header_len);
    if (!in)
        throw std::runtime_error("cudaVisibilityCalibrator: truncated .npy header in " + path);

    if (header.find("'<c8'") == std::string::npos
        || header.find("'fortran_order': False") == std::string::npos)
        throw std::runtime_error("cudaVisibilityCalibrator: " + path
                                 + " must be a C-ordered complex64 (<c8) array");

    std::smatch m;
    if (!std::regex_search(header, m, std::regex("'shape': \\(([^)]*)\\)")))
        throw std::runtime_error("cudaVisibilityCalibrator: no shape in " + path);
    NpyComplex64 out;
    size_t count = 1;
    const std::string dims = m[1];
    const std::regex digits("\\d+");
    for (std::sregex_iterator it(dims.begin(), dims.end(), digits), end; it != end; ++it) {
        out.shape.push_back(std::stoull(it->str()));
        count *= out.shape.back();
    }

    out.data.resize(count * 2);
    in.read(reinterpret_cast<char*>(out.data.data()), count * 2 * sizeof(float));
    if (!in)
        throw std::runtime_error("cudaVisibilityCalibrator: " + path + " ended early");
    return out;
}

} // namespace

cudaVisibilityCalibrator::cudaVisibilityCalibrator(Config& config, const std::string& unique_name,
                                                   bufferContainer& host_buffers,
                                                   cudaDeviceInterface& device, int inst) :
    cudaCommand(config, unique_name, host_buffers, device, inst, no_cuda_command_state,
                "cudaVisibilityCalibrator") {
    _num_local_freq = config.get<int>(unique_name, "num_local_freq");
    _num_blocks = config.get<int>(unique_name, "num_blocks");
    _block_size = config.get_default<int>(unique_name, "block_size", 2);
    _num_data_sets = config.get_default<int>(unique_name, "num_data_sets", 1);
    _freq_offset = config.get_default<int>(unique_name, "freq_offset", 0);
    _gpu_mem_correlation_matrix =
        config.get<std::string>(unique_name, "gpu_mem_correlation_matrix");
    const std::string calibration_file =
        config.get_default<std::string>(unique_name, "calibration_file", "");

    set_command_type(gpuCommandType::KERNEL);

    if (_block_size != 2 || _num_data_sets != 1)
        throw std::runtime_error("cudaVisibilityCalibrator requires block_size 2 and "
                                 "num_data_sets 1, as in the Astron correlator setup");
    if (_freq_offset < 0)
        throw std::runtime_error("cudaVisibilityCalibrator freq_offset must be >= 0");

    const size_t vis_per_freq = (size_t)_num_blocks * _block_size * _block_size;
    _num_visibilities = (size_t)_num_local_freq * vis_per_freq;

    if (calibration_file.empty()) {
        INFO("No calibration_file given: visibilities pass through uncalibrated");
        return;
    }

    NpyComplex64 table = read_npy_complex64(calibration_file);
    if (table.shape.size() != 4 || table.shape[1] != (size_t)_num_blocks || table.shape[2] != 2
        || table.shape[3] != 2)
        throw std::runtime_error("cudaVisibilityCalibrator: table in " + calibration_file
                                 + " must have shape (n_freq, num_blocks, 2, 2)");
    if ((size_t)_freq_offset + _num_local_freq > table.shape[0])
        throw std::runtime_error("cudaVisibilityCalibrator: freq_offset + num_local_freq exceeds "
                                 "the table's channels");

    const float* slice = table.data.data() + (size_t)_freq_offset * vis_per_freq * 2;
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc((void**)&_d_calibration, _num_visibilities * 2 * sizeof(float)));
    CHECK_CUDA_ERROR_NON_OO(cudaMemcpy(_d_calibration, slice, _num_visibilities * 2 * sizeof(float),
                                       cudaMemcpyHostToDevice));
    INFO("Loaded visibility calibration from {} (channels {}..{})", calibration_file, _freq_offset,
         _freq_offset + _num_local_freq - 1);
}

cudaVisibilityCalibrator::~cudaVisibilityCalibrator() {
    if (_d_calibration)
        cudaFree(_d_calibration);
}

cudaEvent_t cudaVisibilityCalibrator::execute(cudaPipelineState&, const std::vector<cudaEvent_t>&) {
    pre_execute();

    const size_t frame_len = _num_visibilities * 2 * sizeof(int32_t);
    void* correlation = device.get_gpu_memory_array(_gpu_mem_correlation_matrix, gpu_frame_id,
                                                    _gpu_buffer_depth, frame_len);

    record_start_event();
    if (_d_calibration) {
        launch_apply_visibility_calibration(static_cast<int32_t*>(correlation), _d_calibration,
                                            _num_visibilities, device.getStream(cuda_stream_id));
    }
    return record_end_event();
}
