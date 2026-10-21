#include "cudaChartsBeamformer.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace beamformer {
namespace {

void check_cuda(const cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
    }
}

constexpr unsigned int tiled_beam_tile = 32;
constexpr unsigned int tiled_time_tile = 8;
constexpr unsigned int tiled_threads = tiled_beam_tile * tiled_time_tile;
constexpr unsigned int tiled_max_antennas = 64;

// =============================================================================
// 1. Tiled Packed int4x2 Beamformer Kernels (Unintegrated & Fused Integrated)
// =============================================================================

__global__ void tiled_packed_voltage_beamformer_kernel(
    const std::uint8_t* packed_voltage, const ComplexFloat* weights, float* intensity,
    const std::size_t n_time, const std::size_t n_freq,
    const std::size_t n_ant, const std::size_t n_beams) {
    extern __shared__ unsigned char shared_storage[];
    auto* shared_weights = reinterpret_cast<ComplexFloat*>(shared_storage);
    auto* shared_voltage =
        shared_storage + tiled_max_antennas * tiled_beam_tile * sizeof(ComplexFloat);

    const std::size_t local_beam = threadIdx.x;
    const std::size_t local_time = threadIdx.y;
    const std::size_t frequency = blockIdx.y;
    const std::size_t beam = static_cast<std::size_t>(blockIdx.x) * tiled_beam_tile + local_beam;
    const std::size_t time = static_cast<std::size_t>(blockIdx.z) * tiled_time_tile + local_time;
    const std::size_t linear_thread = local_time * tiled_beam_tile + local_beam;

    for (std::size_t index = linear_thread; index < n_ant * tiled_beam_tile;
         index += tiled_threads) {
        const std::size_t antenna = index / tiled_beam_tile;
        const std::size_t tile_beam = index % tiled_beam_tile;
        const std::size_t global_beam =
            static_cast<std::size_t>(blockIdx.x) * tiled_beam_tile + tile_beam;
        const auto shared_index = antenna * tiled_beam_tile + tile_beam;
        if (global_beam < n_beams) {
            const std::size_t tw_idx =
                (((frequency * ((n_beams + tiled_beam_tile - 1) / tiled_beam_tile)
                   + static_cast<std::size_t>(blockIdx.x))
                  * n_ant
                  + antenna)
                     * tiled_beam_tile
                 + tile_beam);
            shared_weights[shared_index] = weights[tw_idx];
        } else {
            shared_weights[shared_index] = ComplexFloat{0.0F, 0.0F};
        }
    }

    for (std::size_t index = linear_thread; index < tiled_time_tile * n_ant;
         index += tiled_threads) {
        const std::size_t tile_time = index / n_ant;
        const std::size_t antenna = index % n_ant;
        const std::size_t global_time =
            static_cast<std::size_t>(blockIdx.z) * tiled_time_tile + tile_time;
        shared_voltage[tile_time * n_ant + antenna] =
            global_time < n_time
                ? packed_voltage[(global_time * n_freq + frequency) * n_ant + antenna]
                : 0U;
    }
    __syncthreads();

    if (time >= n_time || beam >= n_beams) {
        return;
    }

    float sum_real = 0.0F;
    float sum_imag = 0.0F;
    for (std::size_t antenna = 0; antenna < n_ant; ++antenna) {
        const ComplexInt4 packed_sample =
            unpack_complex_int4(shared_voltage[local_time * n_ant + antenna]);
        const float sample_real = static_cast<float>(packed_sample.real);
        const float sample_imag = static_cast<float>(packed_sample.imag);
        const ComplexFloat weight = shared_weights[antenna * tiled_beam_tile + local_beam];
        sum_real += weight.real * sample_real - weight.imag * sample_imag;
        sum_imag += weight.real * sample_imag + weight.imag * sample_real;
    }
    intensity[(time * n_freq + frequency) * n_beams + beam] =
        sum_real * sum_real + sum_imag * sum_imag;
}

__global__ void tiled_packed_integrated_voltage_beamformer_kernel(
    const std::uint8_t* packed_voltage, const ComplexFloat* weights,
    float* integrated_intensity, const std::size_t n_time,
    const std::size_t n_freq, const std::size_t n_ant,
    const std::size_t n_beams, const std::size_t integration_spectra) {
    extern __shared__ unsigned char shared_storage[];
    auto* shared_weights = reinterpret_cast<ComplexFloat*>(shared_storage);
    auto* shared_voltage =
        shared_storage + tiled_max_antennas * tiled_beam_tile * sizeof(ComplexFloat);
    auto* shared_power = reinterpret_cast<float*>(
        shared_voltage + tiled_time_tile * tiled_max_antennas * sizeof(std::uint8_t));

    const std::size_t local_beam = threadIdx.x;
    const std::size_t local_time = threadIdx.y;
    const std::size_t frequency = blockIdx.y;
    const std::size_t beam = static_cast<std::size_t>(blockIdx.x) * tiled_beam_tile + local_beam;
    const std::size_t window = blockIdx.z;
    const std::size_t linear_thread = local_time * tiled_beam_tile + local_beam;

    for (std::size_t index = linear_thread; index < n_ant * tiled_beam_tile;
         index += tiled_threads) {
        const std::size_t antenna = index / tiled_beam_tile;
        const std::size_t tile_beam = index % tiled_beam_tile;
        const std::size_t global_beam =
            static_cast<std::size_t>(blockIdx.x) * tiled_beam_tile + tile_beam;
        const auto shared_index = antenna * tiled_beam_tile + tile_beam;
        if (global_beam < n_beams) {
            const std::size_t tw_idx =
                (((frequency * ((n_beams + tiled_beam_tile - 1) / tiled_beam_tile)
                   + static_cast<std::size_t>(blockIdx.x))
                  * n_ant
                  + antenna)
                     * tiled_beam_tile
                 + tile_beam);
            shared_weights[shared_index] = weights[tw_idx];
        } else {
            shared_weights[shared_index] = ComplexFloat{0.0F, 0.0F};
        }
    }
    __syncthreads();

    const std::size_t first_time = window * integration_spectra;
    const std::size_t last_time = first_time + integration_spectra < n_time
                                      ? first_time + integration_spectra
                                      : n_time;
    float integrated = 0.0F;
    for (std::size_t group_offset = 0; group_offset < integration_spectra;
         group_offset += tiled_time_tile) {
        for (std::size_t index = linear_thread; index < tiled_time_tile * n_ant;
             index += tiled_threads) {
            const std::size_t tile_time = index / n_ant;
            const std::size_t antenna = index % n_ant;
            const std::size_t time = first_time + group_offset + tile_time;
            shared_voltage[tile_time * n_ant + antenna] =
                time < last_time
                    ? packed_voltage[(time * n_freq + frequency) * n_ant + antenna]
                    : 0U;
        }
        __syncthreads();

        float power = 0.0F;
        const std::size_t time = first_time + group_offset + local_time;
        if (beam < n_beams && time < last_time) {
            float sum_real = 0.0F;
            float sum_imag = 0.0F;
            for (std::size_t antenna = 0; antenna < n_ant; ++antenna) {
                const ComplexInt4 packed_sample =
                    unpack_complex_int4(shared_voltage[local_time * n_ant + antenna]);
                const float sample_real = static_cast<float>(packed_sample.real);
                const float sample_imag = static_cast<float>(packed_sample.imag);
                const ComplexFloat weight =
                    shared_weights[antenna * tiled_beam_tile + local_beam];
                sum_real += weight.real * sample_real - weight.imag * sample_imag;
                sum_imag += weight.real * sample_imag + weight.imag * sample_real;
            }
            power = sum_real * sum_real + sum_imag * sum_imag;
        }
        shared_power[local_time * tiled_beam_tile + local_beam] = power;
        __syncthreads();

        if (local_time == 0 && beam < n_beams) {
            for (std::size_t tile_time = 0; tile_time < tiled_time_tile; ++tile_time) {
                if (first_time + group_offset + tile_time < last_time) {
                    integrated += shared_power[tile_time * tiled_beam_tile + local_beam];
                }
            }
        }
        __syncthreads();
    }

    if (local_time == 0 && beam < n_beams) {
        integrated_intensity[(window * n_freq + frequency) * n_beams + beam] = integrated;
    }
}

// =============================================================================
// 2. CHIME-Style 1 x 16 x 16 int8 Chunk Quantizer Kernel
// =============================================================================

constexpr unsigned int quantization_threads = 32;
constexpr unsigned int samples_per_thread = 8;

__device__ float sane_fmin(const float lhs, const float rhs) {
    return (isnan(lhs) || isnan(rhs)) ? nanf("") : fminf(lhs, rhs);
}

__device__ float sane_fmax(const float lhs, const float rhs) {
    return (isnan(lhs) || isnan(rhs)) ? nanf("") : fmaxf(lhs, rhs);
}

__global__ void quantize_integrated_intensity_kernel(
    const float* __restrict__ integrated_intensity,
    std::int8_t* __restrict__ quantized_intensity,
    Int8QuantizationParameters* __restrict__ parameters,
    const std::size_t n_time, const std::size_t n_freq,
    const std::size_t n_beams, const std::size_t frequency_tiles,
    const std::size_t beam_tiles) {
    const std::size_t chunk = blockIdx.x;
    const std::size_t lane = threadIdx.x;
    const std::size_t beam_tile = chunk % beam_tiles;
    const std::size_t time_frequency_tile = chunk / beam_tiles;
    const std::size_t frequency_tile = time_frequency_tile % frequency_tiles;
    const std::size_t time_tile = time_frequency_tile / frequency_tiles;

    float values[samples_per_thread];
    std::size_t output_indices[samples_per_thread];
    bool valid[samples_per_thread];
    constexpr float float_max = 3.402823466e+38F;
    float minimum = float_max;
    float maximum = -float_max;
    bool non_finite = false;

    for (unsigned int item = 0; item < samples_per_thread; ++item) {
        const std::size_t sample = lane + item * quantization_threads;
        const std::size_t local_time =
            sample / (quantization_chunk_frequency * quantization_chunk_beam);
        const std::size_t local_frequency =
            (sample / quantization_chunk_beam) % quantization_chunk_frequency;
        const std::size_t local_beam = sample % quantization_chunk_beam;
        const std::size_t time = time_tile * quantization_chunk_time + local_time;
        const std::size_t frequency = frequency_tile * quantization_chunk_frequency + local_frequency;
        const std::size_t beam = beam_tile * quantization_chunk_beam + local_beam;
        valid[item] = time < n_time && frequency < n_freq && beam < n_beams;
        if (!valid[item]) {
            continue;
        }
        output_indices[item] = (time * n_freq + frequency) * n_beams + beam;
        values[item] = integrated_intensity[output_indices[item]];
        if (!isfinite(values[item])) {
            non_finite = true;
        } else {
            minimum = sane_fmin(minimum, values[item]);
            maximum = sane_fmax(maximum, values[item]);
        }
    }

    constexpr unsigned int mask = 0xffffffffU;
    non_finite = __any_sync(mask, non_finite);
    minimum = sane_fmin(minimum, __shfl_xor_sync(mask, minimum, 0x01));
    maximum = sane_fmax(maximum, __shfl_xor_sync(mask, maximum, 0x01));
    minimum = sane_fmin(minimum, __shfl_xor_sync(mask, minimum, 0x02));
    maximum = sane_fmax(maximum, __shfl_xor_sync(mask, maximum, 0x02));
    minimum = sane_fmin(minimum, __shfl_xor_sync(mask, minimum, 0x04));
    maximum = sane_fmax(maximum, __shfl_xor_sync(mask, maximum, 0x04));
    minimum = sane_fmin(minimum, __shfl_xor_sync(mask, minimum, 0x08));
    maximum = sane_fmax(maximum, __shfl_xor_sync(mask, maximum, 0x08));
    minimum = sane_fmin(minimum, __shfl_xor_sync(mask, minimum, 0x10));
    maximum = sane_fmax(maximum, __shfl_xor_sync(mask, maximum, 0x10));

    constexpr float output_min = static_cast<float>(quantized_min_code) - 0.5F;
    constexpr float output_max = static_cast<float>(quantized_max_code) + 0.5F;
    constexpr float output_range = output_max - output_min;
    Int8QuantizationParameters chunk_parameters{};
    if (!non_finite && isfinite(minimum) && isfinite(maximum)) {
        chunk_parameters.scale = (maximum - minimum) / output_range;
        chunk_parameters.offset = minimum - output_min * chunk_parameters.scale;
        if (!isfinite(chunk_parameters.offset) || !isfinite(chunk_parameters.scale)) {
            chunk_parameters = {};
        }
    }
    if (lane == 0) {
        parameters[chunk] = chunk_parameters;
    }

    for (unsigned int item = 0; item < samples_per_thread; ++item) {
        if (!valid[item]) {
            continue;
        }
        int code = 0;
        if (!isfinite(values[item])) {
            code = quantized_invalid_code;
        } else if (chunk_parameters.scale != 0.0F) {
            code = __float2int_rn(
                (values[item] - chunk_parameters.offset) / chunk_parameters.scale);
            code = max(static_cast<int>(quantized_min_code),
                       min(static_cast<int>(quantized_max_code), code));
        }
        quantized_intensity[output_indices[item]] = static_cast<std::int8_t>(code);
    }
}

} // namespace

// =============================================================================
// 3. Host Geometry & Tiled Steering Weight Generation
// =============================================================================

std::vector<Vec3> default_positions(const std::size_t n_ant, const float spacing_m) {
    if (spacing_m <= 0.0F) {
        throw std::invalid_argument("antenna spacing must be positive");
    }
    const std::size_t rows = (n_ant == 32) ? 4 : ((n_ant == 64) ? 8 : 0);
    const std::size_t cols = 8;
    if (rows == 0) {
        throw std::invalid_argument("default geometry supports 32 or 64 antennas only");
    }
    std::vector<Vec3> positions;
    positions.reserve(rows * cols);
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            positions.push_back({static_cast<float>(c) * spacing_m,
                                 static_cast<float>(r) * spacing_m, 0.0F});
        }
    }
    return positions;
}

std::vector<Vec3> default_beam_grid(const std::size_t n_beams, const float l_step, const float m) {
    if (n_beams == 0 || n_beams > maximum_beams) {
        throw std::invalid_argument("beam count must be between 1 and 128");
    }
    if (!std::isfinite(l_step) || l_step <= 0.0F) {
        throw std::invalid_argument("beam l_step must be positive and finite");
    }
    std::vector<Vec3> directions;
    directions.reserve(n_beams);
    const auto center = static_cast<long long>(n_beams / 2);
    for (std::size_t beam = 0; beam < n_beams; ++beam) {
        const float l = static_cast<float>(static_cast<long long>(beam) - center) * l_step;
        const float trans_sq = l * l + m * m;
        if (trans_sq > 1.0F) {
            throw std::invalid_argument("direction cosines exceed visible hemisphere (l^2 + m^2 > 1)");
        }
        directions.push_back({l, m, std::sqrt(1.0F - trans_sq)});
    }
    return directions;
}

std::vector<double> shard_frequencies_hz(const BeamformerInputMode mode,
                                         const std::size_t shard_id,
                                         const double first_coarse_center_hz,
                                         const double coarse_channel_width_hz) {
    const std::size_t n_freq =
        (mode == BeamformerInputMode::RawCoarse) ? local_frequency_channels : fine_frequency_channels;
    std::vector<double> freqs(n_freq);
    for (std::size_t f = 0; f < n_freq; ++f) {
        freqs[f] = channel_frequency_hz(mode, shard_id, f, first_coarse_center_hz,
                                        coarse_channel_width_hz);
    }
    return freqs;
}

TiledWeights repack_weights_to_tiled(const Weights& canonical_weights, const Dimensions& dims) {
    if (canonical_weights.size() != dims.n_beams * dims.n_freq * dims.n_ant) {
        throw std::invalid_argument("canonical weight count mismatch");
    }
    TiledWeights tiled(tiled_weight_count(dims), ComplexFloat{0.0F, 0.0F});
    for (std::size_t f = 0; f < dims.n_freq; ++f) {
        for (std::size_t b = 0; b < dims.n_beams; ++b) {
            for (std::size_t a = 0; a < dims.n_ant; ++a) {
                tiled[tiled_weight_index(f, b / tiled_weight_beam_tile, a,
                                         b % tiled_weight_beam_tile, dims)] =
                    canonical_weights[weight_index(b, f, a, dims)];
            }
        }
    }
    return tiled;
}

TiledWeights generate_tiled_weights(const Dimensions& dims, const std::vector<Vec3>& positions_m,
                                    const std::vector<double>& frequencies_hz,
                                    const std::vector<Vec3>& beam_directions) {
    if (positions_m.size() != dims.n_ant || frequencies_hz.size() != dims.n_freq
        || beam_directions.size() != dims.n_beams) {
        throw std::invalid_argument("generate_tiled_weights dimension mismatch");
    }
    TiledWeights tiled(tiled_weight_count(dims), ComplexFloat{0.0F, 0.0F});
    for (std::size_t f = 0; f < dims.n_freq; ++f) {
        const double wave_number = two_pi * frequencies_hz[f] / speed_of_light_m_per_s;
        for (std::size_t b = 0; b < dims.n_beams; ++b) {
            const auto& dir = beam_directions[b];
            for (std::size_t a = 0; a < dims.n_ant; ++a) {
                const auto& pos = positions_m[a];
                const double delay_m = static_cast<double>(pos[0]) * dir[0]
                                       + static_cast<double>(pos[1]) * dir[1]
                                       + static_cast<double>(pos[2]) * dir[2];
                const double phase = wave_number * delay_m;
                tiled[tiled_weight_index(f, b / tiled_weight_beam_tile, a,
                                         b % tiled_weight_beam_tile, dims)] = {
                    static_cast<float>(std::cos(phase)),
                    static_cast<float>(std::sin(phase)),
                };
            }
        }
    }
    return tiled;
}

// =============================================================================
// 4. Device Launch API Implementations (Tiled Only + Quantization)
// =============================================================================

void launch_tiled_packed_beamformer(const cudaStream_t stream, const std::uint8_t* packed_voltage,
                                    const ComplexFloat* tiled_weights, float* intensity,
                                    const Dimensions& dims) {
    if (!packed_voltage || !tiled_weights || !intensity) {
        throw std::invalid_argument("CUDA tiled beamformer pointers must not be null");
    }
    (void)cudaGetLastError();

    const auto grid_x = (dims.n_beams + tiled_beam_tile - 1) / tiled_beam_tile;
    const auto grid_z = (dims.n_time + tiled_time_tile - 1) / tiled_time_tile;
    constexpr std::size_t shared_bytes =
        tiled_max_antennas * tiled_beam_tile * sizeof(ComplexFloat)
        + tiled_time_tile * tiled_max_antennas * sizeof(std::uint8_t);
    const dim3 grid(static_cast<unsigned int>(grid_x),
                    static_cast<unsigned int>(dims.n_freq),
                    static_cast<unsigned int>(grid_z));
    const dim3 block(tiled_beam_tile, tiled_time_tile, 1);
    tiled_packed_voltage_beamformer_kernel<<<grid, block, shared_bytes, stream>>>(
        packed_voltage, tiled_weights, intensity, dims.n_time, dims.n_freq, dims.n_ant,
        dims.n_beams);
    check_cuda(cudaGetLastError(), "tiled_packed_voltage_beamformer_kernel launch");
}

void launch_tiled_packed_integrated_beamformer(const cudaStream_t stream,
                                               const std::uint8_t* packed_voltage,
                                               const ComplexFloat* tiled_weights,
                                               float* integrated_intensity,
                                               const Dimensions& dims,
                                               const std::size_t integration_spectra) {
    if (!packed_voltage || !tiled_weights || !integrated_intensity) {
        throw std::invalid_argument("CUDA tiled integrated beamformer pointers must not be null");
    }
    if (integration_spectra != 10 && integration_spectra != 320) {
        throw std::invalid_argument("CUDA temporal integration supports 10 or 320 spectra only");
    }
    (void)cudaGetLastError();

    const std::size_t window_count = integrated_time_count(dims.n_time, integration_spectra);
    const auto grid_x = (dims.n_beams + tiled_beam_tile - 1) / tiled_beam_tile;
    constexpr std::size_t shared_bytes =
        tiled_max_antennas * tiled_beam_tile * sizeof(ComplexFloat)
        + tiled_time_tile * tiled_max_antennas * sizeof(std::uint8_t)
        + tiled_time_tile * tiled_beam_tile * sizeof(float);
    const dim3 grid(static_cast<unsigned int>(grid_x),
                    static_cast<unsigned int>(dims.n_freq),
                    static_cast<unsigned int>(window_count));
    const dim3 block(tiled_beam_tile, tiled_time_tile, 1);
    tiled_packed_integrated_voltage_beamformer_kernel<<<grid, block, shared_bytes, stream>>>(
        packed_voltage, tiled_weights, integrated_intensity, dims.n_time, dims.n_freq, dims.n_ant,
        dims.n_beams, integration_spectra);
    check_cuda(cudaGetLastError(), "tiled_packed_integrated_voltage_beamformer_kernel launch");
}

void launch_quantize_integrated_intensity(const cudaStream_t stream,
                                          const float* integrated_intensity,
                                          std::int8_t* quantized_intensity,
                                          Int8QuantizationParameters* parameters,
                                          const Dimensions& integrated_dims) {
    if (!integrated_intensity || !quantized_intensity || !parameters) {
        throw std::invalid_argument("CUDA quantization pointers must not be null");
    }
    (void)cudaGetLastError();

    const std::size_t f_tiles =
        (integrated_dims.n_freq + quantization_chunk_frequency - 1) / quantization_chunk_frequency;
    const std::size_t b_tiles =
        (integrated_dims.n_beams + quantization_chunk_beam - 1) / quantization_chunk_beam;
    const std::size_t chunk_count = quantization_parameter_count(integrated_dims);
    quantize_integrated_intensity_kernel<<<static_cast<unsigned int>(chunk_count),
                                           quantization_threads, 0, stream>>>(
        integrated_intensity, quantized_intensity, parameters, integrated_dims.n_time,
        integrated_dims.n_freq, integrated_dims.n_beams, f_tiles, b_tiles);
    check_cuda(cudaGetLastError(), "quantize_integrated_intensity_kernel launch");
}

} // namespace beamformer
