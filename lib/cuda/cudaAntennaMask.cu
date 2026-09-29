#include "cudaAntennaMask.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace kotekan {

// ============================================================================
// Device Helpers
// ============================================================================

__device__ __forceinline__ void decode_int4x2(uint8_t byte_val, int& re, int& im) {
    // low 4 bits: sign-extended
    int r = static_cast<int>(byte_val & 0x0FU);
    if (r & 0x08) r -= 16;
    // high 4 bits: sign-extended
    int i = static_cast<int>((byte_val >> 4U) & 0x0FU);
    if (i & 0x08) i -= 16;
    re = r;
    im = i;
}

// ============================================================================
// Inspection Kernels
// ============================================================================

__global__ void inspect_antenna_health_kernel(
    const int4x2_t* __restrict__ voltages,
    float* __restrict__ antenna_powers,
    uint32_t* __restrict__ antenna_clips,
    size_t total_spectra,
    size_t n_ant,
    size_t sample_stride)
{
    size_t ant = threadIdx.x;
    if (ant >= n_ant) return;

    size_t num_inspected_spectra = (total_spectra + sample_stride - 1) / sample_stride;
    size_t chunk = (num_inspected_spectra + gridDim.x - 1) / gridDim.x;
    size_t start_s_idx = blockIdx.x * chunk;
    size_t end_s_idx = (start_s_idx + chunk < num_inspected_spectra) ? (start_s_idx + chunk) : num_inspected_spectra;

    float local_power = 0.0f;
    uint32_t local_clips = 0;

    const uint8_t* raw_bytes = reinterpret_cast<const uint8_t*>(voltages);

    for (size_t s_idx = start_s_idx; s_idx < end_s_idx; ++s_idx) {
        size_t s = s_idx * sample_stride;
        size_t offset = s * n_ant + ant;
        uint8_t byte_val = raw_bytes[offset];

        int re, im;
        decode_int4x2(byte_val, re, im);

        local_power += static_cast<float>(re * re + im * im);
        if (re == 7 || re == -8) {
            local_clips++;
        }
        if (im == 7 || im == -8) {
            local_clips++;
        }
    }

    if (local_power > 0.0f) {
        atomicAdd(&antenna_powers[ant], local_power);
    }
    if (local_clips > 0) {
        atomicAdd(&antenna_clips[ant], local_clips);
    }
}

__global__ void normalize_antenna_metrics_kernel(
    float* __restrict__ antenna_powers,
    size_t num_inspected_spectra,
    size_t n_ant)
{
    size_t ant = blockIdx.x * blockDim.x + threadIdx.x;
    if (ant >= n_ant) return;
    if (num_inspected_spectra > 0) {
        antenna_powers[ant] /= static_cast<float>(num_inspected_spectra);
    }
}

// ============================================================================
// Blanking Kernels
// ============================================================================

__global__ void zero_bad_antennas_kernel(
    int4x2_t* __restrict__ voltages,
    const int* __restrict__ bad_antennas,
    int num_bad_antennas,
    size_t total_spectra,
    size_t n_ant)
{
    size_t s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= total_spectra) return;

    size_t base_offset = s * n_ant;
    uint8_t* raw_bytes = reinterpret_cast<uint8_t*>(voltages);
    for (int i = 0; i < num_bad_antennas; ++i) {
        int ant = bad_antennas[i];
        raw_bytes[base_offset + ant] = 0;
    }
}

// ============================================================================
// Launch Wrappers
// ============================================================================

void launch_inspect_antenna_health(
    const int4x2_t* __restrict__ d_voltages,
    float* __restrict__ d_antenna_powers,
    std::uint32_t* __restrict__ d_antenna_clips,
    std::size_t n_time,
    std::size_t n_freq,
    std::size_t n_ant,
    std::size_t sample_stride,
    cudaStream_t stream)
{
    const std::size_t total_spectra = n_time * n_freq;
    if (sample_stride < 1) sample_stride = 1;

    cudaMemsetAsync(d_antenna_powers, 0, n_ant * sizeof(float), stream);
    cudaMemsetAsync(d_antenna_clips, 0, n_ant * sizeof(std::uint32_t), stream);

    // Use 128 blocks along time-freq dimension, 256 threads per block (one thread per antenna)
    const unsigned int num_blocks = 128;
    const unsigned int threads_per_block = static_cast<unsigned int>(n_ant <= 256 ? 256 : n_ant);

    inspect_antenna_health_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        d_voltages, d_antenna_powers, d_antenna_clips, total_spectra, n_ant, sample_stride);

    // Normalize power by inspected count
    const std::size_t num_inspected_spectra = (total_spectra + sample_stride - 1) / sample_stride;
    const unsigned int norm_threads = 256;
    const unsigned int norm_blocks = static_cast<unsigned int>((n_ant + norm_threads - 1) / norm_threads);
    normalize_antenna_metrics_kernel<<<norm_blocks, norm_threads, 0, stream>>>(
        d_antenna_powers, num_inspected_spectra, n_ant);
}

void launch_zero_bad_antennas(
    int4x2_t* __restrict__ d_voltages,
    const int* __restrict__ d_bad_antennas,
    int num_bad_antennas,
    std::size_t total_spectra,
    std::size_t n_ant,
    cudaStream_t stream)
{
    if (num_bad_antennas <= 0) return;

    if (static_cast<std::size_t>(num_bad_antennas) >= n_ant) {
        cudaMemsetAsync(d_voltages, 0, total_spectra * n_ant * sizeof(uint8_t), stream);
        return;
    }

    const unsigned int threads_per_block = 256;
    const unsigned int num_blocks = static_cast<unsigned int>((total_spectra + threads_per_block - 1) / threads_per_block);

    zero_bad_antennas_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        d_voltages, d_bad_antennas, num_bad_antennas, total_spectra, n_ant);
}

__global__ void extract_alive_antennas_kernel(
    const int4x2_t* __restrict__ voltages,
    int4x2_t* __restrict__ alive_voltages,
    const int* __restrict__ alive_antennas,
    int num_to_extract,
    int max_alive_antennas,
    size_t total_spectra,
    size_t n_ant)
{
    size_t s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= total_spectra) return;

    const uint8_t* in_bytes = reinterpret_cast<const uint8_t*>(voltages) + s * n_ant;
    uint8_t* out_bytes = reinterpret_cast<uint8_t*>(alive_voltages) + s * max_alive_antennas;

    for (int i = 0; i < num_to_extract; ++i) {
        int ant = alive_antennas[i];
        out_bytes[i] = in_bytes[ant];
    }
    for (int i = num_to_extract; i < max_alive_antennas; ++i) {
        out_bytes[i] = 0;
    }
}

void launch_extract_alive_antennas(
    const int4x2_t* __restrict__ d_voltages,
    int4x2_t* __restrict__ d_alive_voltages,
    const int* __restrict__ d_alive_antennas,
    int num_to_extract,
    int max_alive_antennas,
    std::size_t total_spectra,
    std::size_t n_ant,
    cudaStream_t stream)
{
    if (max_alive_antennas <= 0 || total_spectra == 0) return;

    const unsigned int threads_per_block = 256;
    const unsigned int num_blocks = static_cast<unsigned int>((total_spectra + threads_per_block - 1) / threads_per_block);

    extract_alive_antennas_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        d_voltages, d_alive_voltages, d_alive_antennas, num_to_extract, max_alive_antennas, total_spectra, n_ant);
}

__global__ void inject_faults_kernel(
    int4x2_t* __restrict__ voltages,
    const uint8_t* __restrict__ fault_types,
    size_t total_spectra,
    size_t n_ant,
    unsigned int seed)
{
    size_t s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= total_spectra) return;

    size_t base = s * n_ant;
    uint8_t* raw = reinterpret_cast<uint8_t*>(voltages);

    for (size_t a = 0; a < n_ant; ++a) {
        uint8_t ft = fault_types[a];
        if (ft == 1) {
            raw[base + a] = 0;
        } else if (ft == 2) {
            uint32_t h = (seed + static_cast<unsigned int>(s) * 31U + static_cast<unsigned int>(a) * 104729U);
            h = (h ^ (h >> 16U)) * 0x45d9f3bU;
            int r = (h & 1U) ? 7 : -8;
            int i = (h & 2U) ? 7 : -8;
            raw[base + a] = static_cast<uint8_t>((r & 0x0FU) | ((i & 0x0FU) << 4U));
        }
    }
}

void launch_inject_faults(
    int4x2_t* __restrict__ d_voltages,
    const std::uint8_t* __restrict__ d_fault_types,
    std::size_t total_spectra,
    std::size_t n_ant,
    unsigned int seed,
    cudaStream_t stream)
{
    const unsigned int tpb = 256;
    const unsigned int nb = static_cast<unsigned int>((total_spectra + tpb - 1) / tpb);
    inject_faults_kernel<<<nb, tpb, 0, stream>>>(
        d_voltages, d_fault_types, total_spectra, n_ant, seed);
}

} // namespace kotekan
