#include "cudaAntennaMask.hpp"
#include "cudaUtils.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>

namespace kotekan {

// ============================================================================
// Device Helpers (Fast Branchless Bitfield Extraction)
// ============================================================================

template <bool SWAPPED_WITH_OFFSET = false>
__device__ __forceinline__ void decode_int4x2_fast(std::uint32_t byte_val, int& re, int& im) {
    if constexpr (SWAPPED_WITH_OFFSET) {
#if defined(__CUDA_ARCH__)
        asm("bfe.u32 %0, %1, 4, 4;" : "=r"(re) : "r"(byte_val));
        asm("bfe.u32 %0, %1, 0, 4;" : "=r"(im) : "r"(byte_val));
        re -= 8;
        im -= 8;
#else
        re = static_cast<int>((byte_val >> 4) & 0x0Fu) - 8;
        im = static_cast<int>(byte_val & 0x0Fu) - 8;
#endif
    } else {
#if defined(__CUDA_ARCH__)
        asm("bfe.s32 %0, %1, 0, 4;" : "=r"(re) : "r"(byte_val));
        asm("bfe.s32 %0, %1, 4, 4;" : "=r"(im) : "r"(byte_val));
#else
        re = (static_cast<int>(byte_val) << 28) >> 28;
        im = (static_cast<int>(byte_val) << 24) >> 28;
#endif
    }
}

// ============================================================================
// Multi-Moment Coalesced Inspection Kernel
// ============================================================================

template <int BLOCK_ANT, int BLOCK_SPEC, bool SWAPPED_WITH_OFFSET = false>
__global__ void __launch_bounds__(512, 2)
inspect_antenna_moments_kernel(
    const std::uint8_t* __restrict__ raw_bytes,
    AntennaRawStats* __restrict__ raw_stats,
    float* __restrict__ antenna_powers,
    std::uint32_t* __restrict__ antenna_clips,
    std::size_t num_inspected_spectra,
    std::size_t n_elem,
    std::size_t sample_stride)
{
    const unsigned int ant = threadIdx.x;
    const unsigned int spec_lane = threadIdx.y;
    if (ant >= n_elem) return;

    unsigned long long local_re2 = 0ULL;
    unsigned long long local_im2 = 0ULL;
    unsigned long long local_p2  = 0ULL;
    long long local_re           = 0LL;
    long long local_im           = 0LL;
    std::uint32_t local_clips    = 0U;

    const std::size_t global_spec_start =
        static_cast<std::size_t>(blockIdx.x) * BLOCK_SPEC + spec_lane;
    const std::size_t global_spec_step =
        static_cast<std::size_t>(gridDim.x) * BLOCK_SPEC;

    for (std::size_t s_idx = global_spec_start; s_idx < num_inspected_spectra; s_idx += global_spec_step) {
        const std::size_t s = s_idx * sample_stride;
        const std::uint32_t byte_val = __ldg(&raw_bytes[s * n_elem + ant]);

        int re, im;
        decode_int4x2_fast<SWAPPED_WITH_OFFSET>(byte_val, re, im);

        const int r2 = re * re;
        const int i2 = im * im;
        const int p  = r2 + i2;

        local_re  += re;
        local_im  += im;
        local_re2 += static_cast<unsigned int>(r2);
        local_im2 += static_cast<unsigned int>(i2);
        local_p2  += static_cast<unsigned int>(p * p);

        local_clips += static_cast<std::uint32_t>((re == 7) | (re == -8))
                     + static_cast<std::uint32_t>((im == 7) | (im == -8));
    }

    // Intra-block reduction across BLOCK_SPEC rows in shared memory
    if constexpr (BLOCK_SPEC > 1) {
        __shared__ unsigned long long s_re2[BLOCK_SPEC][BLOCK_ANT];
        __shared__ unsigned long long s_im2[BLOCK_SPEC][BLOCK_ANT];
        __shared__ unsigned long long s_p2[BLOCK_SPEC][BLOCK_ANT];
        __shared__ long long s_re[BLOCK_SPEC][BLOCK_ANT];
        __shared__ long long s_im[BLOCK_SPEC][BLOCK_ANT];
        __shared__ std::uint32_t s_clips[BLOCK_SPEC][BLOCK_ANT];

        s_re2[spec_lane][ant]   = local_re2;
        s_im2[spec_lane][ant]   = local_im2;
        s_p2[spec_lane][ant]    = local_p2;
        s_re[spec_lane][ant]    = local_re;
        s_im[spec_lane][ant]    = local_im;
        s_clips[spec_lane][ant] = local_clips;
        __syncthreads();

        if (spec_lane == 0) {
            #pragma unroll
            for (int r = 1; r < BLOCK_SPEC; ++r) {
                local_re2   += s_re2[r][ant];
                local_im2   += s_im2[r][ant];
                local_p2    += s_p2[r][ant];
                local_re    += s_re[r][ant];
                local_im    += s_im[r][ant];
                local_clips += s_clips[r][ant];
            }
        }
    }

    if (spec_lane == 0) {
        if (raw_stats != nullptr) {
            atomicAdd(&raw_stats[ant].sum_re2, local_re2);
            atomicAdd(&raw_stats[ant].sum_im2, local_im2);
            atomicAdd(&raw_stats[ant].sum_p2,  local_p2);
            atomicAdd(reinterpret_cast<unsigned long long*>(&raw_stats[ant].sum_re),
                      static_cast<unsigned long long>(local_re));
            atomicAdd(reinterpret_cast<unsigned long long*>(&raw_stats[ant].sum_im),
                      static_cast<unsigned long long>(local_im));
            atomicAdd(&raw_stats[ant].clips, local_clips);
        }
        const unsigned long long total_p = local_re2 + local_im2;
        if (antenna_powers != nullptr && total_p > 0ULL && num_inspected_spectra > 0) {
            const float norm_p =
                static_cast<float>(static_cast<double>(total_p) / static_cast<double>(num_inspected_spectra));
            atomicAdd(&antenna_powers[ant], norm_p);
        }
        if (antenna_clips != nullptr && local_clips > 0U) {
            atomicAdd(&antenna_clips[ant], local_clips);
        }
    }
}

// ============================================================================
// Coalesced Blanking & Extraction Kernels
// ============================================================================

__global__ void zero_bad_antennas_coalesced_kernel(
    std::uint8_t* __restrict__ raw_bytes,
    const int* __restrict__ bad_antennas,
    int num_bad_antennas,
    std::size_t total_spectra,
    std::size_t n_ant)
{
    __shared__ int s_bad[MAX_MASK_ANTENNAS];
    const unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const unsigned int block_threads = blockDim.x * blockDim.y;

    for (unsigned int i = tid; i < static_cast<unsigned int>(num_bad_antennas); i += block_threads) {
        s_bad[i] = bad_antennas[i];
    }
    __syncthreads();

    // 2D mapping: threadIdx.x indexes bad antennas, threadIdx.y indexes spectra
    const std::size_t s = static_cast<std::size_t>(blockIdx.x) * blockDim.y + threadIdx.y;
    if (s >= total_spectra) return;

    std::uint8_t* row_ptr = raw_bytes + s * n_ant;
    for (int i = static_cast<int>(threadIdx.x); i < num_bad_antennas; i += static_cast<int>(blockDim.x)) {
        row_ptr[s_bad[i]] = 0U;
    }
}

__global__ void extract_alive_antennas_coalesced_kernel(
    const std::uint8_t* __restrict__ in_bytes,
    std::uint8_t* __restrict__ out_bytes,
    const int* __restrict__ alive_antennas,
    int num_to_extract,
    int max_alive_antennas,
    std::size_t total_spectra,
    std::size_t n_ant)
{
    __shared__ int s_alive[MAX_MASK_ANTENNAS];
    const unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const unsigned int block_threads = blockDim.x * blockDim.y;

    for (unsigned int i = tid; i < static_cast<unsigned int>(num_to_extract); i += block_threads) {
        s_alive[i] = alive_antennas[i];
    }
    __syncthreads();

    // One warp (blockDim.x = 32) per spectrum s -> 100% coalesced writes to out_row
    const std::size_t s = static_cast<std::size_t>(blockIdx.x) * blockDim.y + threadIdx.y;
    if (s >= total_spectra) return;

    const std::uint8_t* in_row = in_bytes + s * n_ant;
    std::uint8_t* out_row = out_bytes + s * static_cast<std::size_t>(max_alive_antennas);

    for (int i = static_cast<int>(threadIdx.x); i < max_alive_antennas; i += static_cast<int>(blockDim.x)) {
        out_row[i] = (i < num_to_extract) ? __ldg(&in_row[s_alive[i]]) : 0U;
    }
}

__global__ void inject_faults_coalesced_kernel(
    std::uint8_t* __restrict__ raw_bytes,
    const std::uint8_t* __restrict__ fault_types,
    std::size_t total_spectra,
    std::size_t n_ant,
    unsigned int seed)
{
    __shared__ std::uint8_t s_faults[MAX_MASK_ANTENNAS];
    const unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const unsigned int block_threads = blockDim.x * blockDim.y;

    for (unsigned int a = tid; a < static_cast<unsigned int>(n_ant); a += block_threads) {
        s_faults[a] = fault_types[a];
    }
    __syncthreads();

    const std::size_t s = static_cast<std::size_t>(blockIdx.x) * blockDim.y + threadIdx.y;
    if (s >= total_spectra) return;

    std::uint8_t* row_ptr = raw_bytes + s * n_ant;
    for (std::size_t a = threadIdx.x; a < n_ant; a += blockDim.x) {
        const std::uint8_t ft = s_faults[a];
        if (ft == 1U) {
            // Dead antenna
            row_ptr[a] = 0U;
        } else if (ft == 2U) {
            // Saturated ADC rails (+7 / -8)
            std::uint32_t h = seed + static_cast<unsigned int>(s) * 31U + static_cast<unsigned int>(a) * 104729U;
            h = (h ^ (h >> 16U)) * 0x45d9f3bU;
            const int r = (h & 1U) ? 7 : -8;
            const int i = (h & 2U) ? 7 : -8;
            row_ptr[a] = static_cast<std::uint8_t>((r & 0x0FU) | ((i & 0x0FU) << 4U));
        } else if (ft == 3U) {
            // Stuck bit / DC lock (constant 3 + 2j -> power = 13, 0 clips)
            row_ptr[a] = static_cast<std::uint8_t>((3 & 0x0FU) | ((2 & 0x0FU) << 4U));
        } else if (ft == 4U) {
            // Impulsive RFI (1% duty-cycle rail bursts -> normal mean power, high kurtosis)
            if ((s & 127U) == 0U) {
                row_ptr[a] = static_cast<std::uint8_t>((7 & 0x0FU) | ((7 & 0x0FU) << 4U));
            }
        }
    }
}

// ============================================================================
// Launch Wrappers
// ============================================================================

void launch_inspect_antenna_moments(
    const int4x2_t* __restrict__ d_voltages,
    AntennaRawStats* __restrict__ d_raw_stats,
    float* __restrict__ d_antenna_powers,
    std::uint32_t* __restrict__ d_antenna_clips,
    std::size_t n_time,
    std::size_t n_freq,
    std::size_t n_elem,
    std::size_t sample_stride,
    cudaStream_t stream,
    bool is_swapped_withoffset)
{
    const std::size_t total_spectra = n_time * n_freq;
    if (total_spectra == 0 || n_elem == 0) return;
    if (sample_stride < 1) sample_stride = 1;

    if (d_raw_stats != nullptr) {
        cudaMemsetAsync(d_raw_stats, 0, n_elem * sizeof(AntennaRawStats), stream);
    }
    if (d_antenna_powers != nullptr) {
        cudaMemsetAsync(d_antenna_powers, 0, n_elem * sizeof(float), stream);
    }
    if (d_antenna_clips != nullptr) {
        cudaMemsetAsync(d_antenna_clips, 0, n_elem * sizeof(std::uint32_t), stream);
    }

    const std::size_t num_inspected_spectra = (total_spectra + sample_stride - 1) / sample_stride;
    const auto* raw_bytes = reinterpret_cast<const std::uint8_t*>(d_voltages);

    // 256 blocks provides full wave saturation across all SMs
    constexpr unsigned int NUM_BLOCKS = 256;

    auto dispatch_kernel = [&](auto fmt_tag) {
        constexpr bool F_O = decltype(fmt_tag)::value;
        if (n_elem <= 32) {
            const dim3 block(32, 8);
            inspect_antenna_moments_kernel<32, 8, F_O><<<NUM_BLOCKS, block, 0, stream>>>(
                raw_bytes, d_raw_stats, d_antenna_powers, d_antenna_clips,
                num_inspected_spectra, n_elem, sample_stride);
        } else if (n_elem <= 64) {
            const dim3 block(64, 4);
            inspect_antenna_moments_kernel<64, 4, F_O><<<NUM_BLOCKS, block, 0, stream>>>(
                raw_bytes, d_raw_stats, d_antenna_powers, d_antenna_clips,
                num_inspected_spectra, n_elem, sample_stride);
        } else if (n_elem <= 128) {
            const dim3 block(128, 2);
            inspect_antenna_moments_kernel<128, 2, F_O><<<NUM_BLOCKS, block, 0, stream>>>(
                raw_bytes, d_raw_stats, d_antenna_powers, d_antenna_clips,
                num_inspected_spectra, n_elem, sample_stride);
        } else if (n_elem <= 256) {
            const dim3 block(256, 1);
            inspect_antenna_moments_kernel<256, 1, F_O><<<NUM_BLOCKS, block, 0, stream>>>(
                raw_bytes, d_raw_stats, d_antenna_powers, d_antenna_clips,
                num_inspected_spectra, n_elem, sample_stride);
        } else {
            const dim3 block(512, 1);
            inspect_antenna_moments_kernel<512, 1, F_O><<<NUM_BLOCKS, block, 0, stream>>>(
                raw_bytes, d_raw_stats, d_antenna_powers, d_antenna_clips,
                num_inspected_spectra, n_elem, sample_stride);
        }
    };

    if (is_swapped_withoffset) {
        dispatch_kernel(std::true_type{});
    } else {
        dispatch_kernel(std::false_type{});
    }
}

void launch_inspect_antenna_health(
    const int4x2_t* __restrict__ d_voltages,
    float* __restrict__ d_antenna_powers,
    std::uint32_t* __restrict__ d_antenna_clips,
    std::size_t n_time,
    std::size_t n_freq,
    std::size_t n_ant,
    std::size_t sample_stride,
    cudaStream_t stream,
    bool is_swapped_withoffset)
{
    launch_inspect_antenna_moments(
        d_voltages, nullptr, d_antenna_powers, d_antenna_clips,
        n_time, n_freq, n_ant, sample_stride, stream, is_swapped_withoffset);
}

void launch_zero_bad_antennas(
    int4x2_t* __restrict__ d_voltages,
    const int* __restrict__ d_bad_antennas,
    int num_bad_antennas,
    std::size_t total_spectra,
    std::size_t n_ant,
    cudaStream_t stream)
{
    if (num_bad_antennas <= 0 || total_spectra == 0) return;

    if (static_cast<std::size_t>(num_bad_antennas) >= n_ant) {
        cudaMemsetAsync(d_voltages, 0, total_spectra * n_ant * sizeof(std::uint8_t), stream);
        return;
    }

    const unsigned int threads_x = (num_bad_antennas <= 8) ? 8U : ((num_bad_antennas <= 16) ? 16U : 32U);
    const unsigned int threads_y = 256U / threads_x;
    const dim3 block(threads_x, threads_y);
    const unsigned int num_blocks =
        static_cast<unsigned int>((total_spectra + threads_y - 1) / threads_y);

    zero_bad_antennas_coalesced_kernel<<<num_blocks, block, 0, stream>>>(
        reinterpret_cast<std::uint8_t*>(d_voltages),
        d_bad_antennas, num_bad_antennas, total_spectra, n_ant);
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
    if (num_to_extract < 0) num_to_extract = 0;
    if (num_to_extract > static_cast<int>(MAX_MASK_ANTENNAS)) {
        num_to_extract = static_cast<int>(MAX_MASK_ANTENNAS);
    }

    // 32 threads (1 warp) per spectrum row -> coalesced reads & writes
    constexpr unsigned int WARP_SIZE = 32U;
    constexpr unsigned int SPECTRA_PER_BLOCK = 8U;
    const dim3 block(WARP_SIZE, SPECTRA_PER_BLOCK);
    const unsigned int num_blocks =
        static_cast<unsigned int>((total_spectra + SPECTRA_PER_BLOCK - 1) / SPECTRA_PER_BLOCK);

    extract_alive_antennas_coalesced_kernel<<<num_blocks, block, 0, stream>>>(
        reinterpret_cast<const std::uint8_t*>(d_voltages),
        reinterpret_cast<std::uint8_t*>(d_alive_voltages),
        d_alive_antennas, num_to_extract, max_alive_antennas, total_spectra, n_ant);
}

void launch_inject_faults(
    int4x2_t* __restrict__ d_voltages,
    const std::uint8_t* __restrict__ d_fault_types,
    std::size_t total_spectra,
    std::size_t n_ant,
    unsigned int seed,
    cudaStream_t stream)
{
    if (total_spectra == 0 || n_ant == 0) return;

    constexpr unsigned int WARP_SIZE = 32U;
    constexpr unsigned int SPECTRA_PER_BLOCK = 8U;
    const dim3 block(WARP_SIZE, SPECTRA_PER_BLOCK);
    const unsigned int num_blocks =
        static_cast<unsigned int>((total_spectra + SPECTRA_PER_BLOCK - 1) / SPECTRA_PER_BLOCK);

    inject_faults_coalesced_kernel<<<num_blocks, block, 0, stream>>>(
        reinterpret_cast<std::uint8_t*>(d_voltages),
        d_fault_types, total_spectra, n_ant, seed);
}

} // namespace kotekan
