#include "cudaChartsBeamformer.hpp"

#include <mma.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
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
// fp16-weight kernel: shared weights [antenna][beam] padded to 33 beams so the staging
// writes (consecutive threads = consecutive antennas) do not all hit one bank.
constexpr unsigned int fp16_weight_stride = tiled_beam_tile + 1;

// =============================================================================
// 1. Scalar fp32 Reference Beamformer (fp16 inputs, test reference)
// =============================================================================

// Scalar fp32 reference: reads fp16 fine voltages and fp16 weights with a power-of-two scale
// per (b, f), fp32 arithmetic. Kept as the test reference for the tensor-core kernel
// (FP16_TC_ARCHITECTURE.md, WP1-WP3).
__global__ void tiled_fp16_integrated_voltage_beamformer_kernel(
    const __half2* fine_voltage, const __half* weights_fp16, const float* weight_scale,
    float* integrated_intensity, const std::size_t n_time,
    const std::size_t n_freq, const std::size_t n_ant,
    const std::size_t n_beams, const std::size_t integration_spectra) {
    extern __shared__ unsigned char shared_storage[];
    auto* shared_weights = reinterpret_cast<ComplexFloat*>(shared_storage);
    auto* shared_voltage = reinterpret_cast<float2*>(
        shared_storage + tiled_max_antennas * fp16_weight_stride * sizeof(ComplexFloat));
    auto* shared_power = reinterpret_cast<float*>(
        shared_voltage + tiled_time_tile * tiled_max_antennas);

    const std::size_t local_beam = threadIdx.x;
    const std::size_t local_time = threadIdx.y;
    const std::size_t frequency = blockIdx.y;
    const std::size_t beam = static_cast<std::size_t>(blockIdx.x) * tiled_beam_tile + local_beam;
    const std::size_t window = blockIdx.z;
    const std::size_t linear_thread = local_time * tiled_beam_tile + local_beam;

    // [f][plane][b][e] planes: element index fastest for coalesced reads.
    const std::size_t plane_stride = n_beams * n_ant;
    const __half* real_plane = weights_fp16 + frequency * 2 * plane_stride;
    const __half* imag_plane = real_plane + plane_stride;
    for (std::size_t index = linear_thread; index < n_ant * tiled_beam_tile;
         index += tiled_threads) {
        const std::size_t tile_beam = index / n_ant;
        const std::size_t antenna = index % n_ant;
        const std::size_t global_beam =
            static_cast<std::size_t>(blockIdx.x) * tiled_beam_tile + tile_beam;
        const auto shared_index = antenna * fp16_weight_stride + tile_beam;
        if (global_beam < n_beams) {
            const std::size_t w_idx = global_beam * n_ant + antenna;
            shared_weights[shared_index] = ComplexFloat{__half2float(real_plane[w_idx]),
                                                        __half2float(imag_plane[w_idx])};
        } else {
            shared_weights[shared_index] = ComplexFloat{0.0F, 0.0F};
        }
    }
    const float scale = beam < n_beams ? weight_scale[frequency * n_beams + beam] : 0.0F;
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
                    ? __half22float2(fine_voltage[(time * n_freq + frequency) * n_ant + antenna])
                    : make_float2(0.0F, 0.0F);
        }
        __syncthreads();

        float power = 0.0F;
        const std::size_t time = first_time + group_offset + local_time;
        if (beam < n_beams && time < last_time) {
            float sum_real = 0.0F;
            float sum_imag = 0.0F;
            for (std::size_t antenna = 0; antenna < n_ant; ++antenna) {
                const float2 sample = shared_voltage[local_time * n_ant + antenna];
                const float sample_real = sample.x;
                const float sample_imag = sample.y;
                const ComplexFloat weight =
                    shared_weights[antenna * fp16_weight_stride + local_beam];
                sum_real += weight.real * sample_real - weight.imag * sample_imag;
                sum_imag += weight.real * sample_imag + weight.imag * sample_real;
            }
            // Scale before squaring: s^2 alone could under/overflow float32.
            const float scaled_real = scale * sum_real;
            const float scaled_imag = scale * sum_imag;
            power = scaled_real * scaled_real + scaled_imag * scaled_imag;
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
// 1b. fp16 Tensor-Core Integrated Beamformer (WMMA m16n16k16, fp32 accumulation)
// =============================================================================
// FP16_TC_ARCHITECTURE.md WP3, dense time packing and pipelining from BEAM_256_PLAN.md B4.
// Per fine channel f and spectrum t:
//   Y[b, t] = sum_e W16[b,f,e] x[t,f,e] as real GEMMs of A = W16 (beams x elements) and
//   B = x (elements x spectra):  Yr = Wr.Xr + (-Wi).Xi,  Yi = Wr.Xi + Wi.Xr,
//   P[w,f,b] = sum over the supplied spectra of window w of (s*Yr)^2 + (s*Yi)^2.
// One block per fine channel (and block of 128 beams) with the time loop inside, so the
// 32 KiB of fp16 weights of a channel are read once per frame; warp w keeps the A fragments
// of beams 16w .. 16w + 15 of its block in registers for the whole frame. With
// nvcc 12.8, deriving the channel and beam block from blockIdx.x at run time (beam_blocks)
// also gives code that runs ~4 % faster than frequency = blockIdx.x (measured,
// BEAM_256_PLAN.md B4).
// Dense time packing: each 16-column B tile holds 16 consecutive spectra (30 tiles for 480
// spectra), and a K-spectrum window may straddle tiles. Every output column of a WMMA product
// depends only on its own B column, so (Yr, Yi) of a spectrum does not depend on how spectra
// are grouped into tiles. Spectra past n_time are staged as zeros and never summed.
// The accumulators reach the power through store_matrix_sync into a per-warp scratch, so no
// fragment-to-element mapping is assumed. Lane r < 16 owns beam 16w + r and adds
// (s*Yr)^2 + (s*Yi)^2 to its running window sum in time order, starting from 0 (the scalar
// reference's order); the sum is written when the window has K spectra or the frame ends.
// Staging: two shared-memory buffers of tc_tiles_per_round tiles. The next round's voltages
// are loaded into registers while the current round is multiplied; one __syncthreads per
// round. Per warp, the MMAs of tile g are interleaved, one K tile at a time, with the window
// sums of the full tile g - 1, so the tensor pipe has work while the lanes sum.
// Tensor-core products of fp16 values are exact; the accumulation order and rounding inside
// the MMA are hardware-defined, so results are not bit-identical to the scalar reference.
// The output is bit-identical to the earlier one-window-per-tile kernel (f7d1968c5).
namespace wmma = nvcuda::wmma;

constexpr unsigned int tc_tile = 16;             // WMMA m = n = k; spectra per B tile
constexpr unsigned int tc_warps = 8;             // one warp per 16-beam row tile, B <= 128
constexpr unsigned int tc_threads = tc_warps * 32;
constexpr unsigned int tc_max_beams = tc_warps * tc_tile;
constexpr unsigned int tc_tiles_per_round = 2;   // B tiles per stage buffer and __syncthreads
constexpr unsigned int tc_scratch_ld = tc_tile + 4; // fp32 accumulator rows (multiple of 4)

template<unsigned int KTiles>
struct TensorCoreLayout {
    static constexpr unsigned int n_ant = KTiles * tc_tile;
    // Halves per staged spectrum column of a B plane: a multiple of 8 (WMMA ldm) whose
    // 16-byte rows fall on distinct shared-memory banks for 8 consecutive columns.
    static constexpr unsigned int x_ld = n_ant + 8;
    static constexpr unsigned int plane_halves = tc_tile * x_ld; // one 16-spectrum B plane
    // One stage buffer: [tile][re, im][spectrum][element].
    static constexpr unsigned int buffer_halves = tc_tiles_per_round * 2 * plane_halves;
    static constexpr unsigned int chunks_per_row = n_ant / 4; // uint4 = 4 __half2 per load
    static constexpr unsigned int round_rows = tc_tiles_per_round * tc_tile; // spectra
    static constexpr unsigned int round_items = round_rows * chunks_per_row;
    static constexpr unsigned int items_per_thread = (round_items + tc_threads - 1) / tc_threads;
    static_assert((plane_halves * sizeof(__half)) % 32 == 0, "WMMA needs 32-byte planes");
    static_assert(tc_scratch_ld % 4 == 0, "scratch rows are read as float4");
};

struct alignas(8) Half2Pair {
    __half2 lo;
    __half2 hi;
};

// Loads one round of spectra [first_time, first_time + round_rows) of a channel into
// registers: x[t][f][e] (re, im interleaved), each spectrum row as 16-byte chunks. Spectra
// past n_time are zeros, so the fine-voltage buffer is never read past its end.
template<unsigned int KTiles>
__device__ __forceinline__ void tensor_core_stage_load(
    uint4 (&chunks)[TensorCoreLayout<KTiles>::items_per_thread], const __half2* channel_voltage,
    const std::size_t time_stride, const std::size_t first_time, const std::size_t n_time) {
    using L = TensorCoreLayout<KTiles>;
#pragma unroll
    for (unsigned int i = 0; i < L::items_per_thread; ++i) {
        const unsigned int item = threadIdx.x + i * tc_threads;
        chunks[i] = make_uint4(0U, 0U, 0U, 0U);
        if (item < L::round_items) {
            const std::size_t time = first_time + item / L::chunks_per_row;
            if (time < n_time)
                chunks[i] = __ldg(reinterpret_cast<const uint4*>(channel_voltage
                                                                 + time * time_stride)
                                  + item % L::chunks_per_row);
        }
    }
}

// Writes a loaded round to a stage buffer as separate real and imaginary planes.
template<unsigned int KTiles>
__device__ __forceinline__ void tensor_core_stage_store(
    __half* buffer, const uint4 (&chunks)[TensorCoreLayout<KTiles>::items_per_thread]) {
    using L = TensorCoreLayout<KTiles>;
#pragma unroll
    for (unsigned int i = 0; i < L::items_per_thread; ++i) {
        const unsigned int item = threadIdx.x + i * tc_threads;
        if (item < L::round_items) {
            const unsigned int chunk = item % L::chunks_per_row;
            const unsigned int row = item / L::chunks_per_row;
            union {
                uint4 raw;
                __half2 v[4];
            } chunk_data;
            static_assert(sizeof(chunk_data) == sizeof(uint4), "uint4 holds four __half2");
            chunk_data.raw = chunks[i];
            const __half2* const v = chunk_data.v;
            __half* const re_dst = buffer + (row / tc_tile) * 2 * L::plane_halves
                                   + (row % tc_tile) * L::x_ld + chunk * 4;
            *reinterpret_cast<Half2Pair*>(re_dst) =
                Half2Pair{__lows2half2(v[0], v[1]), __lows2half2(v[2], v[3])};
            *reinterpret_cast<Half2Pair*>(re_dst + L::plane_halves) =
                Half2Pair{__highs2half2(v[0], v[1]), __highs2half2(v[2], v[3])};
        }
    }
}

template<unsigned int KTiles>
__global__ void __launch_bounds__(tc_threads, 2) tensor_core_fp16_integrated_beamformer_kernel(
    const __half2* __restrict__ fine_voltage, const __half* __restrict__ weights_fp16,
    const float* __restrict__ weight_scale, float* __restrict__ integrated_intensity,
    const std::size_t n_time, const std::size_t n_freq, const std::size_t n_beams,
    const unsigned int integration_spectra, const unsigned int beam_blocks) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
    __trap(); // WMMA needs sm_70; the launcher is never used on older devices.
#else
    using L = TensorCoreLayout<KTiles>;
    __shared__ __align__(128) unsigned char stage_bytes[2 * L::buffer_halves * sizeof(__half)];
    __shared__ __align__(128) float scratch[tc_warps * 2 * tc_tile * tc_scratch_ld];
    __half* const stage = reinterpret_cast<__half*>(stage_bytes);

    // blockIdx.x = beam_blocks * f + block: the blocks of tc_max_beams beams of a channel are
    // adjacent, so they run close in time and share the channel's voltages in L2
    // (BEAM_256_PLAN.md K1; beam_blocks = 1 while B <= 128).
    const std::size_t frequency = blockIdx.x / beam_blocks;
    const unsigned int beam_block = blockIdx.x % beam_blocks;
    const unsigned int warp = threadIdx.x / 32;
    const unsigned int lane = threadIdx.x % 32;
    const std::size_t first_beam = static_cast<std::size_t>(beam_block) * tc_max_beams
                                   + static_cast<std::size_t>(warp) * tc_tile;
    const bool warp_active = first_beam < n_beams; // warp-uniform
    const unsigned int k = integration_spectra;     // any K >= 1; the launcher accepts K = 10

    // A fragments of this warp's 16 beams, row-major [b][e] straight from the global
    // [f][plane][b][e] layout (ld = E); -Wi by elementwise negation (exact in fp16).
    wmma::fragment<wmma::matrix_a, tc_tile, tc_tile, tc_tile, __half, wmma::row_major>
        w_re[KTiles], w_im[KTiles], w_im_neg[KTiles];
    float scale = 0.0F;
    if (warp_active) {
        const std::size_t plane_stride = n_beams * L::n_ant;
        const __half* re_rows =
            weights_fp16 + frequency * 2 * plane_stride + first_beam * L::n_ant;
        const __half* im_rows = re_rows + plane_stride;
#pragma unroll
        for (unsigned int kt = 0; kt < KTiles; ++kt) {
            wmma::load_matrix_sync(w_re[kt], re_rows + kt * tc_tile, L::n_ant);
            wmma::load_matrix_sync(w_im[kt], im_rows + kt * tc_tile, L::n_ant);
#pragma unroll
            for (int i = 0; i < w_im[kt].num_elements; ++i)
                w_im_neg[kt].x[i] = __hneg(w_im[kt].x[i]);
        }
        if (lane < tc_tile)
            scale = weight_scale[frequency * n_beams + first_beam + lane];
    }

    const std::size_t time_stride = n_freq * L::n_ant; // __half2 between consecutive spectra
    const __half2* channel_voltage = fine_voltage + frequency * L::n_ant;
    float* const y_re_tile = scratch + warp * 2 * tc_tile * tc_scratch_ld;
    float* const y_im_tile = y_re_tile + tc_tile * tc_scratch_ld;
    // Output of beam first_beam + lane (dereferenced by active lanes < 16 only).
    float* const out = integrated_intensity + frequency * n_beams + first_beam + lane;
    const std::size_t window_stride = n_freq * n_beams;

    const std::size_t tile_count = (n_time + tc_tile - 1) / tc_tile;
    const std::size_t round_count = (tile_count + tc_tiles_per_round - 1) / tc_tiles_per_round;

    uint4 chunks[L::items_per_thread];
    tensor_core_stage_load<KTiles>(chunks, channel_voltage, time_stride, 0, n_time);
    tensor_core_stage_store<KTiles>(stage, chunks);
    if (round_count > 1)
        tensor_core_stage_load<KTiles>(chunks, channel_voltage, time_stride, L::round_rows,
                                       n_time);

    // Running window sum of lane r < 16, carried across tiles and rounds.
    float integrated = 0.0F;
    unsigned int in_window = 0;
    std::size_t window = 0;
    auto accumulate = [&](const float y_real, const float y_imag, const std::size_t time) {
        // Scale before squaring (s^2 alone could under/overflow float32).
        const float scaled_real = scale * y_real;
        const float scaled_imag = scale * y_imag;
        const float power = scaled_real * scaled_real + scaled_imag * scaled_imag;
        integrated += power;
        if (++in_window == k || time + 1 == n_time) {
            out[window * window_stride] = integrated;
            integrated = 0.0F;
            in_window = 0;
            ++window;
        }
    };
    // Spectra 4q .. 4q + 3 of the full tile in scratch (rows are 16-byte aligned).
    auto sum_quarter = [&](const unsigned int q, const std::size_t first_time) {
        if (lane < tc_tile) {
            const float4 re4 =
                *reinterpret_cast<const float4*>(y_re_tile + lane * tc_scratch_ld + 4 * q);
            const float4 im4 =
                *reinterpret_cast<const float4*>(y_im_tile + lane * tc_scratch_ld + 4 * q);
            accumulate(re4.x, im4.x, first_time + 4 * q);
            accumulate(re4.y, im4.y, first_time + 4 * q + 1);
            accumulate(re4.z, im4.z, first_time + 4 * q + 2);
            accumulate(re4.w, im4.w, first_time + 4 * q + 3);
        }
    };
    constexpr unsigned int quarters_per_k_tile = 4 / KTiles;
    static_assert(KTiles * quarters_per_k_tile == 4, "KTiles must be 2 or 4");

    // pending: the scratch holds the full tile starting at pending_time, not yet summed.
    bool pending = false;
    std::size_t pending_time = 0;
    for (std::size_t round = 0; round < round_count; ++round) {
        // Buffer (round & 1) is complete; buffer ((round + 1) & 1), read in round - 1, is free.
        __syncthreads();
        if (round + 1 < round_count) {
            tensor_core_stage_store<KTiles>(stage + ((round + 1) & 1) * L::buffer_halves,
                                            chunks);
            if (round + 2 < round_count)
                tensor_core_stage_load<KTiles>(chunks, channel_voltage, time_stride,
                                               (round + 2) * L::round_rows, n_time);
        }
        if (!warp_active)
            continue; // warp-uniform; inactive warps still stage and reach every barrier
        const __half* const current = stage + (round & 1) * L::buffer_halves;
#pragma unroll 1
        for (unsigned int tile = 0; tile < tc_tiles_per_round; ++tile) {
            const std::size_t first_time = (round * tc_tiles_per_round + tile) * tc_tile;
            if (first_time >= n_time)
                break;
            const __half* const x_re = current + tile * 2 * L::plane_halves;
            const __half* const x_im = x_re + L::plane_halves;

            wmma::fragment<wmma::accumulator, tc_tile, tc_tile, tc_tile, float> y_re, y_im;
            wmma::fill_fragment(y_re, 0.0F);
            wmma::fill_fragment(y_im, 0.0F);
#pragma unroll
            for (unsigned int kt = 0; kt < KTiles; ++kt) {
                // B = x as elements x spectra, column-major: element (e, t) at [t*ld + e].
                wmma::fragment<wmma::matrix_b, tc_tile, tc_tile, tc_tile, __half,
                               wmma::col_major>
                    b_re, b_im;
                wmma::load_matrix_sync(b_re, x_re + kt * tc_tile, L::x_ld);
                wmma::load_matrix_sync(b_im, x_im + kt * tc_tile, L::x_ld);
                wmma::mma_sync(y_re, w_re[kt], b_re, y_re);
                wmma::mma_sync(y_re, w_im_neg[kt], b_im, y_re);
                wmma::mma_sync(y_im, w_re[kt], b_im, y_im);
                wmma::mma_sync(y_im, w_im[kt], b_re, y_im);
                if (pending) {
#pragma unroll
                    for (unsigned int qq = 0; qq < quarters_per_k_tile; ++qq)
                        sum_quarter(kt * quarters_per_k_tile + qq, pending_time);
                }
            }
            __syncwarp(); // the pending tile's scratch reads are done
            wmma::store_matrix_sync(y_re_tile, y_re, tc_scratch_ld, wmma::mem_row_major);
            wmma::store_matrix_sync(y_im_tile, y_im, tc_scratch_ld, wmma::mem_row_major);
            __syncwarp();
            if (first_time + tc_tile <= n_time) {
                pending = true;
                pending_time = first_time;
            } else { // the partial last tile: sum its supplied spectra now
                pending = false;
                if (lane < tc_tile)
                    for (std::size_t time = first_time; time < n_time; ++time)
                        accumulate(y_re_tile[lane * tc_scratch_ld + (time - first_time)],
                                   y_im_tile[lane * tc_scratch_ld + (time - first_time)], time);
                __syncwarp();
            }
        }
    }
    if (warp_active && pending) {
#pragma unroll
        for (unsigned int q = 0; q < 4; ++q)
            sum_quarter(q, pending_time);
    }
#endif
}

template<unsigned int KTiles>
TensorCoreKernelResources tensor_core_kernel_resources() {
    TensorCoreKernelResources resources;
    cudaFuncAttributes attributes{};
    check_cuda(cudaFuncGetAttributes(&attributes,
                                     tensor_core_fp16_integrated_beamformer_kernel<KTiles>),
               "tensor-core beamformer attributes");
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                   &resources.blocks_per_sm,
                   tensor_core_fp16_integrated_beamformer_kernel<KTiles>, tc_threads, 0),
               "tensor-core beamformer occupancy");
    resources.registers_per_thread = attributes.numRegs;
    resources.static_shared_bytes = attributes.sharedSizeBytes;
    resources.threads_per_block = tc_threads;
    return resources;
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
    const BeamGroupSelection beam_groups) {
    // One block per output chunk: output beam group g = beam_tile reads the computed group
    // beam_groups.source[g]. Input stride n_beams (all computed beams), output stride 16G.
    const std::size_t beam_tiles = beam_groups.count;
    const std::size_t output_beams = beam_tiles * quantization_chunk_beam;
    const std::size_t chunk = blockIdx.x;
    const std::size_t lane = threadIdx.x;
    const std::size_t beam_tile = chunk % beam_tiles;
    const std::size_t source_tile = beam_groups.source[beam_tile];
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
        const std::size_t beam = source_tile * quantization_chunk_beam + local_beam;
        valid[item] = time < n_time && frequency < n_freq && beam < n_beams;
        if (!valid[item]) {
            continue;
        }
        output_indices[item] = (time * n_freq + frequency) * output_beams
                               + beam_tile * quantization_chunk_beam + local_beam;
        values[item] = integrated_intensity[(time * n_freq + frequency) * n_beams + beam];
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
    if (n_beams == 0 || n_beams > beam_count) {
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

std::vector<double> shard_frequencies_hz(const std::size_t shard_id,
                                         const double first_coarse_center_hz,
                                         const double coarse_channel_width_hz) {
    std::vector<double> freqs(fine_frequency_channels);
    for (std::size_t f = 0; f < fine_frequency_channels; ++f) {
        freqs[f] = channel_frequency_hz(shard_id, f, first_coarse_center_hz,
                                        coarse_channel_width_hz);
    }
    return freqs;
}

Weights generate_weights(const Dimensions& dims, const std::vector<Vec3>& positions_m,
                         const std::vector<double>& frequencies_hz,
                         const std::vector<Vec3>& beam_directions) {
    if (positions_m.size() != dims.n_ant || frequencies_hz.size() != dims.n_freq
        || beam_directions.size() != dims.n_beams) {
        throw std::invalid_argument("generate_weights dimension mismatch");
    }
    Weights weights(dims.n_beams * dims.n_freq * dims.n_ant);
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
                weights[weight_index(b, f, a, dims)] = {
                    static_cast<float>(std::cos(phase)),
                    static_cast<float>(std::sin(phase)),
                };
            }
        }
    }
    return weights;
}

Fp16Weights convert_weights_fp16(const Weights& canonical_weights, const Dimensions& dims) {
    if (canonical_weights.size() != dims.n_beams * dims.n_freq * dims.n_ant) {
        throw std::invalid_argument("canonical weight count mismatch");
    }
    Fp16Weights out;
    out.values.assign(fp16_weight_count(dims), __float2half_rn(0.0F));
    out.scale.assign(dims.n_freq * dims.n_beams, 0.0F);
    double sum_sq_error = 0.0;
    std::size_t error_count = 0;
    for (std::size_t b = 0; b < dims.n_beams; ++b) {
        for (std::size_t f = 0; f < dims.n_freq; ++f) {
            const ComplexFloat* w = &canonical_weights[weight_index(b, f, 0, dims)];
            double max_sq = 0.0;
            for (std::size_t a = 0; a < dims.n_ant; ++a) {
                if (!std::isfinite(w[a].real) || !std::isfinite(w[a].imag)) {
                    throw std::invalid_argument("weights must be finite");
                }
                // Squares of float32 values are exact in double: no FMA dependence.
                const double re2 = static_cast<double>(w[a].real) * w[a].real;
                const double im2 = static_cast<double>(w[a].imag) * w[a].imag;
                max_sq = std::max(max_sq, re2 + im2);
            }
            if (max_sq == 0.0) {
                ++out.zero_beam_channels; // s = 0, W16 = 0: exact zero power
                continue;
            }
            // Smallest k with 2^(2k) >= max_sq. max_sq = m * 2^e, m in [0.5, 1).
            int e = 0;
            const double m = std::frexp(max_sq, &e);
            const int k = (m == 0.5) ? static_cast<int>(std::ceil((e - 1) / 2.0))
                                     : static_cast<int>(std::ceil(e / 2.0));
            if (k < -126 || k > 127) {
                throw std::invalid_argument(
                    "weight magnitude outside the float32 normal range of the fp16 scale");
            }
            const float s = std::ldexp(1.0F, k);
            const float inv_s = std::ldexp(1.0F, -k); // exact: W / s == W * 2^-k
            const double max_abs = std::sqrt(max_sq);
            out.scale[weight_scale_index(f, b, dims)] = s;
            for (std::size_t a = 0; a < dims.n_ant; ++a) {
                const float parts[2] = {w[a].real, w[a].imag};
                double err_sq = 0.0;
                for (std::size_t plane = 0; plane < fp16_weight_plane_count; ++plane) {
                    const float scaled = parts[plane] * inv_s;
                    const __half h = __float2half_rn(scaled);
                    std::uint16_t bits = 0;
                    std::memcpy(&bits, &h, sizeof(bits));
                    if (scaled != 0.0F) {
                        if ((bits & 0x7fffU) == 0)
                            ++out.flushed_components;
                        else if ((bits & 0x7c00U) == 0)
                            ++out.subnormal_components;
                    }
                    out.values[fp16_weight_index(f, plane, b, a, dims)] = h;
                    const double back = static_cast<double>(__half2float(h)) * s;
                    const double d = back - static_cast<double>(parts[plane]);
                    err_sq += d * d;
                }
                const double rel = std::sqrt(err_sq) / max_abs;
                out.max_relative_error = std::max(out.max_relative_error, rel);
                sum_sq_error += rel * rel;
                ++error_count;
            }
        }
    }
    out.rms_relative_error = error_count ? std::sqrt(sum_sq_error / error_count) : 0.0;
    return out;
}

// =============================================================================
// 4. Device Launch API Implementations (Beamformers + Quantization)
// =============================================================================

void launch_tiled_fp16_integrated_beamformer(const cudaStream_t stream,
                                             const __half2* fine_voltage,
                                             const __half* weights_fp16,
                                             const float* weight_scale,
                                             float* integrated_intensity,
                                             const Dimensions& dims,
                                             const std::size_t integration_spectra) {
    if (!fine_voltage || !weights_fp16 || !weight_scale || !integrated_intensity) {
        throw std::invalid_argument("CUDA tiled integrated beamformer pointers must not be null");
    }
    if (integration_spectra != 10) {
        throw std::invalid_argument("CUDA temporal integration supports K = 10 only");
    }
    (void)cudaGetLastError();

    const std::size_t window_count = integrated_time_count(dims.n_time, integration_spectra);
    const auto grid_x = (dims.n_beams + tiled_beam_tile - 1) / tiled_beam_tile;
    constexpr std::size_t shared_bytes =
        tiled_max_antennas * fp16_weight_stride * sizeof(ComplexFloat)
        + tiled_time_tile * tiled_max_antennas * sizeof(float2)
        + tiled_time_tile * tiled_beam_tile * sizeof(float);
    const dim3 grid(static_cast<unsigned int>(grid_x),
                    static_cast<unsigned int>(dims.n_freq),
                    static_cast<unsigned int>(window_count));
    const dim3 block(tiled_beam_tile, tiled_time_tile, 1);
    tiled_fp16_integrated_voltage_beamformer_kernel<<<grid, block, shared_bytes, stream>>>(
        fine_voltage, weights_fp16, weight_scale, integrated_intensity, dims.n_time, dims.n_freq,
        dims.n_ant, dims.n_beams, integration_spectra);
    check_cuda(cudaGetLastError(), "tiled_fp16_integrated_voltage_beamformer_kernel launch");
}

namespace {
void check_tensor_core_shape(const Dimensions& dims, const std::size_t integration_spectra) {
    if (integration_spectra != 10) {
        throw std::invalid_argument("CUDA temporal integration supports K = 10 only");
    }
    if (dims.n_ant != 32 && dims.n_ant != 64) {
        throw std::invalid_argument("tensor-core beamformer supports 32 or 64 elements");
    }
    if (dims.n_beams == 0 || dims.n_beams % tc_tile != 0 || dims.n_beams > tc_max_beams) {
        throw std::invalid_argument(
            "tensor-core beamformer needs a beam count that is a positive multiple of 16, "
            "at most 128");
    }
    if (dims.n_time == 0 || dims.n_freq == 0
        || dims.n_freq > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("tensor-core beamformer needs 1 <= n_freq < 2^31 and n_time > 0");
    }
}
} // namespace

void launch_tensor_core_fp16_integrated_beamformer(const cudaStream_t stream,
                                                   const __half2* fine_voltage,
                                                   const __half* weights_fp16,
                                                   const float* weight_scale,
                                                   float* integrated_intensity,
                                                   const Dimensions& dims,
                                                   const std::size_t integration_spectra) {
    if (!fine_voltage || !weights_fp16 || !weight_scale || !integrated_intensity) {
        throw std::invalid_argument("CUDA tensor-core beamformer pointers must not be null");
    }
    check_tensor_core_shape(dims, integration_spectra);
    // WMMA loads the A fragments straight from global memory (32-byte aligned rows); the
    // staging loop reads 16-byte chunks of fine voltages.
    if (reinterpret_cast<std::uintptr_t>(weights_fp16) % 32 != 0
        || reinterpret_cast<std::uintptr_t>(fine_voltage) % 16 != 0) {
        throw std::invalid_argument(
            "tensor-core beamformer needs 32-byte aligned weights and 16-byte aligned voltages");
    }
    (void)cudaGetLastError();

    // One block per (fine channel, block of tc_max_beams beams), beam blocks of a channel
    // adjacent in x; one beam block while B <= 128.
    const auto beam_blocks =
        static_cast<unsigned int>((dims.n_beams + tc_max_beams - 1) / tc_max_beams);
    if (dims.n_freq * beam_blocks > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("tensor-core beamformer grid exceeds 2^31 - 1 blocks");
    }
    const dim3 grid(static_cast<unsigned int>(dims.n_freq * beam_blocks));
    const dim3 block(tc_threads);
    const auto k = static_cast<unsigned int>(integration_spectra);
    if (dims.n_ant == 64) {
        tensor_core_fp16_integrated_beamformer_kernel<4><<<grid, block, 0, stream>>>(
            fine_voltage, weights_fp16, weight_scale, integrated_intensity, dims.n_time,
            dims.n_freq, dims.n_beams, k, beam_blocks);
    } else {
        tensor_core_fp16_integrated_beamformer_kernel<2><<<grid, block, 0, stream>>>(
            fine_voltage, weights_fp16, weight_scale, integrated_intensity, dims.n_time,
            dims.n_freq, dims.n_beams, k, beam_blocks);
    }
    check_cuda(cudaGetLastError(), "tensor_core_fp16_integrated_beamformer_kernel launch");
}

TensorCoreKernelResources tensor_core_beamformer_resources(const std::size_t n_ant) {
    if (n_ant == 64)
        return tensor_core_kernel_resources<4>();
    if (n_ant == 32)
        return tensor_core_kernel_resources<2>();
    throw std::invalid_argument("tensor-core beamformer supports 32 or 64 elements");
}

void launch_quantize_integrated_intensity(const cudaStream_t stream,
                                          const float* integrated_intensity,
                                          std::int8_t* quantized_intensity,
                                          Int8QuantizationParameters* parameters,
                                          const Dimensions& integrated_dims,
                                          const BeamGroupSelection& beam_groups) {
    if (!integrated_intensity || !quantized_intensity || !parameters) {
        throw std::invalid_argument("CUDA quantization pointers must not be null");
    }
    if (integrated_dims.n_beams % quantization_chunk_beam != 0) {
        throw std::invalid_argument("int8 quantization requires a multiple of 16 beams");
    }
    const std::size_t computed_groups = integrated_dims.n_beams / quantization_chunk_beam;
    if (beam_groups.count == 0 || beam_groups.count > computed_groups) {
        throw std::invalid_argument("int8 beam group selection count out of range");
    }
    for (std::uint32_t g = 0; g < beam_groups.count; ++g) {
        if (beam_groups.source[g] >= computed_groups
            || (g > 0 && beam_groups.source[g] <= beam_groups.source[g - 1])) {
            throw std::invalid_argument(
                "int8 beam groups must be strictly increasing and within the computed beams");
        }
    }
    (void)cudaGetLastError();

    const std::size_t f_tiles =
        (integrated_dims.n_freq + quantization_chunk_frequency - 1) / quantization_chunk_frequency;
    Dimensions written_dims = integrated_dims;
    written_dims.n_beams = beam_groups.written_beams();
    const std::size_t chunk_count = quantization_parameter_count(written_dims);
    quantize_integrated_intensity_kernel<<<static_cast<unsigned int>(chunk_count),
                                           quantization_threads, 0, stream>>>(
        integrated_intensity, quantized_intensity, parameters, integrated_dims.n_time,
        integrated_dims.n_freq, integrated_dims.n_beams, f_tiles, beam_groups);
    check_cuda(cudaGetLastError(), "quantize_integrated_intensity_kernel launch");
}

} // namespace beamformer
