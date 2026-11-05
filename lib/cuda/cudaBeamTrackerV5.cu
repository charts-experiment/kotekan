#include "cudaBeamTrackerV5.hpp"
#include "cudaUtils.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace kotekan {

namespace {

constexpr double speed_of_light_m_per_s = charts::constants::speed_of_light_m_per_s;
constexpr double two_pi = charts::constants::two_pi;

__device__ __forceinline__ float2 unpack_int4_fast(const std::uint32_t byte_val) {
#if defined(__CUDA_ARCH__)
    int r, i;
    asm("bfe.s32 %0, %1, 0, 4;" : "=r"(r) : "r"(byte_val));
    asm("bfe.s32 %0, %1, 4, 4;" : "=r"(i) : "r"(byte_val));
    return make_float2(__int2float_rn(r), __int2float_rn(i));
#else
    const int r = (static_cast<int>(byte_val) << 28) >> 28;
    const int i = ((static_cast<int>(byte_val) << 24) >> 28);
    return make_float2(static_cast<float>(r), static_cast<float>(i));
#endif
}

__device__ __forceinline__ void tracker_weight_v5(const float3 position,
                                                  const float3 direction,
                                                  const double wave_number,
                                                  float* weight_real,
                                                  float* weight_imag) {
    const double delay_m = static_cast<double>(position.x) * direction.x
                         + static_cast<double>(position.y) * direction.y
                         + static_cast<double>(position.z) * direction.z;
    const double phase = wave_number * delay_m;
    double s, c;
    sincos(phase, &s, &c);
    *weight_real = static_cast<float>(c);
    *weight_imag = static_cast<float>(s);
}

template <int N_ANT, int TIME_UNROLL>
__global__ void __launch_bounds__(128, 4)
tracker_v5_multibeam_kernel(
    float2* __restrict__ voltages,
    const float* __restrict__ window_directions,
    const double* __restrict__ wavenumbers,
    const std::uint8_t* __restrict__ packed,
    const std::uint8_t* __restrict__ antenna_mask,
    const float3* __restrict__ antenna_positions,
    const std::size_t n_time,
    const std::size_t n_freq,
    const std::size_t integration_spectra,
    const std::size_t time_chunk_size,
    const std::size_t chunks_per_window,
    const std::size_t num_active_beams,
    const std::size_t max_beams_stride,
    const std::size_t total_warps) {

    constexpr unsigned int ANT_PER_LANE = static_cast<unsigned int>(N_ANT / 32);
    constexpr unsigned int full_mask = 0xFFFFFFFFu;

    const unsigned int lane = threadIdx.x;
    const unsigned int warp_in_block = threadIdx.y;
    const std::size_t warp_id =
        static_cast<std::size_t>(blockIdx.x) * blockDim.y + warp_in_block;

    if (warp_id >= total_warps) {
        return;
    }

    const std::size_t beam_idx = warp_id % num_active_beams;
    const std::size_t rest = warp_id / num_active_beams;
    const std::size_t freq = rest % n_freq;
    const std::size_t chunk_global = rest / n_freq;
    const std::size_t chunk_in_win = chunk_global % chunks_per_window;
    const std::size_t window = chunk_global / chunks_per_window;

    const std::size_t dir_idx = (window * num_active_beams + beam_idx) * 3;
    const float3 direction = make_float3(__ldg(&window_directions[dir_idx + 0]),
                                         __ldg(&window_directions[dir_idx + 1]),
                                         __ldg(&window_directions[dir_idx + 2]));
    const double wave_number = __ldg(&wavenumbers[freq]);

    // Precompute steering weights and pre-negated imaginary components
    float w_r[ANT_PER_LANE];
    float w_i[ANT_PER_LANE];
    float nw_i[ANT_PER_LANE];

    #pragma unroll
    for (unsigned int a = 0; a < ANT_PER_LANE; ++a) {
        const unsigned int elem = lane + a * 32U;
        if (antenna_mask != nullptr && __ldg(&antenna_mask[elem]) == 0) {
            // Dead / masked antenna: zero out steering weights to completely isolate dead element
            w_r[a] = 0.0F;
            w_i[a] = 0.0F;
            nw_i[a] = 0.0F;
        } else {
            const float3 pos = antenna_positions[elem];
            tracker_weight_v5(pos, direction, wave_number, &w_r[a], &w_i[a]);
            nw_i[a] = -w_i[a];
        }
    }

    const std::size_t t_win_start = window * integration_spectra;
    const std::size_t t_win_end = (t_win_start + integration_spectra < n_time)
                                     ? (t_win_start + integration_spectra)
                                     : n_time;

    const std::size_t t_chunk_start = t_win_start + chunk_in_win * time_chunk_size;
    if (t_chunk_start >= t_win_end) {
        return;
    }
    const std::size_t t_chunk_end = (t_chunk_start + time_chunk_size < t_win_end)
                                       ? (t_chunk_start + time_chunk_size)
                                       : t_win_end;

    const std::size_t t_stride = n_freq * N_ANT;
    const std::size_t voltage_stride = n_freq * max_beams_stride;

    const std::uint8_t* packed_ptr = packed + (t_chunk_start * n_freq + freq) * N_ANT + lane;
    float2* voltages_ptr = voltages + (t_chunk_start * n_freq + freq) * max_beams_stride + beam_idx;

    std::size_t t = t_chunk_start;

    for (; t + (TIME_UNROLL - 1) < t_chunk_end; t += TIME_UNROLL) {
        float s_r[TIME_UNROLL] = {0.0F};
        float s_i[TIME_UNROLL] = {0.0F};

        #pragma unroll
        for (unsigned int a = 0; a < ANT_PER_LANE; ++a) {
            const float wra = w_r[a];
            const float wia = w_i[a];
            const float nwi = nw_i[a];
            const unsigned int a_offset = a * 32U;

            #pragma unroll
            for (int k = 0; k < TIME_UNROLL; ++k) {
                const float2 p = unpack_int4_fast(__ldg(&packed_ptr[k * t_stride + a_offset]));
                s_r[k] = fmaf(wra, p.x, fmaf(nwi, p.y, s_r[k]));
                s_i[k] = fmaf(wra, p.y, fmaf(wia, p.x, s_i[k]));
            }
        }

        // Intra-warp shuffle reduction
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            #pragma unroll
            for (int k = 0; k < TIME_UNROLL; ++k) {
                s_r[k] += __shfl_down_sync(full_mask, s_r[k], offset);
                s_i[k] += __shfl_down_sync(full_mask, s_i[k], offset);
            }
        }

        if (lane == 0) {
            #pragma unroll
            for (int k = 0; k < TIME_UNROLL; ++k) {
                voltages_ptr[k * voltage_stride] = make_float2(s_r[k], s_i[k]);
            }
        }

        packed_ptr += TIME_UNROLL * t_stride;
        voltages_ptr += TIME_UNROLL * voltage_stride;
    }

    // Remainder loop
    for (; t < t_chunk_end; ++t) {
        float s_r = 0.0F;
        float s_i = 0.0F;

        #pragma unroll
        for (unsigned int a = 0; a < ANT_PER_LANE; ++a) {
            const float2 p = unpack_int4_fast(__ldg(&packed_ptr[a * 32U]));
            s_r = fmaf(w_r[a], p.x, fmaf(nw_i[a], p.y, s_r));
            s_i = fmaf(w_r[a], p.y, fmaf(w_i[a], p.x, s_i));
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            s_r += __shfl_down_sync(full_mask, s_r, offset);
            s_i += __shfl_down_sync(full_mask, s_i, offset);
        }

        if (lane == 0) {
            *voltages_ptr = make_float2(s_r, s_i);
        }

        packed_ptr += t_stride;
        voltages_ptr += voltage_stride;
    }
}

template <int N_ANT, int TIME_UNROLL>
void dispatch_kernel(
    float2* d_voltages,
    const float* d_window_directions,
    const double* d_wavenumbers,
    const std::uint8_t* d_packed,
    const std::uint8_t* d_antenna_mask,
    const float3* d_antenna_positions,
    std::size_t n_time,
    std::size_t n_freq,
    std::size_t integration_spectra,
    std::size_t time_chunk_size,
    std::size_t chunks_per_window,
    std::size_t num_active_beams,
    std::size_t max_beams_stride,
    std::size_t total_warps,
    cudaStream_t stream) {

    constexpr int WARPS_PER_BLOCK = 4;
    const dim3 block_dim(32, WARPS_PER_BLOCK);
    const unsigned int grid_dim =
        static_cast<unsigned int>((total_warps + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

    tracker_v5_multibeam_kernel<N_ANT, TIME_UNROLL><<<grid_dim, block_dim, 0, stream>>>(
        d_voltages,
        d_window_directions,
        d_wavenumbers,
        d_packed,
        d_antenna_mask,
        d_antenna_positions,
        n_time,
        n_freq,
        integration_spectra,
        time_chunk_size,
        chunks_per_window,
        num_active_beams,
        max_beams_stride,
        total_warps);
}

} // namespace

void launch_beam_tracker_v5_multibeam(
    const int4x2_t* d_packed,
    float2* d_voltages,
    std::size_t n_time,
    std::size_t n_freq,
    std::size_t n_ant,
    std::size_t max_beams_allocated,
    const std::vector<double>& frequencies_hz,
    const MultiBeamTrackerConfig& config,
    cudaStream_t stream,
    std::size_t window_offset) {

    if (config.num_active_beams == 0) {
        return; // Zero active beams: instantaneous bypass
    }

    const std::size_t num_active_beams = std::min(config.num_active_beams, MAX_TRACKER_BEAMS);
    const std::size_t max_beams_stride = std::max(max_beams_allocated, num_active_beams);

    if (frequencies_hz.size() != n_freq) {
        throw std::invalid_argument("frequencies_hz size mismatch with n_freq");
    }

    const std::size_t window_count =
        (n_time + config.integration_spectra - 1) / config.integration_spectra;
    const std::size_t chunks_per_window =
        (config.integration_spectra + config.time_chunk_size - 1) / config.time_chunk_size;
    const std::size_t total_warps =
        window_count * chunks_per_window * n_freq * num_active_beams;

    std::vector<float> h_window_directions(window_count * num_active_beams * 3);
    for (std::size_t w = 0; w < window_count; ++w) {
        const std::size_t global_win = window_offset + w;
        const double center_sample =
            (static_cast<double>(global_win) + 0.5) * static_cast<double>(config.integration_spectra);

        for (std::size_t b = 0; b < num_active_beams; ++b) {
            const auto& traj = config.trajectories[b];
            const double l = traj.direction_start.x + traj.direction_rate_per_sample.dl * center_sample;
            const double m = traj.direction_start.y + traj.direction_rate_per_sample.dm * center_sample;
            const double trans_sq = l * l + m * m;
            const double n = (trans_sq <= 1.0) ? std::sqrt(1.0 - trans_sq) : 0.0;

            const std::size_t base_idx = (w * num_active_beams + b) * 3;
            h_window_directions[base_idx + 0] = static_cast<float>(l);
            h_window_directions[base_idx + 1] = static_cast<float>(m);
            h_window_directions[base_idx + 2] = static_cast<float>(n);
        }
    }

    std::vector<double> h_wavenumbers(n_freq);
    for (std::size_t f = 0; f < n_freq; ++f) {
        h_wavenumbers[f] = two_pi * frequencies_hz[f] / speed_of_light_m_per_s;
    }

    bool has_masked_antennas = false;
    for (std::size_t a = 0; a < n_ant; ++a) {
        if (config.antenna_mask[a] == 0) {
            has_masked_antennas = true;
            break;
        }
    }

    float* d_window_directions = nullptr;
    double* d_wavenumbers = nullptr;
    std::uint8_t* d_antenna_mask = nullptr;
    float3* d_antenna_positions = nullptr;

    CHECK_CUDA_ERROR_NON_OO(
        cudaMallocAsync(&d_window_directions, h_window_directions.size() * sizeof(float), stream));
    CHECK_CUDA_ERROR_NON_OO(
        cudaMallocAsync(&d_wavenumbers, h_wavenumbers.size() * sizeof(double), stream));
    CHECK_CUDA_ERROR_NON_OO(
        cudaMallocAsync(&d_antenna_positions, n_ant * sizeof(float3), stream));

    if (has_masked_antennas) {
        CHECK_CUDA_ERROR_NON_OO(
            cudaMallocAsync(&d_antenna_mask, n_ant * sizeof(std::uint8_t), stream));
        CHECK_CUDA_ERROR_NON_OO(
            cudaMemcpyAsync(d_antenna_mask, config.antenna_mask.data(),
                            n_ant * sizeof(std::uint8_t),
                            cudaMemcpyHostToDevice, stream));
    }

    CHECK_CUDA_ERROR_NON_OO(
        cudaMemcpyAsync(d_window_directions, h_window_directions.data(),
                        h_window_directions.size() * sizeof(float),
                        cudaMemcpyHostToDevice, stream));
    CHECK_CUDA_ERROR_NON_OO(
        cudaMemcpyAsync(d_wavenumbers, h_wavenumbers.data(),
                        h_wavenumbers.size() * sizeof(double),
                        cudaMemcpyHostToDevice, stream));
    CHECK_CUDA_ERROR_NON_OO(
        cudaMemcpyAsync(d_antenna_positions, config.antenna_positions.data(),
                        n_ant * sizeof(float3),
                        cudaMemcpyHostToDevice, stream));

    const auto* packed_bytes = reinterpret_cast<const std::uint8_t*>(d_packed);

    auto dispatch_ant = [&](auto ant_tag) {
        constexpr int N_A = decltype(ant_tag)::value;
        switch (config.time_unroll) {
            case 2:
                dispatch_kernel<N_A, 2>(
                    d_voltages, d_window_directions, d_wavenumbers, packed_bytes, d_antenna_mask,
                    d_antenna_positions, n_time, n_freq, config.integration_spectra,
                    config.time_chunk_size, chunks_per_window, num_active_beams, max_beams_stride,
                    total_warps, stream);
                break;
            case 4:
                dispatch_kernel<N_A, 4>(
                    d_voltages, d_window_directions, d_wavenumbers, packed_bytes, d_antenna_mask,
                    d_antenna_positions, n_time, n_freq, config.integration_spectra,
                    config.time_chunk_size, chunks_per_window, num_active_beams, max_beams_stride,
                    total_warps, stream);
                break;
            case 8:
            default:
                dispatch_kernel<N_A, 8>(
                    d_voltages, d_window_directions, d_wavenumbers, packed_bytes, d_antenna_mask,
                    d_antenna_positions, n_time, n_freq, config.integration_spectra,
                    config.time_chunk_size, chunks_per_window, num_active_beams, max_beams_stride,
                    total_warps, stream);
                break;
        }
    };

    switch (n_ant) {
        case 32:  dispatch_ant(std::integral_constant<int, 32>{}); break;
        case 64:  dispatch_ant(std::integral_constant<int, 64>{}); break;
        case 128: dispatch_ant(std::integral_constant<int, 128>{}); break;
        case 256: dispatch_ant(std::integral_constant<int, 256>{}); break;
        default:
            CHECK_CUDA_ERROR_NON_OO(cudaFreeAsync(d_window_directions, stream));
            CHECK_CUDA_ERROR_NON_OO(cudaFreeAsync(d_wavenumbers, stream));
            CHECK_CUDA_ERROR_NON_OO(cudaFreeAsync(d_antenna_positions, stream));
            if (d_antenna_mask) CHECK_CUDA_ERROR_NON_OO(cudaFreeAsync(d_antenna_mask, stream));
            throw std::invalid_argument("Unsupported n_ant: must be 32, 64, 128, or 256");
    }

    CHECK_CUDA_ERROR_NON_OO(cudaFreeAsync(d_window_directions, stream));
    CHECK_CUDA_ERROR_NON_OO(cudaFreeAsync(d_wavenumbers, stream));
    CHECK_CUDA_ERROR_NON_OO(cudaFreeAsync(d_antenna_positions, stream));
    if (d_antenna_mask) CHECK_CUDA_ERROR_NON_OO(cudaFreeAsync(d_antenna_mask, stream));
}

void launch_beam_tracker_v5(
    const int4x2_t* d_packed,
    float2* d_voltages,
    std::size_t n_time,
    std::size_t n_freq,
    std::size_t n_ant,
    const std::vector<double>& frequencies_hz,
    const BeamTrackerConfig& config,
    cudaStream_t stream,
    std::size_t window_offset) {

    MultiBeamTrackerConfig multi_cfg;
    multi_cfg.num_active_beams = 1;
    multi_cfg.trajectories[0] = config.trajectory;
    multi_cfg.integration_spectra = config.integration_spectra;
    multi_cfg.spacing_m = config.spacing_m;
    multi_cfg.time_chunk_size = config.time_chunk_size;
    multi_cfg.time_unroll = config.time_unroll;
    multi_cfg.enable_cuda_graph = config.enable_cuda_graph;
    multi_cfg.antenna_mask = config.antenna_mask;
    multi_cfg.antenna_positions = config.antenna_positions;
    multi_cfg.site = config.site;

    launch_beam_tracker_v5_multibeam(
        d_packed, d_voltages, n_time, n_freq, n_ant, 1, frequencies_hz, multi_cfg, stream, window_offset);
}

struct CudaBeamTrackerV5Stream::Impl {
    std::size_t n_time_per_batch;
    std::size_t n_freq;
    std::size_t n_ant;
    std::vector<double> freqs;
    MultiBeamTrackerConfig config;

    cudaStream_t stream = nullptr;
    int4x2_t* d_packed = nullptr;
    float2* d_voltages = nullptr;
    float* d_window_directions = nullptr;
    double* d_wavenumbers = nullptr;
    std::uint8_t* d_antenna_mask = nullptr;
    float3* d_antenna_positions = nullptr;

    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event = nullptr;
    float last_time_ms = 0.0F;

    std::size_t window_count = 0;
    std::size_t chunks_per_window = 0;
    std::size_t total_warps = 0;

    std::vector<float> h_window_directions;
    std::vector<double> h_wavenumbers;

    Impl(std::size_t n_time,
         std::size_t n_f,
         std::size_t n_a,
         const std::vector<double>& frequencies_hz,
         const BeamTrackerConfig& cfg)
        : n_time_per_batch(n_time), n_freq(n_f), n_ant(n_a), freqs(frequencies_hz) {

        config.num_active_beams = 1;
        config.trajectories[0] = cfg.trajectory;
        config.integration_spectra = cfg.integration_spectra;
        config.spacing_m = cfg.spacing_m;
        config.time_chunk_size = cfg.time_chunk_size;
        config.time_unroll = cfg.time_unroll;
        config.enable_cuda_graph = cfg.enable_cuda_graph;
        config.antenna_mask = cfg.antenna_mask;
        config.site = cfg.site;

        window_count = (n_time_per_batch + config.integration_spectra - 1) / config.integration_spectra;
        chunks_per_window = (config.integration_spectra + config.time_chunk_size - 1) / config.time_chunk_size;
        total_warps = window_count * chunks_per_window * n_freq * config.num_active_beams;

        CHECK_CUDA_ERROR_NON_OO(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CHECK_CUDA_ERROR_NON_OO(cudaEventCreate(&start_event));
        CHECK_CUDA_ERROR_NON_OO(cudaEventCreate(&stop_event));

        const std::size_t packed_bytes = n_time_per_batch * n_freq * n_ant * sizeof(int4x2_t);
        const std::size_t voltages_bytes = n_time_per_batch * n_freq * sizeof(float2);

        CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&d_packed, packed_bytes));
        CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&d_voltages, voltages_bytes));

        h_window_directions.resize(window_count * 3);
        h_wavenumbers.resize(n_freq);

        for (std::size_t f = 0; f < n_freq; ++f) {
            h_wavenumbers[f] = two_pi * freqs[f] / speed_of_light_m_per_s;
        }

        CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&d_window_directions, h_window_directions.size() * sizeof(float)));
        CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&d_wavenumbers, h_wavenumbers.size() * sizeof(double)));

        bool has_masked_antennas = false;
        for (std::size_t a = 0; a < n_ant; ++a) {
            if (config.antenna_mask[a] == 0) {
                has_masked_antennas = true;
                break;
            }
        }
        if (has_masked_antennas) {
            CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&d_antenna_mask, n_ant * sizeof(std::uint8_t)));
            CHECK_CUDA_ERROR_NON_OO(cudaMemcpyAsync(d_antenna_mask, config.antenna_mask.data(),
                                                    n_ant * sizeof(std::uint8_t),
                                                    cudaMemcpyHostToDevice, stream));
        }

        CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&d_antenna_positions, n_ant * sizeof(float3)));
        CHECK_CUDA_ERROR_NON_OO(cudaMemcpyAsync(d_antenna_positions, config.antenna_positions.data(),
                                                n_ant * sizeof(float3),
                                                cudaMemcpyHostToDevice, stream));

        CHECK_CUDA_ERROR_NON_OO(cudaMemcpyAsync(d_wavenumbers, h_wavenumbers.data(),
                                                h_wavenumbers.size() * sizeof(double),
                                                cudaMemcpyHostToDevice, stream));
    }

    ~Impl() {
        if (d_packed) cudaFree(d_packed);
        if (d_voltages) cudaFree(d_voltages);
        if (d_window_directions) cudaFree(d_window_directions);
        if (d_wavenumbers) cudaFree(d_wavenumbers);
        if (d_antenna_mask) cudaFree(d_antenna_mask);
        if (d_antenna_positions) cudaFree(d_antenna_positions);
        if (start_event) cudaEventDestroy(start_event);
        if (stop_event) cudaEventDestroy(stop_event);
        if (stream) cudaStreamDestroy(stream);
    }

    void execute_internal(std::size_t window_offset, const int4x2_t* in_p, float2* out_v) {
        for (std::size_t w = 0; w < window_count; ++w) {
            const std::size_t global_win = window_offset + w;
            const double center_sample =
                (static_cast<double>(global_win) + 0.5) * static_cast<double>(config.integration_spectra);
            const double l = config.trajectories[0].direction_start.x
                           + config.trajectories[0].direction_rate_per_sample.dl * center_sample;
            const double m = config.trajectories[0].direction_start.y
                           + config.trajectories[0].direction_rate_per_sample.dm * center_sample;
            const double trans_sq = l * l + m * m;
            const double n = (trans_sq <= 1.0) ? std::sqrt(1.0 - trans_sq) : 0.0;
            h_window_directions[w * 3 + 0] = static_cast<float>(l);
            h_window_directions[w * 3 + 1] = static_cast<float>(m);
            h_window_directions[w * 3 + 2] = static_cast<float>(n);
        }

        CHECK_CUDA_ERROR_NON_OO(cudaMemcpyAsync(d_window_directions, h_window_directions.data(),
                                                h_window_directions.size() * sizeof(float),
                                                cudaMemcpyHostToDevice, stream));

        CHECK_CUDA_ERROR_NON_OO(cudaEventRecord(start_event, stream));

        const auto* packed_bytes = reinterpret_cast<const std::uint8_t*>(in_p);
        switch (n_ant) {
            case 32:
                dispatch_kernel<32, 8>(out_v, d_window_directions, d_wavenumbers, packed_bytes, d_antenna_mask,
                                       d_antenna_positions,
                                       n_time_per_batch, n_freq, config.integration_spectra,
                                       config.time_chunk_size, chunks_per_window,
                                       1, 1, total_warps, stream);
                break;
            case 64:
                dispatch_kernel<64, 8>(out_v, d_window_directions, d_wavenumbers, packed_bytes, d_antenna_mask,
                                       d_antenna_positions,
                                       n_time_per_batch, n_freq, config.integration_spectra,
                                       config.time_chunk_size, chunks_per_window,
                                       1, 1, total_warps, stream);
                break;
            case 128:
                dispatch_kernel<128, 8>(out_v, d_window_directions, d_wavenumbers, packed_bytes, d_antenna_mask,
                                       d_antenna_positions,
                                       n_time_per_batch, n_freq, config.integration_spectra,
                                       config.time_chunk_size, chunks_per_window,
                                       1, 1, total_warps, stream);
                break;
            case 256:
                dispatch_kernel<256, 8>(out_v, d_window_directions, d_wavenumbers, packed_bytes, d_antenna_mask,
                                       d_antenna_positions,
                                       n_time_per_batch, n_freq, config.integration_spectra,
                                       config.time_chunk_size, chunks_per_window,
                                       1, 1, total_warps, stream);
                break;
        }

        CHECK_CUDA_ERROR_NON_OO(cudaEventRecord(stop_event, stream));
        CHECK_CUDA_ERROR_NON_OO(cudaEventSynchronize(stop_event));
        CHECK_CUDA_ERROR_NON_OO(cudaEventElapsedTime(&last_time_ms, start_event, stop_event));
    }
};

CudaBeamTrackerV5Stream::CudaBeamTrackerV5Stream(
    std::size_t n_time_per_batch,
    std::size_t n_freq,
    std::size_t n_ant,
    const std::vector<double>& frequencies_hz,
    const BeamTrackerConfig& config)
    : impl_(std::make_unique<Impl>(n_time_per_batch, n_freq, n_ant, frequencies_hz, config)) {}

CudaBeamTrackerV5Stream::~CudaBeamTrackerV5Stream() = default;

void CudaBeamTrackerV5Stream::process_batch(
    std::size_t window_offset,
    const int4x2_t* host_packed,
    float2* host_voltages) {

    const std::size_t packed_bytes = impl_->n_time_per_batch * impl_->n_freq * impl_->n_ant * sizeof(int4x2_t);
    const std::size_t voltages_bytes = impl_->n_time_per_batch * impl_->n_freq * sizeof(float2);

    CHECK_CUDA_ERROR_NON_OO(cudaMemcpyAsync(impl_->d_packed, host_packed, packed_bytes,
                                            cudaMemcpyHostToDevice, impl_->stream));

    impl_->execute_internal(window_offset, impl_->d_packed, impl_->d_voltages);

    CHECK_CUDA_ERROR_NON_OO(cudaMemcpyAsync(host_voltages, impl_->d_voltages, voltages_bytes,
                                            cudaMemcpyDeviceToHost, impl_->stream));
    CHECK_CUDA_ERROR_NON_OO(cudaStreamSynchronize(impl_->stream));
}

void CudaBeamTrackerV5Stream::process_batch_device(
    std::size_t window_offset,
    const int4x2_t* d_packed,
    float2* d_voltages) {
    impl_->execute_internal(window_offset, d_packed, d_voltages);
}

float CudaBeamTrackerV5Stream::last_kernel_time_ms() const { return impl_->last_time_ms; }
cudaStream_t CudaBeamTrackerV5Stream::get_stream() const { return impl_->stream; }
int4x2_t* CudaBeamTrackerV5Stream::device_packed_buffer() { return impl_->d_packed; }
float2* CudaBeamTrackerV5Stream::device_voltages_buffer() { return impl_->d_voltages; }

} // namespace kotekan
