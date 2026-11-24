#include "chartsConstants.hpp"
#include "cudaDirectBeamTracker.hpp"
#include "cudaUtils.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

namespace {

using Clock = std::chrono::high_resolution_clock;

constexpr double SPEED_OF_LIGHT = kotekan::charts::constants::speed_of_light_m_per_s;
constexpr double TWO_PI = kotekan::charts::constants::two_pi;

inline kotekan::int4x2_t pack_int4x2(int re, int im) {
    re = std::max(-7, std::min(7, re));
    im = std::max(-7, std::min(7, im));
    kotekan::int4x2_t p;
    p.val = static_cast<std::uint8_t>((re & 0x0F) | ((im & 0x0F) << 4));
    return p;
}

inline std::complex<double> unpack_int4x2(kotekan::int4x2_t p) {
    int vr = static_cast<int>(p.val & 0x0F);
    if (vr >= 8) {
        vr -= 16;
    }
    int vi = static_cast<int>((p.val >> 4) & 0x0F);
    if (vi >= 8) {
        vi -= 16;
    }
    return {static_cast<double>(vr), static_cast<double>(vi)};
}

std::vector<float3> compute_charts_positions_raw(std::size_t n_ant, float spacing_m,
                                                 bool descending_order) {
    std::vector<float3> pos(n_ant);
    for (std::size_t r = 0; r < n_ant; ++r) {
        const std::size_t phys = descending_order ? (n_ant - 1 - r) : r;
        const std::size_t col = phys & 7U;
        const std::size_t row = phys >> 3U;
        // CHARTS convention: x = -col * d, y = +row * d
        pos[r] = make_float3(-static_cast<float>(col) * spacing_m,
                             static_cast<float>(row) * spacing_m, 0.0f);
    }
    return pos;
}

bool test_wmma_vs_fp32_parity_and_latency() {
    std::cout << "\n======================================================\n";
    std::cout << "[Test 1/3] FP16 WMMA Tensor Core vs FP32 Parity & Latency\n";
    std::cout << "======================================================\n";

    constexpr std::size_t n_ant = 64;
    constexpr std::size_t n_freq = 4;
    constexpr std::size_t n_time = 256;

    std::mt19937 rng(20261124);
    std::uniform_int_distribution<int> sample_dist(-6, 6);

    const std::size_t input_elems = n_time * n_freq * n_ant;
    std::vector<kotekan::int4x2_t> h_packed(input_elems);
    for (std::size_t i = 0; i < input_elems; ++i) {
        h_packed[i] = pack_int4x2(sample_dist(rng), sample_dist(rng));
    }

    const auto h_positions = compute_charts_positions_raw(n_ant, 0.25f, true);
    std::vector<double> h_wavenumbers(n_freq);
    for (std::size_t f = 0; f < n_freq; ++f) {
        const double freq_hz = 300.0e6 + static_cast<double>(f) * 2.0e6;
        h_wavenumbers[f] = TWO_PI * freq_hz / SPEED_OF_LIGHT;
    }

    kotekan::int4x2_t* d_packed = nullptr;
    double* d_wavenumbers = nullptr;
    float3* d_positions = nullptr;
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&d_packed, input_elems * sizeof(kotekan::int4x2_t)));
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&d_wavenumbers, n_freq * sizeof(double)));
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&d_positions, n_ant * sizeof(float3)));
    CHECK_CUDA_ERROR_NON_OO(cudaMemcpy(d_packed, h_packed.data(),
                                       input_elems * sizeof(kotekan::int4x2_t),
                                       cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR_NON_OO(cudaMemcpy(d_wavenumbers, h_wavenumbers.data(),
                                       n_freq * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR_NON_OO(
        cudaMemcpy(d_positions, h_positions.data(), n_ant * sizeof(float3), cudaMemcpyHostToDevice));

    cudaStream_t stream = nullptr;
    CHECK_CUDA_ERROR_NON_OO(cudaStreamCreate(&stream));

    bool ok = true;
    for (const std::size_t num_beams : {std::size_t{16}, std::size_t{64}}) {
        const std::size_t max_beams_stride = 64;
        std::vector<kotekan::DirectDirection3D> h_dirs(num_beams);
        for (std::size_t b = 0; b < num_beams; ++b) {
            const float ang = 0.08f * static_cast<float>(b);
            const float l = 0.35f * std::cos(ang);
            const float m = 0.35f * std::sin(ang);
            h_dirs[b] = {l, m, std::sqrt(std::max(0.0f, 1.0f - l * l - m * m))};
        }

        const std::size_t weight_elems = max_beams_stride * n_freq * n_ant;
        const std::size_t out_elems = n_time * n_freq * max_beams_stride;

        kotekan::DirectDirection3D* d_dirs = nullptr;
        float2* d_weights = nullptr;
        float2* d_voltages_fp32 = nullptr;
        float2* d_voltages_wmma = nullptr;
        CHECK_CUDA_ERROR_NON_OO(
            cudaMalloc(&d_dirs, num_beams * sizeof(kotekan::DirectDirection3D)));
        CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&d_weights, weight_elems * sizeof(float2)));
        CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&d_voltages_fp32, out_elems * sizeof(float2)));
        CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&d_voltages_wmma, out_elems * sizeof(float2)));
        CHECK_CUDA_ERROR_NON_OO(cudaMemcpy(
            d_dirs, h_dirs.data(), num_beams * sizeof(kotekan::DirectDirection3D),
            cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR_NON_OO(cudaMemset(d_voltages_fp32, 0, out_elems * sizeof(float2)));
        CHECK_CUDA_ERROR_NON_OO(cudaMemset(d_voltages_wmma, 0, out_elems * sizeof(float2)));

        kotekan::launch_generate_steering_weights(d_weights, nullptr, d_dirs, d_dirs,
                                                  d_wavenumbers, d_positions, nullptr, nullptr,
                                                  num_beams, n_freq, n_ant, n_time, n_ant, stream);

        kotekan::launch_direct_beamformer(
            d_packed, d_weights, nullptr, d_voltages_fp32, n_time, n_freq, n_ant, num_beams,
            max_beams_stride, 256, 4, 4, stream, 1, kotekan::DirectPolLayout::INTERLEAVED,
            kotekan::DirectInputFormat::INT4X2, false);
        kotekan::launch_direct_beamformer(
            d_packed, d_weights, nullptr, d_voltages_wmma, n_time, n_freq, n_ant, num_beams,
            max_beams_stride, 256, 4, 4, stream, 1, kotekan::DirectPolLayout::INTERLEAVED,
            kotekan::DirectInputFormat::INT4X2, true);
        CHECK_CUDA_ERROR_NON_OO(cudaStreamSynchronize(stream));

        std::vector<float2> h_fp32(out_elems);
        std::vector<float2> h_wmma(out_elems);
        CHECK_CUDA_ERROR_NON_OO(
            cudaMemcpy(h_fp32.data(), d_voltages_fp32, out_elems * sizeof(float2),
                       cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR_NON_OO(
            cudaMemcpy(h_wmma.data(), d_voltages_wmma, out_elems * sizeof(float2),
                       cudaMemcpyDeviceToHost));

        double sum_sq_err = 0.0;
        double sum_sq_sig = 0.0;
        double max_abs_err = 0.0;
        for (std::size_t t = 0; t < n_time; ++t) {
            for (std::size_t f = 0; f < n_freq; ++f) {
                for (std::size_t b = 0; b < num_beams; ++b) {
                    const std::size_t idx = (t * n_freq + f) * max_beams_stride + b;
                    const double er = static_cast<double>(h_wmma[idx].x) - h_fp32[idx].x;
                    const double ei = static_cast<double>(h_wmma[idx].y) - h_fp32[idx].y;
                    const double err = std::sqrt(er * er + ei * ei);
                    max_abs_err = std::max(max_abs_err, err);
                    sum_sq_err += er * er + ei * ei;
                    sum_sq_sig += static_cast<double>(h_fp32[idx].x) * h_fp32[idx].x
                                  + static_cast<double>(h_fp32[idx].y) * h_fp32[idx].y;
                }
            }
        }
        const double rel_rms = std::sqrt(sum_sq_err / (sum_sq_sig + 1e-12));
        std::cout << "  - " << num_beams << " beams: rel_rms=" << std::scientific << rel_rms
                  << ", max_abs_err=" << max_abs_err << std::fixed << "\n";
        if (rel_rms >= 1.0e-2) {
            ok = false;
        }

        cudaFree(d_dirs);
        cudaFree(d_weights);
        cudaFree(d_voltages_fp32);
        cudaFree(d_voltages_wmma);
    }

    // Production scale 64-beam latency benchmark: 15360 x 336 x 64
    constexpr std::size_t bench_time = 15360;
    constexpr std::size_t bench_freq = 336;
    constexpr std::size_t bench_beams = 64;
    kotekan::int4x2_t* d_bench_in = nullptr;
    float2* d_bench_w = nullptr;
    float2* d_bench_out = nullptr;
    CHECK_CUDA_ERROR_NON_OO(
        cudaMalloc(&d_bench_in, bench_time * bench_freq * n_ant * sizeof(kotekan::int4x2_t)));
    CHECK_CUDA_ERROR_NON_OO(
        cudaMalloc(&d_bench_w, bench_beams * bench_freq * n_ant * sizeof(float2)));
    CHECK_CUDA_ERROR_NON_OO(
        cudaMalloc(&d_bench_out, bench_time * bench_freq * bench_beams * sizeof(float2)));
    CHECK_CUDA_ERROR_NON_OO(
        cudaMemset(d_bench_in, 0x11, bench_time * bench_freq * n_ant * sizeof(kotekan::int4x2_t)));
    CHECK_CUDA_ERROR_NON_OO(
        cudaMemset(d_bench_w, 0, bench_beams * bench_freq * n_ant * sizeof(float2)));

    // Warmup both paths
    for (int w = 0; w < 3; ++w) {
        kotekan::launch_direct_beamformer(
            d_bench_in, d_bench_w, nullptr, d_bench_out, bench_time, bench_freq, n_ant, bench_beams,
            bench_beams, 256, 4, 4, stream, 1, kotekan::DirectPolLayout::INTERLEAVED,
            kotekan::DirectInputFormat::INT4X2, false);
        kotekan::launch_direct_beamformer(
            d_bench_in, d_bench_w, nullptr, d_bench_out, bench_time, bench_freq, n_ant, bench_beams,
            bench_beams, 256, 4, 4, stream, 1, kotekan::DirectPolLayout::INTERLEAVED,
            kotekan::DirectInputFormat::INT4X2, true);
    }
    CHECK_CUDA_ERROR_NON_OO(cudaStreamSynchronize(stream));

    cudaEvent_t ev_start = nullptr;
    cudaEvent_t ev_stop = nullptr;
    CHECK_CUDA_ERROR_NON_OO(cudaEventCreate(&ev_start));
    CHECK_CUDA_ERROR_NON_OO(cudaEventCreate(&ev_stop));
    constexpr int iters = 20;

    CHECK_CUDA_ERROR_NON_OO(cudaEventRecord(ev_start, stream));
    for (int i = 0; i < iters; ++i) {
        kotekan::launch_direct_beamformer(
            d_bench_in, d_bench_w, nullptr, d_bench_out, bench_time, bench_freq, n_ant,
            bench_beams, bench_beams, 256, 4, 4, stream, 1, kotekan::DirectPolLayout::INTERLEAVED,
            kotekan::DirectInputFormat::INT4X2, false);
    }
    CHECK_CUDA_ERROR_NON_OO(cudaEventRecord(ev_stop, stream));
    CHECK_CUDA_ERROR_NON_OO(cudaEventSynchronize(ev_stop));
    float fp32_ms = 0.0f;
    CHECK_CUDA_ERROR_NON_OO(cudaEventElapsedTime(&fp32_ms, ev_start, ev_stop));
    const float fp32_per_frame_ms = fp32_ms / static_cast<float>(iters);

    CHECK_CUDA_ERROR_NON_OO(cudaEventRecord(ev_start, stream));
    for (int i = 0; i < iters; ++i) {
        kotekan::launch_direct_beamformer(
            d_bench_in, d_bench_w, nullptr, d_bench_out, bench_time, bench_freq, n_ant,
            bench_beams, bench_beams, 256, 4, 4, stream, 1, kotekan::DirectPolLayout::INTERLEAVED,
            kotekan::DirectInputFormat::INT4X2, true);
    }
    CHECK_CUDA_ERROR_NON_OO(cudaEventRecord(ev_stop, stream));
    CHECK_CUDA_ERROR_NON_OO(cudaEventSynchronize(ev_stop));
    float elapsed_ms = 0.0f;
    CHECK_CUDA_ERROR_NON_OO(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop));
    const float per_frame_ms = elapsed_ms / static_cast<float>(iters);
    std::cout << "  - 64-beam (15360 x 336 x 64) FP32=" << std::setprecision(3) << fp32_per_frame_ms
              << " ms, WMMA=" << per_frame_ms << " ms (speedup " << std::setprecision(2)
              << (fp32_per_frame_ms / std::max(per_frame_ms, 1.0e-3f))
              << "x, budget < 3.000 ms)\n";
    if (per_frame_ms >= 3.0f) {
        ok = false;
    }

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    cudaFree(d_bench_in);
    cudaFree(d_bench_w);
    cudaFree(d_bench_out);
    cudaFree(d_packed);
    cudaFree(d_wavenumbers);
    cudaFree(d_positions);
    cudaStreamDestroy(stream);
    return ok;
}

bool run_2d_fft_case(std::size_t fft_size) {
    constexpr std::size_t n_ant = 64;
    constexpr std::size_t n_freq = 4;
    constexpr std::size_t n_time = 128;
    constexpr std::size_t chunk_size = 64;
    const std::size_t n_beams = fft_size * fft_size;
    const std::size_t max_beams_stride = n_beams;
    const bool descending_order = true;

    std::mt19937 rng(1000 + static_cast<unsigned int>(fft_size));
    std::uniform_real_distribution<float> amp_dist(0.85f, 1.15f);
    std::uniform_real_distribution<float> ph_dist(-0.25f, 0.25f);

    // Non-trivial calibration gains & normalized taper + chunk RFI mask
    std::vector<float2> h_cal(n_freq * n_ant);
    for (std::size_t i = 0; i < h_cal.size(); ++i) {
        const float a = amp_dist(rng);
        const float p = ph_dist(rng);
        h_cal[i] = make_float2(a * std::cos(p), a * std::sin(p));
    }
    std::vector<float> h_taper(n_ant);
    for (std::size_t a = 0; a < n_ant; ++a) {
        h_taper[a] = 0.01f * (1.0f + 0.05f * static_cast<float>(a & 7U));
    }
    const std::size_t n_chunks = (n_time + chunk_size - 1) / chunk_size;
    std::vector<std::uint8_t> h_mask(n_chunks * n_freq, 0);
    // Flag chunk 1 on freq 2 so power accumulation skips it
    h_mask[1 * n_freq + 2] = 1;

    // Synthesize an off-zenith plane wave steered cleanly to beam bin (p_target, q_target)
    const int p_target = static_cast<int>(fft_size / 2) + 2;
    const int q_target = static_cast<int>(fft_size / 2) - 1;
    const double u_x =
        TWO_PI * static_cast<double>(p_target - static_cast<int>(fft_size / 2)) / fft_size;
    const double u_y =
        TWO_PI * static_cast<double>(q_target - static_cast<int>(fft_size / 2)) / fft_size;

    std::vector<kotekan::int4x2_t> h_packed(n_time * n_freq * n_ant);
    for (std::size_t t = 0; t < n_time; ++t) {
        for (std::size_t f = 0; f < n_freq; ++f) {
            const double t_phase = 0.13 * static_cast<double>(t) + 0.07 * static_cast<double>(f);
            for (std::size_t a = 0; a < n_ant; ++a) {
                const std::size_t phys = descending_order ? (63 - a) : a;
                const double col = static_cast<double>(phys & 7U);
                const double row = static_cast<double>(phys >> 3U);
                // Matches kernel convention: col DFT exp(-j 2pi p col / N) & row exp(+j 2pi q row / N)
                const double theta = t_phase + u_x * col - u_y * row;
                const int vr = static_cast<int>(std::llround(6.0 * std::cos(theta)));
                const int vi = static_cast<int>(std::llround(6.0 * std::sin(theta)));
                h_packed[(t * n_freq + f) * n_ant + a] = pack_int4x2(vr, vi);
            }
        }
    }

    // CPU reference 2D DFT (power layout matches factorized_2d_fft_beamformer_kernel: [b * n_freq + f])
    std::vector<float2> cpu_v(n_time * n_freq * max_beams_stride, make_float2(0.0f, 0.0f));
    std::vector<float> cpu_p(n_beams * n_freq, 0.0f);
    std::vector<std::size_t> valid_counts(n_freq, 0);
    for (std::size_t t = 0; t < n_time; ++t) {
        const std::size_t chunk = t / chunk_size;
        for (std::size_t f = 0; f < n_freq; ++f) {
            std::complex<double> grid[16][16];
            for (int r = 0; r < 16; ++r) {
                for (int c = 0; c < 16; ++c) {
                    grid[r][c] = {0.0, 0.0};
                }
            }
            for (std::size_t a = 0; a < n_ant; ++a) {
                const std::size_t phys = descending_order ? (63 - a) : a;
                const std::size_t col = phys & 7U;
                const std::size_t row = phys >> 3U;
                const std::complex<double> z = unpack_int4x2(h_packed[(t * n_freq + f) * n_ant + a]);
                const float2 g = h_cal[f * n_ant + a];
                const double w_scale = static_cast<double>(h_taper[a]);
                const std::complex<double> weighted(
                    (z.real() * g.x - z.imag() * g.y) * w_scale,
                    (z.real() * g.y + z.imag() * g.x) * w_scale);
                grid[row][col] = weighted;
            }
            const bool masked = (h_mask[chunk * n_freq + f] != 0);
            if (!masked) {
                valid_counts[f] += 1;
            }
            for (std::size_t q = 0; q < fft_size; ++q) {
                const int k_row = static_cast<int>((q + fft_size / 2) % fft_size);
                for (std::size_t p = 0; p < fft_size; ++p) {
                    const int k_col = static_cast<int>((p + fft_size / 2) % fft_size);
                    std::complex<double> acc(0.0, 0.0);
                    for (std::size_t r = 0; r < 8; ++r) {
                        const double ang_r =
                            TWO_PI * static_cast<double>(k_row) * static_cast<double>(r) / fft_size;
                        const std::complex<double> wr(std::cos(ang_r), std::sin(ang_r));
                        std::complex<double> row_sum(0.0, 0.0);
                        for (std::size_t c = 0; c < 8; ++c) {
                            const double ang_c = -TWO_PI * static_cast<double>(k_col)
                                                 * static_cast<double>(c) / fft_size;
                            const std::complex<double> wc(std::cos(ang_c), std::sin(ang_c));
                            row_sum += grid[r][c] * wc;
                        }
                        acc += row_sum * wr;
                    }
                    const std::size_t b_idx = q * fft_size + p;
                    cpu_v[(t * n_freq + f) * max_beams_stride + b_idx] = make_float2(
                        static_cast<float>(acc.real()), static_cast<float>(acc.imag()));
                    if (!masked) {
                        cpu_p[b_idx * n_freq + f] += static_cast<float>(std::norm(acc));
                    }
                }
            }
        }
    }
    for (std::size_t b_idx = 0; b_idx < n_beams; ++b_idx) {
        for (std::size_t f = 0; f < n_freq; ++f) {
            if (valid_counts[f] > 0) {
                cpu_p[b_idx * n_freq + f] /= static_cast<float>(valid_counts[f]);
            }
        }
    }

    kotekan::int4x2_t* d_packed = nullptr;
    float2* d_cal = nullptr;
    float* d_taper = nullptr;
    std::uint8_t* d_mask = nullptr;
    float2* d_v = nullptr;
    float* d_p = nullptr;
    CHECK_CUDA_ERROR_NON_OO(
        cudaMalloc(&d_packed, h_packed.size() * sizeof(kotekan::int4x2_t)));
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&d_cal, h_cal.size() * sizeof(float2)));
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&d_taper, h_taper.size() * sizeof(float)));
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&d_mask, h_mask.size() * sizeof(std::uint8_t)));
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&d_v, cpu_v.size() * sizeof(float2)));
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&d_p, cpu_p.size() * sizeof(float)));

    CHECK_CUDA_ERROR_NON_OO(cudaMemcpy(d_packed, h_packed.data(),
                                       h_packed.size() * sizeof(kotekan::int4x2_t),
                                       cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR_NON_OO(
        cudaMemcpy(d_cal, h_cal.data(), h_cal.size() * sizeof(float2), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR_NON_OO(
        cudaMemcpy(d_taper, h_taper.data(), h_taper.size() * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR_NON_OO(cudaMemcpy(
        d_mask, h_mask.data(), h_mask.size() * sizeof(std::uint8_t), cudaMemcpyHostToDevice));

    cudaStream_t stream = nullptr;
    CHECK_CUDA_ERROR_NON_OO(cudaStreamCreate(&stream));
    kotekan::launch_factorized_2d_fft_beamformer(
        d_packed, d_v, d_p, d_cal, d_taper, d_mask, n_time, n_freq, fft_size, max_beams_stride,
        chunk_size, descending_order, stream, kotekan::DirectInputFormat::INT4X2);
    CHECK_CUDA_ERROR_NON_OO(cudaStreamSynchronize(stream));

    std::vector<float2> gpu_v(cpu_v.size());
    std::vector<float> gpu_p(cpu_p.size());
    CHECK_CUDA_ERROR_NON_OO(
        cudaMemcpy(gpu_v.data(), d_v, gpu_v.size() * sizeof(float2), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR_NON_OO(
        cudaMemcpy(gpu_p.data(), d_p, gpu_p.size() * sizeof(float), cudaMemcpyDeviceToHost));

    double max_v_err = 0.0;
    for (std::size_t i = 0; i < cpu_v.size(); ++i) {
        const double er = static_cast<double>(gpu_v[i].x) - cpu_v[i].x;
        const double ei = static_cast<double>(gpu_v[i].y) - cpu_v[i].y;
        max_v_err = std::max(max_v_err, std::sqrt(er * er + ei * ei));
    }

    double max_p_rel_err = 0.0;
    std::size_t peak_beam = 0;
    float peak_power = -1.0f;
    for (std::size_t i = 0; i < cpu_p.size(); ++i) {
        const double rel = std::abs(gpu_p[i] - cpu_p[i]) / std::max(1.0, static_cast<double>(cpu_p[i]));
        max_p_rel_err = std::max(max_p_rel_err, rel);
    }
    for (std::size_t b = 0; b < n_beams; ++b) {
        if (gpu_p[b * n_freq + 0] > peak_power) {
            peak_power = gpu_p[b * n_freq + 0];
            peak_beam = b;
        }
    }
    const std::size_t expected_peak = static_cast<std::size_t>(q_target) * fft_size + p_target;

    std::cout << "  - " << fft_size << "x" << fft_size << " (" << n_beams
              << " beams): max_v_err=" << std::scientific << max_v_err
              << ", max_p_rel_err=" << max_p_rel_err << std::fixed << ", peak_beam=" << peak_beam
              << " (expected " << expected_peak << ")\n";

    cudaFree(d_packed);
    cudaFree(d_cal);
    cudaFree(d_taper);
    cudaFree(d_mask);
    cudaFree(d_v);
    cudaFree(d_p);
    cudaStreamDestroy(stream);

    return (max_v_err < 1.0e-4) && (max_p_rel_err < 1.0e-4) && (peak_beam == expected_peak);
}

bool test_2d_fft_allsky() {
    std::cout << "\n======================================================\n";
    std::cout << "[Test 2/3] Separable 2D FFT All-Sky Mode (8x8 and 16x16)\n";
    std::cout << "======================================================\n";
    const bool ok8 = run_2d_fft_case(8);
    const bool ok16 = run_2d_fft_case(16);
    return ok8 && ok16;
}

bool test_factorized_2d_dft() {
    std::cout << "\n======================================================\n";
    std::cout << "[Test 3/3] Factorized Arbitrary 2D DFT (Nx=10, Ny=6)\n";
    std::cout << "======================================================\n";

    constexpr std::size_t n_ant = 64;
    constexpr std::size_t n_freq = 3;
    constexpr std::size_t n_time = 64;
    constexpr std::size_t nx = 10;
    constexpr std::size_t ny = 6;
    constexpr std::size_t max_beams_stride = nx * ny;
    constexpr bool descending_order = true;

    std::mt19937 rng(777);
    std::uniform_int_distribution<int> sample_dist(-7, 7);
    std::uniform_real_distribution<float> w_dist(-0.5f, 0.5f);

    std::vector<kotekan::int4x2_t> h_packed(n_time * n_freq * n_ant);
    for (auto& v : h_packed) {
        v = pack_int4x2(sample_dist(rng), sample_dist(rng));
    }
    std::vector<float2> h_bx(n_freq * nx * 8);
    std::vector<float2> h_by(n_freq * ny * 8);
    for (auto& w : h_bx) {
        w = make_float2(w_dist(rng), w_dist(rng));
    }
    for (auto& w : h_by) {
        w = make_float2(w_dist(rng), w_dist(rng));
    }

    std::vector<float2> cpu_v(n_time * n_freq * max_beams_stride, make_float2(0.0f, 0.0f));
    for (std::size_t t = 0; t < n_time; ++t) {
        for (std::size_t f = 0; f < n_freq; ++f) {
            std::complex<double> grid[8][8];
            for (std::size_t a = 0; a < n_ant; ++a) {
                const std::size_t phys = descending_order ? (63 - a) : a;
                grid[phys >> 3U][phys & 7U] = unpack_int4x2(h_packed[(t * n_freq + f) * n_ant + a]);
            }
            for (std::size_t by = 0; by < ny; ++by) {
                for (std::size_t bx = 0; bx < nx; ++bx) {
                    std::complex<double> acc(0.0, 0.0);
                    for (std::size_t r = 0; r < 8; ++r) {
                        const float2 wy_f = h_by[(by * n_freq + f) * 8 + r];
                        const std::complex<double> wy(wy_f.x, wy_f.y);
                        std::complex<double> row_acc(0.0, 0.0);
                        for (std::size_t c = 0; c < 8; ++c) {
                            const float2 wx_f = h_bx[(bx * n_freq + f) * 8 + c];
                            const std::complex<double> wx(wx_f.x, wx_f.y);
                            row_acc += grid[r][c] * wx;
                        }
                        acc += row_acc * wy;
                    }
                    cpu_v[(t * n_freq + f) * max_beams_stride + (by * nx + bx)] = make_float2(
                        static_cast<float>(acc.real()), static_cast<float>(acc.imag()));
                }
            }
        }
    }

    kotekan::int4x2_t* d_packed = nullptr;
    float2* d_bx = nullptr;
    float2* d_by = nullptr;
    float2* d_v = nullptr;
    CHECK_CUDA_ERROR_NON_OO(
        cudaMalloc(&d_packed, h_packed.size() * sizeof(kotekan::int4x2_t)));
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&d_bx, h_bx.size() * sizeof(float2)));
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&d_by, h_by.size() * sizeof(float2)));
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&d_v, cpu_v.size() * sizeof(float2)));
    CHECK_CUDA_ERROR_NON_OO(cudaMemcpy(d_packed, h_packed.data(),
                                       h_packed.size() * sizeof(kotekan::int4x2_t),
                                       cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR_NON_OO(
        cudaMemcpy(d_bx, h_bx.data(), h_bx.size() * sizeof(float2), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR_NON_OO(
        cudaMemcpy(d_by, h_by.data(), h_by.size() * sizeof(float2), cudaMemcpyHostToDevice));

    cudaStream_t stream = nullptr;
    CHECK_CUDA_ERROR_NON_OO(cudaStreamCreate(&stream));
    kotekan::launch_factorized_2d_dft_beamformer(
        d_packed, d_bx, d_by, d_v, n_time, n_freq, nx, ny, max_beams_stride, 64,
        descending_order, stream, kotekan::DirectInputFormat::INT4X2);
    CHECK_CUDA_ERROR_NON_OO(cudaStreamSynchronize(stream));

    std::vector<float2> gpu_v(cpu_v.size());
    CHECK_CUDA_ERROR_NON_OO(
        cudaMemcpy(gpu_v.data(), d_v, gpu_v.size() * sizeof(float2), cudaMemcpyDeviceToHost));

    double max_err = 0.0;
    for (std::size_t i = 0; i < cpu_v.size(); ++i) {
        const double er = static_cast<double>(gpu_v[i].x) - cpu_v[i].x;
        const double ei = static_cast<double>(gpu_v[i].y) - cpu_v[i].y;
        max_err = std::max(max_err, std::sqrt(er * er + ei * ei));
    }
    std::cout << "  - Max error vs CPU separable reference: " << std::scientific << max_err
              << std::fixed << "\n";

    cudaFree(d_packed);
    cudaFree(d_bx);
    cudaFree(d_by);
    cudaFree(d_v);
    cudaStreamDestroy(stream);
    return max_err < 1.0e-4;
}

} // namespace

int main() {
    const bool t1 = test_wmma_vs_fp32_parity_and_latency();
    const bool t2 = test_2d_fft_allsky();
    const bool t3 = test_factorized_2d_dft();
    std::cout << "\nSummary: WMMA=" << (t1 ? "PASS" : "FAIL")
              << ", 2D-FFT=" << (t2 ? "PASS" : "FAIL")
              << ", 2D-DFT=" << (t3 ? "PASS" : "FAIL") << "\n";
    return (t1 && t2 && t3) ? 0 : 1;
}
