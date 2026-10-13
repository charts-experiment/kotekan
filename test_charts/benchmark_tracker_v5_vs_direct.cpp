#include "DataType.hpp"
#include "chartsConstants.hpp"
#include "cudaBeamTrackerV5.hpp"
#include "cudaDirectBeamTracker.hpp"
#include "cudaUtils.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::high_resolution_clock;

constexpr double SPEED_OF_LIGHT = kotekan::charts::constants::speed_of_light_m_per_s;
constexpr double TWO_PI = kotekan::charts::constants::two_pi;
constexpr double SAMPLE_TIME_US = 3.333333333333; // 8192 / 2457.6 MHz = 3.3333 us/sample

// Helper to generate synthetic 4-bit packed complex voltage data [time][freq][ant]
std::vector<kotekan::int4x2_t> generate_synthetic_data(std::size_t n_time, std::size_t n_freq,
                                                       std::size_t n_ant) {

    std::vector<kotekan::int4x2_t> data(n_time * n_freq * n_ant);
    for (std::size_t t = 0; t < n_time; ++t) {
        for (std::size_t f = 0; f < n_freq; ++f) {
            for (std::size_t a = 0; a < n_ant; ++a) {
                const double phase = TWO_PI * (0.01 * t + 0.05 * f + 0.1 * a);
                const int real_val = static_cast<int>(std::round(3.5 * std::cos(phase)));
                const int imag_val = static_cast<int>(std::round(3.5 * std::sin(phase)));
                const int clamped_r = std::max(-8, std::min(7, real_val));
                const int clamped_i = std::max(-8, std::min(7, imag_val));
                const uint8_t byte_val =
                    static_cast<uint8_t>((clamped_r & 0x0F) | ((clamped_i & 0x0F) << 4));
                data[(t * n_freq + f) * n_ant + a].val = byte_val;
            }
        }
    }
    return data;
}

// Compute antenna coordinates matching Kotekan geometry
std::vector<float3> compute_antenna_positions(std::size_t n_ant, float spacing_m) {
    std::vector<float3> pos(n_ant);
    for (std::size_t a = 0; a < n_ant; ++a) {
        const unsigned int col = (n_ant <= 64) ? (a & 7U) : (a & 15U);
        const unsigned int row = (n_ant <= 64) ? (a >> 3U) : (a >> 4U);
        pos[a] = make_float3(static_cast<float>(col) * spacing_m,
                             static_cast<float>(row) * spacing_m, 0.0f);
    }
    return pos;
}

struct BenchmarkRecord {
    std::size_t n_ant;
    std::size_t n_time;
    std::size_t n_freq;
    std::size_t n_beams;
    double v5_kernel_ms;
    double v5_eff_gb_s;
    double direct_kernel_ms;
    double direct_eff_gb_s;
    double speedup_ratio;
    double direct_budget_pct;
    double direct_realtime_factor;
    bool numerical_parity_pass;
    float max_numerical_err;
};

void run_comparison_benchmark(std::size_t n_ant, std::size_t n_time, std::size_t n_freq,
                              std::size_t n_beams, std::size_t max_beams_stride, int n_iterations,
                              std::vector<BenchmarkRecord>& results, cudaStream_t stream) {

    const float spacing_m = 0.6f;
    const std::size_t input_bytes = n_time * n_freq * n_ant * sizeof(kotekan::int4x2_t);
    const std::size_t v5_output_bytes = n_time * n_freq * sizeof(float2);
    const std::size_t direct_output_bytes = n_time * n_freq * max_beams_stride * sizeof(float2);
    const std::size_t weights_bytes = max_beams_stride * n_freq * n_ant * sizeof(float2);

    auto h_packed = generate_synthetic_data(n_time, n_freq, n_ant);
    auto h_positions = compute_antenna_positions(n_ant, spacing_m);

    std::vector<double> h_frequencies(n_freq);
    std::vector<double> h_wavenumbers(n_freq);
    for (std::size_t f = 0; f < n_freq; ++f) {
        h_frequencies[f] = 300.0e6 + f * 300.0e3;
        h_wavenumbers[f] = TWO_PI * h_frequencies[f] / SPEED_OF_LIGHT;
    }

    // Direct Beam Directions
    std::vector<kotekan::DirectDirection3D> h_direct_dirs(max_beams_stride);
    for (std::size_t b = 0; b < max_beams_stride; ++b) {
        const float l = 0.03f * static_cast<float>(b);
        const float m = -0.015f * static_cast<float>(b);
        const float r2 = l * l + m * m;
        h_direct_dirs[b] =
            kotekan::DirectDirection3D{l, m, (r2 <= 1.0f) ? std::sqrt(1.0f - r2) : 0.0f};
    }

    // Allocate Device Memory
    kotekan::int4x2_t* d_packed = nullptr;
    float2* d_v5_voltages = nullptr;
    float2* d_direct_voltages = nullptr;
    float2* d_direct_weights = nullptr;
    kotekan::DirectDirection3D* d_direct_dirs = nullptr;
    double* d_wavenumbers = nullptr;
    float3* d_positions = nullptr;

    cudaMalloc(&d_packed, input_bytes);
    cudaMalloc(&d_v5_voltages, v5_output_bytes);
    cudaMalloc(&d_direct_voltages, direct_output_bytes);
    cudaMalloc(&d_direct_weights, weights_bytes);
    cudaMalloc(&d_direct_dirs, max_beams_stride * sizeof(kotekan::DirectDirection3D));
    cudaMalloc(&d_wavenumbers, n_freq * sizeof(double));
    cudaMalloc(&d_positions, n_ant * sizeof(float3));

    cudaMemcpyAsync(d_packed, h_packed.data(), input_bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_direct_dirs, h_direct_dirs.data(),
                    max_beams_stride * sizeof(kotekan::DirectDirection3D), cudaMemcpyHostToDevice,
                    stream);
    cudaMemcpyAsync(d_wavenumbers, h_wavenumbers.data(), n_freq * sizeof(double),
                    cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_positions, h_positions.data(), n_ant * sizeof(float3), cudaMemcpyHostToDevice,
                    stream);

    // Precompute Steering Weights for Direct Tracker
    kotekan::launch_generate_steering_weights(d_direct_weights, d_direct_dirs, d_wavenumbers,
                                              d_positions, nullptr, nullptr, max_beams_stride,
                                              n_freq, n_ant, stream);
    CHECK_CUDA_ERROR_NON_OO(cudaGetLastError());
    // Opt-in persisting L2 policy (safely omitted to prevent driver reject on consumer GPUs)
    // kotekan::set_l2_persisting_weights_policy(stream, d_direct_weights, weights_bytes);
    CHECK_CUDA_ERROR_NON_OO(cudaStreamSynchronize(stream));

    // Setup V5 Config (Windowed Baseline)
    kotekan::BeamTrackerConfig v5_config;
    v5_config.trajectory.direction_start = {h_direct_dirs[0].x, h_direct_dirs[0].y,
                                            h_direct_dirs[0].z};
    v5_config.trajectory.direction_rate_per_sample = {
        0.0f, 0.0f}; // stationary target for exact parity test
    v5_config.integration_spectra = 320;
    v5_config.spacing_m = spacing_m;
    v5_config.time_chunk_size = 80;
    v5_config.time_unroll = 8;
    for (std::size_t a = 0; a < n_ant; ++a) {
        v5_config.antenna_positions[a] = h_positions[a];
    }

    cudaEvent_t start_evt, stop_evt;
    cudaEventCreate(&start_evt);
    cudaEventCreate(&stop_evt);

    // -------------------------------------------------------------------------
    // 1. Numerical Parity Check (Beam 0, stationary)
    // -------------------------------------------------------------------------
    kotekan::launch_beam_tracker_v5(d_packed, d_v5_voltages, n_time, n_freq, n_ant, h_frequencies,
                                    v5_config, stream);
    CHECK_CUDA_ERROR_NON_OO(cudaGetLastError());

    kotekan::launch_direct_beamformer(d_packed, d_direct_weights, d_direct_voltages, n_time, n_freq,
                                      n_ant, 1, max_beams_stride, 256, 4, 4, stream);
    CHECK_CUDA_ERROR_NON_OO(cudaGetLastError());
    CHECK_CUDA_ERROR_NON_OO(cudaStreamSynchronize(stream));

    const std::size_t test_check_samples = std::min(n_time, static_cast<std::size_t>(512));
    const std::size_t check_count = test_check_samples * n_freq;
    std::vector<float2> h_v5_out(check_count);
    std::vector<float2> h_direct_out(check_count * max_beams_stride);

    CHECK_CUDA_ERROR_NON_OO(cudaMemcpy(h_v5_out.data(), d_v5_voltages, check_count * sizeof(float2),
                                       cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR_NON_OO(cudaMemcpy(h_direct_out.data(), d_direct_voltages,
                                       check_count * max_beams_stride * sizeof(float2),
                                       cudaMemcpyDeviceToHost));

    float max_err = 0.0f;
    for (std::size_t i = 0; i < check_count; ++i) {
        float2 v5_v = h_v5_out[i];
        float2 dir_v = h_direct_out[i * max_beams_stride + 0]; // Beam 0
        float dr = std::abs(v5_v.x - dir_v.x);
        float di = std::abs(v5_v.y - dir_v.y);
        max_err = std::max(max_err, std::max(dr, di));
    }
    const bool parity_pass = (max_err < 1e-3f);
    if (!parity_pass) {
        std::cerr << " [PARITY FAIL: max_err=" << max_err << " V5[0]=(" << h_v5_out[0].x << ","
                  << h_v5_out[0].y << ")" << " Dir[0]=(" << h_direct_out[0].x << ","
                  << h_direct_out[0].y << ")] " << std::flush;
    }

    // -------------------------------------------------------------------------
    // 2. Benchmark Beam Tracker V5 Baseline (Windowed)
    // -------------------------------------------------------------------------
    // Warmup
    for (int w = 0; w < 3; ++w) {
        for (std::size_t b = 0; b < n_beams; ++b) {
            kotekan::launch_beam_tracker_v5(d_packed, d_v5_voltages, n_time, n_freq, n_ant,
                                            h_frequencies, v5_config, stream);
        }
    }
    CHECK_CUDA_ERROR_NON_OO(cudaGetLastError());
    CHECK_CUDA_ERROR_NON_OO(cudaStreamSynchronize(stream));

    std::vector<float> v5_times;
    for (int iter = 0; iter < n_iterations; ++iter) {
        cudaEventRecord(start_evt, stream);
        for (std::size_t b = 0; b < n_beams; ++b) {
            kotekan::launch_beam_tracker_v5(d_packed, d_v5_voltages, n_time, n_freq, n_ant,
                                            h_frequencies, v5_config, stream);
        }
        cudaEventRecord(stop_evt, stream);
        CHECK_CUDA_ERROR_NON_OO(cudaGetLastError());
        CHECK_CUDA_ERROR_NON_OO(cudaEventSynchronize(stop_evt));

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start_evt, stop_evt);
        v5_times.push_back(ms);
    }
    std::sort(v5_times.begin(), v5_times.end());
    double v5_avg_ms = std::accumulate(v5_times.begin(), v5_times.end(), 0.0) / v5_times.size();

    // V5 memory traffic: reads input buffer once per beam!
    const double v5_read_bytes = static_cast<double>(input_bytes) * n_beams;
    const double v5_write_bytes = static_cast<double>(n_time * n_freq * sizeof(float2)) * n_beams;
    const double v5_eff_gb_s = ((v5_read_bytes + v5_write_bytes) / (v5_avg_ms * 1e-3)) / 1e9;

    // -------------------------------------------------------------------------
    // 3. Benchmark Direct Beam Tracker (Fused Multi-Beam, Zero Window)
    // -------------------------------------------------------------------------
    // Warmup
    for (int w = 0; w < 3; ++w) {
        kotekan::launch_direct_beamformer(d_packed, d_direct_weights, d_direct_voltages, n_time,
                                          n_freq, n_ant, n_beams, max_beams_stride, 256, 4, 4,
                                          stream);
    }
    CHECK_CUDA_ERROR_NON_OO(cudaGetLastError());
    CHECK_CUDA_ERROR_NON_OO(cudaStreamSynchronize(stream));

    std::vector<float> direct_times;
    for (int iter = 0; iter < n_iterations; ++iter) {
        cudaEventRecord(start_evt, stream);
        kotekan::launch_direct_beamformer(d_packed, d_direct_weights, d_direct_voltages, n_time,
                                          n_freq, n_ant, n_beams, max_beams_stride, 256, 4, 4,
                                          stream);
        cudaEventRecord(stop_evt, stream);
        CHECK_CUDA_ERROR_NON_OO(cudaGetLastError());
        CHECK_CUDA_ERROR_NON_OO(cudaEventSynchronize(stop_evt));

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start_evt, stop_evt);
        direct_times.push_back(ms);
    }
    std::sort(direct_times.begin(), direct_times.end());
    double direct_avg_ms =
        std::accumulate(direct_times.begin(), direct_times.end(), 0.0) / direct_times.size();

    // Direct memory traffic: fused multi-beam register tiling reads input buffer ONCE regardless of
    // beam count!
    const double direct_read_bytes = static_cast<double>(input_bytes);
    const double direct_write_bytes =
        static_cast<double>(n_time * n_freq * n_beams * sizeof(float2));
    const double direct_eff_gb_s =
        ((direct_read_bytes + direct_write_bytes) / (direct_avg_ms * 1e-3)) / 1e9;

    const double speedup = v5_avg_ms / std::max(1e-6, direct_avg_ms);
    const double budget_ms = (static_cast<double>(n_time) * SAMPLE_TIME_US) / 1000.0;
    const double budget_pct = (direct_avg_ms / budget_ms) * 100.0;
    const double realtime_factor = budget_ms / std::max(1e-6, direct_avg_ms);

    results.push_back({n_ant, n_time, n_freq, n_beams, v5_avg_ms, v5_eff_gb_s, direct_avg_ms,
                       direct_eff_gb_s, speedup, budget_pct, realtime_factor, parity_pass,
                       max_err});

    // Clean up
    cudaFree(d_packed);
    cudaFree(d_v5_voltages);
    cudaFree(d_direct_voltages);
    cudaFree(d_direct_weights);
    cudaFree(d_direct_dirs);
    cudaFree(d_wavenumbers);
    cudaFree(d_positions);
    cudaEventDestroy(start_evt);
    cudaEventDestroy(stop_evt);
}

} // namespace

int main(int argc, char** argv) {
    std::size_t n_time =
        3840; // Default: 3,840 time samples (12.8 ms frame duration, low-VRAM 2.23 GB profile)
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--n-time" && i + 1 < argc) {
            n_time = std::stoul(argv[++i]);
        } else if (arg == "--15360" || arg == "--full-frame") {
            n_time = 15360;
        } else if (arg == "--3840" || arg == "--low-vram") {
            n_time = 3840;
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [--n-time <samples>] [--3840] [--15360]\n";
            std::cout << "  --3840, --low-vram : 3,840 time samples (12.8 ms frame, ~2.23 GB VRAM "
                         "at 256 ant) [DEFAULT]\n";
            std::cout << "  --15360, --full    : 15,360 time samples (51.2 ms frame, ~11.89 GB "
                         "VRAM at 256 ant)\n";
            return 0;
        }
    }

    std::cout << "================================================================================="
                 "=======================================\n";
    std::cout << " CHARTS Radio Telescope Beam Tracker Benchmark: Direct Beamformer vs. Beam "
                 "Tracker V5\n";
    std::cout << " Grounded in CHORD FRB Beamformer (Smith 2022) & SPOTLIGHT Multi-Beam Backend "
                 "(Gajendran et al. 2025)\n";
    std::cout << "================================================================================="
                 "=======================================\n\n";

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU Hardware Target : " << prop.name << " (Compute " << prop.major << "."
              << prop.minor << ", " << (prop.totalGlobalMem / (1024 * 1024)) << " MB VRAM, "
              << prop.multiProcessorCount << " SMs)\n";

    const double frame_budget_ms = (static_cast<double>(n_time) * SAMPLE_TIME_US) / 1000.0;
    std::cout << "Configuration Target:\n";
    std::cout << "  - Frame Time Samples : " << n_time << " (" << std::fixed << std::setprecision(2)
              << frame_budget_ms << " ms real-time frame duration)\n";
    std::cout << "  - Frequency Channels : 672 channels (full local band)\n";
    std::cout << "  - Profile Mode       : "
              << ((n_time == 3840)
                      ? "Low-VRAM & Low-Latency Profile (2.23 GB at N_ant=256, buffer_depth=3)"
                      : "Standard Extended Payload Profile")
              << "\n\n";

    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    std::vector<BenchmarkRecord> results;
    const int iterations = 15;

    // Test matrix across array scales and beam configurations
    const std::vector<std::size_t> ant_counts = {32, 64, 128, 256};
    const std::vector<std::size_t> beam_counts = {1, 2, 4};
    const std::size_t n_freq = 672; // Full local frequency band

    std::cout << "Executing Head-to-Head Benchmarks across Matrix (" << iterations
              << " iterations per configuration)...\n\n";

    for (std::size_t n_ant : ant_counts) {
        for (std::size_t n_beams : beam_counts) {
            std::cout << " -> Running N_ant=" << std::setw(3) << n_ant << ", Beams=" << n_beams
                      << ", N_time=" << n_time << ", N_freq=" << n_freq << " ... " << std::flush;

            run_comparison_benchmark(n_ant, n_time, n_freq, n_beams, 4, iterations, results,
                                     stream);
            std::cout << "DONE (Speedup: " << std::fixed << std::setprecision(2)
                      << results.back().speedup_ratio << "x)\n";
        }
    }

    std::cout << "\n";
    std::cout << "================================================================================="
                 "=======================================\n";
    std::cout << "| Ant | Beams |   V5 Time  |  Direct Time |  Speedup  |  V5 GB/s  | Direct GB/s "
                 "| Real-Time Factor | Budget % | Parity |\n";
    std::cout << "================================================================================="
                 "=======================================\n";

    for (const auto& r : results) {
        std::cout << "| " << std::setw(3) << r.n_ant << " | " << std::setw(5) << r.n_beams << " | "
                  << std::fixed << std::setprecision(3) << std::setw(8) << r.v5_kernel_ms
                  << " ms | " << std::setw(8) << r.direct_kernel_ms << " ms | "
                  << std::setprecision(2) << std::setw(7) << r.speedup_ratio << "x | "
                  << std::setprecision(1) << std::setw(7) << r.v5_eff_gb_s << " | " << std::setw(9)
                  << r.direct_eff_gb_s << " | " << std::setprecision(2) << std::setw(14)
                  << r.direct_realtime_factor << "x | " << std::setprecision(1) << std::setw(6)
                  << r.direct_budget_pct << "% | "
                  << (r.numerical_parity_pass ? " PASS " : " FAIL ") << "|\n";
    }
    std::cout << "================================================================================="
                 "=======================================\n\n";

    // Export to JSON for downstream plotting & Slurm integration
    const std::string json_filename = "tracker_benchmark_comparison.json";
    std::ofstream jout(json_filename);
    if (jout.is_open()) {
        jout << "{\n";
        jout << "  \"gpu_device\": \"" << prop.name << "\",\n";
        jout << "  \"n_iterations\": " << iterations << ",\n";
        jout << "  \"n_time\": " << n_time << ",\n";
        jout << "  \"frame_budget_ms\": " << frame_budget_ms << ",\n";
        jout << "  \"records\": [\n";
        for (std::size_t i = 0; i < results.size(); ++i) {
            const auto& r = results[i];
            jout << "    {\n";
            jout << "      \"n_ant\": " << r.n_ant << ",\n";
            jout << "      \"n_time\": " << r.n_time << ",\n";
            jout << "      \"n_freq\": " << r.n_freq << ",\n";
            jout << "      \"n_beams\": " << r.n_beams << ",\n";
            jout << "      \"v5_kernel_ms\": " << r.v5_kernel_ms << ",\n";
            jout << "      \"v5_eff_gb_s\": " << r.v5_eff_gb_s << ",\n";
            jout << "      \"direct_kernel_ms\": " << r.direct_kernel_ms << ",\n";
            jout << "      \"direct_eff_gb_s\": " << r.direct_eff_gb_s << ",\n";
            jout << "      \"speedup_ratio\": " << r.speedup_ratio << ",\n";
            jout << "      \"direct_budget_pct\": " << r.direct_budget_pct << ",\n";
            jout << "      \"direct_realtime_factor\": " << r.direct_realtime_factor << ",\n";
            jout << "      \"numerical_parity_pass\": "
                 << (r.numerical_parity_pass ? "true" : "false") << ",\n";
            jout << "      \"max_numerical_err\": " << r.max_numerical_err << "\n";
            jout << "    }" << (i + 1 < results.size() ? "," : "") << "\n";
        }
        jout << "  ]\n";
        jout << "}\n";
        jout.close();
        std::cout << "Successfully exported benchmark records to: " << json_filename << "\n";
    }

    cudaStreamDestroy(stream);
    return 0;
}
