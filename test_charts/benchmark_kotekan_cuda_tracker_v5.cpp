#include "DataType.hpp"
#include "chartsConstants.hpp"
#include "cudaBeamTrackerV5.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr double speed_of_light = kotekan::charts::constants::speed_of_light_m_per_s;
constexpr double two_pi = kotekan::charts::constants::two_pi;

// Helper to create synthetic 4-bit packed data
std::vector<kotekan::int4x2_t> generate_synthetic_data(std::size_t n_time, std::size_t n_freq,
                                                       std::size_t n_ant) {
    std::vector<kotekan::int4x2_t> data(n_time * n_freq * n_ant);
    for (std::size_t t = 0; t < n_time; ++t) {
        for (std::size_t f = 0; f < n_freq; ++f) {
            for (std::size_t a = 0; a < n_ant; ++a) {
                // Generate a simple sinusoidal pattern encoded into 4-bit signed ints
                const double phase = two_pi * (0.01 * t + 0.05 * f + 0.1 * a);
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

// CPU Reference for numerical validation of complex voltages
void cpu_reference_beam_tracker(const std::vector<kotekan::int4x2_t>& packed, float2* voltages,
                                std::size_t n_time, std::size_t n_freq, std::size_t n_ant,
                                const std::vector<double>& frequencies_hz,
                                const kotekan::BeamTrackerConfig& config) {

    for (std::size_t t = 0; t < n_time; ++t) {
        const std::size_t win = t / config.integration_spectra;
        const double center_sample =
            (static_cast<double>(win) + 0.5) * static_cast<double>(config.integration_spectra);
        const float l =
            static_cast<float>(config.trajectory.direction_start.x
                               + center_sample * config.trajectory.direction_rate_per_sample.dl);
        const float m =
            static_cast<float>(config.trajectory.direction_start.y
                               + center_sample * config.trajectory.direction_rate_per_sample.dm);
        const float trans_sq = l * l + m * m;
        const float n = (trans_sq <= 1.0f) ? std::sqrt(1.0f - trans_sq) : 0.0f;
        (void)n;

        for (std::size_t f = 0; f < n_freq; ++f) {
            const double freq_hz =
                (f < frequencies_hz.size()) ? frequencies_hz[f] : (300.0e6 + f * 300.0e3);
            const double k = two_pi * freq_hz / speed_of_light;

            double sum_r = 0.0;
            double sum_i = 0.0;

            for (std::size_t a = 0; a < n_ant; ++a) {
                // Geometry
                float pos_x = 0.0f, pos_y = 0.0f;
                if (n_ant == 32 || n_ant == 64) {
                    pos_x = static_cast<float>(a & 7) * config.spacing_m;
                    pos_y = static_cast<float>(a >> 3) * config.spacing_m;
                } else {
                    pos_x = static_cast<float>(a & 15) * config.spacing_m;
                    pos_y = static_cast<float>(a >> 4) * config.spacing_m;
                }

                const double delay_m = pos_x * l + pos_y * m;
                const double phase = k * delay_m;
                const double w_r = std::cos(phase);
                const double w_i = std::sin(phase);

                // Unpack int4
                const uint8_t byte_val = packed[(t * n_freq + f) * n_ant + a].val;
                int v_r = static_cast<int>(byte_val & 0x0F);
                if (v_r >= 8)
                    v_r -= 16;
                int v_i = static_cast<int>((byte_val >> 4) & 0x0F);
                if (v_i >= 8)
                    v_i -= 16;

                sum_r += w_r * static_cast<double>(v_r) - w_i * static_cast<double>(v_i);
                sum_i += w_r * static_cast<double>(v_i) + w_i * static_cast<double>(v_r);
            }

            voltages[t * n_freq + f] =
                make_float2(static_cast<float>(sum_r), static_cast<float>(sum_i));
        }
    }
}

bool check_tolerance_complex(const float2* ref, const float2* test, std::size_t count,
                             float rel_tol = 1e-3f, float abs_tol = 1e-4f) {
    for (std::size_t i = 0; i < count; ++i) {
        const float diff_r = std::abs(ref[i].x - test[i].x);
        const float diff_i = std::abs(ref[i].y - test[i].y);
        if ((diff_r > abs_tol && diff_r > rel_tol * std::abs(ref[i].x))
            || (diff_i > abs_tol && diff_i > rel_tol * std::abs(ref[i].y))) {
            std::fprintf(stderr, "Mismatch at [%zu]: ref=(%f, %f), test=(%f, %f)\n", i, ref[i].x,
                         ref[i].y, test[i].x, test[i].y);
            return false;
        }
    }
    return true;
}

void benchmark_case(std::size_t n_ant, std::size_t n_time, std::size_t n_freq = 336,
                    std::size_t integration_spectra = 320, int repeat = 10) {
    std::cout << "=================================================================\n";
    std::cout << "Benchmarking Kotekan V5 Complex Voltage Tracker: n_ant=" << n_ant
              << ", n_time=" << n_time << ", n_freq=" << n_freq
              << ", integration=" << integration_spectra << "\n";
    std::cout << "=================================================================\n";

    kotekan::BeamTrackerConfig config;
    config.trajectory.direction_start = {0.0f, 0.0f, 1.0f};
    config.trajectory.direction_rate_per_sample = {1.0e-5f, 0.0f};
    config.integration_spectra = integration_spectra;
    config.spacing_m = 0.6f;
    config.time_chunk_size = 80;
    config.time_unroll = 8;

    std::vector<double> freqs(n_freq);
    for (std::size_t f = 0; f < n_freq; ++f) {
        freqs[f] = 300.0e6 + f * 300.0e3;
    }

    const auto host_packed = generate_synthetic_data(n_time, n_freq, n_ant);
    const std::size_t total_out = n_time * n_freq;
    std::vector<float2> cpu_ref(total_out);
    std::vector<float2> gpu_out(total_out);

    // 1. Compute CPU reference for smaller subset / validation
    cpu_reference_beam_tracker(host_packed, cpu_ref.data(), std::min(n_time, std::size_t(640)),
                               n_freq, n_ant, freqs, config);

    // 2. Validate Direct Kernel Launch
    kotekan::int4x2_t* d_packed = nullptr;
    float2* d_voltages = nullptr;
    const std::size_t v_bytes = n_time * n_freq * n_ant * sizeof(kotekan::int4x2_t);
    const std::size_t out_bytes = total_out * sizeof(float2);

    cudaMalloc(&d_packed, v_bytes);
    cudaMalloc(&d_voltages, out_bytes);
    cudaMemcpy(d_packed, host_packed.data(), v_bytes, cudaMemcpyHostToDevice);

    kotekan::launch_beam_tracker_v5(d_packed, d_voltages, n_time, n_freq, n_ant, freqs, config);
    cudaMemcpy(gpu_out.data(), d_voltages, out_bytes, cudaMemcpyDeviceToHost);

    const bool direct_match = check_tolerance_complex(cpu_ref.data(), gpu_out.data(),
                                                      std::min(n_time, std::size_t(640)) * n_freq);
    if (!direct_match) {
        std::cout << "  [FAIL] Direct launch numerical mismatch with CPU reference!\n";
    } else {
        std::cout << "  [PASS] Direct launch numerically matches CPU reference.\n";
    }

    // 3. Validate and Benchmark CudaBeamTrackerV5Stream (Standard)
    {
        config.enable_cuda_graph = false;
        kotekan::CudaBeamTrackerV5Stream stream_engine(n_time, n_freq, n_ant, freqs, config);

        // Warmup
        for (int i = 0; i < 2; ++i) {
            stream_engine.process_batch(0, host_packed.data(), gpu_out.data());
        }

        const bool stream_match = check_tolerance_complex(
            cpu_ref.data(), gpu_out.data(), std::min(n_time, std::size_t(640)) * n_freq);
        if (!stream_match) {
            std::cout << "  [FAIL] Stream engine numerical mismatch!\n";
        } else {
            std::cout << "  [PASS] Stream engine numerically matches CPU reference.\n";
        }

        // Measure GPU kernel-only latency
        std::vector<float> kernel_times;
        for (int i = 0; i < repeat; ++i) {
            stream_engine.process_batch_device(0, d_packed, d_voltages);
            kernel_times.push_back(stream_engine.last_kernel_time_ms());
        }
        float sum_ms = 0.0f;
        for (float t : kernel_times)
            sum_ms += t;
        float avg_kernel_ms = sum_ms / repeat;

        // Measure Host-to-Host end-to-end latency
        const auto start = Clock::now();
        for (int i = 0; i < repeat; ++i) {
            stream_engine.process_batch(0, host_packed.data(), gpu_out.data());
        }
        const auto end = Clock::now();
        const double avg_e2e_ms =
            std::chrono::duration<double, std::milli>(end - start).count() / repeat;

        const double data_mb = static_cast<double>(v_bytes) / (1024.0 * 1024.0);
        const double throughput_gb_s = (data_mb / 1024.0) / (avg_kernel_ms / 1000.0);

        std::cout << std::fixed << std::setprecision(3);
        std::cout << "  -> Stream (Standard): Kernel = " << avg_kernel_ms
                  << " ms | End-to-End = " << avg_e2e_ms
                  << " ms | Compute Throughput = " << throughput_gb_s << " GB/s\n";
    }

    // 4. Benchmark CudaBeamTrackerV5Stream (with CUDA Graph)
    {
        config.enable_cuda_graph = true;
        kotekan::CudaBeamTrackerV5Stream graph_engine(n_time, n_freq, n_ant, freqs, config);

        // Warmup & capture
        for (int i = 0; i < 2; ++i) {
            graph_engine.process_batch_device(0, d_packed, d_voltages);
        }

        std::vector<float> kernel_times;
        for (int i = 0; i < repeat; ++i) {
            graph_engine.process_batch_device(0, d_packed, d_voltages);
            kernel_times.push_back(graph_engine.last_kernel_time_ms());
        }
        float sum_ms = 0.0f;
        for (float t : kernel_times)
            sum_ms += t;
        float avg_kernel_ms = sum_ms / repeat;

        const double data_mb = static_cast<double>(v_bytes) / (1024.0 * 1024.0);
        const double throughput_gb_s = (data_mb / 1024.0) / (avg_kernel_ms / 1000.0);

        std::cout << "  -> Stream (CUDA Graph): Kernel = " << avg_kernel_ms
                  << " ms | Compute Throughput = " << throughput_gb_s << " GB/s\n";
    }

    cudaFree(d_packed);
    cudaFree(d_voltages);
}

} // namespace

int main() {
    std::cout << "=================================================================\n";
    std::cout << "Kotekan V5 CUDA Beam Tracker Validation & Performance Benchmark\n";
    std::cout << "=================================================================\n\n";

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Device 0: " << prop.name << " (Compute " << prop.major << "." << prop.minor
              << ", " << (prop.totalGlobalMem / (1024 * 1024)) << " MB VRAM)\n\n";

    // Run test matrix
    benchmark_case(32, 3200);
    benchmark_case(64, 3200);
    benchmark_case(128, 3200);
    benchmark_case(256, 3200);

    std::cout << "\n=================================================================\n";
    std::cout << "Full Payload Benchmark (n_time = 15360)\n";
    std::cout << "=================================================================\n";
    benchmark_case(64, 15360);

    return 0;
}
