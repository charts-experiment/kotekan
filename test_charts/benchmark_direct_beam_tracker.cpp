#include "DataType.hpp"
#include "chartsConstants.hpp"
#include "cudaDirectBeamTracker.hpp"

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

constexpr double speed_of_light = kotekan::charts::constants::speed_of_light_m_per_s;
constexpr double two_pi = kotekan::charts::constants::two_pi;

// Helper to generate synthetic 4-bit packed complex voltage data
std::vector<kotekan::int4x2_t> generate_synthetic_data(std::size_t n_time, std::size_t n_freq,
                                                       std::size_t n_ant) {

    std::vector<kotekan::int4x2_t> data(n_time * n_freq * n_ant);
    for (std::size_t t = 0; t < n_time; ++t) {
        for (std::size_t f = 0; f < n_freq; ++f) {
            for (std::size_t a = 0; a < n_ant; ++a) {
                const double phase = two_pi * (0.01 * t + 0.05 * f + 0.1 * a);
                const int real_val = static_cast<int>(std::round(3.5 * std::cos(phase)));
                const int imag_val = static_cast<int>(std::round(3.5 * std::sin(phase)));
                const int clamped_r = std::max(-7, std::min(7, real_val));
                const int clamped_i = std::max(-7, std::min(7, imag_val));
                const uint8_t byte_val =
                    static_cast<uint8_t>((clamped_r & 0x0F) | ((clamped_i & 0x0F) << 4));
                data[(t * n_freq + f) * n_ant + a].val = byte_val;
            }
        }
    }
    return data;
}

// CPU Reference for direct beamforming numerical validation
void cpu_reference_direct_beamformer(const std::vector<kotekan::int4x2_t>& packed,
                                     const std::vector<float2>& weights,
                                     std::vector<float2>& voltages, std::size_t n_time,
                                     std::size_t n_freq, std::size_t n_ant, std::size_t num_beams,
                                     std::size_t max_beams_stride) {

    voltages.resize(n_time * n_freq * max_beams_stride);

    for (std::size_t t = 0; t < n_time; ++t) {
        for (std::size_t f = 0; f < n_freq; ++f) {
            for (std::size_t b = 0; b < num_beams; ++b) {
                double sum_r = 0.0;
                double sum_i = 0.0;

                const std::size_t w_base = (b * n_freq + f) * n_ant;

                for (std::size_t a = 0; a < n_ant; ++a) {
                    const float2 w = weights[w_base + a];

                    const uint8_t byte_val = packed[(t * n_freq + f) * n_ant + a].val;
                    int v_r = static_cast<int>(byte_val & 0x0F);
                    if (v_r >= 8)
                        v_r -= 16;
                    int v_i = static_cast<int>((byte_val >> 4) & 0x0F);
                    if (v_i >= 8)
                        v_i -= 16;

                    sum_r += static_cast<double>(w.x) * static_cast<double>(v_r)
                             - static_cast<double>(w.y) * static_cast<double>(v_i);
                    sum_i += static_cast<double>(w.x) * static_cast<double>(v_i)
                             + static_cast<double>(w.y) * static_cast<double>(v_r);
                }

                const std::size_t out_idx = (t * n_freq + f) * max_beams_stride + b;
                voltages[out_idx] =
                    make_float2(static_cast<float>(sum_r), static_cast<float>(sum_i));
            }
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    std::size_t n_time = 3840;      // Default: 3,840 samples (~12.8 ms)
    std::size_t n_freq = 336;       // Default: 336 channels
    std::size_t n_ant = 64;         // Default: 64 antennas
    std::size_t max_beams = 16;     // Default stride
    std::size_t n_iterations = 25;  // Default iterations
    std::vector<std::size_t> test_beam_counts;
    std::string json_output = "";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--n-time" || arg == "-t") && i + 1 < argc) {
            n_time = std::stoul(argv[++i]);
        } else if (arg == "--15360" || arg == "--full") {
            n_time = 15360;
        } else if (arg == "--3840" || arg == "--low-vram") {
            n_time = 3840;
        } else if ((arg == "--n-ant" || arg == "-a") && i + 1 < argc) {
            n_ant = std::stoul(argv[++i]);
        } else if ((arg == "--n-freq" || arg == "-f") && i + 1 < argc) {
            n_freq = std::stoul(argv[++i]);
        } else if (arg == "--max-beams" && i + 1 < argc) {
            max_beams = std::stoul(argv[++i]);
        } else if ((arg == "--beams" || arg == "-b") && i + 1 < argc) {
            std::string beams_str = argv[++i];
            std::stringstream ss(beams_str);
            std::string item;
            while (std::getline(ss, item, ',')) {
                if (!item.empty()) {
                    test_beam_counts.push_back(std::stoul(item));
                }
            }
        } else if (arg == "--iterations" && i + 1 < argc) {
            n_iterations = std::stoul(argv[++i]);
        } else if (arg == "--json-output" && i + 1 < argc) {
            json_output = argv[++i];
        }
    }

    if (test_beam_counts.empty()) {
        for (std::size_t b : {1, 2, 4, 10, 16, 64}) {
            if (b <= max_beams) test_beam_counts.push_back(b);
        }
        if (test_beam_counts.empty()) test_beam_counts.push_back(max_beams);
    } else {
        std::size_t highest = 0;
        for (std::size_t b : test_beam_counts) {
            if (b > highest) highest = b;
        }
        if (highest > max_beams) max_beams = highest;
    }

    std::cout << "================================================================================="
                 "==================\n";
    std::cout << " CHARTS Direct Beam Tracker (Zero Integration Window) Benchmark & Numerical "
                 "Validation\n";
    std::cout << "================================================================================="
                 "==================\n\n";

    const double real_time_budget_ms = (static_cast<double>(n_time) * 3.333333333333) / 1000.0;

    const std::size_t input_bytes = n_time * n_freq * n_ant * sizeof(kotekan::int4x2_t);
    const std::size_t output_bytes = n_time * n_freq * max_beams * sizeof(float2);
    const std::size_t weights_bytes = max_beams * n_freq * n_ant * sizeof(float2);

    std::cout << "Configuration Parameters:\n";
    std::cout << "  - Time Samples per Frame (n_time)    : " << n_time << "\n";
    std::cout << "  - Frequency Channels (n_freq)        : " << n_freq << "\n";
    std::cout << "  - Antenna Elements (n_ant)           : " << n_ant << "\n";
    std::cout << "  - Max Beams (Allocated Stride)       : " << max_beams << "\n";
    std::cout << "  - Real-Time Frame Duration (Budget)  : " << std::fixed << std::setprecision(2)
              << real_time_budget_ms << " ms\n";
    std::cout << "  - Input Buffer Size                  : " << (input_bytes / (1024.0 * 1024.0))
              << " MB\n";
    std::cout << "  - Output Buffer Size                 : " << (output_bytes / (1024.0 * 1024.0))
              << " MB\n";
    std::cout << "  - Weights Buffer Size                : " << (weights_bytes / 1024.0)
              << " KB\n\n";

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Device: " << prop.name << " (" << (prop.totalGlobalMem / (1024 * 1024))
              << " MB VRAM)\n\n";

    // VRAM Safety Check before allocation
    std::size_t vram_free_before = 0, vram_total = 0;
    cudaMemGetInfo(&vram_free_before, &vram_total);
    const std::size_t required_bytes = input_bytes + output_bytes + weights_bytes;
    if (required_bytes > static_cast<std::size_t>(vram_free_before * 0.95)) {
        std::cerr << "OUT_OF_VRAM: Required allocations (" << (required_bytes / (1024.0 * 1024.0))
                  << " MB) exceed physical GPU headroom (" << (vram_free_before / (1024.0 * 1024.0))
                  << " MB free).\n";
        return 2;
    }

    // Prepare synthetic input data
    std::cout << "[1/4] Generating synthetic test data..." << std::flush;
    auto h_packed = generate_synthetic_data(n_time, n_freq, n_ant);
    std::cout << " DONE.\n";

    // Setup host weights
    std::cout << "[2/4] Setting up steering directions and wavenumbers..." << std::flush;
    std::vector<kotekan::DirectDirection3D> h_dirs(max_beams);
    for (std::size_t b = 0; b < max_beams; ++b) {
        const float l = 0.05f * static_cast<float>(b);
        const float m = -0.02f * static_cast<float>(b);
        const float r2 = l * l + m * m;
        h_dirs[b] = kotekan::DirectDirection3D{l, m, (r2 <= 1.0f) ? std::sqrt(1.0f - r2) : 0.0f};
    }

    std::vector<double> h_wavenumbers(n_freq);
    for (std::size_t f = 0; f < n_freq; ++f) {
        const double freq_hz = 300.0e6 + f * 300.0e3;
        h_wavenumbers[f] = two_pi * freq_hz / speed_of_light;
    }

    std::vector<float3> h_positions(n_ant);
    for (std::size_t a = 0; a < n_ant; ++a) {
        const unsigned int col = a & 7U;
        const unsigned int row = a >> 3U;
        h_positions[a] =
            make_float3(static_cast<float>(col) * 0.6f, static_cast<float>(row) * 0.6f, 0.0f);
    }
    std::cout << " DONE.\n";

    // Allocate GPU buffers
    kotekan::int4x2_t* d_packed = nullptr;
    float2* d_voltages = nullptr;
    float2* d_weights = nullptr;
    kotekan::DirectDirection3D* d_dirs = nullptr;
    double* d_wavenumbers = nullptr;
    float3* d_positions = nullptr;

    cudaMalloc(&d_packed, input_bytes);
    cudaMalloc(&d_voltages, output_bytes);
    cudaMalloc(&d_weights, weights_bytes);
    cudaMalloc(&d_dirs, max_beams * sizeof(kotekan::DirectDirection3D));
    cudaMalloc(&d_wavenumbers, n_freq * sizeof(double));
    cudaMalloc(&d_positions, n_ant * sizeof(float3));

    std::size_t vram_free_after = 0;
    cudaMemGetInfo(&vram_free_after, &vram_total);
    const double vram_allocated_mb = static_cast<double>(vram_free_before - vram_free_after) / (1024.0 * 1024.0);

    cudaMemcpy(d_packed, h_packed.data(), input_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dirs, h_dirs.data(), max_beams * sizeof(kotekan::DirectDirection3D),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_wavenumbers, h_wavenumbers.data(), n_freq * sizeof(double),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_positions, h_positions.data(), n_ant * sizeof(float3), cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    // Compute GPU weights
    kotekan::launch_generate_steering_weights(d_weights, d_dirs, d_wavenumbers, d_positions,
                                              nullptr, nullptr, max_beams, n_freq, n_ant, stream);
    cudaStreamSynchronize(stream);

    // Read back weights for CPU reference validation
    std::vector<float2> h_weights(max_beams * n_freq * n_ant);
    cudaMemcpy(h_weights.data(), d_weights, weights_bytes, cudaMemcpyDeviceToHost);

    // Run numerical equivalence test against CPU reference (on first 512 samples)
    std::cout
        << "[3/4] Validating numerical equivalence against CPU reference (first 512 samples)..."
        << std::flush;
    const std::size_t test_samples = std::min(n_time, std::size_t(512));
    std::vector<kotekan::int4x2_t> h_test_packed(h_packed.begin(),
                                                 h_packed.begin() + test_samples * n_freq * n_ant);
    std::vector<float2> cpu_test_voltages;
    cpu_reference_direct_beamformer(h_test_packed, h_weights, cpu_test_voltages, test_samples,
                                    n_freq, n_ant, max_beams, max_beams);

    // Run on GPU with fused multi-beam register tiling (B_tile = 4)
    kotekan::launch_direct_beamformer(d_packed, d_weights, d_voltages, test_samples, n_freq, n_ant,
                                      max_beams, max_beams, 256, 4, 4, stream);
    cudaStreamSynchronize(stream);

    std::vector<float2> gpu_test_voltages(test_samples * n_freq * max_beams);
    cudaMemcpy(gpu_test_voltages.data(), d_voltages,
               test_samples * n_freq * max_beams * sizeof(float2), cudaMemcpyDeviceToHost);

    double sum_diff_sq = 0.0;
    double sum_ref_sq = 0.0;
    double max_phase_err_deg = 0.0;
    for (std::size_t i = 0; i < gpu_test_voltages.size(); ++i) {
        const double dr = static_cast<double>(gpu_test_voltages[i].x) - static_cast<double>(cpu_test_voltages[i].x);
        const double di = static_cast<double>(gpu_test_voltages[i].y) - static_cast<double>(cpu_test_voltages[i].y);
        sum_diff_sq += dr * dr + di * di;
        const double r_sq = static_cast<double>(cpu_test_voltages[i].x) * cpu_test_voltages[i].x +
                            static_cast<double>(cpu_test_voltages[i].y) * cpu_test_voltages[i].y;
        sum_ref_sq += r_sq;
        if (r_sq > 1.0) {
            double pg = std::atan2(static_cast<double>(gpu_test_voltages[i].y), static_cast<double>(gpu_test_voltages[i].x));
            double pc = std::atan2(static_cast<double>(cpu_test_voltages[i].y), static_cast<double>(cpu_test_voltages[i].x));
            double dp = std::abs(pg - pc);
            if (dp > M_PI) dp = 2.0 * M_PI - dp;
            max_phase_err_deg = std::max(max_phase_err_deg, dp * (180.0 / M_PI));
        }
    }
    const double rel_l2_err = std::sqrt(sum_diff_sq / std::max(1e-12, sum_ref_sq));
    const double snr_db = 10.0 * std::log10(sum_ref_sq / std::max(1e-12, sum_diff_sq));

    if (rel_l2_err < 1.0e-4 && max_phase_err_deg < 0.01) {
        std::cout << " PASSED! (Rel L2: " << std::scientific << rel_l2_err
                  << ", MaxPhase: " << std::fixed << std::setprecision(4) << max_phase_err_deg << " deg, SNR: "
                  << std::setprecision(1) << snr_db << " dB)\n\n";
    } else {
        std::cout << " FAILED! (Rel L2: " << rel_l2_err << ", MaxPhase: " << max_phase_err_deg << " deg)\n\n";
    }

    // Benchmark across beam counts
    std::cout << "[4/4] Benchmarking Direct Beam Tracker throughput (Blackwell SM120 Fused "
                 "Multi-Beam)...\n\n";

    cudaEvent_t start_evt, stop_evt;
    cudaEventCreate(&start_evt);
    cudaEventCreate(&stop_evt);

    std::cout << "---------------------------------------------------------------------------------"
                 "----------------------------------------------------\n";
    std::cout << "| Active Beams |  Min (ms)  |  Avg (ms)  |  Med (ms)  |  P99 (ms)  |  Max (ms)  | Budget % "
                 "| Realtime Speedup | Eff. GB/s | GFLOPS   |\n";
    std::cout << "---------------------------------------------------------------------------------"
                 "----------------------------------------------------\n";

    struct BeamBenchmarkRecord {
        std::size_t active_beams = 0;
        double min_ms = 0.0;
        double avg_ms = 0.0;
        double med_ms = 0.0;
        double p99_ms = 0.0;
        double max_ms = 0.0;
        double eff_bandwidth_gb_s = 0.0;
        double compute_gflops = 0.0;
        double budget_utilization_pct = 0.0;
        double realtime_speedup = 0.0;
    };
    std::vector<BeamBenchmarkRecord> beam_records;

    for (std::size_t b_count : test_beam_counts) {
        // Warmup
        for (int w = 0; w < 5; ++w) {
            kotekan::launch_direct_beamformer(d_packed, d_weights, d_voltages, n_time, n_freq,
                                              n_ant, b_count, max_beams, 256, 4, 4, stream);
        }
        cudaStreamSynchronize(stream);

        std::vector<double> runs;
        for (std::size_t iter = 0; iter < n_iterations; ++iter) {
            cudaEventRecord(start_evt, stream);
            kotekan::launch_direct_beamformer(d_packed, d_weights, d_voltages, n_time, n_freq,
                                              n_ant, b_count, max_beams, 256, 4, 4, stream);
            cudaEventRecord(stop_evt, stream);
            cudaEventSynchronize(stop_evt);

            float ms = 0.0f;
            cudaEventElapsedTime(&ms, start_evt, stop_evt);
            runs.push_back(ms);
        }

        std::sort(runs.begin(), runs.end());
        double min_ms = runs.front();
        double max_ms = runs.back();
        double med_ms = runs[runs.size() / 2];
        std::size_t p99_idx = std::min(runs.size() - 1, static_cast<std::size_t>(runs.size() * 0.99));
        double p99_ms = runs[p99_idx];
        double sum = std::accumulate(runs.begin(), runs.end(), 0.0);
        double avg_ms = sum / runs.size();

        const double read_bytes = static_cast<double>(input_bytes);
        const double write_bytes = static_cast<double>(n_time * n_freq * b_count * sizeof(float2));
        const double total_effective_bytes = read_bytes + write_bytes;
        const double eff_bandwidth_gb_s = (total_effective_bytes / (avg_ms * 1e-3)) / 1e9;
        const double realtime_factor = real_time_budget_ms / avg_ms;
        const double total_mac_ops = static_cast<double>(n_time) * n_freq * n_ant * b_count;
        const double compute_gflops = (total_mac_ops * 8.0) / (avg_ms * 1e-3) / 1e9;
        const double budget_pct = (avg_ms / real_time_budget_ms) * 100.0;

        BeamBenchmarkRecord rec;
        rec.active_beams = b_count;
        rec.min_ms = min_ms;
        rec.avg_ms = avg_ms;
        rec.med_ms = med_ms;
        rec.p99_ms = p99_ms;
        rec.max_ms = max_ms;
        rec.eff_bandwidth_gb_s = eff_bandwidth_gb_s;
        rec.compute_gflops = compute_gflops;
        rec.budget_utilization_pct = budget_pct;
        rec.realtime_speedup = realtime_factor;
        beam_records.push_back(rec);

        std::string label = std::to_string(b_count) + " Beam" + (b_count > 1 ? "s" : "");
        std::cout << "| " << std::left << std::setw(12) << label << " | " << std::right
                  << std::fixed << std::setprecision(3) << std::setw(10) << min_ms << " | "
                  << std::setw(10) << avg_ms << " | " << std::setw(10) << med_ms << " | "
                  << std::setw(10) << p99_ms << " | " << std::setw(10) << max_ms << " | "
                  << std::setw(7) << budget_pct << "% | " << std::setw(15)
                  << realtime_factor << "x | " << std::setw(9) << eff_bandwidth_gb_s << " | "
                  << std::setw(8) << compute_gflops << " |\n";
    }
    std::cout << "---------------------------------------------------------------------------------"
                 "----------------------------------------------------\n\n";

    if (!json_output.empty()) {
        std::ofstream jf(json_output);
        if (jf.is_open()) {
            jf << "{\n";
            jf << "  \"device\": \"" << prop.name << "\",\n";
            jf << "  \"vram_total_mb\": " << (prop.totalGlobalMem / (1024 * 1024)) << ",\n";
            jf << "  \"vram_allocated_mb\": " << vram_allocated_mb << ",\n";
            jf << "  \"n_ant\": " << n_ant << ",\n";
            jf << "  \"n_freq\": " << n_freq << ",\n";
            jf << "  \"n_time\": " << n_time << ",\n";
            jf << "  \"max_beams\": " << max_beams << ",\n";
            jf << "  \"frame_budget_ms\": " << real_time_budget_ms << ",\n";
            jf << "  \"rel_l2_error\": " << rel_l2_err << ",\n";
            jf << "  \"max_phase_err_deg\": " << max_phase_err_deg << ",\n";
            jf << "  \"snr_db\": " << snr_db << ",\n";
            jf << "  \"benchmarks\": [\n";
            for (std::size_t i = 0; i < beam_records.size(); ++i) {
                const auto& b = beam_records[i];
                jf << "    {\n";
                jf << "      \"active_beams\": " << b.active_beams << ",\n";
                jf << "      \"min_ms\": " << b.min_ms << ",\n";
                jf << "      \"avg_ms\": " << b.avg_ms << ",\n";
                jf << "      \"med_ms\": " << b.med_ms << ",\n";
                jf << "      \"p99_ms\": " << b.p99_ms << ",\n";
                jf << "      \"max_ms\": " << b.max_ms << ",\n";
                jf << "      \"eff_bandwidth_gb_s\": " << b.eff_bandwidth_gb_s << ",\n";
                jf << "      \"compute_gflops\": " << b.compute_gflops << ",\n";
                jf << "      \"budget_utilization_pct\": " << b.budget_utilization_pct << ",\n";
                jf << "      \"realtime_speedup\": " << b.realtime_speedup << "\n";
                jf << "    }" << (i + 1 < beam_records.size() ? "," : "") << "\n";
            }
            jf << "  ]\n";
            jf << "}\n";
            jf.close();
            std::cout << "[JSON] Successfully wrote benchmark metrics to: " << json_output << "\n";
        }
    }

    // Clean up
    cudaFree(d_packed);
    cudaFree(d_voltages);
    cudaFree(d_weights);
    cudaFree(d_dirs);
    cudaFree(d_wavenumbers);
    cudaFree(d_positions);
    cudaEventDestroy(start_evt);
    cudaEventDestroy(stop_evt);
    cudaStreamDestroy(stream);

    std::cout << "Direct Beam Tracker benchmark completed successfully.\n";
    return 0;
}
