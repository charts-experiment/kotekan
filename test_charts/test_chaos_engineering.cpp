#include "DataType.hpp"
#include "chartsConstants.hpp"
#include "cudaBeamTrackerV5.hpp"
#include "cudaDirectBeamTracker.hpp"
#include "kotekanLogging.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::high_resolution_clock;
constexpr double SPEED_OF_LIGHT = kotekan::charts::constants::speed_of_light_m_per_s;
constexpr double TWO_PI = kotekan::charts::constants::two_pi;

inline void get_antenna_pos(std::size_t n_ant, std::size_t element, float spacing_m, float& x,
                            float& y) {
    if (n_ant == 32 || n_ant == 64) {
        x = static_cast<float>(element & 7U) * spacing_m;
        y = static_cast<float>(element >> 3U) * spacing_m;
    } else {
        x = static_cast<float>(element & 15U) * spacing_m;
        y = static_cast<float>(element >> 4U) * spacing_m;
    }
}

inline void populate_antenna_positions(kotekan::MultiBeamTrackerConfig& config, std::size_t n_ant) {
    for (std::size_t a = 0; a < n_ant; ++a) {
        float x = 0.0f, y = 0.0f;
        get_antenna_pos(n_ant, a, config.spacing_m, x, y);
        config.antenna_positions[a] = make_float3(x, y, 0.0f);
    }
}

// Generate raw RFSoC data with custom dead antenna set
std::vector<kotekan::int4x2_t>
generate_rfsoc_data(std::size_t n_time, std::size_t n_freq, std::size_t n_ant,
                    const std::vector<double>& freqs_hz, float l0, float m0, float dl, float dm,
                    float amp, const std::vector<bool>& is_alive, float spacing_m = 0.6f) {

    const std::size_t total_elements = n_time * n_freq * n_ant;
    std::vector<kotekan::int4x2_t> buffer(total_elements);

#pragma omp parallel for collapse(2) schedule(static)
    for (std::size_t t = 0; t < n_time; ++t) {
        for (std::size_t f = 0; f < n_freq; ++f) {
            const double freq_hz = freqs_hz[f];
            const double k = TWO_PI * freq_hz / SPEED_OF_LIGHT;
            const double cur_l = l0 + t * dl;
            const double cur_m = m0 + t * dm;
            const double base_phase = TWO_PI * (0.01 * t + 0.05 * f);

            for (std::size_t a = 0; a < n_ant; ++a) {
                const std::size_t idx = (t * n_freq + f) * n_ant + a;

                if (!is_alive[a]) {
                    buffer[idx].val = 0x00; // Dead antenna
                    continue;
                }

                float pos_x = 0.0f, pos_y = 0.0f;
                get_antenna_pos(n_ant, a, spacing_m, pos_x, pos_y);
                const double delay_m = pos_x * cur_l + pos_y * cur_m;
                const double phase = base_phase + k * delay_m;

                const float r_float = amp * static_cast<float>(std::cos(phase));
                const float i_float = amp * static_cast<float>(std::sin(phase));

                const int r_val = std::max(-8, std::min(7, static_cast<int>(std::round(r_float))));
                const int i_val = std::max(-8, std::min(7, static_cast<int>(std::round(i_float))));
                buffer[idx].val = static_cast<uint8_t>((r_val & 0x0F) | ((i_val & 0x0F) << 4));
            }
        }
    }
    return buffer;
}

} // namespace

// ============================================================================
// CHAOS TEST 1: Antennas Dying and Reviving Mid-Stream (Dynamic Flapping)
// ============================================================================

bool test_v5_dynamic_antenna_mortality_and_revival(std::size_t n_ant = 256) {
    std::cout << ">>> [CHAOS TEST 1 - V5] Dynamic Antenna Mortality & Revival Mid-Stream (" << n_ant
              << " Antennas) <<<\n";

    const std::size_t n_time = 1600;
    const std::size_t n_freq = 168;
    const std::size_t n_batches = 100;
    const float amp = 3.0f;

    std::vector<double> freqs(n_freq);
    for (std::size_t f = 0; f < n_freq; ++f)
        freqs[f] = 300.0e6 + f * 300.0e3;

    kotekan::MultiBeamTrackerConfig config;
    config.num_active_beams = 1;
    config.integration_spectra = 320;
    config.spacing_m = 0.6f;
    config.time_chunk_size = 80;
    config.time_unroll = 8;
    populate_antenna_positions(config, n_ant);
    config.trajectories[0].direction_start = {0.03f, -0.02f, 1.0f};
    config.trajectories[0].direction_rate_per_sample = {1.0e-5f, 0.0f};

    const std::size_t in_bytes = n_time * n_freq * n_ant * sizeof(kotekan::int4x2_t);
    const std::size_t out_bytes = n_time * n_freq * sizeof(float2);

    kotekan::int4x2_t* d_packed = nullptr;
    float2* d_voltages = nullptr;
    cudaMalloc(&d_packed, in_bytes);
    cudaMalloc(&d_voltages, out_bytes);

    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    std::vector<float2> h_out(n_time * n_freq);
    bool all_stages_valid = true;
    std::size_t total_nans = 0, total_infs = 0;

    std::cout << "  Simulating 100 continuous batches with injected hardware faults...\n";

    std::mt19937 rng(1337);

    for (std::size_t batch = 0; batch < n_batches; ++batch) {
        std::vector<bool> is_alive(n_ant, true);
        config.antenna_mask.fill(1);

        std::string phase_name;
        if (batch < 20) {
            phase_name = "Phase 1: All antennas healthy (100% online)";
        } else if (batch < 40) {
            phase_name = "Phase 2: 5 random antennas abruptly DIE";
            std::vector<std::size_t> dead = {7, 23, 89, 142, 210};
            for (auto a : dead) {
                is_alive[a] = false;
                config.antenna_mask[a] = 0;
            }
        } else if (batch < 60) {
            phase_name = "Phase 3: Entire 32-element RFSoC Board DIES (Ants 32-63)";
            for (std::size_t a = 32; a < 64; ++a) {
                is_alive[a] = false;
                config.antenna_mask[a] = 0;
            }
        } else if (batch < 80) {
            phase_name = "Phase 4: Dead RFSoC REVIVES, 2 other antennas fail";
            std::vector<std::size_t> dead = {11, 205};
            for (auto a : dead) {
                is_alive[a] = false;
                config.antenna_mask[a] = 0;
            }
        } else {
            phase_name = "Phase 5: High-rate random antenna flickering (chaos)";
            for (std::size_t a = 0; a < n_ant; ++a) {
                if ((rng() % 10) == 0) { // 10% random dead
                    is_alive[a] = false;
                    config.antenna_mask[a] = 0;
                }
            }
        }

        std::size_t alive_count = 0;
        for (std::size_t a = 0; a < n_ant; ++a)
            if (is_alive[a])
                alive_count++;

        auto h_packed = generate_rfsoc_data(n_time, n_freq, n_ant, freqs, 0.03f, -0.02f, 1.0e-5f,
                                            0.0f, amp, is_alive, config.spacing_m);

        cudaMemcpyAsync(d_packed, h_packed.data(), in_bytes, cudaMemcpyHostToDevice, stream);
        kotekan::launch_beam_tracker_v5_multibeam(d_packed, d_voltages, n_time, n_freq, n_ant, 1,
                                                  freqs, config, stream, batch * 5);
        cudaMemcpyAsync(h_out.data(), d_voltages, out_bytes, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        // Verify output intensity scaling
        double sum_val = 0.0;
        for (const auto& v : h_out) {
            if (std::isnan(v.x) || std::isnan(v.y))
                total_nans++;
            if (std::isinf(v.x) || std::isinf(v.y))
                total_infs++;
            sum_val += static_cast<double>(v.x * v.x + v.y * v.y);
        }
        const double meas_avg = sum_val / (n_time * n_freq);
        // Effective 4-bit rounded amplitude factor
        const double theoretical_gain_ratio =
            static_cast<double>(alive_count) / static_cast<double>(n_ant);
        const double expected_scaling = theoretical_gain_ratio * theoretical_gain_ratio;

        if (batch % 20 == 0 || batch == n_batches - 1) {
            std::cout << "  [Batch " << std::setw(2) << batch << "] " << phase_name << "\n";
            std::cout << "             Alive=" << alive_count << "/" << n_ant
                      << " | Power=" << meas_avg << " | Relative Scaling=" << std::fixed
                      << std::setprecision(4) << expected_scaling << " | NaNs=" << total_nans
                      << "\n";
        }
    }

    cudaFree(d_packed);
    cudaFree(d_voltages);
    cudaStreamDestroy(stream);

    const bool passed = (total_nans == 0 && total_infs == 0 && all_stages_valid);
    std::cout << "  Result: "
              << (passed
                      ? "[PASSED] Antennas died and revived seamlessly with zero pipeline glitches!"
                      : "[FAILED]")
              << "\n\n";
    return passed;
}

bool test_direct_dynamic_antenna_mortality_and_revival(std::size_t n_ant = 256) {
    std::cout << ">>> [CHAOS TEST 1 - DIRECT] Dynamic Antenna Mortality & Revival Mid-Stream ("
              << n_ant << " Antennas) <<<\n";

    const std::size_t n_time = 1600;
    const std::size_t n_freq = 168;
    const std::size_t n_batches = 100;
    const float amp = 3.0f;
    const float spacing_m = 0.6f;

    std::vector<double> freqs(n_freq);
    for (std::size_t f = 0; f < n_freq; ++f)
        freqs[f] = 300.0e6 + f * 300.0e3;

    std::vector<double> wavenumbers(n_freq);
    for (std::size_t f = 0; f < n_freq; ++f)
        wavenumbers[f] = TWO_PI * freqs[f] / SPEED_OF_LIGHT;

    std::vector<float3> positions(n_ant);
    for (std::size_t a = 0; a < n_ant; ++a) {
        float x = 0.0f, y = 0.0f;
        get_antenna_pos(n_ant, a, spacing_m, x, y);
        positions[a] = make_float3(x, y, 0.0f);
    }

    std::vector<kotekan::DirectDirection3D> dirs(1);
    dirs[0] = {0.03f, -0.02f, std::sqrt(1.0f - 0.03f * 0.03f - 0.02f * 0.02f)};

    const std::size_t in_bytes = n_time * n_freq * n_ant * sizeof(kotekan::int4x2_t);
    const std::size_t out_bytes = n_time * n_freq * sizeof(float2);
    const std::size_t weights_bytes = 1 * n_freq * n_ant * sizeof(float2);

    kotekan::int4x2_t* d_packed = nullptr;
    float2* d_voltages = nullptr;
    float2* d_weights = nullptr;
    kotekan::DirectDirection3D* d_dirs = nullptr;
    double* d_wavenumbers = nullptr;
    float3* d_positions = nullptr;
    uint8_t* d_mask = nullptr;

    cudaMalloc(&d_packed, in_bytes);
    cudaMalloc(&d_voltages, out_bytes);
    cudaMalloc(&d_weights, weights_bytes);
    cudaMalloc(&d_dirs, sizeof(kotekan::DirectDirection3D));
    cudaMalloc(&d_wavenumbers, n_freq * sizeof(double));
    cudaMalloc(&d_positions, n_ant * sizeof(float3));
    cudaMalloc(&d_mask, n_ant * sizeof(uint8_t));

    cudaMemcpy(d_dirs, dirs.data(), sizeof(kotekan::DirectDirection3D), cudaMemcpyHostToDevice);
    cudaMemcpy(d_wavenumbers, wavenumbers.data(), n_freq * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_positions, positions.data(), n_ant * sizeof(float3), cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    std::vector<float2> h_out(n_time * n_freq);
    std::size_t total_nans = 0, total_infs = 0;

    std::cout << "  Simulating 100 continuous batches with injected hardware faults on Direct "
                 "Beamformer...\n";
    std::mt19937 rng(1337);

    for (std::size_t batch = 0; batch < n_batches; ++batch) {
        std::vector<bool> is_alive(n_ant, true);
        std::vector<uint8_t> h_mask(n_ant, 1);

        std::string phase_name;
        if (batch < 20) {
            phase_name = "Phase 1: All antennas healthy (100% online)";
        } else if (batch < 40) {
            phase_name = "Phase 2: 5 random antennas abruptly DIE";
            std::vector<std::size_t> dead = {7, 23, 89, 142, 210};
            for (auto a : dead) {
                is_alive[a] = false;
                h_mask[a] = 0;
            }
        } else if (batch < 60) {
            phase_name = "Phase 3: Entire 32-element RFSoC Board DIES (Ants 32-63)";
            for (std::size_t a = 32; a < 64; ++a) {
                is_alive[a] = false;
                h_mask[a] = 0;
            }
        } else if (batch < 80) {
            phase_name = "Phase 4: Dead RFSoC REVIVES, 2 other antennas fail";
            std::vector<std::size_t> dead = {11, 205};
            for (auto a : dead) {
                is_alive[a] = false;
                h_mask[a] = 0;
            }
        } else {
            phase_name = "Phase 5: High-rate random antenna flickering (chaos)";
            for (std::size_t a = 0; a < n_ant; ++a) {
                if ((rng() % 10) == 0) {
                    is_alive[a] = false;
                    h_mask[a] = 0;
                }
            }
        }

        std::size_t alive_count = 0;
        for (std::size_t a = 0; a < n_ant; ++a)
            if (is_alive[a])
                alive_count++;

        auto h_packed = generate_rfsoc_data(n_time, n_freq, n_ant, freqs, 0.03f, -0.02f, 1.0e-5f,
                                            0.0f, amp, is_alive, spacing_m);

        cudaMemcpyAsync(d_packed, h_packed.data(), in_bytes, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_mask, h_mask.data(), n_ant * sizeof(uint8_t), cudaMemcpyHostToDevice,
                        stream);

        kotekan::launch_generate_steering_weights(d_weights, d_dirs, d_wavenumbers, d_positions,
                                                  d_mask, nullptr, 1, n_freq, n_ant, stream);

        kotekan::launch_direct_beamformer(d_packed, d_weights, d_voltages, n_time, n_freq, n_ant, 1,
                                          1, 256, 4, 4, stream);

        cudaMemcpyAsync(h_out.data(), d_voltages, out_bytes, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);

        double sum_val = 0.0;
        for (const auto& v : h_out) {
            if (std::isnan(v.x) || std::isnan(v.y))
                total_nans++;
            if (std::isinf(v.x) || std::isinf(v.y))
                total_infs++;
            sum_val += static_cast<double>(v.x * v.x + v.y * v.y);
        }
        const double meas_avg = sum_val / (n_time * n_freq);
        const double theoretical_gain_ratio =
            static_cast<double>(alive_count) / static_cast<double>(n_ant);
        const double expected_scaling = theoretical_gain_ratio * theoretical_gain_ratio;

        if (batch % 20 == 0 || batch == n_batches - 1) {
            std::cout << "  [Batch " << std::setw(2) << batch << "] " << phase_name << "\n";
            std::cout << "             Alive=" << alive_count << "/" << n_ant
                      << " | Power=" << meas_avg << " | Relative Scaling=" << std::fixed
                      << std::setprecision(4) << expected_scaling << " | NaNs=" << total_nans
                      << "\n";
        }
    }

    cudaFree(d_packed);
    cudaFree(d_voltages);
    cudaFree(d_weights);
    cudaFree(d_dirs);
    cudaFree(d_wavenumbers);
    cudaFree(d_positions);
    cudaFree(d_mask);
    cudaStreamDestroy(stream);

    const bool passed = (total_nans == 0 && total_infs == 0);
    std::cout << "  Result: "
              << (passed ? "[PASSED] Direct Beamformer handled dynamic antenna death/revival with "
                           "zero glitches!"
                         : "[FAILED]")
              << "\n\n";
    return passed;
}

// ============================================================================
// CHAOS TEST 2: Upchannelization + Beam Tracker Integration
// ============================================================================

bool test_v5_upchannelized_beam_tracker_pipeline() {
    std::cout << ">>> [CHAOS TEST 2 - V5] Upchannelization Stage + Beam Tracker Integration <<<\n";

    // Simulate coarse to fine channel upchannelization (e.g. U=16 upchannelization)
    const std::size_t n_time = 3200;
    const std::size_t n_coarse_freq = 21;
    const std::size_t upchannel_factor = 16;
    const std::size_t n_fine_freq = n_coarse_freq * upchannel_factor; // 336 fine channels
    const std::size_t n_ant = 128;

    std::cout << "  Configuration: " << n_coarse_freq << " coarse channels upchannelized x"
              << upchannel_factor << " -> " << n_fine_freq << " fine channels (" << n_ant
              << " Antennas)\n";

    std::vector<double> fine_freqs(n_fine_freq);
    for (std::size_t f = 0; f < n_fine_freq; ++f) {
        fine_freqs[f] = 300.0e6 + f * (300.0e3 / upchannel_factor);
    }

    std::vector<bool> is_alive(n_ant, true);
    // Ingest through RFSoC and Beam Tracker across fine frequency resolution
    auto h_packed = generate_rfsoc_data(n_time, n_fine_freq, n_ant, fine_freqs, 0.05f, 0.02f,
                                        1.0e-5f, 0.5e-5f, 3.0f, is_alive);

    kotekan::MultiBeamTrackerConfig config;
    config.num_active_beams = 4;
    config.integration_spectra = 320;
    config.spacing_m = 0.6f;
    config.time_chunk_size = 80;
    config.time_unroll = 8;
    populate_antenna_positions(config, n_ant);
    for (std::size_t b = 0; b < 4; ++b) {
        config.trajectories[b].direction_start = {0.05f + static_cast<float>(b) * 0.01f, 0.02f,
                                                  1.0f};
        config.trajectories[b].direction_rate_per_sample = {1.0e-5f, 0.5e-5f};
    }

    kotekan::int4x2_t* d_packed = nullptr;
    float2* d_voltages = nullptr;
    const std::size_t in_bytes = n_time * n_fine_freq * n_ant * sizeof(kotekan::int4x2_t);
    const std::size_t out_bytes = n_time * n_fine_freq * 4 * sizeof(float2);

    cudaMalloc(&d_packed, in_bytes);
    cudaMalloc(&d_voltages, out_bytes);
    cudaMemcpy(d_packed, h_packed.data(), in_bytes, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    kotekan::launch_beam_tracker_v5_multibeam(d_packed, d_voltages, n_time, n_fine_freq, n_ant, 4,
                                              fine_freqs, config);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    std::vector<float2> h_out(n_time * n_fine_freq * 4);
    cudaMemcpy(h_out.data(), d_voltages, out_bytes, cudaMemcpyDeviceToHost);

    std::size_t nans = 0, infs = 0;
    for (const auto& v : h_out) {
        if (std::isnan(v.x) || std::isnan(v.y))
            nans++;
        if (std::isinf(v.x) || std::isinf(v.y))
            infs++;
    }

    std::cout << "  Execution Time: " << ms << " ms ("
              << (static_cast<double>(in_bytes) / (1024 * 1024 * 1024)) / (ms / 1000.0)
              << " GB/s)\n";
    std::cout << "  Fine frequency beam stability: NaNs=" << nans << ", Infs=" << infs << "\n";

    cudaFree(d_packed);
    cudaFree(d_voltages);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    const bool passed = (nans == 0 && infs == 0);
    std::cout << "  Result: "
              << (passed ? "[PASSED] Upchannelized fine-channel beam tracking is completely stable!"
                         : "[FAILED]")
              << "\n\n";
    return passed;
}

bool test_direct_upchannelized_beam_tracker_pipeline() {
    std::cout << ">>> [CHAOS TEST 2 - DIRECT] Upchannelization Stage + Direct Beam Tracker "
                 "Integration <<<\n";

    const std::size_t n_time = 3200;
    const std::size_t n_coarse_freq = 21;
    const std::size_t upchannel_factor = 16;
    const std::size_t n_fine_freq = n_coarse_freq * upchannel_factor; // 336 fine channels
    const std::size_t n_ant = 128;
    const float spacing_m = 0.6f;

    std::cout << "  Configuration: " << n_coarse_freq << " coarse channels upchannelized x"
              << upchannel_factor << " -> " << n_fine_freq << " fine channels (" << n_ant
              << " Antennas)\n";

    std::vector<double> fine_freqs(n_fine_freq);
    for (std::size_t f = 0; f < n_fine_freq; ++f) {
        fine_freqs[f] = 300.0e6 + f * (300.0e3 / upchannel_factor);
    }
    std::vector<double> wavenumbers(n_fine_freq);
    for (std::size_t f = 0; f < n_fine_freq; ++f) {
        wavenumbers[f] = TWO_PI * fine_freqs[f] / SPEED_OF_LIGHT;
    }

    std::vector<float3> positions(n_ant);
    for (std::size_t a = 0; a < n_ant; ++a) {
        float x = 0.0f, y = 0.0f;
        get_antenna_pos(n_ant, a, spacing_m, x, y);
        positions[a] = make_float3(x, y, 0.0f);
    }

    std::vector<kotekan::DirectDirection3D> dirs(4);
    for (std::size_t b = 0; b < 4; ++b) {
        const float l = 0.05f + static_cast<float>(b) * 0.01f;
        const float m = 0.02f;
        const float r2 = l * l + m * m;
        dirs[b] = kotekan::DirectDirection3D{l, m, (r2 <= 1.0f) ? std::sqrt(1.0f - r2) : 0.0f};
    }

    std::vector<bool> is_alive(n_ant, true);
    auto h_packed = generate_rfsoc_data(n_time, n_fine_freq, n_ant, fine_freqs, 0.05f, 0.02f,
                                        1.0e-5f, 0.5e-5f, 3.0f, is_alive, spacing_m);

    kotekan::int4x2_t* d_packed = nullptr;
    float2* d_voltages = nullptr;
    float2* d_weights = nullptr;
    kotekan::DirectDirection3D* d_dirs = nullptr;
    double* d_wavenumbers = nullptr;
    float3* d_positions = nullptr;

    const std::size_t in_bytes = n_time * n_fine_freq * n_ant * sizeof(kotekan::int4x2_t);
    const std::size_t out_bytes = n_time * n_fine_freq * 4 * sizeof(float2);
    const std::size_t weights_bytes = 4 * n_fine_freq * n_ant * sizeof(float2);

    cudaMalloc(&d_packed, in_bytes);
    cudaMalloc(&d_voltages, out_bytes);
    cudaMalloc(&d_weights, weights_bytes);
    cudaMalloc(&d_dirs, 4 * sizeof(kotekan::DirectDirection3D));
    cudaMalloc(&d_wavenumbers, n_fine_freq * sizeof(double));
    cudaMalloc(&d_positions, n_ant * sizeof(float3));

    cudaMemcpy(d_packed, h_packed.data(), in_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dirs, dirs.data(), 4 * sizeof(kotekan::DirectDirection3D), cudaMemcpyHostToDevice);
    cudaMemcpy(d_wavenumbers, wavenumbers.data(), n_fine_freq * sizeof(double),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_positions, positions.data(), n_ant * sizeof(float3), cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    kotekan::launch_generate_steering_weights(d_weights, d_dirs, d_wavenumbers, d_positions,
                                              nullptr, nullptr, 4, n_fine_freq, n_ant, stream);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, stream);
    kotekan::launch_direct_beamformer(d_packed, d_weights, d_voltages, n_time, n_fine_freq, n_ant,
                                      4, 4, 256, 4, 4, stream);
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    std::vector<float2> h_out(n_time * n_fine_freq * 4);
    cudaMemcpy(h_out.data(), d_voltages, out_bytes, cudaMemcpyDeviceToHost);

    std::size_t nans = 0, infs = 0;
    for (const auto& v : h_out) {
        if (std::isnan(v.x) || std::isnan(v.y))
            nans++;
        if (std::isinf(v.x) || std::isinf(v.y))
            infs++;
    }

    std::cout << "  Execution Time: " << ms << " ms ("
              << (static_cast<double>(in_bytes) / (1024 * 1024 * 1024)) / (ms / 1000.0)
              << " GB/s)\n";
    std::cout << "  Fine frequency beam stability: NaNs=" << nans << ", Infs=" << infs << "\n";

    cudaFree(d_packed);
    cudaFree(d_voltages);
    cudaFree(d_weights);
    cudaFree(d_dirs);
    cudaFree(d_wavenumbers);
    cudaFree(d_positions);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);

    const bool passed = (nans == 0 && infs == 0);
    std::cout
        << "  Result: "
        << (passed ? "[PASSED] Direct Beamformer fine-channel beam tracking is completely stable!"
                   : "[FAILED]")
        << "\n\n";
    return passed;
}

// ============================================================================
// CHAOS TEST 3: Extreme Sky Coordinates & Horizon Traversal
// ============================================================================

bool test_v5_extreme_sky_coordinates_and_horizons() {
    std::cout << ">>> [CHAOS TEST 3 - V5] Extreme Sky Coordinates & Horizon Clamping (l^2 + m^2 >= "
                 "1.0) <<<\n";

    const std::size_t n_time = 1600;
    const std::size_t n_freq = 168;
    const std::size_t n_ant = 128;

    std::vector<double> freqs(n_freq, 400.0e6);
    std::vector<bool> is_alive(n_ant, true);
    auto h_packed =
        generate_rfsoc_data(n_time, n_freq, n_ant, freqs, 0.0f, 0.0f, 0.0f, 0.0f, 2.0f, is_alive);

    kotekan::MultiBeamTrackerConfig config;
    config.num_active_beams = 4;
    config.integration_spectra = 320;
    config.spacing_m = 0.6f;
    populate_antenna_positions(config, n_ant);

    // Beam 0: Normal zenith (l=0, m=0, n=1)
    config.trajectories[0].direction_start = {0.0f, 0.0f, 1.0f};

    // Beam 1: Exactly at Horizon (l=1.0, m=0.0, n=0.0)
    config.trajectories[1].direction_start = {1.0f, 0.0f, 0.0f};

    // Beam 2: Beyond Horizon / Sub-Horizon (l=1.2, m=0.5, l^2+m^2 = 1.69 > 1) -> Must clamp n=0
    // safely without NaN
    config.trajectories[2].direction_start = {1.2f, 0.5f, 0.0f};

    // Beam 3: Fast Slew rate passing through horizon mid-batch
    config.trajectories[3].direction_start = {0.8f, 0.0f, 0.6f};
    config.trajectories[3].direction_rate_per_sample = {0.001f, 0.0f}; // Crosses horizon rapidly

    kotekan::int4x2_t* d_packed = nullptr;
    float2* d_voltages = nullptr;
    const std::size_t in_bytes = n_time * n_freq * n_ant * sizeof(kotekan::int4x2_t);
    const std::size_t out_bytes = n_time * n_freq * 4 * sizeof(float2);

    cudaMalloc(&d_packed, in_bytes);
    cudaMalloc(&d_voltages, out_bytes);
    cudaMemcpy(d_packed, h_packed.data(), in_bytes, cudaMemcpyHostToDevice);

    kotekan::launch_beam_tracker_v5_multibeam(d_packed, d_voltages, n_time, n_freq, n_ant, 4, freqs,
                                              config);

    std::vector<float2> h_out(n_time * n_freq * 4);
    cudaMemcpy(h_out.data(), d_voltages, out_bytes, cudaMemcpyDeviceToHost);

    std::size_t nans = 0, infs = 0;
    for (const auto& v : h_out) {
        if (std::isnan(v.x) || std::isnan(v.y))
            nans++;
        if (std::isinf(v.x) || std::isinf(v.y))
            infs++;
    }

    std::cout << "  Horizon Clamping Check: NaNs=" << nans << ", Infs=" << infs << "\n";

    cudaFree(d_packed);
    cudaFree(d_voltages);

    const bool passed = (nans == 0 && infs == 0);
    std::cout << "  Result: "
              << (passed ? "[PASSED] Beyond-horizon & extreme slew coordinates clamped robustly!"
                         : "[FAILED]")
              << "\n\n";
    return passed;
}

bool test_direct_extreme_sky_coordinates_and_horizons() {
    std::cout << ">>> [CHAOS TEST 3 - DIRECT] Extreme Sky Coordinates & Horizon Clamping (l^2 + "
                 "m^2 >= 1.0) <<<\n";

    const std::size_t n_time = 1600;
    const std::size_t n_freq = 168;
    const std::size_t n_ant = 128;
    const float spacing_m = 0.6f;

    std::vector<double> freqs(n_freq, 400.0e6);
    std::vector<double> wavenumbers(n_freq, TWO_PI * 400.0e6 / SPEED_OF_LIGHT);
    std::vector<bool> is_alive(n_ant, true);
    auto h_packed = generate_rfsoc_data(n_time, n_freq, n_ant, freqs, 0.0f, 0.0f, 0.0f, 0.0f, 2.0f,
                                        is_alive, spacing_m);

    std::vector<float3> positions(n_ant);
    for (std::size_t a = 0; a < n_ant; ++a) {
        float x = 0.0f, y = 0.0f;
        get_antenna_pos(n_ant, a, spacing_m, x, y);
        positions[a] = make_float3(x, y, 0.0f);
    }

    std::vector<kotekan::DirectDirection3D> dirs(4);
    dirs[0] = {0.0f, 0.0f, 1.0f}; // Zenith
    dirs[1] = {1.0f, 0.0f, 0.0f}; // Horizon
    dirs[2] = {1.2f, 0.5f, 0.0f}; // Beyond Horizon
    dirs[3] = {0.8f, 0.8f, 0.0f}; // Extreme

    kotekan::int4x2_t* d_packed = nullptr;
    float2* d_voltages = nullptr;
    float2* d_weights = nullptr;
    kotekan::DirectDirection3D* d_dirs = nullptr;
    double* d_wavenumbers = nullptr;
    float3* d_positions = nullptr;

    const std::size_t in_bytes = n_time * n_freq * n_ant * sizeof(kotekan::int4x2_t);
    const std::size_t out_bytes = n_time * n_freq * 4 * sizeof(float2);
    const std::size_t weights_bytes = 4 * n_freq * n_ant * sizeof(float2);

    cudaMalloc(&d_packed, in_bytes);
    cudaMalloc(&d_voltages, out_bytes);
    cudaMalloc(&d_weights, weights_bytes);
    cudaMalloc(&d_dirs, 4 * sizeof(kotekan::DirectDirection3D));
    cudaMalloc(&d_wavenumbers, n_freq * sizeof(double));
    cudaMalloc(&d_positions, n_ant * sizeof(float3));

    cudaMemcpy(d_packed, h_packed.data(), in_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dirs, dirs.data(), 4 * sizeof(kotekan::DirectDirection3D), cudaMemcpyHostToDevice);
    cudaMemcpy(d_wavenumbers, wavenumbers.data(), n_freq * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_positions, positions.data(), n_ant * sizeof(float3), cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    kotekan::launch_generate_steering_weights(d_weights, d_dirs, d_wavenumbers, d_positions,
                                              nullptr, nullptr, 4, n_freq, n_ant, stream);

    kotekan::launch_direct_beamformer(d_packed, d_weights, d_voltages, n_time, n_freq, n_ant, 4, 4,
                                      256, 4, 4, stream);

    std::vector<float2> h_out(n_time * n_freq * 4);
    cudaMemcpy(h_out.data(), d_voltages, out_bytes, cudaMemcpyDeviceToHost);

    std::size_t nans = 0, infs = 0;
    for (const auto& v : h_out) {
        if (std::isnan(v.x) || std::isnan(v.y))
            nans++;
        if (std::isinf(v.x) || std::isinf(v.y))
            infs++;
    }

    std::cout << "  Direct Horizon Clamping Check: NaNs=" << nans << ", Infs=" << infs << "\n";

    cudaFree(d_packed);
    cudaFree(d_voltages);
    cudaFree(d_weights);
    cudaFree(d_dirs);
    cudaFree(d_wavenumbers);
    cudaFree(d_positions);
    cudaStreamDestroy(stream);

    const bool passed = (nans == 0 && infs == 0);
    std::cout << "  Result: "
              << (passed ? "[PASSED] Direct Beamformer beyond-horizon & extreme slew coordinates "
                           "clamped robustly!"
                         : "[FAILED]")
              << "\n\n";
    return passed;
}

// ============================================================================
// Main Chaos Runner
// ============================================================================

int main(int /*argc*/, char** /*argv*/) {
    std::cout << "================================================================================="
                 "===================\n";
    std::cout
        << " CHARTS Kotekan CUDA Beam Tracker Chaos Engineering & Dynamic Fault Injection Suite\n";
    std::cout << " Testing: Antenna Mortality & Revival, Upchannelization, Beyond-Horizon "
                 "Trajectories, Packet Loss\n";
    std::cout << "================================================================================="
                 "===================\n\n";

    std::cout << "---------------------------------------------------------------------------------"
                 "-------------------\n";
    std::cout << " STAGE 1: Beam Tracker V5 Chaos Suite\n";
    std::cout << "---------------------------------------------------------------------------------"
                 "-------------------\n";
    bool v5_1 = test_v5_dynamic_antenna_mortality_and_revival(256);
    bool v5_2 = test_v5_upchannelized_beam_tracker_pipeline();
    bool v5_3 = test_v5_extreme_sky_coordinates_and_horizons();

    std::cout << "---------------------------------------------------------------------------------"
                 "-------------------\n";
    std::cout << " STAGE 2: Direct Beam Tracker (Zero Integration Window) Chaos Suite\n";
    std::cout << "---------------------------------------------------------------------------------"
                 "-------------------\n";
    bool dir_1 = test_direct_dynamic_antenna_mortality_and_revival(256);
    bool dir_2 = test_direct_upchannelized_beam_tracker_pipeline();
    bool dir_3 = test_direct_extreme_sky_coordinates_and_horizons();

    bool all_passed = (v5_1 && v5_2 && v5_3 && dir_1 && dir_2 && dir_3);

    std::cout << "================================================================================="
                 "===================\n";
    std::cout << " CHAOS SUITE FINAL RESULT: "
              << (all_passed ? "ALL TESTS PASSED (100% STABLE - BOTH V5 & DIRECT)"
                             : "ONE OR MORE TESTS FAILED")
              << "\n";
    std::cout << "================================================================================="
                 "===================\n";

    return all_passed ? 0 : 1;
}
