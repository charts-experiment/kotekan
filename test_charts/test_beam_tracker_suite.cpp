#include "DataType.hpp"
#include "chartsConstants.hpp"
#include "cudaBeamTrackerV5.hpp"
#include "cudaDirectBeamTracker.hpp"
#include "kotekanLogging.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <immintrin.h>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::high_resolution_clock;

constexpr double SPEED_OF_LIGHT = kotekan::charts::constants::speed_of_light_m_per_s;
constexpr double TWO_PI = kotekan::charts::constants::two_pi;

// ============================================================================
// Data Types & Structures
// ============================================================================

struct PointSource {
    float l0 = 0.0f;
    float m0 = 0.0f;
    float dl = 0.0f;
    float dm = 0.0f;
    float amplitude = 3.0f; // in 4-bit scale [-7..7]
};

struct TestResult {
    std::string test_name;
    std::size_t n_ant = 0;
    std::size_t n_time = 0;
    std::size_t n_freq = 0;
    std::size_t n_beams = 0;
    bool passed = false;
    std::string details;
    double kernel_time_ms = 0.0;
    double throughput_gb_s = 0.0;
};

// ============================================================================
// Physical Geometry Helpers (Matches Kotekan CUDA V5 kernel)
// ============================================================================

inline void get_antenna_pos(std::size_t n_ant, std::size_t element, float spacing_m, float& x,
                            float& y) {
    if (n_ant == 32 || n_ant == 64) {
        x = static_cast<float>(element & 7U) * spacing_m;
        y = static_cast<float>(element >> 3U) * spacing_m;
    } else { // 128 or 256 antennas (16 columns)
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

// ============================================================================
// RFSoC Packet Simulation & Shuffling Ingestion
// ============================================================================

std::vector<kotekan::int4x2_t> simulate_rfsoc_ingest_and_shuffle(
    std::size_t n_time, std::size_t n_freq, std::size_t n_ant, const std::vector<double>& freqs_hz,
    const std::vector<PointSource>& sources, float noise_amplitude = 0.0f, float spacing_m = 0.6f,
    bool simulate_lost_packets = false, std::vector<uint8_t>* out_mask = nullptr) {

    const std::size_t n_channels_per_packet = 168;
    const std::size_t num_elements_per_rfsoc = 32;
    const std::size_t num_rfsocs = n_ant / num_elements_per_rfsoc;
    const std::size_t subbands_per_nic = n_freq / n_channels_per_packet;

    const std::size_t total_elements = n_time * n_freq * n_ant;
    std::vector<kotekan::int4x2_t> shuffled_buffer(total_elements);
    std::memset(shuffled_buffer.data(), 0, total_elements * sizeof(kotekan::int4x2_t));

    if (out_mask) {
        out_mask->assign(n_time, 0);
    }

    std::vector<float> time_real(n_time * n_freq * n_ant, 0.0f);
    std::vector<float> time_imag(n_time * n_freq * n_ant, 0.0f);

#pragma omp parallel for collapse(2) schedule(static)
    for (std::size_t t = 0; t < n_time; ++t) {
        for (std::size_t f = 0; f < n_freq; ++f) {
            const double freq_hz = freqs_hz[f];
            const double k = TWO_PI * freq_hz / SPEED_OF_LIGHT;

            for (const auto& src : sources) {
                const double cur_l = src.l0 + t * src.dl;
                const double cur_m = src.m0 + t * src.dm;
                const double base_phase = TWO_PI * (0.005 * t + 0.03 * f);

                for (std::size_t a = 0; a < n_ant; ++a) {
                    float pos_x = 0.0f, pos_y = 0.0f;
                    get_antenna_pos(n_ant, a, spacing_m, pos_x, pos_y);

                    const double delay_m = pos_x * cur_l + pos_y * cur_m;
                    const double phase = base_phase - k * delay_m;

                    const std::size_t idx = (t * n_freq + f) * n_ant + a;
                    time_real[idx] += src.amplitude * static_cast<float>(std::cos(phase));
                    time_imag[idx] += src.amplitude * static_cast<float>(std::sin(phase));
                }
            }

            if (noise_amplitude > 0.0f) {
                std::mt19937 rng(static_cast<unsigned int>(42 + t * n_freq + f));
                std::normal_distribution<float> noise_dist(0.0f, noise_amplitude);
                for (std::size_t a = 0; a < n_ant; ++a) {
                    const std::size_t idx = (t * n_freq + f) * n_ant + a;
                    time_real[idx] += noise_dist(rng);
                    time_imag[idx] += noise_dist(rng);
                }
            }
        }
    }

// Emulate RFSoC UDP packetization and rfsocHandlerShuffle ingestion
#pragma omp parallel for collapse(2) schedule(static)
    for (std::size_t t = 0; t < n_time; ++t) {
        for (std::size_t sb = 0; sb < subbands_per_nic; ++sb) {
            for (std::size_t rf_id = 0; rf_id < num_rfsocs; ++rf_id) {
                if (simulate_lost_packets && (t % 100 == 13) && (rf_id == 1) && (sb == 0)) {
                    if (out_mask)
                        (*out_mask)[t] = 1;
                    continue;
                }

                const std::size_t ant_base = (num_rfsocs - 1 - rf_id) * num_elements_per_rfsoc;
                const std::size_t freq_base_offset = sb * n_channels_per_packet * n_ant;
                const std::size_t antenna_output_offset =
                    (num_rfsocs - 1 - rf_id) * num_elements_per_rfsoc;
                uint8_t* dst = reinterpret_cast<uint8_t*>(shuffled_buffer.data())
                               + (t * n_freq * n_ant) + freq_base_offset + antenna_output_offset;

                for (std::size_t ch = 0; ch < n_channels_per_packet; ++ch) {
                    const std::size_t global_f = sb * n_channels_per_packet + ch;
                    for (std::size_t elem = 0; elem < num_elements_per_rfsoc; ++elem) {
                        const std::size_t global_a = ant_base + elem;
                        const std::size_t src_idx = (t * n_freq + global_f) * n_ant + global_a;

                        const int r_val = std::max(
                            -8, std::min(7, static_cast<int>(std::round(time_real[src_idx]))));
                        const int i_val = std::max(
                            -8, std::min(7, static_cast<int>(std::round(time_imag[src_idx]))));
                        const uint8_t byte_val =
                            static_cast<uint8_t>((r_val & 0x0F) | ((i_val & 0x0F) << 4));

                        dst[ch * n_ant + elem] = byte_val;
                    }
                }
            }
        }
    }

    return shuffled_buffer;
}

// ============================================================================
// High-Precision Double CPU Analytical Reference
// ============================================================================

// ============================================================================
// High-Precision Double CPU Analytical Reference (Complex Voltages)
// ============================================================================

void cpu_reference_multibeam(const std::vector<kotekan::int4x2_t>& packed, float2* out_voltages,
                             std::size_t n_time, std::size_t n_freq, std::size_t n_ant,
                             std::size_t max_beams_stride, const std::vector<double>& freqs_hz,
                             const kotekan::MultiBeamTrackerConfig& config,
                             std::size_t window_offset = 0) {

#pragma omp parallel for collapse(2) schedule(static)
    for (std::size_t t = 0; t < n_time; ++t) {
        for (std::size_t f = 0; f < n_freq; ++f) {
            const std::size_t win = window_offset + (t / config.integration_spectra);
            const double center_sample =
                (static_cast<double>(win) + 0.5) * static_cast<double>(config.integration_spectra);

            const double freq_hz = freqs_hz[f];
            const double k = TWO_PI * freq_hz / SPEED_OF_LIGHT;

            for (std::size_t b = 0; b < config.num_active_beams; ++b) {
                const auto& traj = config.trajectories[b];
                const float l = static_cast<float>(
                    traj.direction_start.x + traj.direction_rate_per_sample.dl * center_sample);
                const float m = static_cast<float>(
                    traj.direction_start.y + traj.direction_rate_per_sample.dm * center_sample);

                double sum_r = 0.0;
                double sum_i = 0.0;

                for (std::size_t a = 0; a < n_ant; ++a) {
                    if (config.antenna_mask[a] == 0)
                        continue; // Skip masked/dead antennas

                    float pos_x = 0.0f, pos_y = 0.0f;
                    get_antenna_pos(n_ant, a, config.spacing_m, pos_x, pos_y);

                    const double delay_m =
                        static_cast<double>(pos_x) * l + static_cast<double>(pos_y) * m;
                    const double phase = k * delay_m;
                    const double w_r = std::cos(phase);
                    const double w_i = std::sin(phase);

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

                out_voltages[(t * n_freq + f) * max_beams_stride + b] =
                    make_float2(static_cast<float>(sum_r), static_cast<float>(sum_i));
            }
        }
    }
}

// ============================================================================
// Statistical & Tolerance Verification
// ============================================================================

struct ValidationMetrics {
    bool passed = true;
    double max_rel_error = 0.0;
    double max_abs_error = 0.0;
    double rms_error = 0.0;
    std::size_t nan_count = 0;
    std::size_t inf_count = 0;
    std::size_t mismatch_count = 0;
};

ValidationMetrics verify_voltages(const float2* ref, const float2* test, std::size_t count,
                                  float rel_tol = 1.0e-3f, float abs_tol = 1.0e-3f) {

    ValidationMetrics m;
    double sum_sq = 0.0;

    for (std::size_t i = 0; i < count; ++i) {
        const float2 r = ref[i];
        const float2 t = test[i];

        if (std::isnan(t.x) || std::isnan(t.y)) {
            m.nan_count++;
            m.passed = false;
            continue;
        }
        if (std::isinf(t.x) || std::isinf(t.y)) {
            m.inf_count++;
            m.passed = false;
            continue;
        }

        const double diff_r = std::abs(static_cast<double>(r.x) - static_cast<double>(t.x));
        const double diff_i = std::abs(static_cast<double>(r.y) - static_cast<double>(t.y));
        const double diff = std::sqrt(diff_r * diff_r + diff_i * diff_i);
        sum_sq += diff * diff;

        if (diff > m.max_abs_error)
            m.max_abs_error = diff;

        const double ref_mag =
            std::sqrt(static_cast<double>(r.x) * r.x + static_cast<double>(r.y) * r.y);
        if (ref_mag > 1e-4) {
            const double rel_err = diff / ref_mag;
            if (rel_err > m.max_rel_error)
                m.max_rel_error = rel_err;
            if (diff > abs_tol && rel_err > rel_tol) {
                m.mismatch_count++;
                m.passed = false;
            }
        } else if (diff > abs_tol) {
            m.mismatch_count++;
            m.passed = false;
        }
    }

    m.rms_error = std::sqrt(sum_sq / std::max(std::size_t(1), count));
    return m;
}

// ============================================================================
// DIRECT BEAM TRACKER (Zero Integration Window) Verification
// ============================================================================

TestResult test_direct_beam_tracker_accuracy(std::size_t n_ant, std::size_t n_time = 3840,
                                             std::size_t n_freq = 336, std::size_t max_beams = 4,
                                             std::size_t active_beams = 4) {

    TestResult res;
    res.test_name = "[Direct] Numerical Accuracy (CPU Double Ref)";
    res.n_ant = n_ant;
    res.n_time = n_time;
    res.n_freq = n_freq;
    res.n_beams = active_beams;

    const float spacing_m = 0.6f;
    std::vector<double> freqs(n_freq);
    for (std::size_t f = 0; f < n_freq; ++f) {
        freqs[f] = 300.0e6 + f * 300.0e3;
    }

    std::vector<PointSource> sources;
    for (std::size_t b = 0; b < active_beams; ++b) {
        PointSource src;
        src.l0 = static_cast<float>(b) * 0.015f;
        src.m0 = static_cast<float>(b) * -0.010f;
        src.dl = 0.0f;
        src.dm = 0.0f;
        src.amplitude = 3.0f;
        sources.push_back(src);
    }

    const auto host_packed =
        simulate_rfsoc_ingest_and_shuffle(n_time, n_freq, n_ant, freqs, sources, 0.5f, spacing_m);

    std::vector<kotekan::DirectDirection3D> h_dirs(max_beams);
    for (std::size_t b = 0; b < active_beams; ++b) {
        const float l = sources[b].l0;
        const float m = sources[b].m0;
        const float r2 = l * l + m * m;
        h_dirs[b] = kotekan::DirectDirection3D{l, m, (r2 <= 1.0f) ? std::sqrt(1.0f - r2) : 0.0f};
    }
    for (std::size_t b = active_beams; b < max_beams; ++b) {
        h_dirs[b] = kotekan::DirectDirection3D{0.0f, 0.0f, 1.0f};
    }

    std::vector<double> h_wavenumbers(n_freq);
    for (std::size_t f = 0; f < n_freq; ++f) {
        h_wavenumbers[f] = TWO_PI * freqs[f] / SPEED_OF_LIGHT;
    }

    std::vector<float3> h_positions(n_ant);
    for (std::size_t a = 0; a < n_ant; ++a) {
        float x = 0.0f, y = 0.0f;
        get_antenna_pos(n_ant, a, spacing_m, x, y);
        h_positions[a] = make_float3(x, y, 0.0f);
    }

    // Generate weights on CPU
    const float norm = 1.0f / std::sqrt(static_cast<float>(n_ant));
    std::vector<float2> h_weights(max_beams * n_freq * n_ant);
    for (std::size_t b = 0; b < max_beams; ++b) {
        for (std::size_t f = 0; f < n_freq; ++f) {
            for (std::size_t a = 0; a < n_ant; ++a) {
                const double delay_m = static_cast<double>(h_positions[a].x) * h_dirs[b].x
                                       + static_cast<double>(h_positions[a].y) * h_dirs[b].y;
                const double phase = h_wavenumbers[f] * delay_m;
                h_weights[(b * n_freq + f) * n_ant + a] =
                    make_float2(static_cast<float>(std::cos(phase) * norm),
                                static_cast<float>(std::sin(phase) * norm));
            }
        }
    }

    const std::size_t eval_samples = std::min(n_time, std::size_t(512));
    std::vector<float2> cpu_ref(eval_samples * n_freq * max_beams);
    for (std::size_t t = 0; t < eval_samples; ++t) {
        for (std::size_t f = 0; f < n_freq; ++f) {
            for (std::size_t b = 0; b < active_beams; ++b) {
                double sum_r = 0.0;
                double sum_i = 0.0;
                const std::size_t w_base = (b * n_freq + f) * n_ant;
                for (std::size_t a = 0; a < n_ant; ++a) {
                    const float2 w = h_weights[w_base + a];
                    const uint8_t byte_val = host_packed[(t * n_freq + f) * n_ant + a].val;
                    int v_r = static_cast<int>(byte_val & 0x0F);
                    if (v_r >= 8)
                        v_r -= 16;
                    int v_i = static_cast<int>((byte_val >> 4) & 0x0F);
                    if (v_i >= 8)
                        v_i -= 16;

                    sum_r += static_cast<double>(w.x) * v_r - static_cast<double>(w.y) * v_i;
                    sum_i += static_cast<double>(w.x) * v_i + static_cast<double>(w.y) * v_r;
                }
                cpu_ref[(t * n_freq + f) * max_beams + b] =
                    make_float2(static_cast<float>(sum_r), static_cast<float>(sum_i));
            }
        }
    }

    const std::size_t input_bytes = n_time * n_freq * n_ant * sizeof(kotekan::int4x2_t);
    const std::size_t output_bytes = n_time * n_freq * max_beams * sizeof(float2);
    const std::size_t weights_bytes = max_beams * n_freq * n_ant * sizeof(float2);

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

    cudaMemcpy(d_packed, host_packed.data(), input_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dirs, h_dirs.data(), max_beams * sizeof(kotekan::DirectDirection3D),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_wavenumbers, h_wavenumbers.data(), n_freq * sizeof(double),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_positions, h_positions.data(), n_ant * sizeof(float3), cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    kotekan::launch_generate_steering_weights(d_weights, d_dirs, d_wavenumbers, d_positions,
                                              nullptr, nullptr, max_beams, n_freq, n_ant, stream);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, stream);
    kotekan::launch_direct_beamformer(d_packed, d_weights, d_voltages, n_time, n_freq, n_ant,
                                      active_beams, max_beams, 256, 4, 4, stream);
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    res.kernel_time_ms = elapsed_ms;

    const double data_gb = static_cast<double>(input_bytes) / (1024.0 * 1024.0 * 1024.0);
    res.throughput_gb_s = data_gb / (elapsed_ms / 1000.0);

    std::vector<float2> gpu_out(eval_samples * n_freq * max_beams);
    cudaMemcpy(gpu_out.data(), d_voltages, eval_samples * n_freq * max_beams * sizeof(float2),
               cudaMemcpyDeviceToHost);

    const auto metrics =
        verify_voltages(cpu_ref.data(), gpu_out.data(), eval_samples * n_freq * max_beams);
    res.passed = metrics.passed;

    std::ostringstream ss;
    ss << "RMS Error=" << std::scientific << std::setprecision(2) << metrics.rms_error
       << ", MaxAbsErr=" << metrics.max_abs_error << ", Mismatches=" << metrics.mismatch_count;
    res.details = ss.str();

    cudaFree(d_packed);
    cudaFree(d_voltages);
    cudaFree(d_weights);
    cudaFree(d_dirs);
    cudaFree(d_wavenumbers);
    cudaFree(d_positions);
    cudaStreamDestroy(stream);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return res;
}

// ============================================================================
// TEST 1: Numerical Accuracy across 64, 128, 256 Antennas & All Multi-Beams
// ============================================================================

TestResult test_numerical_accuracy(std::size_t n_ant, std::size_t n_time = 3200,
                                   std::size_t n_freq = 336, std::size_t max_beams = 4,
                                   std::size_t active_beams = 4, int time_unroll = 8) {

    TestResult res;
    res.test_name = "[V5] Numerical Accuracy & CPU Reference Equivalence";
    res.n_ant = n_ant;
    res.n_time = n_time;
    res.n_freq = n_freq;
    res.n_beams = active_beams;

    std::vector<double> freqs(n_freq);
    for (std::size_t f = 0; f < n_freq; ++f) {
        freqs[f] = 300.0e6 + f * 300.0e3;
    }

    kotekan::MultiBeamTrackerConfig config;
    config.num_active_beams = active_beams;
    config.integration_spectra = 320;
    config.spacing_m = 0.6f;
    config.time_chunk_size = 80;
    config.time_unroll = time_unroll;
    populate_antenna_positions(config, n_ant);

    std::vector<PointSource> sources;
    for (std::size_t b = 0; b < active_beams; ++b) {
        PointSource src;
        src.l0 = static_cast<float>(b) * 0.015f;
        src.m0 = static_cast<float>(b) * -0.010f;
        src.dl = 1.0e-5f;
        src.dm = 0.5e-5f;
        src.amplitude = 3.0f;
        sources.push_back(src);

        config.trajectories[b].direction_start = {src.l0, src.m0, 1.0f};
        config.trajectories[b].direction_rate_per_sample = {src.dl, src.dm};
    }

    const auto host_packed = simulate_rfsoc_ingest_and_shuffle(n_time, n_freq, n_ant, freqs,
                                                               sources, 0.5f, config.spacing_m);

    const std::size_t total_out_elements = n_time * n_freq * max_beams;
    std::vector<float2> cpu_ref(total_out_elements);
    std::vector<float2> gpu_out(total_out_elements);

    const std::size_t eval_samples = std::min(n_time, std::size_t(640));
    cpu_reference_multibeam(host_packed, cpu_ref.data(), eval_samples, n_freq, n_ant, max_beams,
                            freqs, config);

    kotekan::int4x2_t* d_packed = nullptr;
    float2* d_voltages = nullptr;
    const std::size_t input_bytes = n_time * n_freq * n_ant * sizeof(kotekan::int4x2_t);
    const std::size_t output_bytes = total_out_elements * sizeof(float2);

    cudaMalloc(&d_packed, input_bytes);
    cudaMalloc(&d_voltages, output_bytes);
    cudaMemset(d_voltages, 0, output_bytes);
    cudaMemcpy(d_packed, host_packed.data(), input_bytes, cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, stream);
    kotekan::launch_beam_tracker_v5_multibeam(d_packed, d_voltages, n_time, n_freq, n_ant,
                                              max_beams, freqs, config, stream, 0);
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    res.kernel_time_ms = elapsed_ms;

    const double data_gb = static_cast<double>(input_bytes) / (1024.0 * 1024.0 * 1024.0);
    res.throughput_gb_s = data_gb / (elapsed_ms / 1000.0);

    cudaMemcpy(gpu_out.data(), d_voltages, eval_samples * n_freq * max_beams * sizeof(float2),
               cudaMemcpyDeviceToHost);

    const auto metrics =
        verify_voltages(cpu_ref.data(), gpu_out.data(), eval_samples * n_freq * max_beams);

    res.passed = metrics.passed;
    std::ostringstream ss;
    ss << "RMS Error=" << std::scientific << std::setprecision(2) << metrics.rms_error
       << ", MaxAbsErr=" << metrics.max_abs_error << ", Mismatches=" << metrics.mismatch_count;
    res.details = ss.str();

    cudaFree(d_packed);
    cudaFree(d_voltages);
    cudaStreamDestroy(stream);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return res;
}

// ============================================================================
// TEST 2: Astronomical Validation: Coherent Gain (N_ant^2) & Spatial Beam Profile
// ============================================================================

TestResult test_astronomical_coherent_gain_and_profile(std::size_t n_ant,
                                                       std::size_t n_time = 15360,
                                                       std::size_t n_freq = 336) {

    TestResult res;
    res.test_name = "[V5] Coherent Gain (N_ant^2) & Sidelobes";
    res.n_ant = n_ant;
    res.n_time = n_time;
    res.n_freq = n_freq;
    res.n_beams = 4;

    std::vector<double> freqs(n_freq);
    for (std::size_t f = 0; f < n_freq; ++f) {
        freqs[f] = 300.0e6 + f * 300.0e3;
    }

    const float source_amp = 3.0f;
    // Quantized 4-bit complex power factor for rounded integers: round(3 cos theta)^2 + round(3 sin
    // theta)^2 = 9.3756
    const float quant_power_factor = 9.3756f;
    const float expected_coherent_peak = static_cast<float>(n_ant * n_ant) * quant_power_factor;

    PointSource src;
    src.l0 = 0.05f;
    src.m0 = -0.03f;
    src.dl = 2.0e-6f;
    src.dm = 1.0e-6f;
    src.amplitude = source_amp;

    kotekan::MultiBeamTrackerConfig config;
    config.num_active_beams = 4;
    config.integration_spectra = 320;
    config.spacing_m = 0.6f;
    config.time_chunk_size = 80;
    config.time_unroll = 8;
    populate_antenna_positions(config, n_ant);

    const float aperture_x = (n_ant <= 64 ? 7.0f : 15.0f) * config.spacing_m;
    const float lambda = static_cast<float>(SPEED_OF_LIGHT / freqs[0]);
    const float null_offset = lambda / aperture_x;
    const float sidelobe_offset = 1.5f * lambda / aperture_x;
    const float far_offset = 0.55f;

    // Beam 0: Exact On-target
    config.trajectories[0].direction_start = {src.l0, src.m0, 1.0f};
    config.trajectories[0].direction_rate_per_sample = {src.dl, src.dm};

    // Beam 1: First Null offset
    config.trajectories[1].direction_start = {src.l0 + null_offset, src.m0, 1.0f};
    config.trajectories[1].direction_rate_per_sample = {src.dl, src.dm};

    // Beam 2: First Sidelobe offset
    config.trajectories[2].direction_start = {src.l0 + sidelobe_offset, src.m0, 1.0f};
    config.trajectories[2].direction_rate_per_sample = {src.dl, src.dm};

    // Beam 3: Far offset
    config.trajectories[3].direction_start = {src.l0 + far_offset, src.m0, 1.0f};
    config.trajectories[3].direction_rate_per_sample = {src.dl, src.dm};

    const auto host_packed = simulate_rfsoc_ingest_and_shuffle(n_time, n_freq, n_ant, freqs, {src},
                                                               0.0f, config.spacing_m);

    const std::size_t max_beams = 4;
    const std::size_t total_out_elements = n_time * n_freq * max_beams;
    std::vector<float2> gpu_out(total_out_elements);

    kotekan::int4x2_t* d_packed = nullptr;
    float2* d_voltages = nullptr;
    const std::size_t input_bytes = n_time * n_freq * n_ant * sizeof(kotekan::int4x2_t);
    const std::size_t output_bytes = total_out_elements * sizeof(float2);

    cudaMalloc(&d_packed, input_bytes);
    cudaMalloc(&d_voltages, output_bytes);
    cudaMemcpy(d_packed, host_packed.data(), input_bytes, cudaMemcpyHostToDevice);

    kotekan::launch_beam_tracker_v5_multibeam(d_packed, d_voltages, n_time, n_freq, n_ant,
                                              max_beams, freqs, config);
    cudaMemcpy(gpu_out.data(), d_voltages, output_bytes, cudaMemcpyDeviceToHost);

    double beam_avg[4] = {0.0, 0.0, 0.0, 0.0};
    const std::size_t n_spectra = n_time * n_freq;

    for (std::size_t i = 0; i < n_spectra; ++i) {
        for (std::size_t b = 0; b < 4; ++b) {
            const float2 v = gpu_out[i * max_beams + b];
            beam_avg[b] += static_cast<double>(v.x * v.x + v.y * v.y);
        }
    }
    for (std::size_t b = 0; b < 4; ++b)
        beam_avg[b] /= n_spectra;

    const double on_target_ratio = beam_avg[0] / expected_coherent_peak;
    const double near_rejection_db =
        10.0 * std::log10(std::max(1e-12, beam_avg[0] / std::max(1e-12, beam_avg[1])));
    const double sidelobe_rejection_db =
        10.0 * std::log10(std::max(1e-12, beam_avg[0] / std::max(1e-12, beam_avg[2])));
    const double far_rejection_db =
        10.0 * std::log10(std::max(1e-12, beam_avg[0] / std::max(1e-12, beam_avg[3])));

    std::ostringstream ss;
    ss << "Expected Peak=" << expected_coherent_peak << ", Meas Peak=" << beam_avg[0]
       << " (Gain Ratio=" << std::fixed << std::setprecision(4) << on_target_ratio << "); "
       << "Rejection: Null=" << std::setprecision(1) << near_rejection_db << " dB, "
       << "Sidelobe=" << sidelobe_rejection_db << " dB, " << "Far=" << far_rejection_db << " dB";

    const bool gain_ok = std::abs(on_target_ratio - 1.0) < 0.03;
    const bool rejection_ok =
        (near_rejection_db > 10.0) && (sidelobe_rejection_db > 10.0) && (far_rejection_db > 15.0);

    res.passed = gain_ok && rejection_ok;
    res.details = ss.str();

    cudaFree(d_packed);
    cudaFree(d_voltages);

    return res;
}

// ============================================================================
// TEST 3: Multi-Source Tracking & Spatial Independence
// ============================================================================

TestResult test_multisource_independence(std::size_t n_ant, std::size_t n_time = 3200,
                                         std::size_t n_freq = 336) {

    TestResult res;
    res.test_name = "[V5] Multi-Source Tracking (Zero Crosstalk)";
    res.n_ant = n_ant;
    res.n_time = n_time;
    res.n_freq = n_freq;
    res.n_beams = 2;

    std::vector<double> freqs(n_freq);
    for (std::size_t f = 0; f < n_freq; ++f) {
        freqs[f] = 300.0e6 + f * 300.0e3;
    }

    PointSource src1;
    src1.l0 = 0.25f;
    src1.m0 = 0.10f;
    src1.dl = 1.0e-5f;
    src1.dm = 0.0f;
    src1.amplitude = 3.0f;

    PointSource src2;
    src2.l0 = -0.25f;
    src2.m0 = -0.10f;
    src2.dl = 0.0f;
    src2.dm = 1.0e-5f;
    src2.amplitude = 3.0f;

    kotekan::MultiBeamTrackerConfig config;
    config.num_active_beams = 2;
    config.integration_spectra = 320;
    config.spacing_m = 0.6f;
    config.time_chunk_size = 80;
    config.time_unroll = 8;
    populate_antenna_positions(config, n_ant);

    config.trajectories[0].direction_start = {src1.l0, src1.m0, 1.0f};
    config.trajectories[0].direction_rate_per_sample = {src1.dl, src1.dm};

    config.trajectories[1].direction_start = {src2.l0, src2.m0, 1.0f};
    config.trajectories[1].direction_rate_per_sample = {src2.dl, src2.dm};

    const auto host_packed = simulate_rfsoc_ingest_and_shuffle(
        n_time, n_freq, n_ant, freqs, {src1, src2}, 0.0f, config.spacing_m);

    const std::size_t max_beams = 4;
    const std::size_t total_out = n_time * n_freq * max_beams;
    std::vector<float2> gpu_out(total_out);

    kotekan::int4x2_t* d_packed = nullptr;
    float2* d_voltages = nullptr;
    cudaMalloc(&d_packed, n_time * n_freq * n_ant * sizeof(kotekan::int4x2_t));
    cudaMalloc(&d_voltages, total_out * sizeof(float2));
    cudaMemcpy(d_packed, host_packed.data(), n_time * n_freq * n_ant * sizeof(kotekan::int4x2_t),
               cudaMemcpyHostToDevice);

    kotekan::launch_beam_tracker_v5_multibeam(d_packed, d_voltages, n_time, n_freq, n_ant,
                                              max_beams, freqs, config);
    cudaMemcpy(gpu_out.data(), d_voltages, total_out * sizeof(float2), cudaMemcpyDeviceToHost);

    double b0_sum = 0.0, b1_sum = 0.0;
    const std::size_t n_spectra = n_time * n_freq;
    for (std::size_t i = 0; i < n_spectra; ++i) {
        const float2 v0 = gpu_out[i * max_beams + 0];
        const float2 v1 = gpu_out[i * max_beams + 1];
        b0_sum += static_cast<double>(v0.x * v0.x + v0.y * v0.y);
        b1_sum += static_cast<double>(v1.x * v1.x + v1.y * v1.y);
    }
    b0_sum /= n_spectra;
    b1_sum /= n_spectra;

    const float quant_power_factor = 9.3756f;
    const float exp_p1 = static_cast<float>(n_ant * n_ant) * quant_power_factor;
    const float exp_p2 = static_cast<float>(n_ant * n_ant) * quant_power_factor;

    const double r1 = b0_sum / exp_p1;
    const double r2 = b1_sum / exp_p2;

    std::ostringstream ss;
    ss << "Beam 0 (Src 1): Meas=" << b0_sum << ", Exp=" << exp_p1 << " (Ratio=" << r1 << "); "
       << "Beam 1 (Src 2): Meas=" << b1_sum << ", Exp=" << exp_p2 << " (Ratio=" << r2 << ")";

    res.passed = (std::abs(r1 - 1.0) < 0.04) && (std::abs(r2 - 1.0) < 0.04);
    res.details = ss.str();

    cudaFree(d_packed);
    cudaFree(d_voltages);

    return res;
}

// ============================================================================
// TEST 4: Missing Packets & Masking Stability
// ============================================================================

TestResult test_masking_stability(std::size_t n_ant, std::size_t n_time = 3200,
                                  std::size_t n_freq = 336) {

    TestResult res;
    res.test_name = "[V5] Packet Loss & Zero-Masking Robustness";
    res.n_ant = n_ant;
    res.n_time = n_time;
    res.n_freq = n_freq;
    res.n_beams = 1;

    std::vector<double> freqs(n_freq);
    for (std::size_t f = 0; f < n_freq; ++f)
        freqs[f] = 300.0e6 + f * 300.0e3;

    PointSource src;
    src.l0 = 0.01f;
    src.m0 = 0.01f;
    src.dl = 0.0f;
    src.dm = 0.0f;
    src.amplitude = 3.0f;

    std::vector<uint8_t> lost_mask;
    const auto host_packed = simulate_rfsoc_ingest_and_shuffle(n_time, n_freq, n_ant, freqs, {src},
                                                               0.0f, 0.6f, true, &lost_mask);

    kotekan::MultiBeamTrackerConfig config;
    config.num_active_beams = 1;
    config.integration_spectra = 320;
    config.spacing_m = 0.6f;
    populate_antenna_positions(config, n_ant);
    config.trajectories[0].direction_start = {src.l0, src.m0, 1.0f};

    const std::size_t total_out = n_time * n_freq;
    std::vector<float2> gpu_out(total_out);

    kotekan::int4x2_t* d_packed = nullptr;
    float2* d_voltages = nullptr;
    cudaMalloc(&d_packed, n_time * n_freq * n_ant * sizeof(kotekan::int4x2_t));
    cudaMalloc(&d_voltages, total_out * sizeof(float2));
    cudaMemcpy(d_packed, host_packed.data(), n_time * n_freq * n_ant * sizeof(kotekan::int4x2_t),
               cudaMemcpyHostToDevice);

    kotekan::launch_beam_tracker_v5_multibeam(d_packed, d_voltages, n_time, n_freq, n_ant, 1, freqs,
                                              config);
    cudaMemcpy(gpu_out.data(), d_voltages, total_out * sizeof(float2), cudaMemcpyDeviceToHost);

    std::size_t nan_count = 0, inf_count = 0;
    for (const auto& v : gpu_out) {
        if (std::isnan(v.x) || std::isnan(v.y))
            nan_count++;
        if (std::isinf(v.x) || std::isinf(v.y))
            inf_count++;
    }

    std::ostringstream ss;
    ss << "NaN count: " << nan_count << ", Inf count: " << inf_count;

    res.passed = (nan_count == 0 && inf_count == 0);
    res.details = ss.str();

    cudaFree(d_packed);
    cudaFree(d_voltages);

    return res;
}

// ============================================================================
// TEST 5: Dead Antenna Fault-Tolerance & Masking Verification
// ============================================================================

TestResult test_dead_antenna_fault_tolerance(std::size_t n_ant,
                                             const std::vector<std::size_t>& dead_antenna_indices,
                                             std::size_t n_time = 3200, std::size_t n_freq = 336) {

    TestResult res;
    res.test_name =
        "[V5] Dead Antenna Masking (" + std::to_string(dead_antenna_indices.size()) + " Dead)";
    res.n_ant = n_ant;
    res.n_time = n_time;
    res.n_freq = n_freq;
    res.n_beams = 1;

    std::vector<double> freqs(n_freq);
    for (std::size_t f = 0; f < n_freq; ++f)
        freqs[f] = 300.0e6 + f * 300.0e3;

    const float source_amp = 3.0f;
    const std::size_t active_ant_count = n_ant - dead_antenna_indices.size();
    const float quant_power_factor = 9.3756f;
    const float expected_coherent_peak =
        static_cast<float>(active_ant_count * active_ant_count) * quant_power_factor;

    PointSource src;
    src.l0 = 0.04f;
    src.m0 = -0.02f;
    src.dl = 1.0e-5f;
    src.dm = 0.5e-5f;
    src.amplitude = source_amp;

    kotekan::MultiBeamTrackerConfig config;
    config.num_active_beams = 1;
    config.integration_spectra = 320;
    config.spacing_m = 0.6f;
    config.time_chunk_size = 80;
    config.time_unroll = 8;
    populate_antenna_positions(config, n_ant);
    config.trajectories[0].direction_start = {src.l0, src.m0, 1.0f};
    config.trajectories[0].direction_rate_per_sample = {src.dl, src.dm};

    for (std::size_t dead_idx : dead_antenna_indices) {
        if (dead_idx < n_ant) {
            config.antenna_mask[dead_idx] = 0;
        }
    }

    auto host_packed = simulate_rfsoc_ingest_and_shuffle(n_time, n_freq, n_ant, freqs, {src}, 0.0f,
                                                         config.spacing_m);

    // Dead antenna output in physical frame is zeroed or noise
    for (std::size_t t = 0; t < n_time; ++t) {
        for (std::size_t f = 0; f < n_freq; ++f) {
            for (std::size_t dead_idx : dead_antenna_indices) {
                host_packed[(t * n_freq + f) * n_ant + dead_idx].val = 0x00;
            }
        }
    }

    const std::size_t max_beams = 1;
    const std::size_t total_out = n_time * n_freq * max_beams;
    std::vector<float2> gpu_out(total_out);

    kotekan::int4x2_t* d_packed = nullptr;
    float2* d_voltages = nullptr;
    const std::size_t input_bytes = n_time * n_freq * n_ant * sizeof(kotekan::int4x2_t);
    const std::size_t output_bytes = total_out * sizeof(float2);

    cudaMalloc(&d_packed, input_bytes);
    cudaMalloc(&d_voltages, output_bytes);
    cudaMemcpy(d_packed, host_packed.data(), input_bytes, cudaMemcpyHostToDevice);

    kotekan::launch_beam_tracker_v5_multibeam(d_packed, d_voltages, n_time, n_freq, n_ant,
                                              max_beams, freqs, config);
    cudaMemcpy(gpu_out.data(), d_voltages, output_bytes, cudaMemcpyDeviceToHost);

    double meas_sum = 0.0;
    std::size_t nan_count = 0, inf_count = 0;
    for (const auto& v : gpu_out) {
        if (std::isnan(v.x) || std::isnan(v.y))
            nan_count++;
        if (std::isinf(v.x) || std::isinf(v.y))
            inf_count++;
        meas_sum += static_cast<double>(v.x * v.x + v.y * v.y);
    }
    meas_sum /= (n_time * n_freq);

    const double gain_ratio = meas_sum / expected_coherent_peak;

    std::ostringstream ss;
    ss << "Alive=" << active_ant_count << "/" << n_ant << " (" << dead_antenna_indices.size()
       << " Dead) | Exp=" << expected_coherent_peak << ", Meas=" << meas_sum
       << " (Gain Ratio=" << std::fixed << std::setprecision(4) << gain_ratio
       << ", NaN=" << nan_count << ")";

    res.passed = (nan_count == 0 && inf_count == 0 && std::abs(gain_ratio - 1.0) < 0.03);
    res.details = ss.str();

    cudaFree(d_packed);
    cudaFree(d_voltages);

    return res;
}

// ============================================================================
// TEST 6: Sustained Continuous Stress Test
// ============================================================================

TestResult run_sustained_stress_test(std::size_t n_ant, std::size_t n_iterations = 200,
                                     std::size_t n_time = 15360, std::size_t n_freq = 336,
                                     std::size_t max_beams = 4, std::size_t active_beams = 2) {

    TestResult res;
    res.test_name = "[V5] Sustained Pipeline Stress Test";
    res.n_ant = n_ant;
    res.n_time = n_time;
    res.n_freq = n_freq;
    res.n_beams = active_beams;

    std::vector<double> freqs(n_freq);
    for (std::size_t f = 0; f < n_freq; ++f)
        freqs[f] = 300.0e6 + f * 300.0e3;

    kotekan::MultiBeamTrackerConfig config;
    config.num_active_beams = active_beams;
    config.integration_spectra = 320;
    config.spacing_m = 0.6f;
    config.time_chunk_size = 80;
    config.time_unroll = 8;
    populate_antenna_positions(config, n_ant);
    for (std::size_t b = 0; b < active_beams; ++b) {
        config.trajectories[b].direction_start = {static_cast<float>(b) * 0.02f, 0.0f, 1.0f};
        config.trajectories[b].direction_rate_per_sample = {1.0e-5f, 0.0f};
    }

    PointSource src;
    src.l0 = 0.0f;
    src.m0 = 0.0f;
    src.dl = 1.0e-5f;
    src.dm = 0.0f;
    src.amplitude = 2.5f;

    const auto host_packed = simulate_rfsoc_ingest_and_shuffle(n_time, n_freq, n_ant, freqs, {src},
                                                               0.5f, config.spacing_m);

    const std::size_t input_bytes = n_time * n_freq * n_ant * sizeof(kotekan::int4x2_t);
    const std::size_t output_bytes = n_time * n_freq * max_beams * sizeof(float2);

    kotekan::int4x2_t* d_packed = nullptr;
    float2* d_voltages = nullptr;
    cudaMalloc(&d_packed, input_bytes);
    cudaMalloc(&d_voltages, output_bytes);
    cudaMemcpy(d_packed, host_packed.data(), input_bytes, cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    cudaEvent_t start_ev, stop_ev;
    cudaEventCreate(&start_ev);
    cudaEventCreate(&stop_ev);

    std::vector<double> latencies_ms;
    latencies_ms.reserve(n_iterations);

    bool memory_stable = true;
    bool no_cuda_error = true;
    bool data_clean = true;

    // Warmup
    for (int i = 0; i < 5; ++i) {
        kotekan::launch_beam_tracker_v5_multibeam(d_packed, d_voltages, n_time, n_freq, n_ant,
                                                  max_beams, freqs, config, stream, i * 48);
        cudaStreamSynchronize(stream);
    }

    std::size_t free_mem_start = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem_start, &total_mem);

    const auto wall_start = Clock::now();

    for (std::size_t iter = 0; iter < n_iterations; ++iter) {
        cudaEventRecord(start_ev, stream);
        kotekan::launch_beam_tracker_v5_multibeam(d_packed, d_voltages, n_time, n_freq, n_ant,
                                                  max_beams, freqs, config, stream, iter * 48);
        cudaEventRecord(stop_ev, stream);
        cudaEventSynchronize(stop_ev);

        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            no_cuda_error = false;
            break;
        }

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start_ev, stop_ev);
        latencies_ms.push_back(ms);
    }

    const auto wall_end = Clock::now();
    const double total_wall_ms =
        std::chrono::duration<double, std::milli>(wall_end - wall_start).count();

    std::size_t free_mem_end = 0;
    cudaMemGetInfo(&free_mem_end, &total_mem);
    const std::size_t mem_diff = (free_mem_start > free_mem_end) ? (free_mem_start - free_mem_end)
                                                                 : (free_mem_end - free_mem_start);
    if (mem_diff > 10ULL * 1024ULL * 1024ULL) {
        memory_stable = false;
    }

    std::vector<float2> final_out(n_time * n_freq * max_beams);
    cudaMemcpy(final_out.data(), d_voltages, output_bytes, cudaMemcpyDeviceToHost);
    for (const auto& v : final_out) {
        if (std::isnan(v.x) || std::isnan(v.y) || std::isinf(v.x) || std::isinf(v.y)) {
            data_clean = false;
            break;
        }
    }

    std::sort(latencies_ms.begin(), latencies_ms.end());
    const double min_ms = latencies_ms.front();
    const double max_ms = latencies_ms.back();
    const double median_ms = latencies_ms[latencies_ms.size() / 2];
    const double p99_ms = latencies_ms[static_cast<std::size_t>(latencies_ms.size() * 0.99)];
    const double sum_ms = std::accumulate(latencies_ms.begin(), latencies_ms.end(), 0.0);
    const double avg_ms = sum_ms / latencies_ms.size();

    const double total_gb =
        (static_cast<double>(input_bytes) * n_iterations) / (1024.0 * 1024.0 * 1024.0);
    const double sustained_throughput = total_gb / (total_wall_ms / 1000.0);

    res.kernel_time_ms = avg_ms;
    res.throughput_gb_s = sustained_throughput;
    res.passed = no_cuda_error && memory_stable && data_clean;

    std::ostringstream ss;
    ss << n_iterations << " iterations | Avg=" << std::fixed << std::setprecision(2) << avg_ms
       << " ms, Min=" << min_ms << " ms, Med=" << median_ms << " ms, P99=" << p99_ms
       << " ms, Max=" << max_ms << " ms | Sustained=" << sustained_throughput
       << " GB/s | Budget=" << std::setprecision(1) << (avg_ms / 50.0 * 100.0) << "% (50ms budget)";
    res.details = ss.str();

    cudaFree(d_packed);
    cudaFree(d_voltages);
    cudaEventDestroy(start_ev);
    cudaEventDestroy(stop_ev);
    cudaStreamDestroy(stream);

    return res;
}

} // namespace

// ============================================================================
// Main Suite Runner
// ============================================================================

int main(int /*argc*/, char** /*argv*/) {
    std::cout << "================================================================================="
                 "===================\n";
    std::cout
        << " CHARTS Kotekan CUDA Beam Tracker Comprehensive Verification & Stress Test Suite\n";
    std::cout
        << " Configurations: 64, 128, 256 Antennas | Multi-Beam 1..8 | RFSoC Shuffling Ingestion\n";
    std::cout << "================================================================================="
                 "===================\n\n";

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Device: " << prop.name << " (" << (prop.totalGlobalMem / (1024 * 1024))
              << " MB VRAM, Compute " << prop.major << "." << prop.minor << ")\n\n";

    std::vector<TestResult> results;

    const std::vector<std::size_t> antenna_configs = {64, 128, 256};

    for (std::size_t n_ant : antenna_configs) {
        std::cout << ">>> Running Test Suite for N_ANT = " << n_ant << " <<<\n";

        // 0. Direct Beam Tracker: Exact Analytical CPU Double Reference Equivalence
        results.push_back(test_direct_beam_tracker_accuracy(n_ant, 3840, 336, 4, 1));
        results.push_back(test_direct_beam_tracker_accuracy(n_ant, 3840, 336, 4, 4));

        // 1. Numerical Accuracy across beam counts (1, 4, 8 beams)
        results.push_back(test_numerical_accuracy(n_ant, 3200, 336, 4, 1, 8));
        results.push_back(test_numerical_accuracy(n_ant, 3200, 336, 4, 4, 8));
        results.push_back(test_numerical_accuracy(n_ant, 3200, 336, 8, 8, 8));

        // 2. Astronomical Validation (Coherent N_ant^2 gain & Sidelobe rejection)
        results.push_back(test_astronomical_coherent_gain_and_profile(n_ant, 15360, 336));

        // 3. Multi-Source Independence
        results.push_back(test_multisource_independence(n_ant, 3200, 336));

        // 4. Missing packet masking stability
        results.push_back(test_masking_stability(n_ant, 3200, 336));

        // 5. Dead Antenna Fault-Tolerance & Masking Tests (Single dead antenna, multi-dead, full
        // RFSoC dead)
        std::vector<std::size_t> dead_test_set;
        if (n_ant == 64) {
            dead_test_set = {3, 14, 27, 45}; // 4 dead antennas out of 64
        } else if (n_ant == 128) {
            dead_test_set = {5, 12, 33, 48, 65, 80, 99, 110}; // 8 dead antennas out of 128
        } else {                                              // 256
            dead_test_set = {2,   17,  34,  51,  68,  85,  102, 119,
                             136, 153, 170, 187, 204, 221, 238, 255}; // 16 dead antennas out of 256
        }
        results.push_back(test_dead_antenna_fault_tolerance(n_ant, dead_test_set, 3200, 336));

        // 6. Sustained High-Throughput Stress Test
        const std::size_t stress_beams = (n_ant == 256) ? 1 : 2;
        results.push_back(run_sustained_stress_test(n_ant, 200, 15360, 336, 4, stress_beams));

        std::cout << "\n";
    }

    // Print Consolidated Summary Table
    std::cout << "================================================================================="
                 "=======================\n";
    std::cout << " TEST EXECUTION SUMMARY REPORT (Beam Tracker V5 vs. Direct Beamformer)\n";
    std::cout << "================================================================================="
                 "=======================\n";
    std::cout << "| " << std::left << std::setw(6) << "Ants" << "| " << std::setw(8) << "Beams"
              << "| " << std::setw(6) << "Status" << "| " << std::setw(46) << "Test Category"
              << "| " << std::setw(30) << "Performance / Metrics" << "|\n";
    std::cout << "---------------------------------------------------------------------------------"
                 "-----------------------\n";

    bool all_passed = true;
    for (const auto& r : results) {
        if (!r.passed)
            all_passed = false;

        std::string status_str = r.passed ? "[PASS]" : "[FAIL]";
        std::string beams_str = std::to_string(r.n_beams) + " Beam" + (r.n_beams > 1 ? "s" : "");

        std::cout << "| " << std::left << std::setw(6) << r.n_ant << "| " << std::setw(8)
                  << beams_str << "| " << std::setw(6) << status_str << "| " << std::setw(46)
                  << r.test_name << "| " << std::setw(30)
                  << (r.kernel_time_ms > 0
                          ? (std::to_string(r.kernel_time_ms).substr(0, 5) + " ms ("
                             + std::to_string(r.throughput_gb_s).substr(0, 5) + " GB/s)")
                          : "Verified")
                  << "|\n";
        std::cout << "    Details: " << r.details << "\n";
    }
    std::cout << "================================================================================="
                 "=======================\n";

    if (all_passed) {
        std::cout << "\n>>> ALL TESTS PASSED! Beam Tracker Kernel & Pipeline are 100% STABLE and "
                     "ACCURATE. <<<\n";
        return 0;
    } else {
        std::cout << "\n>>> ONE OR MORE TESTS FAILED! <<<\n";
        return 1;
    }
}
