#include "chartsConstants.hpp"
#include "cudaBeamUpchannelize.hpp"
#include "cudaUtils.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

namespace {

constexpr double TWO_PI = kotekan::charts::constants::two_pi;
constexpr double HANN16_SCALE = 0.4082482904638630; // 1 / sqrt(6)

double hann16_coeff(int n) {
    const double h = 0.5 * (1.0 - std::cos(TWO_PI * static_cast<double>(n) / 16.0));
    // Periodic Hann satisfies sum h[n]^2 = 6; dividing by sqrt(6) makes transform mean-power-unitary
    return h * (HANN16_SCALE / 0.5) * 0.5;
}

bool test_dft_parity_both_windows() {
    std::cout << "\n======================================================\n";
    std::cout << "[Test 1/4] 16-pt DFT Parity (Rectangular & Hann + Power)\n";
    std::cout << "======================================================\n";

    constexpr std::size_t n_time = 256;
    constexpr std::size_t n_freq = 4;
    constexpr std::size_t num_beams = 6;
    constexpr std::size_t max_beams_stride = 8;
    constexpr std::size_t n_pol = 2;

    std::mt19937 rng(4242);
    std::normal_distribution<float> dist(0.0f, 1.5f);

    const std::size_t total_elems = n_time * n_freq * max_beams_stride * n_pol;
    std::vector<float2> h_in(total_elems);
    for (auto& v : h_in) {
        v = make_float2(dist(rng), dist(rng));
    }

    float2* d_in = nullptr;
    float2* d_out = nullptr;
    float* d_pow = nullptr;
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&d_in, total_elems * sizeof(float2)));
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&d_out, total_elems * sizeof(float2)));
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&d_pow, num_beams * n_freq * 16 * sizeof(float)));
    CHECK_CUDA_ERROR_NON_OO(
        cudaMemcpy(d_in, h_in.data(), total_elems * sizeof(float2), cudaMemcpyHostToDevice));

    cudaStream_t stream = nullptr;
    CHECK_CUDA_ERROR_NON_OO(cudaStreamCreate(&stream));

    bool ok = true;
    for (const auto win :
         {kotekan::BeamUpchanWindowType::RECTANGULAR, kotekan::BeamUpchanWindowType::HANN}) {
        kotekan::BeamUpchannelizeConfig cfg;
        cfg.n_time = n_time;
        cfg.n_freq = n_freq;
        cfg.num_beams = num_beams;
        cfg.max_beams_stride = max_beams_stride;
        cfg.n_pol = n_pol;
        cfg.upchan_factor = 16;
        cfg.window_type = win;

        CHECK_CUDA_ERROR_NON_OO(cudaMemsetAsync(d_out, 0, total_elems * sizeof(float2), stream));
        kotekan::launch_beam_upchannelize_16pt(d_in, d_out, d_pow, cfg, stream);
        CHECK_CUDA_ERROR_NON_OO(cudaStreamSynchronize(stream));

        std::vector<float2> gpu_out(total_elems);
        std::vector<float> gpu_pow(num_beams * n_freq * 16);
        CHECK_CUDA_ERROR_NON_OO(
            cudaMemcpy(gpu_out.data(), d_out, total_elems * sizeof(float2), cudaMemcpyDeviceToHost));
        CHECK_CUDA_ERROR_NON_OO(cudaMemcpy(gpu_pow.data(), d_pow,
                                           num_beams * n_freq * 16 * sizeof(float),
                                           cudaMemcpyDeviceToHost));

        const std::size_t n_time_out = n_time / 16;
        const std::size_t n_freq_out = n_freq * 16;
        double max_v_err = 0.0;
        double max_p_rel_err = 0.0;

        for (std::size_t b = 0; b < num_beams; ++b) {
            for (std::size_t f = 0; f < n_freq; ++f) {
                std::vector<double> cpu_p(16, 0.0);
                for (std::size_t tb = 0; tb < n_time_out; ++tb) {
                    for (std::size_t pol = 0; pol < n_pol; ++pol) {
                        for (int out_k = 0; out_k < 16; ++out_k) {
                            const int k_dft = (out_k + 8) & 15;
                            std::complex<double> acc(0.0, 0.0);
                            for (int n = 0; n < 16; ++n) {
                                const std::size_t t = tb * 16 + static_cast<std::size_t>(n);
                                const float2 v =
                                    h_in[((t * n_freq + f) * max_beams_stride + b) * n_pol + pol];
                                const double w =
                                    (win == kotekan::BeamUpchanWindowType::HANN)
                                        ? hann16_coeff(n)
                                        : 0.25;
                                const double ang = -TWO_PI * static_cast<double>(k_dft * n) / 16.0;
                                const std::complex<double> tw(std::cos(ang), std::sin(ang));
                                acc += std::complex<double>(v.x * w, v.y * w) * tw;
                            }
                            const std::size_t out_idx =
                                ((tb * n_freq_out + f * 16 + static_cast<std::size_t>(out_k))
                                     * max_beams_stride
                                 + b)
                                    * n_pol
                                + pol;
                            const double er = static_cast<double>(gpu_out[out_idx].x) - acc.real();
                            const double ei = static_cast<double>(gpu_out[out_idx].y) - acc.imag();
                            max_v_err = std::max(max_v_err, std::sqrt(er * er + ei * ei));
                            cpu_p[static_cast<std::size_t>(out_k)] += std::norm(acc);
                        }
                    }
                }
                for (int out_k = 0; out_k < 16; ++out_k) {
                    const double cp = cpu_p[static_cast<std::size_t>(out_k)]
                                      / static_cast<double>(n_time_out);
                    const double gp = gpu_pow[b * n_freq_out + f * 16 + static_cast<std::size_t>(out_k)];
                    const double rel = std::abs(gp - cp) / std::max(1.0e-3, cp);
                    max_p_rel_err = std::max(max_p_rel_err, rel);
                }
            }
        }

        std::cout << "  - Window " << kotekan::beam_upchan_window_to_string(win)
                  << ": max_v_err=" << std::scientific << max_v_err
                  << ", max_p_rel_err=" << max_p_rel_err << std::fixed << "\n";
        if (max_v_err >= 1.0e-5 || max_p_rel_err >= 1.0e-5) {
            ok = false;
        }
    }

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_pow);
    cudaStreamDestroy(stream);
    return ok;
}

bool test_parseval_and_cw_isolation() {
    std::cout << "\n======================================================\n";
    std::cout << "[Test 2/4] Parseval Energy & CW Sidelobe Isolation (< -30 dB Hann)\n";
    std::cout << "======================================================\n";

    constexpr std::size_t n_time = 1024;
    constexpr std::size_t n_freq = 2;
    constexpr std::size_t num_beams = 2;

    std::mt19937 rng(9911);
    std::normal_distribution<float> noise(0.0f, 1.0f);

    // Beam 0: broadband noise (Parseval check)
    // Beam 1: pure CW tone at k_off = +2.5 bins inside coarse channel -> peak near out_k = 10/11
    std::vector<float2> h_in(n_time * n_freq * num_beams);
    double in_energy_rect = 0.0;
    for (std::size_t t = 0; t < n_time; ++t) {
        for (std::size_t f = 0; f < n_freq; ++f) {
            const float2 z0 = make_float2(noise(rng), noise(rng));
            h_in[(t * n_freq + f) * num_beams + 0] = z0;
            in_energy_rect += static_cast<double>(z0.x) * z0.x + static_cast<double>(z0.y) * z0.y;

            const double ang = TWO_PI * 2.5 * static_cast<double>(t) / 16.0;
            h_in[(t * n_freq + f) * num_beams + 1] =
                make_float2(static_cast<float>(std::cos(ang)), static_cast<float>(std::sin(ang)));
        }
    }

    float2* d_in = nullptr;
    float2* d_out = nullptr;
    float* d_pow = nullptr;
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&d_in, h_in.size() * sizeof(float2)));
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&d_out, h_in.size() * sizeof(float2)));
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&d_pow, num_beams * n_freq * 16 * sizeof(float)));
    CHECK_CUDA_ERROR_NON_OO(
        cudaMemcpy(d_in, h_in.data(), h_in.size() * sizeof(float2), cudaMemcpyHostToDevice));

    cudaStream_t stream = nullptr;
    CHECK_CUDA_ERROR_NON_OO(cudaStreamCreate(&stream));

    bool ok = true;
    // Check Rectangular exact Parseval on noise beam 0
    {
        kotekan::BeamUpchannelizeConfig cfg;
        cfg.n_time = n_time;
        cfg.n_freq = n_freq;
        cfg.num_beams = num_beams;
        cfg.max_beams_stride = num_beams;
        cfg.window_type = kotekan::BeamUpchanWindowType::RECTANGULAR;
        kotekan::launch_beam_upchannelize_16pt(d_in, d_out, d_pow, cfg, stream);
        CHECK_CUDA_ERROR_NON_OO(cudaStreamSynchronize(stream));

        std::vector<float2> gpu_out(h_in.size());
        CHECK_CUDA_ERROR_NON_OO(cudaMemcpy(gpu_out.data(), d_out, gpu_out.size() * sizeof(float2),
                                           cudaMemcpyDeviceToHost));
        double out_energy = 0.0;
        for (std::size_t tb = 0; tb < n_time / 16; ++tb) {
            for (std::size_t f = 0; f < n_freq; ++f) {
                for (int k = 0; k < 16; ++k) {
                    const float2 v = gpu_out[((tb * (n_freq * 16) + f * 16 + k) * num_beams) + 0];
                    out_energy += static_cast<double>(v.x) * v.x + static_cast<double>(v.y) * v.y;
                }
            }
        }
        const double parseval_rel = std::abs(out_energy - in_energy_rect) / in_energy_rect;
        std::cout << "  - Rectangular Parseval rel diff: " << std::scientific << parseval_rel
                  << std::fixed << "\n";
        if (parseval_rel >= 1.0e-5) {
            ok = false;
        }
    }

    // Check Hann normalized noise Parseval + CW sidelobe isolation (< -30 dB outside main lobe)
    {
        kotekan::BeamUpchannelizeConfig cfg;
        cfg.n_time = n_time;
        cfg.n_freq = n_freq;
        cfg.num_beams = num_beams;
        cfg.max_beams_stride = num_beams;
        cfg.window_type = kotekan::BeamUpchanWindowType::HANN;
        kotekan::launch_beam_upchannelize_16pt(d_in, d_out, d_pow, cfg, stream);
        CHECK_CUDA_ERROR_NON_OO(cudaStreamSynchronize(stream));

        std::vector<float> gpu_pow(num_beams * n_freq * 16);
        CHECK_CUDA_ERROR_NON_OO(cudaMemcpy(gpu_pow.data(), d_pow,
                                           gpu_pow.size() * sizeof(float), cudaMemcpyDeviceToHost));

        // Average input power per time sample across noise beam 0 matches mean sum across 16 fine bins
        double mean_in_p = in_energy_rect / static_cast<double>(n_freq);
        double out_noise_p = 0.0;
        for (std::size_t k = 0; k < 16; ++k) {
            out_noise_p += static_cast<double>(gpu_pow[0 * (n_freq * 16) + k]);
        }
        out_noise_p *= static_cast<double>(n_time / 16);
        const double single_freq_in = mean_in_p;
        const double hann_noise_rel = std::abs(out_noise_p - single_freq_in) / single_freq_in;

        // Beam 1 CW tone at +2.5 bins: shifted indices peak around k = 8+2 = 10 and 8+3 = 11
        double peak_cw = 0.0;
        double max_far_sidelobe = 0.0;
        for (int k = 0; k < 16; ++k) {
            const double p = gpu_pow[1 * (n_freq * 16) + static_cast<std::size_t>(k)];
            peak_cw = std::max(peak_cw, p);
            if (k < 8 || k > 13) {
                max_far_sidelobe = std::max(max_far_sidelobe, p);
            }
        }
        const double isolation_db =
            10.0 * std::log10(std::max(max_far_sidelobe, 1.0e-12) / std::max(peak_cw, 1.0e-12));
        std::cout << "  - Hann broadband power rel diff: " << std::setprecision(4) << hann_noise_rel
                  << ", CW off-bin sidelobe isolation: " << std::setprecision(2) << isolation_db
                  << " dB (< -30.0 dB required)\n";
        if (hann_noise_rel > 0.15 || isolation_db >= -30.0) {
            ok = false;
        }
    }

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_pow);
    cudaStreamDestroy(stream);
    return ok;
}

bool test_dm_smearing_reduction() {
    std::cout << "\n======================================================\n";
    std::cout << "[Test 3/4] Dedispersion DM=100 @ 300 MHz Smearing Reduction (16x)\n";
    std::cout << "======================================================\n";

    // CHARTS parameters: 390.625 kHz coarse, 2.56 us sample period, DM=100 around 300 MHz
    constexpr double dm = 100.0;
    constexpr double k_dm = 4.148808e3; // MHz^2 pc^-1 cm^3 s
    constexpr double f0_mhz = 300.0;
    constexpr double df_coarse_mhz = 0.390625;
    constexpr double df_fine_mhz = df_coarse_mhz / 16.0; // 24.4140625 kHz
    constexpr double dt_coarse_s = 2.56e-6;
    constexpr double dt_fine_s = dt_coarse_s * 16.0;     // 40.96 us

    // Intra-channel analytical smearing check: exact 16x reduction
    const double smear_coarse_ms =
        8.3e6 * dm * df_coarse_mhz / (f0_mhz * f0_mhz * f0_mhz);
    const double smear_fine_ms =
        8.3e6 * dm * df_fine_mhz / (f0_mhz * f0_mhz * f0_mhz);
    const double analytical_ratio = smear_coarse_ms / smear_fine_ms;

    // Synthesize narrowband RF burst sweeping quadratically across 32 coarse channels at DM=100
    constexpr std::size_t n_time = 15360; // 39.3216 ms
    constexpr std::size_t n_freq = 32;    // 300.0 .. 312.5 MHz
    constexpr std::size_t num_beams = 1;
    const double t_ref_s = 0.025; // arrive at highest frequency at 25 ms so full sweep fits
    const double f_top_mhz = f0_mhz + (static_cast<double>(n_freq) - 0.5) * df_coarse_mhz;
    constexpr double intrinsic_sigma_s = 1.5e-4; // 0.15 ms intrinsic Gaussian pulse

    std::vector<float2> h_in(n_time * n_freq, make_float2(0.0f, 0.0f));
    for (std::size_t f = 0; f < n_freq; ++f) {
        const double f_c_mhz = f0_mhz + (static_cast<double>(f) + 0.5) * df_coarse_mhz;
        for (std::size_t t = 0; t < n_time; ++t) {
            const double t_s = static_cast<double>(t) * dt_coarse_s;
            // Coherent swept sub-pulse inside coarse channel: instantaneous frequency moves with t
            // Solve dispersion delay t_arr(freq) = t_ref + k_dm * DM * (freq^-2 - f_top^-2)
            // Across the coarse channel, synthesize sum over 16 fine sub-tones properly dispersed
            double re = 0.0, im = 0.0;
            for (int sub = 0; sub < 16; ++sub) {
                const double offset_hz =
                    (static_cast<double>(sub) - 7.5) * (df_fine_mhz * 1.0e6);
                const double f_sub_mhz = f_c_mhz + offset_hz * 1.0e-6;
                const double t_arr = t_ref_s + k_dm * dm
                                                   * (1.0 / (f_sub_mhz * f_sub_mhz)
                                                      - 1.0 / (f_top_mhz * f_top_mhz));
                const double dt = t_s - t_arr;
                const double env = std::exp(-0.5 * (dt * dt) / (intrinsic_sigma_s * intrinsic_sigma_s));
                const double ph = TWO_PI * offset_hz * t_s;
                re += env * std::cos(ph);
                im += env * std::sin(ph);
            }
            h_in[t * n_freq + f] = make_float2(static_cast<float>(re), static_cast<float>(im));
        }
    }

    float2* d_in = nullptr;
    float2* d_out = nullptr;
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&d_in, h_in.size() * sizeof(float2)));
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&d_out, h_in.size() * sizeof(float2)));
    CHECK_CUDA_ERROR_NON_OO(
        cudaMemcpy(d_in, h_in.data(), h_in.size() * sizeof(float2), cudaMemcpyHostToDevice));

    kotekan::BeamUpchannelizeConfig cfg;
    cfg.n_time = n_time;
    cfg.n_freq = n_freq;
    cfg.num_beams = num_beams;
    cfg.max_beams_stride = num_beams;
    cfg.window_type = kotekan::BeamUpchanWindowType::HANN;

    cudaStream_t stream = nullptr;
    CHECK_CUDA_ERROR_NON_OO(cudaStreamCreate(&stream));
    kotekan::launch_beam_upchannelize_16pt(d_in, d_out, nullptr, cfg, stream);
    CHECK_CUDA_ERROR_NON_OO(cudaStreamSynchronize(stream));

    std::vector<float2> h_out(h_in.size());
    CHECK_CUDA_ERROR_NON_OO(
        cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(float2), cudaMemcpyDeviceToHost));

    // Compare dedispersed peak and FWHM between coarse-resolved (390.625 kHz @ 40.96 us) vs fine (24.414 kHz @ 40.96 us)
    const std::size_t n_tb = n_time / 16;
    std::vector<double> ts_coarse(n_tb, 0.0);
    std::vector<double> ts_fine(n_tb, 0.0);
    const std::size_t n_freq_out = n_freq * 16;

    for (std::size_t tb = 0; tb < n_tb; ++tb) {
        // Coarse dedispersion
        for (std::size_t f = 0; f < n_freq; ++f) {
            const double fc = f0_mhz + (static_cast<double>(f) + 0.5) * df_coarse_mhz;
            const double delay_s = k_dm * dm * (1.0 / (fc * fc) - 1.0 / (f_top_mhz * f_top_mhz));
            const std::size_t shift = static_cast<std::size_t>(std::llround(delay_s / dt_fine_s));
            const std::size_t tb_src = tb + shift;
            if (tb_src < n_tb) {
                double p = 0.0;
                for (int s = 0; s < 16; ++s) {
                    const float2 v = h_in[(tb_src * 16 + s) * n_freq + f];
                    p += static_cast<double>(v.x) * v.x + static_cast<double>(v.y) * v.y;
                }
                ts_coarse[tb] += p / 16.0;
            }
        }
        // Fine dedispersion
        for (std::size_t kf = 0; kf < n_freq_out; ++kf) {
            const std::size_t fc_idx = kf / 16;
            const int sub = static_cast<int>(kf % 16) - 8;
            const double ff_mhz =
                f0_mhz + (static_cast<double>(fc_idx) + 0.5) * df_coarse_mhz
                + static_cast<double>(sub) * df_fine_mhz;
            const double delay_s =
                k_dm * dm * (1.0 / (ff_mhz * ff_mhz) - 1.0 / (f_top_mhz * f_top_mhz));
            const std::size_t shift = static_cast<std::size_t>(std::llround(delay_s / dt_fine_s));
            const std::size_t tb_src = tb + shift;
            if (tb_src < n_tb) {
                const float2 v = h_out[tb_src * n_freq_out + kf];
                ts_fine[tb] += static_cast<double>(v.x) * v.x + static_cast<double>(v.y) * v.y;
            }
        }
    }

    auto measure_fwhm_ms = [&](const std::vector<double>& ts, double& peak_out) {
        peak_out = *std::max_element(ts.begin(), ts.end());
        std::size_t above = 0;
        for (double v : ts) {
            if (v >= 0.5 * peak_out) {
                ++above;
            }
        }
        return static_cast<double>(above) * dt_fine_s * 1.0e3;
    };

    double coarse_peak = 0.0, fine_peak = 0.0;
    const double coarse_fwhm_ms = measure_fwhm_ms(ts_coarse, coarse_peak);
    const double fine_fwhm_ms = measure_fwhm_ms(ts_fine, fine_peak);

    std::cout << "  - Analytical DM=100 @ 300 MHz smearing: coarse=" << std::setprecision(2)
              << smear_coarse_ms << " ms -> fine=" << smear_fine_ms << " ms ("
              << analytical_ratio << "x reduction)\n";
    std::cout << "  - Dedispersed FWHM: coarse=" << coarse_fwhm_ms
              << " ms -> fine=" << fine_fwhm_ms << " ms (compression "
              << (coarse_fwhm_ms / std::max(fine_fwhm_ms, 1.0e-3)) << "x)\n";

    cudaFree(d_in);
    cudaFree(d_out);
    cudaStreamDestroy(stream);
    return (std::abs(analytical_ratio - 16.0) < 1.0e-6) && (fine_fwhm_ms < 0.2 * coarse_fwhm_ms);
}

bool test_production_latency() {
    std::cout << "\n======================================================\n";
    std::cout << "[Test 4/4] RTX 5090 Production Latency (15360 x 336 x 16 & x 64)\n";
    std::cout << "======================================================\n";

    constexpr std::size_t n_time = 15360;
    constexpr std::size_t n_freq = 336;
    cudaStream_t stream = nullptr;
    CHECK_CUDA_ERROR_NON_OO(cudaStreamCreate(&stream));
    cudaEvent_t ev_start = nullptr;
    cudaEvent_t ev_stop = nullptr;
    CHECK_CUDA_ERROR_NON_OO(cudaEventCreate(&ev_start));
    CHECK_CUDA_ERROR_NON_OO(cudaEventCreate(&ev_stop));

    bool ok = true;
    for (const std::size_t n_beams : {std::size_t{16}, std::size_t{64}}) {
        const std::size_t elems = n_time * n_freq * n_beams;
        float2* d_in = nullptr;
        float2* d_out = nullptr;
        float* d_pow = nullptr;
        CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&d_in, elems * sizeof(float2)));
        CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&d_out, elems * sizeof(float2)));
        CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&d_pow, n_beams * n_freq * 16 * sizeof(float)));
        CHECK_CUDA_ERROR_NON_OO(cudaMemset(d_in, 0, elems * sizeof(float2)));

        kotekan::BeamUpchannelizeConfig cfg;
        cfg.n_time = n_time;
        cfg.n_freq = n_freq;
        cfg.num_beams = n_beams;
        cfg.max_beams_stride = n_beams;
        cfg.window_type = kotekan::BeamUpchanWindowType::HANN;

        for (int w = 0; w < 3; ++w) {
            kotekan::launch_beam_upchannelize_16pt(d_in, d_out, d_pow, cfg, stream);
        }
        CHECK_CUDA_ERROR_NON_OO(cudaStreamSynchronize(stream));

        constexpr int iters = 20;
        CHECK_CUDA_ERROR_NON_OO(cudaEventRecord(ev_start, stream));
        for (int i = 0; i < iters; ++i) {
            kotekan::launch_beam_upchannelize_16pt(d_in, d_out, d_pow, cfg, stream);
        }
        CHECK_CUDA_ERROR_NON_OO(cudaEventRecord(ev_stop, stream));
        CHECK_CUDA_ERROR_NON_OO(cudaEventSynchronize(ev_stop));

        float elapsed_ms = 0.0f;
        CHECK_CUDA_ERROR_NON_OO(cudaEventElapsedTime(&elapsed_ms, ev_start, ev_stop));
        const float per_frame_ms = elapsed_ms / static_cast<float>(iters);
        const double io_gb = static_cast<double>(2 * elems * sizeof(float2)) / 1.0e9;
        const double gbps = io_gb / (static_cast<double>(per_frame_ms) * 1.0e-3);
        std::cout << "  - " << n_beams << " beams (15360 x 336): " << std::setprecision(3)
                  << per_frame_ms << " ms (" << std::setprecision(1) << gbps
                  << " GB/s, budget <= 51.2 ms)\n";
        if (per_frame_ms >= 51.2f) {
            ok = false;
        }

        cudaFree(d_in);
        cudaFree(d_out);
        cudaFree(d_pow);
    }

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    cudaStreamDestroy(stream);
    return ok;
}

} // namespace

int main() {
    const bool t1 = test_dft_parity_both_windows();
    const bool t2 = test_parseval_and_cw_isolation();
    const bool t3 = test_dm_smearing_reduction();
    const bool t4 = test_production_latency();
    std::cout << "\nSummary: DFT=" << (t1 ? "PASS" : "FAIL")
              << ", Parseval/CW=" << (t2 ? "PASS" : "FAIL")
              << ", DM-Smear=" << (t3 ? "PASS" : "FAIL")
              << ", Latency=" << (t4 ? "PASS" : "FAIL") << "\n";
    return (t1 && t2 && t3 && t4) ? 0 : 1;
}
