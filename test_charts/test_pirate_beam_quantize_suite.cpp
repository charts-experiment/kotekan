#include "cudaPirateBeamQuantize.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <numeric>
#include <random>
#include <vector>

using namespace kotekan;

namespace {

#define CUDA_CHECK(call)                                                                           \
    do {                                                                                           \
        cudaError_t err = (call);                                                                  \
        if (err != cudaSuccess) {                                                                  \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,                 \
                         cudaGetErrorString(err));                                                 \
            return 1;                                                                              \
        }                                                                                          \
    } while (0)

inline float half_bits_to_float(std::uint16_t h) {
    const std::uint32_t sign = (static_cast<std::uint32_t>(h & 0x8000U)) << 16;
    const std::uint32_t exp = (h >> 10) & 0x1fU;
    const std::uint32_t mant = h & 0x03ffU;
    std::uint32_t f_bits = 0;
    if (exp == 0) {
        if (mant == 0) {
            f_bits = sign;
        } else {
            std::uint32_t m = mant;
            std::uint32_t e = 127 - 15 + 1;
            while ((m & 0x0400U) == 0) {
                m <<= 1;
                --e;
            }
            m &= 0x03ffU;
            f_bits = sign | (e << 23) | (m << 13);
        }
    } else if (exp == 31) {
        f_bits = sign | 0x7f800000U | (mant << 13);
    } else {
        f_bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float out = 0.0f;
    std::memcpy(&out, &f_bits, sizeof(float));
    return out;
}

inline int decode_nibble_signed(std::uint8_t byte, bool high) {
    int n = high ? ((byte >> 4) & 0x0F) : (byte & 0x0F);
    return (n >= 8) ? (n - 16) : n;
}

struct PirateBuffers {
    float2* d_formed = nullptr;
    std::uint8_t* d_rfi = nullptr;
    float* d_int = nullptr;
    std::uint8_t* d_val = nullptr;
    std::uint8_t* d_out = nullptr;

    ~PirateBuffers() {
        if (d_formed)
            cudaFree(d_formed);
        if (d_rfi)
            cudaFree(d_rfi);
        if (d_int)
            cudaFree(d_int);
        if (d_val)
            cudaFree(d_val);
        if (d_out)
            cudaFree(d_out);
    }
};

bool run_quantize(const std::vector<float2>& h_formed,
                  const std::vector<std::uint8_t>& h_rfi,
                  const PirateBeamQuantizeConfig& cfg,
                  std::vector<std::uint8_t>& h_out) {
    const std::size_t n_frb = cfg.samples_per_data_set / cfg.seq_per_frb_time_sample;
    const std::size_t n_mc = n_frb / PIRATE_SAMPLES_PER_MINICHUNK;
    const std::size_t out_bytes = n_mc * cfg.num_beams * cfg.num_freq * PIRATE_BYTES_PER_CHAN;
    h_out.assign(out_bytes, 0);

    PirateBuffers bufs;
    if (cudaMalloc(&bufs.d_formed, h_formed.size() * sizeof(float2)) != cudaSuccess)
        return false;
    if (!h_rfi.empty()
        && cudaMalloc(&bufs.d_rfi, h_rfi.size() * sizeof(std::uint8_t)) != cudaSuccess)
        return false;
    if (cudaMalloc(&bufs.d_int, cfg.num_beams * cfg.num_freq * n_frb * sizeof(float))
        != cudaSuccess)
        return false;
    if (cudaMalloc(&bufs.d_val, cfg.num_freq * n_frb * sizeof(std::uint8_t)) != cudaSuccess)
        return false;
    if (cudaMalloc(&bufs.d_out, out_bytes) != cudaSuccess)
        return false;

    if (cudaMemcpy(bufs.d_formed, h_formed.data(), h_formed.size() * sizeof(float2),
                   cudaMemcpyHostToDevice)
        != cudaSuccess)
        return false;
    if (!h_rfi.empty()
        && cudaMemcpy(bufs.d_rfi, h_rfi.data(), h_rfi.size() * sizeof(std::uint8_t),
                      cudaMemcpyHostToDevice)
               != cudaSuccess)
        return false;

    launch_pirate_beam_quantize(bufs.d_formed, h_rfi.empty() ? nullptr : bufs.d_rfi, bufs.d_int,
                                bufs.d_val, bufs.d_out, cfg, nullptr);
    if (cudaDeviceSynchronize() != cudaSuccess)
        return false;
    if (cudaMemcpy(h_out.data(), bufs.d_out, out_bytes, cudaMemcpyDeviceToHost) != cudaSuccess)
        return false;
    return true;
}

void unpack_minichunk(const std::vector<std::uint8_t>& buf,
                      std::size_t mc,
                      std::size_t num_beams,
                      std::size_t num_freq,
                      std::size_t b,
                      std::size_t f,
                      float& out_scale,
                      float& out_offset,
                      std::vector<int>& out_q) {
    const std::size_t mc_bytes = num_beams * num_freq * PIRATE_BYTES_PER_CHAN;
    const std::uint8_t* mc_ptr = buf.data() + mc * mc_bytes;
    const std::uint8_t* s_ptr = mc_ptr + (b * num_freq + f) * PIRATE_SCALES_BYTES_PER_CHAN;
    std::uint16_t raw_so[2] = {0, 0};
    std::memcpy(raw_so, s_ptr, sizeof(raw_so));
    out_scale = half_bits_to_float(raw_so[0]);  // scale first
    out_offset = half_bits_to_float(raw_so[1]); // offset second

    const std::uint8_t* d_ptr =
        mc_ptr + num_beams * num_freq * PIRATE_SCALES_BYTES_PER_CHAN
        + (b * num_freq + f) * PIRATE_DATA_BYTES_PER_CHAN;
    out_q.resize(PIRATE_SAMPLES_PER_MINICHUNK);
    for (std::size_t i = 0; i < PIRATE_DATA_BYTES_PER_CHAN; ++i) {
        out_q[2 * i + 0] = decode_nibble_signed(d_ptr[i], false);
        out_q[2 * i + 1] = decode_nibble_signed(d_ptr[i], true);
    }
}

} // namespace

int main() {
    std::printf("==================================================================================\n");
    std::printf(" CHARTS Pirate FRB Beam Quantizer Test Suite (GPU-4.1)\n");
    std::printf("==================================================================================\n");

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count <= 0) {
        std::fprintf(stderr, "No CUDA device available\n");
        return 1;
    }

    // --------------------------------------------------------------------------
    // Test 1: Wire order (scale, offset), nibble packing & reconstruction fidelity
    // --------------------------------------------------------------------------
    {
        PirateBeamQuantizeConfig cfg;
        cfg.num_beams = 1;
        cfg.max_beams = 1;
        cfg.num_freq = 2;
        cfg.num_pols = 1;
        cfg.samples_per_data_set = 1024;
        cfg.seq_per_frb_time_sample = 4; // 256 FRB samples -> 1 minichunk
        cfg.enable_incoherent_subtraction = false;
        cfg.enable_highpass_detrend = false;
        cfg.use_robust_mad_scale = false;
        cfg.stddev_cutoff = 3.0f;

        std::vector<float2> h_formed(cfg.samples_per_data_set * cfg.num_freq * cfg.max_beams,
                                     make_float2(0.0f, 0.0f));
        std::vector<float> expected_f0(256, 0.0f);
        for (int tau = 0; tau < 256; ++tau) {
            const float target_i0 = 50.0f + 12.0f * std::sin(2.0f * 3.14159265f * tau / 32.0f);
            expected_f0[tau] = target_i0;
            const float amp0 = std::sqrt(target_i0);
            for (int dt = 0; dt < 4; ++dt) {
                const std::size_t t = static_cast<std::size_t>(tau * 4 + dt);
                h_formed[t * cfg.num_freq + 0] = make_float2(amp0, 0.0f);
                h_formed[t * cfg.num_freq + 1] = make_float2(3.0f, 4.0f); // intensity = 25
            }
        }

        std::vector<std::uint8_t> h_out;
        if (!run_quantize(h_formed, {}, cfg, h_out))
            return 1;
        if (h_out.size() != cfg.num_beams * cfg.num_freq * PIRATE_BYTES_PER_CHAN) {
            std::fprintf(stderr, "Test 1 FAILED: unexpected byte count %zu\n", h_out.size());
            return 1;
        }

        float scale = 0.0f, offset = 0.0f;
        std::vector<int> q;
        unpack_minichunk(h_out, 0, cfg.num_beams, cfg.num_freq, 0, 0, scale, offset, q);
        if (!(offset > 48.0f && offset < 52.0f) || !(scale > 2.5f && scale < 4.5f)) {
            std::fprintf(stderr, "Test 1 FAILED: wrong (scale, offset) order or range: s=%.4f o=%.4f\n",
                         scale, offset);
            return 1;
        }
        double max_abs_err = 0.0;
        for (int tau = 0; tau < 256; ++tau) {
            if (q[tau] < -7 || q[tau] > 7) {
                std::fprintf(stderr, "Test 1 FAILED: unexpected q[%d]=%d\n", tau, q[tau]);
                return 1;
            }
            const double recon = static_cast<double>(scale) * q[tau] + offset;
            max_abs_err = std::max(max_abs_err, std::abs(recon - expected_f0[tau]));
        }
        if (max_abs_err > 1.0 * scale) {
            std::fprintf(stderr, "Test 1 FAILED: max_abs_err=%.4f > scale=%.4f\n", max_abs_err,
                         scale);
            return 1;
        }
        std::printf("[PASS] Test 1: Wire format (scale=%.3f, offset=%.2f) & reconstruction err=%.3f\n",
                    scale, offset, max_abs_err);
    }

    // --------------------------------------------------------------------------
    // Test 2: RFI mask propagation -> -8 sentinel & outlier immunity
    // --------------------------------------------------------------------------
    {
        PirateBeamQuantizeConfig cfg;
        cfg.num_beams = 1;
        cfg.max_beams = 1;
        cfg.num_freq = 2;
        cfg.num_pols = 1;
        cfg.samples_per_data_set = 1024;
        cfg.seq_per_frb_time_sample = 4;
        cfg.rfi_chunk_size = 16; // 64 RFI chunks of 4 FRB samples each
        cfg.rfi_mask_fraction_threshold = 0.5f;
        cfg.enable_incoherent_subtraction = false;
        cfg.enable_highpass_detrend = false;
        cfg.use_robust_mad_scale = true;
        cfg.stddev_cutoff = 3.0f;

        std::vector<float2> h_formed(cfg.samples_per_data_set * cfg.num_freq, make_float2(0.0f, 0.0f));
        const std::size_t n_chunks = cfg.samples_per_data_set / cfg.rfi_chunk_size;
        std::vector<std::uint8_t> h_rfi(n_chunks * cfg.num_freq, 0);

        for (int tau = 0; tau < 256; ++tau) {
            const bool masked = (tau >= 40 && tau < 60);
            const float val = masked ? 1.0e6f : (20.0f + 4.0f * std::cos(0.1f * tau));
            const float amp = std::sqrt(val);
            for (int dt = 0; dt < 4; ++dt) {
                const std::size_t t = static_cast<std::size_t>(tau * 4 + dt);
                h_formed[t * cfg.num_freq + 0] = make_float2(amp, 0.0f);
                h_formed[t * cfg.num_freq + 1] = make_float2(2.0f, 0.0f);
                if (masked) {
                    const std::size_t c = t / cfg.rfi_chunk_size;
                    h_rfi[c * cfg.num_freq + 0] = 1;
                }
            }
        }
        // Fully mask channel 1 to verify zero scale/offset and all -8 codes.
        for (std::size_t c = 0; c < n_chunks; ++c) {
            h_rfi[c * cfg.num_freq + 1] = 1;
        }

        std::vector<std::uint8_t> h_out;
        if (!run_quantize(h_formed, h_rfi, cfg, h_out))
            return 1;

        float s0 = 0.0f, o0 = 0.0f, s1 = 0.0f, o1 = 0.0f;
        std::vector<int> q0, q1;
        unpack_minichunk(h_out, 0, cfg.num_beams, cfg.num_freq, 0, 0, s0, o0, q0);
        unpack_minichunk(h_out, 0, cfg.num_beams, cfg.num_freq, 0, 1, s1, o1, q1);

        if (!(o0 > 18.0f && o0 < 22.0f) || !(s0 > 0.5f && s0 < 4.0f)) {
            std::fprintf(stderr, "Test 2 FAILED: masked RFI biased stats (s0=%.3f o0=%.3f)\n",
                         s0, o0);
            return 1;
        }
        for (int tau = 0; tau < 256; ++tau) {
            const bool expect_mask = (tau >= 40 && tau < 60);
            if (expect_mask && q0[tau] != -8) {
                std::fprintf(stderr, "Test 2 FAILED: expected -8 at tau=%d, got %d\n", tau, q0[tau]);
                return 1;
            }
            if (!expect_mask && q0[tau] == -8) {
                std::fprintf(stderr, "Test 2 FAILED: unexpected -8 at tau=%d\n", tau);
                return 1;
            }
            if (q1[tau] != -8) {
                std::fprintf(stderr, "Test 2 FAILED: fully masked ch1 tau=%d got %d\n", tau, q1[tau]);
                return 1;
            }
        }
        if (s1 != 0.0f || o1 != 0.0f) {
            std::fprintf(stderr, "Test 2 FAILED: fully masked ch1 non-zero s1=%.3f o1=%.3f\n", s1, o1);
            return 1;
        }
        std::printf("[PASS] Test 2: RFI mask -> -8 sentinel & zero outlier contamination (s0=%.3f o0=%.2f)\n",
                    s0, o0);
    }

    // --------------------------------------------------------------------------
    // Test 3: Incoherent-beam subtraction removes common-mode modulation across 4 beams
    // --------------------------------------------------------------------------
    {
        PirateBeamQuantizeConfig cfg;
        cfg.num_beams = 4;
        cfg.max_beams = 6; // Also exercise non-packed max_beams stride!
        cfg.num_freq = 2;
        cfg.num_pols = 1;
        cfg.samples_per_data_set = 512;
        cfg.seq_per_frb_time_sample = 2; // 256 FRB samples
        cfg.enable_incoherent_subtraction = true;
        cfg.enable_highpass_detrend = false;
        cfg.use_robust_mad_scale = false;

        std::vector<float2> h_formed(cfg.samples_per_data_set * cfg.num_freq * cfg.max_beams,
                                     make_float2(0.0f, 0.0f));
        for (int tau = 0; tau < 256; ++tau) {
            const float common = 40.0f + 15.0f * std::sin(2.0f * 3.14159265f * tau / 16.0f);
            for (int t_off = 0; t_off < 2; ++t_off) {
                const std::size_t t = static_cast<std::size_t>(tau * 2 + t_off);
                for (std::size_t f = 0; f < cfg.num_freq; ++f) {
                    for (std::size_t b = 0; b < cfg.num_beams; ++b) {
                        const float frb_only = (b == 0 && tau == 128) ? 20.0f : 0.0f;
                        const float amp = std::sqrt(common + frb_only);
                        const std::size_t idx = (t * cfg.num_freq + f) * cfg.max_beams + b;
                        h_formed[idx] = make_float2(amp, 0.0f);
                    }
                }
            }
        }

        std::vector<std::uint8_t> h_out;
        if (!run_quantize(h_formed, {}, cfg, h_out))
            return 1;

        float s_b0 = 0.0f, o_b0 = 0.0f, s_b1 = 0.0f, o_b1 = 0.0f;
        std::vector<int> q_b0, q_b1;
        unpack_minichunk(h_out, 0, cfg.num_beams, cfg.num_freq, 0, 0, s_b0, o_b0, q_b0);
        unpack_minichunk(h_out, 0, cfg.num_beams, cfg.num_freq, 1, 0, s_b1, o_b1, q_b1);
        // After incoherent mean subtraction:
        //   tau != 128 -> 0 for all beams; tau == 128 -> +15 in b0, -5 in b1.
        if (q_b0[128] != 7 || q_b1[128] != -7 || std::abs(q_b0[10]) > 1) {
            std::fprintf(stderr, "Test 3 FAILED: q_b0[128]=%d q_b1[128]=%d q_b0[10]=%d\n",
                         q_b0[128], q_b1[128], q_b0[10]);
            return 1;
        }
        std::printf("[PASS] Test 3: Incoherent subtraction cancelled common mode (b0[128]=%d, b1[128]=%d, b0[10]=%d)\n",
                    q_b0[128], q_b1[128], q_b0[10]);
    }

    // --------------------------------------------------------------------------
    // Test 4: High-pass detrending removes slow ramp while keeping narrow FRB pulse
    // --------------------------------------------------------------------------
    {
        PirateBeamQuantizeConfig cfg;
        cfg.num_beams = 1;
        cfg.max_beams = 1;
        cfg.num_freq = 1;
        cfg.samples_per_data_set = 512;
        cfg.seq_per_frb_time_sample = 2;
        cfg.enable_incoherent_subtraction = false;
        cfg.enable_highpass_detrend = true;
        cfg.highpass_window_samples = 65;
        cfg.use_robust_mad_scale = true;

        std::vector<float2> h_formed(cfg.samples_per_data_set, make_float2(0.0f, 0.0f));
        std::mt19937 rng(42);
        std::normal_distribution<float> noise(0.0f, 0.4f);
        for (int tau = 0; tau < 256; ++tau) {
            const float baseline_ramp = 20.0f + 80.0f * (static_cast<float>(tau) / 255.0f);
            const float pulse = (tau >= 120 && tau <= 122) ? 18.0f : 0.0f;
            const float amp = std::sqrt(std::max(0.1f, baseline_ramp + noise(rng) + pulse));
            h_formed[2 * tau + 0] = make_float2(amp, 0.0f);
            h_formed[2 * tau + 1] = make_float2(amp, 0.0f);
        }

        std::vector<std::uint8_t> h_out;
        if (!run_quantize(h_formed, {}, cfg, h_out))
            return 1;

        float s = 0.0f, o = 0.0f;
        std::vector<int> q;
        unpack_minichunk(h_out, 0, 1, 1, 0, 0, s, o, q);
        if (std::abs(o) > 1.5f || q[121] != 7 || std::abs(q[60]) > 3 || std::abs(q[200]) > 3) {
            std::fprintf(stderr, "Test 4 FAILED: o=%.3f q[121]=%d q[60]=%d q[200]=%d\n",
                         o, q[121], q[60], q[200]);
            return 1;
        }
        std::printf("[PASS] Test 4: High-pass detrending removed ramp (offset=%.3f) and preserved pulse (q[121]=%d)\n",
                    o, q[121]);
    }

    // --------------------------------------------------------------------------
    // Test 5: Robust MAD scaling immune to a 50-sigma burst on background RMS
    // --------------------------------------------------------------------------
    {
        PirateBeamQuantizeConfig cfg;
        cfg.num_beams = 1;
        cfg.max_beams = 1;
        cfg.num_freq = 1;
        cfg.samples_per_data_set = 512;
        cfg.seq_per_frb_time_sample = 2;
        cfg.enable_incoherent_subtraction = false;
        cfg.enable_highpass_detrend = false;
        cfg.use_robust_mad_scale = true;
        cfg.stddev_cutoff = 3.0f;

        std::vector<float2> h_formed(cfg.samples_per_data_set);
        std::mt19937 rng(7);
        std::normal_distribution<float> g(0.0f, 1.0f);
        for (int tau = 0; tau < 256; ++tau) {
            const float x = (tau == 180) ? 150.0f : (30.0f + g(rng));
            const float amp = std::sqrt(std::max(0.1f, x));
            h_formed[2 * tau + 0] = make_float2(amp, 0.0f);
            h_formed[2 * tau + 1] = make_float2(amp, 0.0f);
        }

        std::vector<std::uint8_t> h_out;
        if (!run_quantize(h_formed, {}, cfg, h_out))
            return 1;

        float s_mad = 0.0f, o_mad = 0.0f;
        std::vector<int> q_mad;
        unpack_minichunk(h_out, 0, 1, 1, 0, 0, s_mad, o_mad, q_mad);

        cfg.use_robust_mad_scale = false;
        if (!run_quantize(h_formed, {}, cfg, h_out))
            return 1;
        float s_std = 0.0f, o_std = 0.0f;
        std::vector<int> q_std;
        unpack_minichunk(h_out, 0, 1, 1, 0, 0, s_std, o_std, q_std);

        // Background sigma=1 -> ideal scale = 2*3*1/15 = 0.4; non-robust stddev is ~7.5x inflated!
        if (!(s_mad < 0.7f && s_std > 2.0f && q_mad[180] == 7)) {
            std::fprintf(stderr, "Test 5 FAILED: s_mad=%.3f s_std=%.3f q_mad[180]=%d\n",
                         s_mad, s_std, q_mad[180]);
            return 1;
        }
        std::printf("[PASS] Test 5: Robust MAD scale immune to 50-sigma spike (s_mad=%.3f vs s_std=%.3f)\n",
                    s_mad, s_std);
    }

    // --------------------------------------------------------------------------
    // Test 6: Multi-minichunk + beam slice layout matching pirateNetworkSend
    // --------------------------------------------------------------------------
    {
        PirateBeamQuantizeConfig cfg;
        cfg.num_beams = 4;
        cfg.max_beams = 4;
        cfg.num_freq = 8;
        cfg.num_pols = 2; // dual-pol Stokes I = |Vx|^2 + |Vy|^2
        cfg.samples_per_data_set = 1024;
        cfg.seq_per_frb_time_sample = 2; // 512 FRB samples -> 2 minichunks
        cfg.enable_incoherent_subtraction = false;
        cfg.enable_highpass_detrend = false;
        cfg.use_robust_mad_scale = false;

        std::vector<float2> h_formed(
            cfg.samples_per_data_set * cfg.num_freq * cfg.max_beams * cfg.num_pols);
        for (std::size_t t = 0; t < cfg.samples_per_data_set; ++t) {
            const float mc_boost = (t >= 512) ? 50.0f : 0.0f;
            for (std::size_t f = 0; f < cfg.num_freq; ++f) {
                for (std::size_t b = 0; b < cfg.num_beams; ++b) {
                    const std::size_t base =
                        ((t * cfg.num_freq + f) * cfg.max_beams + b) * cfg.num_pols;
                    const float pwr = 10.0f + 5.0f * b + 2.0f * f + mc_boost;
                    h_formed[base + 0] = make_float2(std::sqrt(0.5f * pwr), 0.0f);
                    h_formed[base + 1] = make_float2(0.0f, std::sqrt(0.5f * pwr));
                }
            }
        }

        std::vector<std::uint8_t> h_out;
        if (!run_quantize(h_formed, {}, cfg, h_out))
            return 1;

        const std::size_t expected_bytes = 2 * cfg.num_beams * cfg.num_freq * 132;
        if (h_out.size() != expected_bytes) {
            std::fprintf(stderr, "Test 6 FAILED: size=%zu != expected=%zu\n", h_out.size(),
                         expected_bytes);
            return 1;
        }
        float s_mc0 = 0.0f, o_mc0 = 0.0f, s_mc1 = 0.0f, o_mc1 = 0.0f;
        std::vector<int> q_mc0, q_mc1;
        unpack_minichunk(h_out, 0, cfg.num_beams, cfg.num_freq, 2, 5, s_mc0, o_mc0, q_mc0);
        unpack_minichunk(h_out, 1, cfg.num_beams, cfg.num_freq, 2, 5, s_mc1, o_mc1, q_mc1);
        // Expected power at b=2, f=5: mc0 = 10 + 10 + 10 = 30; mc1 = 80.
        if (std::abs(o_mc0 - 30.0f) > 0.2f || std::abs(o_mc1 - 80.0f) > 0.2f) {
            std::fprintf(stderr, "Test 6 FAILED: o_mc0=%.2f o_mc1=%.2f\n", o_mc0, o_mc1);
            return 1;
        }
        std::printf("[PASS] Test 6: Multi-minichunk dual-pol wire layout verified (%zu bytes, o_mc0=%.1f, o_mc1=%.1f)\n",
                    h_out.size(), o_mc0, o_mc1);
    }

    std::printf("\nAll 6 Pirate FRB Quantizer tests PASSED!\n");
    return 0;
}
