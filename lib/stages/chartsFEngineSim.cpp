#include "chartsFEngineSim.hpp"

#include "StageFactory.hpp"
#include "chordMetadata.hpp"
#include "kotekanLogging.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <unistd.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

using kotekan::Config;
using kotekan::Stage;
using kotekan::bufferContainer;

REGISTER_KOTEKAN_STAGE(chartsFEngineSim);

namespace {
constexpr double C_LIGHT = 299792458.0;
constexpr double PI = 3.14159265358979323846;
constexpr double K_DM = 4148.808; // s * MHz^2 / (pc * cm^-3)

struct TargetInfo {
    const char* name;
    double ra_deg;
    double dec_deg;
    double nominal_amp;
    double dm;
    double drift_dl;
    double drift_dm;
    bool is_frb;
};

const TargetInfo* lookup_target(const std::string& scenario_name) {
    static const std::vector<std::pair<std::string, TargetInfo>> catalog = {
        {"vela", {"Vela Pulsar", 128.836, -45.176, 3.0, 0.0, 0.0, 0.0, false}},
        {"sun", {"The Sun", 0.0, 0.0, 5.5, 0.0, 0.0, 0.0, false}},
        {"sgr_a", {"Sagittarius A*", 266.417, -29.008, 4.5, 0.0, 0.0, 0.0, false}},
        {"cen_a", {"Centaurus A", 201.365, -43.019, 4.0, 0.0, 0.0, 0.0, false}},
        {"crab", {"Taurus A / Crab", 83.633, +22.014, 3.5, 0.0, 0.0, 0.0, false}},
        {"pictor_a", {"Pictor A", 79.958, -45.779, 3.2, 0.0, 0.0, 0.0, false}},
        {"puppis_a", {"Puppis A", 125.617, -42.983, 3.2, 0.0, 0.0, 0.0, false}},
        {"psr_j0437", {"PSR J0437-4715", 69.316, -47.252, 2.8, 2.64, 0.0, 0.0, false}},
        {"psr_j1644", {"PSR J1644-4559", 251.205, -45.987, 2.5, 478.0, 0.0, 0.0, false}},
        {"psr_j0737", {"PSR J0737-3039", 114.463, -30.661, 2.2, 48.9, 0.0, 0.0, false}},
        {"zenith", {"Zenith Calibration", 24.346, -33.4211146, 3.0, 0.0, 0.0, 0.0, false}},
        {"frb", {"Fast Radio Burst", 150.0, -35.0, 6.0, 300.0, 0.0, 0.0, true}},
        {"rfi_leo", {"RFI LEO Satellite", 0.0, -33.4211146, 15.0, 0.0, 5.0e-5, 2.5e-5, false}}
    };

    std::string key = scenario_name;
    auto remove_substr = [&](const std::string& sub) {
        size_t pos = key.find(sub);
        if (pos != std::string::npos) key.erase(pos, sub.length());
    };
    remove_substr("_no_noise");
    remove_substr("_with_noise");
    remove_substr("_saturated");
    remove_substr("saturated_");

    for (const auto& entry : catalog) {
        if (key == entry.first) {
            return &entry.second;
        }
    }
    // Default fallback to zenith
    return &catalog[10].second;
}

inline uint8_t pack_int4x2(int8_t real, int8_t imag) {
    uint8_t r_nibble = static_cast<uint8_t>(real & 0x0F);
    uint8_t i_nibble = static_cast<uint8_t>((imag & 0x0F) << 4);
    return r_nibble | i_nibble;
}
} // namespace

chartsFEngineSim::chartsFEngineSim(Config& config, const std::string& unique_name,
                                   bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&chartsFEngineSim::main_thread, this)) {

    out_buf = get_buffer("out_buf");
    out_buf->register_producer(unique_name);

    _scenario = config.get_default<std::string>(unique_name, "scenario", "vela");
    _use_noise = config.get_default<bool>(unique_name, "use_noise", true);
    if (_scenario.find("_no_noise") != std::string::npos) _use_noise = false;

    _saturate = config.get_default<bool>(unique_name, "saturate", false);
    if (_scenario.find("saturated") != std::string::npos) _saturate = true;

    _saturated_antennas = config.get_default<std::vector<int>>(
        unique_name, "saturated_antennas", std::vector<int>{7, 23, 42, 55});
    _saturation_factor = config.get_default<float>(unique_name, "saturation_factor", 12.0f);

    _num_elements = config.get_default<int>(unique_name, "num_elements", 64);
    _num_polarizations = config.get_default<int>(unique_name, "num_polarizations", 1);
    if (_num_polarizations < 1 || _num_polarizations > 2) {
        throw std::runtime_error("chartsFEngineSim: num_polarizations must be 1 or 2");
    }
    if (_num_elements % _num_polarizations != 0) {
        throw std::runtime_error("chartsFEngineSim: num_elements must be a multiple of num_polarizations");
    }

    _pol_layout = config.get_default<std::string>(unique_name, "pol_layout", "interleaved");
    int num_dishes = _num_elements / _num_polarizations;

    _grid_x = config.get_default<int>(unique_name, "grid_x", 0);
    if (_grid_x <= 0) {
        if (num_dishes == 64) {
            _grid_x = 8;
        } else if (num_dishes == 128) {
            _grid_x = 16;
        } else if (num_dishes == 256) {
            _grid_x = 16;
        } else {
            _grid_x = static_cast<int>(std::round(std::sqrt(num_dishes)));
            if (_grid_x <= 0) _grid_x = 1;
        }
    }
    _grid_y = (num_dishes + _grid_x - 1) / _grid_x;

    _num_local_freq = config.get_default<int>(unique_name, "num_local_freq", 336);
    _samples_per_data_set = config.get_default<int>(unique_name, "samples_per_data_set", 1536);
    _num_frames = config.get_default<int>(unique_name, "num_frames", 1);

    _freq_start_mhz = config.get_default<double>(unique_name, "freq_start_mhz", 300.0);
    _delta_freq_mhz = config.get_default<double>(unique_name, "delta_freq_mhz", 0.3);
    _delta_time_us = config.get_default<double>(unique_name, "delta_time_us", 10.0 / 3.0);
    _spacing_m = config.get_default<double>(unique_name, "spacing_m", 0.6);
    _site_lat_deg = config.get_default<double>(unique_name, "site_lat_deg", -33.4211146);
    _seed = config.get_default<int>(unique_name, "seed", 42);

    // Tunable noise, amplitude, and pulse parameters
    _noise_sigma_min = config.get_default<float>(unique_name, "noise_sigma_min", 0.40f);
    _noise_sigma_max = config.get_default<float>(unique_name, "noise_sigma_max", 0.70f);
    _signal_amplitude = config.get_default<float>(unique_name, "signal_amplitude", -1.0f);
    _pulse_dm = config.get_default<double>(unique_name, "pulse_dm", -1.0);
    _pulse_width_s = config.get_default<double>(unique_name, "pulse_width_s", 0.001);
    _pulse_center_fraction = config.get_default<double>(unique_name, "pulse_center_fraction", 0.4);
    _pol_x_amp = config.get_default<float>(unique_name, "pol_x_amp", 1.0f);
    _pol_y_amp = config.get_default<float>(unique_name, "pol_y_amp", 1.0f);

    INFO("chartsFEngineSim: Initialized '{:s}' (elements={:d}, dishes={:d}, pols={:d}, layout={:s}, grid={:d}x{:d}, noise={:d}, sat={:d})",
         _scenario, _num_elements, num_dishes, _num_polarizations, _pol_layout, _grid_x, _grid_y,
         _use_noise ? 1 : 0, _saturate ? 1 : 0);
}

chartsFEngineSim::~chartsFEngineSim() {}

void chartsFEngineSim::main_thread() {
    int num_dishes = _num_elements / _num_polarizations;

    // Physical antenna dish positions on the ground (E-W x N-S)
    std::vector<double> dish_pos_x(num_dishes), dish_pos_y(num_dishes);
    for (int d = 0; d < num_dishes; ++d) {
        int col = d % _grid_x;
        int row = d / _grid_x;
        dish_pos_x[d] = col * _spacing_m;
        dish_pos_y[d] = row * _spacing_m;
    }

    // Element to dish and polarization index mapping
    std::vector<int> elem_dish(_num_elements);
    std::vector<int> elem_pol(_num_elements);
    for (int a = 0; a < _num_elements; ++a) {
        if (_num_polarizations == 1) {
            elem_dish[a] = a;
            elem_pol[a] = 0;
        } else if (_pol_layout == "block") {
            elem_pol[a] = a / num_dishes;
            elem_dish[a] = a % num_dishes;
        } else { // "interleaved"
            elem_dish[a] = a / _num_polarizations;
            elem_pol[a] = a % _num_polarizations;
        }
    }

    // Initialize per-element independent receiver thermal noise variance
    std::mt19937 init_rng(_seed);
    std::uniform_real_distribution<float> noise_sigma_dist(_noise_sigma_min, _noise_sigma_max);
    std::vector<float> elem_noise_sigmas(_num_elements, 0.0f);
    if (_use_noise) {
        for (int a = 0; a < _num_elements; ++a) {
            elem_noise_sigmas[a] = noise_sigma_dist(init_rng);
        }
    }

    const TargetInfo* target = lookup_target(_scenario);
    double effective_dm = (_pulse_dm >= 0.0) ? _pulse_dm : target->dm;
    bool is_dispersed = target->is_frb || (_pulse_dm > 0.0) || (target->dm > 0.0);
    double base_nominal_amp = (_signal_amplitude > 0.0f) ? static_cast<double>(_signal_amplitude) : target->nominal_amp;

    INFO("chartsFEngineSim: Running target '{:s}' (RA={:.2f} deg, Dec={:.2f} deg, amp={:.2f}, DM={:.2f}, dispersed={:d})",
         target->name, target->ra_deg, target->dec_deg, base_nominal_amp, effective_dm, is_dispersed ? 1 : 0);

    // Transit direction cosines
    double delta_rad = (target->dec_deg - _site_lat_deg) * (PI / 180.0);
    double l0 = 0.0;
    double m0 = std::sin(delta_rad);
    if (std::string(target->name).find("Zenith") != std::string::npos) {
        m0 = 0.0;
    }

    // Precalculate frequency array [Hz]
    std::vector<double> freqs_hz(_num_local_freq);
    for (int f = 0; f < _num_local_freq; ++f) {
        freqs_hz[f] = (_freq_start_mhz + f * _delta_freq_mhz) * 1e6;
    }

    const double dt_s = _delta_time_us * 1e-6;
    const double c_inv = 1.0 / C_LIGHT;
    const double two_pi = 2.0 * PI;

    int frame_id = 0;

    for (int frame_idx = 0; frame_idx < _num_frames && !stop_thread; ++frame_idx) {
        uint8_t* frame_ptr = (uint8_t*)out_buf->wait_for_empty_frame(unique_name, frame_id);
        if (frame_ptr == nullptr) break;

        // Initialize metadata object for downstream stages (e.g. rawFileWrite, cudaInputData)
        out_buf->allocate_new_metadata_object(frame_id);
        auto meta = get_chord_metadata(out_buf, frame_id);
        if (meta) {
            meta->set_fpga_seq_num(static_cast<uint64_t>(frame_idx));
            std::vector<int> coarse_freq(_num_local_freq);
            for (int f = 0; f < _num_local_freq; ++f) {
                coarse_freq[f] = f;
            }
            meta->set_coarse_freq(coarse_freq);
        }

        int64_t global_t_start = static_cast<int64_t>(frame_idx) * _samples_per_data_set;

        #pragma omp parallel
        {
            #if defined(_OPENMP)
            int tid = omp_get_thread_num();
            #else
            int tid = 0;
            #endif

            // Thread-private deterministic PRNG stream (thread-safe, zero lock contention)
            std::mt19937 thread_rng(static_cast<uint32_t>(_seed) + static_cast<uint32_t>(tid) * 10007U +
                                    static_cast<uint32_t>(frame_idx) * 999983U);
            std::normal_distribution<float> thread_norm_dist(0.0f, 1.0f);

            #pragma omp for collapse(2) schedule(static)
            for (int t = 0; t < _samples_per_data_set; ++t) {
                for (int f = 0; f < _num_local_freq; ++f) {
                    int64_t t_global = global_t_start + t;
                    double freq_hz = freqs_hz[f];

                    double l_t = l0 + target->drift_dl * t;
                    double m_t = m0 + target->drift_dm * t;

                    double base_phase = two_pi * (t_global * 0.005) * (freq_hz * 1e-8);

                    double pulse_envelope = 1.0;
                    if (is_dispersed && effective_dm > 0.0) {
                        double f_ref = freqs_hz.back();
                        double dm_delay_s = (K_DM * 1e-6) * effective_dm *
                            (1.0 / std::pow(freq_hz / 1e9, 2.0) - 1.0 / std::pow(f_ref / 1e9, 2.0));
                        double t_physical_s = t * dt_s;
                        double pulse_center_s = (_samples_per_data_set * dt_s) * _pulse_center_fraction;
                        double t_diff = t_physical_s - (pulse_center_s + dm_delay_s);
                        pulse_envelope = std::exp(-0.5 * std::pow(t_diff / _pulse_width_s, 2.0));
                    }

                    for (int a = 0; a < _num_elements; ++a) {
                        int dish = elem_dish[a];
                        int pol = elem_pol[a];

                        double delay_s = (l_t * dish_pos_x[dish] + m_t * dish_pos_y[dish]) * c_inv;
                        double total_phase = base_phase - (two_pi * freq_hz * delay_s);

                        float v_r = 0.0f;
                        float v_i = 0.0f;

                        if (_use_noise) {
                            float n_r = thread_norm_dist(thread_rng) * elem_noise_sigmas[a];
                            float n_i = thread_norm_dist(thread_rng) * elem_noise_sigmas[a];
                            v_r += n_r;
                            v_i += n_i;
                        }

                        float pol_gain = (pol == 0) ? _pol_x_amp : _pol_y_amp;
                        float sig_amp = static_cast<float>(base_nominal_amp * pulse_envelope) * pol_gain;
                        v_r += sig_amp * std::cos(total_phase);
                        v_i += sig_amp * std::sin(total_phase);

                        if (_saturate) {
                            for (int bad_idx : _saturated_antennas) {
                                if (a == bad_idx || dish == bad_idx) {
                                    v_r *= _saturation_factor;
                                    v_i *= _saturation_factor;
                                    break;
                                }
                            }
                        }

                        // Quantize to signed 4-bit [-7, +7]
                        int r_q = std::clamp(static_cast<int>(std::round(v_r)), -7, 7);
                        int i_q = std::clamp(static_cast<int>(std::round(v_i)), -7, 7);

                        size_t out_offset = (static_cast<size_t>(t) * _num_local_freq + f) * _num_elements + a;
                        frame_ptr[out_offset] = pack_int4x2(static_cast<int8_t>(r_q), static_cast<int8_t>(i_q));
                    }
                }
            }
        }

        out_buf->mark_frame_full(unique_name, frame_id);
        INFO("chartsFEngineSim: Frame {:d}/{:d} successfully generated into buffer", frame_idx + 1, _num_frames);
        frame_id = (frame_id + 1) % out_buf->num_frames;
    }

    // Keep stage alive until pipeline shutdown
    while (!stop_thread) {
        usleep(100000);
    }
}
