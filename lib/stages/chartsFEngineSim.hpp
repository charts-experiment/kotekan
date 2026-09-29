#ifndef CHARTS_FENGINE_SIM_HPP
#define CHARTS_FENGINE_SIM_HPP

#include "Config.hpp"
#include "Stage.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

/**
 * @class chartsFEngineSim
 * @brief Native Kotekan Stage simulating physical complex baseband voltage streams
 *        for the CHARTS telescope array (64, 128, or 256 antenna layouts, 1 or 2 polarizations).
 *
 * Simulates geometric wavefront delays, celestial radio targets (Sgr A*, Cen A,
 * Crab, Vela, Sun, southern pulsars, FRBs, LEO RFI), independent per-antenna
 * thermal noise, polarization configurations, and ADC saturation, outputting directly
 * into Kotekan's ring buffers as 4-bit packed complex integers (int4x2_t).
 */
class chartsFEngineSim : public kotekan::Stage {
public:
    chartsFEngineSim(kotekan::Config& config, const std::string& unique_name,
                     kotekan::bufferContainer& buffer_container);
    virtual ~chartsFEngineSim();
    void main_thread() override;

private:
    Buffer* out_buf;

    std::string _scenario;
    bool _use_noise;
    bool _saturate;
    std::vector<int> _saturated_antennas;
    float _saturation_factor;

    int _num_elements;
    int _num_polarizations;
    std::string _pol_layout;
    int _grid_x;
    int _grid_y;

    int _num_local_freq;
    int _samples_per_data_set;
    int _num_frames;

    double _freq_start_mhz;
    double _delta_freq_mhz;
    double _delta_time_us;
    double _spacing_m;
    double _site_lat_deg;
    int _seed;

    // YAML-configurable noise and pulse parameters
    float _noise_sigma_min;
    float _noise_sigma_max;
    float _signal_amplitude;
    double _pulse_dm;
    double _pulse_width_s;
    double _pulse_center_fraction;
    float _pol_x_amp;
    float _pol_y_amp;
};

#endif // CHARTS_FENGINE_SIM_HPP
