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
 *        for the CHARTS 64-antenna array (8x8 grid, dx=dy=0.6m, Carén Observatory).
 *
 * Simulates geometric wavefront delays, celestial radio targets (Sgr A*, Cen A,
 * Crab, Vela, Sun, southern pulsars, FRBs, LEO RFI), independent per-antenna
 * thermal noise, and ADC saturation, outputting directly into Kotekan's ring buffers
 * as 4-bit packed complex integers (int4x2_t).
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

    int _num_elements;
    int _num_local_freq;
    int _samples_per_data_set;
    int _num_frames;

    double _freq_start_mhz;
    double _delta_freq_mhz;
    double _delta_time_us;
    double _spacing_m;
    double _site_lat_deg;
    int _seed;
    double _frb_dm;
    double _frb_arrival_time_s;
    double _frb_pulse_width_s;
};

#endif // CHARTS_FENGINE_SIM_HPP
