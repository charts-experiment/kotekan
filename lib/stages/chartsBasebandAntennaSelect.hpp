#ifndef CHARTS_BASEBAND_ANTENNA_SELECT_HPP
#define CHARTS_BASEBAND_ANTENNA_SELECT_HPP

#include "Stage.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"

#include <cstddef>
#include <vector>

// Packs selected logical CHARTS antennas from reversed-order baseband frames.
class chartsBasebandAntennaSelect : public kotekan::Stage {
public:
    chartsBasebandAntennaSelect(kotekan::Config& config, const std::string& unique_name,
                                kotekan::bufferContainer& buffer_container);

    void main_thread() override;

private:
    Buffer* in_buf;
    Buffer* out_buf;
    std::vector<int> selected_antennas;
    int input_elements;
    std::size_t samples_per_frame;
};

#endif
