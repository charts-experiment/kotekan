#ifndef CHARTS_BASEBAND_CAPTURE_SELECT_HPP
#define CHARTS_BASEBAND_CAPTURE_SELECT_HPP

#include "Stage.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"

#include <cstdint>
#include <vector>

/**
 * Stream selected logical CHARTS antennas from raw reversed-order input.
 *
 * @conf selected_antennas Logical CHARTS antenna IDs in output row order.
 *                         Logical antenna a reads raw element num_elements - 1 - a.
 *                         For 64 elements, selecting 0 reads raw element 63.
 */
class chartsBasebandCaptureSelect : public kotekan::Stage {
public:
    chartsBasebandCaptureSelect(kotekan::Config& config, const std::string& unique_name,
                                kotekan::bufferContainer& buffer_container);

    void main_thread() override;

private:
    Buffer* in_buf;
    Buffer* out_buf;
    std::vector<int> selected_antennas;
    std::vector<int> source_indices;
    int num_elements;
    int64_t samples_per_data_set;
    int64_t capture_samples;
    uint64_t event_id;
    uint64_t freq_id;
    int64_t bytes_per_sample;
};

#endif
