#ifndef CHARTS_ACCUMULATE_HPP
#define CHARTS_ACCUMULATE_HPP

#include "Config.hpp"
#include "Stage.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

class chartsMetadata;

/**
 * Accumulate int32 CHARTS frames while keeping the output format unchanged.
 *
 * Input frames are accumulated in int64_t and converted back to int32_t only
 * when the output frame is complete.
 */
class chartsAccumulate : public kotekan::Stage {
public:
    chartsAccumulate(kotekan::Config& config, const std::string& unique_name,
                     kotekan::bufferContainer& buffer_container);
    ~chartsAccumulate();
    void main_thread() override;

private:
    void validate_metadata(const chartsMetadata& reference,
                           const chartsMetadata& current) const;

    Buffer* in_buf;
    Buffer* out_buf;

    int32_t _num_frames_to_accumulate;
    std::size_t _num_values;
};

#endif
