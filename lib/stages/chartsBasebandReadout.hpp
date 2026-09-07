#ifndef CHARTS_BASEBAND_READOUT_HPP
#define CHARTS_BASEBAND_READOUT_HPP

#include "Config.hpp"
#include "Stage.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"
#include "visUtil.hpp"
#include "chartsMetadata.hpp"

#include <cstdint>
#include <mutex>
#include <vector>


// For now hardcoding the trigger request as a struct, but this should be replaced
struct ChartsTriggerRequest {
    uint64_t event_id;
    int64_t start_fpga;   // spex index
    int64_t length_fpga;  // number of specs
};

class chartsBasebandReadout : public kotekan::Stage {
public:
    chartsBasebandReadout(kotekan::Config& config, const std::string& unique_name,
                          kotekan::bufferContainer& buffer_container);
    virtual ~chartsBasebandReadout() = default;

    void main_thread() override;

private:
    int _num_frames_buffer;
    int _samples_per_data_set;   // specs per frame
    int _num_elements;
    int64_t _max_dump_samples;
    uint64_t _freq_id;
    size_t _bytes_per_spec;

    // Buffers
    Buffer* in_buf;
    Buffer* out_buf;
    frameID out_frame_id;


    int next_frame;
    int oldest_frame;
    std::vector<std::mutex> frame_locks;
    std::mutex manager_lock;

    // Test trigger request for development without the API manager
    bool test_trigger_sent;
    ChartsTriggerRequest test_trigger;


    int add_replace_frame(int frame_id);
    void lock_range(int start_frame, int end_frame);
    void unlock_range(int start_frame, int end_frame);

    bool wait_for_data(ChartsTriggerRequest& trigger,
                       int& dump_start_frame,
                       int& dump_end_frame);

    bool extract_data(const ChartsTriggerRequest& trigger,
                      int dump_start_frame,
                      int dump_end_frame);
};

#endif