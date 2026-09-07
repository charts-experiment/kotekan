#include "chartsBasebandReadout.hpp"

#include "StageFactory.hpp"
#include "kotekanLogging.hpp"
#include "BasebandMetadata.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <stdexcept>


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(chartsBasebandReadout);

chartsBasebandReadout::chartsBasebandReadout(Config& config, const std::string& unique_name,
                                             bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container,
          std::bind(&chartsBasebandReadout::main_thread, this)),
    _num_frames_buffer(config.get<int>(unique_name, "num_frames_buffer")),
    _samples_per_data_set(config.get<int>(unique_name, "samples_per_data_set")),
    _num_elements(config.get<int>(unique_name, "num_elements")),
    _max_dump_samples(config.get_default<int64_t>(unique_name, "max_dump_samples", 1 << 20)),
    _freq_id(config.get<uint64_t>(unique_name, "freq_id")),

    in_buf(get_buffer("in_buf")),
    out_buf(get_buffer("out_buf")),
    out_frame_id(out_buf),
    next_frame(0),
    oldest_frame(-1),
    frame_locks(_num_frames_buffer),
    test_trigger_sent(false) {

    // Assert that the buffers exist and are the right size, and register as consumer/producer
    if (!in_buf || !out_buf) {
        throw std::runtime_error("chartsBasebandReadout: missing in_buf or out_buf");
    }

    in_buf->register_consumer(unique_name);
    out_buf->register_producer(unique_name);

    if (in_buf->num_frames <= _num_frames_buffer) {
        throw std::runtime_error("chartsBasebandReadout: input buffer too small");
    }

    if (in_buf->frame_size % _samples_per_data_set != 0) {
        throw std::runtime_error(
            "chartsBasebandReadout: in_buf->frame_size must be divisible by samples_per_data_set");
    }

    _bytes_per_spec = in_buf->frame_size / _samples_per_data_set;

    // Trigger hardcoded
    // start_fpga = -1 => use the oldest available spec
    // length_fpga in units of spec
    test_trigger = {1, -1, _max_dump_samples};
}

void chartsBasebandReadout::main_thread() {
    int frame_id = 0;

    // Loop until shutdown signal is received. 
    while (!stop_thread) {
        int in_buf_frame = frame_id % in_buf->num_frames;

        // Wait for the next frame of input data to be full
        if (in_buf->wait_for_full_frame(unique_name, in_buf_frame) == nullptr) {
            break;
        }

        // Add the frame to the manager and get the ID of any frame that was replaced
        int done_frame = add_replace_frame(frame_id);
        if (done_frame >= 0) {
            in_buf->mark_frame_empty(unique_name, done_frame % in_buf->num_frames);
        }

        // Initially for testing, trigger a readout after a few frames have been added
        if (!test_trigger_sent && next_frame >= _max_dump_samples / _samples_per_data_set) {
            int dump_start_frame = 0;
            int dump_end_frame = 0;

            ChartsTriggerRequest trig = test_trigger;

            if (wait_for_data(trig, dump_start_frame, dump_end_frame)) {
                INFO("chartsBasebandReadout: launching test trigger event_id={} start_fpga={} "
                     "length_fpga={} frames=[{}, {})",
                     trig.event_id, trig.start_fpga, trig.length_fpga,
                     dump_start_frame, dump_end_frame);

                if (!extract_data(trig, dump_start_frame, dump_end_frame)) {
                    WARN("chartsBasebandReadout: extract_data failed");
                }

                test_trigger_sent = true;
            } else {
                WARN("chartsBasebandReadout: test trigger could not be satisfied yet");
            }
        }

        frame_id++;
    }
}

// Returns the frame_id of the replaced frame, or -1 if no frame was replaced
int chartsBasebandReadout::add_replace_frame(int frame_id) {
    std::lock_guard<std::mutex> lock(manager_lock);
    int replaced_frame = -1;

    assert(frame_id == next_frame);

    frame_locks[frame_id % _num_frames_buffer].lock();

    bool replace_oldest =
        (frame_id % _num_frames_buffer ==
         (oldest_frame + _num_frames_buffer) % _num_frames_buffer);

    if (replace_oldest) {
        replaced_frame = oldest_frame;
        oldest_frame++;
    }

    frame_locks[frame_id % _num_frames_buffer].unlock();

    next_frame++;
    return replaced_frame;
}

void chartsBasebandReadout::lock_range(int start_frame, int end_frame) {
    for (int frame_index = start_frame; frame_index < end_frame; frame_index++) {
        frame_locks[frame_index % _num_frames_buffer].lock();
    }
}

void chartsBasebandReadout::unlock_range(int start_frame, int end_frame) {
    for (int frame_index = start_frame; frame_index < end_frame; frame_index++) {
        frame_locks[frame_index % _num_frames_buffer].unlock();
    }
}

// Returns true if the trigger could be satisfied, and fills in dump_start_frame and dump_end_frame with the range of frames to dump (end exclusive)
// bool chartsBasebandReadout::wait_for_data(ChartsTriggerRequest& trigger,
//                                           int& dump_start_frame,
//                                           int& dump_end_frame) {
//     if (trigger.length_fpga <= 0) {
//         return false;
//     }

//     if (trigger.length_fpga > _max_dump_samples) {
//         WARN("chartsBasebandReadout: trigger too long");
//         return false;
//     }

//     std::lock_guard<std::mutex> lock(manager_lock);

//     dump_start_frame = (oldest_frame > 0) ? oldest_frame : 0;
//     dump_end_frame = dump_start_frame;

//     int64_t trigger_start_fpga = trigger.start_fpga;
//     int64_t trigger_end_fpga = -1;

//     for (int frame_index = dump_start_frame; frame_index < next_frame; frame_index++) {
//         int in_buf_frame = frame_index % in_buf->num_frames;
//         auto meta = get_charts_metadata(in_buf, in_buf_frame);
//         if (!meta || !meta->has_fpga_seq_num()) {
//             continue;
//         }

//         int64_t frame_fpga_seq = meta->get_fpga_seq_num();

//         if (trigger_start_fpga < 0) {
//             trigger_start_fpga = frame_fpga_seq;
//             trigger_end_fpga = trigger_start_fpga + trigger.length_fpga;
//         }

//         if (trigger_end_fpga <= frame_fpga_seq) {
//             continue;
//         }

//         if (trigger_start_fpga >= frame_fpga_seq + _samples_per_data_set) {
//             dump_start_frame = frame_index + 1;
//             continue;
//         }

//         dump_end_frame = frame_index + 1;
//     }

//     if (trigger_start_fpga < 0 || dump_start_frame >= dump_end_frame) {
//         return false;
//     }

//     trigger.start_fpga = trigger_start_fpga;
//     lock_range(dump_start_frame, dump_end_frame);
//     return true;
// }


bool chartsBasebandReadout::wait_for_data(ChartsTriggerRequest& trigger,
                                          int& dump_start_frame,
                                          int& dump_end_frame) {
    std::lock_guard<std::mutex> lock(manager_lock);

    dump_start_frame = (oldest_frame > 0) ? oldest_frame : 0;
    dump_end_frame = next_frame;

    if (dump_start_frame >= dump_end_frame) {
        return false;
    }

    int first_buf_frame = dump_start_frame % in_buf->num_frames;
    auto first_meta = get_charts_metadata(in_buf, first_buf_frame);
    if (!first_meta || !first_meta->has_fpga_seq_num()) {
        return false;
    }

    trigger.start_fpga = first_meta->get_fpga_seq_num();
    trigger.length_fpga = (dump_end_frame - dump_start_frame) * _samples_per_data_set;

    lock_range(dump_start_frame, dump_end_frame);
    return true;
}


// Returns true if data was successfully extracted and written to the output buffer
bool chartsBasebandReadout::extract_data(const ChartsTriggerRequest& trigger,
                                         int dump_start_frame,
                                         int dump_end_frame) {
    if (dump_start_frame >= dump_end_frame) {
        return false;
    }

    int in_buf_frame = dump_start_frame % in_buf->num_frames;
    auto first_meta = get_charts_metadata(in_buf, in_buf_frame);
    if (!first_meta) {
        unlock_range(dump_start_frame, dump_end_frame);
        return false;
    }

    const int64_t time0_fpga = first_meta->get_time0_fpga();

    const int64_t data_start_fpga =
        std::max(trigger.start_fpga, first_meta->get_fpga_seq_num());
    const int64_t data_end_fpga = trigger.start_fpga + trigger.length_fpga;

    const size_t out_frame_specs = out_buf->frame_size / _bytes_per_spec;

    uint8_t* out_frame = nullptr;
    BasebandMetadata* out_meta = nullptr;

    int64_t out_start = 0;
    int64_t out_remaining = 0;
    bool stop_extract = false;

    for (int frame_index = dump_start_frame; !stop_thread && frame_index < dump_end_frame;
         frame_index++) {

        if (stop_extract) {
            frame_locks[frame_index % _num_frames_buffer].unlock();
            continue;
        }

        in_buf_frame = frame_index % in_buf->num_frames;
        auto meta = get_charts_metadata(in_buf, in_buf_frame);
        if (!meta) {
            frame_locks[frame_index % _num_frames_buffer].unlock();
            continue;
        }

        uint8_t* in_frame = in_buf->frames[in_buf_frame];
        int64_t frame_fpga_seq = meta->get_fpga_seq_num();

        int64_t in_start = std::max<int64_t>(data_start_fpga - frame_fpga_seq, 0); // maximum of 0 and the offset to the start of the data in this frame
        int64_t in_end = std::min<int64_t>(data_end_fpga - frame_fpga_seq, _samples_per_data_set);

        while (in_start < in_end) {
            if (out_remaining == 0) {
                if (!out_buf->is_frame_empty(out_frame_id)) {
                    WARN("chartsBasebandReadout: output buffer full");
                    stop_extract = true;
                    break;
                }

                out_frame = out_buf->wait_for_empty_frame(unique_name, out_frame_id);
                if (out_frame == nullptr) {
                    stop_extract = true;
                    break;
                }

                out_start = 0;
                out_remaining = out_frame_specs;

                out_buf->allocate_new_metadata_object(out_frame_id);
                out_meta = (BasebandMetadata*)(out_buf->get_metadata(out_frame_id).get());

                if (!out_meta) {
                    WARN("chartsBasebandReadout: output metadata is not chartsMetadata");
                    stop_extract = true;
                    break;
                }

                out_meta->event_id = trigger.event_id;
                out_meta->freq_id = _freq_id;
                out_meta->time0_fpga = time0_fpga;
                out_meta->event_start_fpga = trigger.start_fpga;
                out_meta->event_end_fpga = trigger.start_fpga + trigger.length_fpga;
                out_meta->time0_ctime = 0;
                out_meta->time0_ctime_offset = 0;
                out_meta->first_packet_recv_time = 0;
                out_meta->valid_to = 0;
                out_meta->fpga0_ns = 0;
                out_meta->num_elements = _num_elements;
                out_meta->reserved = -1;
                out_meta->frame_fpga_seq = frame_fpga_seq + in_start;
            }

            const int64_t copy_len = std::min<int64_t>(in_end - in_start, out_remaining);

            std::memcpy(out_frame + out_start * _bytes_per_spec,
                        in_frame + in_start * _bytes_per_spec,
                        copy_len * _bytes_per_spec);

            in_start += copy_len;
            out_start += copy_len;
            out_meta->valid_to = out_start;
            out_remaining -= copy_len;

            if (out_remaining == 0) {
                out_buf->mark_frame_full(unique_name, out_frame_id++);
            }
        }

        frame_locks[frame_index % _num_frames_buffer].unlock();
    }

    if (out_remaining > 0 && out_frame != nullptr) {
        std::memset(out_frame + out_start * _bytes_per_spec, 0,
                    out_remaining * _bytes_per_spec);
        out_buf->mark_frame_full(unique_name, out_frame_id++);
    }

    unlock_range(dump_start_frame, dump_end_frame);
    return !stop_thread;
}