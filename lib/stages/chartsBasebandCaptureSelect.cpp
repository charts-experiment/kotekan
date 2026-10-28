#include "chartsBasebandCaptureSelect.hpp"

#include "BasebandMetadata.hpp"
#include "StageFactory.hpp"
#include "chartsMetadata.hpp"

#include <algorithm>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(chartsBasebandCaptureSelect);

chartsBasebandCaptureSelect::chartsBasebandCaptureSelect(
    Config& config, const std::string& unique_name, bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container,
          std::bind(&chartsBasebandCaptureSelect::main_thread, this)),
    in_buf(get_buffer("in_buf")),
    out_buf(get_buffer("out_buf")),
    selected_antennas(config.get<std::vector<int>>(unique_name, "selected_antennas")),
    num_elements(config.get<int>(unique_name, "num_elements")),
    samples_per_data_set(config.get<int64_t>(unique_name, "samples_per_data_set")),
    capture_samples(config.get<int64_t>(unique_name, "capture_samples")),
    event_id(config.get<uint64_t>(unique_name, "event_id")),
    freq_id(config.get<uint64_t>(unique_name, "freq_id")),
    bytes_per_sample(0) {
    if (!in_buf || !out_buf || in_buf == out_buf)
        throw std::invalid_argument("chartsBasebandCaptureSelect requires distinct buffers");
    if (!metadata_is_charts(in_buf, 0) || !out_buf->metadata_pool
        || out_buf->metadata_pool->type_name != "BasebandMetadata")
        throw std::invalid_argument("chartsBasebandCaptureSelect requires CHARTS input and baseband output metadata");
    if (num_elements <= 0 || samples_per_data_set <= 0 || capture_samples <= 0
        || selected_antennas.empty())
        throw std::invalid_argument("chartsBasebandCaptureSelect requires positive dimensions and capture length");
    std::vector<bool> seen(num_elements, false);
    source_indices.reserve(selected_antennas.size());
    for (int antenna : selected_antennas) {
        if (antenna < 0 || antenna >= num_elements || seen[antenna])
            throw std::invalid_argument("chartsBasebandCaptureSelect antenna index invalid or repeated");
        seen[antenna] = true;
        source_indices.push_back(num_elements - 1 - antenna);
    }
    if (in_buf->frame_size == 0 || in_buf->frame_size % samples_per_data_set != 0)
        throw std::invalid_argument("chartsBasebandCaptureSelect input frame has invalid size");
    bytes_per_sample = in_buf->frame_size / samples_per_data_set;
    if (bytes_per_sample % num_elements != 0
        || out_buf->frame_size != static_cast<uint64_t>(
            samples_per_data_set * (bytes_per_sample / num_elements) * selected_antennas.size()))
        throw std::invalid_argument("chartsBasebandCaptureSelect output frame has invalid size");

    in_buf->register_consumer(unique_name);
    out_buf->register_producer(unique_name);
}

void chartsBasebandCaptureSelect::main_thread() {
    int in_frame_id = 0;
    int out_frame_id = 0;
    int64_t captured_samples = 0;
    int64_t event_start = -1;
    int64_t time0_fpga = 0;
    const int64_t frequencies_per_sample = bytes_per_sample / num_elements;
    const int64_t selected_count = selected_antennas.size();

    while (!stop_thread && captured_samples < capture_samples) {
        uint8_t* input = in_buf->wait_for_full_frame(unique_name, in_frame_id);
        if (!input)
            break;
        auto input_metadata = get_charts_metadata(in_buf, in_frame_id);
        if (!input_metadata || !input_metadata->has_fpga_seq_num()
            || !input_metadata->has_time0_fpga())
            throw std::runtime_error("chartsBasebandCaptureSelect input is missing FPGA metadata");

        if (event_start < 0) {
            event_start = input_metadata->get_fpga_seq_num();
            time0_fpga = input_metadata->get_time0_fpga();
        }
        const int64_t valid_samples = std::min(samples_per_data_set,
                                               capture_samples - captured_samples);
        uint8_t* output = out_buf->wait_for_empty_frame(unique_name, out_frame_id);
        if (!output)
            break;

        out_buf->allocate_new_metadata_object(out_frame_id);
        auto output_metadata = std::static_pointer_cast<BasebandMetadata>(
            out_buf->get_metadata(out_frame_id));
        output_metadata->event_id = event_id;
        output_metadata->freq_id = freq_id;
        output_metadata->event_start_fpga = event_start;
        output_metadata->event_end_fpga = event_start + capture_samples;
        output_metadata->time0_fpga = time0_fpga;
        output_metadata->time0_ctime = 0;
        output_metadata->time0_ctime_offset = 0;
        output_metadata->first_packet_recv_time = 0;
        output_metadata->frame_fpga_seq = input_metadata->get_fpga_seq_num();
        output_metadata->valid_to = valid_samples;
        output_metadata->fpga0_ns = 0;
        output_metadata->num_elements = selected_count;
        output_metadata->reserved = -1;

        for (int64_t sample = 0; sample < valid_samples; ++sample) {
            const uint8_t* source = input + sample * bytes_per_sample;
            uint8_t* destination = output + sample * frequencies_per_sample * selected_count;
            for (int64_t freq = 0; freq < frequencies_per_sample; ++freq) {
                const uint8_t* source_frequency = source + freq * num_elements;
                uint8_t* destination_frequency = destination + freq * selected_count;
                for (int64_t element = 0; element < selected_count; ++element)
                    destination_frequency[element] = source_frequency[source_indices[element]];
            }
        }
        if (valid_samples < samples_per_data_set)
            std::memset(output + valid_samples * frequencies_per_sample * selected_count, 0,
                        (samples_per_data_set - valid_samples) * frequencies_per_sample
                            * selected_count);

        out_buf->mark_frame_full(unique_name, out_frame_id);
        in_buf->mark_frame_empty(unique_name, in_frame_id);
        out_frame_id = (out_frame_id + 1) % out_buf->num_frames;
        in_frame_id = (in_frame_id + 1) % in_buf->num_frames;
        captured_samples += valid_samples;
    }

    // The correlator must keep receiving frames after this finite capture ends.
    if (!stop_thread && captured_samples == capture_samples)
        in_buf->unregister_consumer(unique_name, true);
}
