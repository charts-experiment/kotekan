#include "chartsBasebandAntennaSelect.hpp"

#include "BasebandMetadata.hpp"
#include "StageFactory.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(chartsBasebandAntennaSelect);

chartsBasebandAntennaSelect::chartsBasebandAntennaSelect(
    Config& config, const std::string& unique_name, bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container,
          std::bind(&chartsBasebandAntennaSelect::main_thread, this)),
    in_buf(get_buffer("in_buf")),
    out_buf(get_buffer("out_buf")),
    selected_antennas(config.get<std::vector<int>>(unique_name, "selected_antennas")),
    input_elements(config.get<int>(unique_name, "num_elements")),
    samples_per_frame(0) {
    if (!in_buf || !out_buf || in_buf == out_buf)
        throw std::invalid_argument("chartsBasebandAntennaSelect requires distinct input and output buffers");
    if (input_elements <= 0 || selected_antennas.empty())
        throw std::invalid_argument("chartsBasebandAntennaSelect requires elements to select");
    if (!in_buf->metadata_pool || !out_buf->metadata_pool
        || in_buf->metadata_pool->type_name != "BasebandMetadata"
        || out_buf->metadata_pool->type_name != "BasebandMetadata")
        throw std::invalid_argument("chartsBasebandAntennaSelect requires BasebandMetadata buffers");

    std::vector<bool> seen(input_elements, false);
    for (int antenna : selected_antennas) {
        if (antenna < 0 || antenna >= input_elements
            || seen[antenna])
            throw std::invalid_argument("chartsBasebandAntennaSelect antenna index invalid or repeated");
        seen[antenna] = true;
    }

    if (in_buf->frame_size == 0 || in_buf->frame_size % input_elements != 0)
        throw std::invalid_argument("chartsBasebandAntennaSelect input frame has invalid size");
    samples_per_frame = in_buf->frame_size / input_elements;
    if (out_buf->frame_size != samples_per_frame * selected_antennas.size())
        throw std::invalid_argument("chartsBasebandAntennaSelect output frame has invalid size");

    in_buf->register_consumer(unique_name);
    out_buf->register_producer(unique_name);
}

void chartsBasebandAntennaSelect::main_thread() {
    int in_frame_id = 0;
    int out_frame_id = 0;
    const std::size_t output_elements = selected_antennas.size();

    while (!stop_thread) {
        uint8_t* input = in_buf->wait_for_full_frame(unique_name, in_frame_id);
        if (!input)
            break;

        auto input_metadata = std::dynamic_pointer_cast<BasebandMetadata>(
            in_buf->get_metadata(in_frame_id));
        if (!input_metadata || input_metadata->num_elements != static_cast<int32_t>(input_elements))
            throw std::runtime_error("chartsBasebandAntennaSelect input metadata element count mismatch");

        uint8_t* output = out_buf->wait_for_empty_frame(unique_name, out_frame_id);
        if (!output)
            break;

        // Keep input metadata intact for any other consumers of the merged buffer.
        out_buf->allocate_new_metadata_object(out_frame_id);
        in_buf->copy_metadata(in_frame_id, out_buf, out_frame_id);
        auto output_metadata = std::static_pointer_cast<BasebandMetadata>(
            out_buf->get_metadata(out_frame_id));
        output_metadata->num_elements = static_cast<int32_t>(output_elements);

        for (std::size_t sample = 0; sample < samples_per_frame; ++sample) {
            const uint8_t* source = input + sample * input_elements;
            uint8_t* destination = output + sample * output_elements;
            for (std::size_t element = 0; element < output_elements; ++element)
                destination[element] = source[input_elements - 1 - selected_antennas[element]];
        }

        out_buf->mark_frame_full(unique_name, out_frame_id);
        in_buf->mark_frame_empty(unique_name, in_frame_id);
        out_frame_id = (out_frame_id + 1) % out_buf->num_frames;
        in_frame_id = (in_frame_id + 1) % in_buf->num_frames;
    }
}
