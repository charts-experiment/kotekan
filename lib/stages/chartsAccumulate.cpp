#include "chartsAccumulate.hpp"

#include "StageFactory.hpp"
#include "chartsMetadata.hpp"

#include <algorithm>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

namespace {

void add_checked(int64_t& accumulator, int32_t value) {
    constexpr int64_t max_value = std::numeric_limits<int64_t>::max();
    constexpr int64_t min_value = std::numeric_limits<int64_t>::min();

    if ((value > 0 && accumulator > max_value - value)
        || (value < 0 && accumulator < min_value - value)) {
        throw std::overflow_error("chartsAccumulate int64 accumulation overflow");
    }

    accumulator += value;
}

} // namespace

REGISTER_KOTEKAN_STAGE(chartsAccumulate);

chartsAccumulate::chartsAccumulate(Config& config, const std::string& unique_name,
                                   bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container,
          std::bind(&chartsAccumulate::main_thread, this)),
    in_buf(get_buffer("in_buf")),
    out_buf(get_buffer("out_buf")),
    _num_frames_to_accumulate(
        config.get<int32_t>(unique_name, "num_frames_to_accumulate")),
    _num_values(0) {

    if (!in_buf || !out_buf)
        throw std::runtime_error("chartsAccumulate requires in_buf and out_buf");
    if (in_buf == out_buf)
        throw std::runtime_error("chartsAccumulate requires distinct input and output buffers");
    if (_num_frames_to_accumulate <= 0)
        throw std::invalid_argument("chartsAccumulate num_frames_to_accumulate must be positive");
    if (!in_buf->metadata_pool || !out_buf->metadata_pool
        || in_buf->metadata_pool->type_name != "chartsMetadata"
        || out_buf->metadata_pool->type_name != "chartsMetadata") {
        throw std::invalid_argument(
            "chartsAccumulate requires chartsMetadata input and output buffers");
    }
    if (in_buf->frame_size != out_buf->frame_size)
        throw std::invalid_argument("chartsAccumulate input and output frame sizes must match");
    if (in_buf->frame_size == 0 || in_buf->frame_size % sizeof(int32_t) != 0)
        throw std::invalid_argument(
            "chartsAccumulate frame size must be a non-zero multiple of int32_t");

    _num_values = in_buf->frame_size / sizeof(int32_t);

    in_buf->register_consumer(unique_name);
    out_buf->register_producer(unique_name);
}

chartsAccumulate::~chartsAccumulate() {}

void chartsAccumulate::validate_metadata(const chartsMetadata& reference,
                                         const chartsMetadata& current) const {

    if (std::strncmp(reference.name, current.name, CHARTS_META_MAX_DIMNAME) != 0
        || reference.type != current.type || reference.dims != current.dims
        || reference.offset != current.offset)
        throw std::invalid_argument("chartsAccumulate received incompatible array metadata");

    for (int d = 0; d < reference.dims; ++d) {
        if (reference.dim[d] != current.dim[d]
            || std::strncmp(reference.dim_name[d], current.dim_name[d], CHARTS_META_MAX_DIMNAME) != 0
            || reference.stride[d] != current.stride[d])
            throw std::invalid_argument("chartsAccumulate received incompatible array metadata");
    }

    if (reference.has_coarse_freq() != current.has_coarse_freq())
        throw std::invalid_argument("chartsAccumulate received incompatible frequency metadata");
    if (reference.has_coarse_freq() && reference.get_coarse_freq() != current.get_coarse_freq())
        throw std::invalid_argument("chartsAccumulate received incompatible frequency metadata");
}

void chartsAccumulate::main_thread() {
    int in_frame_id = 0;
    int out_frame_id = 0;
    int frames_in_group = 0;
    int64_t lost_samples = 0;
    std::vector<int64_t> accumulation(_num_values, 0);

    while (!stop_thread) {
        uint8_t* in_frame = in_buf->wait_for_full_frame(unique_name, in_frame_id);
        if (in_frame == nullptr)
            break;

        auto input_metadata = get_charts_metadata(in_buf, in_frame_id);
        if (!input_metadata)
            throw std::runtime_error("chartsAccumulate received a frame without chartsMetadata");

        uint8_t* out_frame = nullptr;
        std::shared_ptr<chartsMetadata> output_metadata;
        if (frames_in_group == 0) {
            out_frame = out_buf->wait_for_empty_frame(unique_name, out_frame_id);
            if (out_frame == nullptr)
                break;

            out_buf->allocate_new_metadata_object(out_frame_id);
            in_buf->copy_metadata(in_frame_id, out_buf, out_frame_id);
            output_metadata = get_charts_metadata(out_buf, out_frame_id);
            if (!output_metadata)
                throw std::runtime_error(
                    "chartsAccumulate could not create output chartsMetadata");

            validate_metadata(*output_metadata, *input_metadata);
            std::fill(accumulation.begin(), accumulation.end(), 0);
            lost_samples = 0;
        } else {
            output_metadata = get_charts_metadata(out_buf, out_frame_id);
            if (!output_metadata)
                throw std::runtime_error(
                    "chartsAccumulate lost output chartsMetadata during accumulation");
            validate_metadata(*output_metadata, *input_metadata);
            out_frame = out_buf->frames[out_frame_id];
        }

        const auto* input = reinterpret_cast<const int32_t*>(in_frame);
        for (std::size_t i = 0; i < _num_values; ++i)
            add_checked(accumulation[i], input[i]);

        if (input_metadata->has_lost_timesamples())
            add_checked(lost_samples, input_metadata->get_lost_timesamples());

        in_buf->mark_frame_empty(unique_name, in_frame_id);
        in_frame_id = (in_frame_id + 1) % in_buf->num_frames;
        ++frames_in_group;

        if (frames_in_group == _num_frames_to_accumulate) {
            auto* output = reinterpret_cast<int32_t*>(out_frame);
            for (std::size_t i = 0; i < _num_values; ++i) {
                if (accumulation[i] < std::numeric_limits<int32_t>::min()
                    || accumulation[i] > std::numeric_limits<int32_t>::max())
                    throw std::overflow_error(
                        "chartsAccumulate result does not fit in int32_t");
                output[i] = static_cast<int32_t>(accumulation[i]);
            }

            if (output_metadata->has_lost_timesamples()) {
                if (lost_samples < std::numeric_limits<int32_t>::min()
                    || lost_samples > std::numeric_limits<int32_t>::max())
                    throw std::overflow_error(
                        "chartsAccumulate lost_timesamples does not fit in int32_t");
                output_metadata->set_lost_timesamples(static_cast<int32_t>(lost_samples));
            }

            out_buf->mark_frame_full(unique_name, out_frame_id);
            out_frame_id = (out_frame_id + 1) % out_buf->num_frames;
            frames_in_group = 0;
        }
    }
}
