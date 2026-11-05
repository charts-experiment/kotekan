#include "correlationAVX.hpp"

#include "StageFactory.hpp"
#include "kotekanLogging.hpp"

#include <cstring>
#include <thread>

REGISTER_KOTEKAN_STAGE(correlationAVX);

correlationAVX::correlationAVX(Config& config, const std::string& unique_name,
                               bufferContainer& buffer_container) :

    Stage(config, unique_name, buffer_container, std::bind(&correlationAVX::main_thread, this)) {

    buf_in = get_buffer("corr_in");
    buf_out = get_buffer("corr_out");

    buf_in->register_consumer(unique_name);
    buf_out->register_producer(unique_name);

    _num_samples = config.get<uint32_t>(unique_name, "num_samples");
    _integration = config.get<uint32_t>(unique_name, "integration");
}

correlationAVX::~correlationAVX() {}

void correlationAVX::main_thread() {
    uint32_t frame_in = 0;
    uint32_t frame_out = 0;

    while (!stop_thread) {

        void* in_frame = buf_in->wait_for_full_frame(unique_name, frame_in);
        if (!in_frame)
            break;

        void* out_frame = buf_out->wait_for_empty_frame(unique_name, frame_out);
        if (!out_frame)
            break;

        int8_t* raw = (int8_t*)in_frame;
        float* out = (float*)out_frame;

        // Supongamos layout: [X samples][Y samples]
        int8_t* X = raw;
        int8_t* Y = raw + 2 * _num_samples; // cada muestra usa 2 bytes (re, im)

        compute_correlations_avx(X, Y, _num_samples, out);

        buf_in->mark_frame_empty(unique_name, frame_in);
        buf_out->mark_frame_full(unique_name, frame_out);

        frame_in = (frame_in + 1) % buf_in->num_frames;
        frame_out = (frame_out + 1) % buf_out->num_frames;
    }
}
