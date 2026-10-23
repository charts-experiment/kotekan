#include "chartsMVDRBeams.hpp"

#include "StageFactory.hpp"
#include "kotekanLogging.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(chartsMVDRBeams);

chartsMVDRBeams::chartsMVDRBeams(Config& config, const std::string& unique_name,
                                 bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&chartsMVDRBeams::main_thread, this)),
    in_buf(get_buffer("in_buf")), out_buf(get_buffer("out_buf")), weights_buf(nullptr) {

    if (!in_buf || !out_buf || in_buf == out_buf)
        throw std::invalid_argument("chartsMVDRBeams requires distinct in_buf and out_buf");
    if (config.exists(unique_name, "weights_buf"))
        weights_buf = get_buffer("weights_buf");

    charts_mvdr::MvdrParams p;
    p.num_elements = config.get<int>(unique_name, "num_elements");
    p.num_freq = config.get<int>(unique_name, "num_local_freq");
    p.block_size = config.get_default<int>(unique_name, "block_size", 2);
    p.samples_per_frame = config.get_default<double>(unique_name, "samples_per_frame", -1.0);
    if (p.samples_per_frame <= 0.0)
        p.samples_per_frame = config.get<int>(unique_name, "samples_per_data_set");

    const double f0_mhz = config.get<double>(unique_name, "freq_start_mhz");
    const double df_mhz = config.get<double>(unique_name, "delta_freq_mhz");
    for (int k = 0; k < p.num_freq; ++k)
        p.freqs_hz.push_back((f0_mhz + k * df_mhz) * 1e6);

    p.pos_x_m = config.get<std::vector<double>>(unique_name, "positions_x_m");
    p.pos_y_m = config.get<std::vector<double>>(unique_name, "positions_y_m");
    for (const auto& lm : config.get<std::vector<std::vector<double>>>(unique_name, "beams")) {
        if (lm.size() != 2)
            throw std::invalid_argument("chartsMVDRBeams: each beam must be [l, m]");
        p.beam_l.push_back(lm[0]);
        p.beam_m.push_back(lm[1]);
    }
    p.loading = config.get_default<double>(unique_name, "loading", 10.0);
    p.loading_ref = charts_mvdr::parse_loading_ref(
        config.get_default<std::string>(unique_name, "loading_ref", "noise"));

    const std::string gains_file = config.get_default<std::string>(unique_name, "gains_file", "");
    if (!gains_file.empty()) {
        const int first = config.get_default<int>(unique_name, "gains_first_channel", 0);
        p.gains = charts_mvdr::load_gains_npy(gains_file, first, p.num_freq, p.num_elements);
        INFO("Loaded gains from {:s} (rows {:d}-{:d})", gains_file, first, first + p.num_freq - 1);
    }

    half_window = config.get_default<int>(unique_name, "cov_half_window", 10);
    guard = config.get_default<int>(unique_name, "cov_guard", 1);
    min_frames = config.get_default<int>(unique_name, "cov_min_frames", half_window - guard);
    update_every = config.get_default<int>(unique_name, "update_every", 1);
    wng_warn_db = config.get_default<double>(unique_name, "wng_warn_db", 3.0);
    flush_after_frames = config.get_default<int64_t>(unique_name, "flush_after_frames", 0);
    if (half_window < 1 || guard < 0 || guard >= half_window)
        throw std::invalid_argument("chartsMVDRBeams: need cov_half_window >= 1 and "
                                    "0 <= cov_guard < cov_half_window");
    if (update_every < 1)
        throw std::invalid_argument("chartsMVDRBeams: update_every must be >= 1");

    engine = std::make_unique<charts_mvdr::MvdrEngine>(p);

    const int n_rec = p.num_elements / p.block_size;
    const std::size_t in_size = static_cast<std::size_t>(p.num_freq) * n_rec * (n_rec + 1) / 2
                                * p.block_size * p.block_size * 2 * sizeof(int32_t);
    if (in_buf->frame_size != in_size)
        throw std::invalid_argument("chartsMVDRBeams: in_buf frame size does not match the "
                                    "correlator layout");
    const std::size_t out_size =
        2 * static_cast<std::size_t>(p.num_freq) * engine->num_beams() * sizeof(float);
    if (out_buf->frame_size != out_size)
        throw std::invalid_argument("chartsMVDRBeams: out_buf frame size must be "
                                    + std::to_string(out_size));
    if (weights_buf && weights_buf->frame_size != engine->weights_frame_size())
        throw std::invalid_argument("chartsMVDRBeams: weights_buf frame size must be "
                                    + std::to_string(engine->weights_frame_size()));
    if (in_buf->num_frames < 2 * half_window + 2)
        throw std::invalid_argument("chartsMVDRBeams: in_buf needs at least "
                                    + std::to_string(2 * half_window + 2)
                                    + " frames (2 cov_half_window + 2)");

    in_buf->register_consumer(unique_name);
    out_buf->register_producer(unique_name);
    if (weights_buf)
        weights_buf->register_producer(unique_name);
}

chartsMVDRBeams::~chartsMVDRBeams() {}

namespace {
void copy_meta(Buffer* src, int src_id, Buffer* dst, int dst_id) {
    if (!dst->metadata_pool)
        return;
    dst->allocate_new_metadata_object(dst_id);
    if (src->metadata_pool && src->metadata_pool->type_name == dst->metadata_pool->type_name)
        src->copy_metadata(src_id, dst, dst_id);
}
} // namespace

bool chartsMVDRBeams::process(int64_t t, int64_t last_index) {
    const int nf = in_buf->num_frames;
    const int in_id = static_cast<int>(t % nf);

    const auto k = charts_mvdr::window_indices(t, last_index, half_window, guard);
    if (charts_mvdr::should_update(t, engine->has_weights(), update_every, k.size(), min_frames)) {
        std::vector<const int32_t*> frames;
        frames.reserve(k.size());
        for (int64_t i : k)
            frames.push_back(reinterpret_cast<const int32_t*>(in_buf->frames[i % nf]));
        if (!engine->update(frames)) {
            WARN("Frame {:d}: covariance not positive definite, weights not updated", t);
        } else {
            const auto wng = engine->wng();
            const float thr = std::pow(10.0f, static_cast<float>(wng_warn_db) / 10.0f);
            const auto n_high =
                std::count_if(wng.begin(), wng.end(), [&](float v) { return v > thr; });
            if (n_high > 0)
                WARN("Frame {:d}: WNG above {:.1f} dB in {:d} (channel, beam) pairs, max {:.1f} "
                     "dB",
                     t, wng_warn_db, n_high,
                     10.0 * std::log10(*std::max_element(wng.begin(), wng.end())));
            if (weights_buf) {
                uint8_t* w = weights_buf->wait_for_empty_frame(unique_name, weights_frame_id);
                if (w == nullptr)
                    return false;
                copy_meta(in_buf, in_id, weights_buf, weights_frame_id);
                engine->write_weights(w, t, static_cast<int64_t>(k.size()));
                weights_buf->mark_frame_full(unique_name, weights_frame_id);
                weights_frame_id = (weights_frame_id + 1) % weights_buf->num_frames;
            }
        }
    }

    uint8_t* out = out_buf->wait_for_empty_frame(unique_name, out_frame_id);
    if (out == nullptr)
        return false;
    copy_meta(in_buf, in_id, out_buf, out_frame_id);
    engine->beam_power(reinterpret_cast<const int32_t*>(in_buf->frames[in_id]),
                       reinterpret_cast<float*>(out));
    out_buf->mark_frame_full(unique_name, out_frame_id);
    out_frame_id = (out_frame_id + 1) % out_buf->num_frames;
    return true;
}

void chartsMVDRBeams::main_thread() {
    const int nf = in_buf->num_frames;
    int64_t received = 0;  // input frames held or released
    int64_t next_out = 0;  // next output frame
    int64_t released = 0;  // input frames marked empty

    while (!stop_thread) {
        const bool flushing = flush_after_frames > 0 && received >= flush_after_frames;
        if (!flushing) {
            uint8_t* frame = in_buf->wait_for_full_frame(unique_name, received % nf);
            if (frame == nullptr)
                break;
            ++received;
        }
        const int64_t last = received - 1;
        while (next_out < received && (flushing || next_out + half_window <= last)) {
            if (!process(next_out, last))
                return;
            ++next_out;
            // frame k is needed by outputs t >= k - H only
            while (released < next_out - half_window) {
                in_buf->mark_frame_empty(unique_name, released % nf);
                ++released;
            }
        }
        if (flushing) {
            while (released < received) {
                in_buf->mark_frame_empty(unique_name, released % nf);
                ++released;
            }
            INFO("Flushed after {:d} input frames ({:d} output frames)", received, next_out);
            break;
        }
    }
}
