#include "chartsTriggeredBasebandReadout.hpp"

#include "BasebandMetadata.hpp"
#include "StageFactory.hpp"
#include "chartsMetadata.hpp"
#include "restServer.hpp"
#include "visUtil.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <set>
#include <stdexcept>
#include <thread>

using kotekan::basebandDumpStatus;
using kotekan::basebandReadoutManager;
using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using nlohmann::json;

REGISTER_KOTEKAN_STAGE(chartsTriggeredBasebandReadout);

namespace {
int64_t capture_limit(Config& config, const std::string& name) {
    // Evaluate duration/rate expressions before converting to an integer sample count.
    const double samples = config.get<double>(name, "max_dump_samples");
    if (!std::isfinite(samples) || samples < 1
        || static_cast<long double>(samples) >= std::numeric_limits<int64_t>::max())
        throw std::invalid_argument("max_dump_samples must be a positive spectrum count");
    return static_cast<int64_t>(samples);
}
}

chartsTriggeredBasebandReadout::chartsTriggeredBasebandReadout(Config& config, const std::string& unique_name,
                                             bufferContainer& buffers) :
    Stage(config, unique_name, buffers, std::bind(&chartsTriggeredBasebandReadout::main_thread, this)),
    samples_per_frame(config.get<int64_t>(unique_name, "samples_per_data_set")),
    output_spectra(config.get<int64_t>(unique_name, "output_spectra_per_frame")),
    retained_frames(config.get<int>(unique_name, "num_frames_buffer")),
    elements(config.get<int>(unique_name, "num_elements")),
    spectrum_rate(config.get<double>(unique_name, "spectra_per_second")),
    capture_timeout(config.get_default<double>(unique_name, "capture_timeout", 30.0)),
    max_capture_spectra(capture_limit(config, unique_name)),
    selected_antennas(config.get<std::vector<int>>(unique_name, "selected_antennas")),
    endpoint(config.get_default<std::string>(unique_name, "endpoint", unique_name + "/trigger")),
    manager(basebandReadoutManager::named(config.get<std::string>(unique_name, "readout_status"))) {
    if (samples_per_frame <= 0 || output_spectra <= 0 || retained_frames < 1 || elements < 1
        || !std::isfinite(spectrum_rate) || spectrum_rate <= 0 || max_capture_spectra <= 0
        || !std::isfinite(capture_timeout) || capture_timeout <= 0 || selected_antennas.empty()
        || endpoint.empty() || endpoint.front() != '/'
        || config.get<std::string>(unique_name, "readout_status").empty())
        throw std::invalid_argument("chartsTriggeredBasebandReadout: invalid dimensions, timing or endpoint");
    std::set<int> selected;
    for (int antenna : selected_antennas) {
        if (antenna < 0 || antenna >= elements || !selected.insert(antenna).second)
            throw std::invalid_argument("chartsTriggeredBasebandReadout: invalid or repeated antenna");
    }

    auto inputs = config.get<std::vector<std::string>>(unique_name, "in_bufs");
    auto outputs = config.get<std::vector<std::string>>(unique_name, "out_bufs");
    auto frequencies = config.get<std::vector<uint64_t>>(unique_name, "freq_ids");
    if (inputs.empty() || inputs.size() != outputs.size() || inputs.size() != frequencies.size())
        throw std::invalid_argument("chartsTriggeredBasebandReadout: stream lists must have equal lengths");
    std::set<Buffer*> used;
    std::set<uint64_t> freq_ids;
    for (size_t i = 0; i < inputs.size(); ++i) {
        Buffer* input = buffers.get_buffer(inputs[i]);
        Buffer* output = buffers.get_buffer(outputs[i]);
        if (!used.insert(input).second || !used.insert(output).second
            || !freq_ids.insert(frequencies[i]).second || input->num_frames <= retained_frames
            || !input->metadata_pool || input->metadata_pool->type_name != "chartsMetadata"
            || !output->metadata_pool || output->metadata_pool->type_name != "BasebandMetadata")
            throw std::invalid_argument("chartsTriggeredBasebandReadout: invalid buffers or frequency IDs");
        if (static_cast<uint64_t>(samples_per_frame) > std::numeric_limits<uint64_t>::max() / elements)
            throw std::invalid_argument("chartsTriggeredBasebandReadout: input dimensions overflow");
        const uint64_t divisor = static_cast<uint64_t>(samples_per_frame) * elements;
        if (!input->frame_size || input->frame_size % divisor != 0)
            throw std::invalid_argument("chartsTriggeredBasebandReadout: input dimensions do not match frame size");
        const size_t num_freq = input->frame_size / divisor;
        const size_t out_stride = num_freq * selected_antennas.size();
        if (static_cast<uint64_t>(output_spectra) > output->frame_size / out_stride
            || output->frame_size != static_cast<uint64_t>(output_spectra) * out_stride)
            throw std::invalid_argument("chartsTriggeredBasebandReadout: output dimensions do not match frame size");
        streams.push_back({input, output, frequencies[i], num_freq,
                           num_freq * static_cast<size_t>(elements), out_stride, 0, {}, true});
    }
    // Register only after all dimensions have been validated.
    for (auto& stream : streams) {
        stream.input->register_consumer(unique_name);
        stream.output->register_producer(unique_name);
    }
    auto& server = kotekan::restServer::instance();
    server.register_get_callback(endpoint, [this](kotekan::connectionInstance& conn) {
        conn.send_json_reply(status());
    });
    server.register_post_callback(endpoint, [this](kotekan::connectionInstance& conn, json& request) {
        try {
            conn.send_json_reply(trigger(request));
        } catch (const std::exception& ex) {
            conn.send_error(ex.what(), kotekan::HTTP_RESPONSE::BAD_REQUEST);
        }
    });
}

chartsTriggeredBasebandReadout::~chartsTriggeredBasebandReadout() {
    kotekan::restServer::instance().remove_get_callback(endpoint);
    kotekan::restServer::instance().remove_json_callback(endpoint);
    stop();
    join();
}

json chartsTriggeredBasebandReadout::trigger(const json& request) {
    if (!request.at("event_id").is_number_integer()
        || (!request.at("event_id").is_number_unsigned()
            && request.at("event_id").get<int64_t>() < 0))
        throw std::invalid_argument("event_id must be a nonnegative integer");
    const uint64_t event_id = request.at("event_id").get<uint64_t>();
    const double seconds = request.at("duration_seconds").get<double>();
    const long double count = static_cast<long double>(seconds) * spectrum_rate;
    if (!std::isfinite(seconds) || seconds <= 0 || count < 0.5L
        || count >= static_cast<long double>(std::numeric_limits<int64_t>::max()) - 0.5L)
        throw std::invalid_argument("duration_seconds exceeds max_dump_samples or rounds to zero");
    const int64_t length = std::llround(count);
    if (length > max_capture_spectra)
        throw std::invalid_argument("duration_seconds exceeds max_dump_samples");
    const std::string mode = request.value("mode", std::string("next"));
    if (mode != "next" && mode != "previous" && mode != "spectrum")
        throw std::invalid_argument("mode must be next, previous or spectrum");

    std::lock_guard<std::mutex> lock(mutex);
    if (!accepting || stop_thread)
        throw std::runtime_error("readout is stopping");
    if (manager->find(event_id))
        throw std::invalid_argument("event_id has already been used");
    if (worker_busy)
        throw std::runtime_error("another capture is being extracted");
    if (have_event) {
        auto previous = manager->find(last_event);
        if (previous && (previous->state == basebandDumpStatus::State::WAITING
                         || previous->state == basebandDumpStatus::State::INPROGRESS))
            throw std::runtime_error("previous capture has not finished writing to disk");
    }

    int64_t latest_end = 0;
    int64_t common_end = std::numeric_limits<int64_t>::max();
    int64_t common_start = 0;
    size_t bytes_per_spectrum = 0;
    for (const auto& stream : streams) {
        if (!stream.running || stream.frames.empty())
            throw std::runtime_error("all input streams must be receiving before triggering");
        const int64_t end = stream.frames.back().start + samples_per_frame;
        latest_end = std::max(latest_end, end);
        common_end = std::min(common_end, end);
        common_start = std::max(common_start, stream.frames.front().start);
        bytes_per_spectrum += stream.output_stride;
        const int64_t chunks = 1 + (length - 1) / output_spectra;
        if (chunks > stream.output->num_frames)
            throw std::invalid_argument("capture exceeds staging capacity; shorten duration or enlarge buffers");
        for (int64_t i = 0; i < chunks; ++i) {
            if (!stream.output->is_frame_empty((stream.output_id + i) % stream.output->num_frames))
                throw std::runtime_error("baseband staging buffers have not drained");
        }
    }
    int64_t start = latest_end;
    if (mode == "previous")
        start = common_end - length;
    if (mode == "spectrum") {
        if (!request.at("start_spectrum").is_number_integer())
            throw std::invalid_argument("start_spectrum must be a nonnegative integer");
        start = request.at("start_spectrum").get<int64_t>();
    }
    if (start < common_start || start > std::numeric_limits<int64_t>::max() - length)
        throw std::invalid_argument("requested start is outside available history or spectrum range");
    if (static_cast<uint64_t>(length) > std::numeric_limits<size_t>::max() / bytes_per_spectrum)
        throw std::invalid_argument("capture byte count overflows size_t");

    manager->add({event_id, start, length, "", ""});
    worker_busy = true;
    have_event = true;
    last_event = event_id;
    return {{"event_id", event_id}, {"status", "waiting"}, {"start_spectrum", start},
            {"capture_spectra", length}, {"duration_seconds", length / spectrum_rate}};
}

json chartsTriggeredBasebandReadout::status() {
    json reply;
    {
        std::lock_guard<std::mutex> lock(mutex);
        reply = {{"accepting", accepting && !stop_thread && manager->accepting_requests()}, {"spectra_per_second", spectrum_rate},
                 {"extracting", worker_busy},
                 {"max_capture_spectra", max_capture_spectra},
                 {"input_spectra_per_frame", samples_per_frame},
                 {"output_spectra_per_frame", output_spectra}, {"streams", json::array()}};
        for (const auto& stream : streams) {
            json entry = {{"freq_id", stream.freq_id}, {"receiving", stream.running}};
            if (!stream.frames.empty()) {
                entry["oldest_spectrum"] = stream.frames.front().start;
                entry["end_spectrum"] = stream.frames.back().start + samples_per_frame;
            }
            reply["streams"].push_back(entry);
        }
    }
    reply["events"] = json::array();
    for (const auto& event : manager->all()) {
        const char* state = "waiting";
        switch (event.state) {
            case basebandDumpStatus::State::WAITING: break;
            case basebandDumpStatus::State::INPROGRESS:
                state = event.bytes_staged == event.bytes_total ? "staged" : "capturing";
                break;
            case basebandDumpStatus::State::DONE: state = "done"; break;
            case basebandDumpStatus::State::ERROR: state = "error"; break;
        }
        reply["events"].push_back({{"event_id", event.request.event_id}, {"status", state},
            {"start_spectrum", event.request.start_fpga}, {"capture_spectra", event.request.length_fpga},
            {"duration_seconds", event.request.length_fpga / spectrum_rate},
            {"bytes_total", event.bytes_total}, {"bytes_staged", event.bytes_staged},
            {"bytes_remaining", event.bytes_remaining}, {"reason", event.reason}});
    }
    return reply;
}

void chartsTriggeredBasebandReadout::ingest(size_t index) {
    auto& stream = streams[index];
    int id = 0;
    while (!stop_thread) {
        const auto timeout = double_to_ts(current_time() + 0.1);
        const int result = stream.input->wait_for_full_frame_timeout(unique_name, id, timeout);
        if (result == 1)
            continue;
        if (result == -1)
            break;
        const auto meta = get_charts_metadata(stream.input, id);
        if (!meta || !meta->has_fpga_seq_num() || !meta->has_time0_fpga()
            || meta->get_fpga_seq_num() < 0
            || meta->get_fpga_seq_num() > std::numeric_limits<int64_t>::max() - samples_per_frame) {
            ERROR("CHARTS input is missing valid spectrum/time metadata");
            break;
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            // Sequence resets invalidate retained history, not an unrelated correlation consumer.
            if (!stream.frames.empty() && meta->get_fpga_seq_num() <= stream.frames.back().start) {
                for (const auto& frame : stream.frames)
                    stream.input->mark_frame_empty(unique_name, frame.id);
                stream.frames.clear();
            }
            stream.frames.push_back({id, meta->get_fpga_seq_num()});
            if (stream.frames.size() > static_cast<size_t>(retained_frames)) {
                stream.input->mark_frame_empty(unique_name, stream.frames.front().id);
                stream.frames.pop_front();
            }
        }
        data_ready.notify_all();
        id = (id + 1) % stream.input->num_frames;
    }
    {
        std::lock_guard<std::mutex> lock(mutex);
        stream.running = false;
        accepting = false;
        for (const auto& frame : stream.frames)
            stream.input->mark_frame_empty(unique_name, frame.id);
        stream.frames.clear();
    }
    stream.input->unregister_consumer(unique_name, true);
    data_ready.notify_all();
}

void chartsTriggeredBasebandReadout::main_thread() {
    std::thread worker(&chartsTriggeredBasebandReadout::readout, this);
    std::vector<std::thread> receivers;
    for (size_t i = 0; i < streams.size(); ++i)
        receivers.emplace_back(&chartsTriggeredBasebandReadout::ingest, this, i);
    for (auto& receiver : receivers)
        receiver.join();
    manager->stop();
    data_ready.notify_all();
    worker.join();
    manager->fail_pending("capture cancelled during readout shutdown");
    std::lock_guard<std::mutex> lock(mutex);
    worker_busy = false;
}

void chartsTriggeredBasebandReadout::fail(basebandDumpStatus& event, std::mutex& event_mutex,
                                  const std::string& reason) {
    std::lock_guard<std::mutex> lock(event_mutex);
    event.state = basebandDumpStatus::State::ERROR;
    event.reason = reason;
    event.finished = std::make_shared<std::chrono::system_clock::time_point>(
        std::chrono::system_clock::now());
    WARN("Baseband event {} failed: {}", event.request.event_id, reason);
}

void chartsTriggeredBasebandReadout::readout() {
    while (auto request = manager->get_next_waiting_request()) {
        auto& event = request->first;
        auto& event_mutex = request->second;
        try {
            extract(event, event_mutex);
        } catch (const std::exception& ex) {
            fail(event, event_mutex, ex.what());
        }
        std::lock_guard<std::mutex> lock(mutex);
        worker_busy = false;
    }
}

void chartsTriggeredBasebandReadout::extract(basebandDumpStatus& event, std::mutex& event_mutex) {
    const auto request = event.request;
    const int64_t end = request.start_fpga + request.length_fpga;
    size_t total_stride = 0;
    for (const auto& stream : streams)
        total_stride += stream.output_stride;
    {
        std::lock_guard<std::mutex> lock(event_mutex);
        event.bytes_total = request.length_fpga * total_stride;
        event.bytes_remaining = event.bytes_total;
        event.state = basebandDumpStatus::State::INPROGRESS;
        event.started = std::make_shared<std::chrono::system_clock::time_point>(
            std::chrono::system_clock::now());
    }
    const auto deadline = std::chrono::steady_clock::now()
                          + std::chrono::duration<double>(request.length_fpga / spectrum_rate
                                                          + capture_timeout);
    std::vector<int64_t> cursor(streams.size(), request.start_fpga);
    std::vector<int64_t> filled(streams.size(), 0);
    std::vector<uint8_t*> outputs(streams.size(), nullptr);
    size_t remaining = streams.size();
    while (remaining) {
        if (stop_thread)
            throw std::runtime_error("capture cancelled during shutdown");
        if (std::chrono::steady_clock::now() >= deadline)
            throw std::runtime_error("timed out waiting for requested spectra");
        bool progressed = false;
        for (size_t i = 0; i < streams.size(); ++i) {
            if (cursor[i] == end)
                continue;
            auto& stream = streams[i];
            std::unique_lock<std::mutex> lock(mutex);
            if (!stream.running)
                throw std::runtime_error("input stream stopped before capture completed");
            auto frame = std::find_if(stream.frames.begin(), stream.frames.end(), [&](const RetainedFrame& f) {
                return f.start <= cursor[i] && cursor[i] < f.start + samples_per_frame;
            });
            if (frame == stream.frames.end()) {
                if (!stream.frames.empty() && cursor[i] < stream.frames.back().start + samples_per_frame)
                    throw std::runtime_error("requested spectra expired or input has a sequence gap");
                continue;
            }
            const auto input_meta = get_charts_metadata(stream.input, frame->id);
            const int64_t offset = cursor[i] - frame->start;
            if (!outputs[i]) {
                // Never wait on disk while retaining the input buffer lock.
                if (!stream.output->is_frame_empty(stream.output_id))
                    throw std::runtime_error("baseband output full; capture incomplete");
                outputs[i] = stream.output->wait_for_empty_frame(unique_name, stream.output_id);
                if (!outputs[i])
                    throw std::runtime_error("baseband output shutting down");
                stream.output->allocate_new_metadata_object(stream.output_id);
                auto meta = std::static_pointer_cast<BasebandMetadata>(stream.output->get_metadata(stream.output_id));
                meta->event_id = request.event_id;
                meta->freq_id = stream.freq_id;
                meta->event_start_fpga = request.start_fpga;
                meta->event_end_fpga = end;
                // CHARTS time0_fpga is Unix microseconds at the input frame start.
                meta->time0_fpga = input_meta->get_time0_fpga()
                    + std::llround(offset * 1e6 / spectrum_rate);
                meta->time0_ctime = meta->time0_fpga / 1e6;
                meta->time0_ctime_offset = 0;
                meta->first_packet_recv_time = 0;
                meta->frame_fpga_seq = cursor[i];
                meta->valid_to = 0;
                meta->fpga0_ns = 0;
                meta->num_elements = selected_antennas.size();
                meta->reserved = -1;
            }
            // Bound lock hold time even when input frames contain many spectra.
            const int64_t count = std::min({end - cursor[i], samples_per_frame - offset,
                                           output_spectra - filled[i], int64_t(4096)});
            const uint8_t* source = stream.input->frames[frame->id] + offset * stream.input_stride;
            uint8_t* destination = outputs[i] + filled[i] * stream.output_stride;
            for (int64_t t = 0; t < count; ++t) {
                for (size_t f = 0; f < stream.frequencies; ++f) {
                    for (size_t a = 0; a < selected_antennas.size(); ++a)
                        destination[t * stream.output_stride + f * selected_antennas.size() + a]
                            = source[t * stream.input_stride + f * elements
                                     + elements - 1 - selected_antennas[a]];
                }
            }
            cursor[i] += count;
            filled[i] += count;
            const bool last = cursor[i] == end;
            if (filled[i] == output_spectra || last) {
                auto meta = std::static_pointer_cast<BasebandMetadata>(stream.output->get_metadata(stream.output_id));
                meta->valid_to = filled[i];
                if (filled[i] < output_spectra)
                    std::memset(outputs[i] + filled[i] * stream.output_stride, 0,
                                (output_spectra - filled[i]) * stream.output_stride);
                {
                    std::lock_guard<std::mutex> event_lock(event_mutex);
                    event.bytes_staged += filled[i] * stream.output_stride;
                }
                stream.output->mark_frame_full(unique_name, stream.output_id);
                stream.output_id = (stream.output_id + 1) % stream.output->num_frames;
                outputs[i] = nullptr;
                filled[i] = 0;
                if (last)
                    --remaining;
            }
            progressed = true;
        }
        if (!progressed) {
            std::unique_lock<std::mutex> lock(mutex);
            data_ready.wait_for(lock, std::chrono::milliseconds(20));
        }
        {
            std::lock_guard<std::mutex> lock(event_mutex);
            if (event.state == basebandDumpStatus::State::ERROR)
                return; // The disk writer has reported a failure.
        }
    }
}
