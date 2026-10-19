#define BOOST_TEST_MODULE "test_chartsTriggeredBasebandReadout"

#include "BasebandMetadata.hpp"
#include "BasebandWriter.hpp"
#include "bufferMerge.hpp"
#include "chartsMetadata.hpp"
#include "chartsTriggeredBasebandReadout.hpp"
#include "metadata.hpp"
#include "visUtil.hpp"

#include <boost/test/included/unit_test.hpp>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <sched.h>
#include <thread>
#include <unistd.h>

using namespace kotekan;
using nlohmann::json;

namespace {
void eventually(const std::function<bool()>& predicate) {
    const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!predicate()) {
        BOOST_REQUIRE_MESSAGE(std::chrono::steady_clock::now() < end, "timed out waiting for stage");
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
}

uint8_t sample(int64_t spectrum, int frequency, int raw_element, int stream) {
    return (spectrum + frequency * 17 + raw_element * 3 + stream * 53) & 255;
}

struct Pipeline {
    int input_spectra;
    int chunk_spectra;
    int input_id[2] = {0, 0};
    int output_id[2] = {0, 0};
    int correlation_frames = 0;
    std::string root;
    Config config;
    bufferContainer buffers;
    std::shared_ptr<metadataPool> input_pool;
    std::shared_ptr<metadataPool> output_pool;
    std::vector<std::unique_ptr<Buffer>> inputs;
    std::vector<std::unique_ptr<Buffer>> outputs;
    std::unique_ptr<Buffer> merged;
    std::unique_ptr<chartsTriggeredBasebandReadout> readout;
    std::unique_ptr<bufferMerge> merge;
    std::unique_ptr<BasebandWriter> writer;

    Pipeline(int input_size = 7, int chunk_size = 5, bool start_writer = true) :
        input_spectra(input_size), chunk_spectra(chunk_size) {
        char path[] = "/tmp/charts-triggered-test-XXXXXX";
        const char* directory = mkdtemp(path);
        BOOST_REQUIRE(directory != nullptr);
        root = directory;
        input_pool = metadataPool::create(64, sizeof(chartsMetadata), root + "-in", "chartsMetadata");
        output_pool = metadataPool::create(64, sizeof(BasebandMetadata), root + "-out", "BasebandMetadata");
        for (int i = 0; i < 2; ++i) {
            inputs.emplace_back(std::make_unique<Buffer>(10, input_size * 2 * 4, input_pool,
                "in" + std::to_string(i), "standard", 0, false, false, std::vector<int>{}, true));
            outputs.emplace_back(std::make_unique<Buffer>(16, chunk_size * 2 * 2, output_pool,
                "out" + std::to_string(i), "standard", 0, false, false, std::vector<int>{}, true));
            buffers.add_buffer("in" + std::to_string(i), inputs.back().get());
            buffers.add_buffer("out" + std::to_string(i), outputs.back().get());
            inputs.back()->register_producer("network");
            inputs.back()->register_consumer("correlator");
            outputs.back()->register_consumer("inspect");
        }
        merged = std::make_unique<Buffer>(8, chunk_size * 2 * 2, output_pool,
            "merged", "standard", 0, false, false, std::vector<int>{}, true);
        buffers.add_buffer("merged", merged.get());
        json cfg = {{"log_level", "ERROR"}, {"cpu_affinity", {sched_getcpu()}},
            {"samples_per_data_set", input_size}, {"num_elements", 4},
            {"capture", {{"in_bufs", {"in0", "in1"}}, {"out_bufs", {"out0", "out1"}},
                {"freq_ids", {0, 2}}, {"selected_antennas", {0, 2}},
                {"output_spectra_per_frame", chunk_size}, {"num_frames_buffer", 6},
                {"spectra_per_second", 10.0}, {"max_dump_samples", "99.9 * spectra_per_second"},
                {"capture_timeout", 0.1}, {"readout_status", root}, {"endpoint", "/charts/baseband-test"}}},
            {"merge", {{"in_bufs", {"out0", "out1"}}, {"out_buf", "merged"}, {"timeout", 0.01}}},
            {"writer", {{"in_buf", "merged"}, {"num_elements", 2},
                {"selected_antennas", {0, 2}}, {"root_path", root},
                {"spectra_per_frame", chunk_size}, {"readout_status", root}}}};
        const auto writer_name = "writer_" + root.substr(root.find_last_of('/') + 1);
        cfg[writer_name] = cfg["writer"];
        cfg.erase("writer");
        config.update_config(cfg);
        readout = std::make_unique<chartsTriggeredBasebandReadout>(config, "/capture", buffers);
        merge = std::make_unique<bufferMerge>(config, "/merge", buffers);
        writer = std::make_unique<BasebandWriter>(config, "/" + writer_name, buffers);
        if (start_writer) {
            writer->start();
            merge->start();
        }
        readout->start();
    }

    ~Pipeline() {
        readout->stop();
        merge->stop();
        writer->stop();
        for (auto& buffer : inputs)
            buffer->send_shutdown_signal();
        for (auto& buffer : outputs)
            buffer->send_shutdown_signal();
        merged->send_shutdown_signal();
        readout->join();
        merge->join();
        writer->join();
        readout.reset();
        merge.reset();
        writer.reset();
        std::filesystem::remove_all(root);
    }

    void feed(int64_t start, int only_stream = -1) {
        for (int stream = 0; stream < 2; ++stream) {
            if (only_stream >= 0 && stream != only_stream)
                continue;
            auto& buffer = *inputs[stream];
            const int id = input_id[stream];
            uint8_t* data = buffer.wait_for_empty_frame("network", id);
            BOOST_REQUIRE(data != nullptr);
            for (int t = 0; t < input_spectra; ++t)
                for (int f = 0; f < 2; ++f)
                    for (int a = 0; a < 4; ++a)
                        data[(t * 2 + f) * 4 + a] = sample(start + t, f, a, stream);
            buffer.allocate_new_metadata_object(id);
            auto meta = get_charts_metadata(&buffer, id);
            meta->set_fpga_seq_num(start);
            meta->set_time0_fpga(1000000 + start * 100000);
            buffer.mark_frame_full("network", id);
            BOOST_REQUIRE(buffer.wait_for_full_frame("correlator", id) != nullptr);
            buffer.mark_frame_empty("correlator", id);
            ++correlation_frames;
            input_id[stream] = (id + 1) % buffer.num_frames;
        }
        eventually([&] {
            const auto state = readout->status();
            for (int i = 0; i < 2; ++i) {
                if (only_stream >= 0 && i != only_stream)
                    continue;
                if (state["streams"][i].value("end_spectrum", int64_t(-1)) != start + input_spectra)
                    return false;
            }
            return true;
        });
    }

    json trigger(uint64_t id, int64_t length, const std::string& mode = "previous") {
        return readout->trigger({{"event_id", id}, {"duration_seconds", length / 10.0}, {"mode", mode}});
    }

    void inspect(uint64_t event, int64_t start, int64_t length) {
        for (int stream = 0; stream < 2; ++stream) {
            int64_t copied = 0;
            while (copied < length) {
                auto& buffer = *outputs[stream];
                int& id = output_id[stream];
                BOOST_REQUIRE_EQUAL(buffer.wait_for_full_frame_timeout("inspect", id,
                    double_to_ts(current_time() + 5)), 0);
                auto meta = std::static_pointer_cast<BasebandMetadata>(buffer.get_metadata(id));
                BOOST_CHECK_EQUAL(meta->event_id, event);
                BOOST_CHECK_EQUAL(meta->freq_id, stream * 2);
                BOOST_CHECK_EQUAL(meta->event_start_fpga, start);
                BOOST_CHECK_EQUAL(meta->event_end_fpga, start + length);
                BOOST_CHECK_EQUAL(meta->frame_fpga_seq, start + copied);
                const int64_t valid = std::min<int64_t>(chunk_spectra, length - copied);
                BOOST_CHECK_EQUAL(meta->valid_to, valid);
                for (int64_t t = 0; t < valid; ++t)
                    for (int f = 0; f < 2; ++f)
                        for (int a = 0; a < 2; ++a)
                            BOOST_CHECK_EQUAL(buffer.frames[id][(t * 2 + f) * 2 + a],
                                sample(start + copied + t, f, a == 0 ? 3 : 1, stream));
                for (size_t byte = valid * 4; byte < buffer.frame_size; ++byte)
                    BOOST_CHECK_EQUAL(buffer.frames[id][byte], 0);
                copied += valid;
                buffer.mark_frame_empty("inspect", id);
                id = (id + 1) % buffer.num_frames;
            }
        }
        eventually([&] { return readout->status()["events"][0]["status"] == "done"; });
        const auto state = readout->status()["events"][0];
        BOOST_CHECK_EQUAL(state["bytes_total"].get<size_t>(), length * 8);
        BOOST_CHECK_EQUAL(state["bytes_staged"].get<size_t>(), length * 8);
        BOOST_CHECK_EQUAL(state["bytes_remaining"].get<size_t>(), 0);
        const size_t record_size = chunk_spectra * 4 + BasebandMetadata().get_serialized_size();
        for (int stream = 0; stream < 2; ++stream) {
            const auto path = root + "/baseband_raw_" + std::to_string(event) + "/baseband_"
                              + std::to_string(event) + "_" + std::to_string(stream * 2) + ".data";
            BOOST_CHECK_EQUAL(std::filesystem::file_size(path),
                              ((length + chunk_spectra - 1) / chunk_spectra) * record_size);
            std::ifstream file(path, std::ios::binary);
            std::vector<char> bytes(BasebandMetadata().get_serialized_size());
            file.read(bytes.data(), bytes.size());
            BasebandMetadata metadata;
            metadata.set_from_bytes(bytes.data(), bytes.size());
            BOOST_CHECK_EQUAL(metadata.frame_fpga_seq, start);
            uint8_t first;
            file.read(reinterpret_cast<char*>(&first), 1);
            BOOST_CHECK_EQUAL(first, sample(start, 0, 3, stream));
        }
        eventually([&] { return !readout->status()["extracting"].get<bool>(); });
    }
};
}

BOOST_AUTO_TEST_CASE(previous_capture_rechunks_arbitrary_input_frames) {
    for (const auto& sizes : std::vector<std::pair<int, int>>{{7, 5}, {3, 7}, {8, 8}}) {
        Pipeline p(sizes.first, sizes.second);
        p.feed(0);
        p.feed(sizes.first);
        p.feed(2 * sizes.first);
        const int64_t length = sizes.first + 2;
        const int64_t start = 3 * sizes.first - length;
        BOOST_CHECK_EQUAL(p.trigger(1, length)["start_spectrum"].get<int64_t>(), start);
        p.inspect(1, start, length);
        BOOST_CHECK_EQUAL(p.correlation_frames, 6);
    }
}

BOOST_AUTO_TEST_CASE(next_capture_aligns_streams_and_allows_repeated_events) {
    Pipeline p;
    p.feed(0);
    p.feed(7, 0); // Stream 0 is one frame ahead.
    BOOST_CHECK_EQUAL(p.trigger(10, 9, "next")["start_spectrum"].get<int64_t>(), 14);
    p.feed(7, 1);
    p.feed(14);
    p.feed(21);
    p.inspect(10, 14, 9);
    eventually([&] { return p.outputs[0]->is_frame_empty(0) && p.outputs[1]->is_frame_empty(0); });
    p.trigger(11, 2);
    p.inspect(11, 26, 2);
    // Continue well beyond capture and ring wrap; the correlation consumer keeps receiving.
    for (int64_t s = 28; s < 140; s += 7)
        p.feed(s);
    BOOST_CHECK_GT(p.correlation_frames, 30);
}

BOOST_AUTO_TEST_CASE(reject_invalid_busy_expired_and_oversized_requests) {
    Pipeline p;
    BOOST_CHECK_THROW(p.trigger(1, 2), std::runtime_error);
    for (int64_t s = 0; s < 70; s += 7)
        p.feed(s);
    BOOST_CHECK_THROW(p.trigger(1, 1001), std::invalid_argument);
    BOOST_CHECK_THROW(p.trigger(1, 81), std::invalid_argument); // Staging holds only 80 spectra.
    BOOST_CHECK_THROW(p.trigger(1, 0), std::invalid_argument);
    BOOST_CHECK_THROW(p.trigger(1, 2, "bad-mode"), std::invalid_argument);
    BOOST_CHECK_THROW(p.readout->trigger({{"event_id", 1}, {"mode", "spectrum"},
        {"start_spectrum", 0}, {"duration_seconds", 0.2}}), std::invalid_argument);
    p.trigger(2, 9, "next");
    BOOST_CHECK_THROW(p.trigger(2, 2), std::invalid_argument);
    BOOST_CHECK_THROW(p.trigger(3, 2), std::runtime_error);
    p.feed(70);
    p.feed(77);
    p.inspect(2, 70, 9);
}

BOOST_AUTO_TEST_CASE(staging_is_not_reported_as_disk_completion_and_capture_can_timeout) {
    Pipeline p(7, 5, false);
    p.feed(0);
    p.trigger(1, 2);
    eventually([&] { return p.readout->status()["events"][0]["status"] == "staged"; });
    BOOST_CHECK_EQUAL(p.readout->status()["events"][0]["bytes_remaining"].get<int>(), 16);
    BOOST_CHECK_THROW(p.trigger(2, 2), std::runtime_error);
    // Explicit disk failure releases admission and remains visible in status.
    basebandReadoutManager::named(p.root)->written(1, 0, "injected disk failure");
    eventually([&] { return p.readout->status()["events"][0]["status"] == "error"; });
    eventually([&] { return !p.readout->status()["extracting"].get<bool>(); });
    p.trigger(3, 1, "next");
    eventually([&] { return p.readout->status()["events"][0]["status"] == "error"; });
    BOOST_CHECK(p.readout->status()["events"][0]["reason"].get<std::string>().find("timed out")
                != std::string::npos);
}

BOOST_AUTO_TEST_CASE(sequence_gaps_fail_and_shutdown_cancels_future_capture) {
    Pipeline p;
    p.feed(0);
    p.trigger(1, 9, "next");
    p.feed(14); // Missing spectra [7,14).
    eventually([&] { return p.readout->status()["events"][0]["status"] == "error"; });
    eventually([&] { return !p.readout->status()["extracting"].get<bool>(); });
    p.trigger(2, 9, "next");
    p.readout->stop();
    eventually([&] { return p.readout->status()["events"][0]["status"] == "error"; });
}

BOOST_AUTO_TEST_CASE(drain_closes_admission_and_preserves_accepted_capture) {
    Pipeline p;
    p.feed(0);
    p.trigger(1, 2, "next");
    auto manager = basebandReadoutManager::named(p.root);
    manager->close_requests();
    BOOST_CHECK(!p.readout->status()["accepting"].get<bool>());
    p.feed(7);
    p.inspect(1, 7, 2);
    BOOST_CHECK_THROW(p.trigger(2, 2), std::runtime_error);
}

BOOST_AUTO_TEST_CASE(duration_rounding_and_explicit_unaligned_start) {
    Pipeline p;
    p.feed(0);
    p.feed(7);
    BOOST_CHECK_EQUAL(p.readout->status()["max_capture_spectra"].get<int64_t>(), 999);
    BOOST_CHECK_THROW(p.readout->trigger({{"event_id", -1}, {"duration_seconds", 0.1}}),
                      std::invalid_argument);
    BOOST_CHECK_THROW(p.readout->trigger({{"event_id", 1}, {"duration_seconds", 0.01}}),
                      std::invalid_argument);
    const auto response = p.readout->trigger({{"event_id", 1}, {"mode", "spectrum"},
        {"start_spectrum", 6}, {"duration_seconds", 0.26}});
    BOOST_CHECK_EQUAL(response["capture_spectra"].get<int64_t>(), 3);
    BOOST_CHECK_CLOSE(response["duration_seconds"].get<double>(), 0.3, 0.0001);
    p.inspect(1, 6, 3);
}

BOOST_AUTO_TEST_CASE(writer_failure_does_not_stop_input_or_subsequent_captures) {
    Pipeline p(7, 5, false);
    p.feed(0);
    // A regular file obstructs the event directory and forces a real writer error.
    std::ofstream(p.root + "/baseband_raw_1") << "not a directory";
    p.trigger(1, 2);
    eventually([&] { return p.readout->status()["events"][0]["status"] == "staged"; });
    p.writer->start();
    p.merge->start();
    eventually([&] { return p.readout->status()["events"][0]["status"] == "error"; });
    for (int i = 0; i < 2; ++i) {
        BOOST_REQUIRE_EQUAL(p.outputs[i]->wait_for_full_frame_timeout("inspect", 0,
            double_to_ts(current_time() + 5)), 0);
        p.outputs[i]->mark_frame_empty("inspect", 0);
        p.output_id[i] = 1;
    }
    eventually([&] { return !p.readout->status()["extracting"].get<bool>(); });
    p.feed(7);
    p.trigger(2, 2);
    p.inspect(2, 12, 2);
    BOOST_CHECK_EQUAL(p.correlation_frames, 4);
}
