#define BOOST_TEST_MODULE "test_pirateNetworkSend"

#include "Config.hpp"            // for Config
#include "buffer.hpp"            // for Buffer
#include "bufferContainer.hpp"   // for bufferContainer
#include "chartsMetadata.hpp"    // for chartsMetadata
#include "chordMetadata.hpp"     // for chordMetadata
#include "errors.h"              // for get_exit_code, ReturnCode
#include "kotekanLogging.hpp"    // for kotekanLogging, FatalError
#include "metadata.hpp"          // for metadataPool
#include "pirateNetworkSend.hpp" // for pirateNetworkSend

#include "json.hpp" // for json

#include <algorithm>                         // for equal
#include <arpa/inet.h>                       // for htonl, ntohs
#include <boost/test/included/unit_test.hpp> // for BOOST_CHECK, BOOST_REQUIRE
#include <chrono>                            // for milliseconds
#include <csignal>                           // for signal, SIGTERM, SIG_IGN
#include <cstdint>                           // for uint8_t, uint16_t, uint32_t, uint64_t
#include <cstring>                           // for memcpy
#include <fstream>                           // for ifstream
#include <iterator>                          // for istreambuf_iterator
#include <memory>                            // for shared_ptr
#include <netinet/in.h>                      // for sockaddr_in, INADDR_LOOPBACK
#include <string>                            // for string
#include <sys/socket.h>                      // for socket, bind, listen, accept, recv, send
#include <sys/time.h>                        // for timeval
#include <thread>                            // for sleep_for
#include <unistd.h>                          // for close
#include <vector>                            // for vector

using kotekan::bufferContainer;
using kotekan::Config;
using json = nlohmann::json;

// No kotekan_logging_fixture: it fails the test on an ERROR logged from another thread, and the
// stage logs one with its FATAL_ERROR and when the receiver closes the connection.

// FATAL_ERROR raises SIGTERM (exit_kotekan) before it throws.
struct ignore_sigterm {
    ignore_sigterm() {
        std::signal(SIGTERM, SIG_IGN);
    }
};
BOOST_GLOBAL_FIXTURE(ignore_sigterm);

// 4 beams and 4 channels per frame; beams [1, 3) are sent.
static constexpr int num_beams = 4;
static constexpr int num_freq = 4;
static constexpr int beam_begin = 1;
static constexpr int beam_end = 3;
static constexpr int frame_size = num_beams * num_freq * (4 + 128);
// 256 * seq_per_frb_time_sample
static constexpr uint64_t seq_per_minichunk = 256 * 320;

static int listen_localhost(uint16_t& port) {
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    BOOST_REQUIRE(fd >= 0);
    sockaddr_in addr = {};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    BOOST_REQUIRE(bind(fd, (sockaddr*)&addr, sizeof(addr)) == 0);
    socklen_t len = sizeof(addr);
    BOOST_REQUIRE(getsockname(fd, (sockaddr*)&addr, &len) == 0);
    port = ntohs(addr.sin_port);
    BOOST_REQUIRE(listen(fd, 1) == 0);
    return fd;
}

// Accepts a connection whose reads time out after 5 s.
static int accept_connection(int listener) {
    const int fd = accept(listener, nullptr, nullptr);
    BOOST_REQUIRE(fd >= 0);
    timeval timeout = {5, 0};
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    return fd;
}

static std::vector<uint8_t> recv_bytes(int fd, size_t n) {
    std::vector<uint8_t> out(n);
    size_t got = 0;
    while (got < n) {
        const ssize_t r = recv(fd, out.data() + got, n - got, 0);
        BOOST_REQUIRE_MESSAGE(r > 0, "connection closed or timed out");
        got += r;
    }
    return out;
}

template<typename T>
static T read_le(const uint8_t* p) {
    T v;
    std::memcpy(&v, p, sizeof(T));
    return v;
}

static json xengine_metadata() {
    return {{"zone_nfreq", {21504}},
            {"zone_freq_edges", {299.85, 501.75}},
            {"beamset", 1},
            {"beam_ids", {7, 8}},
            {"beam_positions_x", {0.0, 0.1}},
            {"beam_positions_y", {0.0, 0.0}},
            {"unix_ns_at_seq_0", 1772483060000000000},
            {"dt_ns_per_seq", 3333},
            {"seq_per_frb_time_sample", 320},
            {"noise_variance", 1.0}};
}

static json sender_config(const std::string& name, uint16_t port) {
    json cfg;
    cfg["type"] = "config";
    cfg["log_level"] = "WARN";
    cfg["cpu_affinity"] = std::vector<int>{0};
    cfg[name] = {{"kotekan_stage", "pirateNetworkSend"},
                 {"in_buf", "pirate_buf"},
                 {"server_ip", "127.0.0.1"},
                 {"server_port", port},
                 {"num_beams", num_beams},
                 {"beam_range", {beam_begin, beam_end}},
                 {"freq_channel_range", {100, 100 + num_freq}},
                 {"reconnect_time", 1},
                 {"xengine_metadata", xengine_metadata()}};
    return cfg;
}

static std::shared_ptr<metadataPool> chord_pool() {
    return metadataPool::create(8, sizeof(chordMetadata), "pirate_pool", "chordMetadata");
}

// Frame `index`: float16 (scale, offset) per beam and channel, then int4 codes, two per byte with
// the first sample in the low nibble. testdata/pirate_network_send_golden.py builds the same
// values.
static std::vector<uint8_t> frame_contents(int index) {
    std::vector<uint8_t> frame(frame_size);
    uint8_t* p = frame.data();
    for (int b = 0; b < num_beams; ++b)
        for (int f = 0; f < num_freq; ++f) {
            const uint16_t bits = 16 * b + f + index;
            const uint16_t scale_offset[2] = {uint16_t(0x3800 + bits), uint16_t(0x2000 + bits)};
            std::memcpy(p, scale_offset, sizeof(scale_offset));
            p += sizeof(scale_offset);
        }
    for (int b = 0; b < num_beams; ++b)
        for (int f = 0; f < num_freq; ++f)
            for (int t = 0; t < 256; t += 2) {
                const int q0 = (7 * b + 3 * f + t + 5 * index) % 16 - 8;
                const int q1 = (7 * b + 3 * f + t + 1 + 5 * index) % 16 - 8;
                *p++ = uint8_t((q0 & 0x0f) | ((q1 & 0x0f) << 4));
            }
    return frame;
}

static void produce(Buffer& buf, int frame_id, int64_t fpga_seq_num, int index) {
    uint8_t* frame = buf.wait_for_empty_frame("test", frame_id);
    BOOST_REQUIRE(frame != nullptr);
    buf.allocate_new_metadata_object(frame_id);
    const auto meta = buf.get_metadata(frame_id);
    if (const auto chord = std::dynamic_pointer_cast<chordMetadata>(meta))
        chord->set_fpga_seq_num(fpga_seq_num);
    else
        std::dynamic_pointer_cast<chartsMetadata>(meta)->set_fpga_seq_num(fpga_seq_num);
    const std::vector<uint8_t> contents = frame_contents(index);
    std::memcpy(frame, contents.data(), frame_size);
    buf.mark_frame_full("test", frame_id);
}

// Reads the connection header and returns its metadata.
static json recv_header(int fd, uint32_t expected_flags) {
    const std::vector<uint8_t> start = recv_bytes(fd, 12);
    BOOST_CHECK_EQUAL(read_le<uint32_t>(start.data()), 0xf4bf4b02);
    BOOST_CHECK_EQUAL(read_le<uint32_t>(start.data() + 4), expected_flags);
    const uint32_t len = read_le<uint32_t>(start.data() + 8);
    BOOST_CHECK_EQUAL(len % 8, 0);
    const std::vector<uint8_t> text = recv_bytes(fd, len);
    BOOST_REQUIRE_EQUAL(text.back(), 0);
    return json::parse(std::string((const char*)text.data()));
}

// Reads one minichunk and checks it against frame `index`, sent with `seq`.
static void check_minichunk(int fd, uint64_t seq, int index) {
    const size_t scales_bytes = (beam_end - beam_begin) * num_freq * 4;
    const size_t data_bytes = (beam_end - beam_begin) * num_freq * 128;
    const std::vector<uint8_t> mc = recv_bytes(fd, 12 + scales_bytes + data_bytes);
    BOOST_CHECK_EQUAL(read_le<uint32_t>(mc.data()), 0xf4bf4b02);
    BOOST_CHECK_EQUAL(read_le<uint64_t>(mc.data() + 4), seq);

    const std::vector<uint8_t> frame = frame_contents(index);
    const uint8_t* scales = frame.data() + beam_begin * num_freq * 4;
    const uint8_t* data = frame.data() + num_beams * num_freq * 4 + beam_begin * num_freq * 128;
    BOOST_CHECK(std::equal(mc.begin() + 12, mc.begin() + 12 + scales_bytes, scales));
    BOOST_CHECK(std::equal(mc.begin() + 12 + scales_bytes, mc.end(), data));
}

BOOST_AUTO_TEST_CASE(test_matches_reference_sender, *boost::unit_test::timeout(20)) {
    uint16_t port = 0;
    const int listener = listen_localhost(port);

    json cfg = sender_config("send_basic", port);
    cfg["send_basic"]["drop_frames"] = false;
    Config config;
    config.update_config(cfg);

    Buffer buf(4, frame_size, chord_pool(), "pirate_buf", "standard", 0, false, false,
               std::vector<int>{}, true);
    buf.register_producer("test");
    bufferContainer bc;
    bc.add_buffer("pirate_buf", &buf);
    pirateNetworkSend send(config, "/send_basic", bc);
    send.start();

    const int conn = accept_connection(listener);

    // The second frame skips a minichunk, which pirate treats as masked data.
    produce(buf, 0, 3 * seq_per_minichunk, 0);
    produce(buf, 1, 5 * seq_per_minichunk, 1);

    // The whole stream matches the bytes of pirate's reference sender (protocol.py).
    std::ifstream golden_file(TEST_DATA_DIR "/pirate_network_send_golden.bin", std::ios::binary);
    BOOST_REQUIRE(golden_file);
    const std::vector<uint8_t> golden((std::istreambuf_iterator<char>(golden_file)),
                                      std::istreambuf_iterator<char>());
    const std::vector<uint8_t> stream = recv_bytes(conn, golden.size());
    BOOST_CHECK(stream == golden);

    // The configured metadata passes through, with version and freq_channels added.
    const json meta = json::parse(std::string((const char*)stream.data() + 12));
    BOOST_CHECK_EQUAL(meta.at("version"), 2);
    BOOST_CHECK(meta.at("freq_channels") == json({100, 101, 102, 103}));
    const json configured = xengine_metadata();
    for (const auto& [key, value] : configured.items())
        BOOST_CHECK_MESSAGE(meta.at(key) == value, key);
    BOOST_CHECK_EQUAL(meta.size(), configured.size() + 2);

    // Nothing else was sent.
    uint8_t byte;
    BOOST_CHECK_LT(recv(conn, &byte, 1, MSG_DONTWAIT), 0);

    send.stop();
    buf.send_shutdown_signal();
    send.join();
    close(conn);
    close(listener);
}

BOOST_AUTO_TEST_CASE(test_ack_and_seq_divisor, *boost::unit_test::timeout(20)) {
    uint16_t port = 0;
    const int listener = listen_localhost(port);

    json cfg = sender_config("send_ack", port);
    cfg["send_ack"]["drop_frames"] = false;
    cfg["send_ack"]["ack"] = true;
    cfg["send_ack"]["seq_divisor"] = 3;
    Config config;
    config.update_config(cfg);

    Buffer buf(4, frame_size, chord_pool(), "pirate_buf", "standard", 0, false, false,
               std::vector<int>{}, true);
    buf.register_producer("test");
    bufferContainer bc;
    bc.add_buffer("pirate_buf", &buf);
    pirateNetworkSend send(config, "/send_ack", bc);
    send.start();

    const int conn = accept_connection(listener);
    produce(buf, 0, 3 * seq_per_minichunk, 0);
    produce(buf, 1, 6 * seq_per_minichunk, 1);

    recv_header(conn, 1);
    check_minichunk(conn, seq_per_minichunk, 0);

    // The second minichunk waits for the first one's acknowledgement.
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    uint8_t byte;
    BOOST_CHECK_LT(recv(conn, &byte, 1, MSG_DONTWAIT), 0);
    const uint8_t assembled = 1;
    BOOST_REQUIRE_EQUAL(::send(conn, &assembled, 1, 0), 1);
    check_minichunk(conn, 2 * seq_per_minichunk, 1);
    BOOST_REQUIRE_EQUAL(::send(conn, &assembled, 1, 0), 1);

    send.stop();
    buf.send_shutdown_signal();
    send.join();
    close(conn);
    close(listener);
}

BOOST_AUTO_TEST_CASE(test_charts_metadata, *boost::unit_test::timeout(20)) {
    uint16_t port = 0;
    const int listener = listen_localhost(port);

    json cfg = sender_config("send_charts", port);
    cfg["send_charts"]["drop_frames"] = false;
    Config config;
    config.update_config(cfg);

    Buffer buf(4, frame_size,
               metadataPool::create(8, sizeof(chartsMetadata), "charts_pool", "chartsMetadata"),
               "pirate_buf", "standard", 0, false, false, std::vector<int>{}, true);
    buf.register_producer("test");
    bufferContainer bc;
    bc.add_buffer("pirate_buf", &buf);
    pirateNetworkSend send(config, "/send_charts", bc);
    send.start();

    const int conn = accept_connection(listener);
    produce(buf, 0, 2 * seq_per_minichunk, 0);

    recv_header(conn, 0);
    check_minichunk(conn, 2 * seq_per_minichunk, 0);

    send.stop();
    buf.send_shutdown_signal();
    send.join();
    close(conn);
    close(listener);
}

BOOST_AUTO_TEST_CASE(test_reconnect_resends_header, *boost::unit_test::timeout(30)) {
    uint16_t port = 0;
    const int listener = listen_localhost(port);

    json cfg = sender_config("send_reconnect", port);
    cfg["send_reconnect"]["drop_frames"] = false;
    Config config;
    config.update_config(cfg);

    Buffer buf(4, frame_size, chord_pool(), "pirate_buf", "standard", 0, false, false,
               std::vector<int>{}, true);
    buf.register_producer("test");
    bufferContainer bc;
    bc.add_buffer("pirate_buf", &buf);
    pirateNetworkSend send(config, "/send_reconnect", bc);
    send.start();

    int conn = accept_connection(listener);
    produce(buf, 0, 0, 0);
    recv_header(conn, 0);
    check_minichunk(conn, 0, 0);
    close(conn);

    // Sends to the closed connection fail within a few frames; the stage then reconnects.
    int64_t seq = 0;
    int frame_id = 1;
    int index = 1;
    timeval no_wait = {0, 200000};
    setsockopt(listener, SOL_SOCKET, SO_RCVTIMEO, &no_wait, sizeof(no_wait));
    for (conn = -1; conn < 0; ++index, frame_id = (frame_id + 1) % 4) {
        seq += seq_per_minichunk;
        produce(buf, frame_id, seq, index);
        conn = accept(listener, nullptr, nullptr);
    }
    timeval timeout = {5, 0};
    setsockopt(conn, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));

    // The new connection starts with the header, then continues with a later minichunk.
    const json meta = recv_header(conn, 0);
    BOOST_CHECK(meta.at("freq_channels") == json({100, 101, 102, 103}));
    const std::vector<uint8_t> mc = recv_bytes(conn, 12);
    BOOST_CHECK_EQUAL(read_le<uint32_t>(mc.data()), 0xf4bf4b02);
    BOOST_CHECK_GT(read_le<uint64_t>(mc.data() + 4), 0);

    send.stop();
    buf.send_shutdown_signal();
    send.join();
    close(conn);
    close(listener);
}

BOOST_AUTO_TEST_CASE(test_bad_config, *boost::unit_test::timeout(10)) {
    Buffer buf(4, frame_size + 1, chord_pool(), "pirate_buf", "standard", 0, false, false,
               std::vector<int>{}, true);
    bufferContainer bc;
    bc.add_buffer("pirate_buf", &buf);

    Config config;
    config.update_config(sender_config("send_bad_size", 11024));
    BOOST_CHECK_THROW(pirateNetworkSend(config, "/send_bad_size", bc), FatalError);
    BOOST_CHECK_EQUAL(kotekan::kotekanLogging::get_error_message(),
                      "Buffer pirate_buf has frames of 2113 bytes, but 4 beams and 4 channels "
                      "need 2112");

    Buffer good_buf(4, frame_size, chord_pool(), "good_buf", "standard", 0, false, false,
                    std::vector<int>{}, true);
    bufferContainer good_bc;
    good_bc.add_buffer("pirate_buf", &good_buf);
    json cfg = sender_config("send_bad_beams", 11024);
    cfg["send_bad_beams"]["xengine_metadata"]["beam_ids"] = {7, 8, 9};
    config.update_config(cfg);
    BOOST_CHECK_THROW(pirateNetworkSend(config, "/send_bad_beams", good_bc), FatalError);
    BOOST_CHECK_EQUAL(kotekan::kotekanLogging::get_error_message(),
                      "xengine_metadata must set beam_ids, one per beam of beam_range (2)");
}

BOOST_AUTO_TEST_CASE(test_misaligned_seq, *boost::unit_test::timeout(10)) {
    uint16_t port = 0;
    const int listener = listen_localhost(port);

    json cfg = sender_config("send_misaligned", port);
    cfg["send_misaligned"]["drop_frames"] = false;
    Config config;
    config.update_config(cfg);

    Buffer buf(4, frame_size, chord_pool(), "pirate_buf", "standard", 0, false, false,
               std::vector<int>{}, true);
    buf.register_producer("test");
    bufferContainer bc;
    bc.add_buffer("pirate_buf", &buf);
    pirateNetworkSend send(config, "/send_misaligned", bc);
    send.start();

    const int conn = accept_connection(listener);
    produce(buf, 0, seq_per_minichunk + 320, 0);

    // The stage closes the connection without sending anything, not even the header.
    uint8_t byte;
    BOOST_CHECK_EQUAL(recv(conn, &byte, 1, 0), 0);

    // The stage thread ends with the FATAL_ERROR.
    send.stop();
    buf.send_shutdown_signal();
    send.join();
    BOOST_CHECK(get_exit_code() == ReturnCode::FATAL_ERROR);
    BOOST_CHECK_EQUAL(kotekan::kotekanLogging::get_error_message(),
                      "Frame pirate_buf[0]: seq 82240 must be a multiple of 81920 and follow the "
                      "previous seq -1");
    close(conn);
    close(listener);
}
