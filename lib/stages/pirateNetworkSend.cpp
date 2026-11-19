#include "pirateNetworkSend.hpp"

#include "Config.hpp"            // for Config
#include "PipelineGraph.hpp"     // for PipelineGraph, GraphNode
#include "StageFactory.hpp"      // for REGISTER_KOTEKAN_STAGE
#include "Telescope.hpp"         // for Telescope
#include "buffer.hpp"            // for Buffer
#include "bufferContainer.hpp"   // for bufferContainer
#include "chartsMetadata.hpp"    // for chartsMetadata
#include "chordMetadata.hpp"     // for chordMetadata
#include "kotekanLogging.hpp"    // for FATAL_ERROR, ERROR, WARN, INFO, DEBUG
#include "metadata.hpp"          // for metadataObject
#include "prometheusMetrics.hpp" // for Counter, Metrics

#include "fmt.hpp" // for format

#include <arpa/inet.h>  // for htons, inet_pton
#include <cerrno>       // for errno, EINTR
#include <chrono>       // for seconds
#include <cstring>      // for memcpy, strerror
#include <functional>   // for bind
#include <memory>       // for shared_ptr
#include <sys/socket.h> // for sendmsg, recv, connect, bind, setsockopt, socket, MSG_NOSIGNAL
#include <sys/time.h>   // for timeval
#include <sys/uio.h>    // for iovec
#include <thread>       // for thread
#include <unistd.h>     // for close

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using kotekan::prometheus::Metrics;

REGISTER_KOTEKAN_STAGE(pirateNetworkSend);

// Bytes per beam and channel in a frame: 2 float16 (scale, offset) and 256 int4 samples
static constexpr std::ptrdiff_t scales_bytes_per_channel = 2 * 2;
static constexpr std::ptrdiff_t data_bytes_per_channel =
    pirateNetworkSend::samples_per_minichunk / 2;

// Sends all of `iov`, resuming after partial sends. Returns false on a socket error, with errno
// set.
static bool send_all(int fd, iovec* iov, int iovcnt) {
    while (iovcnt > 0) {
        msghdr msg = {};
        msg.msg_iov = iov;
        msg.msg_iovlen = iovcnt;
        const ssize_t n = sendmsg(fd, &msg, MSG_NOSIGNAL);
        if (n < 0) {
            if (errno == EINTR)
                continue;
            return false;
        }
        size_t sent = n;
        while (iovcnt > 0 && sent >= iov->iov_len) {
            sent -= iov->iov_len;
            ++iov;
            --iovcnt;
        }
        if (iovcnt > 0) {
            iov->iov_base = static_cast<uint8_t*>(iov->iov_base) + sent;
            iov->iov_len -= sent;
        }
    }
    return true;
}

// fpga_seq_num of a frame with chordMetadata or chartsMetadata, or -1 if it has none
static int64_t frame_fpga_seq_num(const std::shared_ptr<metadataObject>& meta) {
    if (const auto chord = std::dynamic_pointer_cast<chordMetadata>(meta))
        return chord->has_fpga_seq_num() ? chord->get_fpga_seq_num() : -1;
    if (const auto charts = std::dynamic_pointer_cast<chartsMetadata>(meta))
        return charts->has_fpga_seq_num() ? charts->get_fpga_seq_num() : -1;
    return -1;
}

pirateNetworkSend::pirateNetworkSend(Config& config, const std::string& unique_name,
                                     bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&pirateNetworkSend::main_thread, this)),
    sent_counter(Metrics::instance().add_counter("kotekan_pirate_network_send_sent_minichunk_count",
                                                 unique_name)),
    dropped_counter(Metrics::instance().add_counter(
        "kotekan_pirate_network_send_dropped_frame_count", unique_name)),
    unassembled_counter(Metrics::instance().add_counter(
        "kotekan_pirate_network_send_unassembled_minichunk_count", unique_name)),
    socket_fd(-1), connected(false), header_sent(false) {

    in_buf = get_buffer("in_buf");
    if (!in_buf->metadata_pool)
        FATAL_ERROR("Buffer {:s} has no metadata_pool: the seq of each frame comes from its "
                    "metadata",
                    in_buf->buffer_name);
    in_buf->register_consumer(unique_name);

    server_ip = config.get<std::string>(unique_name, "server_ip");
    server_port = config.get<uint32_t>(unique_name, "server_port");
    local_ip = config.get_default<std::string>(unique_name, "local_ip", "");

    num_beams = config.get<std::ptrdiff_t>(unique_name, "num_beams");
    const auto beam_range =
        config.get_default<std::vector<std::ptrdiff_t>>(unique_name, "beam_range", {0, num_beams});
    if (beam_range.size() != 2 || beam_range[0] < 0 || beam_range[0] >= beam_range[1]
        || beam_range[1] > num_beams)
        FATAL_ERROR("beam_range must be [begin, end) with 0 <= begin < end <= num_beams = {:d}",
                    num_beams);
    beam_begin = beam_range[0];
    beam_end = beam_range[1];

    const auto freq_channel_range =
        config.get<std::vector<std::ptrdiff_t>>(unique_name, "freq_channel_range");
    if (freq_channel_range.size() != 2 || freq_channel_range[0] < 0
        || freq_channel_range[0] >= freq_channel_range[1])
        FATAL_ERROR("freq_channel_range must be [begin, end) with 0 <= begin < end");
    num_freq = freq_channel_range[1] - freq_channel_range[0];

    const std::ptrdiff_t frame_size =
        num_beams * num_freq * (scales_bytes_per_channel + data_bytes_per_channel);
    if (in_buf->frame_size != (size_t)frame_size)
        FATAL_ERROR("Buffer {:s} has frames of {:d} bytes, but {:d} beams and {:d} channels need "
                    "{:d}",
                    in_buf->buffer_name, in_buf->frame_size, num_beams, num_freq, frame_size);

    seq_divisor = config.get_default<int64_t>(unique_name, "seq_divisor", 1);
    if (seq_divisor < 1)
        FATAL_ERROR("seq_divisor must be positive, got {:d}", seq_divisor);
    ack = config.get_default<bool>(unique_name, "ack", false);

    send_timeout = config.get_default<uint32_t>(unique_name, "send_timeout", 20);
    reconnect_time = config.get_default<uint32_t>(unique_name, "reconnect_time", 5);
    drop_frames = config.get_default<bool>(unique_name, "drop_frames", true);
    drop_threshold = config.get_default<float>(unique_name, "drop_threshold", 0.6);

    nlohmann::json metadata = config.get<nlohmann::json>(unique_name, "xengine_metadata");
    if (!metadata.is_object())
        FATAL_ERROR("xengine_metadata must be an object");
    if (metadata.contains("version") || metadata.contains("freq_channels"))
        FATAL_ERROR("xengine_metadata must not set version or freq_channels: this stage sets them");
    if (!metadata.contains("seq_per_frb_time_sample")
        || !metadata["seq_per_frb_time_sample"].is_number_integer()
        || metadata["seq_per_frb_time_sample"].get<int64_t>() < 1)
        FATAL_ERROR("xengine_metadata must set seq_per_frb_time_sample to a positive integer");
    if (!metadata.contains("beam_ids") || !metadata["beam_ids"].is_array()
        || (std::ptrdiff_t)metadata["beam_ids"].size() != beam_end - beam_begin)
        FATAL_ERROR("xengine_metadata must set beam_ids, one per beam of beam_range ({:d})",
                    beam_end - beam_begin);
    seq_per_minichunk = samples_per_minichunk * metadata["seq_per_frb_time_sample"].get<int64_t>();

    std::vector<std::ptrdiff_t> freq_channels(num_freq);
    for (std::ptrdiff_t f = 0; f < num_freq; ++f)
        freq_channels[f] = freq_channel_range[0] + f;
    metadata["version"] = 2;
    metadata["freq_channels"] = freq_channels;
    if (!metadata.contains("unix_ns_at_seq_0"))
        metadata["unix_ns_at_seq_0"] = Telescope::instance().to_time_ns(0);

    // pirate reads a zero-terminated string; pad with at least one zero to a multiple of 8 bytes.
    const std::string text = metadata.dump();
    const uint32_t text_bytes = text.size() + 8 - text.size() % 8;
    const uint32_t flags = ack ? 1 : 0;
    header.resize(3 * sizeof(uint32_t) + text_bytes, 0);
    std::memcpy(header.data(), &magic, sizeof(uint32_t));
    std::memcpy(header.data() + 4, &flags, sizeof(uint32_t));
    std::memcpy(header.data() + 8, &text_bytes, sizeof(uint32_t));
    std::memcpy(header.data() + 12, text.data(), text.size());

    server_addr = {};
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(server_port);
    if (inet_pton(AF_INET, server_ip.c_str(), &server_addr.sin_addr) != 1)
        FATAL_ERROR("server_ip {:s} is not an IPv4 address", server_ip);
}

pirateNetworkSend::~pirateNetworkSend() {}

void pirateNetworkSend::main_thread() {
    int frame_id = 0;
    // seq of the last frame sent or dropped
    int64_t last_seq = -1;

    std::thread connect_thread(&pirateNetworkSend::connect_to_server, this);

    while (!stop_thread) {
        const uint8_t* frame = in_buf->wait_for_full_frame(unique_name, frame_id);
        if (frame == nullptr)
            break;

        const int64_t fpga_seq_num = frame_fpga_seq_num(in_buf->get_metadata(frame_id));
        std::string error;
        int64_t seq = -1;
        if (fpga_seq_num < 0) {
            error = fmt::format("Frame {:s}[{:d}] has no fpga_seq_num in a chordMetadata or "
                                "chartsMetadata",
                                in_buf->buffer_name, frame_id);
        } else if (fpga_seq_num % seq_divisor != 0) {
            error = fmt::format("Frame {:s}[{:d}]: fpga_seq_num {:d} is not a multiple of "
                                "seq_divisor {:d}",
                                in_buf->buffer_name, frame_id, fpga_seq_num, seq_divisor);
        } else {
            seq = fpga_seq_num / seq_divisor;
            if (seq % seq_per_minichunk != 0 || seq <= last_seq)
                error =
                    fmt::format("Frame {:s}[{:d}]: seq {:d} must be a multiple of {:d} and "
                                "follow the previous seq {:d}",
                                in_buf->buffer_name, frame_id, seq, seq_per_minichunk, last_seq);
        }
        if (!error.empty()) {
            // FATAL_ERROR throws out of this function, so end the connect thread first.
            stop();
            close_connection();
            connect_thread.join();
            // The connect thread may have connected after the first close.
            close_connection();
            FATAL_ERROR("{:s}", error);
        }

        const float full_fraction = (float)in_buf->get_num_full_frames() / in_buf->num_frames;
        if (drop_frames && (full_fraction > drop_threshold || !connected)) {
            INFO("Dropping frame {:s}[{:d}] (seq {:d}): {:s}", in_buf->buffer_name, frame_id, seq,
                 connected ? "sending is too slow" : "not connected");
            dropped_counter.inc();
        } else if (connected) {
            if (!header_sent) {
                if (!send_header()) {
                    ERROR("Failed to send the header to {:s}:{:d}: {:s}", server_ip, server_port,
                          std::strerror(errno));
                    close_connection();
                    continue;
                }
                header_sent = true;
            }
            if (!send_minichunk(frame, seq)) {
                ERROR("Failed to send seq {:d} to {:s}:{:d}: {:s}", seq, server_ip, server_port,
                      std::strerror(errno));
                close_connection();
                continue;
            }
            sent_counter.inc();
            DEBUG("Sent frame {:s}[{:d}] (seq {:d}) to {:s}:{:d}", in_buf->buffer_name, frame_id,
                  seq, server_ip, server_port);
        } else {
            // Without dropping, hold the frame until connected.
            std::unique_lock<std::mutex> lock(connection_state_mutex);
            connection_state_cv.wait_for(lock, std::chrono::seconds(1),
                                         [&]() { return stop_thread || connected; });
            continue;
        }

        last_seq = seq;
        in_buf->mark_frame_empty(unique_name, frame_id);
        frame_id = (frame_id + 1) % in_buf->num_frames;
    }

    close_connection();
    connect_thread.join();
    close_connection();
}

bool pirateNetworkSend::send_header() {
    iovec iov = {header.data(), header.size()};
    return send_all(socket_fd, &iov, 1);
}

bool pirateNetworkSend::send_minichunk(const uint8_t* frame, const uint64_t seq) {
    uint8_t minichunk_header[sizeof(uint32_t) + sizeof(uint64_t)];
    std::memcpy(minichunk_header, &magic, sizeof(uint32_t));
    std::memcpy(minichunk_header + sizeof(uint32_t), &seq, sizeof(uint64_t));

    const std::ptrdiff_t scales_bytes_per_beam = num_freq * scales_bytes_per_channel;
    const std::ptrdiff_t data_bytes_per_beam = num_freq * data_bytes_per_channel;
    const uint8_t* const scales = frame + beam_begin * scales_bytes_per_beam;
    const uint8_t* const data =
        frame + num_beams * scales_bytes_per_beam + beam_begin * data_bytes_per_beam;
    const std::ptrdiff_t nbeams = beam_end - beam_begin;

    iovec iov[3] = {
        {minichunk_header, sizeof(minichunk_header)},
        {const_cast<uint8_t*>(scales), size_t(nbeams * scales_bytes_per_beam)},
        {const_cast<uint8_t*>(data), size_t(nbeams * data_bytes_per_beam)},
    };
    if (!send_all(socket_fd, iov, 3))
        return false;

    if (ack) {
        uint8_t assembled;
        ssize_t n;
        while ((n = recv(socket_fd, &assembled, 1, 0)) < 0 && errno == EINTR) {
        }
        if (n != 1) {
            if (n == 0)
                errno = ECONNRESET;
            return false;
        }
        if (!assembled) {
            WARN("{:s}:{:d} did not assemble seq {:d}", server_ip, server_port, seq);
            unassembled_counter.inc();
        }
    }
    return true;
}

void pirateNetworkSend::close_connection() {
    if (socket_fd >= 0)
        close(socket_fd);
    socket_fd = -1;
    {
        std::unique_lock<std::mutex> lock(connection_state_mutex);
        connected = false;
    }
    connection_state_cv.notify_all();
}

void pirateNetworkSend::connect_to_server() {
    while (!stop_thread) {
        DEBUG("Connecting to {:s}:{:d}", server_ip, server_port);

        const int fd = socket(AF_INET, SOCK_STREAM, 0);
        bool ok = fd != -1;
        if (!ok)
            ERROR("Could not create a socket: {:s}", std::strerror(errno));
        if (ok && !local_ip.empty()) {
            sockaddr_in local_addr = {};
            local_addr.sin_family = AF_INET;
            if (inet_pton(AF_INET, local_ip.c_str(), &local_addr.sin_addr) != 1
                || bind(fd, (sockaddr*)&local_addr, sizeof(local_addr)) == -1) {
                WARN("Could not bind to local_ip {:s}: {:s}", local_ip, std::strerror(errno));
                ok = false;
            }
        }
        if (ok && connect(fd, (sockaddr*)&server_addr, sizeof(server_addr)) == -1) {
            WARN("Could not connect to {:s}:{:d}: {:s}; retrying in {:d} s", server_ip, server_port,
                 std::strerror(errno), reconnect_time);
            ok = false;
        }
        if (!ok) {
            if (fd != -1)
                close(fd);
            std::unique_lock<std::mutex> lock(connection_state_mutex);
            connection_state_cv.wait_for(lock, std::chrono::seconds(reconnect_time),
                                         [&]() { return (bool)stop_thread; });
            continue;
        }

        timeval timeout = {};
        timeout.tv_sec = send_timeout;
        if (setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout)) < 0
            || setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) < 0)
            ERROR("Could not set the socket timeouts: {:s}", std::strerror(errno));

        INFO("Connected to {:s}:{:d}, sending beams [{:d}, {:d}) of {:s}", server_ip, server_port,
             beam_begin, beam_end, in_buf->buffer_name);
        {
            std::unique_lock<std::mutex> lock(connection_state_mutex);
            socket_fd = fd;
            header_sent = false;
            connected = true;
        }
        connection_state_cv.notify_all();

        std::unique_lock<std::mutex> lock(connection_state_mutex);
        connection_state_cv.wait(lock, [&]() { return !connected || stop_thread; });
    }
}

void pirateNetworkSend::add_graph_details(kotekan::PipelineGraph& graph) const {
    const std::string dest_id = fmt::format("{:s}/destination", get_unique_name());
    graph.add_node(dest_id)
        .add_line(fmt::format(fmt("{:s}:{:d}"), server_ip, server_port))
        .set_category(kotekan::GraphCategory::Endpoint);
    graph.add_edge(get_unique_name(), dest_id);
}
