/**
 * @file
 * @brief Sends quantized FRB beams to a pirate FRB search node over TCP
 *  - pirateNetworkSend : public kotekan::Stage
 */
#ifndef PIRATE_NETWORK_SEND_HPP
#define PIRATE_NETWORK_SEND_HPP

#include "Config.hpp"            // for Config
#include "Stage.hpp"             // for Stage
#include "buffer.hpp"            // for Buffer
#include "bufferContainer.hpp"   // for bufferContainer
#include "prometheusMetrics.hpp" // for Counter

#include "json.hpp" // for json

#include <atomic>             // for atomic
#include <condition_variable> // for condition_variable
#include <cstddef>            // for size_t
#include <cstdint>            // for int64_t, uint32_t, uint8_t
#include <mutex>              // for mutex
#include <netinet/in.h>       // for sockaddr_in
#include <string>             // for string
#include <vector>             // for vector

/**
 * @class pirateNetworkSend
 * @brief Sends quantized FRB beams to one pirate receiver, using pirate's X->FRB network protocol
 *        (version 2).
 *
 * pirate (github.com/kmsmith137/pirate, `notes/network_protocol.md`) receives intensity over one
 * persistent TCP connection per sender. Each sender sends a subset of the frequency channels for
 * all beams of the receiver, and the receiver assembles the channels of all its senders. When the
 * connection opens, this stage sends the magic `0xf4bf4b02`, the flags, the metadata length and
 * the zero-terminated metadata, in the format of pirate's `configs/xengine_metadata.yml`. It then
 * sends one minichunk per frame: the magic, the uint64 `seq` of the first sample, and the
 * scales/offsets and int4 data of the beams in `beam_range`. The header is sent again on every new
 * connection. Everything is little-endian.
 *
 * Each frame holds one minichunk of 256 time samples of all `num_beams` beams and the channels in
 * `freq_channel_range`, already in pirate's conventions:
 *  - float16 `[num_beams][num_freq][2]`: (scale, offset), reconstructed as `scale * q + offset`;
 *  - int4 `[num_beams][num_freq][256]`, two's complement, two per byte with the first sample in the
 *    low nibble; -8 marks a masked sample.
 * Beam is the slowest index, so the beams of `beam_range` are one contiguous block in each part
 * and are sent without copying. Several instances may read the same buffer to send different beam
 * ranges to different receivers.
 *
 * The metadata is `xengine_metadata` as given, sent as JSON (which pirate's YAML parser accepts),
 * with `version`, `freq_channels` (the channels of `freq_channel_range`) and, unless given,
 * `unix_ns_at_seq_0` (from the Telescope) added. pirate refuses unknown keys and requires every
 * sender to agree on all fields but `freq_channels`, which must be disjoint across senders; pirate
 * does not check that.
 *
 * `seq` is the frame's `fpga_seq_num` divided by `seq_divisor`. It must be a multiple of
 * 256 * `seq_per_frb_time_sample` and increase from frame to frame; anything else is a
 * FATAL_ERROR. Frames dropped while the connection is down or slow leave a gap in `seq`, which
 * pirate treats as masked data.
 *
 * @par Buffers
 * @buffer in_buf The minichunks to send, one per frame.
 *     @buffer_format float16 scales/offsets followed by int4 data, as above; the frame size must be
 *                    num_beams * num_freq * 132 bytes.
 *     @buffer_metadata chordMetadata
 *
 * @conf server_ip           String. IP address of the pirate receiver.
 * @conf server_port         Int. TCP port of the pirate receiver.
 * @conf local_ip            String, optional. Local address to send from, to choose the interface.
 * @conf num_beams           Int. Number of beams in each frame.
 * @conf beam_range          Array of 2 Ints, default [0, num_beams]. Half-open range of the beams
 *                           of each frame sent to this receiver.
 * @conf freq_channel_range  Array of 2 Ints. Half-open range of pirate's global frequency channel
 *                           indices held by each frame, in frame order.
 * @conf seq_divisor         Int, default 1. Number of FPGA samples per pirate `seq` count.
 * @conf xengine_metadata    Object. pirate's X-engine metadata, without `version` and
 *                           `freq_channels`. Must contain `seq_per_frb_time_sample` and one
 *                           `beam_ids` entry per beam of `beam_range`.
 * @conf ack                 Bool, default false. Ask the receiver to answer one byte per minichunk
 *                           (1 if it assembled it). For tests only.
 * @conf send_timeout        Int, default 20. Seconds before a `send()` (or the wait for an ack)
 *                           times out and the connection is closed.
 * @conf reconnect_time      Int, default 5. Seconds between connection attempts.
 * @conf drop_frames         Bool, default true. Drop frames while the connection is down, and when
 *                           more than `drop_threshold` of the buffer is full, instead of blocking.
 * @conf drop_threshold      Float, default 0.6. Fraction of full frames above which frames are
 *                           dropped.
 *
 * @par Metrics
 * @metric kotekan_pirate_network_send_sent_minichunk_count
 *         The number of minichunks sent.
 * @metric kotekan_pirate_network_send_dropped_frame_count
 *         The number of frames dropped because the connection was down or too slow.
 * @metric kotekan_pirate_network_send_unassembled_minichunk_count
 *         With `ack`, the number of minichunks the receiver got but did not assemble.
 */
class pirateNetworkSend : public kotekan::Stage {
public:
    pirateNetworkSend(kotekan::Config& config, const std::string& unique_name,
                      kotekan::bufferContainer& buffer_container);
    ~pirateNetworkSend();

    void main_thread() override;

    /// Adds the receiver to the pipeline graph
    void add_graph_details(kotekan::PipelineGraph& graph) const override;

    /// Magic number of pirate's protocol version 2, at the start of the header and every minichunk
    static constexpr uint32_t magic = 0xf4bf4b02;
    /// Number of time samples per minichunk
    static constexpr int64_t samples_per_minichunk = 256;

private:
    /// Sends the connection header; returns false on a socket error
    bool send_header();
    /// Sends the minichunk in frame `frame`; returns false on a socket error
    bool send_minichunk(const uint8_t* frame, uint64_t seq);
    /// Closes the connection and wakes the connect thread to reconnect
    void close_connection();
    /// Thread that connects to the receiver whenever the connection is down
    void connect_to_server();

    Buffer* in_buf;

    std::string server_ip;
    uint32_t server_port;
    std::string local_ip;

    std::ptrdiff_t num_beams;
    std::ptrdiff_t beam_begin, beam_end;
    std::ptrdiff_t num_freq;
    int64_t seq_divisor;
    int64_t seq_per_minichunk;
    bool ack;

    uint32_t send_timeout;
    uint32_t reconnect_time;
    bool drop_frames;
    float drop_threshold;

    /// Connection header: magic, flags, length and the zero-padded metadata
    std::vector<uint8_t> header;

    kotekan::prometheus::Counter& sent_counter;
    kotekan::prometheus::Counter& dropped_counter;
    kotekan::prometheus::Counter& unassembled_counter;

    struct sockaddr_in server_addr;
    int socket_fd;
    std::atomic<bool> connected;
    /// Whether the header has been sent on the current connection; reset by the connect thread
    std::atomic<bool> header_sent;

    std::mutex connection_state_mutex;
    std::condition_variable connection_state_cv;
};

#endif
