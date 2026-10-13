/**
 * @file
 * @brief A simple handler to capture uniformly sized packets
 * - captureHandler : public dpdkRXhandler
 */

#ifndef CAPTURE_HANDLER_HPP
#define CAPTURE_HANDLER_HPP

#include "Config.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"
#include "dpdkCore.hpp"
#include "packet_copy.h"
#include "prometheusMetrics.hpp"

#include "json.hpp"

#include <arpa/inet.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <endian.h>
#include <fstream>
#include <rte_mbuf.h>
#include <vector>

/**
 * @brief A simple handler to capture uniformly sized packets into a kotekan buffer
 *
 * This handler doesn't try to account for lost packets, or try to align frames.
 * It is simply for dumping any packets it gets into a buffer frame.
 *
 * @todo This is designed to capture FPGA packets which have a fixed length that is
 *       divisible by 32.  This class could be made more general.
 *
 * @par Buffers
 * @buffer out_buf  Kotekan buffer to place the packets in.
 *                  The frame size must be a multiple of the packet_size
 *       @buffer_format unit8_t array of packet contents
 *       @buffer_metadata none
 *
 * @conf packet_size    Int.  The size of the packet must be divisible by 32.
 *                              Includes Eth/IP/UDP headers.
 *
 * @author Andre Renard
 */
class captureHandler : public dpdkRXhandler {
public:
    /// Default constructor
    captureHandler(kotekan::Config& config, const std::string& unique_name,
                   kotekan::bufferContainer& buffer_container, int port);

    /// Processes the incoming packets
    int handle_packet(struct rte_mbuf* mbuf) override;

    /// Update stats, not used by this handler yet.
    virtual void update_stats() override {};

protected:
    /// The output buffer
    Buffer* out_buf;

    /// The current frame
    uint8_t* out_frame;

    /// The ID of the current frame
    int32_t out_frame_id = 0;

    /// Expected Packet size
    uint32_t packet_size;

    /// The location in the out_frame to put the next packet
    uint32_t packet_location = 0;

    /// Flag to setup variables for the first run
    bool first_run = true;

    // Packet loss metric
    uint64_t packets_received = 0;
    uint64_t packets_lost = 0;
    uint64_t expected_seq = 0;
    uint64_t bytes_received = 0;
    bool first_packet = true;


    std::chrono::steady_clock::time_point last_report;
    std::ofstream log_file;
    std::vector<u_int64_t> seq_buffer;
    static constexpr size_t SEQ_BUFFER_SIZE_MAX = 4096;
};

inline captureHandler::captureHandler(kotekan::Config& config, const std::string& unique_name,
                                      kotekan::bufferContainer& buffer_container, int port) :
    dpdkRXhandler(config, unique_name, buffer_container, port) {

    out_buf = buffer_container.get_buffer(config.get<std::string>(unique_name, "out_buf"));
    out_buf->register_producer(unique_name.c_str());

    packet_size = config.get<uint32_t>(unique_name, "packet_size");

    if (packet_size > (uint32_t)out_buf->frame_size) {
        throw std::runtime_error("The packet size must be less than the frame size");
    }

    if ((out_buf->frame_size % packet_size) != 0) {
        throw std::runtime_error("The buffer frame size must be a multiple of the packet size");
    }

    // TODO this seems overly restrictive, but removing this requires a generalized `copy_block`
    // function
    if ((packet_size % 32) != 0) {
        throw std::runtime_error("The packet_size must be a multiple of 32 bytes");
    }
    last_report = std::chrono::steady_clock::now();

    log_file.open("seq_log_port_" + std::to_string(port) + ".txt", std::ios::out | std::ios::trunc);

    if (!log_file.is_open()) {
        throw std::runtime_error("Failed to open log file for captureHandler");
    }
}

inline int captureHandler::handle_packet(struct rte_mbuf* mbuf) {

    // Get the first frame.
    if (first_run) {
        out_frame = out_buf->wait_for_empty_frame(unique_name, out_frame_id);
        if (out_frame == nullptr)
            return -1;
        first_run = false;
    }

#ifndef OLD_DPDK
    if (unlikely((mbuf->ol_flags & RTE_MBUF_F_RX_IP_CKSUM_MASK) == RTE_MBUF_F_RX_IP_CKSUM_BAD)) {
        WARN("Port: {:d}; Got bad packet IP checksum", port);
        return 0;
    }
    // Check the UDP checksum
    if (unlikely((mbuf->ol_flags & RTE_MBUF_F_RX_L4_CKSUM_MASK) == RTE_MBUF_F_RX_L4_CKSUM_BAD)) {
        WARN("Port: {:d}; Got bad packet UDP checksum", port);
        return 0;
    }
#else
    if (unlikely((mbuf->ol_flags | PKT_RX_IP_CKSUM_BAD) == 1)) {
        WARN("Port: {:d}; Got bad packet IP checksum", port);
        return 0;
    }
#endif

    // Copy the packet.
    const int ip_udp_header_size = 42; // skip header IP/UDP
    int offset = ip_udp_header_size;
    assert((packet_location + 1) * packet_size <= (uint32_t)out_buf->frame_size);

    // Packet loss calculation
    constexpr int SEQ_OFFSET = 0;
    constexpr uint64_t MAX_SEQ_JUMP = 4096;

    uint64_t seq_num_le;
    std::memcpy(&seq_num_le,
                rte_pktmbuf_mtod_offset(mbuf, uint8_t*, ip_udp_header_size + SEQ_OFFSET),
                sizeof(uint64_t));

    uint64_t seq_num = le64toh(seq_num_le);

    // Log sequence numbers
    seq_buffer.push_back(seq_num);
    if (seq_buffer.size() >= SEQ_BUFFER_SIZE_MAX) {
        for (const auto& seq : seq_buffer) {
            log_file << seq << "\n";
        }
        log_file.flush();
        seq_buffer.clear();
    }

    packets_received++;

    if (first_packet) {
        expected_seq = seq_num + 1;
        first_packet = false;
    } else {

        if (seq_num > expected_seq) {
            uint64_t gap = seq_num - expected_seq;
            if (gap < MAX_SEQ_JUMP) {
                packets_lost += gap;
            }
            expected_seq = seq_num + 1;
        } else if (seq_num == expected_seq) {
            expected_seq++;
        }
    }
    INFO("Packet seq: {}, expected seq: {}", seq_num, expected_seq);
    copy_block(&mbuf, &out_frame[packet_location * packet_size], packet_size, (int*)&offset);
    bytes_received += packet_size;
    packet_location++;

    if (packet_location * packet_size == (uint32_t)out_buf->frame_size) {
        out_buf->mark_frame_full(unique_name.c_str(), out_frame_id);
        out_frame_id = (out_frame_id + 1) % out_buf->num_frames;

        out_frame = out_buf->wait_for_empty_frame(unique_name, out_frame_id);
        if (out_frame == nullptr)
            return -1;

        packet_location = 0;
    }
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_report).count();

    if (elapsed > 1) {

        double pps = packets_received / (double)elapsed;
        double bps = (bytes_received * 8) / (double)elapsed;

        INFO("Capture Handler Port {}, Packets Received: {}, Packets Lost: {}, Bits per second: "
             "{}, Packets per second: {}",
             port, packets_received, packets_lost, bps, pps);
        last_report = now;

        packets_received = 0;
        bytes_received = 0;
        packets_lost = 0;
    }

    return 0;
}
#endif
