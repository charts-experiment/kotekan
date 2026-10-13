#ifndef RFSOC_HANDLER_CPT_HPP
#define RFSOC_HANDLER_CPT_HPP

#include "BasebandMetadata.hpp" // for BasebandMetadata
#include "Config.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"
#include "chordMetadata.hpp"
#include "dpdkCore.hpp"
#include "prometheusMetrics.hpp"

#include "json.hpp"

#include <arpa/inet.h>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <endian.h>
#include <fstream>
#include <rte_mbuf.h>
#include <string>
#include <util.h>
#include <vector>

class rfsocHandlerCPT : public dpdkRXhandler {
public:
    rfsocHandlerCPT(kotekan::Config& config, const std::string& unique_name,
                    kotekan::bufferContainer& buffer_container, int port);

    int handle_packet(struct rte_mbuf* mbuf) override;
    void update_stats() override {};

protected:
    // Output buffer
    Buffer* out_buf = nullptr;
    uint8_t* out_frame = nullptr;
    int out_frame_id = 0;

    // Config packet constants
    static constexpr uint32_t ETH_IP_UDP_HDR = 42;
    uint32_t packet_size = 5418;
    uint32_t frame_packets = 20480;

    uint32_t spectrum_size = 672;
    uint32_t blocks_per_packet = 8;
    uint32_t num_pol = 2;
    uint32_t payload_offset = 0;
    const uint32_t time_sampels_per_packet =
        blocks_per_packet / num_pol; // 4 time samples per packet
    bool zero_new_frames = true;

    uint32_t packets_written_in_frame = 0;

    std::chrono::steady_clock::time_point last_report;
    std::chrono::steady_clock::time_point last_rate_time;

    // Metrics
    uint64_t rx_packets_total = 0;
    uint64_t rx_bytes_total = 0;
    uint64_t rx_error_total = 0;
    uint64_t rx_len_error_total = 0;
    uint64_t rx_bytes_last = 0;

    // Baseband Metadata pointer
    uint64_t event_id = 0;
    uint64_t freq_id = 0;

    // Check basic packet validity
    inline bool check_packet_basic(struct rte_mbuf* mbuf) {
        if (unlikely(mbuf == nullptr)) {
            rx_error_total += 1;
            return false;
        }
        const uint32_t pkt_len = rte_pktmbuf_pkt_len(mbuf);
        if (unlikely(pkt_len != packet_size)) {
            rx_error_total += 1;
            rx_len_error_total += 1;
            return false;
        }
        rx_packets_total += 1;
        rx_bytes_total += pkt_len;
        return true;
    };

    // Acquire new output frame
    inline bool acquire_new_frame() {
        if (out_frame != nullptr) {
            return true;
        }
        out_frame = out_buf->wait_for_empty_frame(unique_name, out_frame_id);
        if (out_frame == nullptr) {
            rx_error_total += 1;
            return false;
        }

        if (zero_new_frames) {
            memset(out_frame, 0, out_buf->frame_size);
        }

        packets_written_in_frame = 0;
        return true;
    };

    // Finalize current output frame, mark full and fill metadata
    inline bool finalize_frame() {
        BasebandMetadata* out_metadata = nullptr;

        out_buf->allocate_new_metadata_object(out_frame_id);
        // auto meta = get_chord_metadata(out_buf, out_frame_id);

        out_metadata = (BasebandMetadata*)(out_buf->get_metadata(out_frame_id).get());
        // if (meta == nullptr) {
        //     rx_error_total += 1;
        //     WARN("rfsocHandlerCPT: Port {:d}; Failed to get chord metadata for frame {:d}",
        //          port, out_frame_id);
        //     return false;
        // }

        out_metadata->event_id = 0; // TODO: Set proper event ID (only testing)
        out_metadata->freq_id = 0;  // TODO: Set proper frequency ID (only testing)

        static uint64_t fpga_seq_counter = 0;
        // meta->set_fpga_seq_num(fpga_seq_counter);
        fpga_seq_counter += (packets_written_in_frame * time_sampels_per_packet);

        out_buf->mark_frame_full(unique_name, out_frame_id);
        out_frame = nullptr;
        out_frame_id = (out_frame_id + 1) % out_buf->num_frames;

        packets_written_in_frame = 0;
        return true;
    };

    inline void print_status() {
        const auto now = std::chrono::steady_clock::now();
        const auto elapsed_sec_1 =
            std::chrono::duration_cast<std::chrono::seconds>(now - last_report).count();
        const auto elapsed_sec_10 =
            std::chrono::duration_cast<std::chrono::seconds>(now - last_rate_time).count();

        if (elapsed_sec_1 >= 1) {
            INFO("rfsocHandlerCPT: Total duration: {:d} sec", (uint32_t)elapsed_sec_1);
            last_report = now;
        }

        if (elapsed_sec_10 >= 10) {
            // Calcula rate basado en intervalo completo de 10 segundos
            const auto rate_elapsed_sec =
                std::chrono::duration_cast<std::chrono::duration<double>>(now - last_rate_time)
                    .count();

            const uint64_t bytes_diff = rx_bytes_total - rx_bytes_last;
            const double bps = (rate_elapsed_sec > 0) ? (8.0 * bytes_diff / rate_elapsed_sec) : 0.0;

            // Print todo junto cada 10 segundos
            INFO("rfsocHandlerCPT: RX Stats - Total Packets: {:d}, "
                 "Errors: {:d}, Len Errors: {:d}, Bytes: {:d}, "
                 "Rate: {:.2f} Mbps ({:.2e} bps)",
                 rx_packets_total, rx_error_total, rx_len_error_total, rx_bytes_total, bps / 1e6,
                 bps);

            // Reset timers y contadores
            last_report = now;
            last_rate_time = now;
            rx_bytes_last = rx_bytes_total;
        }
    };
};

// Constructor
inline rfsocHandlerCPT::rfsocHandlerCPT(kotekan::Config& config, const std::string& unique_name,
                                        kotekan::bufferContainer& buffer_container, int port) :
    dpdkRXhandler(config, unique_name, buffer_container, port) {

    out_buf = buffer_container.get_buffer(config.get<std::string>(unique_name, "out_buffer"));

    if (!out_buf)
        FATAL_ERROR("rfsocHandlerCPT: Could not find output buffer {:s} for handler {:s}",
                    config.get<std::string>(unique_name, "out_buffer"), unique_name);

    out_buf->register_producer(unique_name.c_str());

    payload_offset = ETH_IP_UDP_HDR;
    last_report = std::chrono::steady_clock::now();
    last_rate_time = std::chrono::steady_clock::now();
    rx_bytes_last = rx_bytes_total;
};


inline int rfsocHandlerCPT::handle_packet(struct rte_mbuf* mbuf) {

    if (unlikely(!check_packet_basic(mbuf))) {
        print_status();
        return 0;
    }
    // End current frame if full
    if (packets_written_in_frame == frame_packets) {
        if (!finalize_frame()) {
            print_status();
            return -1;
        }
    }
    // Adquiere frame si es necesario
    if (out_frame == nullptr) {
        if (!acquire_new_frame()) {
            print_status();
            return -1;
        }
    }
    // Copy packet data to output frame
    const uint32_t packet_index = packets_written_in_frame;        // Current packet index in frame
    for (uint32_t block = 0; block < blocks_per_packet; block++) { // 8 blocks per packet
        const uint32_t pol = block % num_pol;
        const uint32_t t_in_pkt = block / num_pol;
        const uint64_t t =
            (uint64_t)packet_index * time_sampels_per_packet + t_in_pkt; // Global time sample index
        const uint64_t dst_offset = (t * num_pol + pol) * spectrum_size; //

        if (dst_offset + spectrum_size > (uint64_t)out_buf->frame_size) {
            WARN("rfsocHandlerCPT: Port {:d}; Destination offset exceeds frame size. "
                 "Dst Offset: {:d}, Frame Size: {:d}, Packet Index: {:d}, Block: {:d}",
                 port, dst_offset + spectrum_size, out_buf->frame_size, packet_index, block);
            return 0;
        }

        // Destination pointer in output frame
        uint8_t* dst = out_frame + dst_offset;
        const uint32_t src_offset = payload_offset + block * spectrum_size;

        // Copy data from packet to output frame
        const void* ret = rte_pktmbuf_read(mbuf, src_offset, spectrum_size, dst);
        if (unlikely(ret == nullptr)) {
            rx_error_total += 1;
            WARN("rfsocHandlerCPT: Port {:d}; Failed to read packet data. "
                 "Packet Index: {:d}, Block: {:d}",
                 port, packet_index, block);
            print_status();
            return 0;
        }

        if (ret != dst) {
            std::memcpy(dst, ret, spectrum_size);
        }
    }

    packets_written_in_frame++;
    print_status();
    return 0;
};

#endif // RFSOC_HANDLER_CPT_HPP