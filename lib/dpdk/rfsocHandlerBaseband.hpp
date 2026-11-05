#ifndef RFSOC_HANDLER_BASEBAND_HPP
#define RFSOC_HANDLER_BASEBAND_HPP

#include "BasebandMetadata.hpp" // for BasebandMetadata
#include "Config.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"
#include "chartsMetadata.hpp"
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
#include <packet_copy.h>
#include <string>
#include <util.h>
#include <vector>

// This is the RFoC Handler Class for the 32 antenna CHARTS deployment

class rfsocHandlerBaseband : public dpdkRXhandler {
public:
    /// Default constructor
    rfsocHandlerBaseband(kotekan::Config& config, const std::string& unique_name,
                         kotekan::bufferContainer& buffer_container, int port);

    int handle_packet(struct rte_mbuf* mbuf) override;
    void update_stats() override;

protected:
    // Output buffer
    Buffer* out_buf = nullptr;
    uint8_t* out_frame = nullptr;
    int out_frame_id = 0;

    // Config packet constants
    static constexpr uint32_t ETH_IP_UDP_HDR = 42;
    uint32_t packet_size = 5482; // ip/udp + rfsoc header + payload

    bool zero_new_frames = true;
    uint64_t packet_index = 0;
    uint32_t packets_written_in_frame = 0;


    // RFSoC parameters
    static constexpr uint32_t subbands = 4;
    uint32_t bytes_per_sb = 0;
    uint32_t bytes_per_spec = 0;

    static constexpr uint32_t rfsoc_header = 64;
    uint32_t payload_len = packet_size - ETH_IP_UDP_HDR - rfsoc_header;
    uint32_t payload_offset = ETH_IP_UDP_HDR + rfsoc_header;

    // Packet tracking
    bool first_packet = false;
    uint64_t cur_seq = 0;
    uint64_t last_seq = 0;

    uint64_t cur_spec = 0;
    uint64_t last_spec = 0;
    uint64_t frame_start_spec = 0;

    uint8_t subband = 0;
    uint32_t timestamp_sec = 0;
    uint32_t timestamp_micro = 0;
    uint32_t spectra_count = 0; // Spectra counter within the second, used for timestamp only.
    uint64_t time0_fpga = 0;
    uint32_t header_offset = 0;

    // Lost packet metric
    uint64_t rx_lost_packets_total = 0;
    uint64_t rx_packets_total = 0;
    uint64_t rx_bytes_total = 0;
    uint64_t rx_out_of_order_total = 0;
    uint64_t rx_error_total = 0;
    uint64_t rx_len_error_total = 0;
    uint64_t rx_samples_total = 0;
    uint64_t rx_lost_samples_total = 0;

    uint64_t rx_bytes_last = 0;
    uint64_t rx_lost_packets_last = 0;
    uint64_t rx_out_of_order_last = 0;
    uint64_t rx_packets_last = 0;

    Buffer* mask_buf = nullptr; // to track the missing samples
    uint8_t* mask_frame = nullptr;
    int mask_frame_id = 0;

    std::string mask_name;
    uint64_t specs_per_frame = 0;
    std::vector<uint8_t> packets_seen;
    uint64_t valid_specs_total = 0;


    // Capture control
    uint64_t num_frames_captured;
    uint64_t capture_n_frames;

    // Alignment (startup)
    bool got_first_packet = false;
    uint64_t alignment = 0;

    // For test and start
    double warmup_time = 10.0; // seconds
    std::chrono::steady_clock::time_point start_time;
    bool in_warmup = true;

    // Baseband Metadata pointer
    uint64_t event_id = 0;
    uint64_t freq_id = 0;

    // Prometheus metrics
    kotekan::prometheus::MetricFamily<kotekan::prometheus::Gauge>& rx_packets_total_metric;
    kotekan::prometheus::MetricFamily<kotekan::prometheus::Gauge>& rx_samples_total_metric;
    kotekan::prometheus::MetricFamily<kotekan::prometheus::Gauge>& rx_lost_packets_total_metric;
    kotekan::prometheus::MetricFamily<kotekan::prometheus::Gauge>& rx_lost_samples_total_metric;

    kotekan::prometheus::MetricFamily<kotekan::prometheus::Gauge>& rx_bytes_total_metric;
    kotekan::prometheus::MetricFamily<kotekan::prometheus::Gauge>& rx_error_total_metric;


    inline uint64_t extract_seq_le64(const uint8_t* p) const {
        uint64_t v = 0;
        std::memcpy(&v, p, sizeof(uint64_t));
        return le64toh(v);
    }

    inline uint8_t extract_subband_le8(const uint8_t* p) const {
        uint8_t s = 0;
        std::memcpy(&s, p, sizeof(uint8_t));
        return s;
    }

    inline uint64_t extract_timestamp_le32(const uint8_t* p) const {
        uint64_t t = 0;
        std::memcpy(&t, p, sizeof(uint32_t));
        return le32toh(t);
    }

    inline uint32_t count_bits(uint8_t mask) const {
        uint32_t count = 0;
        while (mask != 0) {
            count += mask & 1u;
            mask >>= 1;
        }
        return count;
    }


    inline bool check_packet_basic(struct rte_mbuf* mbuf) {
        if (unlikely(mbuf == nullptr)) {

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

    bool align_first_packet(uint64_t spec);

    bool advance_frame(uint64_t new_seq, bool first_time = false);

    bool copy_packet(struct rte_mbuf* mbuf);
};

inline rfsocHandlerBaseband::rfsocHandlerBaseband(kotekan::Config& config,
                                                  const std::string& unique_name,
                                                  kotekan::bufferContainer& buffer_container,
                                                  int port) :
    dpdkRXhandler(config, unique_name, buffer_container, port),
    rx_packets_total_metric(kotekan::prometheus::Metrics::instance().add_gauge(
        "kotekan_dpdk_rx_packets_total", unique_name, {"port"})),
    rx_samples_total_metric(kotekan::prometheus::Metrics::instance().add_gauge(
        "kotekan_dpdk_rx_samples_total", unique_name, {"port"})),
    rx_lost_packets_total_metric(kotekan::prometheus::Metrics::instance().add_gauge(
        "kotekan_dpdk_rx_lost_packets_total", unique_name, {"port"})),
    rx_lost_samples_total_metric(kotekan::prometheus::Metrics::instance().add_gauge(
        "kotekan_dpdk_rx_lost_samples_total", unique_name, {"port"})),
    rx_bytes_total_metric(kotekan::prometheus::Metrics::instance().add_gauge(
        "kotekan_dpdk_rx_bytes_total", unique_name, {"port"})),
    rx_error_total_metric(kotekan::prometheus::Metrics::instance().add_gauge(
        "kotekan_dpdk_rx_error_total", unique_name, {"port"})) {
    out_buf = buffer_container.get_buffer(config.get<std::string>(unique_name, "out_buffer"));
    if (!out_buf)
        FATAL_ERROR("rfsocHandlerBaseband: Could not find output buffer {:s} for handler {:s}",
                    config.get<std::string>(unique_name, "out_buffer"), unique_name);
    out_buf->register_producer(unique_name.c_str());

    mask_buf = buffer_container.get_buffer(config.get<std::string>(unique_name, "mask_buf"));
    if (!mask_buf)
        FATAL_ERROR("rfsocHandlerBaseband: Could not find mask buffer {:s} for handler {:s}",
                    config.get<std::string>(unique_name, "mask_buf"), unique_name);
    mask_name = unique_name + "_mask";
    mask_buf->register_producer(mask_name.c_str());
    mask_buf->zero_frames();

    alignment = config.get_default<uint64_t>(unique_name, "alignment", 0);

    // Number of frames to capture before stopping, 0 = unlimited
    capture_n_frames = config.get_default<uint64_t>(unique_name, "capture_n_frames", 0);
    num_frames_captured = 0;

    first_packet = false;
    rx_bytes_last = rx_bytes_total;
    header_offset = payload_offset;

    bytes_per_sb = payload_len;
    bytes_per_spec = subbands * bytes_per_sb;
    specs_per_frame = (out_buf->frame_size) / bytes_per_spec;
    packets_seen.resize(specs_per_frame, 0);

    start_time = std::chrono::steady_clock::now();
};


inline int rfsocHandlerBaseband::handle_packet(struct rte_mbuf* mbuf) {

    if (in_warmup) {
        const auto now = std::chrono::steady_clock::now();
        const double elapsed_time =
            std::chrono::duration_cast<std::chrono::duration<double>>(now - start_time).count();
        if (elapsed_time >= warmup_time) {
            in_warmup = false;
            INFO("rfsocHandlerBaseband: Warmup period of {:.2f} seconds ended. Starting capture.",
                 warmup_time);
        } else {
            return 0; // discard packets during warmup
        }
    }

    if (unlikely(!check_packet_basic(mbuf)))
        return 0;

    const uint8_t* pkt = rte_pktmbuf_mtod(mbuf, const uint8_t*);
    cur_seq = extract_seq_le64(pkt + ETH_IP_UDP_HDR);
    subband = extract_subband_le8(pkt + ETH_IP_UDP_HDR + 8);
    // DEBUG("rfsocHandlerBaseband: Packet seq: {:d}, subband: {:d}", cur_seq, subband);

    // The sequence number is the spectrum ID; all packets for the same spectrum
    // share it across subbands.
    cur_spec = cur_seq;

    timestamp_sec = extract_timestamp_le32(pkt + ETH_IP_UDP_HDR + 13);
    spectra_count = extract_timestamp_le32(pkt + ETH_IP_UDP_HDR + 9);
    timestamp_micro = (static_cast<uint64_t>(spectra_count) * 1000000ULL) / 300000ULL;
    time0_fpga = ((uint64_t)timestamp_sec) * 1000000ULL + ((uint64_t)timestamp_micro);

    if (unlikely(subband >= subbands)) {
        INFO("rfsocHandlerBaseband: Invalid subband number {:d} in packet. Discarding packet.",
             subband);
        rx_error_total += 1;
        return 0;
    }


    if (unlikely(!got_first_packet)) {
        if (!align_first_packet(cur_spec))
            return 0;
    }

    // Copy the packet data to the output buffer. Packet loss is tracked by the
    // per-spectrum completion mask because all subbands can share the same seq.
    if (unlikely(!copy_packet(mbuf))) {
        return -1;
    }

    if (cur_seq > last_seq) {
        last_seq = cur_seq;
    }

    return 0;
}

inline bool rfsocHandlerBaseband::align_first_packet(uint64_t spec) {

    if (alignment == 0) {
        FATAL_ERROR("rfsocHandlerBaseband: Alignment parameter must be set and greater than zero.");
    }

    if ((spec % alignment) <= 1000) {
        INFO("rfsocHandlerBaseband: Aligned at spectrum {:d}", spec);
        INFO("rfsocHandlerBaseband: Time0 FPGA timestamp: {:d} microseconds", time0_fpga);

        last_seq = cur_seq;
        got_first_packet = true;

        const uint64_t start_spec = spec - (spec % alignment);

        if (unlikely(!advance_frame(start_spec, true))) {
            got_first_packet = false;
            return false;
        }
        return true;
    }
    return false;
}


inline bool rfsocHandlerBaseband::advance_frame(uint64_t new_spec, bool first_time) {

    if (!first_time) {

        // Check for lost samples in the previous frame.
        for (uint64_t i = 0; i < specs_per_frame; i++) {
            const uint8_t missing_packet_mask = uint8_t(0x0F & ~packets_seen[i]);
            if (missing_packet_mask == 0) {
                mask_frame[i] = 0;
            } else {
                mask_frame[i] = 1;
                rx_lost_samples_total += 1;
                rx_lost_packets_total += count_bits(missing_packet_mask);
            }
        }

        DEBUG("advance_frame: closing frame_id {} start_spec {} valid_specs_total {}", out_frame_id,
              frame_start_spec, valid_specs_total);

        // out_buf->allocate_new_metadata_object(out_frame_id);
        out_buf->mark_frame_full(unique_name, out_frame_id);
        out_frame_id = (out_frame_id + 1) % out_buf->num_frames;

        // Mask buffer handling
        mask_buf->mark_frame_full(mask_name.c_str(), mask_frame_id);
        mask_frame_id = (mask_frame_id + 1) % mask_buf->num_frames;

        num_frames_captured++;
    }

    std::fill(packets_seen.begin(), packets_seen.end(), 0); // reset tracking

    if (capture_n_frames != 0 && num_frames_captured >= capture_n_frames) {
        INFO("rfsocHandlerBaseband: Reached the configured number of frames to capture ({:d}). "
             "Stopping capture.",
             capture_n_frames);
        return false; // stop capturing
    }

    out_frame = out_buf->wait_for_empty_frame(unique_name, out_frame_id);
    if (out_frame == nullptr) {
        return false;
    }

    mask_frame = mask_buf->wait_for_empty_frame(mask_name.c_str(), mask_frame_id);
    if (mask_frame == nullptr) {
        return false;
    }

    std::memset(mask_frame, 1, specs_per_frame); // set all to missing
    frame_start_spec = new_spec;

    // Set metadata values
    BasebandMetadata* out_metadata = nullptr;
    out_buf->allocate_new_metadata_object(out_frame_id);
    out_metadata = (BasebandMetadata*)(out_buf->get_metadata(out_frame_id).get());
    out_metadata->event_id = 0;
    out_metadata->freq_id = 0;
    out_metadata->time0_fpga = time0_fpga;
    out_metadata->frame_fpga_seq = frame_start_spec; // spec (sample) number
    return true;
}


inline bool rfsocHandlerBaseband::copy_packet(struct rte_mbuf* mbuf) {

    // The packet sequence number is already the spectrum ID.
    const uint64_t spec_id = cur_spec;

    if (spec_id < frame_start_spec) {
        rx_out_of_order_total++;
        return true;
    }

    uint64_t spec_loc = spec_id - frame_start_spec; // location in specs

    if (spec_loc >= specs_per_frame) { // need to advance frame
        if (!advance_frame(spec_id))
            return false;
        spec_loc = 0;
    }

    // Track received packets to the mask buffer.
    uint8_t& seen = packets_seen[spec_loc];
    const uint8_t bit = uint8_t(1u << subband);
    if (seen & bit) {
        return true;
    }

    const size_t byte_offset =
        spec_loc * bytes_per_spec + subband * bytes_per_sb; // location in bytes

    if (unlikely(byte_offset + bytes_per_sb > out_buf->frame_size)) {
        rx_error_total += 1;
        return true; // fuera de rango
    }

    int pkt_offset = payload_offset;
    copy_block(&mbuf, &out_frame[byte_offset], bytes_per_sb, &pkt_offset);


    seen |= bit;
    if (seen == 0x0F) { // 4 subbands recibidas
        mask_frame[spec_loc] = 0;
        valid_specs_total++;
        rx_samples_total += 1;
    }
    return true;
}


inline void rfsocHandlerBaseband::update_stats() {
    std::vector<std::string> port_label = {std::to_string(port)};
    rx_packets_total_metric.labels(port_label).set(rx_packets_total);
    rx_samples_total_metric.labels(port_label).set(rx_samples_total);
    rx_lost_packets_total_metric.labels(port_label).set(rx_lost_packets_total);
    rx_lost_samples_total_metric.labels(port_label).set(rx_lost_samples_total);

    rx_bytes_total_metric.labels(port_label).set(rx_bytes_total);
    rx_error_total_metric.labels(port_label).set(rx_error_total);

    double time_now = e_time();
    static double last_status_message_time = 0.0;
    const double status_cadence = 1.0; // seconds

    if ((time_now - last_status_message_time) > status_cadence) {

        const uint64_t d_packets = rx_packets_total - rx_packets_last;
        const uint64_t d_lost = rx_lost_packets_total - rx_lost_packets_last;
        const uint64_t d_bytes = rx_bytes_total - rx_bytes_last;

        const double mbps = (double)d_bytes * 8.0 / (status_cadence * 1e6);

        INFO("RFSoC port {:d} | RX pkt {:d} (+{:d}) | lost {:d} (+{:d}) | "
             " {:.2f} Mb/s | Samples {:d}",
             port, rx_packets_total, d_packets, rx_lost_packets_total, d_lost, mbps,
             rx_samples_total);

        rx_packets_last = rx_packets_total;
        rx_lost_packets_last = rx_lost_packets_total;
        rx_bytes_last = rx_bytes_total;

        last_status_message_time = time_now;
    }
}
#endif
