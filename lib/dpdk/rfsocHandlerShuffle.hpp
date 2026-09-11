#ifndef RFSOC_HANDLER_SHUFFLE_HPP
#define RFSOC_HANDLER_SHUFFLE_HPP

#include "Config.hpp"
#include "dpdkCore.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"
#include "prometheusMetrics.hpp"
#include "chartsMetadata.hpp"
#include "json.hpp"
#include <util.h>
#include <packet_copy.h>
#include <endian.h>
#include <arpa/inet.h>
#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstring>
#include <fstream>
#include <immintrin.h>
#include <limits>
#include <mutex>
#include <vector>
#include <string>
#include <cstdint>

// This is the RFSoC Handler Class for the 64 antenna CHARTS deployment.
// Each handler receives two subbands from two RFSoCs and writes one output
// spectrum in [time][freq][antenna] order.

class rfsocHandlerShuffle : public dpdkRXhandler {
public:
    /// Default constructor
    rfsocHandlerShuffle(kotekan::Config& config, const std::string& unique_name,
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

        //RFSoC parameters
        static constexpr uint32_t subbands = 4;
        static constexpr uint32_t rfsoc_header = 64;
        static constexpr uint32_t rfsoc_id_header_offset = 17;
        uint32_t payload_len = packet_size - ETH_IP_UDP_HDR - rfsoc_header;
        uint32_t bytes_per_sb = 0;
        uint32_t bytes_per_spec = 0; // or bytes_per_sample
        uint32_t payload_offset = ETH_IP_UDP_HDR + rfsoc_header;
        uint32_t num_rfsocs = 2;
        uint32_t num_elements_per_rfsoc = 32;
        uint32_t num_elements = 64;
        uint32_t n_channels_per_packet = 168;
        uint32_t subbands_per_nic = 2;
        uint32_t first_subband = 0;
        uint8_t rfsoc_id = 0;
        uint8_t complete_packet_mask = 0x0F;
        std::vector<uint8_t> packet_payload;
        struct PacketLayout {
            uint8_t bit = 0;
            uint32_t freq_base_offset = 0;
            uint32_t antenna_output_offset = 0;
        };
        std::array<PacketLayout, 8> packet_layout;

        // Packet tracking
        bool first_packet = false;
        uint64_t cur_seq = 0;
        uint64_t last_seq = 0;

        // Sample or spec tracking (used interchangeably)
        uint64_t cur_spec = 0;
        uint64_t last_spec = 0;
        uint64_t frame_start_spec = 0;

        // Other tracking variables
        uint8_t subband = 0;
        uint32_t timestamp_sec = 0;
        uint32_t timestamp_micro = 0;
        uint32_t spectra_count = 0; // Spectra counter within the second, used for timestamp only.
        uint32_t last_timestamp_sec = 0;
        uint32_t last_timestamp_micro = 0;
        uint64_t last_spectra_id = 0;
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
        uint64_t rx_packets_last = 0;
        uint64_t dpdk_ipackets_last = 0;
        uint64_t dpdk_imissed_last = 0;
        uint64_t dpdk_ierrors_last = 0;
        uint64_t dpdk_rx_nombuf_last = 0;

        // Mask buffer and frame to track missing samples
        Buffer* mask_buf = nullptr; //to track the missing samples
        uint8_t* mask_frame = nullptr;
        int mask_frame_id = 0;

        std::string mask_name;
        uint64_t specs_per_frame = 0;
        std::vector<uint8_t> packets_seen; // To track if the spectrum (or sample) is complete
        uint64_t valid_specs_in_frame = 0;
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
        double last_status_message_time = 0.0;

        // Metadata
        uint32_t num_local_freq = 0;
        uint32_t coarse_freq_start = 0;
        std::vector<int> frame_coarse_freq;


        // Prometheus metrics
        kotekan::prometheus::MetricFamily<kotekan::prometheus::Gauge>& rx_packets_total_metric;
        kotekan::prometheus::MetricFamily<kotekan::prometheus::Gauge>& rx_samples_total_metric;
        kotekan::prometheus::MetricFamily<kotekan::prometheus::Gauge>& rx_lost_packets_total_metric;
        kotekan::prometheus::MetricFamily<kotekan::prometheus::Gauge>& rx_lost_samples_total_metric;
        kotekan::prometheus::MetricFamily<kotekan::prometheus::Gauge>& rx_bytes_total_metric;
        kotekan::prometheus::MetricFamily<kotekan::prometheus::Gauge>& rx_error_total_metric;

        // Helper functions
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

        inline uint8_t extract_rfsoc_id(const uint8_t* p) const {
            uint8_t id = 0;
            std::memcpy(&id, p, sizeof(uint8_t));
            return id;
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

        template <bool aligned_dst>
        inline void copy_rfsoc_payload_32x64(uint8_t* dst, const uint8_t* src) const {
            constexpr uint32_t src_stride = 32;
            constexpr uint32_t dst_stride = 64;
            constexpr uint32_t unroll = 8;
            uint32_t freq = 0;

            // Each packet fills one 32-byte half of a 64-byte output cache line.
            // Cached stores let the companion RFSoC packet complete that line in cache.
            for (; freq + unroll <= n_channels_per_packet; freq += unroll) {
                const __m256i value0 =
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 0 * src_stride));
                const __m256i value1 =
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 1 * src_stride));
                const __m256i value2 =
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 2 * src_stride));
                const __m256i value3 =
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 3 * src_stride));
                const __m256i value4 =
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 4 * src_stride));
                const __m256i value5 =
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 5 * src_stride));
                const __m256i value6 =
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 6 * src_stride));
                const __m256i value7 =
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + 7 * src_stride));

                if (aligned_dst) {
                    _mm256_store_si256(reinterpret_cast<__m256i*>(dst + 0 * dst_stride), value0);
                    _mm256_store_si256(reinterpret_cast<__m256i*>(dst + 1 * dst_stride), value1);
                    _mm256_store_si256(reinterpret_cast<__m256i*>(dst + 2 * dst_stride), value2);
                    _mm256_store_si256(reinterpret_cast<__m256i*>(dst + 3 * dst_stride), value3);
                    _mm256_store_si256(reinterpret_cast<__m256i*>(dst + 4 * dst_stride), value4);
                    _mm256_store_si256(reinterpret_cast<__m256i*>(dst + 5 * dst_stride), value5);
                    _mm256_store_si256(reinterpret_cast<__m256i*>(dst + 6 * dst_stride), value6);
                    _mm256_store_si256(reinterpret_cast<__m256i*>(dst + 7 * dst_stride), value7);
                } else {
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + 0 * dst_stride), value0);
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + 1 * dst_stride), value1);
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + 2 * dst_stride), value2);
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + 3 * dst_stride), value3);
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + 4 * dst_stride), value4);
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + 5 * dst_stride), value5);
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + 6 * dst_stride), value6);
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + 7 * dst_stride), value7);
                }

                src += unroll * src_stride;
                dst += unroll * dst_stride;
            }

            for (; freq < n_channels_per_packet; ++freq) {
                const __m256i value =
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src));
                if (aligned_dst) {
                    _mm256_store_si256(reinterpret_cast<__m256i*>(dst), value);
                } else {
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst), value);
                }
                src += src_stride;
                dst += dst_stride;
            }
        }

        inline void copy_rfsoc_payload(uint8_t* dst, const uint8_t* src) const {
            if (likely(num_elements_per_rfsoc == 32 && num_elements == 64)) {
                if (likely((reinterpret_cast<uintptr_t>(dst) & 0x1F) == 0)) {
                    copy_rfsoc_payload_32x64<true>(dst, src);
                } else {
                    copy_rfsoc_payload_32x64<false>(dst, src);
                }
                return;
            }

            for (uint32_t freq = 0; freq < n_channels_per_packet; ++freq) {
                std::memcpy(dst, src, num_elements_per_rfsoc);
                src += num_elements_per_rfsoc;
                dst += num_elements;
            }
        }


        inline bool check_packet_basic(struct rte_mbuf* mbuf){
            if (unlikely(mbuf == nullptr)) {

                return false;
            }

        const uint32_t pkt_len = rte_pktmbuf_pkt_len(mbuf);
        if (unlikely(pkt_len != packet_size)) {
            rx_error_total +=1;
            rx_len_error_total +=1;
            return false;
          }

        rx_packets_total +=1;
        rx_bytes_total += pkt_len;
        return true;
        };

    bool align_first_packet(uint64_t spec);

    /**
     * Check startup alignment across all RFSoC shuffle handlers in this process.
     * UINT64_MAX resets the shared state.
     * Reset only during construction, before any packet processing starts.
     * The first boundary is accepted; subsequent boundaries must match it.
     */
    inline bool check_cross_handler_alignment(uint64_t seq_num) {
        static std::mutex alignment_mutex;
        static uint64_t alignment_first_seq = std::numeric_limits<uint64_t>::max();
        static uint64_t first_alignment = 0;
        static uint64_t first_specs_per_frame = 0;
        std::lock_guard<std::mutex> alignment_lock(alignment_mutex);

        if (seq_num == std::numeric_limits<uint64_t>::max()) {
            alignment_first_seq = std::numeric_limits<uint64_t>::max();
            return true;
        }

        if (alignment_first_seq == std::numeric_limits<uint64_t>::max()) {
            alignment_first_seq = seq_num;
            first_alignment = alignment;
            first_specs_per_frame = specs_per_frame;
            DEBUG("rfsocHandlerShuffle: Port {:d} set startup alignment to {:d}", port, seq_num);
            return true;
        }

        if (alignment != first_alignment || specs_per_frame != first_specs_per_frame) {
            ERROR("rfsocHandlerShuffle: Port {:d} has alignment {:d} and spectra per frame {:d}, "
                  "expected alignment {:d} and spectra per frame {:d}",
                  port, alignment, specs_per_frame, first_alignment, first_specs_per_frame);
            return false;
        }

        if (seq_num != alignment_first_seq) {
            ERROR("rfsocHandlerShuffle: Port {:d} got startup alignment {:d}, expected {:d}",
                  port, seq_num, alignment_first_seq);
            return false;
        }

        return true;
    }

    bool advance_frame(uint64_t new_seq, bool first_time=false);

    bool copy_packet(struct rte_mbuf* mbuf);
};

inline rfsocHandlerShuffle::rfsocHandlerShuffle(kotekan::Config& config, const std::string& unique_name,
                                      kotekan::bufferContainer& buffer_container, int port) :
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
        "kotekan_dpdk_rx_error_total", unique_name, {"port"}))
    {
        out_buf = buffer_container.get_buffer(
            config.get<std::string>(unique_name, "out_buffer"));
        if (!out_buf)
            FATAL_ERROR("rfsocHandlerShuffle: Could not find output buffer {:s} for handler {:s}",
                        config.get<std::string>(unique_name, "out_buffer"), unique_name);
        out_buf->register_producer(unique_name.c_str());

        if (out_buf->metadata_pool == nullptr) {
            FATAL_ERROR("rfsocHandlerShuffle: Output buffer {:s} requires a metadata pool.",
                        out_buf->buffer_name);
        }

        const std::string out_pool_type = out_buf->metadata_pool->type_name;
        if (out_pool_type != "chartsMetadata") {
            FATAL_ERROR("rfsocHandlerShuffle: Output buffer {:s} must use chartsMetadata, got {:s}.",
                        out_buf->buffer_name, out_pool_type);
        }
        INFO("rfsocHandlerShuffle: Using chartsMetadata for output buffer {:s}.", out_buf->buffer_name);

        mask_buf = buffer_container.get_buffer(
            config.get<std::string>(unique_name, "mask_buf"));
        if (!mask_buf)
            FATAL_ERROR("rfsocHandlerShuffle: Could not find mask buffer {:s} for handler {:s}",
                        config.get<std::string>(unique_name, "mask_buf"), unique_name);
        mask_name = unique_name + "_mask";
        mask_buf->register_producer(mask_name.c_str());
        mask_buf->zero_frames();

        alignment = config.get_default<uint64_t>(unique_name, "alignment", 0);
        if (alignment == 0) {
            FATAL_ERROR("rfsocHandlerShuffle: Alignment parameter must be set and greater than zero.");
        }
        // dpdkCore constructs every handler before launching packet-processing threads.
        check_cross_handler_alignment(std::numeric_limits<uint64_t>::max());

        // Number of frames to capture before stopping, 0 = unlimited
        capture_n_frames = config.get_default<uint64_t>(unique_name, "capture_n_frames", 0);
        num_frames_captured = 0;

        first_packet = false;
        rx_bytes_last = rx_bytes_total;

        num_rfsocs = config.get_default<uint32_t>(unique_name, "num_rfsocs", 2);
        num_elements_per_rfsoc =
            config.get_default<uint32_t>(unique_name, "num_elements_per_rfsoc", 32);
        num_elements = config.get_default<uint32_t>(
            unique_name, "num_elements", num_rfsocs * num_elements_per_rfsoc);
        n_channels_per_packet =
            config.get_default<uint32_t>(unique_name, "n_channels_per_packet", 168);
        subbands_per_nic = config.get_default<uint32_t>(unique_name, "subbands_per_nic", 2);
        first_subband =
            config.get_default<uint32_t>(unique_name, "first_subband", port * subbands_per_nic);
        num_local_freq = config.get_default<uint32_t>(
            unique_name, "num_local_freq", n_channels_per_packet * subbands_per_nic);
        coarse_freq_start = config.get_default<uint32_t>(
            unique_name, "coarse_freq_start", first_subband * n_channels_per_packet);

        if (num_rfsocs == 0 || num_rfsocs > 8) {
            FATAL_ERROR("rfsocHandlerShuffle: num_rfsocs must be in the range 1..8, got {:d}.",
                        num_rfsocs);
        }
        if (subbands_per_nic == 0 || first_subband + subbands_per_nic > subbands) {
            FATAL_ERROR(
                "rfsocHandlerShuffle: Invalid subband range first_subband {:d}, subbands_per_nic {:d}.",
                first_subband, subbands_per_nic);
        }
        if (num_elements != num_rfsocs * num_elements_per_rfsoc) {
            FATAL_ERROR(
                "rfsocHandlerShuffle: num_elements {:d} must equal num_rfsocs {:d} * "
                "num_elements_per_rfsoc {:d}.",
                num_elements, num_rfsocs, num_elements_per_rfsoc);
        }
        if (num_local_freq != n_channels_per_packet * subbands_per_nic) {
            FATAL_ERROR(
                "rfsocHandlerShuffle: num_local_freq {:d} must equal n_channels_per_packet {:d} * "
                "subbands_per_nic {:d}.",
                num_local_freq, n_channels_per_packet, subbands_per_nic);
        }

        bytes_per_sb = n_channels_per_packet * num_elements_per_rfsoc;
        bytes_per_spec = num_local_freq * num_elements;
        packet_payload.resize(bytes_per_sb);
        if (payload_len != bytes_per_sb) {
            FATAL_ERROR(
                "rfsocHandlerShuffle: Packet payload length {:d} does not match expected subband size "
                "{:d}.",
                payload_len, bytes_per_sb);
        }
        if (out_buf->frame_size % bytes_per_spec != 0) {
            FATAL_ERROR(
                "rfsocHandlerShuffle: Output frame size {:d} is not divisible by sample size {:d}.",
                out_buf->frame_size, bytes_per_spec);
        }

        specs_per_frame = (out_buf->frame_size) / bytes_per_spec; // how many specs fit in one frame
        if (specs_per_frame == 0) {
            FATAL_ERROR("rfsocHandlerShuffle: Output frame must contain at least one spectrum.");
        }
        if (mask_buf->frame_size != specs_per_frame) {
            FATAL_ERROR(
                "rfsocHandlerShuffle: Mask frame size {:d} must match spectra per frame {:d}.",
                mask_buf->frame_size, specs_per_frame);
        }
        packets_seen.resize(specs_per_frame, 0); // 0 means no packets seen for that spec

        // Metadata frequency setup
        frame_coarse_freq.resize(num_local_freq);
        for (uint32_t i = 0; i < num_local_freq; ++i) {
            frame_coarse_freq[i] = static_cast<int>(coarse_freq_start + i);
        }
        const uint32_t expected_packets_per_spec = subbands_per_nic * num_rfsocs;
        if (expected_packets_per_spec > 8) {
            FATAL_ERROR(
                "rfsocHandlerShuffle: packets per spectrum {:d} exceeds the 8-bit tracking mask.",
                expected_packets_per_spec);
        }
        complete_packet_mask = static_cast<uint8_t>((1u << expected_packets_per_spec) - 1u);
        packet_layout.fill({});
        for (uint32_t subband_slot = 0; subband_slot < subbands_per_nic; ++subband_slot) {
            for (uint32_t rf_id = 0; rf_id < num_rfsocs; ++rf_id) {
                const uint32_t bit_index = subband_slot * num_rfsocs + rf_id;
                packet_layout[bit_index].bit = uint8_t(1u << bit_index);
                packet_layout[bit_index].freq_base_offset =
                    subband_slot * n_channels_per_packet * num_elements;
                packet_layout[bit_index].antenna_output_offset =
                    (num_rfsocs - 1 - rf_id) * num_elements_per_rfsoc;
            }
        }
        INFO(
            "rfsocHandlerShuffle: Port {:d} first_subband {:d}, coarse_freq_start {:d}, "
            "layout {:d} freq x {:d} elements, {:d} packets/spec.",
            port, first_subband, coarse_freq_start, num_local_freq, num_elements,
            expected_packets_per_spec);

        start_time = std::chrono::steady_clock::now();

} ;


inline int rfsocHandlerShuffle::handle_packet(struct rte_mbuf* mbuf) {

    // Warmup period handling, we discard all packets during the warmup time to allow the system to stabilize, and
    // then start processing packets after that.
    if (in_warmup) {
        const auto now = std::chrono::steady_clock::now();
        const double elapsed_time =
            std::chrono::duration_cast<std::chrono::duration<double>>(now - start_time).count();
        if (elapsed_time >= warmup_time) {
            in_warmup = false;
            last_status_message_time = 0.0;
            rx_packets_last = rx_packets_total;
            rx_lost_packets_last = rx_lost_packets_total;
            rx_bytes_last = rx_bytes_total;
        } else {
            return 0; // discard packets during warmup
        }
    }

    // Basic packet checks
    if (unlikely(!check_packet_basic(mbuf))) return 0;

    // Extract sequence number, subband, and timestamp from the packet header.
    // The sequence number is the spectrum ID; all packets for the same spectrum
    // share it across subbands and RFSoCs.
    const uint8_t* pkt = rte_pktmbuf_mtod(mbuf, const uint8_t*);
    cur_seq = extract_seq_le64(pkt + ETH_IP_UDP_HDR);
    subband = extract_subband_le8(pkt + ETH_IP_UDP_HDR + 8);
    rfsoc_id = extract_rfsoc_id(pkt + ETH_IP_UDP_HDR + rfsoc_id_header_offset);
    //DEBUG("rfsocHandlerShuffle: Packet seq: {:d}, subband: {:d}", cur_seq, subband);

    cur_spec = cur_seq;

    // Extract timestamp fields for metadata and status logging.
    timestamp_sec = extract_timestamp_le32(pkt + ETH_IP_UDP_HDR + 13);
    spectra_count = extract_timestamp_le32(pkt + ETH_IP_UDP_HDR + 9);
    timestamp_micro =
        (static_cast<uint64_t>(spectra_count) * 1000000ULL) / 300000ULL;
    time0_fpga = ((uint64_t)timestamp_sec) * 1000000ULL + ((uint64_t)timestamp_micro);
    last_timestamp_sec = timestamp_sec;
    last_timestamp_micro = timestamp_micro;
    last_spectra_id = cur_spec;

    // Check for valid subband number
    if (unlikely(subband >= subbands)) {
        INFO("rfsocHandlerShuffle: Invalid subband number {:d} in packet. Discarding packet.", subband);
        rx_error_total +=1;
        return 0;
    }
    if (unlikely(subband < first_subband || subband >= first_subband + subbands_per_nic)) {
        INFO(
            "rfsocHandlerShuffle: Port {:d} received subband {:d} outside configured range "
            "[{:d}, {:d}). Discarding packet.",
            port, subband, first_subband, first_subband + subbands_per_nic);
        rx_error_total += 1;
        return 0;
    }
    if (unlikely(rfsoc_id >= num_rfsocs)) {
        INFO("rfsocHandlerShuffle: Invalid RFSoC id {:d} in packet. Discarding packet.", rfsoc_id);
        rx_error_total += 1;
        return 0;
    }

    // If we haven't got the first packet yet, we need to align to the first packet that is a
    // multiple of the alignment parameter.
    if (unlikely(!got_first_packet)) {
        if (!align_first_packet(cur_spec)) return 0;
    }

    // Copy the packet data to the output buffer. Packet loss is tracked by the
    // per-spectrum completion mask because both RFSoCs can send the same seq.
    if (unlikely(!copy_packet(mbuf))) {
        return -1;
    }

    if (cur_seq > last_seq) {
        last_seq = cur_seq;
    }
    return 0;
}

// Select an alignment boundary and validate it against the other handlers before
// opening the first output frame. Missing spectra before the received packet
// remain flagged in the frame's mask.
inline bool rfsocHandlerShuffle::align_first_packet(uint64_t spec) {

    if (alignment == 0){
        FATAL_ERROR("rfsocHandlerShuffle: Alignment parameter must be set and greater than zero.");
    }

    if ((spec % alignment) <= 1000) {
            const uint64_t start_spec = spec - (spec % alignment);
            INFO("rfsocHandlerShuffle: Port {:d} received spectrum {:d}, selected start spectrum {:d}",
                 port, spec, start_spec);

            if (!check_cross_handler_alignment(start_spec)) {
                FATAL_ERROR("rfsocHandlerShuffle: Port {:d} failed startup alignment between handlers, "
                            "closing kotekan!", port);
                return false;
            }

            last_seq = cur_seq;
            got_first_packet = true;

            if (unlikely(!advance_frame(start_spec, true))) {
                got_first_packet = false;
                return false;
            }
            return true;
        }
        return false;
    }


// This function do the change of the active frame: it marks the current frame as full and moves to the next one,
// and also handles the mask buffer for lost samples.
inline bool rfsocHandlerShuffle::advance_frame(uint64_t new_spec, bool first_time) {

    struct timeval now;
    gettimeofday(&now, nullptr);

    if (!first_time) {
        // Check for lost samples in the previous frame.
        for (uint64_t i =0; i < specs_per_frame; i++) {
            const uint8_t missing_packet_mask = complete_packet_mask & ~packets_seen[i];
            if (missing_packet_mask == 0) {
                mask_frame[i] = 0;
            } else {
                mask_frame[i] = 1;
                rx_lost_samples_total +=1;
                rx_lost_packets_total += count_bits(missing_packet_mask);
            }
        }

        //DEBUG("advance_frame: closing frame_id {} start_spec {} valid_specs_total {}", out_frame_id, frame_start_spec, valid_specs_total);

        // Mark the current output frame as full and move to the next one
        out_buf->mark_frame_full(unique_name, out_frame_id);
        out_frame_id = (out_frame_id + 1) % out_buf->num_frames; // move to the next frame

        // Mask buffer handling
        mask_buf->mark_frame_full(mask_name.c_str(), mask_frame_id);
        mask_frame_id = (mask_frame_id + 1) % mask_buf->num_frames;

        num_frames_captured++;
    }

    std::fill(packets_seen.begin(), packets_seen.end(), 0); // reset tracking for the new frame
    valid_specs_in_frame = 0;

    // Check if we have reached the configured number of frames to capture
    if (capture_n_frames != 0 && num_frames_captured >= capture_n_frames) {
        INFO("rfsocHandlerShuffle: Reached the configured number of frames to capture ({:d}). Stopping capture.",
             capture_n_frames);
        return false; // stop capturing
    }

    // Wait for the new output frame to be empty and get its pointer
    out_frame = out_buf->wait_for_empty_frame(unique_name, out_frame_id);
    if (out_frame == nullptr) {
        return false;
    }

    mask_frame = mask_buf->wait_for_empty_frame(mask_name.c_str(), mask_frame_id);
    if (mask_frame == nullptr) {
        return false;
    }

    // Initialize the new frame (zero it out)
    std::memset(mask_frame, 1, specs_per_frame); // set all to missing
    frame_start_spec = new_spec;

    // Set metadata values
    out_buf->allocate_new_metadata_object(out_frame_id);
    auto out_meta_obj = out_buf->get_metadata(out_frame_id);

    auto charts_meta = std::dynamic_pointer_cast<chartsMetadata>(out_meta_obj);
    if (!charts_meta) {
        FATAL_ERROR("rfsocHandlerShuffle: Failed to cast output metadata to chartsMetadata.");
        return false;
    }

    charts_meta->set_first_packet_recv_time(now); // the time when the first packet of the frame is received
    charts_meta->set_fpga_seq_num(frame_start_spec); // the frame start spec
    charts_meta->set_time_downsampling_fpga(1);
    charts_meta->set_coarse_freq(frame_coarse_freq);
    charts_meta->set_lost_timesamples(0);
    charts_meta->set_time0_fpga(time0_fpga); // Time0 when the FPGA sent the first packet
    charts_meta->dims = 3;
    charts_meta->type = kotekan::int4x2;
    std::strncpy(charts_meta->dim_name[0], "T", sizeof charts_meta->dim_name[0]); // Time dimension
    std::strncpy(charts_meta->dim_name[1], "F", sizeof charts_meta->dim_name[1]); // Frequency dimension
    std::strncpy(charts_meta->dim_name[2], "E", sizeof charts_meta->dim_name[2]); // Element dimension

    // Lost samples buffer metadata
    if (mask_buf->metadata_pool != nullptr && mask_buf->metadata_pool->type_name == "chartsMetadata") {
        mask_buf->allocate_new_metadata_object(mask_frame_id);
        auto mask_meta = std::dynamic_pointer_cast<chartsMetadata>(mask_buf->get_metadata(mask_frame_id));
        if (mask_meta) {
            mask_meta->set_fpga_seq_num(frame_start_spec);
            mask_meta->set_time_downsampling_fpga(1);
            mask_meta->set_first_packet_recv_time(now);
        }
    }

    return true;
}

// This function copies the packet data from the mbuf to the output frame in the correct location based
// on the sequence number and subband.
inline bool rfsocHandlerShuffle::copy_packet(struct rte_mbuf* mbuf) {

    // The packet sequence number is already the spectrum ID.
    const uint64_t spec_id = cur_spec;

    // If the spec_id is less than the frame_start_spec, then this packet is out of order and belongs
    // to a previous frame, so we discard it.
    if (spec_id < frame_start_spec) {
        rx_out_of_order_total++;
        return true;
    }

    // Location in specs relative to the start of the frame
    uint64_t spec_loc = spec_id - frame_start_spec;

    // Keep every frame on the original sequence grid, including completely
    // missing frames. advance_frame finalizes their masks and enforces the
    // capture limit before opening another frame.
    while (spec_loc >= specs_per_frame) {
        if (!advance_frame(frame_start_spec + specs_per_frame)) return false;
        spec_loc = spec_id - frame_start_spec;
    }

    const uint32_t subband_slot = subband - first_subband;
    const uint32_t bit_index = subband_slot * num_rfsocs + rfsoc_id;
    const PacketLayout& layout = packet_layout[bit_index];
    const size_t spec_offset = spec_loc * bytes_per_spec;

    // Track received packets to the mask buffer.
    uint8_t& seen = packets_seen[spec_loc];
    const uint8_t bit = layout.bit;
    if (seen & bit) {
        return true;
    }

    // Check if the spectrum offset is within the bounds of the output frame.
    if (unlikely(spec_offset + bytes_per_spec > out_buf->frame_size)) {
        rx_error_total +=1;
        return true; //out of frame bounds
    }

    const uint8_t* payload = nullptr;
    if (likely(mbuf->data_len >= payload_offset + bytes_per_sb)) {
        payload = rte_pktmbuf_mtod_offset(mbuf, const uint8_t*, payload_offset);
    } else {
        payload = static_cast<const uint8_t*>(
            rte_pktmbuf_read(mbuf, payload_offset, bytes_per_sb, packet_payload.data()));
        if (unlikely(payload == nullptr)) {
            rx_error_total += 1;
            return true;
        }
    }

    uint8_t* dst =
        &out_frame[spec_offset + layout.freq_base_offset + layout.antenna_output_offset];
    copy_rfsoc_payload(dst, payload);

    seen |= bit;
    if (seen == complete_packet_mask) {
        mask_frame[spec_loc] = 0;
        valid_specs_in_frame++;
        valid_specs_total++;
        rx_samples_total += 1;

    }
    return true;
}


inline void rfsocHandlerShuffle::update_stats() {
    std::vector<std::string> port_label = {std::to_string(port)};
    rx_packets_total_metric.labels(port_label).set(rx_packets_total);
    rx_samples_total_metric.labels(port_label).set(rx_samples_total);
    rx_lost_packets_total_metric.labels(port_label).set(rx_lost_packets_total);
    rx_lost_samples_total_metric.labels(port_label).set(rx_lost_samples_total);

    rx_bytes_total_metric.labels(port_label).set(rx_bytes_total);
    rx_error_total_metric.labels(port_label).set(rx_error_total);

    double time_now = e_time();
    const double status_cadence = 1.0; // seconds

    if (in_warmup) {
        const auto now = std::chrono::steady_clock::now();
        const double elapsed_time =
            std::chrono::duration_cast<std::chrono::duration<double>>(now - start_time).count();

        if (elapsed_time < warmup_time) {
            if ((time_now - last_status_message_time) > status_cadence) {
                const double remaining_time = warmup_time - elapsed_time;
                INFO(
                    "rfsocHandlerShuffle: Warmup in progress ({:.2f} s remaining).",
                    remaining_time);
                last_status_message_time = time_now;
            }
            return;
        }

        in_warmup = false;
        INFO("rfsocHandlerShuffle: Warmup period of {:.2f} seconds ended. Starting capture.", warmup_time);
        last_status_message_time = 0.0;
    }

    if (last_status_message_time == 0.0) {
        last_status_message_time = time_now;
        rx_packets_last = rx_packets_total;
        rx_lost_packets_last = rx_lost_packets_total;
        rx_bytes_last = rx_bytes_total;

        struct rte_eth_stats eth_stats;
        std::memset(&eth_stats, 0, sizeof eth_stats);
        if (rte_eth_stats_get(port, &eth_stats) == 0) {
            dpdk_ipackets_last = eth_stats.ipackets;
            dpdk_imissed_last = eth_stats.imissed;
            dpdk_ierrors_last = eth_stats.ierrors;
            dpdk_rx_nombuf_last = eth_stats.rx_nombuf;
        }
        return;
    }

    if ((time_now - last_status_message_time) > status_cadence) {

        const uint64_t d_packets = rx_packets_total - rx_packets_last;
        const uint64_t d_lost = rx_lost_packets_total - rx_lost_packets_last;
        const uint64_t d_bytes =rx_bytes_total - rx_bytes_last;
        const double dt = time_now - last_status_message_time;

        const double mbps = (double)d_bytes * 8.0 / (dt * 1e6);

        INFO(
            "RFSoC port {:d} | RX pkt {:d} (+{:d}) | lost {:d} (+{:d}) | "
            " {:.2f} Mb/s | Samples {:d}",
            port,
            rx_packets_total, d_packets,
            rx_lost_packets_total, d_lost,
            mbps,
            rx_samples_total
        );

        INFO(
            "Last timestamp: sec {:d} micro {:d} | spectra_id {:d} | Frame start spec {:d}",
            last_timestamp_sec, last_timestamp_micro, last_spectra_id, frame_start_spec);

        struct rte_eth_stats eth_stats;
        std::memset(&eth_stats, 0, sizeof eth_stats);
        if (rte_eth_stats_get(port, &eth_stats) == 0) {
            INFO(
                "DPDK port {:d} stats | ipackets {:d} (+{:d}) | imissed {:d} (+{:d}) | "
                "ierrors {:d} (+{:d}) | rx_nombuf {:d} (+{:d})",
                port,
                eth_stats.ipackets, eth_stats.ipackets - dpdk_ipackets_last,
                eth_stats.imissed, eth_stats.imissed - dpdk_imissed_last,
                eth_stats.ierrors, eth_stats.ierrors - dpdk_ierrors_last,
                eth_stats.rx_nombuf, eth_stats.rx_nombuf - dpdk_rx_nombuf_last);

            dpdk_ipackets_last = eth_stats.ipackets;
            dpdk_imissed_last = eth_stats.imissed;
            dpdk_ierrors_last = eth_stats.ierrors;
            dpdk_rx_nombuf_last = eth_stats.rx_nombuf;
        } else {
            WARN("rfsocHandlerShuffle: Failed to read DPDK stats for port {:d}.", port);
        }

        rx_packets_last = rx_packets_total;
        rx_lost_packets_last = rx_lost_packets_total;
        rx_bytes_last = rx_bytes_total;

        last_status_message_time = time_now;
    }
}
#endif
