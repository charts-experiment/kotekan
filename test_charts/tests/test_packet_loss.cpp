#define BOOST_TEST_MODULE "charts_packet_loss"

#include "chartsPacketLoss.hpp"
#include "zeroSamples.hpp"
#include "visUtil.hpp"

#include <boost/test/included/unit_test.hpp>
#include <algorithm>
#include <cstring>
#include <limits>
#include <numeric>
#include <sched.h>

using namespace kotekan;
using beamformer::validate_charts_mask_pair;

namespace {
void identify(chartsMetadata& meta, int64_t seq, int shard, bool frequencies = true) {
    meta.set_fpga_seq_num(seq);
    meta.set_time_downsampling_fpga(1);
    if (frequencies) {
        std::vector<int> f(336);
        std::iota(f.begin(), f.end(), shard * 336);
        meta.set_coarse_freq(f);
    }
}

struct Pair {
    chartsMetadata raw, mask;
    std::vector<uint8_t> flags = std::vector<uint8_t>(320, 0);
    Pair() { identify(raw, 1000, 0); identify(mask, 1000, 0, false); }
    int64_t validate(size_t shard = 0, size_t mask_shard = 0) {
        return validate_charts_mask_pair(&raw, &mask, flags.data(), flags.size(), 320,
                                        shard, mask_shard);
    }
};
}

BOOST_AUTO_TEST_CASE(pair_metadata_not_ring_slot_and_preserve_polarity) {
    Pair p;
    p.flags[0] = p.flags[319] = 1;
    BOOST_CHECK_EQUAL(p.validate(), 1000);
    BOOST_CHECK_EQUAL(p.flags[0], 1);
    BOOST_CHECK_EQUAL(p.flags[1], 0);
    identify(p.raw, 1000, 1);
    BOOST_CHECK_EQUAL(p.validate(1, 1), 1000);
    BOOST_CHECK_THROW(p.validate(1, 0), std::invalid_argument);
    BOOST_CHECK_THROW(p.validate(0, 0), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(reject_sequence_step_shape_and_nonbinary_flags) {
    Pair p;
    p.mask.set_fpga_seq_num(1001);
    BOOST_CHECK_THROW(p.validate(), std::invalid_argument);
    p.mask.set_fpga_seq_num(1000);
    p.mask.set_time_downsampling_fpga(32);
    BOOST_CHECK_THROW(p.validate(), std::invalid_argument);
    p.mask.set_time_downsampling_fpga(1);
    p.raw.set_time_downsampling_fpga(2);
    BOOST_CHECK_THROW(p.validate(), std::invalid_argument);
    p.raw.set_time_downsampling_fpga(1);
    p.flags.pop_back();
    BOOST_CHECK_THROW(p.validate(), std::invalid_argument);
    p.flags.push_back(2);
    BOOST_CHECK_THROW(p.validate(), std::invalid_argument);
    p.flags.back() = 255;
    BOOST_CHECK_THROW(p.validate(), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(reject_absent_metadata_and_overflow) {
    Pair p;
    chartsMetadata empty;
    BOOST_CHECK_THROW(validate_charts_mask_pair(nullptr, &p.mask, p.flags.data(), 320, 320, 0, 0),
                      std::invalid_argument);
    BOOST_CHECK_THROW(validate_charts_mask_pair(&p.raw, &empty, p.flags.data(), 320, 320, 0, 0),
                      std::invalid_argument);
    BOOST_CHECK_THROW(validate_charts_mask_pair(&empty, &p.mask, p.flags.data(), 320, 320, 0, 0),
                      std::invalid_argument);
    p.raw.set_fpga_seq_num(std::numeric_limits<int64_t>::max() - 100);
    p.mask.set_fpga_seq_num(p.raw.get_fpga_seq_num());
    BOOST_CHECK_THROW(p.validate(), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(reject_missing_or_wrong_frequency_map) {
    Pair p;
    chartsMetadata no_map;
    identify(no_map, 1000, 0, false);
    BOOST_CHECK_THROW(validate_charts_mask_pair(&no_map, &p.mask, p.flags.data(), 320, 320, 0, 0),
                      std::invalid_argument);
    p.raw.set_coarse_freq({0, 1});
    BOOST_CHECK_THROW(p.validate(), std::invalid_argument);
    identify(p.raw, 1000, 1);
    BOOST_CHECK_THROW(p.validate(), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(native_metadata_fixture_matches_actual_serializer) {
    // packet_loss.py uses this existing Linux LP64 native raw-file ABI.
    chartsMetadata meta;
    identify(meta, 12345, 1);
    BOOST_REQUIRE_EQUAL(meta.get_serialized_size(), 3032);
    std::vector<char> bytes(meta.get_serialized_size());
    BOOST_REQUIRE_EQUAL(meta.serialize(bytes.data()), bytes.size());
    int64_t sequence;
    int32_t step, nfreq, first_frequency;
    std::memcpy(&sequence, bytes.data() + 8, sizeof(sequence));
    std::memcpy(&step, bytes.data() + 16, sizeof(step));
    std::memcpy(&nfreq, bytes.data() + 312, sizeof(nfreq));
    std::memcpy(&first_frequency, bytes.data() + 316, sizeof(first_frequency));
    BOOST_CHECK_EQUAL(sequence, 12345);
    BOOST_CHECK_EQUAL(step, 1);
    BOOST_CHECK_EQUAL(nfreq, 336);
    BOOST_CHECK_EQUAL(first_frequency, 336);
}

BOOST_AUTO_TEST_CASE(original_mask_has_independent_consumers_through_wrap_and_backpressure) {
    constexpr int nt = 4;
    constexpr size_t bytes = 336 * 64;
    auto pool = metadataPool::create(32, sizeof(chartsMetadata), "direct-mask", "chartsMetadata");
    Buffer raw(3, nt * bytes, pool, "raw", "standard", 0, false, false, {}, true);
    Buffer mask(5, nt, pool, "mask", "standard", 0, false, false, {}, true);
    bufferContainer buffers;
    buffers.add_buffer("raw", &raw);
    buffers.add_buffer("mask", &mask);
    raw.register_producer("handler");
    mask.register_producer("handler");
    raw.register_consumer("h2d");
    mask.register_consumer("h2d");
    mask.register_consumer("validate");
    Config cfg;
    cfg.update_config({{"log_level", "ERROR"}, {"cpu_affinity", {sched_getcpu()}},
        {"zero", {{"out_buf", "raw"}, {"lost_samples_buf", "mask"},
                  {"sample_size", bytes}, {"zero_value", 0}}}});
    zeroSamples zero(cfg, "/zero", buffers);
    zero.start();
    // Always stop/join even if a BOOST_REQUIRE throws.
    struct Shutdown {
        zeroSamples& zero; Buffer& raw; Buffer& mask;
        ~Shutdown() {
            zero.stop(); raw.send_shutdown_signal(); mask.send_shutdown_signal(); zero.join();
        }
    } shutdown{zero, raw, mask};
    for (int frame = 0; frame < 30; ++frame) {
        int r = frame % raw.num_frames, m = frame % mask.num_frames;
        auto* data = raw.wait_for_empty_frame("handler", r);
        auto* flags = mask.wait_for_empty_frame("handler", m);
        BOOST_REQUIRE(data);
        BOOST_REQUIRE(flags);
        std::fill(data, data + raw.frame_size, 0x37);
        for (int t = 0; t < nt; ++t) flags[t] = (t + frame) % 3 == 0;
        raw.allocate_new_metadata_object(r);
        mask.allocate_new_metadata_object(m);
        auto rm = get_charts_metadata(&raw, r), mm = get_charts_metadata(&mask, m);
        identify(*rm, frame * nt, 0);
        identify(*mm, frame * nt, 0, false);
        // Match handler order. zeroSamples is a second voltage producer.
        raw.mark_frame_full("handler", r);
        mask.mark_frame_full("handler", m);
        BOOST_REQUIRE(mask.wait_for_full_frame("h2d", m));
        BOOST_REQUIRE(mask.wait_for_full_frame("validate", m));
        BOOST_CHECK_EQUAL(validate_charts_mask_pair(rm.get(), mm.get(), flags, nt, nt, 0, 0),
                          frame * nt);
        mask.mark_frame_empty("h2d", m); // Simulated copy completion before zeroing releases it.
        BOOST_REQUIRE_EQUAL(raw.wait_for_full_frame_timeout("h2d", r,
                            double_to_ts(current_time() + 5)), 0);
        // Slow validation consumer retains the original mask after other users release it.
        BOOST_CHECK(!mask.is_frame_empty(m));
        BOOST_CHECK(mask.get_metadata(m) == mm);
        for (int t = 0; t < nt; ++t) {
            const auto expected = flags[t] ? 0 : 0x37;
            BOOST_CHECK(std::all_of(data + t * bytes, data + (t + 1) * bytes,
                                   [expected](uint8_t value) { return value == expected; }));
        }
        mask.mark_frame_empty("validate", m);
        raw.mark_frame_empty("h2d", r);
    }
}
