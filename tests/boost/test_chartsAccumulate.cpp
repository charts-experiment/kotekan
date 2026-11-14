#define BOOST_TEST_MODULE "test_chartsAccumulate"

#include "Config.hpp"
#include "StageFactory.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"
#include "chartsMetadata.hpp"
#include "kotekanLoggingFixture.hpp"

#include <boost/test/included/unit_test.hpp>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

BOOST_GLOBAL_FIXTURE(kotekan_logging_fixture);

BOOST_AUTO_TEST_CASE(float32_average_skips_zero_and_nonfinite_samples) {
    chartsMetadata metadata_dummy;
    constexpr int32_t kNumFreq = 4;
    constexpr int32_t kFramesPerGroup = 4;

    auto pool = metadataPool::create(16, sizeof(chartsMetadata), "charts_pool", "chartsMetadata");
    Buffer in_buf(4, kNumFreq * sizeof(float), pool, "in", "standard", 0, false, false,
                  std::vector<int>{}, true);
    Buffer out_buf(4, kNumFreq * sizeof(float), pool, "out", "standard", 0, false, false,
                   std::vector<int>{}, true);

    kotekan::bufferContainer buffers;
    buffers.add_buffer(in_buf.buffer_name, &in_buf);
    buffers.add_buffer(out_buf.buffer_name, &out_buf);

    in_buf.register_producer("test");
    out_buf.register_consumer("test");

    kotekan::Config config;
    nlohmann::json cfg = {
        {"cpu_affinity", {0}},
        {"log_level", "ERROR"},
        {"accum",
         {{"in_buf", "in"},
          {"out_buf", "out"},
          {"data_type", "float32"},
          {"average", true},
          {"num_frames_to_accumulate", kFramesPerGroup}}},
    };
    config.update_config(cfg);

    auto makers = kotekan::StageFactoryRegistry::get_registered_stages();
    std::unique_ptr<kotekan::Stage> stage(makers.at("chartsAccumulate")->create(config, "/accum", buffers));
    stage->start();

    auto push_group = [&](int group_idx, const std::vector<std::vector<float>>& group_vals) {
        for (int32_t f_idx = 0; f_idx < kFramesPerGroup; ++f_idx) {
            const int slot = (group_idx * kFramesPerGroup + f_idx) % in_buf.num_frames;
            BOOST_REQUIRE(in_buf.wait_for_empty_frame("test", slot));
            in_buf.allocate_new_metadata_object(slot);
            auto meta = get_charts_metadata(&in_buf, slot);
            meta->dims = 1;
            meta->offset = 0;
            meta->dim[0] = kNumFreq;
            meta->stride[0] = 1;
            std::strncpy(meta->name, "beam_power", CHARTS_META_MAX_DIMNAME);
            std::strncpy(meta->dim_name[0], "freq", CHARTS_META_MAX_DIMNAME);
            meta->type = kotekan::float32;
            std::memcpy(in_buf.frames[slot], group_vals[f_idx].data(), kNumFreq * sizeof(float));
            in_buf.mark_frame_full("test", slot);
        }
    };

    const float nan_v = std::numeric_limits<float>::quiet_NaN();
    const float inf_v = std::numeric_limits<float>::infinity();

    // Group 1: partial zero-blanked (ch0: 2/4), all zero-blanked (ch1: 0/4), non-finite (ch2: 2/4), clean (ch3: 4/4)
    push_group(0, {
        {10.0f, 0.0f, 5.0f, 4.0f},
        {0.0f, 0.0f, nan_v, 4.0f},
        {20.0f, 0.0f, 7.0f, 4.0f},
        {0.0f, 0.0f, inf_v, 4.0f},
    });

    int out_slot0 = 0;
    uint8_t* raw_out0 = out_buf.wait_for_full_frame("test", out_slot0);
    BOOST_REQUIRE(raw_out0 != nullptr);
    const auto* out0 = reinterpret_cast<const float*>(raw_out0);
    BOOST_CHECK_CLOSE(out0[0], 15.0f, 1e-4f);
    BOOST_CHECK_EQUAL(out0[1], 0.0f);
    BOOST_CHECK_CLOSE(out0[2], 6.0f, 1e-4f);
    BOOST_CHECK_CLOSE(out0[3], 4.0f, 1e-4f);
    out_buf.mark_frame_empty("test", out_slot0);

    // Group 2: verify valid_counts resets completely on next group
    push_group(1, {
        {0.0f, 8.0f, 1.0f, 2.0f},
        {0.0f, 0.0f, 0.0f, 6.0f},
        {0.0f, 12.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 4.0f},
    });

    int out_slot1 = (out_slot0 + 1) % out_buf.num_frames;
    uint8_t* raw_out1 = out_buf.wait_for_full_frame("test", out_slot1);
    BOOST_REQUIRE(raw_out1 != nullptr);
    const auto* out1 = reinterpret_cast<const float*>(raw_out1);
    BOOST_CHECK_EQUAL(out1[0], 0.0f);
    BOOST_CHECK_CLOSE(out1[1], 10.0f, 1e-4f);
    BOOST_CHECK_CLOSE(out1[2], 1.0f, 1e-4f);
    BOOST_CHECK_CLOSE(out1[3], 4.0f, 1e-4f);
    out_buf.mark_frame_empty("test", out_slot1);

    in_buf.send_shutdown_signal();
    out_buf.send_shutdown_signal();
    stage->stop();
    stage->join();
}
