#define BOOST_TEST_MODULE "test_charts_beam_metadata"
#include <boost/test/included/unit_test.hpp>
#include "chartsBeamMetadata.hpp"

namespace {
nlohmann::json product() {
    return {{"schema","charts.beam.intensity.v1"},{"mode","coarse"},{"datatype","float32_le"},
        {"layout","TFB"},{"shape",{2,336,2}},{"strides",{672,2,1}},{"payload_bytes",5376},
        {"raw_n_time",11},{"n_elements",64},{"integration_spectra",10},{"raw_frame_id",0},
        {"sequence_start",15360},{"sequence_gap_ticks",0},{"tick_period_us",{10,3}},
        {"timestamp_step_ticks",10},{"shard_id",1},{"coarse_origin",336},
        {"frequency_first_hz",400800000.0},{"frequency_step_hz",300000.0},
        {"weights_id","fnv1a64:0123456789abcdef"},{"weights_layout","BFE_complex_float32_le"},
        {"beam_order_id","fixture"},{"intensity_convention","sum_power_all_supplied"},
        {"history","none"},{"voltage_gain",1},
        {"validity_layout","T_integrated_broadcast_frequency_beam"},
        {"supplied_spectra",{10,1}},{"valid_spectra",{8,0}}};
}
}
BOOST_AUTO_TEST_CASE(roundtrip_and_deep_copy) {
    auto source=std::make_shared<chartsBeamMetadata>(); source->document=product();
    std::vector<char> bytes(source->get_serialized_size());
    BOOST_CHECK_EQUAL(source->serialize(bytes.data()),bytes.size());
    chartsBeamMetadata target;
    BOOST_CHECK_EQUAL(target.set_from_bytes(bytes.data(),bytes.size()),bytes.size());
    BOOST_CHECK(target.to_json()==source->to_json());
    target.deepCopy(source);
    source->document["sequence_start"]=99;
    BOOST_CHECK_EQUAL(target.document["sequence_start"].get<int>(),15360);
}
BOOST_AUTO_TEST_CASE(reject_malformed_products) {
    for(auto bad : {nlohmann::json{{"shape",{2,672,2}}}, nlohmann::json{{"valid_spectra",{11,1}}},
                    nlohmann::json{{"supplied_spectra",{10,10}}}, nlohmann::json{{"schema","v99"}},
                    nlohmann::json{{"raw_n_time",-1}}, nlohmann::json{{"strides",{1,2,672}}},
                    nlohmann::json{{"coarse_origin",0}}, nlohmann::json{{"payload_bytes",1}},
                    nlohmann::json{{"integration_spectra",16}}}) {
        chartsBeamMetadata value;value.document=product();value.document.update(bad);
        BOOST_CHECK_THROW(value.validate(), std::exception);
    }
    chartsBeamMetadata value;value.document=product();
    BOOST_CHECK_THROW(value.set_from_bytes("{",1),std::exception);
    BOOST_CHECK(value.document==product()); // failed parse leaves previous metadata intact
    BOOST_CHECK_THROW(value.set_from_bytes(nullptr,1),std::exception);
}
