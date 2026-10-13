#include "chartsBeamMetadata.hpp"
#include "factory.hpp"
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

REGISTER_TYPE_WITH_FACTORY(metadataObject, chartsBeamMetadata);

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}
uint64_t positive_integer(const nlohmann::json& value, bool allow_zero = false) {
    require(value.is_number_integer(), "beam metadata requires integer counts");
    if (!value.is_number_unsigned()) require(value.get<int64_t>() >= 0, "negative beam count");
    auto result = value.get<uint64_t>();
    require(allow_zero || result > 0, "zero beam count");
    return result;
}
}

void chartsBeamMetadata::validate() const {
    const auto& d = document;
    require(d.at("schema") == "charts.beam.intensity.v1", "unknown beam metadata schema");
    require(d.at("mode") == "coarse" && d.at("datatype") == "float32_le",
            "unsupported beam product mode/type");
    require(d.at("layout") == "TFB" && d.at("intensity_convention") == "sum_power_all_supplied",
            "unsupported beam product layout/convention");
    auto t = positive_integer(d.at("raw_n_time"));
    auto e = positive_integer(d.at("n_elements"));
    require(e == 32 || e == 64, "unsupported element count");
    auto k = positive_integer(d.at("integration_spectra"));
    require(k == 10 || k == 320, "unsupported coarse integration");
    auto rows = t / k + (t % k != 0);
    auto shard = positive_integer(d.at("shard_id"), true);
    require(shard < 2, "invalid shard");
    const auto& shape = d.at("shape");
    require(shape.is_array() && shape.size() == 3, "invalid beam shape");
    require(positive_integer(shape[0]) == rows && positive_integer(shape[1]) == 336,
            "inconsistent beam shape");
    auto beams = positive_integer(shape[2]);
    require(beams <= 128, "invalid beam count");
    require(d.at("strides") == nlohmann::json::array({336 * beams, beams, uint64_t(1)}),
            "invalid beam strides");
    require(rows <= std::numeric_limits<uint64_t>::max() / (336 * beams * 4), "output size overflow");
    require(positive_integer(d.at("payload_bytes")) == rows * 336 * beams * 4,
            "invalid payload byte count");
    auto seq = positive_integer(d.at("sequence_start"), true);
    require(t <= std::numeric_limits<uint64_t>::max() - seq, "sequence overflow");
    positive_integer(d.at("raw_frame_id"), true);
    positive_integer(d.at("sequence_gap_ticks"), true);
    require(d.at("tick_period_us") == nlohmann::json::array({10, 3}), "invalid tick period");
    require(positive_integer(d.at("timestamp_step_ticks")) == k, "invalid time step");
    require(positive_integer(d.at("coarse_origin"), true) == shard * 336, "invalid coarse origin");
    for (const auto key : {"frequency_first_hz", "frequency_step_hz"}) {
        require(d.at(key).is_number() && std::isfinite(d.at(key).get<double>()), "invalid frequency grid");
    }
    require(d.at("frequency_step_hz").get<double>() > 0, "invalid frequency spacing");
    require(d.at("history") == "none" && d.at("voltage_gain") == 1, "invalid coarse history/gain");
    require(d.at("weights_layout") == "BFE_complex_float32_le", "invalid weight layout");
    for (const auto key : {"weights_id", "beam_order_id"})
        require(d.at(key).is_string() && !d.at(key).get<std::string>().empty(), "missing weight identity");
    require(d.at("validity_layout") == "T_integrated_broadcast_frequency_beam",
            "invalid validity layout");
    const auto& supplied = d.at("supplied_spectra");
    const auto& valid = d.at("valid_spectra");
    require(supplied.is_array() && valid.is_array() && supplied.size() == rows && valid.size() == rows,
            "invalid validity lengths");
    for (size_t w = 0; w < rows; ++w) {
        auto count = positive_integer(supplied[w]);
        require(count == std::min(k, t - w * k), "invalid supplied count");
        require(positive_integer(valid[w], true) <= count, "invalid valid count");
    }
}

void chartsBeamMetadata::deepCopy(std::shared_ptr<const metadataObject> other) {
    auto beam = std::dynamic_pointer_cast<const chartsBeamMetadata>(other);
    if (!beam) throw std::invalid_argument("incompatible beam metadata copy");
    beam->validate();
    document = beam->document;
}
size_t chartsBeamMetadata::get_serialized_size() {
    validate();
    const auto size = document.dump().size();
    if (size > 4 * 1024 * 1024) throw std::length_error("beam metadata exceeds 4 MiB");
    return size;
}
size_t chartsBeamMetadata::serialize(char* bytes) {
    const auto size = get_serialized_size();
    if (!bytes) throw std::invalid_argument("null metadata destination");
    const auto text = document.dump();
    std::memcpy(bytes, text.data(), size);
    return size;
}
size_t chartsBeamMetadata::set_from_bytes(const char* bytes, size_t length) {
    if (!bytes || !length || length > 4 * 1024 * 1024)
        throw std::invalid_argument("invalid beam metadata byte range");
    chartsBeamMetadata candidate;
    candidate.document = nlohmann::json::parse(bytes, bytes + length);
    candidate.validate();
    document = std::move(candidate.document);
    return length;
}
nlohmann::json chartsBeamMetadata::to_json() {
    validate();
    return document;
}
