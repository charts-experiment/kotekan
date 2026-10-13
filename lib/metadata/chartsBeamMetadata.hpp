#ifndef CHARTS_BEAM_METADATA_HPP
#define CHARTS_BEAM_METADATA_HPP

#include "metadata.hpp"
#include "json.hpp"

// Versioned, portable JSON metadata for standard-buffer beam intensity products.
// rawFileWrite supplies the uint32 byte-length prefix; this class serializes the
// JSON bytes only. No C++ struct layout or native-endian numeric fields on disk.
class chartsBeamMetadata : public metadataObject {
public:
    nlohmann::json document;
    void validate() const;
    void deepCopy(std::shared_ptr<const metadataObject> other) override;
    size_t get_serialized_size() override;
    size_t serialize(char* bytes) override;
    size_t set_from_bytes(const char* bytes, size_t length) override;
    nlohmann::json to_json() override;
};

#endif
