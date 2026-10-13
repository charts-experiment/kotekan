#ifndef CHARTS_METADATA
#define CHARTS_METADATA

#include "DataType.hpp"
#include "NDArray.hpp"
#include "buffer.hpp"
#include "chartsConstants.hpp"
#include "kotekanLogging.hpp"
#include "metadata.hpp"

#include "json.hpp"
#include "jsonMetadata.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string.h>
#include <string>
#include <vector>

// Maximum number of frequencies in metadata array
const int CHARTS_META_MAX_FREQ =
    static_cast<int>(kotekan::charts::constants::charts_full_band_channels);

// Maximum number of dimensions for arrays
const int CHARTS_META_MAX_DIM = 8;

// Maximum length of dimension names for arrays
const int CHARTS_META_MAX_DIMNAME = 20;

class chartsMetadata : public metadataObject {
public:
    chartsMetadata();

    /// copy object
    void deepCopy(std::shared_ptr<const metadataObject> other) override;

    /// Returns the size of objects of this type when serialized into bytes.
    size_t get_serialized_size() override;

    /// Sets this metadata object's values from the given byte array
    /// of the given length.  Returns the number of bytes consumed.
    size_t set_from_bytes(const char* bytes, size_t length) override;

    /// Serializes this metadata object into the given byte array,
    /// expected to be of length (at least) get_serialized_size().
    size_t serialize(char* bytes) override;

    /// serialize to json
    nlohmann::json to_json() override;

    /// Validates that this metadata's array structure (name, type, dimensions, strides) matches
    /// the given frame descriptor, issuing a non-fatal error for any inconsistencies.
    void check_frame_desc(const std::shared_ptr<const kotekan::GenericNDArray>& frame_desc) const;

    /// Copies array structure information (type, dimensions, dimension names, extents, strides)
    /// from the given frame descriptor into this metadata object. Does not copy the array name.
    void set_from_frame_desc(const std::shared_ptr<const kotekan::GenericNDArray>& frame_desc);

    mutable class almost_copyable_mutex : public std::mutex {
    public:
        almost_copyable_mutex() : std::mutex() {}
        almost_copyable_mutex(const almost_copyable_mutex&) : std::mutex() {}
        almost_copyable_mutex operator=(const almost_copyable_mutex&) {
            return *this;
        }
    } lock;

    // ---- array description ----
    char name[CHARTS_META_MAX_DIMNAME]; // "E", "J", "I", etc
    kotekan::DataType type;

    int dims;
    int dim[CHARTS_META_MAX_DIM];
    char dim_name[CHARTS_META_MAX_DIM][CHARTS_META_MAX_DIMNAME]; // "F", "T", "D", etc
    int64_t stride[CHARTS_META_MAX_DIM]; // The stride counts elements, not bytes
    int64_t offset;                      // The offset counts elements, not bytes
    void set_name(const std::string& name) {
        // Manually copying in a for loop to avoid possibly buggy GCC warning
        // about array bounds and stringop-truncation.

        int len = name.length() < CHARTS_META_MAX_DIMNAME ? name.length() : CHARTS_META_MAX_DIMNAME;

        for (int i = 0; i < len; i++)
            this->name[i] = name[i];
        // Fill the remaining space with 0s
        for (int i = len; i < CHARTS_META_MAX_DIMNAME; i++)
            this->name[i] = '\0';
    }
    std::string get_name() const {
        return std::string(name, strnlen(name, CHARTS_META_MAX_DIMNAME));
    }

    void set_array_dimension(int dim, int size, const std::string& name) {
        assert(dim < CHARTS_META_MAX_DIM);
        this->dim[dim] = size;
        // GCC helpfully tries to warn us that the destination string may end up not
        // NUL-terminated, which we know.
#pragma GCC diagnostic push
#if GCC_VERSION > 80000
#pragma GCC diagnostic ignored "-Wstringop-truncation"
#endif
        strncpy(this->dim_name[dim], name.c_str(), CHARTS_META_MAX_DIMNAME);
#pragma GCC diagnostic pop
    }
    void set_strides_simple() {
        // Compute the strides from the set dims assuming simple contiguous
        // access.
        assert(this->dims >= 0);
        std::ptrdiff_t np = 1;
        for (int d = this->dims - 1; d >= 0; --d) {
            this->stride[d] = np;
            assert(this->dim[d] >= 0);
            np *= this->dim[d];
        }
    }

    /// FPGA sequence number ///
    void set_fpga_seq_num(const int64_t fpga_seq_num) {
        std::lock_guard<std::mutex> lock(this->lock);
        metadata[jsonMetadata::FPGA_SEQ_NUM] = fpga_seq_num;
    }

    bool has_fpga_seq_num() const {
        std::lock_guard<std::mutex> lock(this->lock);
        return metadata.contains(jsonMetadata::FPGA_SEQ_NUM);
    }

    int64_t get_fpga_seq_num() const {
        std::lock_guard<std::mutex> lock(this->lock);
        return metadata.at(jsonMetadata::FPGA_SEQ_NUM).template get<int64_t>();
    }

    /// Frame Counter ///
    void set_frame_counter(const int frame_counter) {
        std::lock_guard<std::mutex> lock(this->lock);
        metadata[jsonMetadata::FRAME_COUNTER] = frame_counter;
    }

    bool has_frame_counter() const {
        std::lock_guard<std::mutex> lock(this->lock);
        return metadata.contains(jsonMetadata::FRAME_COUNTER);
    }

    int get_frame_counter() const {
        std::lock_guard<std::mutex> lock(this->lock);
        return metadata.at(jsonMetadata::FRAME_COUNTER).template get<int>();
    }

    // Time downsampling -- the factor by which the time samples have
    // been downsampled relative to FPGA samples.
    void set_time_downsampling_fpga(const int time_downsampling_fpga) {
        std::lock_guard<std::mutex> lock(this->lock);
        metadata[jsonMetadata::TIME_DOWNSAMPLING_FPGA] = time_downsampling_fpga;
    }

    bool has_time_downsampling_fpga() const {
        std::lock_guard<std::mutex> lock(this->lock);
        return metadata.contains(jsonMetadata::TIME_DOWNSAMPLING_FPGA);
    }

    int get_time_downsampling_fpga() const {
        std::lock_guard<std::mutex> lock(this->lock);
        return metadata.at(jsonMetadata::TIME_DOWNSAMPLING_FPGA);
    }

    /// Frequencies functions ///

    // Save the vector of index of frequencies (e.g. what channels are in this node)
    void set_coarse_freq(const std::vector<int>& coarse_freq) {
        std::lock_guard<std::mutex> lock(this->lock);
        assert(coarse_freq.size() <= CHARTS_META_MAX_FREQ);
        metadata[jsonMetadata::COARSE_FREQ] = coarse_freq;
    }

    // Check if the metadata contains the coarse frequency information
    bool has_coarse_freq() const {
        std::lock_guard<std::mutex> lock(this->lock);
        return metadata.contains(jsonMetadata::COARSE_FREQ);
    }

    // Return the vector of frequency indices (e.g. what channels are in this node)
    std::vector<int> get_coarse_freq() const {
        std::lock_guard<std::mutex> lock(this->lock);
        return metadata.at(jsonMetadata::COARSE_FREQ).template get<std::vector<int>>();
    }

    // Return the number of frequencies in this metadata (size of the coarse frequency vector)
    int get_nfreq() const {
        std::lock_guard<std::mutex> lock(this->lock);
        return static_cast<int>(metadata.at(jsonMetadata::COARSE_FREQ).size());
    }

    // First packet receive time -- the system time when the first packet in the frame was captured
    void set_first_packet_recv_time(const timeval time_v) {
        std::lock_guard<std::mutex> lock(this->lock);
        metadata[jsonMetadata::FIRST_PACKET_RECV_TIME] = time_v;
    }

    bool has_first_packet_recv_time() const {
        std::lock_guard<std::mutex> lock(this->lock);
        return metadata.contains(jsonMetadata::FIRST_PACKET_RECV_TIME);
    }

    timeval get_first_packet_recv_time() const {
        std::lock_guard<std::mutex> lock(this->lock);
        return metadata.at(jsonMetadata::FIRST_PACKET_RECV_TIME).template get<timeval>();
    }

    // Time0 FPGA
    void set_time0_fpga(const int64_t time0_fpga) {
        std::lock_guard<std::mutex> lock(this->lock);
        metadata[jsonMetadata::TIME0_FPGA] = time0_fpga;
    }

    bool has_time0_fpga() const {
        std::lock_guard<std::mutex> lock(this->lock);
        return metadata.contains(jsonMetadata::TIME0_FPGA);
    }

    int64_t get_time0_fpga() const {
        std::lock_guard<std::mutex> lock(this->lock);
        return metadata.at(jsonMetadata::TIME0_FPGA).template get<int64_t>();
    }


    void set_lost_timesamples(int32_t x);
    bool has_lost_timesamples() const;
    int32_t get_lost_timesamples() const;

private:
    jsonMetadata::metadata metadata;

    chartsMetadata& operator=(const chartsMetadata&) = default;
    chartsMetadata(const chartsMetadata&) = default;

    friend void to_json(nlohmann::json& j, const chartsMetadata& m);
    friend void from_json(const nlohmann::json& j, chartsMetadata& m);
};


void to_json(nlohmann::json& j, const chartsMetadata& m);
void from_json(const nlohmann::json& j, chartsMetadata& m);

inline bool metadata_is_charts(Buffer* buf, int) {
    return buf && buf->metadata_pool && (buf->metadata_pool->type_name == "chartsMetadata");
}

inline bool metadata_is_charts(const std::shared_ptr<metadataObject>& mc) {
    if (!mc)
        return false;
    std::shared_ptr<metadataPool> pool = mc->parent_pool.lock();
    assert(pool);
    return (pool->type_name == "chartsMetadata");
}

inline bool metadata_is_charts(const std::shared_ptr<const metadataObject>& mc) {
    if (!mc)
        return false;
    std::shared_ptr<metadataPool> pool = mc->parent_pool.lock();
    assert(pool);
    return (pool->type_name == "chartsMetadata");
}

inline std::shared_ptr<chartsMetadata>
get_charts_metadata(const std::shared_ptr<metadataObject>& mc) {
    if (!mc)
        return std::shared_ptr<chartsMetadata>();
    if (!metadata_is_charts(mc)) {
        std::shared_ptr<metadataPool> pool = mc->parent_pool.lock();
        WARN_NON_OO("Expected metadata to be type \"chartsMetadata\", got \"{:s}\".",
                    pool->type_name);
        return std::shared_ptr<chartsMetadata>();
    }
    return std::static_pointer_cast<chartsMetadata>(mc);
}

inline std::shared_ptr<const chartsMetadata>
get_charts_metadata(const std::shared_ptr<const metadataObject>& mc) {
    if (!mc)
        return std::shared_ptr<const chartsMetadata>();
    if (!metadata_is_charts(mc)) {
        std::shared_ptr<const metadataPool> pool = mc->parent_pool.lock();
        WARN_NON_OO("Expected metadata to be type \"chartsMetadata\", got \"{:s}\".",
                    pool->type_name);
        return std::shared_ptr<const chartsMetadata>();
    }
    return std::static_pointer_cast<const chartsMetadata>(mc);
}

inline std::shared_ptr<chartsMetadata> get_charts_metadata(Buffer* buf, int frame_id) {
    if (!buf || frame_id < 0 || frame_id >= (int)buf->metadata.size())
        return std::shared_ptr<chartsMetadata>();
    std::shared_ptr<metadataObject> meta = buf->metadata.at(frame_id);
    return get_charts_metadata(meta);
}


#endif