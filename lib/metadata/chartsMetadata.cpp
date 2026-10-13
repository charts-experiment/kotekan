#include "chartsMetadata.hpp"

#include "factory.hpp"

#include <algorithm>
#include <cstring>

REGISTER_TYPE_WITH_FACTORY(metadataObject, chartsMetadata);

chartsMetadata::chartsMetadata() : type(kotekan::unknown_type), dims(-1), offset(0) {

    name[0] = '\0';

    for (int d = 0; d < CHARTS_META_MAX_DIM; ++d) {
        dim[d] = -1;
        dim_name[d][0] = '\0';
        stride[d] = -1;
    }
}

// Helper function to set the array name and dimension names, ensuring null termination
void chartsMetadata::check_frame_desc(
    const std::shared_ptr<const kotekan::GenericNDArray>& frame_desc) const {

    bool failed = false;

    if (strncmp(this->name, frame_desc->get_quantity_name().get_c_string(), sizeof(this->name))
        != 0) {
        ERROR("Names differ: {:s} != {:s}",
              std::string(this->name, strnlen(this->name, sizeof(this->name))),
              frame_desc->get_quantity_name());
        failed = true;
    }
    if (this->type != frame_desc->get_value_datatype()) {
        ERROR("Types differ for {:s}: {:s} != {:s}", frame_desc->get_quantity_name(),
              kotekan::type_to_string(this->type),
              kotekan::type_to_string(frame_desc->get_value_datatype()));
        failed = true;
    }
    if (size_t(this->dims) != frame_desc->get_rank()) {
        ERROR("Ranks differ for {:s}: {:d} != {:d}", frame_desc->get_quantity_name(), this->dims,
              frame_desc->get_rank());
        failed = true;
    }
    for (int d = this->dims - 1; d >= 0; --d) {

        if (strncmp(this->dim_name[d], frame_desc->get_dimname(d).get_c_string(),
                    sizeof this->dim_name[d])
            != 0) {
            ERROR("Dim_name[{:d}] differs for {:s}: {:s} != {:s}", d,
                  frame_desc->get_quantity_name(),
                  std::string(this->dim_name[d],
                              strnlen(this->dim_name[d], sizeof(this->dim_name[d]))),
                  frame_desc->get_dimname(d));
            failed = true;
        }

        if (this->dim[d] != frame_desc->get_extent(d)) {
            ERROR("Dim[{:d}] differs for {:s}: {:d} != {:d}", d, frame_desc->get_quantity_name(),
                  this->dim[d], frame_desc->get_extent(d));
            failed = true;
        }
        if (this->stride[d] != frame_desc->get_stride(d)) {
            ERROR("Stride[{:d}] differs for {:s}: {:d} != {:d}", d, frame_desc->get_quantity_name(),
                  this->stride[d], frame_desc->get_stride(d));
            failed = true;
        }
    }

    if (failed)
        FATAL_ERROR("Inconsistent array description between CHARTSMetadata and FrameDesc");
}
void chartsMetadata::deepCopy(std::shared_ptr<const metadataObject> other) {
    auto o = std::dynamic_pointer_cast<const chartsMetadata>(other);
    assert(o);

    if (this == o.get())
        return;

    std::scoped_lock<std::mutex, std::mutex> lock(this->lock, o->lock);
    *this = *o;
}

void chartsMetadata::set_lost_timesamples(int32_t x) {
    std::lock_guard<std::mutex> lock(this->lock);
    metadata[jsonMetadata::LOST_TIMESAMPLES] = x;
}

bool chartsMetadata::has_lost_timesamples() const {
    std::lock_guard<std::mutex> lock(this->lock);
    return metadata.contains(jsonMetadata::LOST_TIMESAMPLES);
}

int32_t chartsMetadata::get_lost_timesamples() const {
    std::lock_guard<std::mutex> lock(this->lock);
    return metadata.at(jsonMetadata::LOST_TIMESAMPLES).get<int32_t>();
}

struct chartsMetadataFormat {
    int32_t frame_counter;
    int64_t fpga_seq_num;
    int32_t time_downsampling_fpga;

    char name[CHARTS_META_MAX_DIMNAME];
    int32_t type;

    int32_t dims;
    int32_t dim[CHARTS_META_MAX_DIM];
    char dim_name[CHARTS_META_MAX_DIM][CHARTS_META_MAX_DIMNAME];
    int64_t stride[CHARTS_META_MAX_DIM];
    int64_t offset;

    int32_t nfreq;
    int32_t coarse_freq[CHARTS_META_MAX_FREQ];

    int32_t lost_timesamples;

    timeval first_packet_recv_time;

    uint64_t time0_fpga;
};

size_t chartsMetadata::get_serialized_size() {
    return sizeof(chartsMetadataFormat);
}
// Helper function to set the array name and dimension names, ensuring null termination
size_t chartsMetadata::serialize(char* bytes) {
    chartsMetadataFormat* fmt = reinterpret_cast<chartsMetadataFormat*>(bytes);
    memset(fmt, 0, sizeof(chartsMetadataFormat));

    fmt->frame_counter = has_frame_counter() ? get_frame_counter() : -1;
    fmt->fpga_seq_num = has_fpga_seq_num() ? get_fpga_seq_num() : -1;
    fmt->time_downsampling_fpga = has_time_downsampling_fpga() ? get_time_downsampling_fpga() : -1;

    memcpy(fmt->name, name, sizeof(name));
    fmt->type = static_cast<int32_t>(type);
    fmt->dims = dims;

    for (int i = 0; i < dims; i++) {
        fmt->dim[i] = dim[i];
        memcpy(fmt->dim_name[i], dim_name[i], CHARTS_META_MAX_DIMNAME);
        fmt->stride[i] = stride[i];
    }

    fmt->offset = offset;

    if (has_coarse_freq()) {
        auto cf = get_coarse_freq();
        fmt->nfreq = cf.size();
        std::copy(cf.begin(), cf.end(), fmt->coarse_freq);
    } else {
        fmt->nfreq = 0;
    }

    fmt->lost_timesamples = has_lost_timesamples() ? get_lost_timesamples() : -1;

    fmt->time0_fpga = has_time0_fpga() ? get_time0_fpga() : 0;


    return sizeof(*fmt);
}

size_t chartsMetadata::set_from_bytes(const char* bytes, size_t length) {
    assert(length >= sizeof(chartsMetadataFormat));
    const auto* fmt = reinterpret_cast<const chartsMetadataFormat*>(bytes);

    if (fmt->frame_counter != -1)
        set_frame_counter(fmt->frame_counter);
    if (fmt->fpga_seq_num != -1)
        set_fpga_seq_num(fmt->fpga_seq_num);
    if (fmt->time_downsampling_fpga != -1)
        set_time_downsampling_fpga(fmt->time_downsampling_fpga);

    memcpy(name, fmt->name, sizeof(name));
    type = static_cast<kotekan::DataType>(fmt->type);
    dims = fmt->dims;

    for (int i = 0; i < dims; i++) {
        dim[i] = fmt->dim[i];
        memcpy(dim_name[i], fmt->dim_name[i], CHARTS_META_MAX_DIMNAME);
        stride[i] = fmt->stride[i];
    }

    offset = fmt->offset;

    if (fmt->nfreq > 0) {
        set_coarse_freq(std::vector<int>(fmt->coarse_freq, fmt->coarse_freq + fmt->nfreq));
    }

    if (fmt->lost_timesamples != -1)
        set_lost_timesamples(fmt->lost_timesamples);

    if (fmt->first_packet_recv_time.tv_sec != 0) // sometime in 1970
        this->set_first_packet_recv_time(fmt->first_packet_recv_time);

    if (fmt->time0_fpga != 0)
        this->set_time0_fpga(fmt->time0_fpga);

    return sizeof(*fmt);
}
nlohmann::json chartsMetadata::to_json() {
    nlohmann::json j;

    j["max_dim"] = CHARTS_META_MAX_DIM;
    j["max_dimname"] = CHARTS_META_MAX_DIMNAME;
    j["max_freq"] = CHARTS_META_MAX_FREQ;

    j["name"] = get_name();
    j["dims"] = dims;
    j["type"] = type;
    j["offset"] = offset;

    j["dim"] = std::vector<int>(dim, dim + dims);
    j["stride"] = std::vector<int64_t>(stride, stride + dims);

    if (has_fpga_seq_num())
        j["fpga_seq_num"] = get_fpga_seq_num();
    if (has_frame_counter())
        j["frame_counter"] = get_frame_counter();
    if (has_coarse_freq())
        j["coarse_freq"] = get_coarse_freq();
    if (has_lost_timesamples())
        j["lost_timesamples"] = get_lost_timesamples();
    if (has_time0_fpga())
        j["time0_fpga"] = get_time0_fpga();
    return j;
}