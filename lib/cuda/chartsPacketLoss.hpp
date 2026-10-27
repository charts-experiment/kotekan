#ifndef CHARTS_PACKET_LOSS_HPP
#define CHARTS_PACKET_LOSS_HPP

#include "chartsMetadata.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace beamformer {

// Handler mask metadata has no frequency map. Its shard is bound by configuration;
// the voltage metadata supplies the independent frequency-map check.
inline std::int64_t validate_charts_mask_pair(
    const chartsMetadata* voltage, const chartsMetadata* mask,
    const std::uint8_t* missing, std::size_t mask_bytes, std::size_t raw_n_time,
    std::size_t shard_id, std::size_t mask_shard_id) {
    if (shard_id >= 2 || mask_shard_id != shard_id)
        throw std::invalid_argument("CHARTS mask shard binding mismatch");
    if (!missing || raw_n_time == 0 || mask_bytes != raw_n_time)
        throw std::invalid_argument("CHARTS mask must contain T_raw bytes");
    if (!voltage || !mask || !voltage->has_fpga_seq_num() || !mask->has_fpga_seq_num()
        || !voltage->has_time_downsampling_fpga() || !mask->has_time_downsampling_fpga())
        throw std::invalid_argument("CHARTS voltage and mask require sequence/timestamp metadata");
    const auto seq = voltage->get_fpga_seq_num();
    if (seq < 0 || mask->get_fpga_seq_num() != seq
        || voltage->get_time_downsampling_fpga() != 1 || mask->get_time_downsampling_fpga() != 1)
        throw std::invalid_argument("CHARTS voltage/mask sequence or timestamp_step mismatch");
    if (raw_n_time > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max() - seq))
        throw std::invalid_argument("CHARTS next sequence overflows int64");
    if (!voltage->has_coarse_freq())
        throw std::invalid_argument("CHARTS voltage requires its shard frequency map");
    const auto frequencies = voltage->get_coarse_freq();
    if (frequencies.size() != 336)
        throw std::invalid_argument("CHARTS voltage requires 336 coarse channels");
    for (std::size_t f = 0; f < frequencies.size(); ++f)
        if (frequencies[f] != static_cast<int>(336 * shard_id + f))
            throw std::invalid_argument("CHARTS voltage frequency map disagrees with shard");
    for (std::size_t t = 0; t < mask_bytes; ++t)
        if (missing[t] > 1)
            throw std::invalid_argument("CHARTS missing flags must be binary (1=missing)");
    return seq;
}

} // namespace beamformer
#endif
