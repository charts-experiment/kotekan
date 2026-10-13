/**
 * @file
 * @brief Write a small per-run metadata manifest.
 *
 * The manifest is intentionally independent of telescope geometry.  It records
 * the antennas known to have signal, antennas with suspect positions, and the
 * effective Kotekan configuration.  A source YAML can optionally be copied
 * alongside the manifest.
 */
#ifndef RUN_METADATA_WRITER_HPP
#define RUN_METADATA_WRITER_HPP

#include "Config.hpp"
#include "Stage.hpp"
#include "bufferContainer.hpp"

#include "json.hpp"

#include <string>
#include <vector>

class runMetadataWriter : public kotekan::Stage {
public:
    runMetadataWriter(kotekan::Config& config, const std::string& unique_name,
                      kotekan::bufferContainer& buffer_container);
    ~runMetadataWriter() override = default;

    void main_thread() override;

private:
    const std::string _base_dir;
    const std::string _config_file;
    const std::string _config_snapshot;
    const int _num_antennas;
    const std::vector<int> _signal_antennas;
    const std::vector<int> _position_outliers;
    const std::vector<nlohmann::json> _position_overrides;
};

#endif // RUN_METADATA_WRITER_HPP
