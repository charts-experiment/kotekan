#include "runMetadataWriter.hpp"

#include "StageFactory.hpp"
#include "kotekanLogging.hpp"

#include "json.hpp"

#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <set>
#include <sstream>
#include <system_error>

using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(runMetadataWriter);

namespace {

bool valid_antennas(const std::vector<int>& antennas, int n_antennas) {
    for (int antenna : antennas) {
        if (antenna < 0 || (n_antennas > 0 && antenna >= n_antennas)) {
            return false;
        }
    }
    return true;
}

std::string run_stamp() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm utc{};
    gmtime_r(&now_time, &utc);
    std::ostringstream stamp;
    stamp << std::put_time(&utc, "%Y%m%dT%H%M%SZ");
    return stamp.str();
}

} // namespace

runMetadataWriter::runMetadataWriter(Config& config, const std::string& unique_name,
                                     kotekan::bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&runMetadataWriter::main_thread, this)),
    _base_dir(config.get<std::string>(unique_name, "base_dir")),
    _config_file(config.get_config_file()), _config_snapshot(config.get_full_config_json().dump(2)),
    _num_antennas(config.get_default<int>(unique_name, "num_antennas",
                                          config.get_default<int>("/", "num_elements", 0))),
    _signal_antennas(config.get_default<std::vector<int>>(unique_name, "signal_antennas", {})),
    _position_outliers(config.get_default<std::vector<int>>(unique_name, "position_outliers", {})),
    _position_overrides(
        config.get_default<std::vector<nlohmann::json>>(unique_name, "position_overrides", {})) {

    if (_num_antennas < 0)
        FATAL_ERROR("runMetadataWriter: num_antennas must be non-negative");
    if (!valid_antennas(_signal_antennas, _num_antennas))
        FATAL_ERROR("runMetadataWriter: signal_antennas contains an index outside 0..{}",
                    _num_antennas - 1);
    if (!valid_antennas(_position_outliers, _num_antennas))
        FATAL_ERROR("runMetadataWriter: position_outliers contains an index outside 0..{}",
                    _num_antennas - 1);
    for (const auto& override : _position_overrides) {
        if (!override.is_object() || !override.contains("antenna")
            || !override["antenna"].is_number_integer() || !override.contains("position_m")
            || !override["position_m"].is_array() || override["position_m"].size() != 3) {
            FATAL_ERROR("runMetadataWriter: each position_overrides entry must contain integer "
                        "antenna and a three-element position_m array");
        }
        const int antenna = override["antenna"].get<int>();
        if (antenna < 0 || (_num_antennas > 0 && antenna >= _num_antennas))
            FATAL_ERROR("runMetadataWriter: position_overrides contains antenna {} outside 0..{}",
                        antenna, _num_antennas - 1);
        for (const auto& coordinate : override["position_m"]) {
            if (!coordinate.is_number() || !std::isfinite(coordinate.get<double>()))
                FATAL_ERROR("runMetadataWriter: position_m must contain finite numbers");
        }
        if (override.contains("note") && !override["note"].is_string())
            FATAL_ERROR("runMetadataWriter: position_overrides note must be a string");
    }
}

void runMetadataWriter::main_thread() {
    namespace fs = std::filesystem;
    const std::string stamp = run_stamp();
    const std::string config_snapshot_name = "kotekan_config_" + stamp + ".json";
    const std::string manifest_name = "run_metadata_" + stamp + ".json";

    std::error_code ec;
    fs::create_directories(_base_dir, ec);
    if (ec)
        FATAL_ERROR("runMetadataWriter: Failed to create directory '{}': {}", _base_dir,
                    ec.message());

    nlohmann::json manifest;
    manifest["schema_version"] = 1;
    manifest["array"]["n_antennas"] = _num_antennas;
    manifest["array"]["index_convention"] = "charts_logical";
    manifest["signal_antennas"] = _signal_antennas;
    std::set<int> outlier_ids(_position_outliers.begin(), _position_outliers.end());
    manifest["run_id"] = stamp;
    manifest["kotekan"]["config_snapshot"] = config_snapshot_name;
    if (!_config_file.empty()) {
        manifest["kotekan"]["config_source"] = _config_file;
        manifest["kotekan"]["config_yaml_copy"] = "kotekan_config_" + stamp + ".yaml";
    }

    for (const auto& override : _position_overrides) {
        const auto antenna = override.at("antenna");
        outlier_ids.insert(antenna.get<int>());
        manifest["position_overrides"][std::to_string(antenna.get<int>())] = {
            {"position_m", override.at("position_m")},
            {"note", override.value("note", "")},
        };
    }
    manifest["position_outliers"] = std::vector<int>(outlier_ids.begin(), outlier_ids.end());

    const fs::path manifest_path = fs::path(_base_dir) / manifest_name;
    const fs::path manifest_tmp = manifest_path.string() + ".tmp";
    {
        std::ofstream output(manifest_tmp);
        if (!output)
            FATAL_ERROR("runMetadataWriter: Cannot open '{}' for writing", manifest_tmp.string());
        output << manifest.dump(2) << '\n';
    }
    fs::rename(manifest_tmp, manifest_path, ec);
    if (ec)
        FATAL_ERROR("runMetadataWriter: Failed to rename manifest: {}", ec.message());

    const fs::path config_snapshot = fs::path(_base_dir) / config_snapshot_name;
    const fs::path config_tmp = config_snapshot.string() + ".tmp";
    {
        std::ofstream output(config_tmp);
        if (!output)
            FATAL_ERROR("runMetadataWriter: Cannot open '{}' for writing", config_tmp.string());
        output << _config_snapshot << '\n';
    }
    fs::rename(config_tmp, config_snapshot, ec);
    if (ec)
        FATAL_ERROR("runMetadataWriter: Failed to rename config snapshot: {}", ec.message());

    if (!_config_file.empty()) {
        const fs::path source(_config_file);
        const fs::path destination = fs::path(_base_dir) / ("kotekan_config_" + stamp + ".yaml");
        fs::copy_file(source, destination, fs::copy_options::overwrite_existing, ec);
        if (ec)
            WARN("runMetadataWriter: Could not copy config YAML '{}': {}", _config_file,
                 ec.message());
    }

    INFO("runMetadataWriter: wrote run metadata to {}", _base_dir);
}
