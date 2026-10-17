#include "cudaAntennaMaskCommand.hpp"

#include "cudaDirectBeamTrackerCommand.hpp"
#include "cudaUtils.hpp"
#include "gpuCommand.hpp"
#include "kotekanLogging.hpp"
#include "restServer.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fmt/format.h>
#include <vector>

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::connectionInstance;
using kotekan::cudaAntennaMaskCommand;
using kotekan::HTTP_RESPONSE;
using kotekan::restServer;

REGISTER_CUDA_COMMAND(cudaAntennaMaskCommand);

namespace kotekan {

std::mutex cudaAntennaMaskCommand::_global_mutex;
AntennaMaskConfig cudaAntennaMaskCommand::_shared_config;
std::array<AntennaHealthMetrics, MAX_MASK_ANTENNAS> cudaAntennaMaskCommand::_shared_metrics{};
std::array<std::uint8_t, MAX_MASK_ANTENNAS> cudaAntennaMaskCommand::_shared_mask{};
bool cudaAntennaMaskCommand::_endpoints_registered = false;
std::mutex cudaAntennaMaskCommand::_registry_mutex;
std::vector<cudaAntennaMaskCommand*> cudaAntennaMaskCommand::_instances;

namespace {

float compute_median(std::vector<float>& vals) {
    if (vals.empty())
        return 0.0f;
    const std::size_t n = vals.size();
    const std::size_t mid = n / 2;
    std::nth_element(vals.begin(), vals.begin() + mid, vals.end());
    const float m1 = vals[mid];
    if ((n & 1U) != 0U)
        return m1;
    std::nth_element(vals.begin(), vals.begin() + (mid - 1), vals.end());
    return 0.5f * (m1 + vals[mid - 1]);
}

} // namespace

cudaAntennaMaskCommand::cudaAntennaMaskCommand(Config& config, const std::string& unique_name,
                                               bufferContainer& host_buffers,
                                               cudaDeviceInterface& device, int inst) :
    cudaCommand(config, unique_name, host_buffers, device, inst, no_cuda_command_state,
                "cudaAntennaMaskCommand") {

    _num_pols = config.get_default<int>(unique_name, "num_pols", 1);
    if (config.exists(unique_name, "num_antennas")) {
        _num_antennas = config.get<int>(unique_name, "num_antennas");
        _num_elements =
            config.get_default<int>(unique_name, "num_elements", _num_antennas * _num_pols);
    } else {
        _num_elements = config.get<int>(unique_name, "num_elements");
        _num_antennas = _num_elements / std::max(1, _num_pols);
    }

    const std::string pol_layout_str =
        config.get_default<std::string>(unique_name, "pol_layout", "interleaved");
    _pol_layout =
        (pol_layout_str == "planar") ? DirectPolLayout::PLANAR : DirectPolLayout::INTERLEAVED;

    _num_local_freq = config.get<int>(unique_name, "num_local_freq");
    _samples_per_data_set = config.get<int>(unique_name, "samples_per_data_set");
    _buffer_depth = config.get<int>(unique_name, "buffer_depth");

    _gpu_mem_voltage = config.get_default<std::string>(unique_name, "gpu_mem_voltage", "voltage");
    _gpu_mem_alive_voltages =
        config.get_default<std::string>(unique_name, "gpu_mem_alive_voltages", "");
    _max_alive_antennas = config.get_default<int>(unique_name, "max_alive_antennas", _num_elements);
    _antenna_order = config.get_default<std::string>(unique_name, "antenna_order", "descending");
    _freq_start_hz = config.get_default<double>(unique_name, "freq_start_hz", 300.0e6);
    _subband_id =
        config.get_default<int>(unique_name, "subband", (_freq_start_hz > 350.0e6 ? 1 : 0));

    _last_active_mask.resize(_num_elements, 1);
    _last_power_rms.resize(_num_elements, 0.0f);
    _last_clip_fraction.resize(_num_elements, 0.0f);

    {
        std::lock_guard<std::mutex> lk(_registry_mutex);
        _instances.push_back(this);
    }

    {
        std::lock_guard<std::mutex> lock(_global_mutex);
        _shared_config.num_antennas = static_cast<std::size_t>(_num_antennas);
        _shared_config.num_pols = static_cast<std::size_t>(_num_pols);
        _shared_config.pol_layout = _pol_layout;
        _shared_config.antenna_order = _antenna_order;
        _shared_config.couple_pol_masks =
            config.get_default<bool>(unique_name, "couple_pol_masks", false);

        _shared_config.auto_detect_enabled =
            config.get_default<bool>(unique_name, "auto_detect_enabled", true);
        _shared_config.blank_voltages_enabled =
            config.get_default<bool>(unique_name, "blank_voltages_enabled", false);
        _shared_config.dead_power_threshold =
            config.get_default<float>(unique_name, "dead_power_threshold", 5.0f);
        _shared_config.sat_power_threshold =
            config.get_default<float>(unique_name, "sat_power_threshold", 80.0f);
        _shared_config.clip_fraction_threshold =
            config.get_default<float>(unique_name, "clip_fraction_threshold", 0.15f);

        _shared_config.enable_dc_lock_check =
            config.get_default<bool>(unique_name, "enable_dc_lock_check", true);
        _shared_config.max_dc_fraction =
            config.get_default<float>(unique_name, "max_dc_fraction", 0.35f);

        _shared_config.enable_iq_imbalance_check =
            config.get_default<bool>(unique_name, "enable_iq_imbalance_check", true);
        _shared_config.max_iq_imbalance =
            config.get_default<float>(unique_name, "max_iq_imbalance", 0.50f);

        _shared_config.enable_kurtosis_check =
            config.get_default<bool>(unique_name, "enable_kurtosis_check", true);
        _shared_config.min_kurtosis_threshold =
            config.get_default<float>(unique_name, "min_kurtosis_threshold", 1.15f);
        _shared_config.max_kurtosis_threshold =
            config.get_default<float>(unique_name, "max_kurtosis_threshold", 4.50f);

        _shared_config.enable_mad_outlier_check =
            config.get_default<bool>(unique_name, "enable_mad_outlier_check", true);
        _shared_config.mad_sigma_threshold =
            config.get_default<float>(unique_name, "mad_sigma_threshold", 6.0f);

        _shared_config.revival_frames =
            static_cast<std::uint16_t>(config.get_default<int>(unique_name, "revival_frames", 5));
        _shared_config.sample_stride =
            static_cast<std::size_t>(config.get_default<int>(unique_name, "sample_stride", 1));

        _shared_mask.fill(1);
        _shared_config.manual_mask.fill(1);

        auto set_phys_ant = [&](int phys_ant, std::uint8_t state) {
            for (int r = 0; r < _num_elements && r < static_cast<int>(MAX_MASK_ANTENNAS); ++r) {
                if (static_cast<int>(_shared_config.raw_elem_to_physical_ant(r)) == phys_ant) {
                    _shared_config.manual_mask[r] = state;
                    _shared_mask[r] = state;
                    _shared_metrics[r].status = (state == 0) ? AntennaHealthStatus::MANUAL_MASK
                                                             : AntennaHealthStatus::HEALTHY;
                }
            }
        };

        auto bad_antennas = config.get_default<std::vector<int>>(unique_name, "bad_antennas", {});
        for (int ant : bad_antennas) {
            set_phys_ant(ant, 0);
        }

        auto bad_elements = config.get_default<std::vector<int>>(unique_name, "bad_elements", {});
        for (int elem : bad_elements) {
            if (elem >= 0 && elem < _num_elements && elem < static_cast<int>(MAX_MASK_ANTENNAS)) {
                _shared_config.manual_mask[elem] = 0;
                _shared_mask[elem] = 0;
                _shared_metrics[elem].status = AntennaHealthStatus::MANUAL_MASK;
            }
        }

        auto active_antennas =
            config.get_default<std::vector<int>>(unique_name, "active_antennas", {});
        if (!active_antennas.empty()) {
            _shared_config.manual_mask.fill(0);
            _shared_mask.fill(0);
            for (std::size_t i = 0; i < MAX_MASK_ANTENNAS; ++i) {
                _shared_metrics[i].status = AntennaHealthStatus::MANUAL_MASK;
                _shared_metrics[i].consecutive_healthy = 0;
            }
            for (int ant : active_antennas) {
                set_phys_ant(ant, 1);
            }
        }

        auto active_raw =
            config.get_default<std::vector<int>>(unique_name, "active_raw_elements", {});
        if (!active_raw.empty()) {
            _shared_config.manual_mask.fill(0);
            _shared_mask.fill(0);
            for (std::size_t i = 0; i < MAX_MASK_ANTENNAS; ++i) {
                _shared_metrics[i].status = AntennaHealthStatus::MANUAL_MASK;
                _shared_metrics[i].consecutive_healthy = 0;
            }
            for (int r : active_raw) {
                if (r >= 0 && r < _num_elements && r < static_cast<int>(MAX_MASK_ANTENNAS)) {
                    _shared_config.manual_mask[r] = 1;
                    _shared_mask[r] = 1;
                }
            }
        }

        for (std::size_t i = 0; i < MAX_MASK_ANTENNAS; ++i) {
            if (_shared_mask[i] != 0) {
                _shared_metrics[i].status = AntennaHealthStatus::HEALTHY;
                _shared_metrics[i].consecutive_healthy = _shared_config.revival_frames;
            }
        }

        _shared_config.mask_generation++;
        int n_active_init = 0;
        for (int i = 0; i < _num_elements && i < static_cast<int>(MAX_MASK_ANTENNAS); ++i) {
            if (_shared_mask[i] != 0)
                n_active_init++;
        }
        cudaDirectBeamTrackerCommand::set_shared_antenna_mask(_shared_mask, n_active_init);

        if (!_endpoints_registered) {
            auto& rest = restServer::instance();

            // 1. GET /antenna_mask/status
            auto status_cb = [](connectionInstance& conn) {
                nlohmann::json reply;
                std::lock_guard<std::mutex> lk(_global_mutex);

                const int n_elem =
                    static_cast<int>(std::min(_shared_config.total_elements(), MAX_MASK_ANTENNAS));

                reply["version"] = "Kotekan Antenna Masking Stage v2.0 (Multi-Moment & MAD)";
                reply["num_antennas"] = _shared_config.num_antennas;
                reply["num_pols"] = _shared_config.num_pols;
                reply["num_elements"] = n_elem;
                reply["antenna_order"] = _shared_config.antenna_order;
                reply["auto_detect_enabled"] = _shared_config.auto_detect_enabled;
                reply["blank_voltages_enabled"] = _shared_config.blank_voltages_enabled;
                reply["couple_pol_masks"] = _shared_config.couple_pol_masks;
                reply["thresholds"] = {
                    {"dead_power_threshold", _shared_config.dead_power_threshold},
                    {"sat_power_threshold", _shared_config.sat_power_threshold},
                    {"clip_fraction_threshold", _shared_config.clip_fraction_threshold},
                    {"max_dc_fraction", _shared_config.max_dc_fraction},
                    {"max_iq_imbalance", _shared_config.max_iq_imbalance},
                    {"min_kurtosis_threshold", _shared_config.min_kurtosis_threshold},
                    {"max_kurtosis_threshold", _shared_config.max_kurtosis_threshold},
                    {"mad_sigma_threshold", _shared_config.mad_sigma_threshold},
                    {"revival_frames", _shared_config.revival_frames},
                    {"sample_stride", _shared_config.sample_stride}};

                reply["manual_mask"] = nlohmann::json::array();
                for (int a = 0; a < n_elem; ++a) {
                    reply["manual_mask"].push_back(_shared_config.manual_mask[a] != 0 ? 1 : 0);
                }

                // Subband telemetry aggregation
                nlohmann::json subbands_json = nlohmann::json::array();
                std::vector<cudaAntennaMaskCommand*> sorted_instances;
                {
                    std::lock_guard<std::mutex> reg_lk(_registry_mutex);
                    sorted_instances = _instances;
                }
                std::sort(sorted_instances.begin(), sorted_instances.end(),
                          [](const cudaAntennaMaskCommand* a, const cudaAntennaMaskCommand* b) {
                              return a->get_freq_start_hz() < b->get_freq_start_hz();
                          });

                if (!sorted_instances.empty()) {
                    for (std::size_t s = 0; s < sorted_instances.size(); ++s) {
                        auto* inst = sorted_instances[s];
                        nlohmann::json sub;
                        sub["subband"] = inst->get_subband_id();
                        sub["freq_start_hz"] = inst->get_freq_start_hz();

                        nlohmann::json act_mask = nlohmann::json::array();
                        nlohmann::json pwr_rms = nlohmann::json::array();
                        nlohmann::json clp_frac = nlohmann::json::array();

                        const auto m_mask = inst->get_last_active_mask();
                        const auto m_pwr = inst->get_last_power_rms();
                        const auto m_clp = inst->get_last_clip_fraction();

                        for (int a = 0; a < n_elem; ++a) {
                            act_mask.push_back((a < static_cast<int>(m_mask.size())) ? m_mask[a]
                                                                                     : 1);
                            pwr_rms.push_back((a < static_cast<int>(m_pwr.size())) ? m_pwr[a]
                                                                                   : 0.0f);
                            clp_frac.push_back((a < static_cast<int>(m_clp.size())) ? m_clp[a]
                                                                                    : 0.0f);
                        }
                        sub["active_mask"] = act_mask;
                        sub["power_rms"] = pwr_rms;
                        sub["clip_fraction"] = clp_frac;
                        subbands_json.push_back(sub);
                    }
                } else {
                    nlohmann::json sub;
                    sub["subband"] = 0;
                    sub["freq_start_hz"] = 300.0e6;
                    nlohmann::json act_mask = nlohmann::json::array();
                    nlohmann::json pwr_rms = nlohmann::json::array();
                    nlohmann::json clp_frac = nlohmann::json::array();
                    for (int a = 0; a < n_elem; ++a) {
                        act_mask.push_back(_shared_mask[a] != 0 ? 1 : 0);
                        pwr_rms.push_back(_shared_metrics[a].power);
                        clp_frac.push_back(_shared_metrics[a].clipping_fraction);
                    }
                    sub["active_mask"] = act_mask;
                    sub["power_rms"] = pwr_rms;
                    sub["clip_fraction"] = clp_frac;
                    subbands_json.push_back(sub);
                }
                reply["subbands"] = subbands_json;

                int active_count = 0;
                int dead_count = 0;
                int sat_count = 0;
                int dc_count = 0;
                int rfi_count = 0;
                int iq_count = 0;
                int outlier_count = 0;
                int manual_count = 0;
                nlohmann::json ant_array = nlohmann::json::array();
                nlohmann::json active_indices = nlohmann::json::array();
                nlohmann::json active_phys = nlohmann::json::array();

                for (int a = 0; a < n_elem; ++a) {
                    const auto& m = _shared_metrics[a];
                    const bool active = (_shared_mask[a] != 0);
                    const std::size_t phys_id = _shared_config.raw_elem_to_physical_ant(a);
                    if (active) {
                        active_count++;
                        active_indices.push_back(a);
                        active_phys.push_back(phys_id);
                    }
                    switch (m.status) {
                        case AntennaHealthStatus::DEAD:
                            dead_count++;
                            break;
                        case AntennaHealthStatus::SATURATED:
                            sat_count++;
                            break;
                        case AntennaHealthStatus::DC_LOCKED:
                            dc_count++;
                            break;
                        case AntennaHealthStatus::RFI_KURTOSIS:
                            rfi_count++;
                            break;
                        case AntennaHealthStatus::IQ_IMBALANCE:
                            iq_count++;
                            break;
                        case AntennaHealthStatus::OUTLIER_POWER:
                            outlier_count++;
                            break;
                        case AntennaHealthStatus::MANUAL_MASK:
                            manual_count++;
                            break;
                        default:
                            break;
                    }

                    ant_array.push_back({{"id", a},
                                         {"antenna_id", a},
                                         {"raw_id", a},
                                         {"physical_id", phys_id},
                                         {"pol", _shared_config.raw_elem_to_pol(a)},
                                         {"active", active},
                                         {"enabled", active},
                                         {"status", antenna_health_status_to_string(m.status)},
                                         {"power", m.power},
                                         {"power_rms", m.power},
                                         {"ac_variance", m.ac_variance},
                                         {"dc_fraction", m.dc_fraction},
                                         {"clipping_fraction", m.clipping_fraction},
                                         {"clip_fraction", m.clipping_fraction},
                                         {"kurtosis", m.kurtosis},
                                         {"iq_imbalance", m.iq_imbalance},
                                         {"robust_z_score", m.robust_z_score},
                                         {"clipped_count", m.clipped_count},
                                         {"consecutive_healthy", m.consecutive_healthy}});
                }

                // Dual keys for complete backwards compatibility
                reply["active_antennas"] = active_count;
                reply["active_elements"] = active_count;
                reply["masked_antennas"] = n_elem - active_count;
                reply["masked_elements"] = n_elem - active_count;
                reply["dead_antennas"] = dead_count;
                reply["dead_elements"] = dead_count;
                reply["saturated_antennas"] = sat_count;
                reply["saturated_elements"] = sat_count;
                reply["dc_locked_elements"] = dc_count;
                reply["rfi_kurtosis_elements"] = rfi_count;
                reply["iq_imbalance_elements"] = iq_count;
                reply["outlier_power_elements"] = outlier_count;
                reply["manual_masked_antennas"] = manual_count;
                reply["manual_masked_elements"] = manual_count;

                reply["active_raw_elements"] = active_indices;
                reply["active_physical_antennas"] = active_phys;
                reply["antennas"] = ant_array;

                conn.send_json_reply(reply);
            };
            rest.register_get_callback("/antenna_mask/status", status_cb);

            // Helper lambda for masking / unmasking via REST
            auto apply_rest_mask = [](connectionInstance& conn, nlohmann::json& j, bool enable) {
                try {
                    std::lock_guard<std::mutex> lk(_global_mutex);
                    const std::size_t n_elem =
                        std::min(_shared_config.total_elements(), MAX_MASK_ANTENNAS);
                    bool is_raw = j.value("is_raw", false);
                    const std::uint8_t val = enable ? 1 : 0;

                    if (j.contains("raw_id") || (j.contains("antenna_id") && is_raw)) {
                        std::size_t raw_id = j.contains("raw_id") ? j["raw_id"] : j["antenna_id"];
                        if (raw_id >= n_elem) {
                            conn.send_error("raw_id exceeds total elements",
                                            HTTP_RESPONSE::BAD_REQUEST);
                            return;
                        }
                        _shared_config.manual_mask[raw_id] = val;
                        _shared_mask[raw_id] = val;
                        _shared_metrics[raw_id].status = enable ? AntennaHealthStatus::HEALTHY
                                                                : AntennaHealthStatus::MANUAL_MASK;
                        _shared_metrics[raw_id].consecutive_healthy =
                            enable ? _shared_config.revival_frames : 0;
                        INFO_NON_OO("AntennaMask: Raw element {:d} {:s} applied via REST", raw_id,
                                    enable ? "UNMASK" : "MANUAL MASK");
                    } else if (j.contains("physical_id") || j.contains("antenna_id")) {
                        std::size_t phys_id =
                            j.contains("physical_id") ? j["physical_id"] : j["antenna_id"];
                        if (phys_id >= _shared_config.num_antennas) {
                            conn.send_error("physical_id exceeds num_antennas",
                                            HTTP_RESPONSE::BAD_REQUEST);
                            return;
                        }
                        for (std::size_t r = 0; r < n_elem; ++r) {
                            if (_shared_config.raw_elem_to_physical_ant(r) == phys_id) {
                                _shared_config.manual_mask[r] = val;
                                _shared_mask[r] = val;
                                _shared_metrics[r].status = enable
                                                                ? AntennaHealthStatus::HEALTHY
                                                                : AntennaHealthStatus::MANUAL_MASK;
                                _shared_metrics[r].consecutive_healthy =
                                    enable ? _shared_config.revival_frames : 0;
                            }
                        }
                        INFO_NON_OO("AntennaMask: Physical antenna {:d} {:s} applied via REST",
                                    phys_id, enable ? "UNMASK" : "MANUAL MASK");
                    } else {
                        conn.send_error("Missing antenna_id, physical_id, or raw_id",
                                        HTTP_RESPONSE::BAD_REQUEST);
                        return;
                    }

                    int n_active = 0;
                    for (std::size_t i = 0; i < n_elem; ++i) {
                        if (_shared_mask[i] != 0)
                            n_active++;
                    }
                    _shared_config.mask_generation++;
                    cudaDirectBeamTrackerCommand::set_shared_antenna_mask(_shared_mask, n_active);
                    conn.send_text_reply(
                        fmt::format("Mask updated: {:d}/{:d} active elements\n", n_active, n_elem));
                } catch (const std::exception& e) {
                    conn.send_error(e.what(), HTTP_RESPONSE::BAD_REQUEST);
                }
            };

            rest.register_post_callback(
                "/antenna_mask/mask",
                [apply_rest_mask](connectionInstance& conn, nlohmann::json& j) {
                    apply_rest_mask(conn, j, false);
                });
            rest.register_post_callback(
                "/antenna_mask/unmask",
                [apply_rest_mask](connectionInstance& conn, nlohmann::json& j) {
                    apply_rest_mask(conn, j, true);
                });
            rest.register_post_callback(
                "/antenna_mask/mask_antenna",
                [apply_rest_mask](connectionInstance& conn, nlohmann::json& j) {
                    bool enabled = j.value("enabled", false);
                    apply_rest_mask(conn, j, enabled);
                });

            // 4. POST /antenna_mask/reset
            auto reset_cb = [](connectionInstance& conn, nlohmann::json& /*j*/) {
                std::lock_guard<std::mutex> lk(_global_mutex);
                const std::size_t n_elem =
                    std::min(_shared_config.total_elements(), MAX_MASK_ANTENNAS);
                _shared_config.manual_mask.fill(1);
                _shared_mask.fill(1);
                for (std::size_t i = 0; i < MAX_MASK_ANTENNAS; ++i) {
                    _shared_metrics[i].status = AntennaHealthStatus::HEALTHY;
                    _shared_metrics[i].consecutive_healthy = _shared_config.revival_frames;
                }
                _shared_config.mask_generation++;
                cudaDirectBeamTrackerCommand::set_shared_antenna_mask(_shared_mask,
                                                                      static_cast<int>(n_elem));
                INFO_NON_OO("AntennaMask: Reset all antenna masks to ACTIVE via REST");
                conn.send_text_reply("All antenna masks reset to active\n");
            };
            rest.register_post_callback("/antenna_mask/reset", reset_cb);

            // 5. POST /antenna_mask/configure
            auto config_cb = [](connectionInstance& conn, nlohmann::json& j) {
                try {
                    std::lock_guard<std::mutex> lk(_global_mutex);
                    if (j.contains("auto_detect_enabled"))
                        _shared_config.auto_detect_enabled = j["auto_detect_enabled"];
                    if (j.contains("blank_voltages_enabled"))
                        _shared_config.blank_voltages_enabled = j["blank_voltages_enabled"];
                    if (j.contains("couple_pol_masks"))
                        _shared_config.couple_pol_masks = j["couple_pol_masks"];
                    if (j.contains("dead_power_threshold"))
                        _shared_config.dead_power_threshold = j["dead_power_threshold"];
                    if (j.contains("sat_power_threshold"))
                        _shared_config.sat_power_threshold = j["sat_power_threshold"];
                    if (j.contains("clip_fraction_threshold"))
                        _shared_config.clip_fraction_threshold = j["clip_fraction_threshold"];
                    if (j.contains("enable_dc_lock_check"))
                        _shared_config.enable_dc_lock_check = j["enable_dc_lock_check"];
                    if (j.contains("max_dc_fraction"))
                        _shared_config.max_dc_fraction = j["max_dc_fraction"];
                    if (j.contains("enable_iq_imbalance_check"))
                        _shared_config.enable_iq_imbalance_check = j["enable_iq_imbalance_check"];
                    if (j.contains("max_iq_imbalance"))
                        _shared_config.max_iq_imbalance = j["max_iq_imbalance"];
                    if (j.contains("enable_kurtosis_check"))
                        _shared_config.enable_kurtosis_check = j["enable_kurtosis_check"];
                    if (j.contains("min_kurtosis_threshold"))
                        _shared_config.min_kurtosis_threshold = j["min_kurtosis_threshold"];
                    if (j.contains("max_kurtosis_threshold"))
                        _shared_config.max_kurtosis_threshold = j["max_kurtosis_threshold"];
                    if (j.contains("enable_mad_outlier_check"))
                        _shared_config.enable_mad_outlier_check = j["enable_mad_outlier_check"];
                    if (j.contains("mad_sigma_threshold"))
                        _shared_config.mad_sigma_threshold = j["mad_sigma_threshold"];
                    if (j.contains("revival_frames"))
                        _shared_config.revival_frames = j["revival_frames"];
                    if (j.contains("sample_stride"))
                        _shared_config.sample_stride = j["sample_stride"];
                    _shared_config.mask_generation++;
                    conn.send_text_reply("Antenna mask configuration updated successfully\n");
                } catch (const std::exception& e) {
                    conn.send_error(e.what(), HTTP_RESPONSE::BAD_REQUEST);
                }
            };
            rest.register_post_callback("/antenna_mask/configure", config_cb);

            _endpoints_registered = true;
        }
    }

    set_command_type(gpuCommandType::KERNEL);
    set_name("cudaAntennaMaskCommand");
    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_voltage, true, true, true));
    if (!_gpu_mem_alive_voltages.empty()) {
        gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_alive_voltages, true, false, true));
    }

    allocate_device_buffers();
    INFO_NON_OO(
        "Kotekan Antenna Masking Stage v2.0 initialized for {:d} elements ({:d} ant x {:d} pol)",
        _num_elements, _num_antennas, _num_pols);
}

cudaAntennaMaskCommand::~cudaAntennaMaskCommand() {
    {
        std::lock_guard<std::mutex> lk(_registry_mutex);
        _instances.erase(std::remove(_instances.begin(), _instances.end(), this), _instances.end());
    }
    free_device_buffers();
}

void cudaAntennaMaskCommand::allocate_device_buffers() {
    device.set_thread_device();
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&_d_raw_stats, _num_elements * sizeof(AntennaRawStats)));
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&_d_antenna_powers, _num_elements * sizeof(float)));
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&_d_antenna_clips, _num_elements * sizeof(std::uint32_t)));
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&_d_bad_antennas, _num_elements * sizeof(int)));
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&_d_alive_antennas, _num_elements * sizeof(int)));

    CHECK_CUDA_ERROR_NON_OO(cudaMallocHost(&_h_raw_stats, _num_elements * sizeof(AntennaRawStats)));
}

void cudaAntennaMaskCommand::free_device_buffers() {
    device.set_thread_device();
    if (_d_raw_stats) {
        cudaFree(_d_raw_stats);
        _d_raw_stats = nullptr;
    }
    if (_d_antenna_powers) {
        cudaFree(_d_antenna_powers);
        _d_antenna_powers = nullptr;
    }
    if (_d_antenna_clips) {
        cudaFree(_d_antenna_clips);
        _d_antenna_clips = nullptr;
    }
    if (_d_bad_antennas) {
        cudaFree(_d_bad_antennas);
        _d_bad_antennas = nullptr;
    }
    if (_d_alive_antennas) {
        cudaFree(_d_alive_antennas);
        _d_alive_antennas = nullptr;
    }
    if (_h_raw_stats) {
        cudaFreeHost(_h_raw_stats);
        _h_raw_stats = nullptr;
    }
}

std::array<std::uint8_t, MAX_MASK_ANTENNAS> cudaAntennaMaskCommand::get_current_mask() {
    std::lock_guard<std::mutex> lk(_global_mutex);
    return _shared_mask;
}

AntennaHealthMetrics cudaAntennaMaskCommand::get_antenna_metrics(std::size_t ant_id) {
    std::lock_guard<std::mutex> lk(_global_mutex);
    if (ant_id < MAX_MASK_ANTENNAS)
        return _shared_metrics[ant_id];
    return AntennaHealthMetrics{};
}

void cudaAntennaMaskCommand::set_manual_mask(std::size_t ant_id, bool enable) {
    std::lock_guard<std::mutex> lk(_global_mutex);
    if (ant_id < MAX_MASK_ANTENNAS) {
        _shared_config.manual_mask[ant_id] = enable ? 1 : 0;
        _shared_mask[ant_id] = enable ? 1 : 0;
        _shared_metrics[ant_id].status =
            enable ? AntennaHealthStatus::HEALTHY : AntennaHealthStatus::MANUAL_MASK;
        _shared_metrics[ant_id].consecutive_healthy = enable ? _shared_config.revival_frames : 0;
        _shared_config.mask_generation++;

        int n_active = 0;
        for (std::size_t i = 0; i < _shared_config.total_elements() && i < MAX_MASK_ANTENNAS; ++i) {
            if (_shared_mask[i] != 0)
                n_active++;
        }
        cudaDirectBeamTrackerCommand::set_shared_antenna_mask(_shared_mask, n_active);
    }
}

void cudaAntennaMaskCommand::reset_all_masks() {
    std::lock_guard<std::mutex> lk(_global_mutex);
    _shared_config.manual_mask.fill(1);
    _shared_mask.fill(1);
    for (std::size_t i = 0; i < MAX_MASK_ANTENNAS; ++i) {
        _shared_metrics[i].status = AntennaHealthStatus::HEALTHY;
        _shared_metrics[i].consecutive_healthy = _shared_config.revival_frames;
    }
    _shared_config.mask_generation++;
    const int n_active =
        static_cast<int>(std::min(_shared_config.total_elements(), MAX_MASK_ANTENNAS));
    cudaDirectBeamTrackerCommand::set_shared_antenna_mask(_shared_mask, n_active);
}

void cudaAntennaMaskCommand::update_config(const AntennaMaskConfig& cfg) {
    std::lock_guard<std::mutex> lk(_global_mutex);
    _shared_config = cfg;
    _shared_config.mask_generation++;
}

AntennaMaskConfig cudaAntennaMaskCommand::get_config() {
    std::lock_guard<std::mutex> lk(_global_mutex);
    return _shared_config;
}

cudaEvent_t cudaAntennaMaskCommand::execute(cudaPipelineState& /*pipestate*/,
                                            const std::vector<cudaEvent_t>&) {
    pre_execute();

    const std::size_t input_bytes =
        static_cast<std::size_t>(_num_elements) * static_cast<std::size_t>(_num_local_freq)
        * static_cast<std::size_t>(_samples_per_data_set) * sizeof(int4x2_t);

    void* input_memory =
        device.get_gpu_memory_array(_gpu_mem_voltage, gpu_frame_id, _gpu_buffer_depth, input_bytes);
    if (!input_memory) {
        return record_end_event();
    }

    record_start_event();
    cudaStream_t stream = device.getStream(cuda_stream_id);
    int4x2_t* d_voltages = reinterpret_cast<int4x2_t*>(input_memory);

    AntennaMaskConfig current_cfg = get_config();

    const std::size_t total_spectra =
        static_cast<std::size_t>(_samples_per_data_set) * _num_local_freq;
    const std::size_t inspected_spectra =
        (total_spectra + current_cfg.sample_stride - 1) / current_cfg.sample_stride;

    if (current_cfg.auto_detect_enabled && inspected_spectra > 0) {
        // 1. Single-pass exact integer moment accumulation on GPU
        launch_inspect_antenna_moments(d_voltages, _d_raw_stats, nullptr, nullptr,
                                       _samples_per_data_set, _num_local_freq, _num_elements,
                                       current_cfg.sample_stride, stream);

        // Single consolidated DMA transfer of AntennaRawStats
        CHECK_CUDA_ERROR_NON_OO(cudaMemcpyAsync(_h_raw_stats, _d_raw_stats,
                                                _num_elements * sizeof(AntennaRawStats),
                                                cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA_ERROR_NON_OO(cudaStreamSynchronize(stream));

        const double inv_n = 1.0 / static_cast<double>(inspected_spectra);

        // Pre-pass: compute per-element moments & gather candidate powers per polarization for MAD
        std::vector<float> pol_powers[2];
        pol_powers[0].reserve(_num_elements);
        pol_powers[1].reserve(_num_elements);

        std::vector<float> elem_power(_num_elements, 0.0f);
        std::vector<float> elem_ac_var(_num_elements, 0.0f);
        std::vector<float> elem_dc_frac(_num_elements, 0.0f);
        std::vector<float> elem_clip_frac(_num_elements, 0.0f);
        std::vector<float> elem_kurtosis(_num_elements, 2.0f);
        std::vector<float> elem_iq_imb(_num_elements, 0.0f);

        for (int a = 0; a < _num_elements; ++a) {
            const auto& st = _h_raw_stats[a];
            const double p_re = static_cast<double>(st.sum_re2) * inv_n;
            const double p_im = static_cast<double>(st.sum_im2) * inv_n;
            const double p_tot = p_re + p_im;

            const double mu_re = static_cast<double>(st.sum_re) * inv_n;
            const double mu_im = static_cast<double>(st.sum_im) * inv_n;
            const double dc_p = mu_re * mu_re + mu_im * mu_im;
            const double ac_v = std::max(0.0, p_tot - dc_p);
            const double m4 = static_cast<double>(st.sum_p2) * inv_n;

            elem_power[a] = static_cast<float>(p_tot);
            elem_ac_var[a] = static_cast<float>(ac_v);
            elem_dc_frac[a] = (p_tot > 1e-6) ? static_cast<float>(dc_p / p_tot) : 0.0f;
            elem_clip_frac[a] = static_cast<float>(static_cast<double>(st.clips) * (0.5 * inv_n));
            elem_kurtosis[a] = (p_tot > 1e-3) ? static_cast<float>(m4 / (p_tot * p_tot)) : 0.0f;
            elem_iq_imb[a] =
                (p_tot > 1e-6) ? static_cast<float>(std::abs(p_re - p_im) / p_tot) : 0.0f;

            // Include non-dead, non-saturated elements in array median/MAD reference
            if (current_cfg.manual_mask[a] != 0 && ac_v > current_cfg.dead_power_threshold
                && p_tot < current_cfg.sat_power_threshold) {
                const std::size_t pol = current_cfg.raw_elem_to_pol(a) & 1U;
                pol_powers[pol].push_back(elem_power[a]);
            }
        }

        // Compute Median and MAD per polarization
        float med_power[2] = {0.0f, 0.0f};
        float mad_sigma[2] = {1.0f, 1.0f};
        for (int p = 0; p < std::min(2, _num_pols); ++p) {
            if (pol_powers[p].size() >= 4) {
                med_power[p] = compute_median(pol_powers[p]);
                std::vector<float> abs_devs(pol_powers[p].size());
                for (std::size_t i = 0; i < pol_powers[p].size(); ++i) {
                    abs_devs[i] = std::abs(pol_powers[p][i] - med_power[p]);
                }
                // Floor sigma at 5% of median power to avoid false triggers on ultra-uniform
                // synthetic data
                mad_sigma[p] = std::max(1.4826f * compute_median(abs_devs), 0.05f * med_power[p]);
            }
        }

        std::lock_guard<std::mutex> lk(_global_mutex);
        bool mask_changed = false;
        std::vector<std::uint8_t> candidate_mask(_num_elements, 1);

        for (int a = 0; a < _num_elements; ++a) {
            const std::size_t pol = current_cfg.raw_elem_to_pol(a) & 1U;
            const float z_score = (pol_powers[pol].size() >= 4)
                                      ? (std::abs(elem_power[a] - med_power[pol]) / mad_sigma[pol])
                                      : 0.0f;

            auto& m = _shared_metrics[a];
            m.power = elem_power[a];
            m.ac_variance = elem_ac_var[a];
            m.dc_fraction = elem_dc_frac[a];
            m.clipping_fraction = elem_clip_frac[a];
            m.kurtosis = elem_kurtosis[a];
            m.iq_imbalance = elem_iq_imb[a];
            m.robust_z_score = z_score;
            m.clipped_count = _h_raw_stats[a].clips;

            const std::uint8_t old_mask_val = _shared_mask[a];
            std::uint8_t new_mask_val = 1;

            if (_shared_config.manual_mask[a] == 0) {
                m.status = AntennaHealthStatus::MANUAL_MASK;
                m.consecutive_healthy = 0;
                new_mask_val = 0;
            } else if (m.ac_variance <= _shared_config.dead_power_threshold
                       || m.power <= _shared_config.dead_power_threshold) {
                m.status = AntennaHealthStatus::DEAD;
                m.consecutive_healthy = 0;
                new_mask_val = 0;
            } else if (m.clipping_fraction >= _shared_config.clip_fraction_threshold
                       || m.power >= _shared_config.sat_power_threshold) {
                m.status = AntennaHealthStatus::SATURATED;
                m.consecutive_healthy = 0;
                new_mask_val = 0;
            } else if (_shared_config.enable_dc_lock_check
                       && m.dc_fraction >= _shared_config.max_dc_fraction) {
                m.status = AntennaHealthStatus::DC_LOCKED;
                m.consecutive_healthy = 0;
                new_mask_val = 0;
            } else if (_shared_config.enable_iq_imbalance_check
                       && m.iq_imbalance >= _shared_config.max_iq_imbalance) {
                m.status = AntennaHealthStatus::IQ_IMBALANCE;
                m.consecutive_healthy = 0;
                new_mask_val = 0;
            } else if (_shared_config.enable_kurtosis_check
                       && (m.kurtosis >= _shared_config.max_kurtosis_threshold
                           || m.kurtosis <= _shared_config.min_kurtosis_threshold)) {
                m.status = AntennaHealthStatus::RFI_KURTOSIS;
                m.consecutive_healthy = 0;
                new_mask_val = 0;
            } else if (_shared_config.enable_mad_outlier_check
                       && z_score >= _shared_config.mad_sigma_threshold) {
                m.status = AntennaHealthStatus::OUTLIER_POWER;
                m.consecutive_healthy = 0;
                new_mask_val = 0;
            } else {
                m.consecutive_healthy++;
                if (old_mask_val == 0) {
                    if (m.consecutive_healthy >= _shared_config.revival_frames) {
                        m.status = AntennaHealthStatus::HEALTHY;
                        new_mask_val = 1;
                    } else {
                        new_mask_val = 0;
                    }
                } else {
                    m.status = AntennaHealthStatus::HEALTHY;
                    new_mask_val = 1;
                }
            }
            candidate_mask[a] = new_mask_val;
        }

        // Optional dual-pol coupling: if one polarization of a physical antenna is bad, mask both
        if (_shared_config.couple_pol_masks && _num_pols > 1) {
            for (int a = 0; a < _num_elements; ++a) {
                if (candidate_mask[a] == 0) {
                    const std::size_t phys_id = _shared_config.raw_elem_to_physical_ant(a);
                    for (int b = 0; b < _num_elements; ++b) {
                        if (_shared_config.raw_elem_to_physical_ant(b) == phys_id) {
                            candidate_mask[b] = 0;
                        }
                    }
                }
            }
        }

        _current_bad_indices.clear();
        _current_alive_indices.clear();
        for (int a = 0; a < _num_elements; ++a) {
            if (_shared_mask[a] != candidate_mask[a]) {
                _shared_mask[a] = candidate_mask[a];
                mask_changed = true;
            }
            if (_shared_mask[a] == 0) {
                _current_bad_indices.push_back(a);
            } else {
                _current_alive_indices.push_back(a);
            }
        }

        if (mask_changed || _applied_mask_generation != _shared_config.mask_generation) {
            if (mask_changed) {
                _shared_config.mask_generation++;
            }
            _applied_mask_generation = _shared_config.mask_generation;
            const int n_active = static_cast<int>(_current_alive_indices.size());
            cudaDirectBeamTrackerCommand::set_shared_antenna_mask(_shared_mask, n_active);
        }

        {
            std::lock_guard<std::mutex> mlk(_metric_mutex);
            _last_power_rms.resize(_num_elements);
            _last_clip_fraction.resize(_num_elements);
            _last_active_mask.resize(_num_elements);
            for (int a = 0; a < _num_elements; ++a) {
                _last_power_rms[a] = elem_power[a];
                _last_clip_fraction[a] = elem_clip_frac[a];
                _last_active_mask[a] = candidate_mask[a];
            }
        }
    } else {
        std::lock_guard<std::mutex> lk(_global_mutex);
        _current_bad_indices.clear();
        _current_alive_indices.clear();
        for (int a = 0; a < _num_elements; ++a) {
            if (_shared_mask[a] == 0) {
                _current_bad_indices.push_back(a);
            } else {
                _current_alive_indices.push_back(a);
            }
        }
        if (_applied_mask_generation != _shared_config.mask_generation) {
            _applied_mask_generation = _shared_config.mask_generation;
            const int n_active = static_cast<int>(_current_alive_indices.size());
            cudaDirectBeamTrackerCommand::set_shared_antenna_mask(_shared_mask, n_active);
        }

        {
            std::lock_guard<std::mutex> mlk(_metric_mutex);
            _last_power_rms.assign(_num_elements, 0.0f);
            _last_clip_fraction.assign(_num_elements, 0.0f);
            _last_active_mask.resize(_num_elements);
            for (int a = 0; a < _num_elements; ++a) {
                _last_active_mask[a] = _shared_mask[a];
            }
        }
    }

    // 2. Coalesced in-place voltage blanking for bad elements
    if (current_cfg.blank_voltages_enabled && !_current_bad_indices.empty()) {
        CHECK_CUDA_ERROR_NON_OO(cudaMemcpyAsync(_d_bad_antennas, _current_bad_indices.data(),
                                                _current_bad_indices.size() * sizeof(int),
                                                cudaMemcpyHostToDevice, stream));

        launch_zero_bad_antennas(d_voltages, _d_bad_antennas,
                                 static_cast<int>(_current_bad_indices.size()), total_spectra,
                                 _num_elements, stream);
    }

    // 3. Coalesced warp-per-spectrum alive antenna baseband extraction
    if (!_gpu_mem_alive_voltages.empty()) {
        const std::size_t alive_bytes = static_cast<std::size_t>(_max_alive_antennas)
                                        * static_cast<std::size_t>(_num_local_freq)
                                        * static_cast<std::size_t>(_samples_per_data_set)
                                        * sizeof(int4x2_t);
        void* alive_memory = device.get_gpu_memory_array(_gpu_mem_alive_voltages, gpu_frame_id,
                                                         _gpu_buffer_depth, alive_bytes);
        if (alive_memory) {
            const int num_to_extract = static_cast<int>(std::min(
                static_cast<std::size_t>(_max_alive_antennas), _current_alive_indices.size()));
            if (num_to_extract > 0) {
                CHECK_CUDA_ERROR_NON_OO(
                    cudaMemcpyAsync(_d_alive_antennas, _current_alive_indices.data(),
                                    num_to_extract * sizeof(int), cudaMemcpyHostToDevice, stream));
            }

            int4x2_t* d_alive_voltages = reinterpret_cast<int4x2_t*>(alive_memory);
            launch_extract_alive_antennas(d_voltages, d_alive_voltages, _d_alive_antennas,
                                          num_to_extract, _max_alive_antennas, total_spectra,
                                          _num_elements, stream);

            std::shared_ptr<metadataObject> meta =
                device.get_gpu_memory_array_metadata(_gpu_mem_voltage, gpu_frame_id);
            if (meta) {
                device.claim_gpu_memory_array_metadata(_gpu_mem_alive_voltages, gpu_frame_id, meta);
            }
        }
    }

    return record_end_event();
}

} // namespace kotekan
