#include "cudaRfiMitigationCommand.hpp"

#include "chartsConstants.hpp"
#include "chartsMetadata.hpp"
#include "chordMetadata.hpp"
#include "cudaAntennaMaskCommand.hpp"
#include "cudaDirectBeamTrackerCommand.hpp"
#include "cudaUtils.hpp"
#include "gpuCommand.hpp"
#include "kotekanLogging.hpp"
#include "restServer.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fmt/format.h>
#include <map>

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::connectionInstance;
using kotekan::cudaRfiMitigationCommand;
using kotekan::HTTP_RESPONSE;
using kotekan::restServer;

REGISTER_CUDA_COMMAND(cudaRfiMitigationCommand);

namespace kotekan {

std::mutex cudaRfiMitigationCommand::_global_mutex;
RfiMitigationConfig cudaRfiMitigationCommand::_shared_config;
std::uint64_t cudaRfiMitigationCommand::_config_generation = 1;
std::vector<cudaRfiMitigationCommand*> cudaRfiMitigationCommand::_instances;
bool cudaRfiMitigationCommand::_endpoints_registered = false;

cudaRfiMitigationCommand::cudaRfiMitigationCommand(Config& config, const std::string& unique_name,
                                                   bufferContainer& host_buffers,
                                                   cudaDeviceInterface& device, int inst) :
    cudaCommand(config, unique_name, host_buffers, device, inst, no_cuda_command_state,
                "cudaRfiMitigationCommand") {

    _num_elements = config.get_default<int32_t>(unique_name, "num_elements", 64);
    _num_local_freq = config.get_default<int32_t>(unique_name, "num_local_freq", 336);
    _samples_per_data_set = config.get_default<int32_t>(unique_name, "samples_per_data_set", 15360);
    _buffer_depth = config.get_default<int32_t>(unique_name, "buffer_depth", 3);

    _gpu_mem_voltage = config.get_default<std::string>(unique_name, "gpu_mem_voltage", "voltage_0");
    _gpu_mem_rfi_mask = config.get_default<std::string>(unique_name, "gpu_mem_rfi_mask", "");
    _antenna_order = config.get_default<std::string>(unique_name, "antenna_order", "descending");

    const bool is_subband_1 = (_gpu_mem_voltage.find("_1") != std::string::npos);
    const double default_f0 =
        is_subband_1 ? (charts::constants::charts_frequency_start_hz
                        + 336.0 * charts::constants::charts_channel_width_hz)
                     : charts::constants::charts_frequency_start_hz;
    _freq_start_hz = config.get_default<double>(unique_name, "freq_start_hz", default_f0);
    _freq_step_hz = config.get_default<double>(unique_name, "freq_step_hz",
                                               charts::constants::charts_channel_width_hz);

    const bool desc = (_antenna_order == "descending");
    auto active_ants = config.get_default<std::vector<int>>(unique_name, "active_antennas", {});
    if (!active_ants.empty()) {
        for (int phys : active_ants) {
            if (phys >= 0 && phys < _num_elements) {
                int raw = desc ? ((_num_elements - 1) - phys) : phys;
                _configured_active_raw.push_back(raw);
            }
        }
    } else {
        for (int r = 0; r < _num_elements; ++r) {
            _configured_active_raw.push_back(r);
        }
    }
    _current_active_raw = _configured_active_raw;

    _excision_exempt_freq_ids = config.get_default<std::vector<int>>(
        unique_name, "rfi_first_stage_excision_exempt_freq_ids", {});
    if (_excision_exempt_freq_ids.empty()) {
        _excision_exempt_freq_ids = config.get_default<std::vector<int>>(
            unique_name, "rfi_excision_exempt_freq_ids", {});
    }
    auto exempt_chans = config.get_default<std::vector<int>>(
        unique_name, "rfi_excision_exempt_local_freqs", {});
    _h_exempt_channels.assign(static_cast<std::size_t>(_num_local_freq), 0);
    for (int ch : exempt_chans) {
        if (ch >= 0 && ch < _num_local_freq) {
            _h_exempt_channels[ch] = 1;
        }
    }

    RfiMitigationConfig init_cfg;
    init_cfg.enabled = config.get_default<bool>(unique_name, "enabled", true);
    init_cfg.blank_voltages_enabled =
        config.get_default<bool>(unique_name, "blank_voltages_enabled", true);
    init_cfg.time_chunk_size = static_cast<std::size_t>(
        std::max<int32_t>(16, config.get_default<int32_t>(unique_name, "time_chunk_size", 256)));
    init_cfg.sample_stride = static_cast<std::size_t>(
        std::max<int32_t>(1, config.get_default<int32_t>(unique_name, "sample_stride", 1)));

    init_cfg.enable_sk_check = config.get_default<bool>(unique_name, "enable_sk_check", true);
    init_cfg.sk_sigma_threshold =
        config.get_default<float>(unique_name, "sk_sigma_threshold", 4.5f);

    init_cfg.enable_time_burst_check =
        config.get_default<bool>(unique_name, "enable_time_burst_check", true);
    init_cfg.burst_sigma_threshold =
        config.get_default<float>(unique_name, "burst_sigma_threshold", 5.0f);

    init_cfg.enable_freq_mad_check =
        config.get_default<bool>(unique_name, "enable_freq_mad_check", true);
    init_cfg.freq_window_half_width =
        std::max<int>(2, config.get_default<int>(unique_name, "freq_window_half_width", 8));
    init_cfg.mad_sigma_threshold =
        config.get_default<float>(unique_name, "mad_sigma_threshold", 6.0f);

    init_cfg.enable_saturation_check =
        config.get_default<bool>(unique_name, "enable_saturation_check", true);
    init_cfg.saturation_fraction_threshold =
        config.get_default<float>(unique_name, "saturation_fraction_threshold", 0.02f);
    init_cfg.enable_morphology_sir =
        config.get_default<bool>(unique_name, "enable_morphology_sir", true);
    init_cfg.edge_channels_to_ignore =
        config.get_default<int>(unique_name, "edge_channels_to_ignore", 16);

    auto notch_flat =
        config.get_default<std::vector<double>>(unique_name, "static_notch_bands_mhz", {});
    for (std::size_t i = 0; i + 1 < notch_flat.size(); i += 2) {
        init_cfg.static_notch_bands_mhz.emplace_back(notch_flat[i], notch_flat[i + 1]);
    }

    _num_chunks =
        (static_cast<std::size_t>(_samples_per_data_set) + init_cfg.time_chunk_size - 1)
        / init_cfg.time_chunk_size;
    _last_channel_stats.resize(static_cast<std::size_t>(_num_local_freq));

    {
        std::lock_guard<std::mutex> lk(_global_mutex);
        _shared_config = init_cfg;
        _config_generation++;
        _instances.push_back(this);

        if (!_endpoints_registered) {
            restServer& rest = restServer::instance();

            rest.register_get_callback("/rfi_mitigation/status", [](connectionInstance& conn) {
                nlohmann::json reply;
                RfiMitigationConfig cfg;
                std::vector<cudaRfiMitigationCommand*> sorted_insts;
                {
                    std::lock_guard<std::mutex> glk(_global_mutex);
                    cfg = _shared_config;
                    std::map<double, cudaRfiMitigationCommand*> freq_map;
                    for (auto* ptr : _instances) {
                        auto it = freq_map.find(ptr->get_freq_start_hz());
                        if (it == freq_map.end()
                            || ptr->get_frame_counter() > it->second->get_frame_counter()) {
                            freq_map[ptr->get_freq_start_hz()] = ptr;
                        }
                    }
                    for (auto& [f0, ptr] : freq_map) {
                        sorted_insts.push_back(ptr);
                    }
                }

                reply["enabled"] = cfg.enabled;
                reply["blank_voltages_enabled"] = cfg.blank_voltages_enabled;
                reply["time_chunk_size"] = cfg.time_chunk_size;
                reply["sample_stride"] = cfg.sample_stride;
                reply["enable_sk_check"] = cfg.enable_sk_check;
                reply["sk_sigma_threshold"] = cfg.sk_sigma_threshold;
                reply["enable_time_burst_check"] = cfg.enable_time_burst_check;
                reply["burst_sigma_threshold"] = cfg.burst_sigma_threshold;
                reply["enable_freq_mad_check"] = cfg.enable_freq_mad_check;
                reply["freq_window_half_width"] = cfg.freq_window_half_width;
                reply["mad_sigma_threshold"] = cfg.mad_sigma_threshold;
                reply["enable_saturation_check"] = cfg.enable_saturation_check;
                reply["saturation_fraction_threshold"] = cfg.saturation_fraction_threshold;
                reply["enable_morphology_sir"] = cfg.enable_morphology_sir;
                reply["edge_channels_to_ignore"] = cfg.edge_channels_to_ignore;

                int total_channels = 0;
                int flagged_channels = 0;
                double sum_flagged_frac = 0.0;
                nlohmann::json top_rfi = nlohmann::json::array();

                struct ChanEntry {
                    double freq_mhz;
                    RfiChannelStats st;
                };
                std::vector<ChanEntry> flagged_list;

                for (auto* inst_ptr : sorted_insts) {
                    const auto stats = inst_ptr->get_last_channel_stats();
                    const double f_start = inst_ptr->get_freq_start_hz();
                    const double f_step = inst_ptr->get_freq_step_hz();
                    for (std::size_t f = 0; f < stats.size(); ++f) {
                        total_channels++;
                        sum_flagged_frac += stats[f].flagged_chunk_fraction;
                        if (stats[f].flag_bits != 0) {
                            flagged_channels++;
                            flagged_list.push_back(
                                {(f_start + static_cast<double>(f) * f_step) * 1.0e-6, stats[f]});
                        }
                    }
                }

                std::sort(flagged_list.begin(), flagged_list.end(),
                          [](const ChanEntry& a, const ChanEntry& b) {
                              if (a.st.flagged_chunk_fraction != b.st.flagged_chunk_fraction)
                                  return a.st.flagged_chunk_fraction > b.st.flagged_chunk_fraction;
                              return a.st.max_sk > b.st.max_sk;
                          });

                const std::size_t n_top = std::min<std::size_t>(20, flagged_list.size());
                for (std::size_t i = 0; i < n_top; ++i) {
                    nlohmann::json item;
                    item["freq_mhz"] = flagged_list[i].freq_mhz;
                    item["mean_power"] = flagged_list[i].st.mean_power;
                    item["mean_sk"] = flagged_list[i].st.mean_sk;
                    item["max_sk"] = flagged_list[i].st.max_sk;
                    item["freq_mad_z"] = flagged_list[i].st.freq_mad_z;
                    item["flagged_chunk_fraction"] = flagged_list[i].st.flagged_chunk_fraction;
                    item["flag_bits"] = flagged_list[i].st.flag_bits;
                    top_rfi.push_back(item);
                }

                reply["total_channels"] = total_channels;
                reply["flagged_channels"] = flagged_channels;
                reply["overall_flagged_chunk_fraction"] =
                    (total_channels > 0) ? (sum_flagged_frac / total_channels) : 0.0;
                reply["top_rfi_channels"] = top_rfi;
                conn.send_json_reply(reply);
            });

            rest.register_post_callback(
                "/rfi_mitigation/configure", [](connectionInstance& conn, nlohmann::json& j) {
                    try {
                        std::lock_guard<std::mutex> glk(_global_mutex);
                        if (j.contains("enabled"))
                            _shared_config.enabled = j["enabled"].get<bool>();
                        if (j.contains("blank_voltages_enabled"))
                            _shared_config.blank_voltages_enabled =
                                j["blank_voltages_enabled"].get<bool>();
                        if (j.contains("enable_sk_check"))
                            _shared_config.enable_sk_check = j["enable_sk_check"].get<bool>();
                        if (j.contains("sk_sigma_threshold"))
                            _shared_config.sk_sigma_threshold = j["sk_sigma_threshold"].get<float>();
                        if (j.contains("enable_time_burst_check"))
                            _shared_config.enable_time_burst_check =
                                j["enable_time_burst_check"].get<bool>();
                        if (j.contains("burst_sigma_threshold"))
                            _shared_config.burst_sigma_threshold =
                                j["burst_sigma_threshold"].get<float>();
                        if (j.contains("enable_freq_mad_check"))
                            _shared_config.enable_freq_mad_check =
                                j["enable_freq_mad_check"].get<bool>();
                        if (j.contains("freq_window_half_width"))
                            _shared_config.freq_window_half_width =
                                std::max<int>(2, j["freq_window_half_width"].get<int>());
                        if (j.contains("mad_sigma_threshold"))
                            _shared_config.mad_sigma_threshold =
                                j["mad_sigma_threshold"].get<float>();
                        if (j.contains("enable_saturation_check"))
                            _shared_config.enable_saturation_check =
                                j["enable_saturation_check"].get<bool>();
                        if (j.contains("saturation_fraction_threshold"))
                            _shared_config.saturation_fraction_threshold =
                                j["saturation_fraction_threshold"].get<float>();
                        if (j.contains("enable_morphology_sir"))
                            _shared_config.enable_morphology_sir =
                                j["enable_morphology_sir"].get<bool>();
                        if (j.contains("edge_channels_to_ignore"))
                            _shared_config.edge_channels_to_ignore =
                                std::max<int>(0, j["edge_channels_to_ignore"].get<int>());

                        _config_generation++;

                        conn.send_text_reply(fmt::format(
                            "RFI mitigation updated: enabled={:s}, blank={:s}, sk_sigma={:.2f}, "
                            "burst_sigma={:.2f}, mad_sigma={:.2f}, sat_check={:s}\n",
                            _shared_config.enabled ? "true" : "false",
                            _shared_config.blank_voltages_enabled ? "true" : "false",
                            _shared_config.sk_sigma_threshold, _shared_config.burst_sigma_threshold,
                            _shared_config.mad_sigma_threshold,
                            _shared_config.enable_saturation_check ? "true" : "false"));
                    } catch (const std::exception& e) {
                        conn.send_error(
                            fmt::format("Error configuring RFI mitigation: {:s}\n", e.what()),
                            HTTP_RESPONSE::BAD_REQUEST);
                    }
                });

            rest.register_post_callback(
                "/rfi_mitigation/set_gains", [](connectionInstance& conn, nlohmann::json& j) {
                    try {
                        if (!j.contains("freq_start_hz") || !j.contains("gains")) {
                            conn.send_error("Missing freq_start_hz or gains in JSON\n",
                                            HTTP_RESPONSE::BAD_REQUEST);
                            return;
                        }
                        const double f0 = j["freq_start_hz"].get<double>();
                        const auto gains = j["gains"].get<std::vector<float>>();
                        const bool ok = set_channel_gains_by_freq(f0, gains);
                        if (ok) {
                            conn.send_text_reply(fmt::format(
                                "Updated {:d} digital calibration gains for subband f0={:.1f} MHz\n",
                                static_cast<int>(gains.size()), f0 * 1.0e-6));
                        } else {
                            conn.send_error(fmt::format(
                                "No active RFI instance matching freq_start_hz={:.1f} MHz\n",
                                f0 * 1.0e-6),
                                HTTP_RESPONSE::NOT_FOUND);
                        }
                    } catch (const std::exception& e) {
                        conn.send_error(fmt::format("Error setting gains: {:s}\n", e.what()),
                                        HTTP_RESPONSE::BAD_REQUEST);
                    }
                });

            _endpoints_registered = true;
        }
    }

    set_command_type(gpuCommandType::KERNEL);
    set_name("cudaRfiMitigationCommand");
    gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_voltage, true, true, true));
    if (!_gpu_mem_rfi_mask.empty()) {
        gpu_buffers_used.push_back(std::make_tuple(_gpu_mem_rfi_mask, false, true, true));
        device.register_gpu_memory_name(_gpu_mem_rfi_mask);
    }

    allocate_device_buffers();
}

cudaRfiMitigationCommand::~cudaRfiMitigationCommand() {
    {
        std::lock_guard<std::mutex> lk(_global_mutex);
        _instances.erase(std::remove(_instances.begin(), _instances.end(), this), _instances.end());
    }
    free_device_buffers();
}

void cudaRfiMitigationCommand::allocate_device_buffers() {
    const std::size_t n_cells = _num_chunks * static_cast<std::size_t>(_num_local_freq);
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&_d_active_antennas, _num_elements * sizeof(int)));
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&_d_chunk_power, n_cells * sizeof(float)));
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&_d_chunk_sk, n_cells * sizeof(float)));
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&_d_chunk_sat_frac, n_cells * sizeof(float)));
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&_d_static_notch, _num_local_freq * sizeof(std::uint8_t)));
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&_d_channel_gains, _num_local_freq * sizeof(float)));
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&_d_chunk_rfi_mask, n_cells * sizeof(std::uint8_t)));
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&_d_channel_stats, _num_local_freq * sizeof(RfiChannelStats)));
    CHECK_CUDA_ERROR_NON_OO(cudaMalloc(&_d_exempt_channels, _num_local_freq * sizeof(std::uint8_t)));
    CHECK_CUDA_ERROR_NON_OO(
        cudaMallocHost(&_h_channel_stats, _num_local_freq * sizeof(RfiChannelStats)));

    CHECK_CUDA_ERROR_NON_OO(cudaMemset(_d_chunk_power, 0, n_cells * sizeof(float)));
    CHECK_CUDA_ERROR_NON_OO(cudaMemset(_d_chunk_sk, 0, n_cells * sizeof(float)));
    CHECK_CUDA_ERROR_NON_OO(cudaMemset(_d_chunk_sat_frac, 0, n_cells * sizeof(float)));
    CHECK_CUDA_ERROR_NON_OO(cudaMemset(_d_channel_gains, 0, _num_local_freq * sizeof(float)));
    CHECK_CUDA_ERROR_NON_OO(cudaMemset(_d_chunk_rfi_mask, 0, n_cells * sizeof(std::uint8_t)));
    CHECK_CUDA_ERROR_NON_OO(cudaMemcpy(_d_exempt_channels, _h_exempt_channels.data(),
                                       _num_local_freq * sizeof(std::uint8_t),
                                       cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR_NON_OO(
        cudaMemset(_d_channel_stats, 0, _num_local_freq * sizeof(RfiChannelStats)));
    for (int f = 0; f < _num_local_freq; ++f) {
        _h_channel_stats[f] = RfiChannelStats{};
    }

    if (!_current_active_raw.empty()) {
        CHECK_CUDA_ERROR_NON_OO(cudaMemcpy(_d_active_antennas, _current_active_raw.data(),
                                           _current_active_raw.size() * sizeof(int),
                                           cudaMemcpyHostToDevice));
    }

    rebuild_static_notch_mask(get_config());
}

void cudaRfiMitigationCommand::free_device_buffers() {
    if (_d_active_antennas) {
        cudaFree(_d_active_antennas);
        _d_active_antennas = nullptr;
    }
    if (_d_chunk_power) {
        cudaFree(_d_chunk_power);
        _d_chunk_power = nullptr;
    }
    if (_d_chunk_sk) {
        cudaFree(_d_chunk_sk);
        _d_chunk_sk = nullptr;
    }
    if (_d_chunk_sat_frac) {
        cudaFree(_d_chunk_sat_frac);
        _d_chunk_sat_frac = nullptr;
    }
    if (_d_static_notch) {
        cudaFree(_d_static_notch);
        _d_static_notch = nullptr;
    }
    if (_d_channel_gains) {
        cudaFree(_d_channel_gains);
        _d_channel_gains = nullptr;
    }
    if (_d_chunk_rfi_mask) {
        cudaFree(_d_chunk_rfi_mask);
        _d_chunk_rfi_mask = nullptr;
    }
    if (_d_exempt_channels) {
        cudaFree(_d_exempt_channels);
        _d_exempt_channels = nullptr;
    }
    if (_d_channel_stats) {
        cudaFree(_d_channel_stats);
        _d_channel_stats = nullptr;
    }
    if (_h_channel_stats) {
        cudaFreeHost(_h_channel_stats);
        _h_channel_stats = nullptr;
    }
}

void cudaRfiMitigationCommand::rebuild_static_notch_mask(const RfiMitigationConfig& cfg) {
    std::vector<std::uint8_t> h_notch(static_cast<std::size_t>(_num_local_freq), 0);
    for (int f = 0; f < _num_local_freq; ++f) {
        const double f_mhz = (_freq_start_hz + static_cast<double>(f) * _freq_step_hz) * 1.0e-6;
        for (const auto& [lo_mhz, hi_mhz] : cfg.static_notch_bands_mhz) {
            if (f_mhz >= lo_mhz && f_mhz <= hi_mhz) {
                h_notch[static_cast<std::size_t>(f)] = 1;
                break;
            }
        }
    }
    CHECK_CUDA_ERROR_NON_OO(cudaMemcpy(_d_static_notch, h_notch.data(),
                                        _num_local_freq * sizeof(std::uint8_t),
                                        cudaMemcpyHostToDevice));
}

void cudaRfiMitigationCommand::set_local_channel_gains(const std::vector<float>& gains) {
    const std::size_t n = std::min<std::size_t>(gains.size(), static_cast<std::size_t>(_num_local_freq));
    if (n > 0 && _d_channel_gains != nullptr) {
        CHECK_CUDA_ERROR_NON_OO(cudaMemcpy(_d_channel_gains, gains.data(), n * sizeof(float),
                                           cudaMemcpyHostToDevice));
    }
}

bool cudaRfiMitigationCommand::set_channel_gains_by_freq(double freq_start_hz,
                                                         const std::vector<float>& gains) {
    std::lock_guard<std::mutex> lk(_global_mutex);
    bool found = false;
    for (auto* ptr : _instances) {
        if (std::abs(ptr->get_freq_start_hz() - freq_start_hz) < 1.0e5) {
            ptr->set_local_channel_gains(gains);
            found = true;
        }
    }
    return found;
}

RfiMitigationConfig cudaRfiMitigationCommand::get_config() {
    std::lock_guard<std::mutex> lk(_global_mutex);
    return _shared_config;
}

void cudaRfiMitigationCommand::update_config(const RfiMitigationConfig& cfg) {
    std::lock_guard<std::mutex> lk(_global_mutex);
    _shared_config = cfg;
    _config_generation++;
}

cudaEvent_t cudaRfiMitigationCommand::execute(cudaPipelineState& /*pipestate*/,
                                              const std::vector<cudaEvent_t>&) {
    pre_execute();

    const std::size_t frame_bytes = static_cast<std::size_t>(_samples_per_data_set)
                                    * static_cast<std::size_t>(_num_local_freq)
                                    * static_cast<std::size_t>(_num_elements)
                                    * sizeof(int4x2_t);

    auto* d_voltages = reinterpret_cast<int4x2_t*>(
        device.get_gpu_memory_array(_gpu_mem_voltage, gpu_frame_id, _gpu_buffer_depth, frame_bytes));
    if (!d_voltages) {
        return record_end_event();
    }

    record_start_event();
    cudaStream_t stream = device.getStream(cuda_stream_id);

    bool is_swapped_withoffset = false;
    auto meta =
        device.get_gpu_memory_array_metadata(_gpu_mem_voltage, gpu_frame_id);
    if (auto c_meta = std::dynamic_pointer_cast<chartsMetadata>(meta)) {
        if (c_meta->type == kotekan::int4x2_swapped_withoffset) {
            is_swapped_withoffset = true;
        }
        if (c_meta->has_coarse_freq()) {
            const auto cf = c_meta->get_coarse_freq();
            if (!cf.empty()) {
                _freq_start_hz = charts::constants::charts_frequency_start_hz
                                 + static_cast<double>(cf.front()) * _freq_step_hz;
                if (!_excision_exempt_freq_ids.empty() && !_exempt_resolved && _d_exempt_channels != nullptr) {
                    for (int f = 0; f < _num_local_freq && f < static_cast<int>(cf.size()); ++f) {
                        if (std::find(_excision_exempt_freq_ids.begin(), _excision_exempt_freq_ids.end(), cf[f])
                            != _excision_exempt_freq_ids.end()) {
                            _h_exempt_channels[f] = 1;
                        }
                    }
                    CHECK_CUDA_ERROR_NON_OO(cudaMemcpyAsync(_d_exempt_channels, _h_exempt_channels.data(),
                                                            _num_local_freq * sizeof(std::uint8_t),
                                                            cudaMemcpyHostToDevice, stream));
                    _exempt_resolved = true;
                }
            }
        }
    } else if (auto ch_meta = std::dynamic_pointer_cast<chordMetadata>(meta)) {
        if (ch_meta->type == kotekan::int4x2_swapped_withoffset) {
            is_swapped_withoffset = true;
        }
    }

    RfiMitigationConfig cfg;
    std::uint64_t cfg_gen = 0;
    {
        std::lock_guard<std::mutex> lk(_global_mutex);
        cfg = _shared_config;
        cfg_gen = _config_generation;
    }

    if (_applied_rfi_cfg_gen != cfg_gen) {
        rebuild_static_notch_mask(cfg);
        _applied_rfi_cfg_gen = cfg_gen;
    }

    // Sync active antenna mask from cudaAntennaMaskCommand and tracker calibration mask
    const auto mask_cfg = cudaAntennaMaskCommand::get_config();
    const auto tracker_cfg = cudaDirectBeamTrackerCommand::get_shared_config();
    const std::uint64_t combined_gen = mask_cfg.mask_generation ^ (tracker_cfg.mask_generation << 16);
    if (_applied_mask_gen != combined_gen) {
        const auto ant_mask = cudaAntennaMaskCommand::get_current_mask();
        const auto tracker_mask = cudaDirectBeamTrackerCommand::get_shared_antenna_mask();
        std::vector<int> alive;
        for (int r = 0; r < _num_elements && r < static_cast<int>(MAX_RFI_ANTENNAS); ++r) {
            bool is_alive = (ant_mask[static_cast<std::size_t>(r)] != 0);
            if (tracker_cfg.calibration_mask_initialized) {
                is_alive = is_alive && (tracker_mask[static_cast<std::size_t>(r)] != 0);
            }
            if (is_alive) {
                alive.push_back(r);
            }
        }
        if (alive.empty()) {
            alive = _configured_active_raw;
        }
        _current_active_raw = alive;
        CHECK_CUDA_ERROR_NON_OO(cudaMemcpyAsync(_d_active_antennas, _current_active_raw.data(),
                                                _current_active_raw.size() * sizeof(int),
                                                cudaMemcpyHostToDevice, stream));
        _applied_mask_gen = combined_gen;
    }

    if (cfg.enabled && !_current_active_raw.empty()) {
        const int num_active = static_cast<int>(_current_active_raw.size());
        const std::size_t stride = std::max<std::size_t>(1, cfg.sample_stride);
        const std::size_t eff_m = (cfg.time_chunk_size + stride - 1) / stride;

        // Pass 1: Per-(chunk, freq) scale-invariant Spectral Kurtosis, mean power, and saturation
        launch_inspect_rfi_chunks(d_voltages, _d_active_antennas, num_active, _d_chunk_power,
                                  _d_chunk_sk, _d_chunk_sat_frac,
                                  static_cast<std::size_t>(_samples_per_data_set),
                                  static_cast<std::size_t>(_num_local_freq),
                                  static_cast<std::size_t>(_num_elements), cfg.time_chunk_size,
                                  stride, stream, is_swapped_withoffset);

        // Pass 2: Fused temporal burst MAD, iterative outlier-rejected passband MAD, SK & saturation
        launch_evaluate_rfi_flags(_d_chunk_power, _d_chunk_sk, _d_chunk_sat_frac, _d_static_notch,
                                  _d_channel_gains, _d_chunk_rfi_mask, _d_channel_stats, _num_chunks,
                                  static_cast<std::size_t>(_num_local_freq), num_active, eff_m, cfg,
                                  stream, _d_exempt_channels);

        // Pass 3: Optional in-place voltage blanking of flagged (chunk, freq) cells
        if (cfg.blank_voltages_enabled) {
            launch_blank_rfi_chunks(d_voltages, _d_chunk_rfi_mask,
                                    static_cast<std::size_t>(_samples_per_data_set),
                                    static_cast<std::size_t>(_num_local_freq),
                                    static_cast<std::size_t>(_num_elements), cfg.time_chunk_size,
                                    stream, is_swapped_withoffset);
        }

        if (!_gpu_mem_rfi_mask.empty()) {
            const std::size_t mask_bytes =
                _num_chunks * static_cast<std::size_t>(_num_local_freq) * sizeof(std::uint8_t);
            void* d_out_mask = device.get_gpu_memory_array(_gpu_mem_rfi_mask, gpu_frame_id,
                                                           _gpu_buffer_depth, mask_bytes);
            if (d_out_mask) {
                CHECK_CUDA_ERROR_NON_OO(cudaMemcpyAsync(d_out_mask, _d_chunk_rfi_mask, mask_bytes,
                                                        cudaMemcpyDeviceToDevice, stream));
                if (meta) {
                    device.claim_gpu_memory_array_metadata(_gpu_mem_rfi_mask, gpu_frame_id, meta);
                }
            }
        }

        // Copy lightweight per-channel summary (336 * 24B = 8 KB) to pinned host for REST telemetry
        CHECK_CUDA_ERROR_NON_OO(cudaMemcpyAsync(_h_channel_stats, _d_channel_stats,
                                                _num_local_freq * sizeof(RfiChannelStats),
                                                cudaMemcpyDeviceToHost, stream));
        {
            std::lock_guard<std::mutex> lk(_stats_mutex);
            for (int f = 0; f < _num_local_freq; ++f) {
                _last_channel_stats[static_cast<std::size_t>(f)] = _h_channel_stats[f];
            }
        }
    }

    _frame_counter++;
    return record_end_event();
}

} // namespace kotekan
