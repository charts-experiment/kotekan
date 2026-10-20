#include "BasebandWriter.hpp"
#include <limits>

#include "BasebandFrameView.hpp" // for BasebandFrameView
#include "BasebandMetadata.hpp"  // for BasebandMetadata
#include "StageFactory.hpp"      // for REGISTER_KOTEKAN_STAGE
#include "kotekanLogging.hpp"    // for DEBUG, INFO, ERROR, FATAL_ERROR, WARN
#include "json.hpp"
#include "visUtil.hpp"           // for current_time, frameID, modulo, movingAverage

#include "fmt.hpp" // for compile_string_to_view, format, format_string

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <math.h>
#include <memory>
#include <ratio>
#include <set>
#include <stddef.h>
#include <stdexcept>
#include <sys/stat.h>
#include <thread>
#include <tuple>
#include <utility>

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using kotekan::prometheus::Metrics;

REGISTER_KOTEKAN_STAGE(BasebandWriter);

BasebandWriter::BasebandWriterDestination::BasebandWriterDestination(const std::string& file_name,
                                                                     const uint32_t& frame_size) :
    file(file_name, frame_size), last_updated(current_time()) {}


BasebandWriter::BasebandWriter(Config& config, const std::string& unique_name,
                               bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&BasebandWriter::main_thread, this)),

    _root_path(config.get_default<std::string>(unique_name, "root_path", ".")),
    _dump_timeout(config.get_default<double>(unique_name, "dump_timeout", 60)),
    _max_frames_per_second(config.get_default<double>(unique_name, "max_frames_per_second", 0)),
    _selected_antennas(config.get_default<std::vector<int>>(unique_name, "selected_antennas", {})),
    _antenna_index_convention(config.get_default<std::string>(
        unique_name, "antenna_index_convention", "charts_logical")),
    in_buf(get_buffer("in_buf")), write_in_progress_metric(Metrics::instance().add_gauge(
                                      "kotekan_baseband_writeout_in_progress", unique_name)),
    active_event_dumps_metric(
        Metrics::instance().add_gauge("kotekan_baseband_writeout_active_events", unique_name)),
    write_time_metric(
        Metrics::instance().add_gauge("kotekan_writer_write_time_seconds", unique_name)),
    bytes_written_metric(
        Metrics::instance().add_counter("kotekan_writer_bytes_total", unique_name)) {
    size_t metadata_size = BasebandMetadata().get_serialized_size();
    // The buffer defines the raw record size, including any frequency axis.
    if (in_buf->frame_size > std::numeric_limits<uint32_t>::max() - metadata_size)
        throw std::invalid_argument("BasebandWriter raw frame exceeds the 32-bit file format limit");
    _frame_size = in_buf->frame_size + metadata_size;
    _spectra_per_frame = config.get_default<uint64_t>(unique_name, "spectra_per_frame", 0);
    const auto readout_name = config.get_default<std::string>(unique_name, "readout_status", "");
    if (!readout_name.empty()) {
        if (_spectra_per_frame == 0 || in_buf->frame_size % _spectra_per_frame != 0)
            throw std::invalid_argument("BasebandWriter readout_status requires valid spectra_per_frame");
        _readout_status = kotekan::basebandReadoutManager::named(readout_name);
    }
    if (!_selected_antennas.empty()) {
        if (_selected_antennas.size() != config.get<uint32_t>(unique_name, "num_elements")
            || _antenna_index_convention.empty())
            throw std::invalid_argument("BasebandWriter antenna mapping does not match num_elements");
        std::set<int> unique_antennas(_selected_antennas.begin(), _selected_antennas.end());
        if (unique_antennas.size() != _selected_antennas.size()
            || *unique_antennas.begin() < 0)
            throw std::invalid_argument("BasebandWriter selected_antennas contains invalid indices");
    }
    in_buf->register_consumer(unique_name);

    INFO("BasebandMetaData size: {:d} bytes", metadata_size);
}

void BasebandWriter::ensure_antenna_mapping(const std::string& event_directory,
                                            uint64_t event_id) {
    if (_selected_antennas.empty())
        return;

    namespace fs = std::filesystem;
    const fs::path path = fs::path(event_directory) / "antenna_selection.json";
    const nlohmann::json expected = {
        {"schema_version", 1},
        {"event_id", event_id},
        {"index_convention", _antenna_index_convention},
        {"selected_antennas", _selected_antennas},
    };

    if (fs::exists(path)) {
        std::ifstream input(path);
        nlohmann::json existing;
        if (!input || !(input >> existing) || existing != expected)
            throw std::runtime_error("BasebandWriter existing antenna mapping differs from configuration: "
                                     + path.string());
        return;
    }

    // Never attach a new mapping to raw files whose original antenna order is unknown.
    for (const auto& entry : fs::directory_iterator(event_directory)) {
        const std::string name = entry.path().filename().string();
        if (name.rfind("baseband_", 0) == 0 && entry.path().extension() == ".data")
            throw std::runtime_error("BasebandWriter found raw data without antenna mapping: "
                                     + event_directory);
    }

    const fs::path temporary = path.string() + ".tmp";
    {
        std::ofstream output(temporary, std::ios::trunc);
        if (!output)
            throw std::runtime_error("BasebandWriter cannot open antenna mapping: "
                                     + temporary.string());
        output << expected.dump(2) << '\n';
        if (!output)
            throw std::runtime_error("BasebandWriter cannot write antenna mapping: "
                                     + temporary.string());
    }
    fs::rename(temporary, path);
}


void BasebandWriter::main_thread() {
    frameID frame_id(in_buf);
    std::thread closing_thread(&BasebandWriter::close_old_events, this);

    auto period_start = std::chrono::steady_clock::now();
    unsigned int frames_in_period = 0;

    while (!stop_thread) {
        // Wait for the buffer to be filled with data
        if (in_buf->wait_for_full_frame(unique_name, frame_id) == nullptr) {
            break;
        }

        // Write the frame to its event+frequency destination file
        if (_readout_status) {
            try {
                write_data(in_buf, frame_id);
            } catch (const std::exception& ex) {
                const auto meta = BasebandFrameView(in_buf, frame_id).metadata();
                _readout_status->written(meta->event_id, 0, ex.what());
                ERROR("Baseband event {} write failed: {}", meta->event_id, ex.what());
            }
        } else {
            write_data(in_buf, frame_id);
        }

        const auto now = std::chrono::steady_clock::now();
        const std::chrono::duration<double> diff = now - period_start;
        if (diff > std::chrono::seconds(1)) {
            DEBUG("Restart step count ({:.3f} s)", diff.count());
            period_start = now;
            frames_in_period = 1;
        } else {
            ++frames_in_period;

            double period_throughput = frames_in_period / diff.count();
            DEBUG("Current step count: {} ({:.2f})", frames_in_period, period_throughput);
            if (_max_frames_per_second > 0 && period_throughput > _max_frames_per_second) {
                WARN("Throughput exceeded: {:.2f} >> {}", period_throughput,
                     _max_frames_per_second);
                const std::chrono::duration<double> remaining_in_period =
                    std::chrono::seconds(1) - diff;
                if (frames_in_period >= _max_frames_per_second) {
                    INFO("Sleep until the end of the period: {:.3}s", remaining_in_period.count());
                } else {
                    const auto throttle_rate =
                        remaining_in_period / (_max_frames_per_second - frames_in_period);
                    INFO("Over the quota after {} frames; ({}s remaining, throttle by {:.3} s)",
                         frames_in_period, remaining_in_period.count(), throttle_rate.count());
                    std::this_thread::sleep_for(throttle_rate);
                }
            }
        }

        // Mark the buffer and move on
        in_buf->mark_frame_empty(unique_name, frame_id++);
    }

    {
        std::lock_guard<std::mutex> lk(mtx);
        closing_stop = true;
    }
    stop_closing.notify_one();
    closing_thread.join();
}


void BasebandWriter::write_data(Buffer* in_buf, int frame_id) {
    const auto frame = BasebandFrameView(in_buf, frame_id);
    const auto metadata = frame.metadata();

    const auto event_id = metadata->event_id;
    const auto freq_id = metadata->freq_id;
    DEBUG("Frame {} from {}/{}", metadata->frame_fpga_seq, event_id, freq_id);

    write_in_progress_metric.set(1);

    // Lock the event->freq->file map
    std::unique_lock lk(mtx);
    active_event_dumps_metric.set(baseband_events.size());

    const std::string event_directory_name =
        fmt::format("{:s}/baseband_raw_{:d}", _root_path, event_id);
    if (baseband_events.count(event_id) == 0) {
        std::filesystem::create_directories(_root_path);
        mkdir(event_directory_name.c_str(), S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);
        ensure_antenna_mapping(event_directory_name, event_id);
    }

    const std::string file_name =
        fmt::format("{:s}/baseband_{:d}_{:d}", event_directory_name, event_id, freq_id);
    if (baseband_events[event_id].count(freq_id) == 0) {
        // NOTE: emplace the file instance or it will get closed by the destructor
        baseband_events[event_id].emplace(std::piecewise_construct, std::forward_as_tuple(freq_id),
                                          std::forward_as_tuple(file_name, _frame_size));
    }
    auto& freq_dump_destination = baseband_events[event_id].at(freq_id);
    BasebandFileRaw& baseband_file = freq_dump_destination.file;

    const double start = current_time();
    int32_t write_status = baseband_file.write_frame({in_buf, frame_id});
    freq_dump_destination.last_updated = current_time();
    const double elapsed = freq_dump_destination.last_updated - start;

    write_in_progress_metric.set(0);

    if (write_status == -1) {
        if (_readout_status)
            _readout_status->written(event_id, 0, "baseband output file is corrupt");
        ERROR("Output file is corrupt, dropping frame going to {:s}", file_name);
    } else if (write_status == 0) {
        if (_readout_status) {
            _readout_status->written(event_id, 0, "baseband disk write failed");
            ERROR("Cannot write baseband file {:s}; capture is incomplete", file_name);
        } else {
            FATAL_ERROR("Cannot write to file {:s}, maybe out of disk space?", file_name);
        }
    } else {
        if (_readout_status) {
            if (metadata->valid_to <= 0
                || static_cast<uint64_t>(metadata->valid_to) > _spectra_per_frame)
                _readout_status->written(event_id, 0, "invalid baseband valid_to");
            else
                _readout_status->written(event_id, metadata->valid_to
                    * (in_buf->frame_size / _spectra_per_frame));
        }
        DEBUG("Written frame with event id {:d} and freq {:d} to {:s}", event_id, freq_id,
              file_name);

        bytes_written_metric.inc(frame.data_size() + metadata->get_serialized_size());

        // Update average write time in prometheus
        write_time.add_sample(elapsed);
        write_time_metric.set(write_time.average());
    }
}


void BasebandWriter::close_old_events() {
    DEBUG("Starting the file-closing thread");
    // Do not run the loop more often than once a minute
    const int sweep_cadence_s = std::max(60.0, round(_dump_timeout / 2));
    while (!stop_thread) {
        double now = current_time();
        DEBUG("Run file-closing thread {:.1f}", now);
        std::unique_lock lk(mtx);
        if (stop_closing.wait_for(lk, std::chrono::seconds(sweep_cadence_s),
                                  [this]() { return closing_stop; })) {
            // notified to exit (the predicate also filters spurious wakeups)
            return;
        }

        // Otherwise, we've waited long enough and can do the sweep
        for (auto event_it = baseband_events.begin(); event_it != baseband_events.end();) {
            for (auto event_freq = event_it->second.begin();
                 event_freq != event_it->second.end();) {
                // close the frequency file that's been inactive for over a minute
                if (now - event_freq->second.last_updated > _dump_timeout) {
                    INFO("Closing {}", event_freq->second.file.name);
                    event_freq = event_it->second.erase(event_freq);
                } else {
                    ++event_freq;
                }
            }
            if (event_it->second.empty()) {
                event_it = baseband_events.erase(event_it);
            } else {
                ++event_it;
            }
        }
        active_event_dumps_metric.set(baseband_events.size());
    }
    DEBUG("File-closing thread done");
}
