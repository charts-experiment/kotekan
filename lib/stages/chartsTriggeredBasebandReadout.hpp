#ifndef CHARTS_TRIGGERED_BASEBAND_READOUT_HPP
#define CHARTS_TRIGGERED_BASEBAND_READOUT_HPP

#include "Stage.hpp"
#include "basebandReadoutManager.hpp"
#include "json.hpp"

#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <vector>

/**
 * @brief Retain CHARTS voltage history and extract REST-triggered intervals while correlating.
 *
 * Input buffers contain [time, frequency, raw element] packed complex bytes. Output buffers
 * contain selected logical antennas in request-independent chunks with BasebandMetadata.
 * One stage coordinates all frequency streams using a common spectrum counter interval.
 * See charts/triggered_baseband.md for configuration and API semantics.
 */
class chartsTriggeredBasebandReadout : public kotekan::Stage {
public:
    chartsTriggeredBasebandReadout(kotekan::Config&, const std::string&, kotekan::bufferContainer&);
    ~chartsTriggeredBasebandReadout() override;
    void main_thread() override;

    /// Submit a validated trigger. Also used by the REST callback.
    nlohmann::json trigger(const nlohmann::json& request);
    /// Return stream frontiers and event capture/disk-write status.
    nlohmann::json status();

private:
    struct RetainedFrame {
        int id;
        int64_t start;
    };
    struct Stream {
        Buffer* input;
        Buffer* output;
        uint64_t freq_id;
        size_t frequencies;
        size_t input_stride;
        size_t output_stride;
        int output_id = 0;
        std::deque<RetainedFrame> frames;
        bool running = true;
    };

    void ingest(size_t stream);
    void readout();
    void extract(kotekan::basebandDumpStatus&, std::mutex&);
    void fail(kotekan::basebandDumpStatus&, std::mutex&, const std::string&);

    int64_t samples_per_frame;
    int64_t output_spectra;
    int retained_frames;
    int elements;
    double spectrum_rate;
    double capture_timeout;
    int64_t max_capture_spectra;
    std::vector<int> selected_antennas;
    std::vector<Stream> streams;
    std::string endpoint;
    std::shared_ptr<kotekan::basebandReadoutManager> manager;
    std::mutex mutex;
    std::condition_variable data_ready;
    bool accepting = true;
    bool worker_busy = false;
    bool have_event = false;
    uint64_t last_event = 0;
};

#endif
