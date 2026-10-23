#ifndef CHARTS_MVDR_BEAMS_HPP
#define CHARTS_MVDR_BEAMS_HPP

#include "Config.hpp"
#include "Stage.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"
#include "chartsMvdr.hpp"

#include <cstdint>
#include <memory>
#include <string>

/**
 * @class chartsMVDRBeams
 * @brief DFT and MVDR beams from cudaCorrelatorAstron correlation frames (CPU).
 *
 * For every input frame t the stage estimates a covariance F from the neighbouring frames
 * k with cov_guard < |k - t| <= cov_half_window (blind local off-pulse window: the frame
 * and its neighbours never enter their own F), computes per channel and beam the DFT
 * weights d / N and the MVDR weights (F + delta I)^-1 d / (d^H (F + delta I)^-1 d), and
 * outputs the beam power w^H R(t) w. Weights are recomputed at the first frame with at
 * least cov_min_frames frames in its window and then every update_every frames.
 * Optional complex antenna gains (charts-utils .npy, v_obs = g v) are removed as
 * G^-1 R G^-H; elements with NaN or zero gain are dropped from that channel.
 *
 * The stage holds input frames t - H ... t + H (H = cov_half_window), so its latency is
 * H frames and in_buf needs at least 2 H + 2 frames. Output frame t is produced when frame
 * t + H has arrived; with flush_after_frames > 0 the remaining frames are produced (with
 * truncated windows) once that many input frames have been received, otherwise the last
 * H frames of a finite stream are not produced.
 *
 * Reference implementation: charts-mvdr, charts_mvdr/stage_reference.py.
 *
 * @par Buffers
 * @buffer in_buf      Correlation frames, int32 (re, im), (freq, num_blocks, block_size,
 *                     block_size, 2) as produced by cudaCorrelatorAstron (+ chartsAccumulate).
 * @buffer out_buf     Beam power, float32 [2][num_local_freq][num_beams] (DFT, MVDR), power
 *                     per sample; MVDR NaN before the first update. One frame per input frame.
 * @buffer weights_buf Optional. One frame per weight update: int64 input frame index,
 *                     int64 frames in the window, complex64 [num_local_freq][num_beams]
 *                     [num_elements] MVDR weights, float32 [num_local_freq][num_beams] WNG.
 *
 * @conf num_elements        Int.    Correlator elements.
 * @conf num_local_freq      Int.    Channels per frame.
 * @conf block_size          Int.    Correlator block size (default 2).
 * @conf samples_per_frame   Double. Samples summed per input frame (default
 *                                   samples_per_data_set).
 * @conf freq_start_mhz      Double. Frequency of channel 0 of the frame [MHz].
 * @conf delta_freq_mhz      Double. Channel spacing [MHz].
 * @conf positions_x_m       Array of double. Element x positions, correlator element order.
 * @conf positions_y_m       Array of double. Element y positions.
 * @conf beams               Array of [l, m]. Look directions (direction cosines).
 * @conf gains_file          String. Optional complex gains .npy (n_freq_file, num_elements).
 * @conf gains_first_channel Int.    Row of the gains file for channel 0 (default 0).
 * @conf cov_half_window     Int.    H (default 40: on the simulated windows the null depth
 *                                   spread and the burst recovery were better than with 10).
 * @conf cov_guard           Int.    g (default 1).
 * @conf cov_min_frames      Int.    Minimum frames in the window for an update (default H - g).
 * @conf update_every        Int.    Update period in frames (default 1).
 * @conf loading             Double. Diagonal loading in units of loading_ref (default 10).
 * @conf loading_ref         String. noise (median eigenvalue of F), trace or absolute.
 * @conf wng_warn_db         Double. Warn when WNG = N ||w||^2 exceeds this (default 3 dB).
 * @conf wng_log_period      Int.    Repeat an unchanged WNG warning at most every this many
 *                                   updates (default 100).
 * @conf flush_after_frames  Int.    Produce all remaining frames after this many inputs
 *                                   (default 0: never).
 */
class chartsMVDRBeams : public kotekan::Stage {
public:
    chartsMVDRBeams(kotekan::Config& config, const std::string& unique_name,
                    kotekan::bufferContainer& buffer_container);
    ~chartsMVDRBeams();
    void main_thread() override;

private:
    /// Produce output frame t using input frames up to last_index. False if stopping.
    bool process(int64_t t, int64_t last_index);

    Buffer* in_buf;
    Buffer* out_buf;
    Buffer* weights_buf;

    std::unique_ptr<charts_mvdr::MvdrEngine> engine;
    int half_window;
    int guard;
    int min_frames;
    int update_every;
    double wng_warn_db;
    int64_t flush_after_frames;
    int wng_log_period;
    long last_wng_count = -1;
    int updates_since_wng_log = 0;
    int out_frame_id = 0;
    int weights_frame_id = 0;
};

#endif
