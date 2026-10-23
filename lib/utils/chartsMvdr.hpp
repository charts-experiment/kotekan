/**
 * @file
 * @brief Correlation-domain DFT and MVDR beams for CHARTS (no Kotekan dependencies).
 *
 * Shared by the chartsMVDRBeams stage. Reference implementation:
 * charts-mvdr, charts_mvdr/stage_reference.py (run_stage); the two are compared with a
 * floating-point tolerance, not an exact match.
 *
 * Conventions:
 *  - Input frames are cudaCorrelatorAstron output: int32 (re, im) pairs with shape
 *    (freq, num_blocks, block_size, block_size, 2), blocks over the lower block triangle in
 *    row-major order (block index ry * (ry + 1) / 2 + rx), entry [py][px] of block (ry, rx)
 *    for elements (ry * block_size + py, rx * block_size + px). The kernel stores
 *    S_ij = sum conj(v_i) v_j; this code uses R_ij = conj(S_ij) = sum v_i conj(v_j).
 *  - Steering vector d_n = exp(-j 2 pi f (l x_n + m y_n) / c).
 *  - Gains follow the charts-utils convention v_obs = g v: covariances are corrected as
 *    G^-1 R G^-H. Elements with a non-finite or zero gain in a channel are left out of that
 *    channel (zero weight).
 *  - MVDR: w = (F + delta I)^-1 d / (d^H (F + delta I)^-1 d), DFT: w = d / N.
 *  - WNG = N ||w||^2 (0 dB for DFT, higher is worse), N = usable elements in the channel.
 */
#ifndef CHARTS_MVDR_HPP
#define CHARTS_MVDR_HPP

#include <complex>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace charts_mvdr {

using cplx = std::complex<double>;

enum class LoadingRef { noise, trace, absolute };

LoadingRef parse_loading_ref(const std::string& name);

struct MvdrParams {
    int num_elements = 0;
    int num_freq = 0;
    int block_size = 2;
    double samples_per_frame = 1.0;
    std::vector<double> freqs_hz;         // num_freq
    std::vector<double> pos_x_m, pos_y_m; // num_elements
    std::vector<double> beam_l, beam_m;   // num_beams
    double loading = 10.0;
    LoadingRef loading_ref = LoadingRef::noise;
    std::vector<cplx> gains; // empty (unity) or num_freq * num_elements
};

/// Frame indices k with guard < |k - t| <= half_window inside [0, last_index].
std::vector<int64_t> window_indices(int64_t t, int64_t last_index, int half_window, int guard);

/// Read complex gains from a NumPy .npy file of shape (n_freq_file, num_elements), dtype
/// complex64 or complex128, little-endian, C order (charts-utils cal_pipeline_gpu
/// --save-avg-gains). Returns channels [first_channel, first_channel + num_freq) row-major.
std::vector<cplx> load_gains_npy(const std::string& path, int first_channel, int num_freq,
                                 int num_elements);

/// Logical antenna of each correlator element: e (ascending) or n - 1 - e (descending, the
/// order of CHARTS baseband frames and the cudaDirectBeamTracker default).
std::vector<int> element_to_antenna(int n, const std::string& antenna_order);

/// Gains [freq][antenna] (file order, logical antennas) reordered to [freq][element].
std::vector<cplx> gains_to_elements(const std::vector<cplx>& gains, int num_freq,
                                    const std::vector<int>& elem_to_ant);

/// Correctors c = 1 / g for a beamformer that multiplies its weights by c and forms
/// y = sum_e w_e v_e (charts-utils convention v_obs = g v), laid out [freq][element]:
/// element e takes the gain of antenna elem_to_ant[e]. Non-finite or zero gains give c = 0
/// (the element is dropped). gains is [freq][antenna] with num_ant antennas.
std::vector<std::complex<float>> calibration_correctors(const std::vector<cplx>& gains,
                                                       int num_freq, int num_ant,
                                                       const std::vector<int>& elem_to_ant,
                                                       int* num_dropped = nullptr);

/// Weight update rule: the first frame whose window holds at least min_frames frames, then
/// frames with t % update_every == 0 (and at least min_frames frames in the window).
inline bool should_update(int64_t t, bool has_weights, int update_every, std::size_t n_window,
                          int min_frames) {
    if (n_window < static_cast<std::size_t>(min_frames))
        return false;
    return !has_weights || (t % update_every == 0);
}

/// Eigenvalues (ascending) of a Hermitian matrix A (n x n, row-major); A is overwritten.
std::vector<double> hermitian_eigenvalues(std::vector<cplx>& A, int n);

/// Solve (A) X = B for Hermitian positive definite A (n x n) and B (n x nrhs), row-major.
/// Returns false if A is not positive definite. A and B are overwritten (B holds X).
bool cholesky_solve(std::vector<cplx>& A, std::vector<cplx>& B, int n, int nrhs);

class MvdrEngine {
public:
    explicit MvdrEngine(const MvdrParams& params);

    int num_beams() const {
        return n_beam_;
    }
    bool has_weights() const {
        return has_weights_;
    }

    /// Recompute the MVDR weights from the sum of the given frames (F = sum / (n K_s)).
    /// Returns false (weights unchanged) if a covariance is not positive definite.
    bool update(const std::vector<const int32_t*>& frames);

    /// Beam power of one frame: out[2][num_freq][num_beams], plane 0 DFT, plane 1 MVDR
    /// (NaN until the first update), power per sample.
    void beam_power(const int32_t* frame, float* out) const;

    /// WNG = N ||w||^2 of the current MVDR weights, [num_freq][num_beams] (0 without weights).
    std::vector<float> wng() const;

    /// Weights frame: int64 frame index, int64 n_window, complex64
    /// [num_freq][num_beams][num_elements], float32 WNG [num_freq][num_beams].
    std::size_t weights_frame_size() const;
    void write_weights(uint8_t* out, int64_t frame_index, int64_t n_window) const;

    /// Covariance of one channel from a frame, gain-corrected, restricted to usable elements.
    void covariance(const int32_t* frame, int freq, std::vector<cplx>& R) const;

private:
    void unpack_channel(const int32_t* frame, int freq, std::vector<cplx>& full) const;

    MvdrParams p_;
    int n_beam_;
    std::size_t floats_per_freq_;             // num_blocks * bs * bs * 2
    std::vector<std::vector<int>> good_;      // usable element indices per channel
    std::vector<std::vector<cplx>> inv_gain_; // per channel, per usable element
    std::vector<std::vector<cplx>> steer_;    // per channel [beam][good element]
    std::vector<std::vector<cplx>> w_mvdr_;   // per channel [beam][good element]
    bool has_weights_ = false;
};

} // namespace charts_mvdr

#endif // CHARTS_MVDR_HPP
