#ifndef CHARTS_INTENSITY_PRODUCT_HPP
#define CHARTS_INTENSITY_PRODUCT_HPP

#include "cudaChartsBeamformer.hpp"
#include "json.hpp"
#include "version.h"

#include <cstddef>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace beamformer {

// Versioned description of the integrated intensity product, written once at startup next
// to the frame files. The raw chartsMetadata forwarded with each frame stays immutable input
// provenance; this sidecar describes what the payload means. Per-frame data (exposure counts,
// int8 offset/scale) are separate products carrying the same raw metadata.
inline constexpr const char* product_schema = "charts-intensity-product-v1";

struct ProductDescription {
    bool fine = true;
    bool int8 = true;
    std::size_t shard_id = 0;
    std::size_t raw_n_time = 0;
    std::size_t integration_spectra = 0;
    std::size_t n_time = 0;
    std::size_t n_freq = 0;
    std::size_t n_beams = 0;
    std::size_t n_elements = 0;
    double freq_zero_hz = 0;
    double freq_step_hz = 0;
    std::vector<double> frequencies_hz;
    float quantize_clamp = 7.0F;
    nlohmann::json weights;
    std::string intensity_buffer;
    std::string exposure_buffer;   // empty when packet loss is not enabled
    std::size_t exposure_bytes = 0;
    std::string quant_params_buffer; // empty when int8 parameters are not exported
    std::size_t quant_param_count = 0;
};

inline nlohmann::json product_descriptor(const ProductDescription& p) {
    using nlohmann::json;
    const std::size_t spectra = p.fine ? p.raw_n_time / upchannelization_factor : p.raw_n_time;
    const std::size_t k = p.integration_spectra;
    json d = {
        {"schema", product_schema},
        {"generator", "cudaChartsBeamformerCommand"},
        {"kotekan_git_commit", get_git_commit_hash()},
        {"shard", p.shard_id},
        {"input_mode", p.fine ? "fine" : "coarse"},
        {"T_raw", p.raw_n_time},
        {"U", p.fine ? upchannelization_factor : 1},
        {"K", k},
        {"spectra_per_frame", spectra},
        {"tail_supplied_spectra", spectra - (p.n_time - 1) * k},
        {"num_elements", p.n_elements},
        {"intensity",
         {{"buffer", p.intensity_buffer},
          {"dtype", p.int8 ? "int8" : "float32"},
          {"byte_order", "little"},
          {"shape", {p.n_time, p.n_freq, p.n_beams}},
          {"axes", {"integration_window", "frequency", "beam"}},
          {"layout", "C-order, beam fastest"},
          {"power_convention",
           "sum over the window of |sum_e W[b,f,e] x[f,e]|^2 for every supplied spectrum; "
           "no averaging, exposure normalization or rejection; arbitrary units of "
           "(weight * int4 code)^2"},
          {"component_convention", "x = low nibble + i*high nibble"}}},
        {"frequency",
         {{"nu0_hz", p.freq_zero_hz},
          {"dnu_hz", p.freq_step_hz},
          {"rule", p.fine ? "nu0 + (336*shard + f + (u - 15.5)/32) * dnu, index f*32 + u"
                          : "nu0 + (336*shard + f) * dnu"},
          {"first_hz", p.frequencies_hz.front()},
          {"last_hz", p.frequencies_hz.back()},
          {"rf_mapping_confirmed", false}}},
        {"timing",
         {{"raw_tick_seconds", {10, 3000000}},
          {"frame_start", "FPGA_SEQ_NUM in the forwarded raw chartsMetadata = raw tick of "
                          "sample 0; time0_fpga is provenance, not calibrated UTC"},
          {"window_rule",
           p.fine ? "window w sums fine spectra [w*K, min((w+1)*K, M)); spectrum m uses raw "
                    "ticks [seq + 32m - 96, seq + 32m + 31]"
                  : "window w sums raw spectra [seq + w*K, seq + min((w+1)*K, T_raw))"}}},
        {"history",
         p.fine ? json{{"pfb", "4-tap sinc-Hanning, L=128, coefficients include 1/32, forward "
                               "FFT, centered half-bin order"},
                       {"fine_quantization",
                        {{"clamp", p.quantize_clamp}, {"rounding", "half to even"}}},
                       {"incomplete_spectra_after_start_or_reset", {0, 1, 2}},
                       {"reset", "a sequence jump on this shard zeroes sample and validity "
                                 "history"}}
                : json{{"pfb", nullptr}}},
        {"association",
         {{"key", {"shard", "FPGA_SEQ_NUM", "T_raw"}},
          {"rule", "intensity, exposure and int8 parameter records of one frame carry the "
                   "same raw metadata; pair by key, never by ring slot or file index"}}},
        {"weights", p.weights},
        {"exposure", nullptr},
        {"int8", nullptr},
    };
    if (!p.exposure_buffer.empty()) {
        d["exposure"] = {{"buffer", p.exposure_buffer},
                         {"dtype", p.fine ? "uint8" : "uint16"},
                         {"shape", {p.n_time, local_frequency_channels}},
                         {"axes", {"integration_window", "coarse_channel"}},
                         {"meaning", "valid supplied spectra per window; annotates, never "
                                     "rescales, sum-power"},
                         {"fine_bins_per_count", p.fine ? upchannelization_factor : 1}};
    }
    if (p.int8) {
        d["int8"] = {
            {"chunk", {quantization_chunk_time, quantization_chunk_frequency,
                       quantization_chunk_beam}},
            {"chunk_axes", {"integration_window", "frequency", "beam"}},
            {"codes", {{"finite", {quantized_min_code, quantized_max_code}},
                       {"invalid", quantized_invalid_code}}},
            {"reconstruction", "I = offset + q * scale"},
            {"encoding", "scale = (max - min) / 255, offset = min + 127.5 * scale, "
                         "q = clamp(round_half_even((I - offset) / scale), -127, 127); "
                         "scale = 0 for a constant chunk (I = offset)"},
            {"error_bound", "|I_hat - I| <= scale/2 plus float32 rounding; the chunk minimum "
                            "and maximum map to -127.5/+127.5 and are clamped, so they "
                            "reconstruct exactly min + scale/2 and max - scale/2: exact zeros in "
                            "a chunk with nonzero range are not preserved (one-sided bias)"},
            {"decode_precision", "float64 recommended"},
            {"invalid_chunk_rule", "a chunk containing any -128 code is invalid as a whole: "
                                   "its offset and scale are 0 and its finite values are "
                                   "coded 0"},
            {"parameters",
             p.quant_params_buffer.empty()
                 ? json(nullptr)
                 : json{{"buffer", p.quant_params_buffer},
                        {"dtype", "float32 pairs (offset, scale)"},
                        {"byte_order", "little"},
                        {"count", p.quant_param_count},
                        {"index", "(window_tile * frequency_tiles + frequency_tile) * "
                                  "beam_tiles + beam_tile"},
                        {"frequency_tiles", (p.n_freq + quantization_chunk_frequency - 1)
                                                / quantization_chunk_frequency},
                        {"beam_tiles", (p.n_beams + quantization_chunk_beam - 1)
                                           / quantization_chunk_beam}}},
            {"reconstructable", !p.quant_params_buffer.empty()}};
    }
    return d;
}

// Write atomically (temporary file then rename) so a reader never sees a partial file.
inline void write_product_descriptor(const std::string& path, const ProductDescription& p) {
    const std::string tmp = path + ".tmp";
    {
        std::ofstream out(tmp);
        if (!out)
            throw std::invalid_argument("cannot write product_descriptor_file " + path);
        out << product_descriptor(p).dump(1) << "\n";
        if (!out)
            throw std::invalid_argument("cannot write product_descriptor_file " + path);
    }
    if (std::rename(tmp.c_str(), path.c_str()) != 0)
        throw std::invalid_argument("cannot rename product_descriptor_file " + path);
}

} // namespace beamformer

#endif
