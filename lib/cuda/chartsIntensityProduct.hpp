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
// v2 (2026-11-02): history.upchan_gain describes the fine-channel gain G(u).
// v3 (2026-11-14, branch beamform-64-fp16-tc): fp16 fine voltages without clamp or int4
// re-quantization (history.fine_format), CHORD PFB window; no fine_quantization block.
// weights.device (fp16 + power-of-two scale), beamformer.{kernel, arithmetic} and
// history.upchan_gain.normalization were added to v3 on the same unreleased branch
// (2026-11-20); the reader requires them for v3.
inline constexpr const char* product_schema = "charts-intensity-product-v3";

struct ProductDescription {
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
    std::vector<float> upchan_gain;  // 32 values per fine index u
    bool upchan_gain_unity = true;
    std::string upchan_gain_source = "unity";
    std::string upchan_gain_reference;
    std::string upchan_gain_normalization = "none"; // declared by the configuration
    std::string upchan_gain_sha256;  // of the applied float32 [336*32] array
    nlohmann::json weights;
    std::string beamformer_kernel = "tensor_core"; // or "scalar_reference" (tests only)
    std::string intensity_buffer;
    std::string exposure_buffer;   // empty when packet loss is not enabled
    std::size_t exposure_bytes = 0;
    std::string quant_params_buffer; // empty when int8 parameters are not exported
    std::size_t quant_param_count = 0;
};

inline nlohmann::json product_descriptor(const ProductDescription& p) {
    using nlohmann::json;
    const std::size_t spectra = p.raw_n_time / upchannelization_factor;
    const std::size_t k = p.integration_spectra;
    json d = {
        {"schema", product_schema},
        {"generator", "cudaChartsBeamformerCommand"},
        {"kotekan_git_commit", get_git_commit_hash()},
        {"shard", p.shard_id},
        {"input_mode", "fine"},
        {"T_raw", p.raw_n_time},
        {"U", upchannelization_factor},
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
           "sum over the window of |sum_e W[b,f,e] x[f,e]|^2 for every supplied spectrum, "
           "where W is the effective weight s*W16 applied on the GPU (weights.device: fp16 "
           "with a power-of-two scale per beam and channel) and x is the fp16 fine voltage "
           "after the gain G(u) (history.upchan_gain); no "
           "averaging, exposure normalization or rejection; units of (weight * fp16 fine "
           "voltage)^2, the fine voltage being the PFB output (1/32 gain included, times "
           "G(u)) of the raw input codes; arithmetic in beamformer.arithmetic"},
          {"component_convention", "x = low nibble + i*high nibble"}}},
        {"frequency",
         {{"nu0_hz", p.freq_zero_hz},
          {"dnu_hz", p.freq_step_hz},
          {"rule", "nu0 + (336*shard + f + (u - 15.5)/32) * dnu, index f*32 + u"},
          {"first_hz", p.frequencies_hz.front()},
          {"last_hz", p.frequencies_hz.back()},
          {"rf_mapping_confirmed", false}}},
        {"timing",
         {{"raw_tick_seconds", {10, 3000000}},
          {"frame_start", "FPGA_SEQ_NUM in the forwarded raw chartsMetadata = raw tick of "
                          "sample 0; time0_fpga is provenance, not calibrated UTC"},
          {"window_rule",
           "window w sums fine spectra [w*K, min((w+1)*K, M)); spectrum m uses raw ticks "
           "[seq + 32m - 96, seq + 32m + 31]"}}},
        {"history",
         json{{"pfb", "4-tap sinc-Hanning, L=128, CHORD upchan.jl window "
                              "s' = (2s - 127)/256, coefficients include 1/32 and no other "
                              "normalization, forward FFT, centered half-bin order, fp32 "
                              "arithmetic"},
                       {"fine_format", "fp16"},
                       {"fine_conversion", "fp32 -> fp16 round to nearest even after G(u); no "
                                           "clamp, no int4 re-quantization"},
                       {"upchan_gain",
                        {{"values", p.upchan_gain},
                         {"unity", p.upchan_gain_unity},
                         {"source", p.upchan_gain_source},
                         {"reference", p.upchan_gain_reference},
                         {"normalization", p.upchan_gain_normalization},
                         {"normalization_note", "declared by the configuration "
                                                "(upchan_gain_normalization) and recorded "
                                                "verbatim; the code applies values as given"},
                         {"applied", "float32 [336*32], G[f*32 + u] = values[u] for every local "
                                     "coarse channel f"},
                         {"sha256", p.upchan_gain_sha256},
                         {"placement", "multiplies the fine voltages in fp32 after the FFT and "
                                       "before the fp16 conversion (CHORD upchan.jl step 7)"},
                         {"purpose", "flux correction of the firmware PFB scalloping; values "
                                     "applied verbatim from the configuration"},
                         {"power", "intensity scales as G(u)^2 (linear path); divide by "
                                   "G(u)^2 to refer power to the ungained upchannelizer"}}},
                       {"incomplete_spectra_after_start_or_reset", {0, 1, 2}},
                       {"reset", "a sequence jump on this shard zeroes sample and validity "
                                 "history"}}},
        {"association",
         {{"key", {"shard", "FPGA_SEQ_NUM", "T_raw"}},
          {"rule", "intensity, exposure and int8 parameter records of one frame carry the "
                   "same raw metadata; pair by key, never by ring slot or file index"}}},
        {"weights", p.weights},
        {"beamformer",
         {{"kernel", p.beamformer_kernel},
          {"arithmetic",
           p.beamformer_kernel == "tensor_core"
               ? "fp16 tensor-core MMA (WMMA m16n16k16): exact fp16 x fp16 products, fp32 "
                 "accumulation whose order and rounding are hardware-defined, so results are "
                 "not bit-reproducible across GPU models; (s*Yr)^2 + (s*Yi)^2 and the window sum "
                 "in fp32, spectra in time order"
               : "scalar fp32 test reference: fp16 inputs converted exactly to fp32, products "
                 "and sums in element order with fp32 FMA; (s*Yr)^2 + (s*Yi)^2 and the window "
                 "sum in fp32, spectra in time order"}}},
        {"exposure", nullptr},
        {"int8", nullptr},
    };
    if (!p.exposure_buffer.empty()) {
        d["exposure"] = {{"buffer", p.exposure_buffer},
                         {"dtype", "uint8"},
                         {"shape", {p.n_time, local_frequency_channels}},
                         {"axes", {"integration_window", "coarse_channel"}},
                         {"meaning", "valid supplied spectra per window; annotates, never "
                                     "rescales, sum-power"},
                         {"fine_bins_per_count", upchannelization_factor}};
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
