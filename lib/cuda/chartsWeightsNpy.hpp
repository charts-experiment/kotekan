#ifndef CHARTS_WEIGHTS_NPY_HPP
#define CHARTS_WEIGHTS_NPY_HPP

#include "cudaChartsBeamformer.hpp"
#include "json.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace beamformer {

// Canonical weights are complex64 [beam][frequency][element] with elements in kotekan_raw
// order, written by charts/scripts/build_beamformer_weights.py. Nothing is inverted,
// conjugated, normalized or steered here: the file holds the final combined coefficients.
struct NpyWeightsExpectation {
    std::string mode;  // "coarse" or "fine"
    std::size_t shard = 0;
    std::size_t n_beams = 0;
    std::size_t n_freq = 0;
    std::size_t n_ant = 0;
};

struct NpyWeightsInfo {
    std::string calibration_note;
    bool calibrated = false;
    std::string sha256; // from the descriptor, checked against the .npy bytes
    std::size_t flagged_channel_antennas = 0;
};

namespace detail {
inline void npy_fail(const std::string& why) {
    throw std::invalid_argument("weights .npy: " + why);
}

inline bool is_finite_weight(const ComplexFloat& w) {
    return std::isfinite(w.real) && std::isfinite(w.imag);
}

// Extract the value text of `'key':` from a Python dict literal header.
inline std::string npy_header_value(const std::string& header, const std::string& key) {
    const std::string needle = "'" + key + "':";
    const auto pos = header.find(needle);
    if (pos == std::string::npos)
        npy_fail("header lacks '" + key + "'");
    auto begin = header.find_first_not_of(' ', pos + needle.size());
    if (begin == std::string::npos)
        npy_fail("malformed header");
    std::size_t end;
    if (header[begin] == '(') {
        end = header.find(')', begin);
        if (end == std::string::npos)
            npy_fail("malformed shape");
        ++end;
    } else if (header[begin] == '\'') {
        end = header.find('\'', begin + 1);
        if (end == std::string::npos)
            npy_fail("malformed string");
        ++end;
    } else {
        end = header.find_first_of(",}", begin);
        if (end == std::string::npos)
            npy_fail("malformed header");
    }
    return header.substr(begin, end - begin);
}

// FIPS 180-4 SHA-256, self-contained so the check does not depend on the optional
// OpenSSL support of kotekan_core.
inline std::string sha256_hex(const char* data, std::size_t size) {
    static const std::uint32_t k[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2};
    std::uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                          0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    auto rotr = [](std::uint32_t x, int n) { return (x >> n) | (x << (32 - n)); };
    auto block = [&](const unsigned char* p) {
        std::uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = (std::uint32_t(p[4 * i]) << 24) | (std::uint32_t(p[4 * i + 1]) << 16)
                   | (std::uint32_t(p[4 * i + 2]) << 8) | std::uint32_t(p[4 * i + 3]);
        for (int i = 16; i < 64; ++i) {
            const std::uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const std::uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        std::uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6],
                      hh = h[7];
        for (int i = 0; i < 64; ++i) {
            const std::uint32_t t1 =
                hh + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + k[i] + w[i];
            const std::uint32_t t2 =
                (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
            hh = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }
        h[0] += a;
        h[1] += b;
        h[2] += c;
        h[3] += d;
        h[4] += e;
        h[5] += f;
        h[6] += g;
        h[7] += hh;
    };
    const auto* bytes = reinterpret_cast<const unsigned char*>(data);
    const std::size_t full = size / 64 * 64;
    for (std::size_t i = 0; i < full; i += 64)
        block(bytes + i);
    // Final padding: 0x80, zeros, then the 64-bit big-endian bit length.
    unsigned char tail[128] = {};
    const std::size_t rest = size - full;
    std::memcpy(tail, bytes + full, rest);
    tail[rest] = 0x80;
    const std::size_t tail_len = rest < 56 ? 64 : 128;
    const std::uint64_t bits = static_cast<std::uint64_t>(size) * 8;
    for (int i = 0; i < 8; ++i)
        tail[tail_len - 1 - i] = static_cast<unsigned char>(bits >> (8 * i));
    for (std::size_t i = 0; i < tail_len; i += 64)
        block(tail + i);
    static const char hex[] = "0123456789abcdef";
    std::string out;
    for (std::uint32_t v : h)
        for (int i = 28; i >= 0; i -= 4)
            out += hex[(v >> i) & 0xf];
    return out;
}
} // namespace detail

// Parse an in-memory .npy image holding exactly the expected complex64 C-order array.
// Rejects unsupported versions, object/pickled arrays, byte order, Fortran order, wrong
// shape or payload length (no trailing bytes) and non-finite coefficients.
inline Weights parse_npy_complex64(const std::vector<char>& bytes,
                                   const NpyWeightsExpectation& expect) {
    // '<c8' payload is copied directly into ComplexFloat.
    static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__, "little-endian host required");
    static const char magic[] = "\x93NUMPY";
    if (bytes.size() < 10 || std::memcmp(bytes.data(), magic, 6) != 0)
        detail::npy_fail("missing \\x93NUMPY magic");
    const auto major = static_cast<unsigned char>(bytes[6]);
    std::size_t header_len, header_start;
    if (major == 1) {
        header_len = static_cast<unsigned char>(bytes[8])
                     | (static_cast<std::size_t>(static_cast<unsigned char>(bytes[9])) << 8);
        header_start = 10;
    } else if (major == 2 || major == 3) {
        if (bytes.size() < 12)
            detail::npy_fail("truncated header");
        header_len = 0;
        for (int i = 0; i < 4; ++i)
            header_len |= static_cast<std::size_t>(static_cast<unsigned char>(bytes[8 + i]))
                          << (8 * i);
        header_start = 12;
    } else {
        detail::npy_fail("unsupported format version " + std::to_string(major));
    }
    if (header_len > bytes.size() || header_start + header_len > bytes.size())
        detail::npy_fail("header exceeds file");
    const std::string header(bytes.data() + header_start, header_len);

    const std::string descr = detail::npy_header_value(header, "descr");
    if (descr != "'<c8'")
        detail::npy_fail("dtype must be little-endian complex64 ('<c8'), got " + descr
                         + " (object arrays and other dtypes are rejected)");
    if (detail::npy_header_value(header, "fortran_order") != "False")
        detail::npy_fail("Fortran order is not supported");

    const std::string shape = detail::npy_header_value(header, "shape");
    const std::string want = "(" + std::to_string(expect.n_beams) + ", "
                             + std::to_string(expect.n_freq) + ", "
                             + std::to_string(expect.n_ant) + ")";
    if (shape != want)
        detail::npy_fail("shape " + shape + " does not match expected " + want
                         + " [beam][frequency][element]");

    const std::size_t count = expect.n_beams * expect.n_freq * expect.n_ant;
    const std::size_t payload = bytes.size() - (header_start + header_len);
    if (payload != count * sizeof(ComplexFloat))
        detail::npy_fail("payload is " + std::to_string(payload) + " bytes, expected "
                         + std::to_string(count * sizeof(ComplexFloat)));

    Weights weights(count);
    std::memcpy(weights.data(), bytes.data() + header_start + header_len, payload);
    for (std::size_t i = 0; i < count; ++i)
        if (!detail::is_finite_weight(weights[i]))
            detail::npy_fail("non-finite coefficient at flat index " + std::to_string(i));
    return weights;
}

// Validate the sidecar produced by build_beamformer_weights.py against the configured
// stage. The .npy header cannot describe calibration semantics, so this is mandatory.
inline NpyWeightsInfo validate_weights_descriptor(const nlohmann::json& d,
                                                  const NpyWeightsExpectation& expect) {
    auto fail = [](const std::string& why) {
        throw std::invalid_argument("weights descriptor: " + why);
    };
    // Integer fields must be JSON integers: 0.9 or false would otherwise truncate to 0.
    auto integer = [&](const nlohmann::json& v, const std::string& what) {
        if (!v.is_number_integer() || v.get<long long>() < 0)
            fail("'" + what + "' must be a non-negative integer");
        return v.get<std::size_t>();
    };
    if (!d.is_object())
        fail("not a JSON object");
    if (d.value("format", "") != "charts_beamformer_weights")
        fail("format must be 'charts_beamformer_weights'");
    // Version 3 introduced the outer conjugate for the beamformer component convention;
    // earlier files are mis-steered.
    if (!d.contains("version") || integer(d["version"], "version") < 3)
        fail("version must be >= 3 (earlier builders omit the component conjugate)");
    if (d.value("mode", "") != expect.mode)
        fail("mode '" + d.value("mode", "") + "' != configured '" + expect.mode + "'");
    if (!d.contains("shard") || integer(d["shard"], "shard") != expect.shard)
        fail("shard does not match configured shard_id");
    if (d.value("dtype", "") != "complex64")
        fail("dtype must be complex64");
    if (d.value("element_order", "") != "kotekan_raw")
        fail("element_order must be kotekan_raw");
    if (d.value("normalization", "") != "none")
        fail("normalization must be 'none'");
    if (!d.contains("shape") || !d["shape"].is_array() || d["shape"].size() != 3
        || integer(d["shape"][0], "shape") != expect.n_beams
        || integer(d["shape"][1], "shape") != expect.n_freq
        || integer(d["shape"][2], "shape") != expect.n_ant)
        fail("shape does not match [beams][frequency][elements]");
    if (!d.contains("upchannelization_factor")
        || integer(d["upchannelization_factor"], "upchannelization_factor")
               != (expect.mode == "fine" ? 32U : 1U))
        fail("upchannelization_factor disagrees with mode");
    if (!d.contains("frequency_hz") || !d["frequency_hz"].contains("first")
        || !d["frequency_hz"].contains("last") || !d["frequency_hz"].contains("nu0_hz")
        || !d["frequency_hz"].contains("dnu_hz"))
        fail("missing frequency_hz first/last/nu0_hz/dnu_hz");

    NpyWeightsInfo info;
    info.sha256 = d.value("sha256", "");
    if (info.sha256.size() != 64
        || info.sha256.find_first_not_of("0123456789abcdef") != std::string::npos)
        fail("missing or malformed 'sha256' (64 lowercase hex digits of the .npy file)");
    if (d.contains("gain")) {
        info.calibrated = d["gain"].value("calibrated", false);
        info.flagged_channel_antennas =
            d["gain"].value("flagged_logical_channel_antenna_count", std::size_t{0});
    }
    if (d.contains("inputs") && d["inputs"].is_object())
        info.calibration_note = d["inputs"].value("calibration_note", "");
    return info;
}

// Check the descriptor's frequency map (nu0, dnu, first and last channel) against the
// stage's own map; matching only the first channel would accept a wrong step on shard 0.
inline void check_descriptor_frequency(const nlohmann::json& d,
                                       const std::vector<double>& stage_frequencies_hz,
                                       double freq_zero_hz, double freq_step_hz) {
    const auto& f = d["frequency_hz"];
    auto near = [](double a, double b) { return std::abs(a - b) <= 1e-3; };
    if (!near(f["nu0_hz"].get<double>(), freq_zero_hz)
        || !near(f["dnu_hz"].get<double>(), freq_step_hz))
        throw std::invalid_argument("weights descriptor: nu0_hz/dnu_hz disagree with the "
                                    "configured frequency_zero_hz/frequency_step_hz");
    if (!near(f["first"].get<double>(), stage_frequencies_hz.front())
        || !near(f["last"].get<double>(), stage_frequencies_hz.back()))
        throw std::invalid_argument("weights descriptor: first/last frequency disagrees with the "
                                    "configured frequency_zero_hz/step and shard");
}

// The descriptor digest covers the whole .npy file (header and payload), as written by
// build_beamformer_weights.py; a mismatch means the sidecar describes another array.
inline void check_npy_sha256(const std::vector<char>& bytes, const std::string& expected) {
    const std::string actual = detail::sha256_hex(bytes.data(), bytes.size());
    if (actual != expected)
        throw std::invalid_argument("weights .npy: sha256 mismatch, file " + actual
                                    + " != descriptor " + expected);
}

inline std::vector<char> read_whole_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in)
        throw std::invalid_argument("cannot open " + path);
    std::vector<char> bytes(static_cast<std::size_t>(in.tellg()));
    in.seekg(0);
    if (!in.read(bytes.data(), static_cast<std::streamsize>(bytes.size())))
        throw std::invalid_argument("cannot read " + path);
    return bytes;
}

} // namespace beamformer

#endif
