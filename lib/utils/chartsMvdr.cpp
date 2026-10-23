#include "chartsMvdr.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace charts_mvdr {

namespace {

constexpr double speed_of_light_m_per_s = 299792458.0;
constexpr double two_pi = 6.283185307179586476925286766559;

/// Eigenvalues of a real symmetric tridiagonal matrix (implicit QL, Numerical Recipes tqli
/// without eigenvectors). d: diagonal, e[i]: element (i, i+1), e[n-1] unused.
void tridiagonal_ql(std::vector<double>& d, std::vector<double>& e) {
    const int n = static_cast<int>(d.size());
    if (n == 0)
        return;
    e[n - 1] = 0.0;
    const double eps = std::numeric_limits<double>::epsilon();
    for (int l = 0; l < n; ++l) {
        int iter = 0;
        int m;
        do {
            for (m = l; m < n - 1; ++m) {
                const double dd = std::fabs(d[m]) + std::fabs(d[m + 1]);
                if (std::fabs(e[m]) <= eps * dd)
                    break;
            }
            if (m != l) {
                if (iter++ == 200)
                    throw std::runtime_error("charts_mvdr: tridiagonal QL did not converge");
                double g = (d[l + 1] - d[l]) / (2.0 * e[l]);
                double r = std::hypot(g, 1.0);
                g = d[m] - d[l] + e[l] / (g + std::copysign(r, g));
                double s = 1.0, c = 1.0, p = 0.0;
                int i;
                for (i = m - 1; i >= l; --i) {
                    const double f = s * e[i];
                    const double b = c * e[i];
                    r = std::hypot(f, g);
                    e[i + 1] = r;
                    if (r == 0.0) {
                        d[i + 1] -= p;
                        e[m] = 0.0;
                        break;
                    }
                    s = f / r;
                    c = g / r;
                    g = d[i + 1] - p;
                    r = (d[i] - g) * s + 2.0 * c * b;
                    p = s * r;
                    d[i + 1] = g + p;
                    g = c * r - b;
                }
                if (r == 0.0 && i >= l)
                    continue;
                d[l] -= p;
                e[l] = g;
                e[m] = 0.0;
            }
        } while (m != l);
    }
}

/// Unpack one channel of the Astron layout into a full Hermitian matrix R = conj(S) * scale.
template<typename T>
void unpack_astron(const T* base, int N, int bs, double scale, std::vector<cplx>& full) {
    const int n_rec = N / bs;
    full.assign(static_cast<std::size_t>(N) * N, cplx(0.0, 0.0));
    std::vector<char> stored(static_cast<std::size_t>(N) * N, 0);
    for (int ry = 0; ry < n_rec; ++ry)
        for (int rx = 0; rx <= ry; ++rx) {
            const std::size_t blk = static_cast<std::size_t>(ry) * (ry + 1) / 2 + rx;
            for (int py = 0; py < bs; ++py)
                for (int px = 0; px < bs; ++px) {
                    const std::size_t idx = ((blk * bs + py) * bs + px) * 2;
                    const int ey = ry * bs + py, ex = rx * bs + px;
                    full[ey * N + ex] = cplx(static_cast<double>(base[idx]),
                                             -static_cast<double>(base[idx + 1]))
                                        * scale;
                    stored[ey * N + ex] = 1;
                }
        }
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            if (!stored[i * N + j])
                full[i * N + j] = std::conj(full[j * N + i]);
}

double median_sorted(const std::vector<double>& v) {
    const std::size_t n = v.size();
    if (n == 0)
        return 0.0;
    return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

} // namespace

LoadingRef parse_loading_ref(const std::string& name) {
    if (name == "noise")
        return LoadingRef::noise;
    if (name == "trace")
        return LoadingRef::trace;
    if (name == "absolute")
        return LoadingRef::absolute;
    throw std::invalid_argument("loading_ref must be one of noise, trace, absolute; got " + name);
}

std::vector<cplx> load_gains_npy(const std::string& path, int first_channel, int num_freq,
                                 int num_elements) {
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("charts_mvdr: cannot open gains file " + path);
    char magic[6];
    in.read(magic, 6);
    if (!in || std::memcmp(magic, "\x93NUMPY", 6) != 0)
        throw std::runtime_error("charts_mvdr: not a .npy file: " + path);
    unsigned char ver[2];
    in.read(reinterpret_cast<char*>(ver), 2);
    uint32_t hlen = 0;
    if (ver[0] == 1) {
        unsigned char b[2];
        in.read(reinterpret_cast<char*>(b), 2);
        hlen = b[0] | (b[1] << 8);
    } else {
        unsigned char b[4];
        in.read(reinterpret_cast<char*>(b), 4);
        hlen = b[0] | (b[1] << 8) | (b[2] << 16) | (static_cast<uint32_t>(b[3]) << 24);
    }
    std::string header(hlen, '\0');
    in.read(&header[0], hlen);
    if (!in)
        throw std::runtime_error("charts_mvdr: truncated .npy header: " + path);
    auto value_of = [&](const std::string& key) {
        const auto k = header.find("'" + key + "'");
        if (k == std::string::npos)
            throw std::runtime_error("charts_mvdr: .npy header without " + key + ": " + path);
        return header.substr(header.find(':', k) + 1);
    };
    const std::string descr = value_of("descr");
    int bytes;
    if (descr.find("'<c8'") == 0 || descr.find(" '<c8'") == 0)
        bytes = 8;
    else if (descr.find("'<c16'") == 0 || descr.find(" '<c16'") == 0)
        bytes = 16;
    else
        throw std::runtime_error("charts_mvdr: gains must be little-endian complex64 or "
                                 "complex128: "
                                 + path);
    if (value_of("fortran_order").find("False") == std::string::npos)
        throw std::runtime_error("charts_mvdr: gains must be in C order: " + path);
    const std::string shape = value_of("shape");
    long rows = -1, cols = -1;
    int used = 0;
    if (std::sscanf(shape.c_str(), " (%ld, %ld%n", &rows, &cols, &used) != 2
        || shape.compare(used, 1, ")") != 0)
        throw std::runtime_error("charts_mvdr: gains must be a 2D array (n_freq, n_ant): " + path);
    if (cols != num_elements)
        throw std::runtime_error("charts_mvdr: gains file has a different number of elements: "
                                 + path);
    if (first_channel < 0 || first_channel + num_freq > rows)
        throw std::runtime_error("charts_mvdr: gains file does not cover the channels: " + path);
    std::vector<cplx> out(static_cast<std::size_t>(num_freq) * num_elements);
    in.seekg(static_cast<std::streamoff>(first_channel) * cols * bytes, std::ios::cur);
    for (auto& v : out) {
        if (bytes == 8) {
            float re_im[2];
            in.read(reinterpret_cast<char*>(re_im), 8);
            v = cplx(re_im[0], re_im[1]);
        } else {
            double re_im[2];
            in.read(reinterpret_cast<char*>(re_im), 16);
            v = cplx(re_im[0], re_im[1]);
        }
    }
    if (!in)
        throw std::runtime_error("charts_mvdr: truncated gains file: " + path);
    return out;
}

std::vector<int64_t> window_indices(int64_t t, int64_t last_index, int half_window, int guard) {
    if (half_window < 1 || guard < 0 || guard >= half_window)
        throw std::invalid_argument("need half_window >= 1 and 0 <= guard < half_window");
    std::vector<int64_t> k;
    const int64_t lo = std::max<int64_t>(0, t - half_window);
    const int64_t hi = std::min<int64_t>(last_index, t + half_window);
    for (int64_t i = lo; i <= hi; ++i)
        if (std::llabs(i - t) > guard)
            k.push_back(i);
    return k;
}

std::vector<double> hermitian_eigenvalues(std::vector<cplx>& A, int n) {
    // Householder reduction to Hermitian tridiagonal form; the eigenvalues equal those of the
    // real symmetric tridiagonal matrix with the same diagonal and |off-diagonal|.
    std::vector<cplx> v(n), p(n), w(n);
    for (int k = 0; k < n - 2; ++k) {
        const int m = n - k - 1; // length of the column below the diagonal
        double xnorm2 = 0.0;
        for (int j = 0; j < m; ++j)
            xnorm2 += std::norm(A[(k + 1 + j) * n + k]);
        const double xnorm = std::sqrt(xnorm2);
        if (xnorm == 0.0)
            continue;
        const cplx x0 = A[(k + 1) * n + k];
        const cplx phase = (std::abs(x0) > 0.0) ? x0 / std::abs(x0) : cplx(1.0, 0.0);
        const cplx alpha = -phase * xnorm;
        for (int j = 0; j < m; ++j)
            v[j] = A[(k + 1 + j) * n + k];
        v[0] -= alpha;
        double vnorm2 = 0.0;
        for (int j = 0; j < m; ++j)
            vnorm2 += std::norm(v[j]);
        if (vnorm2 == 0.0)
            continue;
        const double vnorm = std::sqrt(vnorm2);
        for (int j = 0; j < m; ++j)
            v[j] /= vnorm;
        // trailing block B = A[k+1:, k+1:]: B <- B - 2 (v w^H + w v^H), w = Bv - (v^H B v) v
        for (int i = 0; i < m; ++i) {
            cplx s = 0.0;
            const cplx* row = &A[(k + 1 + i) * n + (k + 1)];
            for (int j = 0; j < m; ++j)
                s += row[j] * v[j];
            p[i] = s;
        }
        cplx K = 0.0;
        for (int i = 0; i < m; ++i)
            K += std::conj(v[i]) * p[i];
        const double Kr = K.real();
        for (int i = 0; i < m; ++i)
            w[i] = p[i] - Kr * v[i];
        for (int i = 0; i < m; ++i) {
            cplx* row = &A[(k + 1 + i) * n + (k + 1)];
            for (int j = 0; j < m; ++j)
                row[j] -= 2.0 * (v[i] * std::conj(w[j]) + w[i] * std::conj(v[j]));
        }
        A[(k + 1) * n + k] = alpha;
        A[k * n + (k + 1)] = std::conj(alpha);
        for (int j = 1; j < m; ++j) {
            A[(k + 1 + j) * n + k] = 0.0;
            A[k * n + (k + 1 + j)] = 0.0;
        }
    }
    std::vector<double> d(n), e(n, 0.0);
    for (int i = 0; i < n; ++i)
        d[i] = A[i * n + i].real();
    for (int i = 0; i < n - 1; ++i)
        e[i] = std::abs(A[(i + 1) * n + i]);
    tridiagonal_ql(d, e);
    std::sort(d.begin(), d.end());
    return d;
}

bool cholesky_solve(std::vector<cplx>& A, std::vector<cplx>& B, int n, int nrhs) {
    // A = L L^H, L stored in the lower triangle of A.
    for (int j = 0; j < n; ++j) {
        double djj = A[j * n + j].real();
        for (int k = 0; k < j; ++k)
            djj -= std::norm(A[j * n + k]);
        if (!(djj > 0.0))
            return false;
        const double ljj = std::sqrt(djj);
        A[j * n + j] = ljj;
        for (int i = j + 1; i < n; ++i) {
            cplx s = A[i * n + j];
            for (int k = 0; k < j; ++k)
                s -= A[i * n + k] * std::conj(A[j * n + k]);
            A[i * n + j] = s / ljj;
        }
    }
    for (int r = 0; r < nrhs; ++r) {
        for (int i = 0; i < n; ++i) { // L y = b
            cplx s = B[i * nrhs + r];
            for (int k = 0; k < i; ++k)
                s -= A[i * n + k] * B[k * nrhs + r];
            B[i * nrhs + r] = s / A[i * n + i].real();
        }
        for (int i = n - 1; i >= 0; --i) { // L^H x = y
            cplx s = B[i * nrhs + r];
            for (int k = i + 1; k < n; ++k)
                s -= std::conj(A[k * n + i]) * B[k * nrhs + r];
            B[i * nrhs + r] = s / A[i * n + i].real();
        }
    }
    return true;
}

MvdrEngine::MvdrEngine(const MvdrParams& params) : p_(params) {
    const int N = p_.num_elements, F = p_.num_freq, bs = p_.block_size;
    if (N <= 0 || F <= 0 || bs <= 0 || N % bs != 0)
        throw std::invalid_argument("charts_mvdr: invalid num_elements/num_freq/block_size");
    if (p_.samples_per_frame <= 0.0)
        throw std::invalid_argument("charts_mvdr: samples_per_frame must be positive");
    if ((int)p_.freqs_hz.size() != F)
        throw std::invalid_argument("charts_mvdr: freqs_hz must have num_freq values");
    if ((int)p_.pos_x_m.size() != N || (int)p_.pos_y_m.size() != N)
        throw std::invalid_argument("charts_mvdr: positions must have num_elements values");
    if (p_.beam_l.empty() || p_.beam_l.size() != p_.beam_m.size())
        throw std::invalid_argument("charts_mvdr: need at least one beam (l, m)");
    if (!p_.gains.empty() && (int)p_.gains.size() != F * N)
        throw std::invalid_argument("charts_mvdr: gains must have num_freq * num_elements values");
    n_beam_ = static_cast<int>(p_.beam_l.size());
    const int n_rec = N / bs;
    floats_per_freq_ = static_cast<std::size_t>(n_rec) * (n_rec + 1) / 2 * bs * bs * 2;

    good_.resize(F);
    inv_gain_.resize(F);
    steer_.resize(F);
    w_mvdr_.resize(F);
    for (int f = 0; f < F; ++f) {
        for (int a = 0; a < N; ++a) {
            const cplx g = p_.gains.empty() ? cplx(1.0, 0.0) : p_.gains[f * N + a];
            if (std::isfinite(g.real()) && std::isfinite(g.imag()) && std::abs(g) > 0.0) {
                good_[f].push_back(a);
                inv_gain_[f].push_back(1.0 / g);
            }
        }
        const int n = static_cast<int>(good_[f].size());
        steer_[f].resize(static_cast<std::size_t>(n_beam_) * n);
        for (int b = 0; b < n_beam_; ++b)
            for (int i = 0; i < n; ++i) {
                const int a = good_[f][i];
                const double delay =
                    (p_.beam_l[b] * p_.pos_x_m[a] + p_.beam_m[b] * p_.pos_y_m[a])
                    / speed_of_light_m_per_s;
                steer_[f][b * n + i] = std::polar(1.0, -two_pi * p_.freqs_hz[f] * delay);
            }
    }
}

void MvdrEngine::unpack_channel(const int32_t* frame, int freq, std::vector<cplx>& full) const {
    unpack_astron(frame + static_cast<std::size_t>(freq) * floats_per_freq_, p_.num_elements,
                  p_.block_size, 1.0, full);
}

void MvdrEngine::covariance(const int32_t* frame, int freq, std::vector<cplx>& R) const {
    std::vector<cplx> full;
    unpack_channel(frame, freq, full);
    const int N = p_.num_elements;
    const auto& idx = good_[freq];
    const auto& ig = inv_gain_[freq];
    const int n = static_cast<int>(idx.size());
    R.resize(static_cast<std::size_t>(n) * n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            R[i * n + j] = full[idx[i] * N + idx[j]] * ig[i] * std::conj(ig[j]);
}

bool MvdrEngine::update(const std::vector<const int32_t*>& frames) {
    if (frames.empty())
        return false;
    const int F = p_.num_freq;
    const std::size_t total = floats_per_freq_ * F;
    // Exact integer sum over the window, scaled to power per sample when unpacked.
    std::vector<int64_t> sum(total, 0);
    for (const int32_t* fr : frames)
        for (std::size_t i = 0; i < total; ++i)
            sum[i] += fr[i];
    const double scale = 1.0 / (static_cast<double>(frames.size()) * p_.samples_per_frame);

    std::vector<std::vector<cplx>> w_new(F);
    const int N = p_.num_elements;
    std::vector<cplx> full;
    for (int f = 0; f < F; ++f) {
        unpack_astron(sum.data() + static_cast<std::size_t>(f) * floats_per_freq_, N,
                      p_.block_size, scale, full);
        const auto& idx = good_[f];
        const auto& ig = inv_gain_[f];
        const int n = static_cast<int>(idx.size());
        if (n == 0)
            continue;
        std::vector<cplx> Fm(static_cast<std::size_t>(n) * n);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                Fm[i * n + j] = full[idx[i] * N + idx[j]] * ig[i] * std::conj(ig[j]);

        double delta = 0.0;
        if (p_.loading_ref == LoadingRef::absolute) {
            delta = p_.loading;
        } else if (p_.loading_ref == LoadingRef::trace) {
            double tr = 0.0;
            for (int i = 0; i < n; ++i)
                tr += Fm[i * n + i].real();
            delta = p_.loading * tr / n;
        } else if (p_.loading != 0.0) {
            std::vector<cplx> tmp(Fm);
            delta = p_.loading * median_sorted(hermitian_eigenvalues(tmp, n));
        }
        for (int i = 0; i < n; ++i)
            Fm[i * n + i] += delta;

        std::vector<cplx> X(static_cast<std::size_t>(n) * n_beam_);
        for (int i = 0; i < n; ++i)
            for (int b = 0; b < n_beam_; ++b)
                X[i * n_beam_ + b] = steer_[f][b * n + i];
        if (!cholesky_solve(Fm, X, n, n_beam_))
            return false;
        w_new[f].resize(static_cast<std::size_t>(n_beam_) * n);
        for (int b = 0; b < n_beam_; ++b) {
            cplx den = 0.0;
            for (int i = 0; i < n; ++i)
                den += std::conj(steer_[f][b * n + i]) * X[i * n_beam_ + b];
            for (int i = 0; i < n; ++i)
                w_new[f][b * n + i] = X[i * n_beam_ + b] / den;
        }
    }
    w_mvdr_.swap(w_new);
    has_weights_ = true;
    return true;
}

void MvdrEngine::beam_power(const int32_t* frame, float* out) const {
    const int F = p_.num_freq;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    std::vector<cplx> R;
    for (int f = 0; f < F; ++f) {
        const int n = static_cast<int>(good_[f].size());
        float* dft = out + static_cast<std::size_t>(f) * n_beam_;
        float* mv = out + (static_cast<std::size_t>(F) + f) * n_beam_;
        if (n == 0) {
            for (int b = 0; b < n_beam_; ++b)
                dft[b] = mv[b] = nan;
            continue;
        }
        covariance(frame, f, R);
        const double inv_ks = 1.0 / p_.samples_per_frame;
        auto quad = [&](const cplx* w, double wscale) {
            cplx acc = 0.0;
            for (int i = 0; i < n; ++i) {
                cplx s = 0.0;
                for (int j = 0; j < n; ++j)
                    s += R[i * n + j] * w[j];
                acc += std::conj(w[i]) * s;
            }
            return acc.real() * wscale * wscale * inv_ks;
        };
        for (int b = 0; b < n_beam_; ++b) {
            dft[b] = static_cast<float>(quad(&steer_[f][b * n], 1.0 / n));
            mv[b] = has_weights_ ? static_cast<float>(quad(&w_mvdr_[f][b * n], 1.0)) : nan;
        }
    }
}

std::vector<float> MvdrEngine::wng() const {
    std::vector<float> out(static_cast<std::size_t>(p_.num_freq) * n_beam_, 0.0f);
    if (!has_weights_)
        return out;
    for (int f = 0; f < p_.num_freq; ++f) {
        const int n = static_cast<int>(good_[f].size());
        if (w_mvdr_[f].empty())
            continue;
        for (int b = 0; b < n_beam_; ++b) {
            double norm2 = 0.0;
            for (int i = 0; i < n; ++i)
                norm2 += std::norm(w_mvdr_[f][b * n + i]);
            out[f * n_beam_ + b] = static_cast<float>(n * norm2);
        }
    }
    return out;
}

std::size_t MvdrEngine::weights_frame_size() const {
    const std::size_t nfb = static_cast<std::size_t>(p_.num_freq) * n_beam_;
    return 2 * sizeof(int64_t) + nfb * p_.num_elements * 2 * sizeof(float)
           + nfb * sizeof(float);
}

void MvdrEngine::write_weights(uint8_t* out, int64_t frame_index, int64_t n_window) const {
    std::memset(out, 0, weights_frame_size());
    std::memcpy(out, &frame_index, sizeof(int64_t));
    std::memcpy(out + sizeof(int64_t), &n_window, sizeof(int64_t));
    const int F = p_.num_freq, N = p_.num_elements;
    float* w = reinterpret_cast<float*>(out + 2 * sizeof(int64_t));
    float* wng_out = w + static_cast<std::size_t>(F) * n_beam_ * N * 2;
    if (has_weights_) {
        for (int f = 0; f < F; ++f) {
            const int n = static_cast<int>(good_[f].size());
            if (w_mvdr_[f].empty())
                continue;
            for (int b = 0; b < n_beam_; ++b)
                for (int i = 0; i < n; ++i) {
                    const cplx v = w_mvdr_[f][b * n + i];
                    const std::size_t o =
                        ((static_cast<std::size_t>(f) * n_beam_ + b) * N + good_[f][i]) * 2;
                    w[o] = static_cast<float>(v.real());
                    w[o + 1] = static_cast<float>(v.imag());
                }
        }
    }
    const auto g = wng();
    std::memcpy(wng_out, g.data(), g.size() * sizeof(float));
}

} // namespace charts_mvdr
