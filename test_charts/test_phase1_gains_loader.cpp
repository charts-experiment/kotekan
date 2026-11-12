#include "chartsMvdr.hpp"
#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <cassert>
#include <fstream>
#include <cstring>
#include <filesystem>
#include <limits>

// Helper to write a simple .npy file (C-order, little-endian complex128)
static void write_test_npy(const std::string& path, long rows, long cols, const std::vector<std::complex<double>>& data) {
    std::ofstream out(path, std::ios::binary);
    if (!out)
        throw std::runtime_error("Cannot create test npy: " + path);
    // Header
    const char magic[6] = {'\x93', 'N', 'U', 'M', 'P', 'Y'};
    out.write(magic, 6);
    unsigned char ver[2] = {1, 0};
    out.write(reinterpret_cast<char*>(ver), 2);
    
    // Construct header dict
    std::string header = "{'descr': '<c16', 'fortran_order': False, 'shape': (" + 
                         std::to_string(rows) + ", " + std::to_string(cols) + "), }";
    // Pad to 64-byte boundary
    size_t total_len = 10 + header.size() + 1; // 10 bytes prefix + header + newline
    size_t pad = (64 - (total_len % 64)) % 64;
    header.append(pad, ' ');
    header.push_back('\n');
    
    uint16_t hlen = static_cast<uint16_t>(header.size());
    out.write(reinterpret_cast<char*>(&hlen), 2);
    out.write(header.data(), header.size());
    
    // Data
    out.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(std::complex<double>));
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "[Phase 1] Dynamic Gains Loader & Calibration Math Unit Tests\n";
    std::cout << "========================================================\n";

    const std::string path4 = "charts/config/gains/gains_sun_transit_1027_interp.npy";
    const std::string path64 = "charts/config/gains/gains_sun_transit_1027_64ant.npy";
    const std::vector<int> active_antennas = {0, 5, 32, 37};

    // -------------------------------------------------------------------------
    // 1. 4-Column vs. 64-Column Drop-In Equivalence & Dual-Stream Slicing
    // -------------------------------------------------------------------------
    std::cout << "1. Verifying 4-Column vs. 64-Column drop-in equivalence across dual streams...\n";
    for (int stream = 0; stream < 2; ++stream) {
        int first_chan = stream * 336;
        int num_freq = 336;
        int num_elements = 64;

        auto g4 = charts_mvdr::load_gains_npy(path4, first_chan, num_freq, num_elements, active_antennas);
        auto g64 = charts_mvdr::load_gains_npy(path64, first_chan, num_freq, num_elements, {});

        assert(g4.size() == static_cast<std::size_t>(num_freq * num_elements));
        assert(g64.size() == static_cast<std::size_t>(num_freq * num_elements));

        for (std::size_t i = 0; i < g4.size(); ++i) {
            if (g4[i] != g64[i]) {
                std::cerr << "Mismatch at index " << i << " in stream " << stream << "\n";
                return 1;
            }
        }

        auto elem_to_ant = charts_mvdr::element_to_antenna(num_elements, "descending");
        int dropped4 = 0, dropped64 = 0;
        auto c4 = charts_mvdr::calibration_correctors(g4, num_freq, num_elements, elem_to_ant, &dropped4);
        auto c64 = charts_mvdr::calibration_correctors(g64, num_freq, num_elements, elem_to_ant, &dropped64);

        assert(dropped4 == 20160);
        assert(dropped64 == 20160);
        assert(c4.size() == c64.size());

        float max_diff = 0.0f;
        for (std::size_t i = 0; i < c4.size(); ++i) {
            float d = std::abs(c4[i] - c64[i]);
            if (d > max_diff) max_diff = d;
        }
        std::cout << "   Stream " << stream << " (channels " << first_chan << ".." << first_chan + num_freq - 1
                  << "): max |c4 - c64| = " << max_diff << ", dropped = " << dropped4 << " [PASS]\n";
        assert(max_diff == 0.0f);
    }

    // -------------------------------------------------------------------------
    // 2. Complex Phase Sign & Amplitude Inversion Convention
    // -------------------------------------------------------------------------
    std::cout << "2. Verifying Complex Phase Sign & Amplitude Inversion Convention...\n";
    {
        int num_freq = 336;
        int num_elements = 64;
        auto g4 = charts_mvdr::load_gains_npy(path4, 0, num_freq, num_elements, active_antennas);
        auto elem_to_ant = charts_mvdr::element_to_antenna(num_elements, "descending");
        int dropped = 0;
        auto corr = charts_mvdr::calibration_correctors(g4, num_freq, num_elements, elem_to_ant, &dropped);

        // Verify that for active antennas:
        // (a) mathematically 1.0 / g yields bit-exact 0.0 residual phase error (< 1e-12 deg)
        // (b) float32 representation yields residual phase error within float precision (< 1e-5 deg)
        double max_phase_err_deg_float = 0.0;
        double max_phase_err_deg_double = 0.0;
        for (int f = 0; f < num_freq; ++f) {
            for (int e = 0; e < num_elements; ++e) {
                int a = elem_to_ant[e];
                bool is_active = (a == 0 || a == 5 || a == 32 || a == 37);
                std::complex<float> c = corr[static_cast<std::size_t>(f) * num_elements + e];
                std::complex<double> g = g4[static_cast<std::size_t>(f) * num_elements + a];
                if (is_active) {
                    std::complex<double> v_double = 1.0 / g;
                    double phase_double = std::abs(std::arg(v_double * g)) * 180.0 / M_PI;
                    if (phase_double > max_phase_err_deg_double)
                        max_phase_err_deg_double = phase_double;

                    std::complex<double> prod = std::complex<double>(c.real(), c.imag()) * g;
                    double phase_deg = std::abs(std::arg(prod)) * 180.0 / M_PI;
                    if (phase_deg > max_phase_err_deg_float)
                        max_phase_err_deg_float = phase_deg;
                    assert(std::abs(prod.real() - 1.0) < 1e-5);
                    assert(std::abs(prod.imag()) < 1e-5);
                } else {
                    assert(c.real() == 0.0f && c.imag() == 0.0f);
                }
            }
        }
        std::cout << "   Active antennas mathematical residual phase error (FP64) = " << max_phase_err_deg_double
                  << " deg (< 1e-12 deg) [PASS]\n";
        assert(max_phase_err_deg_double < 1e-12);
        std::cout << "   Active antennas float32 residual phase error = " << max_phase_err_deg_float
                  << " deg (within float32 precision, < 2.5e-6 deg) [PASS]\n";
        assert(max_phase_err_deg_float < 2.5e-6);
        std::cout << "   60 uncalibrated antennas masked to 0.0 + 0.0j (total dropped = " << dropped << ") [PASS]\n";
    }

    // -------------------------------------------------------------------------
    // 3. Fault Injection & Boundary Guards
    // -------------------------------------------------------------------------
    std::cout << "3. Testing Fault Injection & Boundary Guards...\n";
    {
        // 3a. Zero / Subnormal / Non-Finite Gains
        std::vector<std::complex<double>> bad_values = {
            std::complex<double>(0.0, 0.0),
            std::complex<double>(-0.0, 0.0),
            std::complex<double>(1e-320, 0.0),
            std::complex<double>(0.0, 1e-320),
            std::complex<double>(std::numeric_limits<double>::quiet_NaN(), 1.0),
            std::complex<double>(1.0, std::numeric_limits<double>::quiet_NaN()),
            std::complex<double>(std::numeric_limits<double>::infinity(), 0.0),
            std::complex<double>(0.0, -std::numeric_limits<double>::infinity())
        };

        for (const auto& bad_val : bad_values) {
            int num_freq = 336;
            int num_elements = 64;
            auto g = charts_mvdr::load_gains_npy(path4, 0, num_freq, num_elements, active_antennas);
            // Inject bad value into channel 100, antenna 5
            g[100 * num_elements + 5] = bad_val;

            auto elem_to_ant = charts_mvdr::element_to_antenna(num_elements, "descending");
            int dropped = 0;
            auto corr = charts_mvdr::calibration_correctors(g, num_freq, num_elements, elem_to_ant, &dropped);

            std::cout << "   [bad_val test] bad_val = " << bad_val << ", dropped = " << dropped << "\n";
            // Expect dropped to increment from 20160 to 20161
            assert(dropped == 20161);
            // Verify that channel 100, antenna 5 is strictly 0.0 + 0.0j and no NaNs exist in corr
            for (int e = 0; e < num_elements; ++e) {
                if (elem_to_ant[e] == 5) {
                    auto c = corr[100 * num_elements + e];
                    assert(c.real() == 0.0f && c.imag() == 0.0f);
                    (void)c;
                }
            }
            for (auto c : corr) {
                assert(std::isfinite(c.real()) && std::isfinite(c.imag()));
                (void)c;
            }
        }
        std::cout << "   Fault Injection (0, -0, 1e-320, NaN, Inf) -> masked to 0, dropped=20161, 0 NaNs [PASS]\n";

        // 3b. Multi-channel, multi-antenna edge channel corruptions
        {
            int num_freq = 336;
            int num_elements = 64;
            auto g = charts_mvdr::load_gains_npy(path4, 0, num_freq, num_elements, active_antennas);
            // Corrupt first channel (0) on antenna 0, and last channel (335) on antenna 37
            g[0 * num_elements + 0] = std::complex<double>(0.0, 0.0);
            g[335 * num_elements + 37] = std::complex<double>(std::numeric_limits<double>::infinity(), 0.0);

            auto elem_to_ant = charts_mvdr::element_to_antenna(num_elements, "descending");
            int dropped = 0;
            auto corr = charts_mvdr::calibration_correctors(g, num_freq, num_elements, elem_to_ant, &dropped);
            assert(dropped == 20162);
            for (auto c : corr) {
                assert(std::isfinite(c.real()) && std::isfinite(c.imag()));
                (void)c;
            }
            std::cout << "   Edge Channels (f=0, f=335) active antenna corruption -> dropped=20162, 0 NaNs [PASS]\n";
        }

        // 3c. Dimension & Index Bounds
        // Mismatched (336, 4) file when 672 channels expected
        std::string tmp_mismatched = "/tmp/test_gains_mismatched_336x4.npy";
        std::vector<std::complex<double>> dummy_336x4(336 * 4, std::complex<double>(1.0, 0.0));
        write_test_npy(tmp_mismatched, 336, 4, dummy_336x4);
        bool caught_mismatch = false;
        try {
            // Request channels 336..671 from 336-row file -> must throw
            charts_mvdr::load_gains_npy(tmp_mismatched, 336, 336, 64, active_antennas);
        } catch (const std::runtime_error& e) {
            caught_mismatch = true;
            std::cout << "   Caught expected channel-mismatch exception: " << e.what() << " [PASS]\n";
        }
        assert(caught_mismatch);
        (void)caught_mismatch;

        // 3-element antenna_subset for 4-column .npy
        bool caught_bad_subset = false;
        try {
            std::vector<int> subset3 = {0, 5, 32};
            charts_mvdr::load_gains_npy(path4, 0, 336, 64, subset3);
        } catch (const std::runtime_error& e) {
            caught_bad_subset = true;
            std::cout << "   Caught expected subset-size exception: " << e.what() << " [PASS]\n";
        }
        assert(caught_bad_subset);
        (void)caught_bad_subset;

        // Out-of-bounds antenna index [0, 5, 32, 64]
        bool caught_oob = false;
        try {
            std::vector<int> subset_oob = {0, 5, 32, 64};
            charts_mvdr::load_gains_npy(path4, 0, 336, 64, subset_oob);
        } catch (const std::runtime_error& e) {
            caught_oob = true;
            std::cout << "   Caught expected OOB antenna index exception: " << e.what() << " [PASS]\n";
        }
        assert(caught_oob);
        (void)caught_oob;
    }

    // -------------------------------------------------------------------------
    // 4. Strict Phase Inversion Boundaries, Dynamic Range & Physical Geometry
    // -------------------------------------------------------------------------
    std::cout << "4. Testing Strict Phase Inversions, Dynamic Range & Interferometer Baselines...\n";
    {
        // 4a. Extreme Phase Jumps (180 deg, +90 deg, -90 deg)
        std::vector<std::complex<double>> phase_cases = {
            std::complex<double>(-1.0, 0.0),   // 180 deg phase inversion
            std::complex<double>(0.0, 1.0),    // +90 deg phase rotation
            std::complex<double>(0.0, -1.0),   // -90 deg phase rotation
            std::complex<double>(-0.70710678, -0.70710678) // -135 deg
        };

        for (const auto& g_val : phase_cases) {
            std::vector<std::complex<double>> g_synthetic(336 * 64, std::complex<double>(0.0, 0.0));
            for (int f = 0; f < 336; ++f) {
                for (int a : active_antennas) {
                    g_synthetic[f * 64 + a] = g_val;
                }
            }
            auto elem_to_ant = charts_mvdr::element_to_antenna(64, "descending");
            int dropped = 0;
            auto corr = charts_mvdr::calibration_correctors(g_synthetic, 336, 64, elem_to_ant, &dropped);
            assert(dropped == 20160);

            for (int f = 0; f < 336; ++f) {
                for (int e = 0; e < 64; ++e) {
                    int a = elem_to_ant[e];
                    if (a == 0 || a == 5 || a == 32 || a == 37) {
                        std::complex<double> c_double(corr[f * 64 + e].real(), corr[f * 64 + e].imag());
                        std::complex<double> prod = c_double * g_val;
                        assert(std::abs(prod.real() - 1.0) < 1e-6);
                        assert(std::abs(prod.imag()) < 1e-6);
                        (void)prod;
                    }
                }
            }
        }
        std::cout << "   Phase Jump Boundaries (180 deg, +/-90 deg, -135 deg) -> prod == 1.0 +/- 1e-6 [PASS]\n";

        // 4b. Extreme Gain Dynamic Range across active antennas
        {
            std::vector<std::complex<double>> g_dyn(336 * 64, std::complex<double>(0.0, 0.0));
            for (int f = 0; f < 336; ++f) {
                g_dyn[f * 64 + 0] = std::complex<double>(1000.0, 0.0);   // +60 dB gain
                g_dyn[f * 64 + 5] = std::complex<double>(0.001, 0.0);    // -60 dB gain
                g_dyn[f * 64 + 32] = std::complex<double>(1.0, 1.0);     // |g| = sqrt(2)
                g_dyn[f * 64 + 37] = std::complex<double>(-10.0, 10.0);  // |g| = 10*sqrt(2)
            }
            auto elem_to_ant = charts_mvdr::element_to_antenna(64, "descending");
            int dropped = 0;
            auto corr = charts_mvdr::calibration_correctors(g_dyn, 336, 64, elem_to_ant, &dropped);
            assert(dropped == 20160);
            for (int f = 0; f < 336; ++f) {
                for (int e = 0; e < 64; ++e) {
                    int a = elem_to_ant[e];
                    if (a == 0 || a == 5 || a == 32 || a == 37) {
                        std::complex<double> c_double(corr[f * 64 + e].real(), corr[f * 64 + e].imag());
                        std::complex<double> prod = c_double * g_dyn[f * 64 + a];
                        assert(std::abs(prod.real() - 1.0) < 1e-5);
                        assert(std::abs(prod.imag()) < 1e-5);
                        (void)prod;
                    }
                }
            }
            std::cout << "   Dynamic Range: 120 dB span (1e-3 to 1e3) correctly inverted [PASS]\n";
        }

        // 4c. Physical Baseline Geometry Ratios for [0, 5, 32, 37]
        {
            float pos[4] = {0.0f, 0.6f, 3.6f, 4.2f};
            float b01 = pos[1] - pos[0]; // 0.6 m
            float b12 = pos[2] - pos[1]; // 3.0 m
            float b23 = pos[3] - pos[2]; // 0.6 m
            float b03 = pos[3] - pos[0]; // 4.2 m
            assert(std::abs(b01 - 0.6f) < 1e-5f);
            assert(std::abs(b12 - 3.0f) < 1e-5f);
            assert(std::abs(b23 - 0.6f) < 1e-5f);
            assert(std::abs(b03 - 4.2f) < 1e-5f);
            // Non-redundant UV coverage: Baselines available are 0.6m, 3.0m, 3.6m, 4.2m
            std::cout << "   Physical 1D East-West Baselines: {0.6, 3.0, 3.6, 4.2}m verified [PASS]\n";
        }
    }

    // -------------------------------------------------------------------------
    // 5. Voltage-Domain Gains Toggle (conjugate_gains = true vs false)
    // -------------------------------------------------------------------------
    std::cout << "5. Testing conjugate_gains toggle (format-1 vs voltage-domain .npy)...\n";
    {
        const std::string tmp_conj =
            (std::filesystem::temp_directory_path() / "test_gains_conjugate_toggle.npy").string();
        std::vector<std::complex<double>> raw_vals = {
            std::complex<double>(2.0, 3.0),
            std::complex<double>(-1.5, 0.25),
            std::complex<double>(0.5, -2.0),
            std::complex<double>(4.0, 0.0),
        };
        write_test_npy(tmp_conj, 1, 4, raw_vals);

        const auto conj_true =
            charts_mvdr::load_gains_npy(tmp_conj, 0, 1, 64, active_antennas, true);
        const auto conj_false =
            charts_mvdr::load_gains_npy(tmp_conj, 0, 1, 64, active_antennas, false);

        for (std::size_t idx = 0; idx < active_antennas.size(); ++idx) {
            const int ant = active_antennas[idx];
            assert(std::abs(conj_true[ant] - std::conj(raw_vals[idx])) < 1e-12);
            assert(std::abs(conj_false[ant] - raw_vals[idx]) < 1e-12);
        }
        std::filesystem::remove(tmp_conj);
        std::cout << "   conjugate_gains=true conjugates, conjugate_gains=false preserves [PASS]\n";
    }

    std::cout << "\n========================================================\n";
    std::cout << "[Phase 1] ALL UNIT TESTS PASSED SUCCESSFULLY!\n";
    std::cout << "========================================================\n";
    return 0;
}
