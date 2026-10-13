#ifndef CPU_CORR_AVX_HPP
#define CPU_CORR_AVX_HPP

#include "Config.hpp"
#include "Stage.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"

#include <immintrin.h>
#include <stdint.h>
#include <string>

class correlationAVX : public kotekan::Stage {
public:
    correlationAVX(kotekan::Config& config, const std::string& unique_name,
                   kotekan::bufferContainer& buffer_container);
    ~correlationAVX();
    void main_thread() override;

private:
    void compute_correlations_avx(int8_t* x, int8_t* y, uint32_t nsamp, float* out);

    Buffer* buf_in;
    Buffer* buf_out;

    uint32_t _num_samples;
    uint32_t _integration;
};

#endif
