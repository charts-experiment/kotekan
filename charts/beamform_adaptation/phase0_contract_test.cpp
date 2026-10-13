// Offline golden contract checks; no packet receiver, GPU adapter, or benchmark.
#include "beamformer/indexing.hpp"
#include "beamformer/int4.hpp"
#include "beamformer/fine_channel_pipeline_contract.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>

static void check(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

int main() {
    using namespace beamformer;
    // Literal asymmetric bytes catch swapped nibbles and offset binary decoding.
    struct Byte { unsigned code; int real, imag; };
    for (const auto x : {Byte{0x00,0,0}, Byte{0x17,7,1}, Byte{0x78,-8,7},
                         Byte{0x8f,-1,-8}, Byte{0xf8,-8,-1}, Byte{0x88,-8,-8}}) {
        const auto v = unpack_complex_int4(x.code);
        check(v.real == x.real && v.imag == x.imag, "golden packed decode");
    }
    for (int r=-8; r<=7; ++r) for (int i=-8; i<=7; ++i) {
        const unsigned expected = (r < 0 ? r+16 : r) + 16*(i < 0 ? i+16 : i);
        check(pack_complex_int4(r,i) == expected, "all 256 encodings");
        auto v=unpack_complex_int4(expected);
        check(v.real==r && v.imag==i, "all 256 decodings");
    }
    const Dimensions raw{2,336,64,64};
    // Handler: subband slots have 168 channels; RFSoC 1 precedes RFSoC 0.
    struct Placement { size_t t, slot, channel, rf, e, offset; };
    for (auto p : {Placement{0,0,0,1,0,0}, Placement{0,0,0,0,0,32},
                   Placement{0,0,167,0,31,10751}, Placement{0,1,0,1,0,10752},
                   Placement{1,1,167,0,31,43007}}) {
        const auto element=rfsoc_output_element(p.rf,p.e);
        const auto f=p.slot*168+p.channel;
        for (auto shard : default_shard_descriptors()) {
            check(shard_voltage_index(p.t,f,element,raw,shard)==p.offset,
                  "golden handler destination offset for each independent shard");
            check(absolute_frequency(shard,f)==shard.shard_id*336+f,
                  "shard frequency origin");
        }
    }
    auto input=make_beamformer_input_contract(BeamformerInputMode::ChannelizedFine,352);
    auto md=fine_pipeline_metadata(input);
    check(md.output_dimensions.n_time==2 && md.supplied_spectra(0)==10
          && md.supplied_spectra(1)==1, "partial integration convention");
    const auto cd=channelized_dimensions(RawVoltageDimensions{352,336,64});
    check(channelized_index(1,335,31,63,cd)==1376255, "golden fine offset");
    check(flattened_fine_frequency(335,31,cd)==10751, "fine view endpoint");
    struct Frame { size_t raw_time, fine_time, rows, bytes; double cadence; };
    for (auto frame : {Frame{15360,480,48,132120576,51.2},
                       Frame{153600,4800,480,1321205760,512.0}})
    for (size_t s=0;s<2;++s) {
        auto fine=make_beamformer_input_contract(BeamformerInputMode::ChannelizedFine,frame.raw_time,s);
        auto fm=fine_pipeline_metadata(fine);
        check(fine.dimensions.n_time==frame.fine_time && fm.output_dimensions.n_time==frame.rows,
              "production fine dimensions");
        check(intensity_bytes(fm.output_dimensions)==frame.bytes, "production output bytes");
        check(beamformer_absolute_frequency_start(fine)==s*10752, "fine shard origin");
        check(beamformer_frequency_hz(fine,0)==299854687.5+s*100800000.,
              "golden fine first center");
        check(beamformer_frequency_hz(fine,10751)==400645312.5+s*100800000.,
              "golden fine final center");
        check(beamformer_timestamp_step(fine)==32, "sequence tick time units");
        check(std::abs(beamformer_frame_deadline_ms(fine)-frame.cadence)<1e-12, "cadence");
    }
    // Proposed mask bridge truth table. This specifies the future adapter, it does
    // not exercise a Kotekan implementation. Missing whole spectra mean all F invalid.
    const unsigned missing[]={0,1,0};
    const unsigned expected[]={1,0,1};
    for(size_t t=0;t<3;++t) for(size_t f=0;f<336;++f)
        check(1-missing[t]==expected[t], "mask polarity fixture");
    bool rejected=false;
    try { make_beamformer_input_contract(BeamformerInputMode::ChannelizedFine,33); }
    catch(const std::invalid_argument&) { rejected=true; }
    check(rejected, "unaligned raw time rejected");
    std::cout << "PASS: packed bytes, RFSoC order, shard offsets, fine view/grid, shapes, partial windows, mask specification\n";
}
