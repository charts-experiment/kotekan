#define BOOST_TEST_MODULE "test_charts_beamformer"
#include <boost/test/included/unit_test.hpp>
#include "cudaCommand.hpp"
#include "cudaInputData.hpp"
#include "cudaOutputData.hpp"
#include "cudaSyncInput.hpp"
#include "cudaSyncOutput.hpp"
#include "chartsMetadata.hpp"
#include "chartsBeamMetadata.hpp"
#include "beamformer/cpu_beamformer.hpp"
#include "beamformer/io.hpp"
#include "beamformer/indexing.hpp"
#include "beamformer/int4.hpp"
#include "beamformer/beamformer_input_contract.hpp"
#include "beamformer/temporal_integration.hpp"
#include <filesystem>
#include <cmath>
#include <cstring>
#include <fstream>
#include <future>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <unistd.h>

using namespace beamformer;
using json=nlohmann::json;

namespace {
std::string digest(const Weights& w) {
    uint64_t h=14695981039346656037ULL;
    const auto* bytes=reinterpret_cast<const uint8_t*>(w.data());
    for(size_t i=0;i<w.size()*sizeof(ComplexFloat);++i) {h^=bytes[i];h*=1099511628211ULL;}
    std::ostringstream out;out<<"fnv1a64:"<<std::hex<<std::setfill('0')<<std::setw(16)<<h;return out.str();
}
void file_frame(const std::filesystem::path& path, metadataObject& metadata, const uint8_t* data, size_t size) {
    std::vector<char> bytes(metadata.get_serialized_size());metadata.serialize(bytes.data());
    uint32_t length=bytes.size();std::ofstream file(path,std::ios::binary);
    file.write(reinterpret_cast<const char*>(&length),sizeof length);
    file.write(bytes.data(),bytes.size());file.write(reinterpret_cast<const char*>(data),size);
    if(!file) throw std::runtime_error("fixture write failed");
}
struct Fixture {
    kotekan::Config config;
    kotekan::bufferContainer buffers;
    BeamformerInputContract input;
    Weights weights;
    json settings, manifest;
    std::filesystem::path directory;
    std::shared_ptr<metadataPool> raw_pool,beam_pool;
    std::unique_ptr<Buffer> raw,mask,output;
    std::shared_ptr<cudaDeviceInterface> device;
    std::shared_ptr<cudaCommandState> state;
    std::vector<std::vector<std::unique_ptr<cudaCommand>>> slots;
    static constexpr size_t depth=3;

    Fixture(size_t shard=0,size_t t=11,size_t e=32,size_t b=4,size_t k=10,const std::string& kernel="tiled") {
        static size_t serial=0;
        directory=std::filesystem::temp_directory_path()/ ("charts-adapter-"+std::to_string(getpid())+"-"+std::to_string(serial++));
        if(!std::filesystem::create_directory(directory)) throw std::runtime_error("fixture directory collision");
        input=make_beamformer_input_contract(BeamformerInputMode::RawCoarse,t,shard,e,b);
        weights.resize(b*336*e);
        for(size_t beam=0;beam<b;++beam) for(size_t f=0;f<336;++f) for(size_t a=0;a<e;++a)
            weights[weight_index(beam,f,a,input.dimensions)]={float(1+beam+f%3)*0.25F,float(int(a%3)-1)*0.125F};
        write_weights(directory/"weights.bin",weights,input);
        manifest={{"schema","charts.weights.v1"},{"layout","BFE_complex_float32_le"},{"shard_id",shard},
            {"shape",{b,size_t(336),e}},{"frequency_first_hz",beamformer_frequency_hz(input,0)},
            {"frequency_step_hz",300000.0},{"digest",digest(weights)},{"beam_order_id","fixture_order"}};
        write_manifest();
        settings={{"type","config"},{"log_level","warn"},{"profiling",false},{"buffer_depth",depth},
            {"samples_per_data_set",t},{"num_elements",e},{"num_beams",b},{"integration_spectra",k},
            {"shard_id",shard},{"beamformer_kernel",kernel},
            {"beam",{{"in_buf","raw"},{"mask_buf","missing"},{"out_buf","intensity"},
                     {"gpu_mem_voltage","voltage"},{"gpu_mem_intensity","intensity"},
                     {"weights_file",(directory/"weights.bin").string()},
                     {"weights_manifest",(directory/"weights.json").string()}}},
            {"copy_in",{{"in_buf","raw"},{"gpu_mem","voltage"}}},
            {"copy_out",{{"out_buf","intensity"},{"gpu_mem","intensity"}}}};
        config.update_config(settings);
        raw_pool=metadataPool::create(32,sizeof(chartsMetadata),"raw_pool","chartsMetadata");
        beam_pool=metadataPool::create(32,sizeof(chartsBeamMetadata),"beam_pool","chartsBeamMetadata");
        auto create=[&](const char* name,size_t bytes,std::shared_ptr<metadataPool> pool) {
            return std::make_unique<Buffer>(depth,bytes,pool,name,"standard",0,false,false,std::vector<int>{},false);
        };
        raw=create("raw",packed_voltage_bytes(input.dimensions),raw_pool);
        mask=create("missing",t,raw_pool);
        output=create("intensity",integrated_intensity_count(input.dimensions,k)*sizeof(float),beam_pool);
        raw->register_producer("source");raw->register_producer("zero");mask->register_producer("source");
        output->register_consumer("check");
        buffers.add_buffer("raw",raw.get());buffers.add_buffer("missing",mask.get());buffers.add_buffer("intensity",output.get());
    }
    ~Fixture() {
        slots.clear();state.reset();device.reset(); // finish GPU ownership before releasing host storage
        std::filesystem::remove_all(directory);
    }
    void write_manifest() {std::ofstream(directory/"weights.json")<<manifest.dump();}
    void setup() {
        device=cudaDeviceInterface::get(0,"device_0",config);device->prepareStreams(3);
        state=FACTORY(cudaCommandState)::create_shared("cudaChartsBeamformerCommand",config,"/beam",buffers,*device);
        for(size_t i=0;i<depth;++i) {
            std::vector<std::unique_ptr<cudaCommand>> pipeline;
            pipeline.emplace_back(new cudaInputData(config,"/copy_in",buffers,*device,i));
            pipeline.emplace_back(new cudaSyncInput(config,"/sync_in",buffers,*device,i));
            pipeline.emplace_back(FACTORY_VARIANT(state,cudaCommand)::create_bare(
                "cudaChartsBeamformerCommand",config,"/beam",buffers,*device,i,state));
            pipeline.emplace_back(new cudaSyncOutput(config,"/sync_out",buffers,*device,i));
            pipeline.emplace_back(new cudaOutputData(config,"/copy_out",buffers,*device,i));
            slots.push_back(std::move(pipeline));
        }
    }
    bool missing(size_t t,size_t frame) const {
        return t==(frame+1+input.raw_shard.shard_id)%input.raw_n_time
            || (input.raw_shard.shard_id==1 && t+1==input.raw_n_time);
    }
    PackedVoltage samples(size_t frame) const {
        PackedVoltage data(packed_voltage_bytes(input.dimensions));
        for(size_t t=0;t<input.raw_n_time;++t) for(size_t f=0;f<336;++f) for(size_t a=0;a<input.dimensions.n_ant;++a) {
            int r=int((t+f+a+frame+input.raw_shard.shard_id)%16)-8;
            int im=int((3*t+2*f+a+frame)%16)-8;
            data[voltage_index(t,f,a,input.dimensions)]=missing(t,frame) ? 0 : pack_complex_int4(r,im);
        }
        return data;
    }
    void metadata(chartsMetadata& m,uint64_t seq,bool is_raw) const {
        m.set_fpga_seq_num(seq);m.set_time_downsampling_fpga(1);
        if(is_raw) {
            m.type=kotekan::int4x2;
            std::vector<int> freq(336);std::iota(freq.begin(),freq.end(),input.raw_shard.absolute_frequency_start);
            m.set_coarse_freq(freq); // producer's unset extents/strides are intentionally retained
        }
    }
    void supply(size_t frame,bool finish_zero=true) {
        size_t i=frame%depth;
        auto data=samples(frame);
        std::memcpy(raw->wait_for_empty_frame("source",i),data.data(),data.size());
        auto* flags=mask->wait_for_empty_frame("source",i);
        for(size_t t=0;t<input.raw_n_time;++t) flags[t]=missing(t,frame) ? 1 : 0;
        raw->allocate_new_metadata_object(i);mask->allocate_new_metadata_object(i);
        metadata(*std::dynamic_pointer_cast<chartsMetadata>(raw->get_metadata(i)),15360+frame*input.raw_n_time,true);
        metadata(*std::dynamic_pointer_cast<chartsMetadata>(mask->get_metadata(i)),15360+frame*input.raw_n_time,false);
        raw->mark_frame_full("source",i);mask->mark_frame_full("source",i);
        if(finish_zero) raw->mark_frame_full("zero",i);
    }
};
}

BOOST_AUTO_TEST_CASE(gpu_numerical_and_frame_ownership) {
    // Three frames in flight, six frames total: actual transfers/sync commands and ring reuse.
    for(size_t shard=0;shard<2;++shard) for(const auto kernel : {"tiled","direct"}) {
        const size_t t=std::string(kernel)=="direct" ? 321 : 11;
        const size_t k=std::string(kernel)=="direct" ? 320 : 10;
        const auto beams=std::string(kernel)=="direct" ? 1 : (shard==0 ? 4 : 64);
        Fixture fixture(shard,t,shard==0 ? 32 : 64,beams,k,kernel);fixture.setup();
        for(size_t batch=0;batch<2;++batch) {
            std::vector<cudaEvent_t> completion;
            for(size_t i=0;i<Fixture::depth;++i) {
                const auto frame=batch*Fixture::depth+i;fixture.supply(frame);
                std::vector<cudaEvent_t> events(3,nullptr);cudaPipelineState ps(frame);
                for(auto& command : fixture.slots[i]) {command->start_frame(frame);BOOST_REQUIRE_EQUAL(command->wait_on_precondition(),0);}
                for(auto& command : fixture.slots[i]) {
                    auto event=command->execute_base(ps,events);events[command->get_cuda_stream_id()]=event;
                }
                completion.push_back(events[1]);
            }
            for(size_t i=0;i<Fixture::depth;++i) {
                BOOST_REQUIRE_EQUAL(cudaEventSynchronize(completion[i]),cudaSuccess);
                fixture.slots[i][0]->finalize_frame();
                BOOST_CHECK(!fixture.raw->is_frame_empty(i)); // adapter still retains input
                BOOST_CHECK(!fixture.mask->is_frame_empty(i));
                for(size_t c=1;c<fixture.slots[i].size();++c) fixture.slots[i][c]->finalize_frame();
                BOOST_CHECK(fixture.raw->is_frame_empty(i));BOOST_CHECK(fixture.mask->is_frame_empty(i));
                const auto* actual=reinterpret_cast<const float*>(fixture.output->wait_for_full_frame("check",i));
                auto expected=cpu_integrate_intensity(cpu_beamform_packed_intensity(
                    fixture.samples(batch*Fixture::depth+i),fixture.weights,fixture.input),fixture.input,k);
                for(size_t n=0;n<expected.size();++n)
                    BOOST_CHECK_SMALL(double(actual[n]-expected[n]),1e-3+1e-5*std::abs(double(expected[n])));
                auto meta=std::dynamic_pointer_cast<chartsBeamMetadata>(fixture.output->get_metadata(i));
                BOOST_REQUIRE(meta);meta->validate();
                BOOST_CHECK_EQUAL(meta->document["shard_id"].get<size_t>(),shard);
                BOOST_CHECK_EQUAL(meta->document["supplied_spectra"][1].get<size_t>(),1);
                auto counts=meta->document["valid_spectra"].get<std::vector<size_t>>();
                BOOST_CHECK_EQUAL(std::accumulate(counts.begin(),counts.end(),size_t(0)),t-(shard==0 ? 1 : 2));
                fixture.output->mark_frame_empty("check",i);
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(reject_stale_nonbinary_and_wrong_shard_masks) {
    for(int defect=0;defect<5;++defect) {
        Fixture f;f.setup();f.supply(0);f.slots[0][2]->start_frame(0);
        auto mask_meta=std::dynamic_pointer_cast<chartsMetadata>(f.mask->get_metadata(0));
        if(defect==0) mask_meta->set_fpga_seq_num(15361);
        if(defect==1) f.mask->frames[0][0]=2;
        if(defect==2) mask_meta->set_coarse_freq(std::vector<int>(336,336));
        if(defect==3) f.raw->set_metadata(0,nullptr);
        if(defect==4) std::dynamic_pointer_cast<chartsMetadata>(f.raw->get_metadata(0))->dim[1]=672;
        BOOST_CHECK_THROW(f.slots[0][2]->wait_on_precondition(),std::exception);
    }
}
BOOST_AUTO_TEST_CASE(wait_for_all_producers) {
    Fixture f;f.setup();f.supply(0,false);f.slots[0][2]->start_frame(0);
    auto waiting=std::async(std::launch::async,[&] {return f.slots[0][2]->wait_on_precondition();});
    auto status=waiting.wait_for(std::chrono::milliseconds(20));
    f.raw->mark_frame_full("zero",0);
    BOOST_CHECK(status==std::future_status::timeout);
    BOOST_CHECK_EQUAL(waiting.get(),0);
}
BOOST_AUTO_TEST_CASE(host_registration_lifetime) {
    Fixture f;f.setup();unsigned flags=0;
    for(auto* buffer:{f.raw.get(),f.output.get()})
        for(size_t i=0;i<Fixture::depth;++i)
            BOOST_CHECK_EQUAL(cudaHostGetFlags(&flags,buffer->frames[i]),cudaSuccess);
    BOOST_CHECK_EQUAL(cudaPeekAtLastError(),cudaSuccess);
    f.slots.clear();
    for(auto* buffer:{f.raw.get(),f.output.get()})
        for(size_t i=0;i<Fixture::depth;++i) {
            BOOST_CHECK_EQUAL(cudaHostGetFlags(&flags,buffer->frames[i]),cudaErrorInvalidValue);
            (void)cudaGetLastError(); // expected probe in this test
        }
}
BOOST_AUTO_TEST_CASE(reject_configuration_and_weights) {
    for(int defect=0;defect<10;++defect) {
        Fixture f;
        if(defect==0) f.settings["integration_spectra"]=16;
        if(defect==1) {f.manifest["shard_id"]=1;f.write_manifest();}
        if(defect==2) {f.manifest["digest"]="wrong";f.write_manifest();}
        if(defect==3) {f.manifest["layout"]="tiled";f.write_manifest();}
        if(defect==4) f.settings["beam"]["gpu_mem_intensity"]="voltage";
        if(defect==5) f.output->frame_size-=4;
        if(defect==6) std::filesystem::resize_file(f.directory/"weights.bin",1);
        if(defect==7) {f.manifest["frequency_first_hz"]=123;f.write_manifest();}
        if(defect==8) f.raw->frame_size-=1;
        if(defect==9) f.mask->frame_size-=1;
        f.config.update_config(f.settings);
        BOOST_CHECK_THROW(f.setup(),std::exception);
    }
}
BOOST_AUTO_TEST_CASE(analytical_constant_validity_annotation_and_duplicate_rejection) {
    Fixture f(0,1,64,64,10);
    std::fill(f.weights.begin(),f.weights.end(),ComplexFloat{1,0});
    write_weights(f.directory/"weights.bin",f.weights,f.input);
    f.manifest["digest"]=digest(f.weights);f.write_manifest();f.setup();
    for(size_t frame=0;frame<2;++frame) {
        f.supply(frame);
        // Deliberate nonzero data in an invalid spectrum on frame 1 proves that
        // validity annotates supplied power; it does not exclude/normalize it.
        std::memset(f.raw->frames[frame],0x17,f.raw->frame_size);
        f.mask->frames[frame][0]=frame==0 ? 0 : 1;
        std::vector<cudaEvent_t> events(3,nullptr);cudaPipelineState ps(frame);
        for(auto& command:f.slots[frame]) {
            command->start_frame(frame);
            BOOST_REQUIRE_EQUAL(command->wait_on_precondition(),0);
        }
        for(auto& command:f.slots[frame])
            events[command->get_cuda_stream_id()]=command->execute_base(ps,events);
        BOOST_REQUIRE_EQUAL(cudaEventSynchronize(events[1]),cudaSuccess);
        for(auto& command:f.slots[frame]) command->finalize_frame();
        const auto* result=reinterpret_cast<const float*>(f.output->wait_for_full_frame("check",frame));
        for(size_t i=0;i<336*64;++i) BOOST_CHECK_EQUAL(result[i],204800.0F);
        auto meta=std::dynamic_pointer_cast<chartsBeamMetadata>(f.output->get_metadata(frame));
        BOOST_REQUIRE(meta);
        BOOST_CHECK_EQUAL(meta->document["valid_spectra"][0].get<size_t>(),frame==0 ? 1 : 0);
        f.output->mark_frame_empty("check",frame);
    }
    f.supply(2);
    for(auto* buffer:{f.raw.get(),f.mask.get()})
        std::dynamic_pointer_cast<chartsMetadata>(buffer->get_metadata(2))->set_fpga_seq_num(15360);
    f.slots[2][2]->start_frame(2);
    BOOST_CHECK_THROW(f.slots[2][2]->wait_on_precondition(),std::exception);
}
BOOST_AUTO_TEST_CASE(emit_replay_fixture) {
    auto& suite=boost::unit_test::framework::master_test_suite();
    std::filesystem::path destination;
    for(int i=1;i+1<suite.argc;++i) if(std::string(suite.argv[i])=="--fixture-dir") destination=suite.argv[i+1];
    if(destination.empty()) return;
    std::filesystem::create_directories(destination);
    for(size_t shard=0;shard<2;++shard) {
        Fixture f(shard,11,64,4,10);
        const auto suffix=".shard"+std::to_string(shard);
        std::filesystem::copy_file(f.directory/"weights.bin",destination/("weights"+suffix+".bin"));
        std::filesystem::copy_file(f.directory/"weights.json",destination/("weights"+suffix+".json"));
        for(size_t frame=0;frame<6;++frame) {
            std::ostringstream n;n<<std::setw(7)<<std::setfill('0')<<frame;
            auto data=f.samples(frame);std::vector<uint8_t> mask(11,0);
            for(size_t t=0;t<11;++t) mask[t]=f.missing(t,frame) ? 1 : 0;
            chartsMetadata voltage_meta,mask_meta;
            f.metadata(voltage_meta,15360+frame*11,true);f.metadata(mask_meta,15360+frame*11,false);
            file_frame(destination/("voltage"+suffix+"_"+n.str()+".bin"),voltage_meta,data.data(),data.size());
            file_frame(destination/("missing"+suffix+"_"+n.str()+".bin"),mask_meta,mask.data(),mask.size());
            write_packed_voltage(destination/("packed"+suffix+"_"+n.str()+".bin"),data,f.input);
        }
    }
}
