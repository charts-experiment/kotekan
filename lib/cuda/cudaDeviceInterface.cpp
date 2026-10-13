#include "cudaDeviceInterface.hpp"

#include "cudaUtils.hpp"      // for CHECK_CUDA_ERROR
#include "cuda_runtime_api.h" // for cudaEventCreate, cudaEventRecord, cudaMemcpyAsync, cudaStr...
#include "kotekanLogging.hpp" // for FATAL_ERROR, INFO, DEBUG2

#include "fmt.hpp" // for compile_string_to_view

#include <assert.h> // for assert
#include <cstdlib>
#include <cuda.h> // for cuGetErrorString, cuModuleGetFunction, cuModuleLoadDataEx
#include <filesystem>
#include <mutex>           // for mutex, lock_guard
#include <nvPTXCompiler.h> // for NVPTXCOMPILE_SUCCESS, nvPTXCompilerCompile, nvPTXCompilerC...
#include <nvrtc.h>         // for nvrtcGetErrorString, NVRTC_SUCCESS, nvrtcCompileProgram
#include <set>
#include <sstream>
#include <stdexcept> // for runtime_error
#include <stdio.h>   // for fclose, fopen, fread, fseek, ftell, rewind, FILE, SEEK_END
#include <stdlib.h>  // for free, malloc
#include <system_error>
#include <utility> // for pair

using kotekan::Config;

std::map<int32_t, std::weak_ptr<cudaDeviceInterface>> cudaDeviceInterface::inst_map;

// Protects access to inst_map
static std::mutex cuda_inst_map_mutex;

std::shared_ptr<cudaDeviceInterface>
cudaDeviceInterface::get(int32_t gpu_id, const std::string& name, Config& config) {
    std::lock_guard<std::mutex> lock(cuda_inst_map_mutex);
    auto it = inst_map.find(gpu_id);
    if (it != inst_map.end()) {
        if (auto existing = it->second.lock())
            // it->second is a std::weak_ptr. lock() attempts to create a new
            // shared_ptr that shares ownership with any existing shared owners.
            // If the weak_ptr has not expired, 'existing' becomes a valid shared_ptr.
            return existing;
    }
    auto dev = std::make_shared<cudaDeviceInterface>(config, name,
                                                     gpu_id); // creates an owning std::shared_ptr
    inst_map[gpu_id] = dev; // store weak reference (implicit conversion)
    return dev;
}

cudaDeviceInterface::cudaDeviceInterface(Config& config, const std::string& unique_name,
                                         int32_t gpu_id) :
    gpuDeviceInterface(config, unique_name, gpu_id) {

    // Find out how many GPUs can be probed.
    int max_num_gpus;
    CHECK_CUDA_ERROR(cudaGetDeviceCount(&max_num_gpus));
    INFO("Number of CUDA GPUs: {:d}", max_num_gpus);

    if (gpu_id > max_num_gpus) {
        throw std::runtime_error(
            "Asked for a GPU ID which is higher than the maximum number of GPUs in the system");
    }

    set_thread_device();
}

cudaDeviceInterface::~cudaDeviceInterface() {
    for (auto& stream : streams) {
        CHECK_CUDA_ERROR(cudaStreamDestroy(stream));
    }
    cleanup_memory();
}

void cudaDeviceInterface::set_thread_device() {
    CHECK_CUDA_ERROR(cudaSetDevice(gpu_id));
}

void* cudaDeviceInterface::alloc_gpu_memory(size_t len) {
    void* ret;
    CHECK_CUDA_ERROR(cudaMalloc(&ret, len));
    return ret;
}
void cudaDeviceInterface::free_gpu_memory(void* ptr) {
    CHECK_CUDA_ERROR(cudaFree(ptr));
}

cudaStream_t cudaDeviceInterface::getStream(int32_t cuda_stream_id) {
    return streams[cuda_stream_id];
}

int32_t cudaDeviceInterface::get_num_streams() {
    return streams.size();
}

void cudaDeviceInterface::prepareStreams(uint32_t num_streams) {
    // Create GPU command queues
    for (uint32_t i = streams.size(); i < num_streams; ++i) {
        cudaStream_t stream = nullptr;
        CHECK_CUDA_ERROR(cudaStreamCreate(&stream));
        streams.push_back(stream);
    }
}

void cudaDeviceInterface::async_copy_host_to_gpu(void* dst, void* src, size_t len,
                                                 uint32_t cuda_stream_id, cudaEvent_t pre_event,
                                                 cudaEvent_t* copy_start_event,
                                                 cudaEvent_t* copy_end_event) {
    if (pre_event)
        CHECK_CUDA_ERROR(cudaStreamWaitEvent(getStream(cuda_stream_id), pre_event, 0));
    if (copy_start_event) {
        CHECK_CUDA_ERROR(cudaEventCreate(copy_start_event));
        CHECK_CUDA_ERROR(cudaEventRecord(*copy_start_event, getStream(cuda_stream_id)));
    }
    // Data transfer to GPU
    CHECK_CUDA_ERROR(
        cudaMemcpyAsync(dst, src, len, cudaMemcpyHostToDevice, getStream(cuda_stream_id)));
    if (copy_end_event) {
        CHECK_CUDA_ERROR(cudaEventCreate(copy_end_event));
        CHECK_CUDA_ERROR(cudaEventRecord(*copy_end_event, getStream(cuda_stream_id)));
    }
}

void cudaDeviceInterface::async_copy_gpu_to_host(void* dst, void* src, size_t len,
                                                 uint32_t cuda_stream_id, cudaEvent_t pre_event,
                                                 cudaEvent_t* copy_start_event,
                                                 cudaEvent_t* copy_end_event) {
    if (pre_event)
        CHECK_CUDA_ERROR(cudaStreamWaitEvent(getStream(cuda_stream_id), pre_event, 0));
    if (copy_start_event) {
        CHECK_CUDA_ERROR(cudaEventCreate(copy_start_event));
        CHECK_CUDA_ERROR(cudaEventRecord(*copy_start_event, getStream(cuda_stream_id)));
    }
    // Data transfer from GPU
    CHECK_CUDA_ERROR(
        cudaMemcpyAsync(dst, src, len, cudaMemcpyDeviceToHost, getStream(cuda_stream_id)));
    if (copy_end_event) {
        CHECK_CUDA_ERROR(cudaEventCreate(copy_end_event));
        CHECK_CUDA_ERROR(cudaEventRecord(*copy_end_event, getStream(cuda_stream_id)));
    }
}

static std::vector<std::string> get_cuda_include_paths() {
    std::vector<std::string> inc_paths;
    std::set<std::string> seen;

    auto try_add = [&](const std::string& path) {
        if (path.empty())
            return;
        std::error_code ec;
        if (std::filesystem::is_directory(path, ec)) {
            std::string canon;
            try {
                canon = std::filesystem::canonical(path, ec).string();
            } catch (...) {
                canon = path;
            }
            if (canon.empty())
                canon = path;
            if (seen.find(canon) == seen.end()) {
                seen.insert(canon);
                inc_paths.push_back(path);
            }
            // In modern CUDA (e.g. CUDA 12+), CCCL headers (cuda/std, cooperative_groups, thrust,
            // cub) may reside in cccl/
            std::string cccl = path + "/cccl";
            if (std::filesystem::is_directory(cccl, ec)) {
                std::string canon_cccl;
                try {
                    canon_cccl = std::filesystem::canonical(cccl, ec).string();
                } catch (...) {
                    canon_cccl = cccl;
                }
                if (canon_cccl.empty())
                    canon_cccl = cccl;
                if (seen.find(canon_cccl) == seen.end()) {
                    seen.insert(canon_cccl);
                    inc_paths.push_back(cccl);
                }
            }
        }
    };

    // 1. Environment variables from cluster modules (e.g. Trillium, Compute Canada, SLURM)
    const char* env_vars[] = {"CUDA_HOME", "CUDA_PATH",        "CUDA_ROOT",     "EBROOTCUDA",
                              "CUDADIR",   "CUDA_INCLUDE_DIR", "CUDA_INC_PATH", "CUDAToolkit_ROOT"};
    for (const char* var : env_vars) {
        const char* val = std::getenv(var);
        if (val && *val) {
            std::string root(val);
            try_add(root + "/include");
            try_add(root);
        }
    }

    // 2. Check CPATH / C_INCLUDE_PATH / CPLUS_INCLUDE_PATH
    const char* inc_envs[] = {"CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH"};
    for (const char* var : inc_envs) {
        const char* val = std::getenv(var);
        if (val && *val) {
            std::stringstream ss(val);
            std::string item;
            while (std::getline(ss, item, ':')) {
                if (!item.empty()) {
                    try_add(item);
                }
            }
        }
    }

    // 3. Compile-time definition from CMake FindCUDAToolkit
#ifdef KOTEKAN_CUDA_INCLUDE_DIRS
    {
        std::stringstream ss(KOTEKAN_CUDA_INCLUDE_DIRS);
        std::string item;
        while (std::getline(ss, item, ';')) {
            if (!item.empty()) {
                try_add(item);
            }
        }
    }
#endif

    // 4. Standard default fallback locations
    try_add("/usr/local/cuda/include");
    try_add("/usr/local/include");
    try_add("/opt/cuda/include");
    try_add("/usr/include");

    return inc_paths;
}

void cudaDeviceInterface::build(const std::string& kernel_filename,
                                const std::vector<std::string>& kernel_names,
                                const std::vector<std::string>& opts) {
    size_t program_size;
    FILE* fp;
    char* program_buffer;
    nvrtcResult res;

    for (auto& kernel_name : kernel_names)
        if (runtime_kernels.count(kernel_name))
            FATAL_ERROR("Building CUDA kernels in file {:s}: kernel \"{:s}\" already exists.",
                        kernel_filename, kernel_name);

    // DEBUG("Building! {:s}", kernel_command)
    //  Load the kernel file contents into `program_buffer`
    fp = fopen(kernel_filename.c_str(), "r");
    if (fp == nullptr) {
        FATAL_ERROR("error loading file: {:s}", kernel_filename.c_str());
    }
    fseek(fp, 0, SEEK_END);
    program_size = ftell(fp);
    rewind(fp);

    program_buffer = (char*)malloc(program_size + 1);
    program_buffer[program_size] = '\0';
    int sizeRead = fread(program_buffer, sizeof(char), program_size, fp);
    if (sizeRead < (int32_t)program_size)
        FATAL_ERROR("Error reading the file: {:s}", kernel_filename);
    fclose(fp);

    // Create the program object
    nvrtcProgram prog;
    res = nvrtcCreateProgram(&prog, program_buffer, nullptr, 0, nullptr, nullptr);
    if (res != NVRTC_SUCCESS) {
        const char* error_str = nvrtcGetErrorString(res);
        INFO("ERROR IN nvrtcCreateProgram: {}", error_str);
    }

    free(program_buffer);

    // Target the local GPU's compute capability. A hardcoded -arch (e.g.
    // compute_86) produces PTX the driver JIT may reject on newer GPUs.
    int cc_major = 0, cc_minor = 0;
    CHECK_CUDA_ERROR(cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, gpu_id));
    CHECK_CUDA_ERROR(cudaDeviceGetAttribute(&cc_minor, cudaDevAttrComputeCapabilityMinor, gpu_id));
    const std::string arch_opt = fmt::format("-arch=sm_{:d}{:d}", cc_major, cc_minor);

    std::vector<std::string> local_opts(opts);
    bool arch_found = false;
    for (auto& s : local_opts) {
        if (s.rfind("-arch=", 0) == 0) {
            if (s != arch_opt)
                INFO("GPU[{:d}] device interface: Replacing option {} with {} for local GPU",
                     gpu_id, s, arch_opt);
            s = arch_opt;
            arch_found = true;
        }
    }
    if (!arch_found)
        local_opts.push_back(arch_opt);

    // Dynamically resolved CUDA include paths for NVRTC
    std::vector<std::string> extra_opts;
    for (const auto& inc_path : get_cuda_include_paths()) {
        INFO("cudaDeviceInterface: Using NVRTC include path: {:s}", inc_path);
        extra_opts.push_back(std::string("--include-path=") + inc_path);
    }
    extra_opts.push_back("--std=c++17");

    // Convert compiler options to a c-style array.
    std::vector<const char*> cstrings;
    cstrings.reserve(local_opts.size() + extra_opts.size());

    for (auto& s : local_opts)
        cstrings.push_back(s.c_str());
    for (auto& s : extra_opts)
        cstrings.push_back(s.c_str());

    // Compile the kernel
    res =
        nvrtcCompileProgram(prog, cstrings.size(), cstrings.data()); // TODO Abstract error checking

    if (res != NVRTC_SUCCESS) {

        size_t logSize = 0;
        nvrtcGetProgramLogSize(prog, &logSize);

        std::string log;
        log.resize(logSize > 1 ? logSize : 1); // evita size 0
        nvrtcGetProgramLog(prog, log.data());

        // 2) Mensaje de error + log
        const char* error_str = nvrtcGetErrorString(res);
        FATAL_ERROR("ERROR IN nvrtcCompileProgram: {}\nNVRTC LOG:\n{}", error_str, log);
    }


    // Obtain CUBIN from the program: native SASS for the local GPU, so the
    // driver does not need to JIT-compile PTX.
    size_t cubinSize;
    res = nvrtcGetCUBINSize(prog, &cubinSize);
    if (res != NVRTC_SUCCESS) {
        const char* error_str = nvrtcGetErrorString(res);
        FATAL_ERROR("ERROR IN nvrtcGetCUBINSize: {}", error_str);
    }
    std::vector<char> cubin(cubinSize);
    res = nvrtcGetCUBIN(prog, cubin.data());
    if (res != NVRTC_SUCCESS) {
        const char* error_str = nvrtcGetErrorString(res);
        FATAL_ERROR("ERROR IN nvrtcGetCUBIN: {}", error_str);
    }
    DEBUG2("CUBIN EXTRACTED");
    res = nvrtcDestroyProgram(&prog);
    if (res != NVRTC_SUCCESS) {
        const char* error_str = nvrtcGetErrorString(res);
        FATAL_ERROR("ERROR IN nvrtcDestroyProgram: {}", error_str);
    }

    CUresult err;
    CUmodule module;
    // Get the module with the kernels
    err = cuModuleLoadDataEx(&module, cubin.data(), 0, nullptr, nullptr);
    if (err != CUDA_SUCCESS) {
        const char* errStr;
        cuGetErrorString(err, &errStr);
        FATAL_ERROR("ERROR IN cuModuleLoadDataEx: {}", errStr);
    }

    for (auto& kernel_name : kernel_names) {
        runtime_kernels.emplace(kernel_name, nullptr);
        err = cuModuleGetFunction(&runtime_kernels[kernel_name], module, kernel_name.c_str());
        if (err != CUDA_SUCCESS) {
            const char* errStr;
            cuGetErrorString(err, &errStr);
            FATAL_ERROR("ERROR IN cuModuleGetFunction for correlate: {}", errStr);
        }
        if (runtime_kernels[kernel_name] == nullptr) {
            FATAL_ERROR("Failed to find kernel name \"{}\" in compiled CUBIN module", kernel_name);
        }
    }
}

void cudaDeviceInterface::build_ptx(const std::string& kernel_filename,
                                    const std::vector<std::string>& kernel_names,
                                    const std::vector<std::string>& opts,
                                    const std::string& kernel_name_prefix) {
    size_t program_size;
    FILE* fp;
    char* program_buffer;
    nvPTXCompileResult nv_res;
    CUresult cu_res;
    nvPTXCompilerHandle compiler = nullptr;
    size_t elf_size;
    char* elf;
    CUmodule module;

    for (auto& kernel_name : kernel_names)
        if (runtime_kernels.count(kernel_name_prefix + kernel_name))
            FATAL_ERROR("Building CUDA kernels in file {:s}: kernel \"{:s}\" already exists.",
                        kernel_filename, kernel_name_prefix + kernel_name);
    // DEBUG("Building! {:s}", kernel_command)

    // Load the kernel file contents into `program_buffer`
    fp = fopen(kernel_filename.c_str(), "r");
    if (fp == NULL) {
        FATAL_ERROR("error loading file: {:s}", kernel_filename.c_str());
    }
    fseek(fp, 0, SEEK_END);
    program_size = ftell(fp);
    rewind(fp);

    program_buffer = (char*)malloc(program_size + 1);
    program_buffer[program_size] = '\0';
    int sizeRead = fread(program_buffer, sizeof(char), program_size, fp);
    if (sizeRead < (int32_t)program_size)
        FATAL_ERROR("Error reading the file: {:s}", kernel_filename);
    fclose(fp);

    // Create the compiler
    nv_res = nvPTXCompilerCreate(&compiler, program_size, program_buffer);
    if (nv_res != NVPTXCOMPILE_SUCCESS) {
        // TODO Report ENUM names.
        FATAL_ERROR("Could not create PTX compiler, error code: {:d}", (int)nv_res);
        return;
    }

    // Compile for the compute capability of the GPU that is actually present. Callers
    // pass the --gpu-name their kernel was generated for, but SASS is only compatible
    // within a major architecture (an sm_89 cubin fails to load on an sm_86 device with
    // "no kernel image is available for execution on the device"), so kernels would
    // otherwise only run on the specific GPU model they were generated for.
    cudaDeviceProp prop;
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&prop, gpu_id));
    const std::string gpu_name = fmt::format("--gpu-name=sm_{:d}{:d}", prop.major, prop.minor);
    std::vector<std::string> compile_opts;
    compile_opts.reserve(opts.size() + 1);
    for (const std::string& opt : opts) {
        if (opt.rfind("--gpu-name", 0) == 0) {
            if (opt != gpu_name)
                INFO("Kernel file {:s} was generated for {:s}; compiling for the local GPU with "
                     "{:s} instead",
                     kernel_filename, opt, gpu_name);
        } else {
            compile_opts.push_back(opt);
        }
    }
    compile_opts.push_back(gpu_name);

    // Convert compiler options to a c-style array.
    std::vector<const char*> cstring_opts;
    cstring_opts.reserve(compile_opts.size());
    for (auto& s : compile_opts)
        cstring_opts.push_back(s.c_str());

    // Compile the code
    nv_res = nvPTXCompilerCompile(compiler, cstring_opts.size(), cstring_opts.data());
    // TODO Abstract error checking
    if (nv_res != NVPTXCOMPILE_SUCCESS) {
        size_t error_size;
        char* error_log = nullptr;
        nv_res = nvPTXCompilerGetErrorLogSize(compiler, &error_size);
        if (nv_res != NVPTXCOMPILE_SUCCESS) {
            FATAL_ERROR("Could not get error log size, error code: {:d}", (int)nv_res);
            return;
        }
        if (error_size != 0) {
            error_log = (char*)malloc(error_size + 1);
            assert(error_log != nullptr);
            nv_res = nvPTXCompilerGetErrorLog(compiler, error_log);
            if (nv_res != NVPTXCOMPILE_SUCCESS) {
                FATAL_ERROR("Could not get error log, error code: {:d}", (int)nv_res);
                free(error_log);
                return;
            }
        }
        FATAL_ERROR("Could not compile PTX: \n{:s}", error_log);
        free(error_log);
        return;
    }

    nv_res = nvPTXCompilerGetCompiledProgramSize(compiler, &elf_size);
    if (nv_res != NVPTXCOMPILE_SUCCESS) {
        FATAL_ERROR("Could not get compiled PTX elf size, error code: {:d}", (int)nv_res);
        return;
    }

    elf = (char*)malloc(elf_size);
    assert(elf != nullptr);
    nv_res = nvPTXCompilerGetCompiledProgram(compiler, (void*)elf);
    if (nv_res != NVPTXCOMPILE_SUCCESS) {
        FATAL_ERROR("Could not get compiled PTX elf data, error code: {:d}", (int)nv_res);
        return;
    }

    // Dump Logs
    size_t info_size;
    nv_res = nvPTXCompilerGetInfoLogSize(compiler, &info_size);
    if (nv_res != NVPTXCOMPILE_SUCCESS) {
        FATAL_ERROR("Could not get info log size, error code: {:d}", (int)nv_res);
        return;
    }

    if (info_size != 0) {
        char* info_Log = (char*)malloc(info_size + 1);
        nv_res = nvPTXCompilerGetInfoLog(compiler, info_Log);
        if (nv_res != NVPTXCOMPILE_SUCCESS) {
            FATAL_ERROR("Could not get PTX compiler logs, error code: {:d}", (int)nv_res);
            free(info_Log);
            return;
        }
        INFO("PTX Compiler logs: \n{:s}", info_Log);
        free(info_Log);
    }

    // Cleanup compiler
    nv_res = nvPTXCompilerDestroy(&compiler);
    if (nv_res != NVPTXCOMPILE_SUCCESS) {
        FATAL_ERROR("Could not destroy compiler, error code: {:d}", (int)nv_res);
        return;
    }

    // Extract kernels
    cu_res = cuModuleLoadDataEx(&module, elf, 0, nullptr, nullptr);
    if (cu_res != CUDA_SUCCESS) {
        const char* errStr = nullptr;
        cuGetErrorString(cu_res, &errStr);
        FATAL_ERROR("Could not load module data from elf for kernel file {:s}: {:s}",
                    kernel_filename, errStr);
        return;
    }

    for (auto& kernel_name : kernel_names) {
        runtime_kernels.emplace(kernel_name_prefix + kernel_name, nullptr);
        cu_res = cuModuleGetFunction(&runtime_kernels[kernel_name_prefix + kernel_name], module,
                                     kernel_name.c_str());
        if (cu_res != CUDA_SUCCESS) {
            const char* errStr;
            cuGetErrorString(cu_res, &errStr);
            FATAL_ERROR("ERROR IN cuModuleGetFunction for correlate: {:s}", errStr);
        }
    }

    free(elf);
}
