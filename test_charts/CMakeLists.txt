project(test_charts)

if(CUDAToolkit_FOUND)
    set(COMMON_TEST_SRCS
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib/cuda/cudaBeamTrackerV5.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib/cuda/cudaDirectBeamTracker.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib/cuda/cudaAntennaMask.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib/cuda/cudaBiSLC.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib/cuda/cudaTransientTrigger.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib/core/kotekanLogging.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib/core/errors.c)

    set(COMMON_INCLUDE_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib/cuda
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib/metadata
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib/core
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib/stages
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib/dpdk
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib/utils
        ${CMAKE_CURRENT_SOURCE_DIR}/../lib
        ${CMAKE_CURRENT_SOURCE_DIR}/../external/json
        ${CMAKE_CURRENT_SOURCE_DIR}/../external/fmt
        ${CMAKE_CURRENT_SOURCE_DIR}/../external/gputils/src_lib
        ${CMAKE_CURRENT_SOURCE_DIR}/../external/gputils/src_include)

    # 1. Benchmark Kotekan CUDA Tracker V5
    add_executable(benchmark_kotekan_cuda_tracker_v5 benchmark_kotekan_cuda_tracker_v5.cpp
                                                     ${COMMON_TEST_SRCS})
    target_link_libraries(benchmark_kotekan_cuda_tracker_v5 PRIVATE CUDA::cudart Threads::Threads)
    target_include_directories(benchmark_kotekan_cuda_tracker_v5 PRIVATE ${COMMON_INCLUDE_DIRS})

    # 1. Benchmark Pipeline Stages
    add_executable(benchmark_pipeline_stages benchmark_pipeline_stages.cpp ${COMMON_TEST_SRCS})
    target_link_libraries(benchmark_pipeline_stages PRIVATE CUDA::cudart Threads::Threads)
    target_include_directories(benchmark_pipeline_stages PRIVATE ${COMMON_INCLUDE_DIRS})

    # 1. Extensive Kotekan Beam Tracker Comprehensive Test Suite
    add_executable(test_beam_tracker_suite test_beam_tracker_suite.cpp ${COMMON_TEST_SRCS})
    target_link_libraries(test_beam_tracker_suite PRIVATE CUDA::cudart Threads::Threads)
    target_include_directories(test_beam_tracker_suite PRIVATE ${COMMON_INCLUDE_DIRS})

    # 1. Chaos Engineering & Fault Injection Test Suite
    add_executable(test_chaos_engineering test_chaos_engineering.cpp ${COMMON_TEST_SRCS})
    target_link_libraries(test_chaos_engineering PRIVATE CUDA::cudart Threads::Threads)
    target_include_directories(test_chaos_engineering PRIVATE ${COMMON_INCLUDE_DIRS})

    # 1. Benchmark Direct Beam Tracker (Zero Integration Window)
    add_executable(benchmark_direct_beam_tracker benchmark_direct_beam_tracker.cpp
                                                 ${COMMON_TEST_SRCS})
    target_link_libraries(benchmark_direct_beam_tracker PRIVATE CUDA::cudart Threads::Threads)
    target_include_directories(benchmark_direct_beam_tracker PRIVATE ${COMMON_INCLUDE_DIRS})

    # 1. Head-to-Head Comparative Benchmark: Beam Tracker V5 vs. Direct Beam Tracker
    add_executable(benchmark_tracker_v5_vs_direct benchmark_tracker_v5_vs_direct.cpp
                                                  ${COMMON_TEST_SRCS})
    target_link_libraries(benchmark_tracker_v5_vs_direct PRIVATE CUDA::cudart Threads::Threads)
    target_include_directories(benchmark_tracker_v5_vs_direct PRIVATE ${COMMON_INCLUDE_DIRS})

    # 1. Dead & Saturated Antenna Masking Stage Test Suite
    add_executable(test_antenna_masking_stage test_antenna_masking_stage.cpp ${COMMON_TEST_SRCS})
    target_link_libraries(test_antenna_masking_stage PRIVATE CUDA::cudart Threads::Threads)
    target_include_directories(test_antenna_masking_stage PRIVATE ${COMMON_INCLUDE_DIRS})

    # 1. Big Master Pipeline Test: Masking Stage -> Direct Beam Tracker
    add_executable(benchmark_pipeline_masking_direct benchmark_pipeline_masking_direct.cpp
                                                     ${COMMON_TEST_SRCS})
    target_link_libraries(benchmark_pipeline_masking_direct PRIVATE CUDA::cudart Threads::Threads)
    target_include_directories(benchmark_pipeline_masking_direct PRIVATE ${COMMON_INCLUDE_DIRS})

    # 1. BiSLC (Joint Matrix Unmixing) Coherent Sidelobe Cancellation Suite
    add_executable(test_bislc_suite test_bislc_suite.cpp ${COMMON_TEST_SRCS})
    target_link_libraries(test_bislc_suite PRIVATE CUDA::cudart Threads::Threads)
    target_include_directories(test_bislc_suite PRIVATE ${COMMON_INCLUDE_DIRS})

    # 1. Sub-frame Phase Interpolation Suite
    add_executable(test_phase_interpolation_suite test_phase_interpolation_suite.cpp
                                                  ${COMMON_TEST_SRCS})
    target_link_libraries(test_phase_interpolation_suite PRIVATE CUDA::cudart Threads::Threads)
    target_include_directories(test_phase_interpolation_suite PRIVATE ${COMMON_INCLUDE_DIRS})

    # 1. Transient Trigger Comprehensive Test Suite (SK + Cross-Correlation + Ring Buffer Dump)
    add_executable(test_transient_trigger_suite test_transient_trigger_suite.cpp
                                                ${COMMON_TEST_SRCS})
    target_link_libraries(test_transient_trigger_suite PRIVATE CUDA::cudart Threads::Threads)
    target_include_directories(test_transient_trigger_suite PRIVATE ${COMMON_INCLUDE_DIRS})
endif()
