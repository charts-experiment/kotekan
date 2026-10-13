# Optional local scientific dependency. No network access, source copying, or stages.
include_guard(GLOBAL)

if(USE_CHARTS_BEAMFORMER)
    if(CMAKE_VERSION VERSION_LESS 3.24)
        message(FATAL_ERROR "CHARTS beamformer requires CMake >= 3.24")
    endif()
    if(NOT CUDAToolkit_FOUND OR NOT CMAKE_CUDA_COMPILER)
        message(FATAL_ERROR "USE_CHARTS_BEAMFORMER=ON requires the CUDA backend (USE_CUDA=ON)")
    endif()
    if(NOT EXISTS "${CHARTS_BEAMFORMER_SOURCE_DIR}/include/beamformer/cuda_stage_api.hpp"
       OR NOT EXISTS "${CHARTS_BEAMFORMER_SOURCE_DIR}/CMakeLists.txt")
        message(FATAL_ERROR "Set CHARTS_BEAMFORMER_SOURCE_DIR to the local beamform_project checkout")
    endif()
endif()

# Called from lib/cuda after Kotekan selects its CUDA architecture. Function scope
# keeps embedded-project options from changing standalone/parent cache settings.
function(kotekan_add_charts_beamformer)
    if(NOT USE_CHARTS_BEAMFORMER)
        return()
    endif()
    set(BEAMFORMER_ENABLE_CUDA ON)
    set(BEAMFORMER_BUILD_TOOLS OFF)
    set(BEAMFORMER_BUILD_TESTS OFF)
    add_subdirectory("${CHARTS_BEAMFORMER_SOURCE_DIR}"
                     "${CMAKE_BINARY_DIR}/charts_beamformer" EXCLUDE_FROM_ALL)
    if(NOT TARGET charts::cuda)
        message(FATAL_ERROR "CHARTS dependency must provide charts::cuda; update beamform_project")
    endif()
    target_link_libraries(kotekan_cuda PUBLIC charts::cuda)
    # Explicit, host-only link probe: references real CUDA stage symbols without
    # device discovery or launches. Kept out of production/default build targets.
    add_executable(charts_beamformer_link_check EXCLUDE_FROM_ALL
                   "${KOTEKAN_SOURCE_DIR}/charts/beamform_adaptation/phase1_link_check.cpp")
    target_link_libraries(charts_beamformer_link_check PRIVATE charts::cuda)
endfunction()
