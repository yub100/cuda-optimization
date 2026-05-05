cmake_minimum_required(VERSION 3.18)

project(cuda_gemm_bench LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 按你的 GPU 改：
# T4=75, V100=70, A100=80, A10/A40=86, L20/L40/RTX40=89, H100=90
set(CMAKE_CUDA_ARCHITECTURES 86)

# 输出目录
set(OUTPUT_DIR ${CMAKE_BINARY_DIR}/output)
set(NCU_DIR ${OUTPUT_DIR}/ncu)
set(NSYS_DIR ${OUTPUT_DIR}/nsys)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR})

# NVTX
find_library(NVTOOLSEXT_LIB nvToolsExt
    PATHS
        /usr/local/cuda/lib64
        /usr/local/cuda-12.2/lib64
        /usr/local/cuda-12.3/lib64
        /usr/local/cuda-12.4/lib64
)

if(NOT NVTOOLSEXT_LIB)
    message(FATAL_ERROR "nvToolsExt not found, but your code uses NVTX")
endif()

# 手动列出依赖头文件，方便后面扩展
set(GEMM_HEADERS
    ${CMAKE_SOURCE_DIR}/gemm_cpu.cuh
    ${CMAKE_SOURCE_DIR}/gemm_v1.cuh
    ${CMAKE_SOURCE_DIR}/gemm_v2.cuh
    ${CMAKE_SOURCE_DIR}/gemm_v3.cuh
    ${CMAKE_SOURCE_DIR}/gemm_v4.cuh
    ${CMAKE_SOURCE_DIR}/../utils/utils.cuh
)

# 主程序
add_executable(gemm_bench
    ${CMAKE_SOURCE_DIR}/main.cu
    ${GEMM_HEADERS}
)

target_include_directories(gemm_bench PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/../utils
)

target_compile_definitions(gemm_bench PRIVATE
    USE_NVTX
)

target_compile_options(gemm_bench PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -lineinfo -Xptxas=-v>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

target_link_libraries(gemm_bench PRIVATE
    ${NVTOOLSEXT_LIB}
)

# Nsight Compute
add_custom_target(ncu
    COMMAND ${CMAKE_COMMAND} -E make_directory ${NCU_DIR}
    COMMAND ncu
            --set full
            --target-processes all
            --force-overwrite
            --export ${NCU_DIR}/gemm_bench
            $<TARGET_FILE:gemm_bench>
    DEPENDS gemm_bench
    WORKING_DIRECTORY ${OUTPUT_DIR}
    COMMENT "Running Nsight Compute"
)

# Nsight Systems
add_custom_target(nsys
    COMMAND ${CMAKE_COMMAND} -E make_directory ${NSYS_DIR}
    COMMAND nsys profile
            --trace=cuda,nvtx,osrt
            --cuda-memory-usage=true
            --force-overwrite=true
            --output=${NSYS_DIR}/gemm_bench
            $<TARGET_FILE:gemm_bench>
    DEPENDS gemm_bench
    WORKING_DIRECTORY ${OUTPUT_DIR}
    COMMENT "Running Nsight Systems"
)