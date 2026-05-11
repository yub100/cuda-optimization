cmake_minimum_required(VERSION 3.18)

# # 强制使用 GCC 12
# set(CMAKE_C_COMPILER /usr/bin/gcc-12 CACHE PATH "C compiler")
# set(CMAKE_CXX_COMPILER /usr/bin/g++-12 CACHE PATH "C++ compiler")
# set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-12 CACHE PATH "CUDA host compiler")

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

set(GEMM_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(UTILS_DIR ${GEMM_DIR}/../utils)

find_package(CUDAToolkit REQUIRED)

# NVTX
find_library(NVTOOLSEXT_LIB nvToolsExt
    PATHS
        ${CUDAToolkit_LIBRARY_DIR}
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
    ${GEMM_DIR}/gemm_cpu.cuh
    ${GEMM_DIR}/gemm_v1.cuh
    ${GEMM_DIR}/gemm_v2.cuh
    ${GEMM_DIR}/gemm_v3.cuh
    ${GEMM_DIR}/gemm_v4.cuh
    ${GEMM_DIR}/gemm_v5.cuh
    ${GEMM_DIR}/gemm_v5_p.cuh
    ${GEMM_DIR}/gemm_v5_swizzle.cuh
    ${GEMM_DIR}/gemm_v6_1.cuh
    ${GEMM_DIR}/gemm_v6_2.cuh
    ${GEMM_DIR}/gemm_ex.cuh
    ${GEMM_DIR}/gemm_cublas.cuh
    ${UTILS_DIR}/utils.cuh
)

function(configure_gemm_target target_name)
    target_include_directories(${target_name} PRIVATE
        ${GEMM_DIR}
        ${UTILS_DIR}
        ${CUDAToolkit_INCLUDE_DIRS}
    )

    target_compile_definitions(${target_name} PRIVATE
        USE_NVTX
    )

    target_compile_options(${target_name} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3>
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>
        $<$<COMPILE_LANGUAGE:CXX>:-O3>
    )

    target_link_libraries(${target_name} PRIVATE
        ${NVTOOLSEXT_LIB}
        CUDA::cublas
    )
endfunction()


function(add_ncu_target target_name executable_name output_name)
    add_custom_target(${target_name}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${NCU_DIR}
        COMMAND ncu
                --set full
                --target-processes all
                --force-overwrite
                --export ${NCU_DIR}/${output_name}
                $<TARGET_FILE:${executable_name}>
        DEPENDS ${executable_name}
        WORKING_DIRECTORY ${OUTPUT_DIR}
        COMMENT "Running Nsight Compute for ${executable_name}"
    )
endfunction()

function(add_nsys_target target_name executable_name output_name)
    add_custom_target(${target_name}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${NSYS_DIR}
        COMMAND nsys profile
                --trace=cuda,nvtx,osrt
                --cuda-memory-usage=true
                --force-overwrite=true
                --output=${NSYS_DIR}/${output_name}
                $<TARGET_FILE:${executable_name}>
        DEPENDS ${executable_name}
        WORKING_DIRECTORY ${OUTPUT_DIR}
        COMMENT "Running Nsight Systems for ${executable_name}"
    )
endfunction()

######################main
# add_executable(main
#     ${GEMM_DIR}/main.cu
#     ${GEMM_HEADERS}
# )
# configure_gemm_target(main)

# add_ncu_target(ncu3 main main)
# add_nsys_target(nsys3 main main)


######################benchmark1  
# add_executable(gemm_bench
#     ${GEMM_DIR}/benchmark1.cu
#     ${GEMM_HEADERS}
# )
# configure_gemm_target(gemm_bench)
# # ncu
# add_ncu_target(ncu gemm_bench gemm_bench)
# add_nsys_target(nsys gemm_bench gemm_bench)


######################benchmark2
add_executable(gemm_bench2
    ${GEMM_DIR}/benchmark2.cu
    ${GEMM_HEADERS}
)
configure_gemm_target(gemm_bench2)

add_ncu_target(ncu2 gemm_bench2 gemm_bench2)
add_nsys_target(nsys2 gemm_bench2 gemm_bench2)



