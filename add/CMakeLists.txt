cmake_minimum_required(VERSION 3.18)

# 声明项目名称，并显式指定支持 CUDA 和 CXX (C++) 语言
project(CudaExercise LANGUAGES CUDA CXX)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置 CUDA 编译选项
# 如果你想让程序在特定显卡上跑得最快，可以指定架构
# 例如：RTX 30系列是 86，RTX 40系列是 89
# 如果不确定，可以设为 native（自动探测当前机器显卡）

# 方法1：使用 CUDA_SELECT_NVCC_ARCH_FLAGS（推荐）
find_package(CUDA REQUIRED)
cuda_select_nvcc_arch_flags(ARCH_FLAGS Auto)  # 自动检测
# 或者手动指定，例如：cuda_select_nvcc_arch_flags(ARCH_FLAGS 70 75 80 86)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${ARCH_FLAGS})

# 查找 CUDA 库（通常 CMake 自动处理，但显式查找更稳健）
find_package(CUDAToolkit REQUIRED)

# 添加可执行文件
add_executable(cuda_add add.cu)

# 如果你的代码里用到了 CUDA 运行时库的函数（如 cudaMalloc）
# 现代 CMake 会自动链接，但也可以显式链接：
target_link_libraries(cuda_add PRIVATE CUDA::cudart)