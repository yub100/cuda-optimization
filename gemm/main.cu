#include <iostream>

#include "./gemm_cpu.cuh"
#include "./gemm_v1.cuh"
#include "./gemm_v2.cuh"
#include "./gemm_v3.cuh"
#include "./gemm_v4.cuh"
#include "./gemm_v5.cuh"
#include "./gemm_v5_p.cuh"
#include "./gemm_v5_swizzle.cuh"
#include "./gemm_v6_1.cuh"
#include "./gemm_v6_2.cuh"
#include "../utils/utils.cuh"

void print1() {
    const int M = 1024;
    const int K = 1024;
    const int N = 1024;
    

    srand(42);
    // 分配主机内存
    float* hA = new float[M * K];
    float* hB = new float[K * N];      // B: K x N
    float* hC_cpu = new float[M * N];  // CPU 计算结果
    float* hC_gpu = new float[M * N];  // GPU 计算结果

    // initialize hA and hB
    random_matrix(hA, M * K);
    random_matrix(hB, K * N);

    // float time1 = 0.0;

    // CPU 矩阵乘法
    gemm_cpu(hA, hB, hC_cpu, M, K, N);

    gemm_sharedmemory_v1(hA, hB, hC_gpu, M, K, N);
    gemm_reg_v2(hA, hB, hC_gpu, M, K, N);
    gemm_reg_v3<128, 32, 128, 8, 8>(hA, hB, hC_gpu, M, K, N);
    gemm_reg_v4(hA, hB, hC_gpu, M, K, N);
    gemm_reg_v5(hA, hB, hC_gpu, M, K, N);
    gemm_reg_v5_p<128, 32, 128, 8, 8>(hA, hB, hC_gpu, M, K, N);
    gemm_reg_v5_swizzle(hA, hB, hC_gpu, M, K, N);
    gemm_reg_v6_1<128, 32, 128, 8, 8>(hA, hB, hC_gpu, M, K, N);
    gemm_reg_v6_2<128, 32, 128, 8, 8>(hA, hB, hC_gpu, M, K, N);
    

    for (int i = 0; i < M * N; i++) {
        if (fabs(hC_cpu[i] - hC_gpu[i]) > 1e-3f){
            std::cout << "false" << std::endl;
            break;
        }
    }

    for (int i = 0; i < 16; i++) {
        std::cout << hC_cpu[i] << " ";
    }
    std::cout << std::endl;

    for (int i = 0; i < 16; i++) {
        std::cout << hC_gpu[i] << " ";
    }
    std::cout << std::endl;
    
}

