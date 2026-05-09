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
    gemm_reg_v3(hA, hB, hC_gpu, M, K, N);
    gemm_reg_v4(hA, hB, hC_gpu, M, K, N);
    gemm_reg_v5(hA, hB, hC_gpu, M, K, N);
    gemm_reg_v5_p(hA, hB, hC_gpu, M, K, N);
    gemm_reg_v5_swizzle(hA, hB, hC_gpu, M, K, N);
    gemm_reg_v6_1(hA, hB, hC_gpu, M, K, N);
    gemm_reg_v6_2(hA, hB, hC_gpu, M, K, N);
    

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


// void print2() {
//     srand(42);

//     std::vector<int> sizes = {
//         128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096, 5120, 6144, 7168, 8192
//     };

//     std::ofstream fout("gemm_time.csv");
//     fout << "M,K,N,cpu_ms,shared_ms,reg_ms,cpu_gflops,shared_gflops,reg_gflops\n";

//     for (int size : sizes) {
//         int M = size;
//         int K = size;
//         int N = size;

//         std::cout << "Running M=K=N=" << size << std::endl;

//         float* hA = new float[M * K];
//         float* hB = new float[K * N];
//         float* hC_cpu = new float[M * N];
//         float* hC_shared = new float[M * N];
//         float* hC_reg = new float[M * N];

//         random_matrix(hA, M * K);
//         random_matrix(hB, K * N);

//         float cpu_ms = -1.0f;

//         if (size <= 1024) {
//             cpu_ms = gemm_cpu(hA, hB, hC_cpu, M, K, N);
//         }

//         float shared_ms = gemm_sharedmemory(hA, hB, hC_shared, M, K, N);
//         float reg_ms = gemm_reg_v4(hA, hB, hC_reg, M, K, N);

//         // 正确性检查：只在 CPU 跑过时检查
//         if (cpu_ms > 0.0f) {
//             bool shared_ok = true;
//             bool reg_ok = true;

//             for (int i = 0; i < M * N; i++) {
//                 if (fabs(hC_cpu[i] - hC_shared[i]) > 1e-3f) {
//                     shared_ok = false;
//                     std::cout << "shared false at " << i
//                               << ", cpu=" << hC_cpu[i]
//                               << ", shared=" << hC_shared[i]
//                               << std::endl;
//                     break;
//                 }
//             }

//             for (int i = 0; i < M * N; i++) {
//                 if (fabs(hC_cpu[i] - hC_reg[i]) > 1e-3f) {
//                     reg_ok = false;
//                     std::cout << "reg false at " << i
//                               << ", cpu=" << hC_cpu[i]
//                               << ", reg=" << hC_reg[i]
//                               << std::endl;
//                     break;
//                 }
//             }

//             std::cout << "shared ok: " << shared_ok
//                       << ", reg ok: " << reg_ok
//                       << std::endl;
//         }

//         double flops = 2.0 * M * N * K;

//         double cpu_gflops = cpu_ms > 0.0f
//             ? flops / (cpu_ms / 1000.0) / 1e9
//             : -1.0;

//         double shared_gflops = flops / (shared_ms / 1000.0) / 1e9;
//         double reg_gflops = flops / (reg_ms / 1000.0) / 1e9;

//         std::cout << "cpu_ms = " << cpu_ms
//                   << ", shared_ms = " << shared_ms
//                   << ", reg_ms = " << reg_ms
//                   << std::endl;

//         std::cout << "cpu_gflops = " << cpu_gflops
//                   << ", shared_gflops = " << shared_gflops
//                   << ", reg_gflops = " << reg_gflops
//                   << std::endl;

//         fout << M << ","
//              << K << ","
//              << N << ","
//              << cpu_ms << ","
//              << shared_ms << ","
//              << reg_ms << ","
//              << cpu_gflops << ","
//              << shared_gflops << ","
//              << reg_gflops << "\n";

//         delete[] hA;
//         delete[] hB;
//         delete[] hC_cpu;
//         delete[] hC_shared;
//         delete[] hC_reg;
//     }

//     fout.close();

// }

int main() {
    print1();
}

