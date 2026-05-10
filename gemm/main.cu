#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

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
#include "./gemm_cutlass.cuh"
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


using GemmFn = float (*)(float*, float*, float*, int, int, int);

bool check_result(const float* expected, const float* actual, int size, float eps, float* max_abs_diff) {
    float max_diff = 0.0f;
    bool ok = true;

    for (int i = 0; i < size; i++) {
        float diff = std::fabs(expected[i] - actual[i]);
        max_diff = std::max(max_diff, diff);
        if (diff > eps) {
            ok = false;
        }
    }

    if (max_abs_diff) {
        *max_abs_diff = max_diff;
    }

    return ok;
}

double calc_gflops(int M, int K, int N, float avg_ms) {
    if (avg_ms <= 0.0f) {
        return 0.0;
    }

    double flops = 2.0 * static_cast<double>(M) * K * N;
    return flops / (avg_ms / 1000.0) / 1e9;
}

template <int BM, int BK, int BN, int TM, int TN>
void run_case(std::ofstream* csv,
              const std::string& version,
              GemmFn gemm_fn,
              float* hA,
              float* hB,
              float* hC_cpu,
              float* hC_gpu,
              int M,
              int K,
              int N) {
    std::fill(hC_gpu, hC_gpu + M * N, 0.0f);

    float avg_ms = gemm_fn(hA, hB, hC_gpu, M, K, N);
    double gflops = calc_gflops(M, K, N, avg_ms);

    float max_abs_diff = 0.0f;
    bool ok = check_result(hC_cpu, hC_gpu, M * N, 1e-3f, &max_abs_diff);

    if (write_csv && csv && csv->is_open()) {
        *csv << version << ','
             << BM << ','
             << BK << ','
             << BN << ','
             << TM << ','
             << TN << ','
             << M << ','
             << K << ','
             << N << ','
             << avg_ms << ','
             << gflops << ','
             << (ok ? "true" : "false") << ','
             << max_abs_diff << '\n';
    }

    if (!ok) {
        std::cout << "Validation failed: " << version
                  << " BK=" << BK
                  << " max_abs_diff=" << max_abs_diff
                  << std::endl;
    }
}

void bench() {
    const int M = 1024;
    const int K = 1024;
    const int N = 1024;

    srand(42);

    float* hA = new float[M * K];
    float* hB = new float[K * N];
    float* hC_cpu = new float[M * N];
    float* hC_gpu = new float[M * N];

    random_matrix(hA, M * K);
    random_matrix(hB, K * N);

    float cpu_ms = gemm_cpu(hA, hB, hC_cpu, M, K, N);
    double cpu_gflops = calc_gflops(M, K, N, cpu_ms);

    std::ofstream csv;
    if (write_csv) {
        csv.open("gemm_bench.csv");
        csv << "version,BM,BK,BN,TM,TN,M,K,N,avg_ms,gflops,correct,max_abs_diff\n";
        csv << "gemm_cpu,0,0,0,0,0,"
            << M << ','
            << K << ','
            << N << ','
            << cpu_ms << ','
            << cpu_gflops << ",true,0\n";
    }

    if (print_on_destroy) {
        std::cout << "BENCH_RUNS=" << BENCH_RUNS
                  << ", M=" << M
                  << ", K=" << K
                  << ", N=" << N
                  << std::endl;
        std::cout << "gemm_cpu:\t\t" << cpu_ms << "ms" << std::endl;
    }

    run_case<32, 0, 32, 0, 0>(&csv, "gemm_sharedmemory_v1", gemm_sharedmemory_v1, hA, hB, hC_cpu, hC_gpu, M, K, N);

    run_case<128, 8, 128, 8, 8>(&csv, "gemm_reg_v2", gemm_reg_v2<128, 8, 128, 8, 8>, hA, hB, hC_cpu, hC_gpu, M, K, N);
    run_case<128, 16, 128, 8, 8>(&csv, "gemm_reg_v2", gemm_reg_v2<128, 16, 128, 8, 8>, hA, hB, hC_cpu, hC_gpu, M, K, N);
    run_case<128, 32, 128, 8, 8>(&csv, "gemm_reg_v2", gemm_reg_v2<128, 32, 128, 8, 8>, hA, hB, hC_cpu, hC_gpu, M, K, N);

    run_case<128, 8, 128, 8, 8>(&csv, "gemm_reg_v3", gemm_reg_v3<128, 8, 128, 8, 8>, hA, hB, hC_cpu, hC_gpu, M, K, N);
    run_case<128, 16, 128, 8, 8>(&csv, "gemm_reg_v3", gemm_reg_v3<128, 16, 128, 8, 8>, hA, hB, hC_cpu, hC_gpu, M, K, N);
    run_case<128, 32, 128, 8, 8>(&csv, "gemm_reg_v3", gemm_reg_v3<128, 32, 128, 8, 8>, hA, hB, hC_cpu, hC_gpu, M, K, N);

    // v4/v5/v5_swizzle use a BK=8-specific global-to-shared loading pattern.
    run_case<128, 8, 128, 8, 8>(&csv, "gemm_reg_v4", gemm_reg_v4<128, 8, 128, 8, 8>, hA, hB, hC_cpu, hC_gpu, M, K, N);
    run_case<128, 8, 128, 8, 8>(&csv, "gemm_reg_v5", gemm_reg_v5<128, 8, 128, 8, 8>, hA, hB, hC_cpu, hC_gpu, M, K, N);

    run_case<128, 8, 128, 8, 8>(&csv, "gemm_reg_v5_p", gemm_reg_v5_p<128, 8, 128, 8, 8>, hA, hB, hC_cpu, hC_gpu, M, K, N);
    run_case<128, 16, 128, 8, 8>(&csv, "gemm_reg_v5_p", gemm_reg_v5_p<128, 16, 128, 8, 8>, hA, hB, hC_cpu, hC_gpu, M, K, N);
    run_case<128, 32, 128, 8, 8>(&csv, "gemm_reg_v5_p", gemm_reg_v5_p<128, 32, 128, 8, 8>, hA, hB, hC_cpu, hC_gpu, M, K, N);

    run_case<128, 8, 128, 8, 8>(&csv, "gemm_reg_v5_swizzle", gemm_reg_v5_swizzle<128, 8, 128, 8, 8>, hA, hB, hC_cpu, hC_gpu, M, K, N);

    run_case<128, 8, 128, 8, 8>(&csv, "gemm_reg_v6_1", gemm_reg_v6_1<128, 8, 128, 8, 8>, hA, hB, hC_cpu, hC_gpu, M, K, N);
    run_case<128, 16, 128, 8, 8>(&csv, "gemm_reg_v6_1", gemm_reg_v6_1<128, 16, 128, 8, 8>, hA, hB, hC_cpu, hC_gpu, M, K, N);
    run_case<128, 32, 128, 8, 8>(&csv, "gemm_reg_v6_1", gemm_reg_v6_1<128, 32, 128, 8, 8>, hA, hB, hC_cpu, hC_gpu, M, K, N);

    run_case<128, 8, 128, 8, 8>(&csv, "gemm_reg_v6_2", gemm_reg_v6_2<128, 8, 128, 8, 8>, hA, hB, hC_cpu, hC_gpu, M, K, N);
    run_case<128, 16, 128, 8, 8>(&csv, "gemm_reg_v6_2", gemm_reg_v6_2<128, 16, 128, 8, 8>, hA, hB, hC_cpu, hC_gpu, M, K, N);
    run_case<128, 32, 128, 8, 8>(&csv, "gemm_reg_v6_2", gemm_reg_v6_2<128, 32, 128, 8, 8>, hA, hB, hC_cpu, hC_gpu, M, K, N);

#if GEMM_HAS_CUTLASS
    run_case<128, 8, 128, 0, 0>(&csv, "gemm_cutlass", gemm_cutlass<128, 8, 128, 8, 8>, hA, hB, hC_cpu, hC_gpu, M, K, N);
#else
    if (print_on_destroy) {
        std::cout << "gemm_cutlass skipped: CUTLASS headers were not found." << std::endl;
    }
#endif

    if (write_csv) {
        csv.close();
    }

    delete[] hA;
    delete[] hB;
    delete[] hC_cpu;
    delete[] hC_gpu;

    if (print_on_destroy && write_csv) {
        std::cout << "Wrote gemm_bench.csv" << std::endl;
    }
}

int main() {
    bench();
    return 0;
}
