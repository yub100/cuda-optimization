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
#include "./gemm_cublas.cuh"
#include "../utils/utils.cuh"

struct GemmSharedMemoryV1 {
    template <int BM, int BK, int BN, int TM, int TN>
    static float run(float* hA, float* hB, float* hC, int M, int K, int N) {
        (void)BM;
        (void)BK;
        (void)BN;
        (void)TM;
        (void)TN;
        return gemm_sharedmemory_v1(hA, hB, hC, M, K, N);
    }
};

struct GemmRegV2 {
    template <int BM, int BK, int BN, int TM, int TN>
    static float run(float* hA, float* hB, float* hC, int M, int K, int N) {
        return gemm_reg_v2<BM, BK, BN, TM, TN>(hA, hB, hC, M, K, N);
    }
};

struct GemmRegV3 {
    template <int BM, int BK, int BN, int TM, int TN>
    static float run(float* hA, float* hB, float* hC, int M, int K, int N) {
        return gemm_reg_v3<BM, BK, BN, TM, TN>(hA, hB, hC, M, K, N);
    }
};

struct GemmRegV4 {
    template <int BM, int BK, int BN, int TM, int TN>
    static float run(float* hA, float* hB, float* hC, int M, int K, int N) {
        return gemm_reg_v4<BM, BK, BN, TM, TN>(hA, hB, hC, M, K, N);
    }
};

struct GemmRegV5 {
    template <int BM, int BK, int BN, int TM, int TN>
    static float run(float* hA, float* hB, float* hC, int M, int K, int N) {
        return gemm_reg_v5<BM, BK, BN, TM, TN>(hA, hB, hC, M, K, N);
    }
};

struct GemmRegV5P {
    template <int BM, int BK, int BN, int TM, int TN>
    static float run(float* hA, float* hB, float* hC, int M, int K, int N) {
        return gemm_reg_v5_p<BM, BK, BN, TM, TN>(hA, hB, hC, M, K, N);
    }
};

struct GemmRegV5Swizzle {
    template <int BM, int BK, int BN, int TM, int TN>
    static float run(float* hA, float* hB, float* hC, int M, int K, int N) {
        return gemm_reg_v5_swizzle<BM, BK, BN, TM, TN>(hA, hB, hC, M, K, N);
    }
};

struct GemmRegV61 {
    template <int BM, int BK, int BN, int TM, int TN>
    static float run(float* hA, float* hB, float* hC, int M, int K, int N) {
        return gemm_reg_v6_1<BM, BK, BN, TM, TN>(hA, hB, hC, M, K, N);
    }
};

struct GemmRegV62 {
    template <int BM, int BK, int BN, int TM, int TN>
    static float run(float* hA, float* hB, float* hC, int M, int K, int N) {
        return gemm_reg_v6_2<BM, BK, BN, TM, TN>(hA, hB, hC, M, K, N);
    }
};

struct GemmCublas {
    template <int BM, int BK, int BN, int TM, int TN>
    static float run(float* hA, float* hB, float* hC, int M, int K, int N) {
        return gemm_cublas<BM, BK, BN, TM, TN>(hA, hB, hC, M, K, N);
    }
};

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

template <typename Gemm, int BM, int BK, int BN, int TM, int TN>
void run_case(std::ofstream* csv,
              const std::string& version,
              float* hA,
              float* hB,
              float* hC_cpu,
              float* hC_gpu,
              int M,
              int K,
              int N) {
    std::fill(hC_gpu, hC_gpu + M * N, 0.0f);

    float avg_ms = Gemm::template run<BM, BK, BN, TM, TN>(hA, hB, hC_gpu, M, K, N);
    double gflops = calc_gflops(M, K, N, avg_ms);

    float max_abs_diff = 0.0f;
    // bool ok = check_result(hC_cpu, hC_gpu, M * N, 1e-3f, &max_abs_diff);

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
             << gflops << ',' << '\n';
            //  << (ok ? "true" : "false") << ','
            //  << max_abs_diff << '\n';
    }

    // if (!ok) {
    //     std::cout << "Validation failed: " << version
    //               << " BK=" << BK
    //               << " max_abs_diff=" << max_abs_diff
    //               << std::endl;
    // }
}

void bench() {
    const int M = 16384;
    const int K = 1024;
    const int N = 16384;

    srand(42);

    float* hA = new float[M * K];
    float* hB = new float[K * N];
    float* hC_cpu = new float[M * N];
    float* hC_gpu = new float[M * N];

    // memset(hA, 0, M * K * sizeof(float));
    // memset(hB, 0, K * N * sizeof(float));

    random_matrix(hA, M * K);
    random_matrix(hB, K * N);

    // float cpu_ms = gemm_cpu(hA, hB, hC_cpu, M, K, N);
    // double cpu_gflops = calc_gflops(M, K, N, cpu_ms);

    std::ofstream csv;
    if (write_csv) {
        csv.open("gemm_bench.csv");
        csv << "version,BM,BK,BN,TM,TN,M,K,N,avg_ms,gflops\n";
        // csv << "gemm_cpu,0,0,0,0,0,"
        //     << M << ','
        //     << K << ','
        //     << N << ','
        //     << cpu_ms << ','
        //     << cpu_gflops << ",true,0\n";
    }

    // if (print_on_destroy) {
    //     std::cout << "BENCH_RUNS=" << BENCH_RUNS
    //               << ", M=" << M
    //               << ", K=" << K
    //               << ", N=" << N
    //               << std::endl;
    //     std::cout << "gemm_cpu:\t\t" << cpu_ms << "ms" << std::endl;
    // }

    run_case<GemmSharedMemoryV1, 32, 0, 32, 0, 0>(&csv, "gemm_sharedmemory_v1", hA, hB, hC_cpu, hC_gpu, M, K, N);

    run_case<GemmRegV2, 128, 8, 128, 8, 8>(&csv, "gemm_reg_v2", hA, hB, hC_cpu, hC_gpu, M, K, N);
    run_case<GemmRegV2, 128, 16, 128, 8, 8>(&csv, "gemm_reg_v2", hA, hB, hC_cpu, hC_gpu, M, K, N);
    run_case<GemmRegV2, 128, 32, 128, 8, 8>(&csv, "gemm_reg_v2", hA, hB, hC_cpu, hC_gpu, M, K, N);

    run_case<GemmRegV3, 128, 8, 128, 8, 8>(&csv, "gemm_reg_v3", hA, hB, hC_cpu, hC_gpu, M, K, N);
    run_case<GemmRegV3, 128, 16, 128, 8, 8>(&csv, "gemm_reg_v3", hA, hB, hC_cpu, hC_gpu, M, K, N);
    run_case<GemmRegV3, 128, 32, 128, 8, 8>(&csv, "gemm_reg_v3", hA, hB, hC_cpu, hC_gpu, M, K, N);

    // v4/v5/v5_swizzle use a BK=8-specific global-to-shared loading pattern.
    run_case<GemmRegV4, 128, 8, 128, 8, 8>(&csv, "gemm_reg_v4", hA, hB, hC_cpu, hC_gpu, M, K, N);
    run_case<GemmRegV5, 128, 8, 128, 8, 8>(&csv, "gemm_reg_v5", hA, hB, hC_cpu, hC_gpu, M, K, N);

    run_case<GemmRegV5P, 128, 8, 128, 8, 8>(&csv, "gemm_reg_v5_p", hA, hB, hC_cpu, hC_gpu, M, K, N);
    run_case<GemmRegV5P, 128, 16, 128, 8, 8>(&csv, "gemm_reg_v5_p", hA, hB, hC_cpu, hC_gpu, M, K, N);
    run_case<GemmRegV5P, 128, 32, 128, 8, 8>(&csv, "gemm_reg_v5_p", hA, hB, hC_cpu, hC_gpu, M, K, N);

    run_case<GemmRegV5Swizzle, 128, 8, 128, 8, 8>(&csv, "gemm_reg_v5_swizzle", hA, hB, hC_cpu, hC_gpu, M, K, N);

    run_case<GemmRegV61, 128, 8, 128, 8, 8>(&csv, "gemm_reg_v6_1", hA, hB, hC_cpu, hC_gpu, M, K, N);
    run_case<GemmRegV61, 128, 16, 128, 8, 8>(&csv, "gemm_reg_v6_1", hA, hB, hC_cpu, hC_gpu, M, K, N);
    run_case<GemmRegV61, 128, 32, 128, 8, 8>(&csv, "gemm_reg_v6_1", hA, hB, hC_cpu, hC_gpu, M, K, N);

    run_case<GemmRegV62, 128, 8, 128, 8, 8>(&csv, "gemm_reg_v6_2", hA, hB, hC_cpu, hC_gpu, M, K, N);
    run_case<GemmRegV62, 128, 16, 128, 8, 8>(&csv, "gemm_reg_v6_2", hA, hB, hC_cpu, hC_gpu, M, K, N);
    run_case<GemmRegV62, 128, 32, 128, 8, 8>(&csv, "gemm_reg_v6_2", hA, hB, hC_cpu, hC_gpu, M, K, N);

    run_case<GemmCublas, 0, 0, 0, 0, 0>(&csv, "gemm_cublas", hA, hB, hC_cpu, hC_gpu, M, K, N);

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
