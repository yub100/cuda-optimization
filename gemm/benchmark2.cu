#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

#include <cuda_runtime.h>

#include "./gemm_cublas.cuh"
#include "./gemm_v1.cuh"
#include "./gemm_v2.cuh"
#include "./gemm_v3.cuh"
#include "./gemm_v4.cuh"
#include "./gemm_v5.cuh"
#include "./gemm_v5_p.cuh"
#include "./gemm_v5_swizzle.cuh"
#include "./gemm_v6_1.cuh"
#include "./gemm_v6_2.cuh"
#include "./gemm_ex.cuh"

namespace {

constexpr int kFixedK = 1024;
constexpr int kWarmupRuns = 3;
constexpr int kBenchRuns = 10;

struct GemmSharedMemoryV1;
struct GemmRegV2;
struct GemmRegV3;
struct GemmRegV4;
struct GemmRegV5;
struct GemmRegV5P;
struct GemmRegV5Swizzle;
struct GemmRegV61;
struct GemmRegV62;
struct GemmCublas;
struct GemmEx;

// Change only these lines when you want to benchmark another version.
using BenchGemm = GemmCublas;
constexpr const char* kBenchName = "cutblas";
constexpr int kBM = 128;
constexpr int kBK = 16;
constexpr int kBN = 128;
constexpr int kTM = 8;
constexpr int kTN = 8;

void check_cuda(cudaError_t status, const char* expr) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s: %s\n", expr, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

void check_cublas(cublasStatus_t status, const char* expr) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "cuBLAS error at %s: %s\n", expr, cublas_status_to_string(status));
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_CUDA(expr) check_cuda((expr), #expr)
#define CHECK_CUBLAS(expr) check_cublas((expr), #expr)

cublasHandle_t g_cublas_handle = nullptr;

double gflops(int m, int n, int k, double seconds) {
    if (seconds <= 0.0) {
        return 0.0;
    }
    return 2.0 * static_cast<double>(m) * n * k / seconds / 1.0e9;
}

template <int BM, int BK, int BN, int TM, int TN>
bool exact_tile_supported(int m, int n, int k) {
    return m % BM == 0 && n % BN == 0 && k % BK == 0;
}


struct GemmEx {
    template <int BM, int BK, int BN, int TM, int TN>
    static bool supported(int m, int n, int k) {
        return BK == 8 && exact_tile_supported<BM, BK, BN, TM, TN>(m, n, k);
    }

    template <int BM, int BK, int BN, int TM, int TN>
    static void launch(float* dA, float* dB, float* dC, int m, int k, int n) {
        dim3 block(BN / TN, BM / TM, 1);
        dim3 grid(n / BN, m / BM, 1);
        gemm_ex<BM, BK, BN, TM, TN><<<grid, block>>>(dA, dB, dC, m, n, k);
    }
};

struct GemmSharedMemoryV1 {
    template <int BM, int BK, int BN, int TM, int TN>
    static bool supported(int, int, int) {
        (void)BM;
        (void)BK;
        (void)BN;
        (void)TM;
        (void)TN;
        return true;
    }

    template <int BM, int BK, int BN, int TM, int TN>
    static void launch(float* dA, float* dB, float* dC, int m, int k, int n) {
        (void)BM;
        (void)BK;
        (void)BN;
        (void)TM;
        (void)TN;
        constexpr int BLOCK_DIM = 32;
        dim3 block(BLOCK_DIM, BLOCK_DIM, 1);
        dim3 grid((n + BLOCK_DIM - 1) / BLOCK_DIM, (m + BLOCK_DIM - 1) / BLOCK_DIM, 1);
        gemm_shared_memory_kernel_v1<BLOCK_DIM><<<grid, block>>>(dA, dB, dC, m, k, n);
    }
};

struct GemmRegV2 {
    template <int BM, int BK, int BN, int TM, int TN>
    static bool supported(int, int, int) {
        return true;
    }

    template <int BM, int BK, int BN, int TM, int TN>
    static void launch(float* dA, float* dB, float* dC, int m, int k, int n) {
        dim3 block(BN / TN, BM / TM, 1);
        dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM, 1);
        gemm_reg_kernel_v2<BM, BK, BN, TM, TN><<<grid, block>>>(dA, dB, dC, m, k, n);
    }
};

struct GemmRegV3 {
    template <int BM, int BK, int BN, int TM, int TN>
    static bool supported(int, int, int) {
        return true;
    }

    template <int BM, int BK, int BN, int TM, int TN>
    static void launch(float* dA, float* dB, float* dC, int m, int k, int n) {
        dim3 block(BN / TN, BM / TM, 1);
        dim3 grid((n + BN - 1) / BN, (m + BM - 1) / BM, 1);
        gemm_reg_kernel_v3<BM, BK, BN, TM, TN><<<grid, block>>>(dA, dB, dC, m, k, n);
    }
};

struct GemmRegV4 {
    template <int BM, int BK, int BN, int TM, int TN>
    static bool supported(int m, int n, int k) {
        return BK == 8 && exact_tile_supported<BM, BK, BN, TM, TN>(m, n, k);
    }

    template <int BM, int BK, int BN, int TM, int TN>
    static void launch(float* dA, float* dB, float* dC, int m, int k, int n) {
        dim3 block(BN / TN, BM / TM, 1);
        dim3 grid(n / BN, m / BM, 1);
        gemm_reg_kernel_v4<BM, BK, BN, TM, TN><<<grid, block>>>(dA, dB, dC, m, k, n);
    }
};

struct GemmRegV5 {
    template <int BM, int BK, int BN, int TM, int TN>
    static bool supported(int m, int n, int k) {
        return BK == 8 && exact_tile_supported<BM, BK, BN, TM, TN>(m, n, k);
    }

    template <int BM, int BK, int BN, int TM, int TN>
    static void launch(float* dA, float* dB, float* dC, int m, int k, int n) {
        dim3 block(BN / TN, BM / TM, 1);
        dim3 grid(n / BN, m / BM, 1);
        gemm_reg_kernel_v5<BM, BK, BN, TM, TN><<<grid, block>>>(dA, dB, dC, m, k, n);
    }
};

struct GemmRegV5P {
    template <int BM, int BK, int BN, int TM, int TN>
    static bool supported(int m, int n, int k) {
        return exact_tile_supported<BM, BK, BN, TM, TN>(m, n, k);
    }

    template <int BM, int BK, int BN, int TM, int TN>
    static void launch(float* dA, float* dB, float* dC, int m, int k, int n) {
        dim3 block(BN / TN, BM / TM, 1);
        dim3 grid(n / BN, m / BM, 1);
        gemm_reg_kernel_v5_p<BM, BK, BN, TM, TN><<<grid, block>>>(dA, dB, dC, m, k, n);
    }
};

struct GemmRegV5Swizzle {
    template <int BM, int BK, int BN, int TM, int TN>
    static bool supported(int m, int n, int k) {
        return BK == 8 && exact_tile_supported<BM, BK, BN, TM, TN>(m, n, k);
    }

    template <int BM, int BK, int BN, int TM, int TN>
    static void launch(float* dA, float* dB, float* dC, int m, int k, int n) {
        dim3 block(BN / TN, BM / TM, 1);
        dim3 grid(n / BN, m / BM, 1);
        gemm_reg_kernel_v5_swizzle<BM, BK, BN, TM, TN><<<grid, block>>>(dA, dB, dC, m, k, n);
    }
};

struct GemmRegV61 {
    template <int BM, int BK, int BN, int TM, int TN>
    static bool supported(int m, int n, int k) {
        return exact_tile_supported<BM, BK, BN, TM, TN>(m, n, k);
    }

    template <int BM, int BK, int BN, int TM, int TN>
    static void launch(float* dA, float* dB, float* dC, int m, int k, int n) {
        dim3 block(BN / TN, BM / TM, 1);
        dim3 grid(n / BN, m / BM, 1);
        gemm_reg_kernel_v6_1<BM, BK, BN, TM, TN><<<grid, block>>>(dA, dB, dC, m, k, n);
    }
};

struct GemmRegV62 {
    template <int BM, int BK, int BN, int TM, int TN>
    static bool supported(int m, int n, int k) {
        return exact_tile_supported<BM, BK, BN, TM, TN>(m, n, k);
    }

    template <int BM, int BK, int BN, int TM, int TN>
    static void launch(float* dA, float* dB, float* dC, int m, int k, int n) {
        dim3 block(BN / TN, BM / TM, 1);
        dim3 grid(n / BN, m / BM, 1);
        gemm_reg_kernel_v6_2<BM, BK, BN, TM, TN><<<grid, block>>>(dA, dB, dC, m, k, n);
    }
};

struct GemmCublas {
    template <int BM, int BK, int BN, int TM, int TN>
    static bool supported(int, int, int) {
        (void)BM;
        (void)BK;
        (void)BN;
        (void)TM;
        (void)TN;
        return true;
    }

    template <int BM, int BK, int BN, int TM, int TN>
    static void launch(float* dA, float* dB, float* dC, int m, int k, int n) {
        (void)BM;
        (void)BK;
        (void)BN;
        (void)TM;
        (void)TN;
        CHECK_CUBLAS(gemm_cublas_sgemm(g_cublas_handle, dA, dB, dC, m, k, n));
    }
};

template <typename Gemm, int BM, int BK, int BN, int TM, int TN>
void run_one_shape(const char* name, int m, int n, int k) {
    if (!Gemm::template supported<BM, BK, BN, TM, TN>(m, n, k)) {
        std::printf("M N K = %6d %6d %6d, %s skipped: unsupported tile shape BM=%d BK=%d BN=%d TM=%d TN=%d\n",
                    m,
                    n,
                    k,
                    name,
                    BM,
                    BK,
                    BN,
                    TM,
                    TN);
        return;
    }

    float* dA = nullptr;
    float* dB = nullptr;
    float* dC = nullptr;

    const std::size_t bytesA = static_cast<std::size_t>(m) * k * sizeof(float);
    const std::size_t bytesB = static_cast<std::size_t>(k) * n * sizeof(float);
    const std::size_t bytesC = static_cast<std::size_t>(m) * n * sizeof(float);

    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&dA), bytesA));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&dB), bytesB));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&dC), bytesC));

    CHECK_CUDA(cudaMemset(dA, 0, bytesA));
    CHECK_CUDA(cudaMemset(dB, 0, bytesB));
    CHECK_CUDA(cudaMemset(dC, 0, bytesC));

    for (int i = 0; i < kWarmupRuns; i++) {
        Gemm::template launch<BM, BK, BN, TM, TN>(dA, dB, dC, m, k, n);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start;
    cudaEvent_t stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    std::vector<double> times_s;
    times_s.reserve(kBenchRuns);

    for (int i = 0; i < kBenchRuns; i++) {
        CHECK_CUDA(cudaEventRecord(start));
        Gemm::template launch<BM, BK, BN, TM, TN>(dA, dB, dC, m, k, n);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float milliseconds = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
        times_s.push_back(static_cast<double>(milliseconds) / 1000.0);
    }

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    const double avg_s = std::accumulate(times_s.begin(), times_s.end(), 0.0) / times_s.size();
    const double avg_gflops = gflops(m, n, k, avg_s);

    std::printf("M N K = %6d %6d %6d, Time = %12.8f s, AVG Performance = %11.4f Gflops\n",
                m,
                n,
                k,
                avg_s,
                avg_gflops);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));
}

}  // namespace

int main() {
    const int sizes[] = {
        128,
        256,
        384,
        512,
        768,
        1024,
        1536,
        2048,
        3072,
        4096,
        6144,
        8192,
        12288,
        16384,
    };

    CHECK_CUBLAS(cublasCreate(&g_cublas_handle));
    CHECK_CUBLAS(cublasSetMathMode(g_cublas_handle, CUBLAS_PEDANTIC_MATH));

    printf("Kernel = %s\n", kBenchName);
    for (int size : sizes) {
        run_one_shape<BenchGemm, kBM, kBK, kBN, kTM, kTN>(kBenchName, size, size, kFixedK);
    }

    CHECK_CUBLAS(cublasDestroy(g_cublas_handle));
    return 0;
}
