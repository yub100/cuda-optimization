#pragma once

#include <cublas_v2.h>
#include <iostream>
#include <nvToolsExt.h>
#include "../utils/utils.cuh"

inline const char* cublas_status_to_string(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";
        default:
            return "CUBLAS_STATUS_UNKNOWN";
    }
}

template <int BM = 128, int BK = 8, int BN = 128, int TM = 8, int TN = 8>
float gemm_cublas(float *hA, float *hB, float *hC, int M, int K, int N) {
    (void)BM;
    (void)BK;
    (void)BN;
    (void)TM;
    (void)TN;

    float *dA, *dB, *dC;

    nvtxRangePush("gemm_cublas_start_up_malloc");
    cudaMalloc((void **)&dA, M * K * sizeof(float));
    cudaMalloc((void **)&dB, K * N * sizeof(float));
    cudaMalloc((void **)&dC, M * N * sizeof(float));
    nvtxRangePop();

    cudaMemcpy(dA, hA, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(dC, 0, M * N * sizeof(float));

    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "cublasCreate error: " << cublas_status_to_string(status) << std::endl;
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
        return -1.0f;
    }

    // Keep full FP32 behavior so the existing CPU validation threshold remains meaningful.
    status = cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "cublasSetMathMode error: " << cublas_status_to_string(status) << std::endl;
        cublasDestroy(handle);
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
        return -1.0f;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;

    auto run_gemm = [&]() {
        // cuBLAS uses column-major. Row-major C=A*B is equivalent to
        // column-major C^T = B^T * A^T with output shape N x M.
        return cublasSgemm(handle,
                           CUBLAS_OP_N,
                           CUBLAS_OP_N,
                           N,
                           M,
                           K,
                           &alpha,
                           dB,
                           N,
                           dA,
                           K,
                           &beta,
                           dC,
                           N);
    };

    nvtxRangePush("gemm_cublas_kernel");
    status = run_gemm();
    cudaDeviceSynchronize();
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "cuBLAS warmup error: " << cublas_status_to_string(status) << std::endl;
        nvtxRangePop();
        cublasDestroy(handle);
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
        return -1.0f;
    }

    float avg_ms = 0.0f;
    {
        CudaTimer timer("gemm_cublas", BENCH_RUNS, print_on_destroy);
        for (int run = 0; run < BENCH_RUNS; run++) {
            status = run_gemm();
            if (status != CUBLAS_STATUS_SUCCESS) {
                std::cout << "cuBLAS kernel error: " << cublas_status_to_string(status) << std::endl;
                break;
            }
        }
        avg_ms = timer.stop();
        nvtxRangePop();
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "Kernel launch error: " << cudaGetErrorString(err) << std::endl;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cout << "Kernel execution error: " << cudaGetErrorString(err) << std::endl;
    }

    cudaMemcpy(hC, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    cublasDestroy(handle);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return avg_ms;
}
