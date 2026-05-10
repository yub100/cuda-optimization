#pragma once

#include <iostream>
#include <nvToolsExt.h>
#include "../utils/utils.cuh"

#if __has_include(<cutlass/gemm/device/gemm.h>)
#define GEMM_HAS_CUTLASS 1

#include <cutlass/arch/arch.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/layout/matrix.h>

template <int BM = 128, int BK = 8, int BN = 128, int TM = 8, int TN = 8>
float gemm_cutlass(float *hA, float *hB, float *hC, int M, int K, int N) {
    (void)BM;
    (void)BK;
    (void)BN;
    (void)TM;
    (void)TN;

    static_assert(BM % 4 == 0, "CUTLASS warp shape requires BM to be divisible by 4.");
    static_assert(BN % 2 == 0, "CUTLASS warp shape requires BN to be divisible by 2.");

    float *dA, *dB, *dC;

    nvtxRangePush("gemm_cutlass_start_up_malloc");
    cudaMalloc((void **)&dA, M * K * sizeof(float));
    cudaMalloc((void **)&dB, K * N * sizeof(float));
    cudaMalloc((void **)&dC, M * N * sizeof(float));
    nvtxRangePop();

    cudaMemcpy(dA, hA, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(dC, 0, M * N * sizeof(float));

    using ThreadblockShape = cutlass::gemm::GemmShape<BM, BN, BK>;
    using WarpShape = cutlass::gemm::GemmShape<BM / 4, BN / 2, BK>;
    using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;

    using CutlassGemm = cutlass::gemm::device::Gemm<
        float, cutlass::layout::RowMajor,
        float, cutlass::layout::RowMajor,
        float, cutlass::layout::RowMajor,
        float,
        cutlass::arch::OpClassSimt,
        cutlass::arch::Sm80,
        ThreadblockShape,
        WarpShape,
        InstructionShape>;

    CutlassGemm gemm_op;

    typename CutlassGemm::Arguments args(
        {M, N, K},
        {dA, K},
        {dB, N},
        {dC, N},
        {dC, N},
        {1.0f, 0.0f});

    cutlass::Status status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        std::cout << "CUTLASS can_implement error: " << static_cast<int>(status) << std::endl;
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
        return -1.0f;
    }

    nvtxRangePush("gemm_cutlass_kernel");
    status = gemm_op(args);
    cudaDeviceSynchronize();

    if (status != cutlass::Status::kSuccess) {
        std::cout << "CUTLASS warmup error: " << static_cast<int>(status) << std::endl;
        nvtxRangePop();
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
        return -1.0f;
    }

    float avg_ms = 0.0f;
    {
        CudaTimer timer("gemm_cutlass", BENCH_RUNS, false);
        for (int run = 0; run < BENCH_RUNS; run++) {
            status = gemm_op(args);
            if (status != cutlass::Status::kSuccess) {
                std::cout << "CUTLASS kernel error: " << static_cast<int>(status) << std::endl;
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

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    return avg_ms;
}

#else
#define GEMM_HAS_CUTLASS 0

template <int BM = 128, int BK = 8, int BN = 128, int TM = 8, int TN = 8>
float gemm_cutlass(float*, float*, float*, int, int, int) {
    std::cout << "gemm_cutlass skipped: CUTLASS headers were not found." << std::endl;
    return -1.0f;
}

#endif
