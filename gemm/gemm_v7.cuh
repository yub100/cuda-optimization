/* 
该版本基于v5_p实现双缓冲shared memory。
保持v5_p的smemA布局、计算和store C布局，仅调整K tile流水结构。
*/
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <nvToolsExt.h>
#include "../utils/utils.cuh"

template <int BM = 128, int BK = 8, int BN = 128, int TM = 8, int TN = 8>
__global__ void gemm_reg_kernel_v7(float *dA, float *dB, float *dC, int M, int K, int N) {
    static_assert(BK % 4 == 0, "BK must be a multiple of 4 for FLOAT4 A loads");
    static_assert(TM == 8 && TN == 8, "gemm_reg_kernel_v7 keeps the v6_2 TM=TN=8 layout");

    __shared__ float shared_A[2][BK][BM];
    __shared__ float shared_B[2][BK][BN];
    float regA[TM];
    float regB[TN];
    float regC[TM][TN] = {0.0};

    constexpr int blockDim_x = BN / TN;
    constexpr int blockDim_y = BM / TM;
    constexpr int BLOCK_THREADS = blockDim_x * blockDim_y;
    constexpr int A_LOAD_ITERS = (BM * BK + BLOCK_THREADS * 4 - 1) / (BLOCK_THREADS * 4);
    constexpr int B_LOAD_ITERS = (BN * BK + BLOCK_THREADS * 4 - 1) / (BLOCK_THREADS * 4);

    float load_a_r[A_LOAD_ITERS][4];
    float load_b_r[B_LOAD_ITERS][4];

    int blockIdx_x = blockIdx.x;
    int blockIdx_y = blockIdx.y;
    int tid = threadIdx.y * blockDim_x + threadIdx.x;
    int k_tiles = K / BK;

    #pragma unroll
    for (int load_i = 0; load_i < A_LOAD_ITERS; load_i++) {
        int load_offset = load_i * BLOCK_THREADS * 4 + tid * 4;
        int load_smema_k = load_offset % BK;
        int load_smema_m = load_offset / BK;
        int load_gmem_m = blockIdx_y * BM + load_smema_m;
        int load_gmema_k = load_smema_k;
        FLOAT4(load_a_r[load_i][0]) = FLOAT4(dA[OFFSET(load_gmem_m, load_gmema_k, K)]);
    }

    #pragma unroll
    for (int load_i = 0; load_i < B_LOAD_ITERS; load_i++) {
        int load_offset = load_i * BLOCK_THREADS * 4 + tid * 4;
        int load_smemb_k = load_offset / BN;
        int load_smemb_n = load_offset % BN;
        int load_gmem_n = blockIdx_x * BN + load_smemb_n;
        int load_gmemb_k = load_smemb_k;
        FLOAT4(load_b_r[load_i][0]) = FLOAT4(dB[OFFSET(load_gmemb_k, load_gmem_n, N)]);
    }

    #pragma unroll
    for (int load_i = 0; load_i < A_LOAD_ITERS; load_i++) {
        int load_offset = load_i * BLOCK_THREADS * 4 + tid * 4;
        int load_smema_k = load_offset % BK;
        int load_smema_m = load_offset / BK;

        shared_A[0][load_smema_k][load_smema_m] = load_a_r[load_i][0];
        shared_A[0][load_smema_k + 1][load_smema_m] = load_a_r[load_i][1];
        shared_A[0][load_smema_k + 2][load_smema_m] = load_a_r[load_i][2];
        shared_A[0][load_smema_k + 3][load_smema_m] = load_a_r[load_i][3];
    }

    #pragma unroll
    for (int load_i = 0; load_i < B_LOAD_ITERS; load_i++) {
        int load_offset = load_i * BLOCK_THREADS * 4 + tid * 4;
        int load_smemb_k = load_offset / BN;
        int load_smemb_n = load_offset % BN;
        FLOAT4(shared_B[0][load_smemb_k][load_smemb_n]) = FLOAT4(load_b_r[load_i][0]);
    }

    __syncthreads();

    for (int bk = 1; bk < k_tiles; bk++) {
        int smem_sel = (bk - 1) & 1;
        int smem_sel_next = bk & 1;

        #pragma unroll
        for (int load_i = 0; load_i < A_LOAD_ITERS; load_i++) {
            int load_offset = load_i * BLOCK_THREADS * 4 + tid * 4;
            int load_smema_k = load_offset % BK;
            int load_smema_m = load_offset / BK;
            int load_gmem_m = blockIdx_y * BM + load_smema_m;
            int load_gmema_k = bk * BK + load_smema_k;
            FLOAT4(load_a_r[load_i][0]) = FLOAT4(dA[OFFSET(load_gmem_m, load_gmema_k, K)]);
        }

        #pragma unroll
        for (int load_i = 0; load_i < B_LOAD_ITERS; load_i++) {
            int load_offset = load_i * BLOCK_THREADS * 4 + tid * 4;
            int load_smemb_k = load_offset / BN;
            int load_smemb_n = load_offset % BN;
            int load_gmem_n = blockIdx_x * BN + load_smemb_n;
            int load_gmemb_k = bk * BK + load_smemb_k;
            FLOAT4(load_b_r[load_i][0]) = FLOAT4(dB[OFFSET(load_gmemb_k, load_gmem_n, N)]);
        }

        #pragma unroll
        for (int i = 0; i < BK; i++) {
            FLOAT4(regA[0]) = FLOAT4(shared_A[smem_sel][i][threadIdx.y * TM / 2]);
            FLOAT4(regA[4]) = FLOAT4(shared_A[smem_sel][i][threadIdx.y * TM / 2 + BM / 2]);
            FLOAT4(regB[0]) = FLOAT4(shared_B[smem_sel][i][threadIdx.x * TN / 2]);
            FLOAT4(regB[4]) = FLOAT4(shared_B[smem_sel][i][threadIdx.x * TN / 2 + BN / 2]);

            #pragma unroll
            for (int j = 0; j < TM; j++) {
                #pragma unroll
                for (int k = 0; k < TN; k++) {
                    regC[j][k] += regA[j] * regB[k];
                }
            }
        }

        #pragma unroll
        for (int load_i = 0; load_i < A_LOAD_ITERS; load_i++) {
            int load_offset = load_i * BLOCK_THREADS * 4 + tid * 4;
            int load_smema_k = load_offset % BK;
            int load_smema_m = load_offset / BK;

            shared_A[smem_sel_next][load_smema_k][load_smema_m] = load_a_r[load_i][0];
            shared_A[smem_sel_next][load_smema_k + 1][load_smema_m] = load_a_r[load_i][1];
            shared_A[smem_sel_next][load_smema_k + 2][load_smema_m] = load_a_r[load_i][2];
            shared_A[smem_sel_next][load_smema_k + 3][load_smema_m] = load_a_r[load_i][3];
        }

        #pragma unroll
        for (int load_i = 0; load_i < B_LOAD_ITERS; load_i++) {
            int load_offset = load_i * BLOCK_THREADS * 4 + tid * 4;
            int load_smemb_k = load_offset / BN;
            int load_smemb_n = load_offset % BN;
            FLOAT4(shared_B[smem_sel_next][load_smemb_k][load_smemb_n]) = FLOAT4(load_b_r[load_i][0]);
        }

        __syncthreads();
    }

    int smem_sel = (k_tiles - 1) & 1;

    #pragma unroll
    for (int i = 0; i < BK; i++) {
        FLOAT4(regA[0]) = FLOAT4(shared_A[smem_sel][i][threadIdx.y * TM / 2]);
        FLOAT4(regA[4]) = FLOAT4(shared_A[smem_sel][i][threadIdx.y * TM / 2 + BM / 2]);
        FLOAT4(regB[0]) = FLOAT4(shared_B[smem_sel][i][threadIdx.x * TN / 2]);
        FLOAT4(regB[4]) = FLOAT4(shared_B[smem_sel][i][threadIdx.x * TN / 2 + BN / 2]);

        #pragma unroll
        for (int j = 0; j < TM; j++) {
            #pragma unroll
            for (int k = 0; k < TN; k++) {
                regC[j][k] += regA[j] * regB[k];
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < TM / 2; i++) {
        int store_gmem_m = blockIdx_y * BM + threadIdx.y * TM / 2 + i;
        int store_gmem_n = blockIdx_x * BN + threadIdx.x * TN / 2;
        FLOAT4(dC[OFFSET(store_gmem_m, store_gmem_n, N)]) = FLOAT4(regC[i][0]);
        FLOAT4(dC[OFFSET(store_gmem_m, store_gmem_n + BN / 2, N)]) = FLOAT4(regC[i][4]);
    }

    #pragma unroll
    for (int i = 0; i < TM / 2; i++) {
        int store_gmem_m = blockIdx_y * BM + BM / 2 + threadIdx.y * TM / 2 + i;
        int store_gmem_n = blockIdx_x * BN + threadIdx.x * TN / 2;
        FLOAT4(dC[OFFSET(store_gmem_m, store_gmem_n, N)]) = FLOAT4(regC[i + TM / 2][0]);
        FLOAT4(dC[OFFSET(store_gmem_m, store_gmem_n + BN / 2, N)]) = FLOAT4(regC[i + TM / 2][4]);
    }
}

template <int BM = 128, int BK = 8, int BN = 128, int TM = 8, int TN = 8>
float gemm_reg_v7(float *hA, float *hB, float *hC, int M, int K, int N) {
    float *dA, *dB, *dC;

    nvtxRangePush("gemm_reg_start_up_malloc");
    cudaMalloc((void **)&dA, M * K * sizeof(float));
    cudaMalloc((void **)&dB, N * K * sizeof(float));
    cudaMalloc((void **)&dC, M * N * sizeof(float));
    nvtxRangePop();

    cudaMemcpy(dA, hA, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, N * K * sizeof(float), cudaMemcpyHostToDevice);

    int num_BLOCK_x = (N + BN - 1) / BN;
    int num_BLOCK_y = (M + BM - 1) / BM;

    dim3 block(BN / TN, BM / TM, 1);
    dim3 grid(num_BLOCK_x, num_BLOCK_y, 1);

    nvtxRangePush("gemm_reg_kernel_v7");
    gemm_reg_kernel_v7<BM, BK, BN, TM, TN><<<grid, block>>>(dA, dB, dC, M, K, N);
    cudaDeviceSynchronize();

    float avg_ms = 0.0f;
    {
        CudaTimer timer("gemm_reg_v7", BENCH_RUNS, print_on_destroy);
        for (int run = 0; run < BENCH_RUNS; run++) {
            gemm_reg_kernel_v7<BM, BK, BN, TM, TN><<<grid, block>>>(dA, dB, dC, M, K, N);
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
