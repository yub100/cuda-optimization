/* 
该版本基于v4，优化smemA的store方式，按照KxM也就是转置形式顺序存储，存在2-bankconflict，
但是消除了load smemA过程中存在的bank conflict,
使用FLOAT4取，每个线程计算4x4结果矩阵，load smem过程中无bankconflict,
还在此基础上进一步：使每个thread计算4x4大小C tile，这样可以让thread以4个FLOAT为单位挨着访问smem
smem大小必须固定

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
__global__ void gemm_reg_kernel_v5(float *dA, float *dB, float *dC, int M, int K, int N) {
    __shared__ float shared_A[BK][BM];
    __shared__ float shared_B[BK][BN];
    float regA[TM];
    float regB[TN];
    float regC[TM][TN] = {0.0};
    float load_a_r[4];

    constexpr int blockDim_x = BN / TN;
    constexpr int blockDim_y = BM / TM;
    int blockIdx_x = blockIdx.x;
    int blockIdx_y = blockIdx.y;

    int tid = threadIdx.y * blockDim_x + threadIdx.x;
    constexpr int BLOCK_THREADS = (BM / TM) * (BN / TN);


    int load_smema_k = (tid & 1) << 2; //  y = (tid == 0 ? 0 : 4)
    int load_smema_m = tid / 2;

    int load_smemb_n = (tid % 32) * 4;
    int load_smemb_k = tid / 32;

    int load_gmem_m = blockIdx_y * BM + load_smema_m;
    int load_gmem_n = blockIdx_x * BN + load_smemb_n;

    for (int k = 0; k < K; k += BK) {
         __syncthreads();

        int load_gmema_k = k + load_smema_k;
        int start_gmema = OFFSET(load_gmem_m, load_gmema_k, K);

        FLOAT4(load_a_r[0]) = FLOAT4(dA[start_gmema]);

        shared_A[load_smema_k][load_smema_m] = load_a_r[0];
        shared_A[load_smema_k + 1][load_smema_m] = load_a_r[1];
        shared_A[load_smema_k + 2][load_smema_m] = load_a_r[2];
        shared_A[load_smema_k + 3][load_smema_m] = load_a_r[3];
        
        int load_gmemb_k = k + load_smemb_k;
        FLOAT4(shared_B[load_smemb_k][load_smemb_n]) = FLOAT4(dB[OFFSET(load_gmemb_k, load_gmem_n, N)]);

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < BK; i++) {
            FLOAT4(regA[0]) = FLOAT4(shared_A[i][threadIdx.y * TM / 2]);
            FLOAT4(regA[4]) = FLOAT4(shared_A[i][threadIdx.y * TM / 2 + BM / 2]);
            FLOAT4(regB[0]) = FLOAT4(shared_B[i][threadIdx.x * TN / 2]);
            FLOAT4(regB[4]) = FLOAT4(shared_B[i][threadIdx.x * TN / 2 + BN / 2]);

            // calculate tile C
            #pragma unroll
            for (int j = 0; j < TM; j++) {
                #pragma unroll
                for (int k = 0; k < TN; k++) {
                    regC[j][k] += regA[j] * regB[k];
                }
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
float gemm_reg_v5(float *hA, float *hB, float *hC, int M, int K, int N) {
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

    nvtxRangePush("gemm_reg_kernel_v5");
    gemm_reg_kernel_v5<BM, BK, BN, TM, TN><<<grid, block>>>(dA, dB, dC, M, K, N);
    cudaDeviceSynchronize();

    float avg_ms = 0.0f;
    {
        CudaTimer timer("gemm_reg_v5", BENCH_RUNS, print_on_destroy);
        for (int run = 0; run < BENCH_RUNS; run++) {
            gemm_reg_kernel_v5<BM, BK, BN, TM, TN><<<grid, block>>>(dA, dB, dC, M, K, N);
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
