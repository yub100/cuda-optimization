/* 
该版本基于v5_p实现了swizzle load/store smemA

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
__global__ void gemm_reg_kernel_v6_1(float *dA, float *dB, float *dC, int M, int K, int N) {
    __shared__ float shared_A[BK][BM];
    __shared__ float shared_B[BK][BN];
    float regA[TM];
    float regB[TN];
    float regC[TM][TN];
    float load_a_r[4];

    constexpr int blockDim_x = BN / TN;
    constexpr int blockDim_y = BM / TM;
    int blockIdx_x = blockIdx.x;
    int blockIdx_y = blockIdx.y;

    int tid = threadIdx.y * blockDim_x + threadIdx.x;
    constexpr int BLOCK_THREADS = (BM / TM) * (BN / TN);

    #pragma unroll
    for (int j = 0; j < TM; j++) {
        for (int k = 0; k < TN; k++) {
            regC[j][k] = 0.0f;
        }
    }

    constexpr int K_GROUPS = BK / 4;          // BK=32 -> 8
    constexpr int M_BANK_SPAN = 32 / K_GROUPS; // BK=32 -> 4

    for (int k = 0; k < K; k += BK) {
        // store smemA
        for (int i = 0; i < BM * BK; i += blockDim_x * blockDim_y * 4) {

            int load_smema_k = (i + tid * 4) % BK;
            int load_smema_m = (i + tid * 4) / BK;

            int load_gmem_m = blockIdx_y * BM + load_smema_m;
            int load_gmema_k = k + load_smema_k;
            int start_gmema = OFFSET(load_gmem_m, load_gmema_k, K);

            FLOAT4(load_a_r[0]) = FLOAT4(dA[start_gmema]);

            // swizzle
            int k_group = (load_smema_k >> 2) & (K_GROUPS - 1);
            load_smema_m = load_smema_m ^ (k_group * M_BANK_SPAN);

            shared_A[load_smema_k][load_smema_m] = load_a_r[0];
            shared_A[load_smema_k + 1][load_smema_m] = load_a_r[1];
            shared_A[load_smema_k + 2][load_smema_m] = load_a_r[2];
            shared_A[load_smema_k + 3][load_smema_m] = load_a_r[3];
            
        }

        // store smemB
        for (int i = 0; i < BN * BK; i += blockDim_x * blockDim_y * 4) {

            int load_smemb_k = (i + tid * 4) / BN;
            int load_smemb_n = (i + tid * 4) % BN;

            int load_gmem_n = blockIdx_x * BN + load_smemb_n;
            int load_gmemb_k = k + load_smemb_k;

            FLOAT4(shared_B[load_smemb_k][load_smemb_n]) = FLOAT4(dB[OFFSET(load_gmemb_k, load_gmem_n, N)]);
            
        }

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < BK; i++) {
            int load_smemA_m1 = threadIdx.y * TM / 2;
            int load_smemA_m2 = threadIdx.y * TM / 2 + BM / 2;

            // swizzle
            int k_group = (i >> 2) & (K_GROUPS - 1);
            load_smemA_m1 = load_smemA_m1 ^ (k_group * M_BANK_SPAN);
            load_smemA_m2 = load_smemA_m2 ^ (k_group * M_BANK_SPAN);

            FLOAT4(regA[0]) = FLOAT4(shared_A[i][load_smemA_m1]);
            FLOAT4(regA[4]) = FLOAT4(shared_A[i][load_smemA_m2]);
            FLOAT4(regB[0]) = FLOAT4(shared_B[i][threadIdx.x * TN / 2]);
            FLOAT4(regB[4]) = FLOAT4(shared_B[i][threadIdx.x * TN / 2 + BN / 2]);

            // calculate tile C
            #pragma unroll
            for (int j = 0; j < TM; j++) {
                for (int k = 0; k < TN; k++) {
                    regC[j][k] += regA[j] * regB[k];
                }
            }
        }
        
        __syncthreads();
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
float gemm_reg_v6_1(float *hA, float *hB, float *hC, int M, int K, int N) {
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

    nvtxRangePush("gemm_reg_kernel_v6_1");
    gemm_reg_kernel_v6_1<BM, BK, BN, TM, TN><<<grid, block>>>(dA, dB, dC, M, K, N);
    cudaDeviceSynchronize();

    float avg_ms = 0.0f;
    {
        CudaTimer timer("gemm_reg_v6_1", BENCH_RUNS, print_on_destroy);
        for (int run = 0; run < BENCH_RUNS; run++) {
            gemm_reg_kernel_v6_1<BM, BK, BN, TM, TN><<<grid, block>>>(dA, dB, dC, M, K, N);
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
