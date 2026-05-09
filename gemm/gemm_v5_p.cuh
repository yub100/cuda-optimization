/* 
该版本基于v5,适配了不同BK,但必须保证BK%8 == 0
结果显示BK越大，性能越好
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

template <int BM, int BK, int BN, int TM, int TN>
__global__ void gemm_reg_kernel_v5_p(float *dA, float *dB, float *dC, int M, int K, int N) {
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


    for (int k = 0; k < K; k += BK) {
        // store smemA
        for (int i = 0; i < BM * BK; i += blockDim_x * blockDim_y * 4) {

            int load_smema_k = (i + tid * 4) % BK;
            int load_smema_m = (i + tid * 4) / BK;

            int load_gmem_m = blockIdx_y * BM + load_smema_m;
            int load_gmema_k = k + load_smema_k;
            int start_gmema = OFFSET(load_gmem_m, load_gmema_k, K);

            FLOAT4(load_a_r[0]) = FLOAT4(dA[start_gmema]);
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
            FLOAT4(regA[0]) = FLOAT4(shared_A[i][threadIdx.y * TM / 2]);
            FLOAT4(regA[4]) = FLOAT4(shared_A[i][threadIdx.y * TM / 2 + BM / 2]);
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

void gemm_reg_v5_p(float *hA, float *hB, float *hC, int M, int K, int N) {
    float *dA, *dB, *dC;

    nvtxRangePush("gemm_reg_start_up_malloc");
    cudaMalloc((void **)&dA, M * K * sizeof(float));
    cudaMalloc((void **)&dB, N * K * sizeof(float));
    cudaMalloc((void **)&dC, M * N * sizeof(float));
    nvtxRangePop();

    cudaMemcpy(dA, hA, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, N * K * sizeof(float), cudaMemcpyHostToDevice);


    // one thread calculate 8x8 matrix.
    // one Block calculate BM x BN of C.
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 32;
    
    constexpr int TM = 8;
    constexpr int TN = 8;

    int num_BLOCK_x = (N + BN - 1) / BN;
    int num_BLOCK_y = (M + BM - 1) / BM;

    dim3 block(BN / TN, BM / TM, 1);
    dim3 grid(num_BLOCK_x, num_BLOCK_y, 1);

    nvtxRangePush("gemm_reg_kernel_v5_p");
    gemm_reg_kernel_v5_p<BM, BK, BN, TM, TN><<<grid, block>>>(dA, dB, dC, M, K, N);
    cudaDeviceSynchronize();

    {
        CudaTimer timer("gemm_reg_v5_p", BENCH_RUNS);
        for (int run = 0; run < BENCH_RUNS; run++) {
            gemm_reg_kernel_v5_p<BM, BK, BN, TM, TN><<<grid, block>>>(dA, dB, dC, M, K, N);
        }
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
}
