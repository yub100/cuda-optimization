/* 
该版本采用FLOAT4存取方法,因此smem元素必须为128x8个，且行列都必须为4的整数倍
A、B的shape同前几个版本一致
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
__global__ void gemm_reg_kernel_v4(float *dA, float *dB, float *dC, int M, int K, int N) {
    __shared__ float shared_A[BM][BK];
    __shared__ float shared_B[BK][BN];
    float regA[TM];
    float regB[TN];
    float regC[TM][TN];

    int blockDim_x = blockDim.x;
    int blockDim_y = blockDim.y;
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

    int load_smema_k = (tid & 1) << 2; //  y = (tid == 0 ? 0 : 4)
    int load_smema_m = tid / 2;

    int load_smemb_n = (tid % 32) * 4;
    int load_smemb_k = tid / 32;

    int load_gmem_m = blockIdx_y * BM + load_smema_m;
    int load_gmem_n = blockIdx_x * BN + load_smemb_n;

    for (int k = 0; k < K; k += BK) {
        int load_gmema_k = k + load_smema_k;
        FLOAT4(shared_A[load_smema_m][load_smema_k]) = FLOAT4(dA[OFFSET(load_gmem_m, load_gmema_k, K)]);

        int load_gmemb_k = k + load_smemb_k;
        FLOAT4(shared_B[load_smemb_k][load_smemb_n]) = FLOAT4(dB[OFFSET(load_gmemb_k, load_gmem_n, N)]);

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < BK; i++) {
            int row = threadIdx.y * TM;
            int col = threadIdx.x * TN;

            // store RegA
            #pragma unroll
            for (int j = 0; j < TM; j++) {
                regA[j] = shared_A[row + j][i];
            }

            // store RegA
            #pragma unroll
            for (int j = 0; j < TN; j++) {
                regB[j] = shared_B[i][col + j];
            }

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
    for (int i = 0; i < TM; i++) {
        int store_gmem_m = blockIdx_y * BM + threadIdx.y * TM + i;

        #pragma unroll
        for (int j = 0; j < TN; j += 4) {
            int store_gmem_n = blockIdx_x * BN + threadIdx.x * TN + j;
            FLOAT4(dC[OFFSET(store_gmem_m, store_gmem_n, N)]) = FLOAT4(regC[i][j]);

        }
    }
}

void gemm_reg_v4(float *hA, float *hB, float *hC, int M, int K, int N) {
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
    constexpr int BK = 8;
    
    constexpr int TM = 8;
    constexpr int TN = 8;

    int num_BLOCK_x = (N + BN - 1) / BN;
    int num_BLOCK_y = (M + BM - 1) / BM;

    dim3 block(BN / TN, BM / TM, 1);
    dim3 grid(num_BLOCK_x, num_BLOCK_y, 1);

    nvtxRangePush("gemm_reg_kernel_v4");
    gemm_reg_kernel_v4<BM, BK, BN, TM, TN><<<grid, block>>>(dA, dB, dC, M, K, N);
    cudaDeviceSynchronize();

    {
        CudaTimer timer("gemm_reg_v4");
        gemm_reg_kernel_v4<BM, BK, BN, TM, TN><<<grid, block>>>(dA, dB, dC, M, K, N);
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
