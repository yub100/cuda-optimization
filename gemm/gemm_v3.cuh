/* 
该版本相较于gemm_reg_kernel_v1
采用线性存储方法消除store shared memory B时的2路bank conflict
*/
#pragma once
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
__global__ void gemm_reg_kernel_v2(float *dA, float *dB, float *dC, int M, int K, int N) {
    __shared__ float shared_A[BM][BK];
    __shared__ float shared_B[BK][BN];
    float regA[TM];
    float regB[TN];
    float regC[TM][TN];

    int blockDim_x = blockDim.x;
    int blockDim_y = blockDim.y;
    int blockIdx_x = blockIdx.x;
    int blockIdx_y = blockIdx.y;

    #pragma unroll
    for (int j = 0; j < TM; j++) {
        for (int k = 0; k < TN; k++) {
            regC[j][k] = 0.0f;
        }
    }


    for (int k = 0; k < K; k += BK) {
        // store shared_A
        for (int i = 0; i < BM; i += blockDim_y) {
            for (int j = 0; j < BK; j += blockDim_x) {
                int row = blockIdx_y * BM + i + threadIdx.y;
                int col = k + j + threadIdx.x;
                if (i + threadIdx.y < BM && j + threadIdx.x < BK) {
                    if (row < M && col < K) {
                        shared_A[i + threadIdx.y][j + threadIdx.x] = dA[row * K + col];
                    } else {
                        shared_A[i + threadIdx.y][j + threadIdx.x] = 0.0f;
                    }
                }
            }
        }

        // 优化版本，解决bank conflict
        int tid = threadIdx.y * blockDim_x + threadIdx.x;
        constexpr int BLOCK_THREADS = (BN / TN) * (BM / TM);
        for (int t = tid ; t < BK * BN; t += BLOCK_THREADS) {
            int smem_row = t / BN;
            int smem_col = t % BN;
            int gmem_row = k + smem_row;
            int gmem_col = blockIdx_x * BN + smem_col;
            if (gmem_row < K && gmem_col < N) {
                shared_B[smem_row][smem_col] = dB[gmem_row * N + gmem_col];
            } else {
                shared_B[smem_row][smem_col] = 0.0f;
            }
        }
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
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            int row = blockIdx_y * BM + threadIdx.y * TM + i;
            int col = blockIdx_x * BN + threadIdx.x * TN + j;
            if (row < M && col < N) dC[row * N + col] = regC[i][j];
        }
    }
}

void gemm_reg_v2(float *hA, float *hB, float *hC, int M, int K, int N) {
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

    nvtxRangePush("gemm_reg_kernel_v2");
    gemm_reg_kernel_v2<BM, BK, BN, TM, TN><<<grid, block>>>(dA, dB, dC, M, K, N);
    cudaDeviceSynchronize();
    {
        CudaTimer Timer("gemm_reg_v2");
        gemm_reg_kernel_v2<BM, BK, BN, TM, TN><<<grid, block>>>(dA, dB, dC, M, K, N);
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
