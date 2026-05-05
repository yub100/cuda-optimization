/* 
该版本实现了shared memroy
dA.shape = M x K;
dB.shape = K x N;
*/
#pragma onece
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <nvToolsExt.h>
#include "../utils/utils.cuh"

// dA.shape = [M, K], dB.shape = [K, N]
template <int BLOCK_DIM>
__global__ void gemm_shared_memory_kernel(float *dA, float *dB, float *dC, int M, int K, int N) {
    __shared__ float shared_a[BLOCK_DIM][BLOCK_DIM];
    __shared__ float shared_b[BLOCK_DIM][BLOCK_DIM];

    float temp = 0.0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int step = (K + BLOCK_DIM - 1) / BLOCK_DIM;
    
    for (int i = 0; i < step; i++) {
        if (row < M && i * BLOCK_DIM + threadIdx.x < K) {
            shared_a[threadIdx.y][threadIdx.x] = dA[row * K + i * BLOCK_DIM + threadIdx.x];
        } else {
            shared_a[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (i * BLOCK_DIM + threadIdx.y < K && col < N) {
            shared_b[threadIdx.y][threadIdx.x] = dB[col + (threadIdx.y + i * BLOCK_DIM) * N];
        } else {
            shared_b[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int i = 0; i < BLOCK_DIM; i++) {
            temp += shared_a[threadIdx.y][i] * shared_b[i][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < M && col < N) {
        dC[row * N + col] = temp;
    }
}


void gemm_sharedmemory(float *hA, float *hB, float *hC, int M, int K, int N) {
    float *dA, *dB, *dC;

    nvtxRangePush("gemm_sharedm_start_up_malloc");
    cudaMalloc((void **)&dA, M * K * sizeof(float));
    cudaMalloc((void **)&dB, N * K * sizeof(float));
    cudaMalloc((void **)&dC, M * N * sizeof(float));
    nvtxRangePop();

    cudaMemcpy(dA, hA, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, N * K * sizeof(float), cudaMemcpyHostToDevice);

    int BLOCK_DIM_x = 32;
    int BLOCK_DIM_y = 32;
    int num_BLOCK_x = (N + BLOCK_DIM_x - 1) / BLOCK_DIM_x;
    int num_BLOCK_y = (M + BLOCK_DIM_y - 1) / BLOCK_DIM_y;

    dim3 block(BLOCK_DIM_x, BLOCK_DIM_y, 1);
    dim3 grid(num_BLOCK_x, num_BLOCK_y, 1);

    nvtxRangePush("gemm_shared_memory");
    // warm up
    gemm_shared_memory_kernel<32><<<grid, block>>>(dA, dB, dC, M, K, N);
    cudaDeviceSynchronize();

    {
        CudaTimer Timer("gemm_shared_memory");
        gemm_shared_memory_kernel<32><<<grid, block>>>(dA, dB, dC, M, K, N);
        nvtxRangePop();
    }
    cudaMemcpy(hC, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}
