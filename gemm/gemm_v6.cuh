/* 
该版本采用对于smemB采用"FLOAT4存"方式,但是抛弃"FLOAT4取"方式
同v5:smemA的store方式，按照KxM也就是转置形式顺序存储，存在2-bankconflict，且一次指令可以取更多数据.
store smemB存在2-bank conflict.
在v5基础上优化smem的load方式，每个线程计算2x2结果矩阵，抛弃了FLOAT4，但消除了bankconflict
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
    __shared__ float shared_A[BK][BM];
    __shared__ float shared_B[BK][BN];
    float regA[TM];
    float regB[TN];
    float regC[TM][TN];
    float load_a_r[4];

    int blockDim_x = blockDim.x;
    int blockDim_y = blockDim.y;
    int blockIdx_x = blockIdx.x;
    int blockIdx_y = blockIdx.y;

    int tid = threadIdx.y * blockDim_x + threadIdx.x;
    constexpr int BLOCK_THREADS = (BM / TM) * (BN / TN);

    // #pragma unroll
    // for (int j = 0; j < TM; j++) {
    //     for (int k = 0; k < TN; k++) {
    //         regC[j][k] = 0.0f;
    //     }
    // }

    int load_smema_k = (tid & 1) << 2; //  y = (tid == 0 ? 0 : 4)
    int load_smema_m = tid / 2;

    int load_smemb_n = (tid % 32) * 4;
    int load_smemb_k = tid / 32;

    int load_gmem_m = blockIdx_y * BM + load_smema_m;
    int load_gmem_n = blockIdx_x * BN + load_smemb_n;

    for (int k = 0; k < K; k += BK) {
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

        for (int k = 0; k < BK; k++) {
            for (int i = 0; i < BM / (2 * blockDim_y); i++) {
                int str_reg_m = 2 * i;
                for (int j = 0; j < BN / (2 * blockDim_x); j++) {
                    int str_reg_n = 2 * j;
                    
                    regA[str_reg_m] = shared_A[k][i * 2 * blockDim_y + threadIdx.y * 2];
                    regA[str_reg_m + 1] = shared_A[k][i * 2 * blockDim_y + threadIdx.y * 2 + 1];
                    regB[str_reg_n] = shared_B[k][i * 2 * blockDim_x + threadIdx.x * 2];
                    regB[str_reg_n] = shared_B[k][i * 2 * blockDim_x + threadIdx.x * 2 + 1];
                    
                }
            }
        }


        #pragma unroll
        for (int i = 0; i < BK; i++) {
            int str_regA_m = threadIdx.y * TM;
            int str_regB_n = threadIdx.x * TN;
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
