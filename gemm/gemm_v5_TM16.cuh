/* 
该版本基于v5p,适配了BM=256,TM=16的情况
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
__global__ void gemm_reg_kernel_v5_TM16(float *dA, float *dB, float *dC, int M, int K, int N) {
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

    for (int k = 0; k < K; k += BK) {
        __syncthreads();
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
            FLOAT4(regA[0]) = FLOAT4(shared_A[i][threadIdx.y * TM / 4]);
            FLOAT4(regA[4]) = FLOAT4(shared_A[i][threadIdx.y * TM / 4 + BM / 4]);
            FLOAT4(regA[8]) = FLOAT4(shared_A[i][threadIdx.y * TM / 4 + BM / 2]);
            FLOAT4(regA[12]) = FLOAT4(shared_A[i][threadIdx.y * TM / 4 + BM * 3 / 4]);

            FLOAT4(regB[0]) = FLOAT4(shared_B[i][threadIdx.x * TN / 4]);
            FLOAT4(regB[4]) = FLOAT4(shared_B[i][threadIdx.x * TN / 4 + BN / 4]);
            FLOAT4(regB[8]) = FLOAT4(shared_B[i][threadIdx.x * TN / 4 + BN / 2]);
            FLOAT4(regB[12]) = FLOAT4(shared_B[i][threadIdx.x * TN / 4 + BN * 3 / 4]);

            // calculate tile C
            #pragma unroll
            for (int j = 0; j < TM; j++) {
                for (int k = 0; k < TN; k++) {
                    regC[j][k] += regA[j] * regB[k];
                }
            }
        }
        
        
    }

    for (int j = 0; j < 4; j++) {
      for (int k = 0; k < 4; k++) {
        #pragma unroll
        for (int i = 0; i < TM / 4; i++) {
            int store_gmem_m = blockIdx_y * BM + BM / 4 * j + threadIdx.y * TM / 4 + i;
            int store_gmem_n = blockIdx_x * BN + BN / 4 * k + threadIdx.x * TN / 4;
            // FLOAT4(dC[OFFSET(store_gmem_m, store_gmem_n, N)]) = FLOAT4(regC[i + TM / 2][0]);
            FLOAT4(dC[OFFSET(store_gmem_m, store_gmem_n, N)]) = FLOAT4(regC[i + TM / 4 * j][TN / 4 * k]);
        }
      }
    }
}

template <int BM = 128, int BK = 8, int BN = 128, int TM = 8, int TN = 8>
float gemm_reg_v5_TM16(float *hA, float *hB, float *hC, int M, int K, int N) {
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

    nvtxRangePush("gemm_reg_kernel_v5_TM16");
    gemm_reg_kernel_v5_TM16<BM, BK, BN, TM, TN><<<grid, block>>>(dA, dB, dC, M, K, N);
    cudaDeviceSynchronize();

    float avg_ms = 0.0f;
    {
        CudaTimer timer("gemm_reg_v5_TM16", BENCH_RUNS, print_on_destroy);
        for (int run = 0; run < BENCH_RUNS; run++) {
            gemm_reg_kernel_v5_TM16<BM, BK, BN, TM, TN><<<grid, block>>>(dA, dB, dC, M, K, N);
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
