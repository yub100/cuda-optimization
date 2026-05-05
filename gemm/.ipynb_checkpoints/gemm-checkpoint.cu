#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <nvToolsExt.h>

void random_matrix(float* mat, int size, float min = -1.0f, float max = 1.0f) {
    for (int i = 0; i < size; i++) {
        mat[i] = min + static_cast<float>(rand()) / RAND_MAX * (max - min);
    }
}

float gemm_cpu(const float* hA, const float* hB_T, float* hC, int M, int K, int N) {
    nvtxRangePush("gemm_cpu");
    auto start = std::chrono::high_resolution_clock::now();
    // 初始化 C 为 0
    for (int i = 0; i < M * N; i++) {
        hC[i] = 0.0f;
    }
    
    // 三重循环：C[row][col] = sum_{k=0}^{K-1} A[row][k] * B[k][col]
    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                // A[row][k] 的索引：row * K + k
                // B_T[col][k] 的索引：col * K + k
                // sum += hA[row * K + k] * hB_T[col * K + k];
                sum += hA[row * K + k] * hB_T[k * N + col];
            }
            hC[row * N + col] = sum;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();

    float ms = std::chrono::duration<float, std::milli>(end - start).count();
    nvtxRangePop();
    return ms;
}

// dA.shape = [M, K], dB.shape = [K, N]
template <int BLOCK_DIM>
__global__ void gemm_kernel(float *dA, float *dB, float *dC, int M, int K, int N) {
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

float gemm_sharedmemory(float *hA, float *hB, float *hC, int M, int K, int N) {
    float *dA, *dB, *dC;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

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
    gemm_kernel<32><<<grid, block>>>(dA, dB, dC, M, K, N);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    gemm_kernel<32><<<grid, block>>>(dA, dB, dC, M, K, N);
    nvtxRangePop();

    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, end);
    // std::cout << "Shared memory Kernel time: " << milliseconds << " ms" << std::endl;

    cudaMemcpy(hC, dC, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return milliseconds;
}


template <int BM, int BK, int BN, int TM, int TN>
__global__ void gemm_reg_kernel_v1(float *dA, float *dB, float *dC, int M, int K, int N) {
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
        
        // store shared_B
        for (int i = 0; i < BK; i += blockDim_y) {
            for (int j = 0; j < BN; j += blockDim_x) {
                int row = k + i + threadIdx.y;
                int col = blockIdx_x * BN + j + threadIdx.x;
                if (i + threadIdx.y < BK && j + threadIdx.x < BN) {
                    if (row < K && col < N) {
                        shared_B[i + threadIdx.y][j + threadIdx.x] = dB[row * N + col];
                    } else {
                        shared_B[i + threadIdx.y][j + threadIdx.x] = 0.0f;
                    }
                }
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

float gemm_reg(float *hA, float *hB, float *hC, int M, int K, int N) {
    float *dA, *dB, *dC;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

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

    nvtxRangePush("gemm_reg_kernel");
    gemm_reg_kernel_v1<BM, BK, BN, TM, TN><<<grid, block>>>(dA, dB, dC, M, K, N);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    gemm_reg_kernel_v1<BM, BK, BN, TM, TN><<<grid, block>>>(dA, dB, dC, M, K, N);
    nvtxRangePop();

    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, end);
    // std::cout << "Register Kernel time: " << milliseconds << " ms" << std::endl;


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

    return milliseconds;
}

void print1() {
    const int M = 1024;
    const int K = 1024;
    const int N = 1024;
    

    srand(42);
    // 分配主机内存
    float* hA = new float[M * K];
    float* hB = new float[K * N];      // B: K x N
    float* hC_cpu = new float[M * N];  // CPU 计算结果
    float* hC_gpu = new float[M * N];  // GPU 计算结果

    // initialize hA and hB
    random_matrix(hA, M * K);
    random_matrix(hB, K * N);

    float time1 = 0.0;
    float time2 = 0.0;
    float time3 = 0.0;

    // CPU 矩阵乘法
    time1 = gemm_cpu(hA, hB, hC_cpu, M, K, N);
    std::cout << time1 << std::endl;

    time2 = gemm_sharedmemory(hA, hB, hC_gpu, M, K, N);
    std::cout << time2 << std::endl;

    time3 = gemm_reg(hA, hB, hC_gpu, M, K, N);
    std::cout << time3 << std::endl;

    for (int i = 0; i < M * N; i++) {
        if (fabs(hC_cpu[i] - hC_gpu[i]) > 1e-3f){
            std::cout << "false" << std::endl;
            break;
        }
    }

    for (int i = 0; i < 16; i++) {
        std::cout << hC_cpu[i] << " ";
    }
    std::cout << std::endl;

    for (int i = 0; i < 16; i++) {
        std::cout << hC_gpu[i] << " ";
    }
    std::cout << std::endl;
    

}


void print2() {
    srand(42);

    std::vector<int> sizes = {
        128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096, 5120, 6144, 7168, 8192
    };

    std::ofstream fout("gemm_time.csv");
    fout << "M,K,N,cpu_ms,shared_ms,reg_ms,cpu_gflops,shared_gflops,reg_gflops\n";

    for (int size : sizes) {
        int M = size;
        int K = size;
        int N = size;

        std::cout << "Running M=K=N=" << size << std::endl;

        float* hA = new float[M * K];
        float* hB = new float[K * N];
        float* hC_cpu = new float[M * N];
        float* hC_shared = new float[M * N];
        float* hC_reg = new float[M * N];

        random_matrix(hA, M * K);
        random_matrix(hB, K * N);

        float cpu_ms = -1.0f;

        if (size <= 1024) {
            cpu_ms = gemm_cpu(hA, hB, hC_cpu, M, K, N);
        }

        float shared_ms = gemm_sharedmemory(hA, hB, hC_shared, M, K, N);
        float reg_ms = gemm_reg(hA, hB, hC_reg, M, K, N);

        // 正确性检查：只在 CPU 跑过时检查
        if (cpu_ms > 0.0f) {
            bool shared_ok = true;
            bool reg_ok = true;

            for (int i = 0; i < M * N; i++) {
                if (fabs(hC_cpu[i] - hC_shared[i]) > 1e-3f) {
                    shared_ok = false;
                    std::cout << "shared false at " << i
                              << ", cpu=" << hC_cpu[i]
                              << ", shared=" << hC_shared[i]
                              << std::endl;
                    break;
                }
            }

            for (int i = 0; i < M * N; i++) {
                if (fabs(hC_cpu[i] - hC_reg[i]) > 1e-3f) {
                    reg_ok = false;
                    std::cout << "reg false at " << i
                              << ", cpu=" << hC_cpu[i]
                              << ", reg=" << hC_reg[i]
                              << std::endl;
                    break;
                }
            }

            std::cout << "shared ok: " << shared_ok
                      << ", reg ok: " << reg_ok
                      << std::endl;
        }

        double flops = 2.0 * M * N * K;

        double cpu_gflops = cpu_ms > 0.0f
            ? flops / (cpu_ms / 1000.0) / 1e9
            : -1.0;

        double shared_gflops = flops / (shared_ms / 1000.0) / 1e9;
        double reg_gflops = flops / (reg_ms / 1000.0) / 1e9;

        std::cout << "cpu_ms = " << cpu_ms
                  << ", shared_ms = " << shared_ms
                  << ", reg_ms = " << reg_ms
                  << std::endl;

        std::cout << "cpu_gflops = " << cpu_gflops
                  << ", shared_gflops = " << shared_gflops
                  << ", reg_gflops = " << reg_gflops
                  << std::endl;

        fout << M << ","
             << K << ","
             << N << ","
             << cpu_ms << ","
             << shared_ms << ","
             << reg_ms << ","
             << cpu_gflops << ","
             << shared_gflops << ","
             << reg_gflops << "\n";

        delete[] hA;
        delete[] hB;
        delete[] hC_cpu;
        delete[] hC_shared;
        delete[] hC_reg;
    }

    fout.close();

}

int main() {
    print1();
}
