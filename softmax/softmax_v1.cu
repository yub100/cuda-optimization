#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>
#include "../tools/common.cuh"

void softmax_cpu(float *output, const float *input, int R, int C) {
    for (int i = 0; i < R; i++) {
        float max = input[i * C];
        float sum = 0;
        for (int j = 0; j < C; j++) {
            float n = input[i * C + j];
            float o = max;
            max = n > max ? n : max;
            sum = sum * expf(o - max) + expf(n - max);
        }
        for (int j = 0; j < C; j++) {
            output[i * C + j] = expf(input[i * C + j] - max) / sum;
        }
    }
}

__global__ void softmax_kernel_v1(float *out, const float *in, int R, int C) {
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int x = row * C;
    if (row < R) {
        float max = in[x];
        float sum = 0;
        for (int i = 0; i < C; i++) {
            float n = in[x + i];
            if (n > max) {
                sum = sum * expf(max - n) + 1;
                max = n;
            } else {
                sum = sum + expf(n - max);
            }
        }
        for (int i = 0; i < C; i++) {
            out[x + i] = expf(in[x + i] - max) / sum;
        }
    }
}

double time_softmax_cpu_ms(float* out, const float* in, int R, int C,
                           int warmup=3, int iters=30) {
    // warmup
    for (int i = 0; i < warmup; ++i) softmax_cpu(out, in, R, C);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) softmax_cpu(out, in, R, C);
    auto t1 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> ms = t1 - t0;
    return ms.count() / iters; // 平均每次调用的毫秒数
}

float softmax_v1(int block_size, int R, int C, float* dout, float* din) {
    dim3 block(block_size);
    dim3 grid((R + block_size - 1) / block_size);
    
    // 预热
    for (int i = 0; i < 5; i++) {
        softmax_kernel_v1<<<grid, block>>>(dout, din, R, C);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int iters = 200;  // 多跑几次平均更稳
    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        softmax_kernel_v1<<<grid, block>>>(dout, din, R, C);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= iters;  // 平均每次 kernel 的耗时（ms）

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return ms;
}

int main() {
    // initialize
    int R = 256;
    int C = 256;
    float *hin = (float*)malloc(R * C * sizeof(float));
    float *hout = (float*)malloc(R * C * sizeof(float));
    float *din, *dout;
    cudaMalloc((float**)&din, R * C * sizeof(float));
    cudaMalloc((float**)&dout, R * C * sizeof(float));

    initialData(hin, R * C);
    cudaMemcpy(din, hin, R * C * sizeof(float), cudaMemcpyHostToDevice);
    
    int block_size = 32;

    // execute
    float ms = softmax_v1(block_size, R, C, dout, din);
    cudaMemcpy(hout, dout, R * C * sizeof(float), cudaMemcpyDeviceToHost);

    // for (int i = 0; i < 3; i++) {
    // std::cout << "----------------第" << i << "行-----------------\n";
    //     for (int j = 0; j < C; j++) {
    //         std::cout << std::setw(10)
    //                   << std::fixed
    //                   << std::setprecision(5)
    //                   << hout[i * C + j];
    //         if (((j+1) % 8) == 0) std::cout << "\n";
            
    //     }
    // }

    double ms_cpu = time_softmax_cpu_ms(hout, hin, R, C);

    // 
    std::cout << "cpu_time: " << ms_cpu << std::endl;
    std::cout << "gpu_time: " << ms << std::endl;
    return 0;
}

