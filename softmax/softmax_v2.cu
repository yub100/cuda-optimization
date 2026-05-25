#include <iostream>
#include <cmath>
#include "../tools/common.cuh"

// shared memory
// one block operate one row
__global__ void softmax_kernel_v2(float *out, float *in, int R, int C) {
    // shared.size = block_size
    extern __shared__ float shared[];
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    const float *x = in + bid * C;
    float maxval = -INFINITY;

    // initialize the array that will be reduced
    for (int i = tid; i < C; i += block_size) {
        maxval = fmaxf(maxval, x[i]);
    }
    shared[tid] = maxval;

    for (int stride = block_size / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (tid < stride) {
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }
    }
    __syncthreads();
    float max = shared[0];

    for (int i = tid; i < C; i += block_size) {
        out[bid * C + i] = expf(x[i] - max);
    }
    __syncthreads();

    x = out + bid * C;
    float sumval = 0.0f;
    for (int i = tid; i < C; i += block_size) {
        sumval += x[i];
    }
    shared[tid] = sumval;
    __syncthreads();

    for (int stride = block_size / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
    }
    __syncthreads();
    sumval = shared[0];
    for (int i = tid; i < C; i += block_size) {
        out[bid * C + i] = x[i] / sumval;
    }
}

float time_softmax_v2_ms(int block_size, int R, int C, float* dout, float* din) {
    dim3 block(block_size);
    dim3 grid(R);
    
    // 预热
    for (int i = 0; i < 5; i++) {
        softmax_kernel_v2<<<grid, block, block_size * sizeof(float)>>>(dout, din, R, C);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int iters = 200;  // 多跑几次平均更稳
    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        softmax_kernel_v2<<<grid, block, block_size * sizeof(float)>>>(dout, din, R, C);
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
    int R = 256;
    int C = 1 << 15;
    float *hin = (float*)malloc(R * C * sizeof(float));
    float *hout = (float*)malloc(R * C * sizeof(float));
    float *din, *dout;
    cudaMalloc((float**)&din, R * C * sizeof(float));
    cudaMalloc((float**)&dout, R * C * sizeof(float));

    initialData(hin, R * C);
    cudaMemcpy(din, hin, R * C * sizeof(float), cudaMemcpyHostToDevice);
    
    int block_size = 1024;

    // execute
    float ms = time_softmax_v2_ms(block_size, R, C, dout, din);
    cudaMemcpy(hout, dout, R * C * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "x2_gpu_time: " << ms << std::endl;
}

