#include <iostream>
#include <cmath>
#include "../tools/common.cuh"
#include <cuda.h>
#include <cuda_runtime.h>

__device__ float warpReduceMax(float val) {
    for (int offset = 16; offset >= 1; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

__device__ float warpReduceSum(float val) {
    for (int offset = 16; offset >= 1; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
    
}

// v3: wrap
// same as v2, on block operate one row
__global__ void softmax_kernel_v3(float *out, float *in, int R, int C) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    float *x = in + bid * C;
    int block_size = blockDim.x;
    
    // // initialize the array that will be reduced
    float maxval = -INFINITY;
    for (int i = tid; i < C; i += block_size) {
        maxval = fmaxf(maxval, x[i]);
    }
    //shared[tid] = maxval;

    // find the max value in this row by reduce
    maxval = warpReduceMax(maxval);
    
    // broadcast maxval within the warp
    maxval = __shfl_sync(0xFFFFFFFF, maxval, 0);

    // initialize the array that will be reduced
    float sumval = 0.0f;
    for (int i = tid; i < C; i += block_size) {
        out[bid * C + i] = expf(x[i] - maxval);
    }

    x = out + bid * C;
    for (int i = tid; i < C; i += block_size) {
        sumval += x[i];
    }

    // sum by reduce and broadcast
    sumval = warpReduceSum(sumval);
    sumval = __shfl_sync(0xFFFFFFFF, sumval, 0);

    // calculate softmax
    for (int i = tid; i < C; i += block_size) {
        x[i] = x[i] / sumval;
    }
}

float time_softmax_v3_ms(int block_size, int R, int C, float* dout, float* din) {
    dim3 block(block_size);
    dim3 grid(R);
    
    // 预热
    for (int i = 0; i < 5; i++) {
        softmax_kernel_v3<<<grid, block>>>(dout, din, R, C);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int iters = 200;  // 多跑几次平均更稳
    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        softmax_kernel_v3<<<grid, block>>>(dout, din, R, C);
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
    float ms = time_softmax_v3_ms(block_size, R, C, dout, din);
    cudaMemcpy(hout, dout, R * C * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "x2_gpu_time: " << ms << std::endl;
}

