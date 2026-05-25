#include <iostream>
#include <cmath>
#include "../tools/common.cuh"

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

// v4: wrap + shared memory 
// same as v2, on block operate one row, but blockDim.x > 32
__global__ void softmax_kernel_v4(float *out, float *in, int R, int C) {
    int bid = blockIdx.x;
    int tid = threadIdx.x;
    float *x = in + bid * C;
    int block_size = blockDim.x;
    extern __shared__ float shared[];
    int laneId = tid % 32;
    int warpId = tid / 32;
    
    int warpsPerBlock = blockDim.x / 32;
    float* maxvals = shared;
    float* sumvals = &shared[warpsPerBlock];

    // // initialize the array that will be reduced
    float maxval = -INFINITY;
    for (int i = tid; i < C; i += block_size) {
        maxval = fmaxf(maxval, x[i]);
    }

    // find the max value in this row by reduce from this wrap
    maxval = warpReduceMax(maxval);
    
    // broadcast maxval within the warp
    maxval = __shfl_sync(0xFFFFFFFF, maxval, 0);

    if (laneId == 0) maxvals[warpId] = maxval;
    __syncthreads();

    //  find the max value from maxvals[]
    if (tid == 0) {
        for (int i = 0; i < warpsPerBlock; i++) {
            maxval = fmaxf(maxval, maxvals[i]);
        }
        maxvals[0] = maxval;
    }
    __syncthreads();
    maxval = maxvals[0];

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
    if (laneId == 0) sumvals[warpId] = sumval;
    __syncthreads();

    if (tid == 0) {
        for (int i = 1; i < warpsPerBlock; i++) {
            sumvals[0] += sumvals[i];
        }
    }
    __syncthreads();

    // calculate softmax
    for (int i = tid; i < C; i += block_size) {
        x[i] = x[i] / sumvals[0];
    }
}

float time_softmax_v4_ms(int block_size, int R, int C, float* dout, float* din) {
    dim3 block(block_size);
    dim3 grid(R);
    int sharedSize = block_size / 16 * sizeof(float);
    
    // 预热
    for (int i = 0; i < 5; i++) {
        softmax_kernel_v4<<<grid, block, sharedSize>>>(dout, din, R, C);
    }
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int iters = 200;  // 多跑几次平均更稳
    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        softmax_kernel_v4<<<grid, block, sharedSize>>>(dout, din, R, C);
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
    float ms = time_softmax_v4_ms(block_size, R, C, dout, din);
    cudaMemcpy(hout, dout, R * C * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "x4_gpu_time: " << ms << std::endl;
}

