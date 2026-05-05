#include <iostream>

__global__ void add1(float *x, float *y, float *z) {
    int n = threadIdx.x + blockIdx.x * blockDim.x;
    z[n] = x[n] + y[n];
}

int main() {
    const int N = 32 * 1024 * 1024;
    float *x = (float*)malloc(N * sizeof(float));
    float *y = (float*)malloc(N * sizeof(float));
    float *z = (float*)malloc(N * sizeof(float));
    float *dx, *dy, *dz;
    cudaMalloc((void **)&dx, N * sizeof(float));
    cudaMalloc((void **)&dy, N * sizeof(float));
    cudaMalloc((void **)&dz, N * sizeof(float));
    for (int i = 0; i < N; i++) {
        x[i] = i % 1024;
        y[i] = 1024 - i % 1024;
    }

    cudaMemcpy(dx, x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dy, y, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid(N / 256);
    dim3 block(64);

    for (int i = 0; i < 2; i++) {
        add1<<<grid, block>>>(dx, dy, dz);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(z, dz, N * sizeof(float) / 4, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 64; i++) {
        std::cout << z[i] << " " << std::endl;
    }
    
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dz);
    free(x);
    free(y);
    free(z);
}
