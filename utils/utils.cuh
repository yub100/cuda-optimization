#pragma once
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <mma.h>
#include <string>

#define FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])
#define OFFSET(m, n, ld) (m * ld + n)

void random_matrix(float* mat, int size, float min = -1.0f, float max = 1.0f) {
    for (int i = 0; i < size; i++) {
        mat[i] = min + static_cast<float>(rand()) / RAND_MAX * (max - min);
    }
}

class CudaTimer {
public:
    CudaTimer(std::string name) : _name(name) {
        
        cudaEventCreate(&start);
        cudaEventCreate(&end);
        cudaEventRecord(start);
    }

    ~CudaTimer() {
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        
        float milliseconds = 0.0f;
        cudaEventElapsedTime(&milliseconds, start, end);
        std::cout << _name << ":\t\t" << milliseconds << "ms" << std::endl;
    }

    CudaTimer(const CudaTimer&) = delete;
    CudaTimer& operator=(const CudaTimer&) = delete;

private:
    cudaEvent_t start, end;
    std::string _name;
};