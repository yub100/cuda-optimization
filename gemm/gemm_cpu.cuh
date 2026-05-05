#pragma onece
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <nvToolsExt.h>

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
