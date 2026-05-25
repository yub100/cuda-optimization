#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include "../tools/common.cuh"

void softmax_cpu(float *output, const float *input, int R, int C) {
    float *sumMatrix = (float*)malloc(R * sizeof(float));
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

int main() {
    int R = 30;
    int C = 30;
    float *input = (float*)malloc(R * C * sizeof(float));
    float *output = (float*)malloc(R * C * sizeof(float));
    initialData(input, R * C);
    
    softmax_cpu(output, input, R, C);
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < C; j++) {
            std::cout << std::setw(10)
                      << std::fixed
                      << std::setprecision(5)
                      << output[i * C + j];
            if (((j+1) % 8) == 0) std::cout << "\n";
            
        }
        std::cout << std::endl;
    }
    return 0;
}