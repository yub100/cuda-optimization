#pragma once
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <mma.h>
#include <string>

#define FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])
#define OFFSET(m, n, ld) (m * ld + n)

constexpr int BENCH_RUNS = 4;

constexpr bool print_on_destroy = true;
constexpr bool write_csv = true;

void random_matrix(float* mat, int size, float min = -1.0f, float max = 1.0f) {
    for (int i = 0; i < size; i++) {
        mat[i] = min + static_cast<float>(rand()) / RAND_MAX * (max - min);
    }
}

class CudaTimer {
public:
    CudaTimer(std::string name = "", int repeats = 1, bool timer_print_on_destroy = print_on_destroy)
        : _name(name),
          _repeats(repeats > 0 ? repeats : 1),
          _print_on_destroy(timer_print_on_destroy),
          _stopped(false),
          _milliseconds(0.0f) {
        
        cudaEventCreate(&start);
        cudaEventCreate(&end);
        cudaEventRecord(start);
    }

    ~CudaTimer() {
        if (_print_on_destroy) {
            float average_ms = _stopped ? _milliseconds / _repeats : stop();
            if (!_name.empty()) {
                std::cout << _name << ":\t\t" << average_ms << "ms";
                if (_repeats > 1) {
                    std::cout << " (avg of " << _repeats << ")";
                }
                std::cout << std::endl;
            }
        }

        cudaEventDestroy(start);
        cudaEventDestroy(end);
    }

    float stop() {
        if (!_stopped) {
            cudaEventRecord(end);
            cudaEventSynchronize(end);
            cudaEventElapsedTime(&_milliseconds, start, end);
            _stopped = true;
        }
        return _milliseconds / _repeats;
    }

    CudaTimer(const CudaTimer&) = delete;
    CudaTimer& operator=(const CudaTimer&) = delete;

private:
    cudaEvent_t start, end;
    std::string _name;
    int _repeats;
    bool _print_on_destroy;
    bool _stopped;
    float _milliseconds;
};
