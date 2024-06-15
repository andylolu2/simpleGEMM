#include <cuda_runtime.h>

#include <cute/tensor.hpp>

#include "pointwise.cuh"

namespace ct = cute;

int main(int argc, char const *argv[]) {
    if (argc != 4) {
        std::cout << "Usage: " << argv[0] << " M N iters" << std::endl;
        return 1;
    }
    int64_t M = atoi(argv[1]);
    int64_t N = atoi(argv[2]);
    size_t iters = atoi(argv[3]);

    using T = ct::half_t;

    // Allocate A, B, C
    T *A_ptr;
    T *B_ptr;
    cudaMalloc(&A_ptr, M * N * sizeof(T));
    cudaMalloc(&B_ptr, M * N * sizeof(T));
    auto A = ct::make_tensor(ct::make_gmem_ptr(A_ptr), ct::make_layout(ct::make_shape(M, N), ct::GenRowMajor{}));
    auto B = ct::make_tensor(ct::make_gmem_ptr(B_ptr), ct::make_layout(ct::make_shape(M, N), ct::GenRowMajor{}));

    // Time and benchmark
    std::vector<cudaEvent_t> start_events;
    std::vector<cudaEvent_t> end_events;
    for (size_t i = 0; i < iters; i++) {
        cudaEvent_t start_event;
        cudaEvent_t end_event;
        cudaEventCreate(&start_event);
        cudaEventCreate(&end_event);
        start_events.push_back(start_event);
        end_events.push_back(end_event);
    }

    // Start benchmark
    void *z_ptr;
    cudaMalloc(&z_ptr, 3 * 1024 * 1024);  // Size of my L2 cache
    for (size_t i = 0; i < iters; i++) {
        cudaMemset(z_ptr, 0, 3 * 1024 * 1024);  // Flush L2 cache
        cudaEventRecord(start_events[i]);
        simplegemm::relu(A, B);
        cudaEventRecord(end_events[i]);
    }
    cudaFree(z_ptr);

    // Report benchmark results
    float total_duration = 0;
    for (size_t i = 0; i < iters; i++) {
        cudaEventSynchronize(end_events[i]);
        float duration;  // in ms
        cudaEventElapsedTime(&duration, start_events[i], end_events[i]);
        total_duration += duration;
    }
    float tflops = M * N * iters / (total_duration / 1000) / 1E12;
    float bandwidth = 2 * M * N * iters * sizeof(T) / (total_duration / 1000) / 1E9;
    std::cout << "Time elapse: " << total_duration << "ms" << std::endl;
    std::cout << "TFLOPS: " << tflops << std::endl;
    std::cout << "Bandwidth: " << bandwidth << "GB/s" << std::endl;

    // Deallocate A, B, C
    cudaFree(A_ptr);
    cudaFree(B_ptr);

    return 0;
}