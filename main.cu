#include <cuda_runtime.h>

#include <cute/tensor.hpp>

#include "gemm.cuh"

namespace ct = cute;

int main(int argc, char const *argv[]) {
    if (argc != 5) {
        std::cout << "Usage: " << argv[0] << " M N K iters" << std::endl;
        return 1;
    }
    int64_t M = atoi(argv[1]);
    int64_t N = atoi(argv[2]);
    int64_t K = atoi(argv[3]);
    size_t iters = atoi(argv[4]);

    // Allocate A, B, C
    ct::half_t *A_ptr;
    ct::half_t *B_ptr;
    ct::half_t *C_ptr;
    void *z_ptr;
    cudaMalloc(&A_ptr, M * K * sizeof(ct::half_t));
    cudaMalloc(&B_ptr, N * K * sizeof(ct::half_t));
    cudaMalloc(&C_ptr, M * N * sizeof(ct::half_t));
    cudaMalloc(&z_ptr, 3 * 1024 * 1024);  // Size of my L2 cache
    auto A = ct::make_tensor(ct::make_gmem_ptr(A_ptr), ct::make_layout(ct::make_shape(M, K), ct::GenRowMajor{}));
    auto B = ct::make_tensor(ct::make_gmem_ptr(B_ptr), ct::make_layout(ct::make_shape(N, K), ct::GenRowMajor{}));
    auto C = ct::make_tensor(ct::make_gmem_ptr(C_ptr), ct::make_layout(ct::make_shape(M, N), ct::GenRowMajor{}));

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
    for (size_t i = 0; i < iters; i++) {
        cudaMemset(z_ptr, 0, 3 * 1024 * 1024);  // Flush L2 cache
        cudaEventRecord(start_events[i]);
        simplegemm::gemm(A, B, C);
        cudaEventRecord(end_events[i]);
    }

    // Report benchmark results
    float total_duration = 0;
    for (size_t i = 0; i < iters; i++) {
        cudaEventSynchronize(end_events[i]);
        float duration;  // in ms
        cudaEventElapsedTime(&duration, start_events[i], end_events[i]);
        total_duration += duration;
    }
    float flops = 2 * M * N * K * iters / (total_duration / 1000);
    float bandwidth = ((M * K) * (N / simplegemm::KernelTraits::BLK_N) + (N * K) * (M / simplegemm::KernelTraits::BLK_M) + M * N) * iters * sizeof(ct::half_t) / (total_duration / 1000);
    std::cout << "Time elapse: " << total_duration << "ms" << std::endl;
    std::cout << "TFLOPS: " << flops / 1e12 << std::endl;
    std::cout << "Bandwidth: " << bandwidth / 1e9 << "GB/s" << std::endl;

    // Deallocate A, B, C
    cudaFree(A_ptr);
    cudaFree(B_ptr);
    cudaFree(C_ptr);
    cudaFree(z_ptr);

    return 0;
}