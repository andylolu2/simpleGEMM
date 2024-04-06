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
    cudaMalloc(&A_ptr, M * K * sizeof(ct::half_t));
    cudaMalloc(&B_ptr, N * K * sizeof(ct::half_t));
    cudaMalloc(&C_ptr, M * N * sizeof(ct::half_t));
    auto A = ct::make_tensor(ct::make_gmem_ptr(A_ptr), ct::make_layout(ct::make_shape(M, K), ct::GenRowMajor{}));
    auto B = ct::make_tensor(ct::make_gmem_ptr(B_ptr), ct::make_layout(ct::make_shape(N, K), ct::GenRowMajor{}));
    auto C = ct::make_tensor(ct::make_gmem_ptr(C_ptr), ct::make_layout(ct::make_shape(M, N), ct::GenRowMajor{}));

    // Time and benchmark
    cudaEvent_t start_event;
    cudaEvent_t end_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&end_event);

    // Start benchmark
    cudaEventRecord(start_event);
    for (size_t i = 0; i < iters; i++) {
        gemm<KernelTraits<128, 128, 64, 6>>(A, B, C);
    }
    cudaEventRecord(end_event);

    // Report benchmark results
    cudaEventSynchronize(end_event);
    float total_duration;  // in ms
    cudaEventElapsedTime(&total_duration, start_event, end_event);
    float tflops = 2 * M * N * K * iters / (total_duration / 1000) / 1E12;
    std::cout << "Time elapse: " << total_duration << "ms" << std::endl;
    std::cout << "TFLOPS: " << tflops << std::endl;

    // Deallocate A, B, C
    cudaFree(A_ptr);
    cudaFree(B_ptr);
    cudaFree(C_ptr);

    return 0;
}