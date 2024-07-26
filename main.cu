#include <cute/tensor.hpp>

#include "gemm.cuh"
#include "gemm_config_sm75.cuh"
#include "gemm_config_sm80.cuh"

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
    CUDA_CHECK(cudaMalloc(&A_ptr, M * K * sizeof(ct::half_t)));
    CUDA_CHECK(cudaMalloc(&B_ptr, N * K * sizeof(ct::half_t)));
    CUDA_CHECK(cudaMalloc(&C_ptr, M * N * sizeof(ct::half_t)));
    auto A = ct::make_tensor(ct::make_gmem_ptr(A_ptr), ct::make_layout(ct::make_shape(M, K), ct::GenRowMajor{}));
    auto B = ct::make_tensor(ct::make_gmem_ptr(B_ptr), ct::make_layout(ct::make_shape(N, K), ct::GenRowMajor{}));
    auto C = ct::make_tensor(ct::make_gmem_ptr(C_ptr), ct::make_layout(ct::make_shape(M, N), ct::GenRowMajor{}));

    void *z_ptr;
    int l2_cache_size;
    CUDA_CHECK(cudaDeviceGetAttribute(&l2_cache_size, cudaDevAttrL2CacheSize, 0));
    CUDA_CHECK(cudaMalloc(&z_ptr, l2_cache_size));

    // Time and benchmark
    std::vector<cudaEvent_t> start_events;
    std::vector<cudaEvent_t> end_events;
    for (size_t i = 0; i < iters; i++) {
        cudaEvent_t start_event;
        cudaEvent_t end_event;
        CUDA_CHECK(cudaEventCreate(&start_event));
        CUDA_CHECK(cudaEventCreate(&end_event));
        start_events.push_back(start_event);
        end_events.push_back(end_event);
    }

    using GemmConfig = simplegemm::GemmConfigSm80;
    // using GemmConfig = simplegemm::GemmConfigSm75;

    // Start benchmark
    for (size_t i = 0; i < iters; i++) {
        CUDA_CHECK(cudaMemset(z_ptr, 0, l2_cache_size));
        CUDA_CHECK(cudaEventRecord(start_events[i]));
        simplegemm::gemm<GemmConfig>(A, B, C);
        CUDA_CHECK(cudaEventRecord(end_events[i]));
    }

    // Report benchmark results
    float total_duration = 0;
    for (size_t i = 0; i < iters; i++) {
        CUDA_CHECK(cudaEventSynchronize(end_events[i]));
        float duration;  // in ms
        CUDA_CHECK(cudaEventElapsedTime(&duration, start_events[i], end_events[i]));
        total_duration += duration;
    }
    float flops = 2 * M * N * K * iters / (total_duration / 1000);
    float bandwidth = ((M * K) * (N / GemmConfig::BLK_N) + (N * K) * (M / GemmConfig::BLK_M) + M * N) * iters * sizeof(ct::half_t) / (total_duration / 1000);
    std::cout << "Time elapse: " << total_duration << "ms" << std::endl;
    std::cout << "TFLOPS: " << flops / 1e12 << std::endl;
    std::cout << "Bandwidth: " << bandwidth / 1e9 << "GB/s" << std::endl;

    // Deallocate A, B, C
    CUDA_CHECK(cudaFree(A_ptr));
    CUDA_CHECK(cudaFree(B_ptr));
    CUDA_CHECK(cudaFree(C_ptr));
    CUDA_CHECK(cudaFree(z_ptr));

    return 0;
}