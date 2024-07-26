#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>

#define CUDA_CHECK(ans) \
    { gpuAssert((ans), __FILE__, __LINE__); }
void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        printf("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    }
}

int main(int argc, char const *argv[]) {
    if (argc != 5) {
        std::cout << "Usage: " << argv[0] << " M N K iters" << std::endl;
        return 1;
    }
    int64_t M = atoi(argv[1]);
    int64_t N = atoi(argv[2]);
    int64_t K = atoi(argv[3]);
    size_t iters = atoi(argv[4]);

    using ElementAccumulator = float;       // <- data type of accumulator
    using ElementInputA = cutlass::half_t;  // <- data type of elements in input matrix A
    using ElementInputB = cutlass::half_t;  // <- data type of elements in input matrix B
    using ElementOutput = cutlass::half_t;  // <- data type of elements in output matrix D

    // Row major for Matrix A, Column Major for Matrix B and Row Major for Matrix C.
    // B is column major because the cutlass interface expects B of shape (K, N) instead of (N, K)
    using LayoutInputA = cutlass::layout::RowMajor;
    using LayoutInputB = cutlass::layout::ColumnMajor;
    using LayoutOutput = cutlass::layout::RowMajor;
    // Use tensor cores
    using MMAOp = cutlass::arch::OpClassTensorOp;
    // Compile for SM75 (Turing)
    using SmArch = cutlass::arch::Sm80;

    using Gemm = cutlass::gemm::device::Gemm<ElementInputA,
                                             LayoutInputA,
                                             ElementInputB,
                                             LayoutInputB,
                                             ElementOutput,
                                             LayoutOutput,
                                             ElementAccumulator,
                                             MMAOp,
                                             SmArch,
                                             cutlass::gemm::GemmShape<128, 128, 64> >;

    Gemm gemm_op;

    // Allocate A, B, C
    cutlass::half_t *A_ptr;
    cutlass::half_t *B_ptr;
    cutlass::half_t *C_ptr;
    CUDA_CHECK(cudaMalloc(&A_ptr, M * K * sizeof(cutlass::half_t)));
    CUDA_CHECK(cudaMalloc(&B_ptr, N * K * sizeof(cutlass::half_t)));
    CUDA_CHECK(cudaMalloc(&C_ptr, M * N * sizeof(cutlass::half_t)));
    cutlass::TensorRef<ElementInputA, LayoutInputA> A({A_ptr, K});
    cutlass::TensorRef<ElementInputB, LayoutInputB> B({B_ptr, K});
    cutlass::TensorRef<ElementOutput, LayoutOutput> C({C_ptr, N});

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

    // Start benchmark
    int l2_cache_size;
    CUDA_CHECK(cudaDeviceGetAttribute(&l2_cache_size, cudaDevAttrL2CacheSize, 0));
    void *z_ptr;
    CUDA_CHECK(cudaMalloc(&z_ptr, l2_cache_size));
    for (size_t i = 0; i < iters; i++) {
        CUDA_CHECK(cudaMemset(z_ptr, 0, l2_cache_size));  // Flush L2 cache
        CUDA_CHECK(cudaEventRecord(start_events[i]));
        cutlass::Status status = gemm_op(
            {{static_cast<int>(M), static_cast<int>(N), static_cast<int>(K)}, A, B, C, C});
        if (status != cutlass::Status::kSuccess) {
            printf("Status: %s\n", cutlassGetStatusString(status));
            exit(1);
        }
        CUDA_CHECK(cudaEventRecord(end_events[i]));
    }
    cudaFree(z_ptr);

    // Report benchmark results
    float total_duration = 0;
    for (size_t i = 0; i < iters; i++) {
        CUDA_CHECK(cudaEventSynchronize(start_events[i]));
        CUDA_CHECK(cudaEventSynchronize(end_events[i]));
        float duration;  // in ms
        CUDA_CHECK(cudaEventElapsedTime(&duration, start_events[i], end_events[i]));
        total_duration += duration;
    }
    float flops = 2 * M * N * K * iters / (total_duration / 1000);
    std::cout << "Time elapse: " << total_duration << "ms" << std::endl;
    std::cout << "TFLOPS: " << flops / 1e12 << std::endl;

    // Deallocate A, B, C
    CUDA_CHECK(cudaFree(A_ptr));
    CUDA_CHECK(cudaFree(B_ptr));
    CUDA_CHECK(cudaFree(C_ptr));

    return 0;
}
