#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>

int main(int argc, char const *argv[]) {
    if (argc != 5) {
        std::cout << "Usage: " << argv[0] << " M N K iters" << std::endl;
        return 1;
    }
    int64_t M = atoi(argv[1]);
    int64_t N = atoi(argv[2]);
    int64_t K = atoi(argv[3]);
    size_t iters = atoi(argv[4]);

    using ElementAccumulator = float;                   // <- data type of accumulator
    using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
    using ElementInputA = cutlass::half_t;              // <- data type of elements in input matrix A
    using ElementInputB = cutlass::half_t;              // <- data type of elements in input matrix B
    using ElementOutput = cutlass::half_t;              // <- data type of elements in output matrix D

    // Row major for Matrix A, Column Major for Matrix B and Row Major for Matrix C.
    // B is column major because the cutlass interface expects B of shape (K, N) instead of (N, K)
    using LayoutInputA = cutlass::layout::RowMajor;
    using LayoutInputB = cutlass::layout::ColumnMajor;
    using LayoutOutput = cutlass::layout::RowMajor;
    // Use tensor cores
    using MMAOp = cutlass::arch::OpClassTensorOp;
    // Compile for SM75 (Turing)
    using SmArch = cutlass::arch::Sm75;

    using Gemm = cutlass::gemm::device::Gemm<ElementInputA,
                                             LayoutInputA,
                                             ElementInputB,
                                             LayoutInputB,
                                             ElementOutput,
                                             LayoutOutput,
                                             ElementAccumulator,
                                             MMAOp,
                                             SmArch>;

    Gemm gemm_op;

    // Allocate A, B, C
    cutlass::half_t *A_ptr;
    cutlass::half_t *B_ptr;
    cutlass::half_t *C_ptr;
    cudaMalloc(&A_ptr, M * K * sizeof(cutlass::half_t));
    cudaMalloc(&B_ptr, N * K * sizeof(cutlass::half_t));
    cudaMalloc(&C_ptr, M * N * sizeof(cutlass::half_t));
    cutlass::TensorRef<ElementInputA, LayoutInputA> A({A_ptr, K});
    cutlass::TensorRef<ElementInputB, LayoutInputB> B({B_ptr, K});
    cutlass::TensorRef<ElementOutput, LayoutOutput> C({C_ptr, N});

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
        cutlass::Status status = gemm_op(
            {{static_cast<int>(M), static_cast<int>(N), static_cast<int>(K)}, A, B, C, C});
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
    float flops = 2 * M * N * K * iters / (total_duration / 1000);
    std::cout << "Time elapse: " << total_duration << "ms" << std::endl;
    std::cout << "TFLOPS: " << flops / 1e12 << std::endl;

    // Deallocate A, B, C
    cudaFree(A_ptr);
    cudaFree(B_ptr);
    cudaFree(C_ptr);

    return 0;
}
