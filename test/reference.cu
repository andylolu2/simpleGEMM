#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>

int main(int argc, char const *argv[]) {
    if (argc != 5) {
        std::cout << "Usage: " << argv[0] << " M N K iters" << std::endl;
        return 1;
    }
    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int K = atoi(argv[3]);
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
    cudaMalloc((void **)&A_ptr, M * K * sizeof(cutlass::half_t));
    cudaMalloc((void **)&B_ptr, N * K * sizeof(cutlass::half_t));
    cudaMalloc((void **)&C_ptr, M * N * sizeof(cutlass::half_t));
    cutlass::TensorRef<ElementInputA, LayoutInputA> A({A_ptr, K});
    cutlass::TensorRef<ElementInputB, LayoutInputB> B({B_ptr, K});
    cutlass::TensorRef<ElementOutput, LayoutOutput> C({C_ptr, N});

    // Time and benchmark
    cudaEvent_t start_event;
    cudaEvent_t end_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&end_event);

    // Start benchmark
    cudaEventRecord(start_event);
    for (size_t i = 0; i < iters; i++) {
        cutlass::Status status = gemm_op({{M, N, K}, A, B, C, C});
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
    cudaFree((void *)A_ptr);
    cudaFree((void *)B_ptr);
    cudaFree((void *)C_ptr);

    return 0;
}
