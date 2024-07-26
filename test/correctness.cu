#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/device/gemm.h>
#include <cutlass/util/reference/device/tensor_fill.h>

#include <cute/tensor.hpp>

#include "gemm.cuh"

namespace ct = cute;

int main(int argc, char const *argv[]) {
    if (argc != 4) {
        std::cout << "Usage: " << argv[0] << " M N K" << std::endl;
        return 1;
    }
    int64_t M = atoi(argv[1]);
    int64_t N = atoi(argv[2]);
    int64_t K = atoi(argv[3]);

    // Allocate A, B, C
    cutlass::HostTensor<ct::half_t, cutlass::layout::RowMajor> A_tensor({M, K});
    cutlass::HostTensor<ct::half_t, cutlass::layout::ColumnMajor> B_tensor({K, N});
    cutlass::HostTensor<ct::half_t, cutlass::layout::RowMajor> C_tensor({M, N});
    cutlass::HostTensor<ct::half_t, cutlass::layout::RowMajor> C_ref_tensor({M, N});
    auto A = ct::make_tensor(ct::make_gmem_ptr(A_tensor.device_data()), ct::make_layout(ct::make_shape(M, K), ct::GenRowMajor{}));
    auto B = ct::make_tensor(ct::make_gmem_ptr(B_tensor.device_data()), ct::make_layout(ct::make_shape(N, K), ct::GenRowMajor{}));
    auto C = ct::make_tensor(ct::make_gmem_ptr(C_tensor.device_data()), ct::make_layout(ct::make_shape(M, N), ct::GenRowMajor{}));

    // Fill with random data
    cutlass::reference::device::TensorFillRandomGaussian(A_tensor.device_view(), 0);
    cutlass::reference::device::TensorFillRandomGaussian(B_tensor.device_view(), 1);

    // Test for correctness
    // Ours
    simplegemm::gemm(A, B, C);
    // Reference
    cutlass::reference::device::compute_gemm(
        {static_cast<int>(M), static_cast<int>(N), static_cast<int>(K)},
        1.0f,
        A_tensor.device_ref(),
        B_tensor.device_ref(),
        0.0f,
        C_ref_tensor.device_ref(),
        0.f);
    cudaDeviceSynchronize();

    // Copy output data to host for comparison
    C_tensor.sync_host();
    C_ref_tensor.sync_host();

    // Compare and report metrics
    int64_t rel_err_count = 0;
    int64_t abs_err_count = 0;
    float max_rel_err = 0.0f;
    float max_abs_err = 0.0f;
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            float c = C_tensor.host_ref().at({i, j});
            float c_ref = C_ref_tensor.host_ref().at({i, j});
            float diff = std::abs(c - c_ref);
            float rel = diff / std::abs(c_ref);
            max_abs_err = std::max(max_abs_err, diff);
            max_rel_err = std::max(max_rel_err, rel);
            if (diff > 0.001f) {
                abs_err_count++;
            }
            if (rel > 0.01f) {
                rel_err_count++;
            }
        }
    }
    float rel_err_prop = static_cast<float>(rel_err_count) / static_cast<float>(M * N);
    float abs_err_prop = static_cast<float>(abs_err_count) / static_cast<float>(M * N);
    std::cout << "Max rel err: " << max_rel_err * 100 << "%" << std::endl;
    std::cout << "Rel err prop: " << rel_err_prop * 100 << "%" << std::endl;
    std::cout << "Max abs err: " << max_abs_err << std::endl;
    std::cout << "Abs err prop: " << abs_err_prop * 100 << "%" << std::endl;

    return 0;
}