#pragma once

#include <cute/tensor.hpp>

namespace ct = cute;

// Define some useful aliases
using ct::_;
using ct::Int;
template <typename T>
using Gmem = ct::ViewEngine<ct::gmem_ptr<T *>>;
template <typename T>
using Smem = ct::ViewEngine<ct::smem_ptr<T *>>;

#define CUDA_CHECK(ans) \
    { gpuAssert((ans), __FILE__, __LINE__); }
CUTE_HOST_DEVICE void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        printf("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    }
}

namespace simplegemm {

template <typename GemmConfig>
struct SmemStorage {
    ct::array_aligned<ct::half_t, ct::cosize_v<typename GemmConfig::SmemLayoutA>> a;
    ct::array_aligned<ct::half_t, ct::cosize_v<typename GemmConfig::SmemLayoutB>> b;
};

template <typename GemmConfig, typename LayoutC>
struct SmemGemm {
   private:
    using StrideC = std::decay_t<decltype(LayoutC{}.stride())>;
    using LayoutBlkC = ct::Layout<ct::Shape<Int<GemmConfig::BLK_M>, Int<GemmConfig::BLK_N>>, StrideC>;
    ct::Tensor<Gmem<ct::half_t>, LayoutBlkC> &C;
    typename GemmConfig::TiledMMA tiled_mma;
    typename GemmConfig::SmemCopyA smem_tiled_copy_A;
    typename GemmConfig::SmemCopyB smem_tiled_copy_B;

    decltype(tiled_mma.get_thread_slice(0u)) thread_mma;
    decltype(thread_mma.partition_fragment_C(C)) C_frag;

   public:
    __device__ SmemGemm(ct::Tensor<Gmem<ct::half_t>, LayoutBlkC> &C_)
        : C(C_),
          thread_mma(tiled_mma.get_thread_slice(threadIdx.x)),
          C_frag(thread_mma.partition_fragment_C(C)) {
        ct::clear(C_frag);
    }

    // Perform Smem GEMM: C += A @ B
    __device__ void operator()(
        const ct::Tensor<Smem<ct::half_t>, typename GemmConfig::SmemLayoutBlkA> &sA,
        const ct::Tensor<Smem<ct::half_t>, typename GemmConfig::SmemLayoutBlkB> &sB) {
        // Allocate registers distributed across threads to store operands
        auto A_frag = thread_mma.partition_fragment_A(sA);
        auto B_frag = thread_mma.partition_fragment_B(sB);

        // Load A and B from smem to registers (distributed across threads)
        auto thr_copy_A = smem_tiled_copy_A.get_thread_slice(threadIdx.x);
        auto sA_to_rA_src = thr_copy_A.partition_S(sA);   // COPY_V, COPY_M, COPY_K
        auto sA_to_rA_dst = thr_copy_A.retile_D(A_frag);  // COPY_V, COPY_M, COPY_K
        ct::copy(smem_tiled_copy_A, sA_to_rA_src, sA_to_rA_dst);

        auto thr_copy_B = smem_tiled_copy_B.get_thread_slice(threadIdx.x);
        auto sB_to_rB_src = thr_copy_B.partition_S(sB);   // COPY_V, COPY_N, COPY_K
        auto sB_to_rB_dst = thr_copy_B.retile_D(B_frag);  // COPY_V, COPY_N, COPY_K
        ct::copy(smem_tiled_copy_B, sB_to_rB_src, sB_to_rB_dst);

        // Perform GEMM
        ct::gemm(tiled_mma, A_frag, B_frag, C_frag);

        // Wait until all threads have finished using sA and sB
        __syncthreads();
    }

    // Write back result to gmem
    __device__ void write_back() {
        auto C_frag_out = thread_mma.partition_C(C);  // Corresponding location in output tensor
        ct::copy(C_frag, C_frag_out);
        ct::cp_async_wait<0>();
    }
};

template <typename T, typename SrcLayout, typename DstLayout, typename TiledCopy>
__device__ void load_block_from_gmem_to_smem(
    const ct::Tensor<Gmem<T>, SrcLayout> &src,
    const ct::Tensor<Smem<T>, DstLayout> &dst,
    TiledCopy tiled_copy) {
    auto thread_copy = tiled_copy.get_thread_slice(threadIdx.x);
    auto src_frag = thread_copy.partition_S(src);
    auto dst_frag = thread_copy.partition_D(dst);
    ct::copy(tiled_copy, src_frag, dst_frag);
}

// Reordering the block access pattern helps to improve L2 cache hit rate.
// Triton's doc for matmul has a nice explanation: https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
// For m = 3, n = 4, group_size_m = 2, produces the coordiantes in the following order:
//  |  1 |  3 |  5 |  7 |
//  |  2 |  4 |  6 |  8 |
//  |  9 | 10 | 11 | 12 |
__device__ std::tuple<int, int> threadblock_swizzle(int idx, int m, int n) {
    // We choose group_size_m = sqrt(num_sms) to maximize L2 cache hit rate
    int num_sms;
    CUDA_CHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0));
    int group_size_m = std::sqrt(num_sms);
    int blocks_per_group = group_size_m * n;
    int first_block_idx_m = (idx / blocks_per_group) * group_size_m;
    group_size_m = min(m - first_block_idx_m, group_size_m);  // Min to handle edge case of m % group_size_m != 0
    int block_idx_m = first_block_idx_m + (idx % group_size_m);
    int block_idx_n = (idx % blocks_per_group) / group_size_m;
    return std::make_tuple(block_idx_m, block_idx_n);
}

// Main kernel
template <typename GemmConfig, typename LayoutA, typename LayoutB, typename LayoutC>
__global__ void gemm_kernel(
    ct::Tensor<Gmem<ct::half_t>, LayoutA> A,
    ct::Tensor<Gmem<ct::half_t>, LayoutB> B,
    ct::Tensor<Gmem<ct::half_t>, LayoutC> C) {
    // Threadblock-level paratitioning
    auto [block_idx_m, block_idx_n] = threadblock_swizzle(blockIdx.x, ct::size<0>(A) / GemmConfig::BLK_M, ct::size<0>(B) / GemmConfig::BLK_N);
    auto block_shape_A = ct::Shape<Int<GemmConfig::BLK_M>, Int<GemmConfig::BLK_K>>{};
    auto block_shape_B = ct::Shape<Int<GemmConfig::BLK_N>, Int<GemmConfig::BLK_K>>{};
    auto block_shape_C = ct::Shape<Int<GemmConfig::BLK_M>, Int<GemmConfig::BLK_N>>{};
    auto A_blk = ct::local_tile(A, block_shape_A, ct::make_coord(block_idx_m, _));            // BLK_M, BLK_K, N_BLK_K
    auto B_blk = ct::local_tile(B, block_shape_B, ct::make_coord(block_idx_n, _));            // BLK_N, BLK_K, N_BLK_K
    auto C_blk = ct::local_tile(C, block_shape_C, ct::make_coord(block_idx_m, block_idx_n));  // BLK_M, BLK_N

    // Allocate shared memory for the operands
    extern __shared__ char smem_raw[];
    auto smem_storage = reinterpret_cast<SmemStorage<GemmConfig> *>(smem_raw);
    typename GemmConfig::SmemLayoutA smem_layout_A;
    typename GemmConfig::SmemLayoutB smem_layout_B;
    auto sA = ct::make_tensor(ct::make_smem_ptr(smem_storage->a.data()), smem_layout_A);
    auto sB = ct::make_tensor(ct::make_smem_ptr(smem_storage->b.data()), smem_layout_B);

    typename GemmConfig::GmemCopyA gmem_copy_A;
    typename GemmConfig::GmemCopyB gmem_copy_B;
    SmemGemm<GemmConfig, LayoutC> smem_gemm(C_blk);

    int64_t k_load = 0;  // Index of the next block to load
    int64_t k_read = 0;  // Index of the next block to read
    size_t N_BLK_K = ct::size<2>(A_blk);

    auto load_next_stage = [&]() {
        load_block_from_gmem_to_smem(A_blk(_, _, k_load), sA(_, _, k_load % GemmConfig::NumStages), gmem_copy_A);
        load_block_from_gmem_to_smem(B_blk(_, _, k_load), sB(_, _, k_load % GemmConfig::NumStages), gmem_copy_B);
        k_load++;
        ct::cp_async_fence();
    };

    auto consume_next_stage = [&]() {
        ct::cp_async_wait<GemmConfig::NumStages - 1>();
        smem_gemm(sA(_, _, k_read % GemmConfig::NumStages), sB(_, _, k_read % GemmConfig::NumStages));
        k_read++;
    };

    // Main loop
    // 1. Fill pipeline by issuing loads of A and B blocks from gmem to smem
    CUTE_UNROLL
    for (size_t i = 0; i < GemmConfig::NumStages; i++) {
        load_next_stage();
    }

    // 2. Main loop: Compute GEMM on the k-th block, then issue loads for the (k+NumStages)-th block
    CUTE_UNROLL
    for (size_t i = 0; i < N_BLK_K - GemmConfig::NumStages; i++) {
        consume_next_stage();
        load_next_stage();
    }

    // 3. Consume the remaining blocks
    CUTE_UNROLL
    for (size_t i = 0; i < GemmConfig::NumStages; i++) {
        consume_next_stage();
        ct::cp_async_fence();  // Empty fence to ensure there are always `NumStages` fences
    }

    smem_gemm.write_back();
}

// Host interface
template <typename GemmConfig, typename LayoutA, typename LayoutB, typename LayoutC>
void gemm(
    const ct::Tensor<Gmem<ct::half_t>, LayoutA> &A,
    const ct::Tensor<Gmem<ct::half_t>, LayoutB> &B,
    const ct::Tensor<Gmem<ct::half_t>, LayoutC> &C) {
    assert(ct::size<0>(A) == ct::size<0>(C));  // M
    assert(ct::size<0>(B) == ct::size<1>(C));  // N
    assert(ct::size<1>(A) == ct::size<1>(B));  // K
    int64_t M = ct::size<0>(A);
    int64_t N = ct::size<0>(B);
    int64_t K = ct::size<1>(A);

    // We don't handle predication yet
    assert(M % GemmConfig::BLK_M == 0);
    assert(N % GemmConfig::BLK_N == 0);
    assert(K % GemmConfig::BLK_K == 0);
    dim3 block_dim(M / GemmConfig::BLK_M * N / GemmConfig::BLK_N);
    dim3 thread_dim(GemmConfig::NumThreads);
    size_t smem_size = sizeof(SmemStorage<GemmConfig>);

    CUDA_CHECK(cudaFuncSetAttribute(gemm_kernel<GemmConfig, LayoutA, LayoutB, LayoutC>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    gemm_kernel<GemmConfig><<<block_dim, thread_dim, smem_size>>>(A, B, C);
}
}  // namespace simplegemm
