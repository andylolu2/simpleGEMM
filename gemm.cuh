#pragma once

#include <cuda_runtime.h>

#include <cute/tensor.hpp>

namespace ct = cute;

// Define some useful aliases
using ct::_;
using ct::Int;
template <typename T>
using Gmem = ct::ViewEngine<ct::gmem_ptr<T *>>;

namespace simplegemm {
// GEMM configuration class: Handles the compile-time computation of the kernel parameters.
// Good default values are hard-coded but they might be tuned to give better performance.
struct KernelTraits {
   public:
    // 128x128x64 blocks seems to be a good default
    static constexpr int BLK_M = 128;
    static constexpr int BLK_N = 128;
    static constexpr int BLK_K = 64;
    static constexpr int GroupSizeM = 6;    // Generally want to choose group size ~= sqrt(no. of SMs).
    static constexpr int NumThreads = 128;  // 4 warps

    // Row-major A, B, C
    using LayoutA = ct::Layout<ct::Shape<int64_t, int64_t>, ct::Stride<int64_t, Int<1>>>;
    using LayoutB = ct::Layout<ct::Shape<int64_t, int64_t>, ct::Stride<int64_t, Int<1>>>;
    using LayoutC = ct::Layout<ct::Shape<int64_t, int64_t>, ct::Stride<int64_t, Int<1>>>;
    using BlockShapeA = ct::Shape<Int<BLK_M>, Int<BLK_K>>;
    using BlockShapeB = ct::Shape<Int<BLK_N>, Int<BLK_K>>;
    using BlockShapeC = ct::Shape<Int<BLK_M>, Int<BLK_N>>;
    using LayoutBlkA = ct::Layout<ct::Shape<Int<BLK_M>, Int<BLK_K>, int64_t>, ct::Stride<int64_t, Int<1>, Int<BLK_K>>>;
    using LayoutBlkB = ct::Layout<ct::Shape<Int<BLK_N>, Int<BLK_K>, int64_t>, ct::Stride<int64_t, Int<1>, Int<BLK_K>>>;
    using LayoutBlkC = ct::Layout<ct::Shape<Int<BLK_M>, Int<BLK_N>>, ct::Stride<int64_t, Int<1>>>;

   private:
    static constexpr int AccessSizeBits = 128;
    static constexpr int ElemsPerLoad = AccessSizeBits / ct::sizeof_bits_v<ct::half_t>;
    static constexpr int SmemAtomInner = std::min(64, BLK_K);
    static constexpr int SmemAtomOuter = ElemsPerLoad;
    static constexpr int ThreadsPerRow = SmemAtomInner / ElemsPerLoad;

    // The layout of one tile of the smem block, will be tiled to fill the entire block.
    // The choice of this layout is important for performance.
    // Swizzling reduces shared memory bank conflicts.
    using SmemLayoutAtom = decltype(ct::composition(ct::Swizzle<3, 3, 3>{},
                                                    ct::Layout<
                                                        ct::Shape<Int<SmemAtomOuter>, Int<SmemAtomInner>>,
                                                        ct::Stride<Int<SmemAtomInner>, Int<1>>>{}));

   public:
    // Layout of each block of A/B in shared memory
    using SmemLayoutA = decltype(ct::tile_to_shape(SmemLayoutAtom{}, BlockShapeA{}));
    using SmemLayoutB = decltype(ct::tile_to_shape(SmemLayoutAtom{}, BlockShapeB{}));

   private:
    // The copy atom for gmem -> smem (read A/B) or rmem -> gmem (store C).
    using GmemCopyAtom = ct::Copy_Atom<
        ct::AutoVectorizingCopyWithAssumedAlignment<AccessSizeBits>, ct::half_t>;
    // The thread layout for one tile of the gmem -> smem copy.
    using GmemCopyThreadLayoutA = ct::Layout<ct::Shape<Int<NumThreads / ThreadsPerRow>, Int<ThreadsPerRow>>,
                                             ct::Stride<Int<ThreadsPerRow>, Int<1>>>;
    // The value layout for each thread in the gmem -> smem copy.
    using GmemCopyValLayoutA = ct::Layout<ct::Shape<Int<1>, Int<ElemsPerLoad>>>;

   public:
    // Tiled copy of A from gmem -> smem
    using GmemTiledCopyA = decltype(ct::make_tiled_copy(GmemCopyAtom{},
                                                        GmemCopyThreadLayoutA{},
                                                        GmemCopyValLayoutA{}));
    // Tiled copy of B from gmem -> smem
    using GmemTiledCopyB = GmemTiledCopyA;
    // Copy atom of C from rmem -> gmem
    using GmemCopyC = GmemCopyAtom;

   private:
    // The atom of the smem -> rmem copy for A/B. Loads 4 8x8 matrices (distributed across threads) at a time.
    using SmemCopyAtom = ct::Copy_Atom<ct::SM75_U32x4_LDSM_N, ct::half_t>;
    // The atom for the MMA operation. Each atom is a warp-wise instruction that computes a 16x8x8 mma (with tensor cores).
    using MmaAtom = ct::MMA_Atom<ct::SM75_16x8x8_F32F16F16F32_TN>;
    // We have 128 threads, so we use 4 warps laid out in 2x2x1.
    using MmaAtomLayout = ct::Layout<ct::Shape<Int<2>, Int<2>, Int<1>>>;
    // We want to use the `ldmatrix.x4.m8n8` instruction which loads 4 8x8 matrices for maximum efficiency.
    // To make the operands A and B divisible into 4 8x8 matrices, we expand the problem size for each warp to 16x16x16.
    // Accounting for the fact that we use 4 warps laid out in 2x2x1, the full tile size is 32x32x16.
    using MmaTiledShape = ct::Tile<Int<32>, Int<32>, Int<16>>;

   public:
    // Tiled mma operation
    using TiledMMA = ct::TiledMMA<MmaAtom, MmaAtomLayout, MmaTiledShape>;
    // Tiled copy of A from smem -> rmem
    using SmemTiledCopyA = decltype(ct::make_tiled_copy_A(SmemCopyAtom{}, TiledMMA{}));
    // Tiled copy of B from smem -> rmem
    using SmemTiledCopyB = decltype(ct::make_tiled_copy_B(SmemCopyAtom{}, TiledMMA{}));
};

__device__ std::tuple<int, int> threadblock_swizzle(int idx, int m, int n, int group_size_m) {
    // Reordering the block access pattern helps to improve L2 cache hit rate.
    // Triton's doc for matmul has a nice explanation: https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
    // For m = 3, n = 4, group_size_m = 2, produces the coordiantes in the following order:
    //  |  1 |  3 |  5 |  7 |
    //  |  2 |  4 |  6 |  8 |
    //  |  9 | 10 | 11 | 12 |
    int blocks_per_group = group_size_m * n;
    int first_block_idx_m = (idx / blocks_per_group) * group_size_m;
    group_size_m = min(m - first_block_idx_m, group_size_m);  // Min to handle edge case of m % group_size_m != 0
    int block_idx_m = first_block_idx_m + (idx % group_size_m);
    int block_idx_n = (idx % blocks_per_group) / group_size_m;
    return std::make_tuple(block_idx_m, block_idx_n);
}

// Main kernel
__global__ void gemm_kernel(
    ct::Tensor<Gmem<ct::half_t>, typename KernelTraits::LayoutA> A,
    ct::Tensor<Gmem<ct::half_t>, typename KernelTraits::LayoutB> B,
    ct::Tensor<Gmem<ct::half_t>, typename KernelTraits::LayoutC> C) {
    // Threadblock-level paratitioning
    auto [block_idx_m, block_idx_n] = threadblock_swizzle(
        blockIdx.x,
        ct::ceil_div(ct::size<0>(A), Int<KernelTraits::BLK_M>{}),
        ct::ceil_div(ct::size<0>(B), Int<KernelTraits::BLK_N>{}),
        KernelTraits::GroupSizeM);
    typename KernelTraits::BlockShapeA block_shape_A;
    typename KernelTraits::BlockShapeB block_shape_B;
    typename KernelTraits::BlockShapeC block_shape_C;
    auto A_blk = ct::local_tile(A, block_shape_A, ct::make_coord(block_idx_m, _));            // BLK_M, BLK_K, N_BLK_K
    auto B_blk = ct::local_tile(B, block_shape_B, ct::make_coord(block_idx_n, _));            // BLK_N, BLK_K, N_BLK_K
    auto C_blk = ct::local_tile(C, block_shape_C, ct::make_coord(block_idx_m, block_idx_n));  // BLK_M, BLK_N

    // Allocate shared memory for the operands
    typename KernelTraits::SmemLayoutA smem_layout_A;
    typename KernelTraits::SmemLayoutB smem_layout_B;
    __shared__ ct::half_t sA_data[ct::cosize_v<decltype(smem_layout_A)>];
    __shared__ ct::half_t sB_data[ct::cosize_v<decltype(smem_layout_B)>];
    auto sA = ct::make_tensor(ct::make_smem_ptr(sA_data), smem_layout_A);
    auto sB = ct::make_tensor(ct::make_smem_ptr(sB_data), smem_layout_B);

    typename KernelTraits::GmemTiledCopyA gmem_tiled_copy_A;
    typename KernelTraits::GmemTiledCopyA gmem_tiled_copy_B;
    auto gmem_thr_copy_A = gmem_tiled_copy_A.get_thread_slice(threadIdx.x);
    auto gmem_thr_copy_B = gmem_tiled_copy_B.get_thread_slice(threadIdx.x);

    // Fragments for gmem -> smem copy
    auto gA_to_sA_src = gmem_thr_copy_A.partition_S(A_blk);  // COPY_V, COPY_M, COPY_K, N_BLK_K
    auto gA_to_sA_dst = gmem_thr_copy_A.partition_D(sA);     // COPY_V, COPY_M, COPY_K
    auto gB_to_sB_src = gmem_thr_copy_B.partition_S(B_blk);  // COPY_V, COPY_N, COPY_K, N_BLK_K
    auto gB_to_sB_dst = gmem_thr_copy_B.partition_D(sB);     // COPY_V, COPY_N, COPY_K

    // Fragments of the warp-level tiles (distributed across threads' registers)
    // They will be inputs/outputs of the warp-level GEMM
    typename KernelTraits::TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
    auto rA = thr_mma.partition_fragment_A(sA);     // MMA, MMA_M, MMA_K
    auto rB = thr_mma.partition_fragment_B(sB);     // MMA, MMA_N, MMA_K
    auto rC = thr_mma.partition_fragment_C(C_blk);  // MMA, MMA_M, MMA_N
    auto gC = thr_mma.partition_C(C_blk);           // Corresponding fragment in gmem to write back
    ct::clear(rC);

    // Fragments for smem -> rmem copy
    typename KernelTraits::SmemTiledCopyA smem_tiled_copy_A;
    auto thr_copy_A = smem_tiled_copy_A.get_thread_slice(threadIdx.x);
    auto sA_to_rA_src = thr_copy_A.partition_S(sA);  // COPY_V, COPY_M, COPY_K
    auto sA_to_rA_dst = thr_copy_A.retile_D(rA);     // COPY_V, COPY_M, COPY_K

    typename KernelTraits::SmemTiledCopyB smem_tiled_copy_B;
    auto thr_copy_B = smem_tiled_copy_B.get_thread_slice(threadIdx.x);
    auto sB_to_rB_src = thr_copy_B.partition_S(sB);  // COPY_V, COPY_N, COPY_K
    auto sB_to_rB_dst = thr_copy_B.retile_D(rB);     // COPY_V, COPY_N, COPY_K

    // Main loop
    for (size_t k_blk = 0; k_blk < ct::size<3>(gA_to_sA_src); k_blk++) {
        // Populate sA and sB by copying gmem -> smem (coorperatively within a threadblock)
        ct::copy(gmem_tiled_copy_A, gA_to_sA_src(_, _, _, k_blk), gA_to_sA_dst);
        ct::copy(gmem_tiled_copy_B, gB_to_sB_src(_, _, _, k_blk), gB_to_sB_dst);
        __syncthreads();

        // Load rA and rB by copying smem -> rmem (coorperatively within a warp)
        ct::copy(smem_tiled_copy_A, sA_to_rA_src, sA_to_rA_dst);
        ct::copy(smem_tiled_copy_B, sB_to_rB_src, sB_to_rB_dst);

        // Perform gemm
        ct::gemm(tiled_mma, rA, rB, rC);
    }

    // Write back result
    typename KernelTraits::GmemCopyC gmem_copy_C;
    ct::copy(gmem_copy_C, rC, gC);
}

// Host interface
void gemm(
    const ct::Tensor<Gmem<ct::half_t>, typename KernelTraits::LayoutA> &A,
    const ct::Tensor<Gmem<ct::half_t>, typename KernelTraits::LayoutB> &B,
    const ct::Tensor<Gmem<ct::half_t>, typename KernelTraits::LayoutC> &C) {
    assert(ct::size<0>(A) == ct::size<0>(C));  // M
    assert(ct::size<0>(B) == ct::size<1>(C));  // N
    assert(ct::size<1>(A) == ct::size<1>(B));  // K
    int64_t M = ct::size<0>(A);
    int64_t N = ct::size<0>(B);
    int64_t K = ct::size<1>(A);

    // We don't handle predication yet
    assert(M % KernelTraits::BLK_M == 0);
    assert(N % KernelTraits::BLK_N == 0);
    assert(K % KernelTraits::BLK_K == 0);
    dim3 block_dim(ct::ceil_div(M, KernelTraits::BLK_M) * ct::ceil_div(N, KernelTraits::BLK_N));
    dim3 thread_dim(KernelTraits::NumThreads);

    gemm_kernel<<<block_dim, thread_dim>>>(A, B, C);
}
}  // namespace simplegemm