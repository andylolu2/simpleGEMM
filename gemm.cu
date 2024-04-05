#include <cuda_runtime.h>
#include <cute/tensor.hpp>

namespace ct = cute;

// Define some useful aliases
using ct::_;
using ct::Int;
template <typename T>
using Gmem = ct::ViewEngine<ct::gmem_ptr<T *>>;
template <typename T>
using Smem = ct::ViewEngine<ct::smem_ptr<T *>>;

// GEMM configuration class: Handles the compile-time computation of the kernel parameters.
template <int BLK_M_, int BLK_N_, int BLK_K_, int GroupSizeM_>
struct KernelTraits
{
public:
    static constexpr int BLK_M = BLK_M_;
    static constexpr int BLK_N = BLK_N_;
    static constexpr int BLK_K = BLK_K_;
    static constexpr int GroupSizeM = GroupSizeM_;
    static constexpr int NumThreads = 128; // 4 warps

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
    static constexpr int AccessSizeBits = ct::sizeof_bits_v<ct::uint128_t>;
    static constexpr int ElemsPerLoad = AccessSizeBits / ct::sizeof_bits_v<ct::half_t>;
    static constexpr int SmemAtomInner = 64;
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
    // Tiled copy of A from gmem to smem
    using GmemTiledCopyA = decltype(ct::make_tiled_copy(GmemCopyAtom{},
                                                        GmemCopyThreadLayoutA{},
                                                        GmemCopyValLayoutA{}));
    // Tiled copy of B from gmem to smem
    using GmemTiledCopyB = GmemTiledCopyA;
    // Copy atom of C from rmem to gmem
    using GmemCopyC = GmemCopyAtom;

private:
    // The atom for the MMA operation.
    // Each warp computes a 16x8x8 mma (with tensor cores).
    // We have 128 threads, so we use 4 warps laid out in 2x2x1 to compute 32x16x8 mma.
    // Each thread additional holds more than one element (laid out in 1x2x2).
    // Final size is 32x32x16. (Each warp computes 16x16x16, so each input have be loaded with one m8n8.x4 LDSM instruction)
    using SmemCopyAtom = ct::Copy_Atom<ct::SM75_U32x4_LDSM_N, ct::half_t>;
    using MmaAtom = ct::MMA_Atom<ct::SM75_16x8x8_F32F16F16F32_TN>;
    using MmaThreadLayout = ct::Layout<ct::Shape<Int<2>, Int<2>, Int<1>>>; // 2x2x1 thread group
    using MmaTiledShape = ct::Tile<Int<32>, Int<32>, Int<16>>;             // 1x2x2 value group
    CUTE_STATIC_ASSERT(ct::size_v<MmaThreadLayout> *ct::size_v<MmaAtom::ThrID> == NumThreads);

public:
    // Tiled mma operation
    using TiledMMA = ct::TiledMMA<MmaAtom, MmaThreadLayout, MmaTiledShape>;
    // Tiled copy of A from smem to rmem
    using SmemTiledCopyA = decltype(ct::make_tiled_copy_A(SmemCopyAtom{}, TiledMMA{}));
    // Tiled copy of B from smem to rmem
    using SmemTiledCopyB = SmemTiledCopyA;
};

template <typename KernelTraits, typename LayoutGaSrc, typename LayoutGaDst, typename LayoutGbSrc,
          typename LayoutGbDst>
__device__ void gemm_thread(
    ct::Tensor<Gmem<ct::half_t>, LayoutGaSrc> &gA_to_sA_src,
    ct::Tensor<Smem<ct::half_t>, LayoutGaDst> &gA_to_sA_dst,
    ct::Tensor<Gmem<ct::half_t>, LayoutGbSrc> &gB_to_sB_src,
    ct::Tensor<Smem<ct::half_t>, LayoutGbDst> &gB_to_sB_dst,
    ct::Tensor<Smem<ct::half_t>, typename KernelTraits::SmemLayoutA> &sA,
    ct::Tensor<Smem<ct::half_t>, typename KernelTraits::SmemLayoutB> &sB,
    ct::Tensor<Gmem<ct::half_t>, typename KernelTraits::LayoutBlkC> &C_blk)
{
    typename KernelTraits::GmemTiledCopyA gmem_tiled_copy_A;
    typename KernelTraits::GmemTiledCopyB gmem_tiled_copy_B;
    typename KernelTraits::GmemCopyC gmem_copy_C;
    typename KernelTraits::TiledMMA tiled_mma;

    // Fragments of the warp-level tiles (distributed across threads' registers)
    // They will be inputs/outputs of the warp-level GEMM
    auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
    auto rA = thr_mma.partition_fragment_A(sA);    // MMA, MMA_M, MMA_K
    auto rB = thr_mma.partition_fragment_B(sB);    // MMA, MMA_N, MMA_K
    auto rC = thr_mma.partition_fragment_C(C_blk); // MMA, MMA_M, MMA_N
    auto gC = thr_mma.partition_C(C_blk);          // Corresponding fragment in gmem to write back
    ct::clear(rC);

    // Fragments for smem -> rmem copy
    typename KernelTraits::SmemTiledCopyA smem_tiled_copy_A;
    auto thr_copy_A = smem_tiled_copy_A.get_thread_slice(threadIdx.x);
    auto sA_to_rA_src = thr_copy_A.partition_S(sA); // COPY_V, COPY_M, COPY_K
    auto sA_to_rA_dst = thr_copy_A.retile_D(rA);    // COPY_V, COPY_M, COPY_K
    typename KernelTraits::SmemTiledCopyB smem_tiled_copy_B;
    auto thr_copy_B = smem_tiled_copy_B.get_thread_slice(threadIdx.x);
    auto sB_to_rB_src = thr_copy_B.partition_S(sB); // COPY_V, COPY_N, COPY_K
    auto sB_to_rB_dst = thr_copy_B.retile_D(rB);    // COPY_V, COPY_N, COPY_K

    // Main loop
    for (size_t k_blk = 0; k_blk < ct::size<3>(gA_to_sA_src); k_blk++)
    {
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
    ct::copy(gmem_copy_C, rC, gC);
}

template <typename KernelTraits>
__device__ void gemm_threadblock(
    ct::Tensor<Gmem<ct::half_t>, typename KernelTraits::LayoutBlkA> &A_blk,
    ct::Tensor<Gmem<ct::half_t>, typename KernelTraits::LayoutBlkB> &B_blk,
    ct::Tensor<Gmem<ct::half_t>, typename KernelTraits::LayoutBlkC> &C_blk)
{
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
    auto gA_to_sA_src = gmem_thr_copy_A.partition_S(A_blk); // COPY_V, COPY_M, COPY_K, N_BLK_K
    auto gA_to_sA_dst = gmem_thr_copy_A.partition_D(sA);    // COPY_V, COPY_M, COPY_K
    auto gB_to_sB_src = gmem_thr_copy_B.partition_S(B_blk); // COPY_V, COPY_N, COPY_K, N_BLK_K
    auto gB_to_sB_dst = gmem_thr_copy_B.partition_D(sB);    // COPY_V, COPY_N, COPY_K

    gemm_thread<KernelTraits>(gA_to_sA_src, gA_to_sA_dst, gB_to_sB_src, gB_to_sB_dst, sA, sB,
                              C_blk);
}

template <typename KernelTraits>
__global__ void gemm_kernel(
    ct::Tensor<Gmem<ct::half_t>, typename KernelTraits::LayoutA> A,
    ct::Tensor<Gmem<ct::half_t>, typename KernelTraits::LayoutB> B,
    ct::Tensor<Gmem<ct::half_t>, typename KernelTraits::LayoutC> C)
{
    using BlockShapeA = typename KernelTraits::BlockShapeA;
    using BlockShapeB = typename KernelTraits::BlockShapeB;
    using BlockShapeC = typename KernelTraits::BlockShapeC;

    auto A_blk_all = ct::tiled_divide(A, BlockShapeA{}); // (BLK_M, BLK_K), N_BLK_M, N_BLK_K
    auto B_blk_all = ct::tiled_divide(B, BlockShapeB{}); // (BLK_N, BLK_K), N_BLK_N, N_BLK_K
    auto C_blk_all = ct::tiled_divide(C, BlockShapeC{}); // (BLK_M, BLK_N), N_BLK_M, N_BLK_N

    // Threadblock swizzling
    // Reordering the block access pattern helps to improve L2 cache hit rate.
    // Triton's doc for matmul has a nice explanation: https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
    int N_BLK_M = ct::size<1>(A_blk_all);
    int N_BLK_N = ct::size<1>(B_blk_all);
    int blocks_per_group = KernelTraits::GroupSizeM * N_BLK_N;
    int first_block_idx_m = (blockIdx.x / blocks_per_group) * KernelTraits::GroupSizeM;
    int group_size_m = min(N_BLK_M - first_block_idx_m, KernelTraits::GroupSizeM); // Edge case
    int block_idx_m = first_block_idx_m + (blockIdx.x % group_size_m);
    int block_idx_n = (blockIdx.x % blocks_per_group) / group_size_m;

    auto A_blk = ct::flatten(A_blk_all(_, block_idx_m, _));           // BLK_M, BLK_K, N_BLK_K
    auto B_blk = ct::flatten(B_blk_all(_, block_idx_n, _));           // BLK_N, BLK_K, N_BLK_K
    auto C_blk = ct::flatten(C_blk_all(_, block_idx_m, block_idx_n)); // BLK_M, BLK_N

    gemm_threadblock<KernelTraits>(A_blk, B_blk, C_blk);
}

template <typename KernelTraits>
void gemm(
    ct::Tensor<Gmem<ct::half_t>, typename KernelTraits::LayoutA> &A,
    ct::Tensor<Gmem<ct::half_t>, typename KernelTraits::LayoutB> &B,
    ct::Tensor<Gmem<ct::half_t>, typename KernelTraits::LayoutC> &C)
{
    assert(ct::size<0>(A) == ct::size<0>(C)); // M
    assert(ct::size<0>(B) == ct::size<1>(C)); // N
    assert(ct::size<1>(A) == ct::size<1>(B)); // K
    int64_t M = ct::size<0>(A);
    int64_t N = ct::size<0>(B);
    int64_t K = ct::size<1>(A);

    // We don't handle predication yet
    assert(M % KernelTraits::BLK_M == 0);
    assert(N % KernelTraits::BLK_N == 0);
    assert(K % KernelTraits::BLK_K == 0);
    dim3 block_dim(ct::ceil_div(M, KernelTraits::BLK_M) * ct::ceil_div(N, KernelTraits::BLK_N));
    dim3 thread_dim(KernelTraits::NumThreads);

    gemm_kernel<KernelTraits><<<block_dim, thread_dim>>>(A, B, C);
}

int main(int argc, char const *argv[])
{
    if (argc < 4)
    {
        std::cout << "Usage: " << argv[0] << " M N K iters" << std::endl;
        return 1;
    }
    int64_t M = atoi(argv[1]);
    int64_t N = atoi(argv[2]);
    int64_t K = atoi(argv[3]);
    size_t iters = atoi(argv[4]);

    // Allocate A, B, C
    ct::half_t *A_ptr;
    cudaMalloc((void **)&A_ptr, M * K * sizeof(ct::half_t));
    auto A = ct::make_tensor(ct::make_gmem_ptr(A_ptr), ct::make_layout(ct::make_shape(M, K), ct::GenRowMajor{}));

    ct::half_t *B_ptr;
    cudaMalloc((void **)&B_ptr, N * K * sizeof(ct::half_t));
    auto B = ct::make_tensor(ct::make_gmem_ptr(B_ptr), ct::make_layout(ct::make_shape(N, K), ct::GenRowMajor{}));

    ct::half_t *C_ptr;
    cudaMalloc((void **)&C_ptr, M * N * sizeof(ct::half_t));
    auto C = ct::make_tensor(ct::make_gmem_ptr(C_ptr), ct::make_layout(ct::make_shape(M, N), ct::GenRowMajor{}));

    // Time and benchmark
    cudaEvent_t start_event;
    cudaEvent_t end_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&end_event);

    // Start benchmark
    cudaEventRecord(start_event);
    for (size_t i = 0; i < iters; i++)
    {
        gemm<KernelTraits<128, 128, 64, 6>>(A, B, C);
    }
    cudaEventRecord(end_event);

    // Report benchmark results
    cudaEventSynchronize(end_event);
    float total_duration; // in ms
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
