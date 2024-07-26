#include <cute/tensor.hpp>

namespace ct = cute;

using ct::_;
using ct::Int;

namespace simplegemm {
// GEMM configuration class: Handles the compile-time computation of the kernel parameters.
struct GemmConfigSm80 {
   public:
    // 128x128x64 blocks seems to be a good default
    static constexpr int64_t BLK_M = 128;
    static constexpr int64_t BLK_N = 128;
    static constexpr int64_t BLK_K = 64;
    static constexpr int64_t NumThreads = 128;  // 4 warps
    static constexpr int64_t NumStages = 2;

   private:
    static constexpr int AccessSizeBits = 128;
    static constexpr int ElemsPerLoad = AccessSizeBits / ct::sizeof_bits_v<ct::half_t>;
    static constexpr int SmemAtomInner = std::min(64, static_cast<int>(BLK_K));
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
    using SmemLayoutA = decltype(ct::tile_to_shape(SmemLayoutAtom{}, ct::Shape<Int<BLK_M>, Int<BLK_K>, Int<NumStages>>{}));
    using SmemLayoutB = decltype(ct::tile_to_shape(SmemLayoutAtom{}, ct::Shape<Int<BLK_N>, Int<BLK_K>, Int<NumStages>>{}));
    using SmemLayoutBlkA = decltype(SmemLayoutA{}(_, _, 0));
    using SmemLayoutBlkB = decltype(SmemLayoutB{}(_, _, 0));

   private:
    // The copy atom for gmem -> smem (read A/B) or rmem -> gmem (store C).
    using GmemCopyAtom = ct::Copy_Atom<ct::SM80_CP_ASYNC_CACHEALWAYS<ct::uint128_t>, ct::half_t>;
    // The thread layout for one tile of the gmem -> smem copy.
    using GmemCopyThreadLayoutA = ct::Layout<ct::Shape<Int<NumThreads / ThreadsPerRow>, Int<ThreadsPerRow>>,
                                             ct::Stride<Int<ThreadsPerRow>, Int<1>>>;
    // The value layout for each thread in the gmem -> smem copy.
    using GmemCopyValLayoutA = ct::Layout<ct::Shape<Int<1>, Int<ElemsPerLoad>>>;

   public:
    // Tiled copy of A/B from gmem -> smem
    using GmemCopyA = decltype(ct::make_tiled_copy(GmemCopyAtom{},
                                                   GmemCopyThreadLayoutA{},
                                                   GmemCopyValLayoutA{}));
    using GmemCopyB = GmemCopyA;

   private:
    // The atom of the smem -> rmem copy for A/B. Loads 4 8x8 matrices (distributed across threads) at a time.
    using SmemCopyAtom = ct::Copy_Atom<ct::SM75_U32x4_LDSM_N, ct::half_t>;
    // The atom for the MMA operation. Each atom is a warp-wise instruction that computes a 16x8x16 mma (with tensor cores).
    using MmaAtom = ct::MMA_Atom<ct::SM80_16x8x16_F32F16F16F32_TN>;
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
    using SmemCopyA = decltype(ct::make_tiled_copy_A(SmemCopyAtom{}, TiledMMA{}));
    // Tiled copy of B from smem -> rmem
    using SmemCopyB = decltype(ct::make_tiled_copy_B(SmemCopyAtom{}, TiledMMA{}));
};
}  // namespace simplegemm
