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
template <typename T>
struct KernelTraits {
   public:
    static constexpr int NumThreads = 128;  // 4 warps
    static constexpr int AccessSizeBits = 128;
    static constexpr int ElemsPerLoad = AccessSizeBits / ct::sizeof_bits_v<T>;
    static constexpr int BlockSize = NumThreads * ElemsPerLoad;

    using LayoutA = ct::Layout<ct::Shape<int64_t>>;
    using LayoutB = ct::Layout<ct::Shape<int64_t>>;
    using BlockShapeA = ct::Shape<Int<BlockSize>>;
    using BlockShapeB = ct::Shape<Int<BlockSize>>;
    using LayoutBlkA = ct::Layout<ct::Shape<Int<BlockSize>, int64_t>, ct::Stride<Int<1>, Int<BlockSize>>>;
    using LayoutBlkB = ct::Layout<ct::Shape<Int<BlockSize>, int64_t>, ct::Stride<Int<1>, Int<BlockSize>>>;
    using ValLayout = ct::Layout<ct::Shape<Int<ElemsPerLoad>>>;

    // The copy atom for gmem -> smem (read A/B) or rmem -> gmem (store C).
    using GmemCopyAtom = ct::Copy_Atom<ct::AutoVectorizingCopyWithAssumedAlignment<AccessSizeBits>, T>;
};

// Main kernel
template <typename T>
__global__ void relu_kernel(
    ct::Tensor<Gmem<T>, typename KernelTraits<T>::LayoutA> A,
    ct::Tensor<Gmem<T>, typename KernelTraits<T>::LayoutB> B) {
    typename KernelTraits<T>::BlockShapeA block_shape_A;
    typename KernelTraits<T>::BlockShapeB block_shape_B;
    auto A_blk = ct::local_tile(A, block_shape_A, blockIdx.x);  // BlockSize
    auto B_blk = ct::local_tile(B, block_shape_B, blockIdx.x);  // BlockSize

    typename KernelTraits<T>::ValLayout val_layout;
    auto gA = ct::local_tile(A_blk, val_layout, threadIdx.x);
    auto gB = ct::local_tile(B_blk, val_layout, threadIdx.x);
    auto rA = ct::make_tensor_like(gA);

    typename KernelTraits<T>::GmemCopyAtom gmem_copy;
    ct::copy(gmem_copy, gA, rA);

    // Main loop
    for (size_t i = 0; i < ct::size(rA); i++) {
        rA(i) = rA(i) > 0 ? rA(i) : 0;
    }

    ct::copy(gmem_copy, rA, gB);
}

// Host interface
template <typename T, typename LayoutA, typename LayoutB>
void relu(
    const ct::Tensor<Gmem<T>, LayoutA> &A,
    const ct::Tensor<Gmem<T>, LayoutB> &B) {
    assert(ct::shape(A) == ct::shape(B));
    auto A_flat = ct::make_tensor(ct::make_gmem_ptr(A.data()), ct::make_shape(static_cast<int64_t>(ct::size(A))));
    auto B_flat = ct::make_tensor(ct::make_gmem_ptr(B.data()), ct::make_shape(static_cast<int64_t>(ct::size(B))));

    dim3 block_dim(ct::ceil_div(ct::size(A_flat), KernelTraits<T>::BlockSize));
    dim3 thread_dim(KernelTraits<T>::NumThreads);

    relu_kernel<T><<<block_dim, thread_dim>>>(A_flat, B_flat);
}
}  // namespace simplegemm