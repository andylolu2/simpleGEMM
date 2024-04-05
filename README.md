# NanoGEMM

This is an *extremely* minimalistic but fast implementation of matrix multiplication in CUDA. The implementation is a single, 300-line file [gemm.cu](gemm.cu) which implements half-precision tensor core matrix multiplication, optimised for Turing (SM75) architecture. 

The implementation builds on top of CuTe from CUTLASS, a low-level interface for tensor manipulation in CUDA. The code is well-commented and is meant to be easily readable (minimal CUDA/C++ background knowledge required) and hackable.

Benchmark against SOTA (see [reference.cu](reference.cu)):
```
$ ./gemm 4096 4096 4096 1000
Time elapse: 6043.59ms
TFLOPS: 22.7413

$ ./gemm 8192 8192 8192 100
Time elapse: 4819.51ms
TFLOPS: 22.8138

$ ./reference 4096 4096 4096 1000
Time elapse: 6040.42ms
TFLOPS: 22.7532

$ ./reference 8192 8192 8192 100
Time elapse: 4657.08ms
TFLOPS: 23.6095
```
> The theoretical maximum for the hardware I used (RTX 2060) is 26 TFLOPS.

## Quick start

> Requires CUDA installed. Checkout https://docs.nvidia.com/cuda/cuda-installation-guide-linux/ for instructions.

Compile the source file:
```bash
nvcc \
    --include-path cutlass/include \
    --generate-code=arch=compute_75,code=[compute_75,sm_75] \
    --expt-relaxed-constexpr \
    -forward-unknown-to-host-compiler \
    -std=c++17 \
    -O3 \
    -o build/gemm \
    gemm.cu
```

And run!
```
$ ./gemm
Usage: ./gemm M N K iters

$ ./gemm 4096 4096 4096 1000
Time elapse: 6043.59ms
TFLOPS: 22.7413
```

You can also build with `CMake` (better option for development):
```bash
$ mkdir build
$ cd build/
$ cmake ..
-- Configuring done
-- Generating done
-- Build files have been written to: /workspaces/nanoGEMM/build
$ make gemm 
Consolidate compiler generated dependencies of target gemm
[ 50%] Building CUDA object CMakeFiles/gemm.dir/gemm.cu.o
[100%] Linking CUDA executable gemm
[100%] Built target gemm
$ ./gemm 
Usage: ./gemm M N K iters
```

## What's missing

The code trades off generality for simplicity:
- Only supports half-precision matmul.
- Assumes (asserts) the inputs are divisible by the block size.
- Assumes the inputs are in row-major layout. (Though you probably only want to row-major layout anyway as other combinations are 10-30% slower.)
- Doesn't do software pipelining. (interleaving global memory load for the next tile with computation.)
- Is only optimal for "normal" problem sizes. For more exotic problem sizes like small-M/N with large-K, are split-K kernel is likely to perform better.