cmake_minimum_required(VERSION 3.18)

project(nanoGEMM)
enable_language(CUDA CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

add_executable(gemm gemm.cu)
set_target_properties(gemm PROPERTIES CUDA_ARCHITECTURES 75)
target_compile_options(
    gemm PRIVATE
    -Wno-unused-function
    --expt-relaxed-constexpr
)
target_include_directories(
    gemm PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/cutlass/include
)
add_test(gemm ${CMAKE_CURRENT_BINARY_DIR}/gemm)

add_executable(reference reference.cu)
set_target_properties(reference PROPERTIES CUDA_ARCHITECTURES 75)
target_compile_options(
    reference PRIVATE
    -Wno-unused-function
    --expt-relaxed-constexpr
)
target_include_directories(
    reference PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/cutlass/include
)
add_test(reference ${CMAKE_CURRENT_BINARY_DIR}/reference)
