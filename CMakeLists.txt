cmake_minimum_required(VERSION 3.25.2)

project(simpleGEMM)
enable_language(CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-Wall -Wextra")

function(compile name source_file)
    add_executable(${name} ${source_file})
    set_target_properties(
        ${name} PROPERTIES 
        CUDA_ARCHITECTURES native
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    target_compile_options(
        ${name} PRIVATE
        -Wno-unused-function
        --expt-relaxed-constexpr
        -O3
    )
    target_include_directories(
        ${name} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/cutlass/include
        ${CMAKE_CURRENT_SOURCE_DIR}/cutlass/tools/util/include
    )
endfunction()

compile(main main.cu)
compile(reference test/reference.cu)
compile(correctness test/correctness.cu)
