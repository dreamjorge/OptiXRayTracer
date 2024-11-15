cmake_minimum_required(VERSION 3.18)
project(OptiXProject CUDA CXX)

# Set CUDA and OptiX paths
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
find_package(CUDA REQUIRED)
set(OPTIX_ROOT /usr/local/optix)
include_directories(${OPTIX_ROOT}/include)

# Define the target
add_executable(optix_example main.cu)

# Specify CUDA properties
set_target_properties(optix_example PROPERTIES
    CUDA_STANDARD 11
    CUDA_SEPARABLE_COMPILATION ON
)

# Link OptiX library
target_link_libraries(optix_example PRIVATE ${OPTIX_ROOT}/lib64/liboptix.so)
