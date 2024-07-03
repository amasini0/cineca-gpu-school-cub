cmake_minimum_required(VERSION 3.16)
project(cub_examples LANGUAGES CUDA)

# Set language standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED OFF)
set(CMAKE_CUDA_EXTENSIONS OFF)

# Set output path for binaries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Fetch cccl from official git repo
include(FetchContent)
FetchContent_Declare(
    cccl
    GIT_REPOSITORY https://github.com/NVIDIA/cccl.git
    GIT_TAG        v2.4.0
    GIT_SHALLOW    1
    EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(cccl)

# Add subdirectory with all examples
add_subdirectory(warp)
add_subdirectory(block)
add_subdirectory(device)
