cmake_minimum_required(VERSION 3.20)
project(cub_examples LANGUAGES CUDA)

# Set language standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# Set cuda architecture
set(CMAKE_CUDA_ARCHITECTURES "80;52")

# Set output path for binaries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Fetch cccl from official git repo
include(FetchContent)
FetchContent_Declare(
    CCCL
    GIT_REPOSITORY https://github.com/NVIDIA/cccl.git
    GIT_TAG        v2.4.0
    GIT_SHALLOW    1
    EXCLUDE_FROM_ALL
)
message("-- Fetching CCCL from https://github.com/NVIDIA/cccl.git")
FetchContent_MakeAvailable(cccl)

# Create stub target for cub
add_library(cccl INTERFACE)
target_link_libraries(cccl INTERFACE CCCL::CCCL)

# Add subdirectories with all examples
add_subdirectory(warp)
add_subdirectory(block)
add_subdirectory(device)
