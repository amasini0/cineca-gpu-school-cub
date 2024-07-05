# Add binary for each example
add_executable(block_adjdiff adjdiff.cu)
target_link_libraries(block_adjdiff PRIVATE cccl)

add_executable(block_histogram histogram.cu)
target_link_libraries(block_histogram PRIVATE cccl)

add_executable(block_shuffle shuffle.cu)
target_link_libraries(block_shuffle PRIVATE cccl)

add_executable(block_rle run_length.cu)
target_link_libraries(block_rle PRIVATE cccl)
