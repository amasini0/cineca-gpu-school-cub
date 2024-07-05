# Add binary for each executable
add_executable(warp_reduce reduce.cu)
target_link_libraries(warp_reduce PRIVATE cccl)

add_executable(warp_scan scan.cu)
target_link_libraries(warp_scan PRIVATE cccl)

add_executable(warp_memory memory.cu)
target_link_libraries(warp_memory PRIVATE cccl)

add_executable(warp_sort sort.cu)
target_link_libraries(warp_sort PRIVATE cccl)
