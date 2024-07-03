add_executable(warp_reduce reduce.cu)
add_executable(warp_scan   scan.cu  )
add_executable(warp_memory memory.cu)
add_executable(warp_sort   sort.cu  )

target_link_libraries(warp_reduce warp_scan warp_memory warp_sort PRIVATE cccl::cub)
