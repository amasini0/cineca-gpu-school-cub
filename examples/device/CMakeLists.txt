# Add binary for each example
add_executable(device_partition partition.cu)
target_link_libraries(device_partition PRIVATE cccl)

add_executable(device_for for_each.cu)
target_link_libraries(device_for PRIVATE cccl)

add_executable(device_seg_reduce seg_reduce.cu)
target_link_libraries(device_seg_reduce PRIVATE cccl)

add_executable(device_select select.cu)
target_link_libraries(device_select PRIVATE cccl)
