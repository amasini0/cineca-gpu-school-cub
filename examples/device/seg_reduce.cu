#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#include <cuda_runtime.h>
#include <cub/cub.cuh>

constexpr int num_segments = 4;

int main() {
    // Useful values
    constexpr int num_items = 100;

    // Allocate host vector
    std::vector<int> items(num_items);
    std::vector<int> offsets(num_segments + 1);
    std::vector<int> sums(num_segments);
    std::vector<int> mins(num_segments);

    // Fill input vector
    std::iota(items.begin(), items.end(), 1);

    // Fill offsets
    int segment_length = num_items / num_segments;
    for (int i = 0; i < num_segments; ++i) {
        offsets[i] = i * segment_length;
    }
    offsets[num_segments] = num_items;

    // Allocate arrays on device
    void *p_items, *p_offsets, *p_sums, *p_mins;
    cudaMalloc(&p_items, num_items*sizeof(int));
    cudaMalloc(&p_offsets, (num_segments + 1) * sizeof(int));
    cudaMalloc(&p_sums, num_segments*sizeof(int));
    cudaMalloc(&p_mins, num_segments*sizeof(int));
    int *d_items = static_cast<int*>(p_items);
    int *d_offsets = static_cast<int*>(p_offsets);
    int *d_sums  = static_cast<int*>(p_sums);
    int *d_mins  = static_cast<int*>(p_mins);

    // Copy items to device
    cudaMemcpy(d_items, items.data(), num_items*sizeof(int),  cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, offsets.data(), (num_segments + 1) * sizeof(int), cudaMemcpyHostToDevice);

    // SUM --------------------------------------------------- //
    // Determine temporary device storage requirements
    void* p_temp_storage_sum = nullptr;
    size_t temp_storage_sum_bytes = 0;
    cub::DeviceSegmentedReduce::Sum(
        p_temp_storage_sum, temp_storage_sum_bytes, 
        d_items, d_sums, num_segments, d_offsets, d_offsets + 1);

    // Allocate required temporary storage
    cudaMalloc(&p_temp_storage_sum, temp_storage_sum_bytes);

    // Perform the reduction
    cub::DeviceSegmentedReduce::Sum(
        p_temp_storage_sum, temp_storage_sum_bytes, 
        d_items, d_sums, num_segments, d_offsets, d_offsets + 1);

    // MIN --------------------------------------------------- //
    // Determine temporary device storage requirements
    void* p_temp_storage_min = nullptr;
    size_t temp_storage_min_bytes = 0;
    cub::DeviceSegmentedReduce::Min(
        p_temp_storage_min, temp_storage_min_bytes, 
        d_items, d_mins, num_segments, d_offsets, d_offsets + 1);

    // Allocate required temporary storage
    cudaMalloc(&p_temp_storage_min, temp_storage_min_bytes);

    // Perform the reduction
    cub::DeviceSegmentedReduce::Min(
        p_temp_storage_min, temp_storage_min_bytes, 
        d_items, d_mins, num_segments, d_offsets, d_offsets + 1);

    // Check results
    cudaMemcpy(sums.data(), d_sums, num_segments * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(mins.data(), d_mins, num_segments * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Segment aggregates\n";
    std::cout << "   SUM    MIN\n------------\n";
    for (int s = 0; s < num_segments; ++s) {
        std::cout << std::setw(6) << sums[s] << " "
                  << std::setw(6) << mins[s] << "\n";
    }

    cudaFree(d_items);
    cudaFree(d_offsets);
    cudaFree(d_sums);
    cudaFree(d_mins);
    cudaFree(p_temp_storage_sum);
    cudaFree(p_temp_storage_min);
    return 0;
}
