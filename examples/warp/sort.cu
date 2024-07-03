#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>
#include <cub/cub.cuh>

constexpr int num_blocks = 2;
constexpr int warps_per_block = 2;
constexpr int threads_per_warp = 16;
constexpr int items_per_thread = 4;

using WarpLoader = cub::WarpLoad<int, items_per_thread, cub::WARP_LOAD_VECTORIZE,
                                 threads_per_warp>;
using WarpSorter = cub::WarpMergeSort<int, items_per_thread, threads_per_warp>; 
using WarpStorer = cub::WarpStore<int, items_per_thread, cub::WARP_STORE_VECTORIZE,
                                  threads_per_warp>;

__global__ void warpSort(int* vec, int* out) {
    // Custom sort operation
    auto less = [=](const auto& x, const auto& y) { return x < y; };

    // Array for thread-local items
    int thread_data[items_per_thread];

    // Allocate shared memory for thread communication
    __shared__ WarpLoader::TempStorage load_temp[warps_per_block];
    __shared__ WarpSorter::TempStorage sort_temp[warps_per_block];
    __shared__ WarpStorer::TempStorage stre_temp[warps_per_block];

    // Assign thread local variables and data
    const int warp_lid = threadIdx.x / threads_per_warp;
    const int warp_gid = blockIdx.x * warps_per_block + warp_lid;
    const int warp_offset = warp_gid * threads_per_warp * items_per_thread;
    
    // Load data, sort them and put them back
    WarpLoader(load_temp[warp_lid]).Load(vec + warp_offset, thread_data);
    WarpSorter(sort_temp[warp_lid]).Sort(thread_data, less);
    WarpStorer(stre_temp[warp_lid]).Store(out + warp_offset, thread_data);
}

int main() {
    // Useful values
    constexpr int items_per_warp = threads_per_warp * items_per_thread;
    constexpr int threads_per_block = warps_per_block * threads_per_warp;
    constexpr int allocation_size = num_blocks * warps_per_block * items_per_warp;

    // Create host vectors
    std::vector<int> h_vec(allocation_size);
    std::vector<int> h_out(allocation_size);

    // Fill input vector on host
    for (size_t i = 0; i < allocation_size; ++i) {
        int j = i % items_per_warp;
        if (j % 2 == 0) { 
            h_vec[i] = j / 2; 
        } else {
            h_vec[i] = items_per_warp -1 -j/2;
        } 
    }

    // Allocate memory on device
    void *p_vec, *p_out;
    cudaMalloc(&p_vec, allocation_size * sizeof(int));
    cudaMalloc(&p_out, allocation_size * sizeof(int));
    int* d_vec = static_cast<int*>(p_vec);
    int* d_out = static_cast<int*>(p_out);

    // Copy memory from host to device
    cudaMemcpy(d_vec, h_vec.data(), allocation_size * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel with num_warps warps
    warpSort<<<num_blocks, threads_per_block>>>(d_vec, d_out);

    // Check that execution went well, or print error string
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << cudaGetErrorString(err) << std::endl;
    }

    // Copy result from device to host
    cudaMemcpy(h_out.data(), d_out, allocation_size * sizeof(int), cudaMemcpyDeviceToHost);

    // Check results
    for (size_t i = 0; i < allocation_size; ++i) {
        const int j = i % items_per_warp;
        if (j == 0) {
            std::cout << std::endl;
            std::cout << " Warp: " << i/items_per_warp << "\n";
            std::cout << " INPUT  OUTPUT\n";
            std::cout << "--------------\n";
        }
        std::cout << std::setw(6) << h_vec[i] << "  " 
                  << std::setw(6) << h_out[i] << "\n";
        
    }

    // Free device memory and return
    cudaFree(d_vec);
    cudaFree(d_out);
    return 0;
}
