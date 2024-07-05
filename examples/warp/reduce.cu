#include <iomanip>
#include <iostream>
#include <vector>
#include <cub/cub.cuh>

constexpr int num_blocks = 4;
constexpr int warps_per_block = 4;
constexpr int threads_per_warp = 21;

using WarpReducer = cub::WarpReduce<int, threads_per_warp>;

__global__ void warpReduction(int* vec, int* out) {
    // Allocate shared memory for thread communication
    __shared__ WarpReducer::TempStorage temp[warps_per_block];

    if (threadIdx.x % 32 < threads_per_warp) {
        // Assign thread local variables and data
        int warp_lid = threadIdx.x / 32;
        int warp_gid = blockIdx.x * warps_per_block + warp_lid;
        int thread_gid = warp_gid * threads_per_warp 
                          + threadIdx.x % 32; 
        int thread_data = vec[thread_gid];

        // Compute reduction
        int warp_sum = WarpReducer(temp[warp_lid]).Sum(thread_data);

        // Output from lane0
        if (threadIdx.x % 32 == 0) out[warp_gid] = warp_sum;
    }
}

int main() {
    // Useful values
    constexpr int physical_threads_per_block = warps_per_block * 32;
    constexpr int logical_threads_per_block = warps_per_block * threads_per_warp;
    constexpr int allocation_size = num_blocks * logical_threads_per_block;
    constexpr int num_warps = num_blocks * warps_per_block;

    // Create host vectors
    std::vector<int> h_vec(allocation_size);
    std::vector<int> h_out(num_warps);

    // Fill input host vector
    for (size_t i = 0; i < allocation_size; ++i) {
        h_vec[i] = i;
    }

    // Allocate device memory
    void *p_vec, *p_out;
    cudaMalloc(&p_vec, allocation_size * sizeof(int));
    cudaMalloc(&p_out, num_warps * sizeof(int));
    int* d_vec = static_cast<int*>(p_vec);
    int* d_out = static_cast<int*>(p_out);

    // Copy memory from host to device
    cudaMemcpy(d_vec, h_vec.data(), allocation_size * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel with num_warps warps
    warpReduction<<<num_blocks, physical_threads_per_block>>>(d_vec, d_out);

    // Check that execution went well, or print error string
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << cudaGetErrorString(err) << std::endl;
    }

    // Copy result from device to host
    cudaMemcpy(h_out.data(), d_out, num_warps * sizeof(int), cudaMemcpyDeviceToHost);

    // Check results
    std::cout << "\n WARP     SUM\n";
    std::cout << " ------------\n";
    for (int i = 0; i < num_warps; ++i) {
        std::cout << " " << std::setw(4) << i << "  "
                  << std::setw(6) << h_out[i] << "\n";
    }
    std::cout << "\n CHECK: difference between adjacent warps should be " 
              << threads_per_warp * threads_per_warp << std::endl;
   
    // Free device memory and return
    cudaFree(d_vec);
    cudaFree(d_out);
    return 0;
}
