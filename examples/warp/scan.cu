#include <iomanip>
#include <iostream>
#include <vector>
#include <cub/cub.cuh>

constexpr int num_blocks = 2;
constexpr int warps_per_block = 4;
constexpr int threads_per_warp = 10;

using WarpScanner = cub::WarpScan<int, threads_per_warp>;

__global__ void warpScan(int* vec, int* out, int* agg) {
    // Allocate shared memory for thread communication                         
    __shared__ WarpScanner::TempStorage temp[warps_per_block];

    // Assign thread local variables and data
    if (threadIdx.x % 32 < threads_per_warp) {
        int warp_lid = threadIdx.x / 32;
        int warp_gid = blockIdx.x * warps_per_block + warp_lid;
        int thread_gid = warp_gid * threads_per_warp 
                       + threadIdx.x % 32;
        int thread_data = vec[thread_gid];
        int thread_prod, warp_aggr;
            
        // Compute scan inside each warp
        WarpScanner(temp[warp_lid]).InclusiveScan(
            thread_data, thread_prod, cub::Sum(), warp_aggr
        );

        // Write to output
        out[thread_gid] = thread_prod;
        agg[thread_gid] = warp_aggr;
    }
}

int main() {
    // Useful values
    constexpr int physical_threads_per_block = warps_per_block * 32;
    constexpr int logical_threads_per_block = warps_per_block * threads_per_warp;
    constexpr int allocation_size = num_blocks * logical_threads_per_block;
    
    // Create vectors on host
    std::vector<int> h_vec(allocation_size);
    std::vector<int> h_out(allocation_size);
    std::vector<int> h_agg(allocation_size);

    // Fill host input vector
    for (size_t i = 0; i < allocation_size; ++i) {
        h_vec[i] = i % threads_per_warp + 1.f;
    }

    // Allocate memory on device
    void *p_vec, *p_out, *p_agg;
    cudaMalloc(&p_vec, allocation_size * sizeof(int));
    cudaMalloc(&p_out, allocation_size * sizeof(int));
    cudaMalloc(&p_agg, allocation_size * sizeof(int));
    int* d_vec = static_cast<int*>(p_vec);
    int* d_out = static_cast<int*>(p_out);
    int* d_agg = static_cast<int*>(p_agg);

    // Copy memory from host to device
    cudaMemcpy(d_vec, h_vec.data(), allocation_size * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel with num_warps warps
    warpScan<<<num_blocks, physical_threads_per_block>>>(d_vec, d_out, d_agg);

    // Check that execution went well, or print error string
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << cudaGetErrorString(err) << std::endl;
    }

    // Copy result from device to host
    cudaMemcpy(h_out.data(), d_out, allocation_size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_agg.data(), d_agg, allocation_size * sizeof(int), cudaMemcpyDeviceToHost);

    // Check results
    for (int i = 0; i < allocation_size; ++i) {
        const int j = i % threads_per_warp;
        if (j == 0) {
            std::cout << std::endl;
            std::cout << " Warp: " << i / threads_per_warp << "\n";
            std::cout << " INPUT     SUM    WARP\n";
            std::cout << "----------------------\n";
        }
        std::cout << std::setw(6) << h_vec[i] << "  "
                  << std::setw(6) << h_out[i] << "  " 
                  << std::setw(6) << h_agg[i] << std::endl;
    }

    // Free device memory and return
    cudaFree(d_vec);
    cudaFree(d_out);
    cudaFree(d_agg);
    return 0;
}
