#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>
#include <cub/cub.cuh>

constexpr int num_blocks = 2;
constexpr int warps_per_block = 4;
constexpr int threads_per_warp = 4; // must be power of two
constexpr int items_per_thread = 4;

using WarpLoader = cub::WarpLoad<int, items_per_thread, cub::WARP_LOAD_DIRECT, 
                                 threads_per_warp>;
using WarpExchanger = cub::WarpExchange<int, items_per_thread, /*cub::WARP_ECHANGE_SHMEM,*/
                                        threads_per_warp>;
using WarpStorerBL = cub::WarpStore<int, items_per_thread, cub::WARP_STORE_DIRECT, 
                                         threads_per_warp>;
using WarpStorerST = cub::WarpStore<int, items_per_thread, cub::WARP_STORE_STRIPED, 
                                         threads_per_warp>;


__global__ void warpExchange(int* vec, int* out1, int* out2) {
    // Allocate shared memory for thread communication
    __shared__ WarpLoader::TempStorage ld_temp[warps_per_block];
    __shared__ WarpExchanger::TempStorage ex_temp[warps_per_block];
    __shared__ WarpStorerBL::TempStorage bl_temp[warps_per_block];
    __shared__ WarpStorerST::TempStorage st_temp[warps_per_block];
    
    // Assign thread-local variables and data
    int warp_lid = threadIdx.x / threads_per_warp;
    int warp_gid = blockIdx.x * warps_per_block + warp_lid;
    int warp_offset = warp_gid * threads_per_warp * items_per_thread;
    int thread_data[items_per_thread];

    // Load blocked, exchange, output blocked and striped
    WarpLoader(ld_temp[warp_lid]).Load(vec + warp_offset, thread_data); 
    WarpExchanger(ex_temp[warp_lid]).BlockedToStriped(thread_data, thread_data);
    WarpStorerBL(bl_temp[warp_lid]).Store(out1 + warp_offset, thread_data);
    WarpStorerST(st_temp[warp_lid]).Store(out2 + warp_offset, thread_data);
}

int main() {
    // Useful values
    constexpr int threads_per_block = warps_per_block * threads_per_warp;
    constexpr int allocation_size = num_blocks * threads_per_block * items_per_thread;
    
    // Create vectors on host
    std::vector<int> h_vec (allocation_size);
    std::vector<int> h_out1(allocation_size);
    std::vector<int> h_out2(allocation_size);

    // Fill host input vector
    std::iota(h_vec.begin(), h_vec.end(), 0);

    // Allocate memory on device
    void *p_vec, *p_out1, *p_out2;
    cudaMalloc(&p_vec, allocation_size * sizeof(int));
    cudaMalloc(&p_out1, allocation_size * sizeof(int));
    cudaMalloc(&p_out2, allocation_size * sizeof(int)); 
    int* d_vec  = static_cast<int*>(p_vec);
    int* d_out1 = static_cast<int*>(p_out1);
    int* d_out2 = static_cast<int*>(p_out2);

    // Copy memory from host to device
    cudaMemcpy(d_vec, h_vec.data(), allocation_size * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel with num_warps warps
    warpExchange<<<num_blocks, threads_per_block>>>(d_vec, d_out1, d_out2);

    // Check that execution went well, or print error string
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << cudaGetErrorString(err) << std::endl;
    }

    // Copy result from device to host
    cudaMemcpy(h_out1.data(), d_out1, allocation_size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out2.data(), d_out2, allocation_size * sizeof(int), cudaMemcpyDeviceToHost);


    // Check results
    for (int i = 0; i < allocation_size; ++i) {
        const int block = i / (threads_per_block * items_per_thread);
        const int b_idx = i % (threads_per_block * items_per_thread);
        const int warp  = b_idx / (threads_per_warp * items_per_thread);
        const int w_idx = b_idx % (threads_per_warp * items_per_thread);
        const int thread = w_idx / items_per_thread;
        const int t_idx  = w_idx % items_per_thread;

        if (b_idx == 0) {
            std::cout << std::endl;
            std::cout << "\n\n Block: " << block << "\n";
        }
        if (w_idx == 0) {
            std::cout << "---------------------- Warp: " << warp << "\n";
        }
        if (t_idx == 0) {
            std::cout << "---------------------- Thread: " << thread << "\n";
        }
        std::cout << std::setw(6) << h_vec[i] << "  "
                  << std::setw(6) << h_out1[i] <<  "  "
                  << std::setw(6) << h_out2[i] << std::endl;
    }

    // Free device memory and return
    cudaFree(d_vec);
    cudaFree(d_out1);
    cudaFree(d_out2);
    return 0;
}
