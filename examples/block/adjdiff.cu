#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>
#include <cub/block/block_adjacent_difference.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>

constexpr int num_blocks = 2;
constexpr int block_dim_x = 8;
constexpr int block_dim_y = 8;
constexpr int items_per_thread = 2;

using BlockAdjDiffer = cub::BlockAdjacentDifference<int, block_dim_x, block_dim_y>; 
using BlockLoader = cub::BlockLoad<int, block_dim_x, items_per_thread, 
                                 cub::BLOCK_LOAD_VECTORIZE, block_dim_y>;
using BlockStorer = cub::BlockStore<int, block_dim_x, items_per_thread,
                                    cub::BLOCK_STORE_VECTORIZE, block_dim_y>;

__global__ void blockAdjDiff(int* vec, int* out1, int* out2) {
    // Allocate shared memory for thread communication
    __shared__ BlockAdjDiffer::TempStorage temp;
    __shared__ BlockLoader::TempStorage ld_temp;
    __shared__ BlockStorer::TempStorage st_temp;

    // Assign thread local variables and data
    auto op = [=](auto& x, auto& y){ return x - y; };
    int block_offset = blockIdx.x * blockDim.x * blockDim.y * items_per_thread;
    int thread_data[items_per_thread];
    int diff_result[items_per_thread];

    BlockLoader(ld_temp).Load(vec + block_offset , thread_data);
    BlockAdjDiffer(temp).SubtractLeft(thread_data, diff_result, op);
    BlockStorer(st_temp).Store(out1 + block_offset, diff_result);

    __syncthreads(); // since we have to reuse temp storage for adj diff
    
    BlockAdjDiffer(temp).SubtractRight(thread_data, diff_result, op);
    BlockStorer(st_temp).Store(out2 + block_offset, diff_result);
}

int main() {
    // Useful values
    constexpr dim3 block(block_dim_x, block_dim_y);
    constexpr int items_per_block = block_dim_x * block_dim_y * items_per_thread;
    constexpr int allocation_size = num_blocks * items_per_block;

    
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
    blockAdjDiff<<<num_blocks, block>>>(d_vec, d_out1, d_out2);

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
        if (int j = i % items_per_block; j == 0) {
            std::cout << "\n\n Block: " << i / items_per_block << "\n";
            std::cout << " INPUT  DIFF_L  DIFF_R \n";
            std::cout << "-----------------------\n";
        }
        std::cout << std::setw(6) << h_vec[i]  << "  "
                  << std::setw(6) << h_out1[i] << "  "
                  << std::setw(6) << h_out2[i] << "\n";
    }

    // Free device memory and return
    cudaFree(d_vec);
    cudaFree(d_out1);
    cudaFree(d_out2);
    return 0;
}
