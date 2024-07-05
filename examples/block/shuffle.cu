#include <iomanip>
#include <iostream>
#include <vector>
#include <cub/block/block_shuffle.cuh>
#include <cub/cub.cuh>

constexpr int num_blocks = 2;
constexpr int block_dim_x = 8;
constexpr int block_dim_y = 4;
constexpr int items_per_thread = 2;

using BlockLoadT = cub::BlockLoad<int, block_dim_x, items_per_thread, 
                                  cub::BLOCK_LOAD_VECTORIZE, block_dim_y>;
using BlockStoreT = cub::BlockStore<int, block_dim_x, items_per_thread, 
                                  cub::BLOCK_STORE_VECTORIZE, block_dim_y>;
using BlockShuffleT = cub::BlockShuffle<int, block_dim_x, block_dim_y>; 

__global__ void blockShuffle(int* vec, int* out1, int* out2) {
    // Allocate shared memory for thread communication
    __shared__ BlockShuffleT::TempStorage shuf_temp;
    __shared__ BlockLoadT::TempStorage  load_temp;
    __shared__ BlockStoreT::TempStorage stre_temp;

    // Load thread local data
    int block_offset = blockIdx.x * blockDim.x * blockDim.y * items_per_thread;
    int thread_data[items_per_thread];
    BlockLoadT(load_temp).Load(vec + block_offset , thread_data);

    // Data to be shuffled
    int shuf_item = thread_data[0];
    int shuf_block[items_per_thread];
    
    // Shuffle single value (Offset, Rotate)
    BlockShuffleT(shuf_temp).Offset(shuf_item, shuf_item, 2);
    __syncthreads(); // This is required to get correct results

    // Shuffle entire block of items (Up/Down)
    shuf_block[0] = 111; // This is left unchanged by .Up(...) 
    BlockShuffleT(shuf_temp).Up(thread_data, shuf_block);

    // Store results
    thread_data[0] = shuf_item;
    BlockStoreT(stre_temp).Store(out1 + block_offset, thread_data);
    BlockStoreT(stre_temp).Store(out2 + block_offset, shuf_block);
}

int main() {
    // Useful values
    constexpr dim3 block(block_dim_x, block_dim_y);
    constexpr int items_per_block = block_dim_x * block_dim_y * items_per_thread;
    constexpr int allocation_size = num_blocks * items_per_block;
    
    // Create vectors on host
    std::vector<int> h_vec(allocation_size);
    std::vector<int> h_out1(allocation_size);
    std::vector<int> h_out2(allocation_size);

    // Fill host input vector
    for (int i = 0; i < allocation_size; ++i) {
        h_vec[i] = i % items_per_block;
    }

    // Allocate memory on device
    void *p_vec, *p_out1, *p_out2;
    cudaMalloc(&p_vec,  allocation_size * sizeof(int));
    cudaMalloc(&p_out1, allocation_size * sizeof(int));
    cudaMalloc(&p_out2, allocation_size * sizeof(int)); 
    int* d_vec  = static_cast<int*>(p_vec);
    int* d_out1 = static_cast<int*>(p_out1);
    int* d_out2 = static_cast<int*>(p_out2);

    // Copy memory from host to device
    cudaMemcpy(d_vec, h_vec.data(), allocation_size * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel with num_warps warps
    blockShuffle<<<num_blocks, block>>>(d_vec, d_out1, d_out2);

    // Check for errors during kernel execution
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << cudaGetErrorString(err) << std::endl;
    }

    // Copy result from device to host
    cudaMemcpy(h_out1.data(), d_out1, allocation_size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out2.data(), d_out2, allocation_size * sizeof(int), cudaMemcpyDeviceToHost);

    // Check results
    for (int i = 0; i < allocation_size; ++i) {
        const int j = i % items_per_block;
        if (j == 0) {
            std::cout << "\n  Block: " << i / items_per_block << "\n";
            std::cout << "  INPUT    OUT1    OUT2\n";
            std::cout << "-----------------------\n";
        }
        std::cout << " "
                  << std::setw(6) << h_vec[i] << "  "
                  << std::setw(6) << h_out1[i] << "  "
                  << std::setw(6) << h_out2[i] << "\n";
    }
    
    // Free device memory and return
    cudaFree(d_vec);
    cudaFree(d_out1);
    cudaFree(d_out2);
    return 0;
}
