#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>
#include <cub/cub.cuh>

constexpr int num_blocks = 2;
constexpr int block_dim_x = 8;
constexpr int block_dim_y = 4;
constexpr int items_per_thread = 2;
constexpr int bins = 16;

using BlockHistT = cub::BlockHistogram<int, block_dim_x, items_per_thread, bins,
                                       cub::BLOCK_HISTO_ATOMIC, block_dim_y>; 
using BlockLoadT = cub::BlockLoad<int, block_dim_x, items_per_thread, 
                                  cub::BLOCK_LOAD_VECTORIZE, block_dim_y>;

__global__ void blockHistogram(int* vec, unsigned* out1, unsigned* out2) {
    // Allocate shared memory for thread communication
    __shared__ BlockHistT::TempStorage hi_temp;
    __shared__ BlockLoadT::TempStorage ld_temp;

    // Assign block shared memory for bin counts
    __shared__ unsigned bin_counts1[bins];
    __shared__ unsigned bin_counts2[bins];

    // Load thread local data
    int block_offset = blockIdx.x * blockDim.x * blockDim.y * items_per_thread;
    int thread_data[items_per_thread];
    BlockLoadT(ld_temp).Load(vec + block_offset , thread_data);

    // Create histogram
    BlockHistT(hi_temp).InitHistogram(bin_counts1);
    BlockHistT(hi_temp).Composite(thread_data, bin_counts1);
    
    // Shortcut init + compositing, then keep compositing
    BlockHistT(hi_temp).Histogram(thread_data, bin_counts2);
    BlockHistT(hi_temp).Composite(thread_data, bin_counts2);

    // Output
    if (threadIdx.x < bins)
    for (int i = 0; i < bins; ++i) {
        out1[blockIdx.x * bins + i] = bin_counts1[i];
        out2[blockIdx.x * bins + i] = bin_counts2[i];
    }
}

int main() {
    // Useful values
    constexpr dim3 block(block_dim_x, block_dim_y);
    constexpr int items_per_block = block_dim_x * block_dim_y * items_per_thread;
    constexpr int allocation_size = num_blocks * items_per_block;
    
    // Create vectors on host
    std::vector<int> h_vec(allocation_size);
    std::vector<unsigned> h_out1(num_blocks * bins);
    std::vector<unsigned> h_out2(num_blocks * bins);

    // Fill host input vector
    for (int i = 0; i < allocation_size; ++i) {
        h_vec[i] = i % bins;
    }

    // Allocate memory on device
    void *p_vec, *p_out1, *p_out2;
    cudaMalloc(&p_vec, allocation_size * sizeof(int));
    cudaMalloc(&p_out1, num_blocks * bins * sizeof(unsigned));
    cudaMalloc(&p_out2, num_blocks * bins * sizeof(unsigned)); 
    int* d_vec  = static_cast<int*>(p_vec);
    unsigned* d_out1 = static_cast<unsigned*>(p_out1);
    unsigned* d_out2 = static_cast<unsigned*>(p_out2);

    // Copy memory from host to device
    cudaMemcpy(d_vec, h_vec.data(), allocation_size * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel with num_warps warps
    blockHistogram<<<num_blocks, block>>>(d_vec, d_out1, d_out2);

    // Check that execution went well, or print error string
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << cudaGetErrorString(err) << std::endl;
    }

    // Copy result from device to host
    cudaMemcpy(h_out1.data(), d_out1, num_blocks * bins * sizeof(unsigned), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out2.data(), d_out2, num_blocks * bins * sizeof(unsigned), cudaMemcpyDeviceToHost);

    // Check results
    std::cout << "Check: OUT1 should be " 
              << allocation_size / num_blocks / bins << "\n";
    std::cout << "Check: OUT2 should be 2 * OUT1\n\n";
    std::cout << "Note: this is true if bins <= threads_per_block, otherwise\n";
    std::cout << "OUT1 will contain threads_per_block 1s and 0s for the rest\n";
    for (int i = 0; i < num_blocks * bins; ++i) {
        if (int j = i % bins; j == 0) {
            std::cout << "\n\n Block: " << i / bins << "\n";
            std::cout << "   OUT1    OUT2 \n";
            std::cout << "----------------\n";
        }
        std::cout << " "
                  << std::setw(6) << h_out1[i] << "  "
                  << std::setw(6) << h_out2[i] << "\n";
    }
    
    // Free device memory and return
    cudaFree(d_vec);
    cudaFree(d_out1);
    cudaFree(d_out2);
    return 0;
}
