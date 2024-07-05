#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
#include <cub/cub.cuh>

constexpr int num_blocks = 2;
constexpr int block_dim = 32;
constexpr int items_per_thread = 4;
constexpr int runs_per_thread = items_per_thread; // must be same as items_per_thread

/*--------------ONLY MODIFY THIS FUNCTION --------------------------------------*/        
using DecodeT = cub::BlockRunLengthDecode<int, block_dim, runs_per_thread, items_per_thread>;
using StoreT = cub::BlockStore<int, block_dim, items_per_thread, cub::BLOCK_STORE_VECTORIZE>;

__global__ void blockDecode(int* sizes, int* values, int* lengths, int* output) {
    __shared__ DecodeT::TempStorage dc_temp;
    __shared__ StoreT::TempStorage  st_temp;

    int thread_values[runs_per_thread] = { 0 }; // init to zero
    int thread_lengths[runs_per_thread] = { 0 }; // init to zero

    // Load data from values and lengths inside thread-local arrays
    // Since we (probably) have more threads than required, due to the fact that
    // a thread handles more than one run, we will give the extra threads zero-filled
    // arrays (both for values and lengths, but the most important is lengths), so
    // that they will not influence the decoding
    int block_run_offset = (blockIdx.x != 0) ? sizes[blockIdx.x-1] : 0;
    int block_runs = sizes[blockIdx.x];
    int thread_run_offset = threadIdx.x * runs_per_thread;
    
    for (int i = 0; i < runs_per_thread; ++i) {
        int block_run_idx = thread_run_offset + i;
        int global_run_idx = block_run_offset + block_run_idx;
        if (block_run_idx < block_runs) {
            thread_values[i] = values[global_run_idx];
            thread_lengths[i] = lengths[global_run_idx];  
        }
    }

    // Initialize decoder and get total decoded size from it (not used further)
    // As a check, by construction total_decoded_size should be equal to items_per_thread * block_dim
    int total_decoded_size = 0;
    DecodeT decoder(dc_temp, thread_values, thread_lengths, total_decoded_size);
    
    // Run decoding of a batch of elements (the number is the width of the window, which 
    // depends on the template parameters). 
    // Generally, this should be in a while loop, since more than one batch of decoded 
    // elements may be required to decode all the sequence. 
    // In this case, we specialized the template to get the window size equal to the number
    // of elements to decode (per block), i.e. items_per_thread * 32 (threads in a block)
    // thus we only need one decoding pass to get them all. 
    int decoded_items[items_per_thread];
    decoder.RunLengthDecode(decoded_items, /* offset */ 0);

    // Store results in the correct output position
    const int block_output_offset = blockIdx.x * blockDim.x * items_per_thread;
    StoreT(st_temp).Store(output + block_output_offset, decoded_items);
}


/*--------------DO NOT CHANGE THIS PART OF THE CODE ----------------------------*/
using DiscT = cub::BlockDiscontinuity<int, block_dim>;
using ScanT = cub::BlockScan<int, block_dim, cub::BLOCK_SCAN_WARP_SCANS>;
using LoadT = cub::BlockLoad<int, block_dim, items_per_thread, cub::BLOCK_LOAD_VECTORIZE>;

__global__ void blockEncode (int* input, int* sizes, int* values, int* lengths) {
    // Allocate shared memory
    __shared__ LoadT::TempStorage ld_temp;
    __shared__ DiscT::TempStorage ds_temp;
    __shared__ ScanT::TempStorage sc_temp;

    // Declare thread-local data
    const int block_offset = blockIdx.x * blockDim.x * items_per_thread;
    int thread_data[items_per_thread];
    int thread_discont_mask[items_per_thread];
    int thread_scanned_mask[items_per_thread];
    
    // Load thread-local data
    LoadT(ld_temp).Load(input + block_offset, thread_data);
    
    // Compute discontinuity mask (puts a 1 at first element of each run)
    DiscT(ds_temp).FlagHeads(thread_discont_mask, thread_data, cub::Inequality());

    // Compute inclusive prefix sum of the ones inside the mask
    ScanT(sc_temp).InclusiveSum(thread_discont_mask, thread_scanned_mask);

    // If lengths or values are nullptr, only output size required on each block
    if (values == nullptr || lengths == nullptr) {
        if (threadIdx.x == (blockDim.x -1)) {
            sizes[blockIdx.x] = thread_scanned_mask[items_per_thread - 1]; 
        }
        return;
    }

    // The following executes only if values & lenghts point to valid memory
    // and sizes array is already filled (by a previous call to block_rle)

    // Get offset for writing in output array (equals the number of runs 
    // assigned to the prev block if block_idx != 0, else 0)
    int offset = (blockIdx.x != 0) ? sizes[blockIdx.x-1] : 0;

    // For each element of the thread, if it is the start of a run (i.e. if it has
    // a 1 in the discontinuity mask) write its value in correct position (i.e.
    // the one written in the scanned mask + offset from prev block - 1), and 
    // store the item's idx (thread_idx * items_per_thread + item_in_thread_idx)
    // in the lengths array to later get run length.
    for (int i = 0; i < items_per_thread; ++i) {
        if (thread_discont_mask[i] == 1) {
            int item_idx = threadIdx.x * items_per_thread + i;
            int out_idx = offset + thread_scanned_mask[i] - 1;
            values[out_idx]  = thread_data[i];
            lengths[out_idx] = item_idx;
        }
    }

    // Compute effective lenghts as difference of adjacent starting item indices 
    // obtained in the previous block (stored in lengths).
    // Only the first size-1 (size is the number of runs associated to the block)
    // are working, the others are idle.
    // Each active thread (except the last one) computes difference of starting 
    // indices of its run and the following run, to obtain the run's length.
    // Last active thread (idx = size-1) computes lenght as difference between 
    // its starting index and the index of the last block's item.
    int size = sizes[blockIdx.x];
    for (int thread_lid = threadIdx.x; thread_lid < size; thread_lid += blockDim.x) {
        int thread_gid = offset + thread_lid;
        int run_length = (thread_lid < (size - 1)) ? lengths[thread_gid + 1] - lengths[thread_gid]
                                                   : blockDim.x * items_per_thread - lengths[thread_gid];
        __syncthreads();
        lengths[thread_gid] = run_length;
    }
}

int main() {
    // Useful values
    constexpr int full_size = num_blocks * block_dim * items_per_thread;
    
    // Allocate input, output and sizes vectors on host
    std::vector<int> input(full_size);
    std::vector<int> output(full_size);
    std::vector<int> sizes(num_blocks);

    // Populate input with random sequences of integers
    std::random_device rd;
    std::uniform_int_distribution<int> rand_length(1,9);
    std::uniform_int_distribution<int> rand_value(0,9);

    int i = 0, l = 0, v = 0;
    while (i < full_size) {
        l = std::min(rand_length(rd), full_size - i); // Avoids going out of bounds
        v = rand_value(rd);
        for (int j = 0; j < l; ++j) input[i+j] = v;
        i += l;
    }

    // Allocate input, output and sizes arrays on device
    void *p_input, *p_output, *p_sizes;
    cudaMalloc(&p_input, full_size * sizeof(int));
    cudaMalloc(&p_output, full_size * sizeof(int));
    cudaMalloc(&p_sizes, num_blocks * sizeof(int));
    int* d_input  = static_cast<int*>(p_input);
    int* d_output = static_cast<int*>(p_output);
    int* d_sizes  = static_cast<int*>(p_sizes);
    
    // Copy input to device
    cudaMemcpy(d_input, input.data(), full_size * sizeof(int), cudaMemcpyHostToDevice);
    
    // Get sizes from blockEncode(sizes of encoded seqs on each block)
    blockEncode<<<num_blocks, block_dim>>>(d_input, d_sizes, nullptr, nullptr);
    
    // Check for errors during kernel execution
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "Error: " << cudaGetErrorString(err) << "\n";
    }

    // Copy sizes array to host
    cudaMemcpy(sizes.data(), d_sizes, num_blocks * sizeof(int), cudaMemcpyDeviceToHost);
   
    // Compute total size of output
    const int encoded_size = std::accumulate(sizes.begin(), sizes.end(), 0);

    // Allocate values and lengths vectors on host
    std::vector<int> values(encoded_size);
    std::vector<int> lengths(encoded_size);

    // Allocate values and lengths arrays on device
    void *p_values, *p_lengths;
    cudaMalloc(&p_values, encoded_size * sizeof(int));
    cudaMalloc(&p_lengths, encoded_size * sizeof(int));
    int* d_values = static_cast<int*>(p_values);
    int* d_lengths = static_cast<int*>(p_lengths);

    // Run encoding (this time for real)
    blockEncode<<<num_blocks, block_dim>>>(d_input, d_sizes, d_values, d_lengths);
    
    // Check for errors during kernel execution
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "Error: " << cudaGetErrorString(err) << "\n";
    }

    // Copy encoded sequences to host for printing
    cudaMemcpy(values.data(), d_values, encoded_size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(lengths.data(), d_lengths, encoded_size * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Encoded sequence length: " << std::setw(3) << encoded_size << "\n";
    std::cout << "\n VAL  RUN\n";
    int offset = 0;
    for (int b = 0; b < num_blocks; ++b) {
        const int size = sizes[b];
        std::cout << "--------- Block: " << std::setw(2) << b << "\n";
        for (int i = 0; i < size; ++i) {
            std::cout << std::setw(4) << values[offset + i] << " "
                      << std::setw(4) << lengths[offset + i] << "\n";
        }
        offset += size;
    }

    // Run decoding (use single block, it's easier)
    blockDecode<<<num_blocks, block_dim>>>(d_sizes, d_values, d_lengths, d_output);

    // Check for errors during kernel execution
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "Error: " << cudaGetErrorString(err) << "\n";
    }
    
    // Copy output (decoded) sequence to host for checking
    cudaMemcpy(output.data(), d_output, full_size * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "\n INPUT OUTPUT\n-------------\n";
    for (int i = 0; i < 20; ++i) {
        std::cout << std::setw(6) << input[i] << " "
                  << std::setw(6) << output[i] << "\n";
    }

    // Check that input and output match
    std::cout << "\n\n";
    bool mismatch = false;
    for (int i = 0; i < full_size; ++i) {
        if (input[i] != output[i]) {
            mismatch = true;
            std::cout << "Mismatch at element: " << std::setw(4) << i << "\n";
            std::cout << "Decoding FAILED\n";
            break;
        }
    }
    if (!mismatch) std::cout << "No mismatches found\nDecoding SUCCEDED\n";

    // Release resources and finish execution
    cudaFree(d_input);
    cudaFree(d_sizes);
    cudaFree(d_output);
    cudaFree(d_values);
    cudaFree(d_lengths);
    return 0;
}
