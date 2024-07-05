#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
#include <cub/cub.cuh>
#include <cub/device/device_partition.cuh>

// Define custom op for selection with __device__ call operator
struct Even {
    __device__ bool operator() (const int& x) {
        return static_cast<bool>(x%2 == 0);
    }
};

int main() {
    // Useful values
    constexpr int num_items = 20;

    // Allocate host vector
    std::vector<int>  items(num_items);
    std::vector<char> flags(num_items); // must be castable to bool
    std::vector<int>  out1(num_items);
    std::vector<int>  out2(num_items);

    // Initialize vector and flags with random values
    for (int i = 0; i < num_items; ++i) {
        int val = i + 1;
        items[i] = val;
        flags[i] = static_cast<char>((val%3 == 0) ? 1 : 0); // flag multiples of 3 
    }

    // Allocate input arrays on device
    void *p_items, *p_flags;
    cudaMalloc(&p_items, num_items*sizeof(int) );
    cudaMalloc(&p_flags, num_items*sizeof(bool));
    int  *d_items = static_cast<int* >(p_items);
    char *d_flags = static_cast<char*>(p_flags);

    // Copy items and flags to device
    cudaMemcpy(d_items, items.data(), num_items*sizeof(int),  cudaMemcpyHostToDevice);
    cudaMemcpy(d_flags, flags.data(), num_items*sizeof(char), cudaMemcpyHostToDevice);


    // Using FLAGGED -------------------------------------------- //

    // Allocate output array and single int for writing number of selected items
    void *p_out1, *p_num_selected_1;
    cudaMalloc(&p_out1, num_items*sizeof(int));
    cudaMalloc(&p_num_selected_1, sizeof(int));
    int *d_out1 = static_cast<int*>(p_out1);
    int *d_num_selected_1 = static_cast<int*>(p_num_selected_1);

    // Get memory requirements for algorithm
    void *p_temp_storage_1 = nullptr;
    size_t temp_storage_1_bytes = 0; // This must be size_t according to Flagged signature
    cub::DevicePartition::Flagged(
        p_temp_storage_1, temp_storage_1_bytes,
        d_items, d_flags, d_out1, d_num_selected_1, num_items);

    // Allocate required temporary storage
    cudaMalloc(&p_temp_storage_1, temp_storage_1_bytes);

    // Run selection
    cub::DevicePartition::Flagged(
        p_temp_storage_1, temp_storage_1_bytes,
        d_items, d_flags, d_out1, d_num_selected_1, num_items);

    // Copy back output and num_selected to host
    int num_selected_1 = 0;
    cudaMemcpy(out1.data(), d_out1, num_items * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&num_selected_1, d_num_selected_1, sizeof(int), cudaMemcpyDeviceToHost);

    
    // Using IF ------------------------------------------------- //

    // Allocate output array and single int for writing number of selected items
    void *p_out2, *p_num_selected_2;
    cudaMalloc(&p_out2, num_items*sizeof(int));
    cudaMalloc(&p_num_selected_2, sizeof(int));
    int *d_out2 = static_cast<int*>(p_out2);
    int *d_num_selected_2 = static_cast<int*>(p_num_selected_2);

    // Get memory requirements for algorithm
    void *p_temp_storage_2 = nullptr;
    size_t temp_storage_2_bytes = 0;
    cub::DevicePartition::If(
        p_temp_storage_2, temp_storage_2_bytes,
        d_items, d_out2, d_num_selected_2, num_items, Even());

    // Allocate required temporaty storage
    cudaMalloc(&p_temp_storage_2, temp_storage_2_bytes);

    // Run selection
    cub::DevicePartition::If(
        p_temp_storage_2, temp_storage_2_bytes,
        d_items, d_out2, d_num_selected_2, num_items, Even());
    
    // Copy back output and num_selected to host
    int num_selected_2 = 0;
    cudaMemcpy(out2.data(), d_out2, num_items * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&num_selected_2, d_num_selected_2, sizeof(int), cudaMemcpyDeviceToHost);

    // Checking results
    std::cout << "Results for FLAGGED\n";
    std::cout << "- number of selected items: " << std::setw(3) << num_selected_1 << "\n";
    std::cout << "- selected items: [ ";
    for (int s = 0; s < num_selected_1; ++s) {
        std::cout << out1[s] << " ";
    }
    std::cout << "]\n";
    std::cout << "- excluded elements: [ ";
    for (int s = num_selected_1; s < num_items; ++s) {
        std::cout << out1[s] << " ";
    }
    std::cout << "]\n";

    std::cout << "\nResults for IF\n";
    std::cout << "- number of selected items:" << std::setw(3) << num_selected_2 << "\n";
    std::cout << "- selected items: [ ";
    for (int s = 0; s < num_selected_2; ++s) {
        std::cout << out2[s] << " ";
    }
    std::cout << "]\n";
    std::cout << "- excluded elements: [ ";
    for (int s = num_selected_2; s < num_items; ++s) {
        std::cout << out2[s] << " ";
    }
    std::cout << "]\n";


    cudaFree(d_items);
    cudaFree(d_flags);
    cudaFree(d_out1);
    cudaFree(d_out2);
    cudaFree(d_num_selected_1);
    cudaFree(d_num_selected_2);
    cudaFree(p_temp_storage_1);
    cudaFree(p_temp_storage_2);
    return 0;
}
