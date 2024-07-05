#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>
#include <cub/cub.cuh>
#include <cub/device/device_select.cuh>

// Predicate for if
struct LessThan {
    int ref_value;
    explicit LessThan(int x) : ref_value{x} {}
    __device__ bool operator() (const int& x) { return x < ref_value; }
};

int main() {
    // Useful values
    constexpr int num_items = 60;
    constexpr int threshold = 5;

    // Allocate host vector
    std::vector<int> items(num_items);
    std::vector<int> uniqs(num_items);
    
    // Fill vector with random sequence of values
    std::random_device rd;
    std::uniform_int_distribution<int> random_value(0,9);
    for (auto& el: items) {
        el = random_value(rd);
    }

    // Allocate memory on device
    void *p_items, *p_uniqs;
    cudaMalloc(&p_items, num_items * sizeof(int));
    cudaMalloc(&p_uniqs, num_items * sizeof(int));
    int* d_items = static_cast<int*>(p_items);
    int* d_uniqs = static_cast<int*>(p_uniqs);

    // Copy items to device
    cudaMemcpy(d_items, items.data(), num_items * sizeof(int), cudaMemcpyHostToDevice);
    
    // Allocate pointer to store number of not selected elements
    void *p_num_selected;
    cudaMalloc(&p_num_selected, sizeof(int));
    int* d_num_selected = static_cast<int*>(p_num_selected);

    // Determine temporary storage requirements
    void *p_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceSelect::If(
        p_temp_storage, temp_storage_bytes,
        d_items, d_uniqs, d_num_selected, num_items, LessThan(threshold));

    // Allocate temporary storage
    cudaMalloc(&p_temp_storage, temp_storage_bytes);

    // Run selection
    cub::DeviceSelect::If(
        p_temp_storage, temp_storage_bytes,
        d_items, d_uniqs, d_num_selected, num_items, LessThan(threshold));

    // Copy number of uniques back to host
    int num_selected = 0;
    cudaMemcpy(&num_selected, d_num_selected, sizeof(int), cudaMemcpyDeviceToHost);

    // Copy back output array of uniques
    cudaMemcpy(uniqs.data(), d_uniqs, num_items*sizeof(int), cudaMemcpyDeviceToHost);

    // Check results
    std::cout << "Number of elements less than " << threshold 
              << " is: " << num_selected << "\n";
    std::cout << "Selected elements: [ ";
    for (int s = 0; s < num_selected; ++s) {
        std::cout << uniqs[s] << " ";
    }
    std::cout << "]\n";
    
    // Release resources and finish execution
    cudaFree(d_items);
    cudaFree(d_uniqs);
    cudaFree(d_num_selected);
    cudaFree(p_temp_storage);
    return 0;
}
