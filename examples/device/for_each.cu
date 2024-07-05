#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>
#include <cub/cub.cuh>

struct SquareAndAdd {
    int val;
    CUB_RUNTIME_FUNCTION explicit SquareAndAdd(int x) : val{x} {}
    __device__ void operator() (int& x) { x *= x; x += val; };
};

int main() {
    // Useful values
    constexpr int size = 10000;

    // Initialize host vector
    std::vector<int> numbers(size);
    std::vector<int> results(size);
    std::iota(numbers.begin(), numbers.end(), 0);
    
    // Allocate device memory and copy from host
    void* p_numbers;
    cudaMalloc(&p_numbers, size * sizeof(int));
    int* d_numbers = static_cast<int*>(p_numbers);
    cudaMemcpy(d_numbers, numbers.data(), size * sizeof(int), cudaMemcpyHostToDevice);

    // DeviceFor application
    cub::DeviceFor::ForEachN(d_numbers, size, SquareAndAdd(10));

    // Check for errors during kernel execution
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << cudaGetErrorString(err) << std::endl;
    }

    // Copy result from device to host
    cudaMemcpy(results.data(), d_numbers, size * sizeof(int), cudaMemcpyDeviceToHost);

    // Check results
    std::cout << "\n INPUT  OUTPUT\n";
    std::cout << "--------------\n";
    for (int i = 0; i < 20; ++i) {
        std::cout << std::setw(6) << numbers[i]
                  << std::setw(6) << results[i] << "\n";
    }
    
    // Free device memory and return
    cudaFree(d_numbers);
    return 0;
}
