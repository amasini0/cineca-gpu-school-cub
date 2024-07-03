#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>
#include <cub/cub.cuh>

struct Square {
    __device__ void operator() (int& x) { x *= x; };
};

using blocker = cub::WarpLoad<int, 4, cub::WARP_LOAD_DIRECT, 32>;

int main() {
    // Useful values
    constexpr int size = 10000;

    // Initialize host vector
    std::vector<int> numbers(size);
    std::vector<int> squares(size);
    std::iota(numbers.begin(), numbers.end(), 0);
    
    // Allocate device memory and copy from host
    void* p_numbers;
    cudaMalloc(&p_numbers, size * sizeof(int));
    int* d_numbers = static_cast<int*>(p_numbers);
    cudaMemcpy(d_numbers, numbers.data(), size * sizeof(int), cudaMemcpyHostToDevice);

    // DeviceFor application
    cub::DeviceFor::ForEachN(d_numbers, size, Square());

    // Check that execution went well, or print error string
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << cudaGetErrorString(err) << std::endl;
    }

    // Copy result from device to host
    cudaMemcpy(squares.data(), d_numbers, size * sizeof(int), cudaMemcpyDeviceToHost);

    // Check results
    std::cout << "\n INPUT  OUTPUT\n";
    std::cout << "--------------\n";
    for (int i = 0; i < 20; ++i) {
        std::cout << std::setw(6) << numbers[i]
                  << std::setw(6) << squares[i] << "\n";
    }
    
    // Free device memory and return
    cudaFree(d_numbers);
    return 0;
}
