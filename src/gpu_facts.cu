
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>

int main() {
    int device = 0;

    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: "
                  << cudaGetErrorString(err) << '\n';
        return 1;
    }

    cudaDeviceProp p{};

    err = cudaGetDeviceProperties(&p, device);
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: "
                  << cudaGetErrorString(err) << '\n';
        return 1;
    }

    std::cout << std::fixed << std::setprecision(2);

    std::cout << "GPU: " << p.name << '\n';
    std::cout << "Compute capability: "
              << p.major << "." << p.minor << '\n';
    std::cout << "SM count: "
              << p.multiProcessorCount << '\n';

    std::cout << "Global memory (GiB): "
              << p.totalGlobalMem / (1024.0 * 1024 * 1024)
              << '\n';

    std::cout << "Memory bus width (bits): "
              << p.memoryBusWidth << '\n';

    std::cout << "Shared memory per block (KiB): "
              << p.sharedMemPerBlock / 1024.0 << '\n';

    std::cout << "Registers per block: "
              << p.regsPerBlock << '\n';

    std::cout << "Max threads per block: "
              << p.maxThreadsPerBlock << '\n';

    std::cout << "L2 cache (MiB): "
              << p.l2CacheSize / (1024.0 * 1024) << '\n';

    return 0;
}
