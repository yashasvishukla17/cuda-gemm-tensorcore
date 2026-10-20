
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t err = (call);                             \
        if (err != cudaSuccess) {                             \
            std::cerr << "CUDA error: "                       \
                      << cudaGetErrorString(err) << '\n';     \
            return 1;                                         \
        }                                                     \
    } while (0)

#define CUBLAS_CHECK(call)                                    \
    do {                                                      \
        cublasStatus_t status = (call);                       \
        if (status != CUBLAS_STATUS_SUCCESS) {                \
            std::cerr << "cuBLAS error: " << status << '\n';  \
            return 1;                                         \
        }                                                     \
    } while (0)

int main() {
    const int N = 1024;
    const float alpha = 1.0f;
    const float beta = 0.0f;

    const size_t bytes = static_cast<size_t>(N) * N * sizeof(float);

    std::vector<float> A(N * N, 1.0f);
    std::vector<float> B(N * N, 2.0f);
    std::vector<float> C(N * N, 0.0f);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    CUDA_CHECK(cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), bytes, cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // cuBLAS uses column-major matrices. With these dimensions,
    // this call computes C = A * B for our row-major input buffers.
    CUBLAS_CHECK(cublasSgemm(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, N, N,
        &alpha,
        d_B, N,
        d_A, N,
        &beta,
        d_C, N
    ));

    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int warmup = 5;
    const int iterations = 20;

    for (int i = 0; i < warmup; ++i) {
        CUBLAS_CHECK(cublasSgemm(
            handle, CUBLAS_OP_N, CUBLAS_OP_N,
            N, N, N, &alpha, d_B, N, d_A, N,
            &beta, d_C, N
        ));
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < iterations; ++i) {
        CUBLAS_CHECK(cublasSgemm(
            handle, CUBLAS_OP_N, CUBLAS_OP_N,
            N, N, N, &alpha, d_B, N, d_A, N,
            &beta, d_C, N
        ));
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));

    CUDA_CHECK(cudaMemcpy(C.data(), d_C, bytes, cudaMemcpyDeviceToHost));

    bool correct = true;
    for (int i = 0; i < N * N; ++i) {
        if (std::fabs(C[i] - static_cast<float>(2 * N)) > 1e-3f) {
            correct = false;
            break;
        }
    }

    const double avg_ms = total_ms / iterations;
    const double operations = 2.0 * N * N * N;
    const double tflops = operations / (avg_ms * 1e-3) / 1e12;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "GPU: NVIDIA RTX 3050 A Laptop GPU\n";
    std::cout << "Matrix dimensions: " << N << " x " << N << '\n';
    std::cout << "Average cuBLAS time: " << avg_ms << " ms\n";
    std::cout << "Estimated throughput: " << tflops << " TFLOP/s\n";
    std::cout << "Correctness: " << (correct ? "PASS" : "FAIL") << '\n';

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUBLAS_CHECK(cublasDestroy(handle));

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return correct ? 0 : 1;
}
