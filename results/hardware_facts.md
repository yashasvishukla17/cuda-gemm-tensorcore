\# GPU Hardware Facts



\## Experiment 1: GPU Information Query



\- \*\*Status:\*\* Executed successfully

\- \*\*Source file:\*\* `src/gpu\_facts.cu`



\### Hardware Specifications



| Property | Observed Value |

|---|---|

| GPU | NVIDIA GeForce RTX 3050 A Laptop GPU |

| Compute capability reported by program | 8.9 |

| Streaming Multiprocessors (SMs) | 14 |

| Global memory | 4.00 GiB |

| Memory bus width | 64 bits |

| Shared memory per block | 48 KiB |

| Registers per block | 65,536 |

| Maximum threads per block | 1,024 |

| L2 cache | 16 MiB |



\### What I Learned



\- `cudaGetDeviceProperties()` can retrieve GPU device properties.

\- Streaming Multiprocessors (SMs) execute GPU workloads.

\- Shared memory enables reuse of matrix tiles and can reduce repeated global-memory accesses.

\- Registers store per-thread data and intermediate calculations.

\- L2 cache helps reduce accesses to global memory.

\- Hardware limits influence the design and tuning of custom GEMM kernels.



\### Next Steps



1\. Run the cuBLAS FP32 matrix multiplication benchmark.

2\. Record execution time, estimated TFLOP/s, and correctness.

3\. Implement a custom tiled GEMM kernel and compare it with cuBLAS.



\### Verification Note



The program reported compute capability 8.9. Verify this value before choosing the final CUDA compilation target, because it is unexpected for an RTX 3050 Laptop GPU.





