
# CUDA Basics


Welcome to the CUDA Basics learning module of this journaling repo. This section builds foundational understanding—from why CUDA exists to writing your own GPU kernels.
---
### Now the Question arises, What exactly is CUDA?

CUDA (Compute Unified Device Architecture) is a parallel computing platform & programming model created by NVIDIA. It enables developers to write C/C++ (and now CUDA C++) code that runs on both the CPU (host) and GPU (device).



---

## Why We Need CUDA

* Massive parallelism on everyday hardware
GPUs (Graphics Processing Units) contain hundreds to thousands of cores optimized for parallel tasks like image processing, physics simulations, and deep learning. CPUs aren’t built for this scale.

* High-performance computing democratized
CUDA enables developers to use NVIDIA GPUs for general-purpose computing—no need for specialized supercomputers.

* Real-world speedups
As highlighted in PMPP, naive CUDA implementations already run 10–15× faster than CPU versions; tuning can push 45–100× speedups.

---
##  Basics of CUDA Kernel Development

### 1. Define a Kernel
A **kernel** is defined using the __global__declaration specifier and the number of CUDA threads that execute that kernel for a given kernel call is specified using a new <<<...>>>execution configuration syntax. Each thread that executes the kernel is given a unique thread ID that is accessible within the kernel through built-in variables.

Using the built-in variable threadIdx, adds two vectors A and B of size N and stores the result into vector C.

```cpp
// Kernel definition
__global__ void VecAdd(float* A, float* B, float* C)
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

int main()
{
    ...
    // Kernel invocation with N threads
    VecAdd<<<1, N>>>(A, B, C);
    ...
}

```
Detailed Explanation:

### _ _ _global_ _ _

* This is a CUDA function qualifier.

* __global__ marks a function as a kernel: callable from the host (CPU) and executed on the device (GPU).

* When you call a __global__ function you launch many parallel threads on the GPU.

### Kernel signature: void VecAdd(float* A, float* B, float* C)

* Parameters are device pointers — the kernel expects memory located on the GPU. If you pass host pointers (regular CPU memory) the behavior is invalid.

* float* A, float* B, float* C are arrays in GPU global memory.

### threadIdx.x

* Built-in CUDA variable that gives the index of the thread within its block along x-dimension.

* Ranges from 0 to blockDim.x - 1.

### int i = threadIdx.x;

* The code uses only the per-block thread index to pick which array element to operate on.

* C[i] = A[i] + B[i];

* Each thread reads A[i] and B[i] from global memory and writes sum into C[i].

* If threads do not have unique i, writes will collide (race), or some indices will be untouched.

### VecAdd<<<1, N>>>(A, B, C);

* Kernel launch syntax: <<<gridDim, blockDim>>>.

* gridDim = number of blocks (here 1)

* blockDim = threads per block (here N)

* This launches 1 * N threads.

* Important hardware limits: blockDim is limited by GPU (commonly ≤ 1024 per block). If N > max threads per block, this launch will fail.



