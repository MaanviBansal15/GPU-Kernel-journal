/*
Pinned memory = page-locked RAM, meaning the OS won’t move it around.  
This lets the GPU grab data directly (via DMA) → much faster transfers than normal malloc.  

End result: GPU does the heavy lifting in parallel, CPU just sets things up.  
*/


#include <cassert>
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace std;

// CUDA kernel for vector addition
__global__ void vectorAdd(int* a, int* b, int* c, int N) {
// global thread Id
 int tid = threadIdx.x + (blockIdx.x * blockDim.x);
 if (tid < N) {  // do a boundary check
  c[tid] = a[tid] + b[tid];
  }
}

// Check results
void verify_result(int *a, int *b, int *c, int N) {
  for (int i = 0; i < N; i++) {
    assert(c[i] == a[i] + b[i]);
  }
}

int main() {
  // Array size of 2^26 (67,108,864)
  constexpr int N = 1 << 26;
  size_t bytes = sizeof(int) * N;

// Vectors for holding the host-side (CPU-side) data
  int *h_a, *h_b, *h_c;

  // Allocate pinned memory
  cudaMallocHost(&h_a, bytes);
  cudaMallocHost(&h_b, bytes);
  cudaMallocHost(&h_c, bytes);

 // Initialize random numbers in each array
  for(int i = 0; i < N; i++){
    h_a[i] = rand() % 100;
    h_b[i] = rand() % 100;
  }
  
  // Allocate memory on the device
  // Pinned memory is allocated in a way that it cannot be paged out by the OS.
  // This allows the GPU to use DMA (Direct Memory Access) for faster and more efficient 
  // transfers between host (CPU) and device (GPU) compared to normal malloc.

  int *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);
  cudaMalloc(&d_c, bytes);

  // Copy data from the host to the device (CPU -> GPU)
  cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

  // Threads per CTA (1024 threads per CTA)
  int NUM_THREADS = 1 << 10;

  // CTAs per Grid
  int NUM_BLOCKS = (N + NUM_THREADS - 1) / NUM_THREADS;

  // Launch the kernel on the GPU
  vectorAdd<<<NUM_BLOCKS, NUM_THREADS>>>(d_a, d_b, d_c, N);

  // Copy sum vector from device to host
  cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

  // Check result for errors
  verify_result(h_a, h_b, h_c, N);

  // Free pinned memory
  cudaFreeHost(h_a);
  cudaFreeHost(h_b);
  cudaFreeHost(h_c);

  // Free memory on device
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  cout << "Completed Succesfully\n";

  return 0;
}

