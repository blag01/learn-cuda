#include <cuda_runtime.h>
#include <stdio.h>

__global__ void histogram_kernel(const int* input, int* histogram, int N, int num_bins) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= N) return;
    atomicAdd(histogram + input[i], 1);
}

__global__ void fast_histogram_kernel(const int* input, int* histogram, int N, int num_bins) {
    __shared__ extern int local_hist[];
    const int idx = 4 * (threadIdx.x + blockIdx.x * blockDim.x);
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) local_hist[i] = 0;
    __syncthreads();
    /// reading 4 ints from each thread helps us avoid contention between threads for the same memory locations
    /// some other solutions use int4 here and/or unroll the loop
    for (int i = 0; i < 4 && idx + i < N; i++) atomicAdd(local_hist + input[idx + i], 1);
    __syncthreads();
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) atomicAdd(&histogram[i], local_hist[i]);
}

// input, histogram are device pointers
extern "C" void solve(const int* input, int* histogram, int N, int num_bins) {
    const int numThreads = 64;
    const bool useFast = true;
    // cudaMemset(histogram, 0, sizeof(int) * num_bins);
    
    if (!useFast) {
        const int numBlocks = (N + numThreads - 1) / numThreads;
        histogram_kernel<<<numBlocks, numThreads>>>(input, histogram, N, num_bins);
    } else {
        const int numBlocks = (N + numThreads * 4 - 1) / (numThreads * 4);
        fast_histogram_kernel<<<numBlocks, numThreads, sizeof(int) * num_bins>>>(input, histogram, N, num_bins);
    }
    cudaDeviceSynchronize();
}
