#include <cuda_runtime.h>

// i tried two versions - one with local memory and one without
// they perform the same - this was surprising to me: i assumed accumulating
// the result locally per block and only writing once would be faster than having all threads
// contend for writing to result?

__global__ void dot_kernel(const float *A, const float *B, float *result, int N) {
    __shared__ extern float local_sum[];

    const int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i >= N) return;

    float contrib = A[i] * B[i];

    local_sum[0] = 0.0;
    __syncthreads();

    atomicAdd(&local_sum[0], contrib);
    __syncthreads();
    
    if (!threadIdx.x) atomicAdd(result, local_sum[0]);
}

__global__ void naive_dot_kernel(const float *A, const float *B, float *result, int N) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i >= N) return;

    float contrib = A[i] * B[i];

    atomicAdd(result, contrib);
}

// A, B, result are device pointers
extern "C" void solve(const float* A, const float* B, float* result, int N) {
    const int numThreads = 32;
    const int numBlocks = (N + numThreads - 1) / numThreads;
    
    naive_dot_kernel<<<numBlocks, numThreads, sizeof(float)>>>(A, B, result, N);
}