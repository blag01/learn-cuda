#include <cuda_runtime.h>

__global__ void sum_kernel(const int* input, int* output, int N) {
    const int i = 4 * (threadIdx.x + blockIdx.x * blockDim.x);
    int sum = 0;
    if (i < N) sum += input[i];
    if (i + 1 < N) sum += input[i + 1];
    if (i + 2 < N) sum += input[i + 2];
    if (i + 3 < N) sum += input[i + 3];
    if (sum) atomicAdd(output, sum);
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int S, int E) {
    const int numThreads = 1024;
    const int span = E - S + 1;
    const int numBlocks = (span + numThreads - 1) / numThreads;

    sum_kernel<<<numBlocks, numThreads>>>(input + S, output, span);
}