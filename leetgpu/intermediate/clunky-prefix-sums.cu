#include <cuda_runtime.h>
#include <stdio.h>

// this is clunky
// more efficient solutions exist (see Blelloch scan, Hillis-Steele scan, etc)
// (notably, these avoid extra allocations)
// also: https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda

__global__ void sum_with_prev_kernel(const float* input, float* output, int N, int width) {
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= N) return;
    output[i] = input[i] + (i >= width ? input[i - width] : 0.);
    // printf("sum[%d..%d] = %f\n", max(0, i - 2 * width + 1), i, output[i]);
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    const int numThreads = 1024;
    const int numBlocks = (N + numThreads - 1) / numThreads;

    float *buff[2];

    buff[0] = output; // avoid allocating the first buffer and just use the output
    cudaMalloc(&buff[1], sizeof(float) * N);

    cudaMemcpy(buff[1], input, sizeof(float) * N, cudaMemcpyDeviceToDevice); // we need the input as prev
    
    int i = 0;

    for (int width = 1; width < N; width *= 2, i++) {
        sum_with_prev_kernel<<<numBlocks, numThreads>>>(buff[(i+1) % 2], buff[i % 2], N, width);
    }

    cudaMemcpy(output, buff[(i + 1) % 2], sizeof(float) * N, cudaMemcpyDeviceToDevice); // final output is the prev buffer
    cudaFree(buff[1]); // be nice to the memory
} 