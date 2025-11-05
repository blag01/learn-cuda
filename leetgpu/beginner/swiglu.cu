#include <cuda_runtime.h>
#include <cmath>

__global__ void swiglu_kernel(const float* input, float* output, int halfN) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= halfN) return;
    float x = input[i];
    float silu = x / (1.0 + exp(-x));
    float swiglu = silu * input[halfN + i];
    output[i] = swiglu;
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int halfN = N / 2;
    int threadsPerBlock = 256;
    int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;

    swiglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
    cudaDeviceSynchronize();
}