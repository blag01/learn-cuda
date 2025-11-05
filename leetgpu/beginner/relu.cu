#include <cuda_runtime.h>

__global__ void relu_kernel(const float* input, float* output, int N) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= N) return;
    float val = input[i];
    if (val < 0) val = 0;
    output[i] = val;
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}