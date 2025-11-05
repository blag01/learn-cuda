#include <cuda_runtime.h>

__global__ void reverse_array(float* input, int N) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= N - 1 - i) return;
    float old = input[N - 1 - i];
    input[N - 1 - i] = old;
    input[i] = old;
}

// input is device pointer
extern "C" void solve(float* input, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, N);
    cudaDeviceSynchronize();
}