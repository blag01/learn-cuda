// leetgpu
#include <cuda_runtime.h>

__device__ int index1() {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    return idx;
}

__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int idx = index1();
    if (idx < N) C[idx] = A[idx] + B[idx];
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
