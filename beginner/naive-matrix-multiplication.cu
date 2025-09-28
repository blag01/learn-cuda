#include <cuda_runtime.h>
#include <stdio.h>

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int c = threadIdx.x + blockDim.x * blockIdx.x;
    int r = threadIdx.y + blockDim.y * blockIdx.y;

    // compute C[r][c] = A[r] * B^T[c]
    if (!(r < M && c < K)) return;
    
    float sum = 0.0;

    for (int i = 0, locA = r * N, locB = c; i < N; i++, locA++, locB += K) {
        sum += A[locA] * B[locB];
    }

    C[r * K + c] = sum;
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
