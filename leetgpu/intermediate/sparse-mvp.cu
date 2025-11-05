#include <cuda_runtime.h>
#include <stdio.h>

__global__ void sparse_mvp_kernel(
    const float* A, const float* x, float* y, int M, int N
) {
    const int row = threadIdx.y + blockIdx.y * blockDim.y;
    const int lane = threadIdx.x; // i.e. index within the warp
    if (row >= M) return;
    float sum = 0.0;
    const int base = row * N;
    for (int i = lane; i < N; i += blockDim.x) {
        if (A[row * N + i] != 0.0) {
            sum += A[base + i] * x[i];
        }
    }
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, stride);
    }
    if (!lane) {
        //printf("adding to y[%d] from (%d, %d) in block (%d, %d)\n", row, threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y);
        atomicAdd(y + row, sum);
    }
}

// A, x, y are device pointers
extern "C" void solve(const float* A, const float* x, float* y, int M, int N, int nnz) {
    const dim3 numThreads(32, 32);
    const dim3 numBlocks(
        1,
        (M + numThreads.y - 1) / numThreads.y
    );
    sparse_mvp_kernel<<<numBlocks, numThreads>>>(
        A, x, y, M, N
    );

} 