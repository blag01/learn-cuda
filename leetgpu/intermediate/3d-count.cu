#include <cuda_runtime.h>
#include <stdio.h>

__global__ void slow_count_kernel(const int* input, int* output, int N, int M, int K, int P) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = threadIdx.z + blockIdx.z * blockDim.z;

    // printf("(x, y, z) = (%d, %d, %d)\n", x, y, z);

    if (x >= N || y >= M || z >= K) return;

    const int idx = x * M * K + y * K + z;

    // printf("found %d at %d %d %d\n", input[idx], x, y, z);

    if (input[idx] == P) atomicAdd(output, 1);
}

__global__ void slightly_faster_count_kernel(const int* input, int* output, int N, int M, int K, int P) {
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const int z = 4 * (threadIdx.z + blockIdx.z * blockDim.z);

    if (x >= N || y >= M) return;

    const int idx = x * M * K + y * K + z;

    int inc = 0;

    if (z < K && input[idx] == P) ++inc;
    if (z + 1 < K && input[idx + 1] == P) ++inc;
    if (z + 2 < K && input[idx + 2] == P) ++inc;
    if (z + 3 < K && input[idx + 3] == P) ++inc;

    atomicAdd(output, inc);
}


// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, int* output, int N, int M, int K, int P) {
    const int which = 1; // choice of kernel

    if (which == 0) {
        const dim3 numThreads(16, 16, 4);
        const dim3 numBlocks(
            (N + numThreads.x - 1) / numThreads.x,
            (M + numThreads.y - 1) / numThreads.y,
            (K + numThreads.z - 1) / numThreads.z
        );

        slow_count_kernel<<<numBlocks, numThreads>>>(input, output, N, M, K, P);
    } else if (which == 1){
        const dim3 numThreads(16, 16, 4);
        const dim3 numBlocks(
            (N + numThreads.x - 1) / numThreads.x,
            (M + numThreads.y - 1) / numThreads.y,
            (K + 4 * numThreads.z - 1) / (4 * numThreads.z)
        );

        slightly_faster_count_kernel<<<numBlocks, numThreads>>>(input, output, N, M, K, P);
    }

    cudaDeviceSynchronize();
}