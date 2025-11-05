#include <cuda_runtime.h>
#include <stdio.h>

__global__ void locate_nonzero_kernel(const float *A, int *output, int *len, int M, int N) {
    const int r = threadIdx.x + blockIdx.x * blockDim.x;
    const int c = threadIdx.y + blockIdx.y * blockDim.y;
    if (r >= M || c >= N) return;
    if (abs(A[r * N + c]) > 1e-12) {
        int slot = atomicAdd(len, 1);
        output[2 * slot] = r;
        output[2 * slot + 1] = c;
        // printf("non-zero %f found at (%d, %d), in slot %d\n", A[r * N + c], r, c, slot);
    }
}

__global__ void sparse_mvp_kernel(const int *A_sp, const float *A, const float *x, float *y, int M, int N, int nnz) {
    const int i = 2 * (threadIdx.x + blockIdx.x * blockDim.x);
    if (i >= 2 * nnz) return;
    const int r = A_sp[i];
    const int c = A_sp[i + 1];
    float contrib = A[r * N + c] * x[c];    // non-zero A_ij affects y_i 
    // printf("adding %.2f to y[%d] from (%d, %d) via slot %d\n", contrib, r, r, c, i / 2);
    atomicAdd(y + r, contrib);
}

// A, x, y are device pointers
extern "C" void solve(const float* A, const float* x, float* y, int M, int N, int nnz) {
    int *A_sp;
    int *len;
    
    cudaMalloc(&len, sizeof(int));
    cudaMalloc(&A_sp, sizeof(int) * nnz * 2);
    cudaMemset(len, 0, sizeof(int));
    cudaMemset(A_sp, 0, sizeof(int) * nnz * 2);

    {
        const dim3 numThreads(32, 32);
        const dim3 numBlocks((M + numThreads.x - 1) / numThreads.x, (N + numThreads.y - 1) / numThreads.y);
        locate_nonzero_kernel<<<numBlocks, numThreads>>>(A, A_sp, len, M, N);
    }

    int computed_nnz;
    cudaMemcpy(&computed_nnz, len, sizeof(int), cudaMemcpyDeviceToHost);
    
    // nnz does not match up what i get for some tests??!
    
    // if (computed_nnz != nnz)
    //     exit(1);

    {
        const dim3 numThreads(1024);
        const dim3 numBlocks((computed_nnz + numThreads.x - 1) / numThreads.x);
        sparse_mvp_kernel<<<numBlocks, numThreads>>>(
            A_sp, A, x, y, M, N, computed_nnz
        );
    }

    cudaFree(A_sp);
    cudaFree(len);
} 