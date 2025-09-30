#include <cuda_runtime.h>
#include <stdio.h>

__device__ const int DX[] = {0, 0, +1, -1};
__device__ const int DY[] = {-1, +1, 0, 0};

__global__ void move_kernel(
    const int* grid, int* dist, int rows, int cols, 
    int start_row, int start_col, int end_row, int end_col, int base
) {
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;

    const int idx = x * cols + y;

    if (x >= rows || y >= cols || dist[idx] > rows * cols || dist[idx] < base) return;

    for (int i = 0 ; i < 4 ;i++ ){
        int nx = x + DX[i], ny = y + DY[i];
        const int nidx = nx * cols + ny;
        if (0 <= nx && 0 <= ny && nx < rows && ny < cols && !grid[nidx] && dist[nidx] > 1 + dist[idx]) {
            //printf("will move (%d, %d) -> (%d, %d)\n", x, y, nx, ny);
            atomicMin(dist + nidx, 1 + dist[idx]);
        }
    }
}

__host__ void debugDist(int *dist, int rows, int cols) {
    int *h_dist = new int[rows * cols];
    cudaMemcpy(h_dist, dist, sizeof(int) * rows * cols, cudaMemcpyDeviceToHost);
    for (int i = 0 ; i < rows * cols; i++) {
        if (h_dist[i] <= rows * cols)
            printf("%d ", h_dist[i]);
        else 
            printf("inf ");
    }
    printf("\n");
    delete[] h_dist;
}

// grid, result are device pointers
extern "C" void solve(const int* grid, int* result, int rows, int cols, 
                     int start_row, int start_col, int end_row, int end_col) {
    const dim3 numThreads(32, 32);
    const dim3 numBlocks((rows + numThreads.x - 1) / numThreads.x, (cols + numThreads.y - 1) / numThreads.y);

    int *dist;
    cudaMalloc(&dist, rows * cols * sizeof(int));
    cudaMemset(dist, 0x63, rows * cols * sizeof(int)); // set initial distances to high values, too lazy to write a separate kernel for that
    cudaMemset(dist + start_row * cols + start_col, 0, sizeof(int)); // set initial distance to 0 only for start vertex
    
    int h_finalDist = -1;

    for (int i = 0 ; i < rows * cols && h_finalDist < 0; i++) {
        move_kernel<<<numBlocks, numThreads>>>(grid, dist, rows, cols, start_row, start_col, end_row, end_col, i);
        cudaDeviceSynchronize();

        // printf("after %d: ", i); debugDist(dist, rows, cols);
        cudaMemcpy(&h_finalDist, dist + end_row * cols + end_col, sizeof(int), cudaMemcpyDeviceToHost);
        if(h_finalDist > rows * cols) h_finalDist = -1;
    }

    cudaMemcpy(result, &h_finalDist, sizeof(int), cudaMemcpyHostToDevice);

    cudaFree(dist); // be nice
}