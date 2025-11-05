#include <cuda_runtime.h>
#include <stdio.h>

__global__ void sum_kernel(const float* input, float* output, int N) {
    extern __shared__ float sblock[]; // shmem of size #threads / block

    const int i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i >= N) return; // guarantees threadIdx.x < N too
    
    sblock[threadIdx.x] = input[i];
    __syncthreads(); // ensure all threads are operating on the same non-stale data

    for (int span = blockDim.x / 2; span > 0 ; span /= 2) {
        if (threadIdx.x < span && i + span < N) { // last condition to ensure proper init
            // printf("will access shmem[%d] and shmem[%d]\n", threadIdx.x, threadIdx.x + span);
            sblock[threadIdx.x] = sblock[threadIdx.x] + sblock[threadIdx.x + span];
        }
        __syncthreads();
    }

    output[blockIdx.x] = sblock[0]; // output must have at least #blocks entries
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 32;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // buffer for reduction
    float *reduction_buff[2];
    int buffer_size = max(blocksPerGrid, threadsPerBlock);

    cudaMalloc(&reduction_buff[0], buffer_size * sizeof(float));
    cudaMalloc(&reduction_buff[1], buffer_size * sizeof(float));

    float *next_output;
    const float *last_output = input;
    int num_el = N, i = 0;

    do {
        next_output = reduction_buff[i % 2];
        const int next_num_blocks = (num_el + threadsPerBlock - 1) / threadsPerBlock;
        sum_kernel<<<next_num_blocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
            last_output, next_output, num_el
        );
        last_output = next_output;
        i++;
        num_el = next_num_blocks;
    } while (num_el > 1);
    
    cudaMemcpy(output, next_output, sizeof(float), cudaMemcpyDeviceToDevice);
    
    // skip the next two lines for YOLO
    cudaFree(reduction_buff[0]);
    cudaFree(reduction_buff[1]);

    cudaDeviceSynchronize();
}