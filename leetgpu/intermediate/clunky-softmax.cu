#include <cuda_runtime.h>
#include <stdio.h>

__device__ float myadd(float a, float b) { return a + b; }
__device__ float mymax(float a, float b) { return a > b ? a : b; }

enum class Reduction {
    Sum,
    Max,
};

enum class Mapping {
    Exp,
    Div,
};

__global__ void map_inplace_kernel(float *input, int N, Mapping mapping, float arg) {
    const int i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i >= N) return;

    if (mapping == Mapping::Exp) {
        input[i] = __expf(input[i] - arg);
    } else if (mapping == Mapping::Div) {
        input[i] /= arg;
    }
}

__global__ void reduce_kernel(const float* input, float* output, int N, Reduction red_type) {
    extern __shared__ float sblock[]; // shmem of size #threads / block

    const int i = threadIdx.x + blockDim.x * blockIdx.x;

    if (i >= N) return; // guarantees threadIdx.x < N too
    
    sblock[threadIdx.x] = input[i];
    __syncthreads(); // ensure all threads are operating on the same non-stale data

    // printf("copying %d values into shmem (this was %d in block %d)\n", min(N - blockDim.x * blockIdx.x, blockDim.x), threadIdx.x, blockIdx.x);

    // resolve the reduction type - i couldn't get it to work by passing the function pointer directly
    // so we do an indirection via enum instead
    float (*red_func)(float, float);
    
    if (red_type == Reduction::Sum) red_func = myadd;
    else if (red_type == Reduction::Max) red_func = mymax;
    else red_func = nullptr; // should never happen

    for (int span = blockDim.x / 2; span > 0 ; span /= 2) {
        if (threadIdx.x < span && i + span < N) { // last condition to ensure proper init
            // printf("will access shmem[%d] and shmem[%d]\n", threadIdx.x, threadIdx.x + span);
            sblock[threadIdx.x] = red_func(sblock[threadIdx.x], sblock[threadIdx.x + span]);
        }
        __syncthreads();
    }

    output[blockIdx.x] = sblock[0]; // output must have at least #blocks entries
    if (!threadIdx.x) printf("block %d reduced to %f\n", blockIdx.x, sblock[0]);
}

__global__ void copy_kernel(const float* input, float* output, int N) {
    const int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= N) return;
    output[i] = input[i];
}

__host__ float reduce(const float *input, int blocksPerGrid, int threadsPerBlock, int N, Reduction reduction) {
    // buffer for reduction
    float *reduction_buff[2];
    int buffer_size = max(blocksPerGrid, threadsPerBlock);

    cudaMalloc(&reduction_buff[0], buffer_size * sizeof(float));
    cudaMalloc(&reduction_buff[1], buffer_size * sizeof(float));

    float *next_output;
    
    printf("allocated ping-pong buffers for reduction (size %d)\n", buffer_size);

    for (int num_el = N, i = 0; num_el > 1 || !i; i++) { // see end of loop for the change in num_el!
        const float *next_input = i == 0 ? input : next_output;
        next_output = reduction_buff[i % 2];
        const int next_num_blocks = (num_el + threadsPerBlock - 1) / threadsPerBlock;
        printf("reduction along first %d elements (threads=%d, blocks=%d)\n", num_el, threadsPerBlock, next_num_blocks);
        reduce_kernel<<<next_num_blocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(
            next_input, next_output, num_el, reduction
        );
        cudaDeviceSynchronize();
        num_el = next_num_blocks; // each block gets a slot; we assign num_el here cause we compute next_num_blocks inside the loop
    }

    float value;
    cudaMemcpy(&value, next_output, sizeof(*next_output), cudaMemcpyDeviceToHost);

    printf("computed value via reduction: %f", value);

    cudaFree(reduction_buff[0]);
    cudaFree(reduction_buff[1]);

    return value;
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 2;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    float max_value = reduce(input, blocksPerGrid, threadsPerBlock, N, Reduction::Max);

    copy_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    map_inplace_kernel<<<blocksPerGrid, threadsPerBlock>>>(output, N, Mapping::Exp, max_value);

    float sum = reduce(output, blocksPerGrid, threadsPerBlock, N, Reduction::Sum);
    map_inplace_kernel<<<blocksPerGrid, threadsPerBlock>>>(output, N, Mapping::Div, sum);

    cudaDeviceSynchronize();
}