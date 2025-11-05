#include <cuda_runtime.h>
#include <math.h>

__device__ inline float square(float x) { return x * x; }
 
__global__ void mse_kernel(const float *predictions, const float *targets, float *mse, int N) {
    const int batch = threadIdx.x + blockIdx.x * blockDim.x;
    const int i = 4 * batch;
    float sum = 0.0;

    if (i < N) sum += square(predictions[i] - targets[i]);
    if (i + 1 < N) sum += square(predictions[i + 1] - targets[i + 1]);
    if (i + 2 < N) sum += square(predictions[i + 2] - targets[i + 2]);
    if (i + 3 < N) sum += square(predictions[i + 3] - targets[i + 3]);
    
    for (int stride = 16; stride > 0; stride /= 2){
        sum += __shfl_down_sync(0xFFFFFFFF, sum, stride);
    }

    if (threadIdx.x % 32 == 0)
        atomicAdd(mse, sum / (float) N);
}

// predictions, targets, mse are device pointers
extern "C" void solve(const float* predictions, const float* targets, float* mse, int N) {
    const int numThreads(1024);
    const int numBlocks((N + 4 * numThreads - 1) / (4 * numThreads));
    mse_kernel<<<numBlocks, numThreads>>>(predictions, targets, mse, N);
}
