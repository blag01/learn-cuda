#include <cuda_runtime.h>

__global__ void sum_multiples_kernel(const float *samples, const int n_samples, float *result, float avg) {
    int i = 4 * (threadIdx.x + blockIdx.x * blockDim.x);
    if (i >= n_samples) return;
    float sum;

    if (i + 4 < n_samples) {
        sum = avg * samples[i] + avg * samples[i + 1] + avg * samples[i + 2] + avg * samples[i + 3];
    } else {
        for (; i < n_samples; i++) {
            sum += avg * samples[i];
        }
    }

    atomicAdd(result, sum);
}

// y_samples, result are device pointers
extern "C" void solve(const float* y_samples, float* result, float a, float b, int n_samples) {
    float w = b - a;
    float avg = w / static_cast<float>(n_samples);

    const int numThreads = 128;
    const int numBlocks = (n_samples + 4 * numThreads - 1) / numThreads;

    sum_multiples_kernel<<<numBlocks, numThreads>>>(y_samples, n_samples, result, avg);
}
