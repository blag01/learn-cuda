#include <cuda_runtime.h>
#include <stdio.h>

__global__ void ce_loss_kernel(const float *logits, const int *true_labels, float *loss, int N, int C) {
    const int lane = threadIdx.x;
    const int sample = threadIdx.y + blockIdx.y * blockDim.y;
    if (sample >= N || lane >= C) return;
    float lse = 0.0;
    for (int i = lane; i < C; i += blockDim.x) {
        lse += __expf(logits[sample * C + i]);
    }
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        lse += __shfl_down_sync(0xFFFFFFFF, lse, stride);
    }
    if (!lane) {
        //printf("log sum exp #%d = %.2lf\n", sample, log(lse));
        lse = log(lse) - logits[sample * C + true_labels[sample]];
        atomicAdd(loss, 1. / static_cast<float>(N) * lse);
    }
}

// logits, true_labels, loss are device pointers
extern "C" void solve(const float* logits, const int* true_labels, float* loss, int N, int C) {
    const dim3 numThreads(32, 32);
    const dim3 numBlocks(
        1,
        (N + numThreads.y - 1) / numThreads.y
    );

    ce_loss_kernel<<<numBlocks, numThreads>>>(logits, true_labels, loss, N, C);
}