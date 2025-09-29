#include <cuda_runtime.h>

__global__ void partial_sort(float *data, int N, int half_width) {
    const int index = threadIdx.x + blockIdx.x * blockDim.x;
    const int begin = index * 2 * half_width;
    const int mid = index + half_width;
    const int end = begin + 2 * half_width;
    
    if (mid >= N) return; // nothing to do

    // assumption: [begin, mid) and [mid, end) are already sorted
    // so all we need to do is merge them together

    int l = begin, r = mid;

    while (l < mid && r < end) {
        if (data[l] <= data[r]) l++;
        else {
            // swap the smaller one to the left
            float temp = data[l];
            data[l] = data[r];
            data[r] = temp;
            r++;
        }
    }
}

// data is device pointer
extern "C" void solve(float* data, int N) {
    for (int half_width = 1; half_width * 2 <= N; half_width *= 2) {
        const int numThreads = 128;
        const int span = numThreads * 2 * half_width;
        const int numBlocks = (N + span - 1) / span;
        partial_sort<<<numBlocks, numThreads>>>(data, N, half_width);
    }
}