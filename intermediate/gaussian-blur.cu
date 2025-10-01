#include <cuda_runtime.h>

__global__ void gaussian_blur_kernel(const float *input, const float *kernel, float *output,
int input_rows, int input_cols, int kernel_rows, int kernel_cols) {

    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= input_rows || y >= input_cols) return;

    const int offset_x = x - kernel_rows / 2;
    const int offset_y = y - kernel_cols / 2;

    float smooth = 0;

    for (int i = 0; i < kernel_rows; i++) {
        for (int j = 0 ; j < kernel_cols ; j++) {
            int this_x = offset_x + i;
            int this_y = offset_y + j;

            if (this_x < 0 || this_y < 0 || this_x >= input_rows || this_y >= input_cols) continue;

            smooth += kernel[i * kernel_rows + j] * input[this_x * input_rows + this_y];
        }
    }

    output[x * input_rows + y] = smooth;
}

// input, kernel, output are device pointers
extern "C" void solve(const float* input, const float* kernel, float* output,
           int input_rows, int input_cols, int kernel_rows, int kernel_cols) {
    const dim3 numThreads(32,32);
    const dim3 numBlocks(
        (input_rows + numThreads.x - 1) / numThreads.x,
        (input_cols + numThreads.y - 1) / numThreads.y
    );
    gaussian_blur_kernel<<<numBlocks, numThreads>>>(
        input, kernel, output, input_rows, input_cols, kernel_rows, kernel_cols
    );
}
