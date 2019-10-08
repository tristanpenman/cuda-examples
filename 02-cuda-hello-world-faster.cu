#include <math.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void prepData(double *a, double *b, size_t n)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] = sin(idx) * sin(idx);
        b[idx] = cos(idx) * cos(idx);
    }
}

__global__ void vecAdd(double *a, double *b, double *c, size_t n)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main()
{
    size_t n = 50000000;
    size_t bytes = n * sizeof(double);
    double *h_c = (double *) malloc(bytes);  // output vector

    double *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    const int blockSize = 1024;
    const int gridSize = (int)ceil((float)n/blockSize);

    prepData<<<gridSize, blockSize>>>(d_a, d_b, n);

    cudaDeviceSynchronize();

    vecAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);

    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    double sum = 0;
    for (int i = 0; i < n; i++) {
        sum += h_c[i];
    }

    printf("final result: %f\n", sum / n);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(h_c);

    return 0;
}

