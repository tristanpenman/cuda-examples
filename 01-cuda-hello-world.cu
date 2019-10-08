#include <math.h>
#include <stdio.h>
#include <stdlib.h>

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
    double *h_a = (double *) malloc(bytes);
    double *h_b = (double *) malloc(bytes);
    double *h_c = (double *) malloc(bytes);  // output vector
    for (int i = 0; i < n; i++) {
        h_a[i] = sin(i) * sin(i);
        h_b[i] = cos(i) * cos(i);
    }

    double *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    const int blockSize = 1024;
    const int gridSize = (int)ceil((float)n/blockSize);

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

    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}

