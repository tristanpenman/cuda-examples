#include <cmath>
#include <iostream>
#include <vector>

template<typename T>
class DevicePtr {
    T *ptr = nullptr;

    __device__ __host__ __inline__ DevicePtr(T *ptr) : ptr(ptr) { }
public:
    static DevicePtr<T> fromRaw(T *ptr)
    {
        return { ptr };
    }

    __device__ __inline__ T* operator->() const
    {
        return ptr;
    }

    __device__ __inline__ T& operator*() const
    {
        return *ptr;
    }

    __device__ __host__ __inline__ operator T*() const
    {
        return ptr;
    }
};

template<typename T>
__global__ void vecAdd(
    DevicePtr<T> a, DevicePtr<T> b, DevicePtr<T> c,
    size_t n)
{
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main()
{
    const size_t n = 1000;
    const size_t bytes = n * sizeof(double);

    std::vector<double> h_a(n);
    std::vector<double> h_b(n);
    for (double i = 0; i < n; i++) {
        h_a[i] = sin(i) * sin(i);
        h_b[i] = cos(i) * cos(i);
    }

    double *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

    const size_t blockDim = 1024;
    const size_t gridDim = static_cast<size_t>(
        ceil(static_cast<float>(n) / blockDim));

    vecAdd<double><<<gridDim, blockDim>>>(
        DevicePtr<double>::fromRaw(d_a),
        DevicePtr<double>::fromRaw(d_b),
        DevicePtr<double>::fromRaw(d_c),
        n);

    std::vector<double> h_c(n);
    cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);
    double sum = 0;
    for (double i = 0; i < n; i++) {
        sum += h_c[i];
    }

    std::cout << "final result: " << sum << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
