#include <cmath>
#include <iostream>
#include <vector>

template<typename T>
class DevicePtr
{
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
class DeviceMem
{
    T *ptr = nullptr;
    size_t bytes;

    DeviceMem(std::size_t bytes) : bytes(bytes)
    {
        cudaMalloc(&ptr, bytes);
    }

public:
    static DeviceMem alloc(std::size_t numElements)
    {
        return { numElements * sizeof(T) };
    }

    ~DeviceMem()
    {
        cudaFree(ptr);
    }

    operator DevicePtr<T>() const
    {
        return DevicePtr<T>::fromRaw(ptr);
    }

    operator DevicePtr<const T>() const
    {
        return DevicePtr<const T>::fromRaw(ptr);
    }

    size_t size() const
    {
        return bytes;
    }
};

template<typename T>
void copyElements(DeviceMem<T> &dst, const T *src)
{
    cudaMemcpy(dst.operator DevicePtr<T>(), src, dst.size(),
        cudaMemcpyHostToDevice);
}

template<typename T>
void copyElements(T *dst, DeviceMem<T> &src)
{
    cudaMemcpy(dst, src.operator DevicePtr<T>(), src.size(),
        cudaMemcpyDeviceToHost);
}

template<typename T>
__global__ void vecAdd(
    DevicePtr<const T> a, DevicePtr<const T> b, DevicePtr<T> c,
    size_t n)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main()
{
    const size_t n = 1000;

    std::vector<double> h_a(n);
    std::vector<double> h_b(n);
    for (double i = 0; i < n; i++) {
        h_a[i] = sin(i) * sin(i);
        h_b[i] = cos(i) * cos(i);
    }

    auto d_a = DeviceMem<double>::alloc(n);
    auto d_b = DeviceMem<double>::alloc(n);
    auto d_c = DeviceMem<double>::alloc(n);

    copyElements(d_a, h_a.data());
    copyElements(d_b, h_b.data());

    const size_t blockDim = 1024;
    const size_t gridDim = static_cast<size_t>(
        ceil(static_cast<float>(n) / blockDim));

    vecAdd<double><<<gridDim, blockDim>>>(d_a, d_b, d_c, n);

    std::vector<double> h_c(n);
    copyElements(h_c.data(), d_c);
    double sum = 0;
    for (double i = 0; i < n; i++) {
        sum += h_c[i];
    }

    std::cout << "final result: " << sum << std::endl;
}
