# CUDA Examples

This repo contains a collection of CUDA examples that were first used for a talk at the Melbourne C++ Meetup.

## Listing

1. [00-hello-world.cu](00-hello-world.cu) - Vector addition on a CPU; the hello world of the parallel computing
2. [01-cuda-hello-world.cu](01-cuda-hello-world.cu) - Vector addition using CUDA
3. [02-cuda-hello-world-faster.cu](02-cuda-hello-world-faster.cu) - Vector addition using CUDA, with a CPU bottleneck removed
4. [03-templates-device-ptr.cu](03-templates-device-ptr.cu) - Using C++ templates to make device pointers safer
5. [04-templates-device-mem.cu](04-templates-device-mem.cu) - More templates, allowing for scoped device memory
6. [05-thrust-rand-vectors.cu](05-thrust-rand-vectors.cu) - Using Thrust to add up random numbers

## Compiling

All examples can be compiled with `nvcc`. Only `02-cuda-hello-world-faster.cu` requires an additional compiler option `--expt-relaxed-constexpr` (at least, when compiled on Linux).

A Makefile has been included, so all examples can built using `make`.

## Credits

These are all based on examples found in the wild. 03 and 04, in particular, are based on code from Michael Gopshtein's CppCon talk, [CUDA Kernels in C++](https://www.youtube.com/watch?v=HIJTRrm9nzY). And examples 01 and 02 are based on the Vector Addition sample code included in the CUDA Toolkit.

## License

It seems pretty reasonable to me to consider this example code to be in the public domain.
