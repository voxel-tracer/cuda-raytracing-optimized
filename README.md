# cuda-raytracing-optimized

## Add VisualStudio 2019 and CUDA 10.2 project

- make a small change to expose M_PI constant. in ve3.h

```
#define _USE_MATH_DEFINES // we need this to get M_PI constant
#include <math.h>
```

## Separate CUDA and CPP code

Moving kernel code to a separate file will help us in the long term
as we add more features to the renderer. Main change is introducing
a couple of host functions that will call the kernels. Those functions
declarations need to be prefixed with:
```
extern "C"
```

## Optimization 1: use a different curand generator

Let's run nvprof on our program and see what's using most of the rendering time:
```
$ nvprof ./x64/Release/cuda-raytracing-optimized.exe
Rendering a 1200x800 image with 10 samples per pixel in 8x8 blocks.
took 19.412 seconds.
==3324== NVPROF is profiling process 3324, command: ./x64/Release/cuda-raytracing-optimized.exe
==3324== Profiling application: ./x64/Release/cuda-raytracing-optimized.exe
==3324== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   57.19%  11.1202s         1  11.1202s  11.1202s  11.1202s  render(vec3*, int, int, int, camera**, hitable**, curandStateXORWOW*)
                   42.60%  8.28334s         1  8.28334s  8.28334s  8.28334s  render_init(int, int, curandStateXORWOW*)
                    0.12%  22.521ms         1  22.521ms  22.521ms  22.521ms  create_world(hitable**, hitable**, camera**, int, int, curandStateXORWOW*)
                    0.09%  16.543ms         1  16.543ms  16.543ms  16.543ms  free_world(hitable**, hitable**, camera**)
                    0.00%  1.8240us         1  1.8240us  1.8240us  1.8240us  rand_init(curandStateXORWOW*)
      API calls:   97.76%  19.4312s         5  3.88623s  13.600us  11.1229s  cudaDeviceSynchronize
                    1.80%  357.40ms         1  357.40ms  357.40ms  357.40ms  cudaMallocManaged
                    0.29%  57.263ms         1  57.263ms  57.263ms  57.263ms  cudaDeviceReset
                    0.09%  18.533ms         6  3.0889ms  10.300us  16.676ms  cudaFree
                    0.03%  6.4706ms         5  1.2941ms  29.800us  4.2203ms  cudaLaunchKernel
                    0.02%  4.7847ms         5  956.94us  10.000us  3.9265ms  cudaMalloc
                    0.00%  597.40us        97  6.1580us     100ns  272.50us  cuDeviceGetAttribute
                    0.00%  25.300us         1  25.300us  25.300us  25.300us  cuDeviceTotalMem
                    0.00%  17.700us         1  17.700us  17.700us  17.700us  cuDeviceGetPCIBusId
                    0.00%  3.6000us         5     720ns     300ns  1.2000us  cudaGetLastError
                    0.00%  1.4000us         3     466ns     200ns     800ns  cuDeviceGetCount
                    0.00%  1.4000us         2     700ns     200ns  1.2000us  cuDeviceGet
                    0.00%     600ns         1     600ns     600ns     600ns  cuDeviceGetName
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuDeviceGetUuid
```

`render_init()` is using **42%** of the rendering time which is too much given that the kernel is only calling `curand_init()`

According to CUDA documentation, curand state setup can be an expensive operation. The default curand generator is `curandStateXORWOW`
using a different generator can generally speedup the setup. Let's try `curandStatePhilox4_32_10`:

```
Rendering a 1200x800 image with 10 samples per pixel in 8x8 blocks.
took 9.931 seconds.
==22200== NVPROF is profiling process 22200, command: ./x64/Release/cuda-raytracing-optimized.exe
==22200== Profiling application: ./x64/Release/cuda-raytracing-optimized.exe
==22200== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   99.57%  9.92859s         1  9.92859s  9.92859s  9.92859s  render(vec3*, int, int, int, camera**, hitable**, curandStatePhilox4_32_10*)
                    0.24%  23.975ms         1  23.975ms  23.975ms  23.975ms  create_world(hitable**, hitable**, camera**, int, int, curandStatePhilox4_32_10*)
                    0.16%  16.399ms         1  16.399ms  16.399ms  16.399ms  free_world(hitable**, hitable**, camera**)
                    0.02%  2.1433ms         1  2.1433ms  2.1433ms  2.1433ms  render_init(int, int, curandStatePhilox4_32_10*)
                    0.00%  4.3520us         1  4.3520us  4.3520us  4.3520us  rand_init(curandStatePhilox4_32_10*)
      API calls:   97.30%  9.95541s         5  1.99108s  13.300us  9.92871s  cudaDeviceSynchronize
                    1.95%  199.78ms         1  199.78ms  199.78ms  199.78ms  cudaMallocManaged
                    0.49%  49.969ms         1  49.969ms  49.969ms  49.969ms  cudaDeviceReset
                    0.18%  18.375ms         6  3.0625ms  15.400us  16.547ms  cudaFree
                    0.06%  5.8682ms         5  1.1736ms  14.000us  4.8471ms  cudaMalloc
                    0.01%  1.5005ms         5  300.10us  16.100us  1.2840ms  cudaLaunchKernel
                    0.00%  499.10us        97  5.1450us     100ns  256.20us  cuDeviceGetAttribute
                    0.00%  28.500us         1  28.500us  28.500us  28.500us  cuDeviceTotalMem
                    0.00%  13.100us         1  13.100us  13.100us  13.100us  cuDeviceGetUuid
                    0.00%  11.400us         1  11.400us  11.400us  11.400us  cuDeviceGetPCIBusId
                    0.00%  3.3000us         5     660ns     200ns  1.2000us  cudaGetLastError
                    0.00%  3.1000us         3  1.0330us     500ns  1.9000us  cuDeviceGetCount
                    0.00%  1.8000us         2     900ns     200ns  1.6000us  cuDeviceGet
                    0.00%     600ns         1     600ns     600ns     600ns  cuDeviceGetName
                    0.00%     300ns         1     300ns     300ns     300ns  cuDeviceGetLuid
```

Sure enough the rendering time went down from ~19s to ~10s and now `render()` kernel takes **+99%** of rendering time.