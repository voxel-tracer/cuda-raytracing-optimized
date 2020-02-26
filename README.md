# cuda-raytracing-optimized

This is based off Roger Allen's [Raytracing in a weekend in CUDA](https://github.com/rogerallen/raytracinginoneweekendincuda)

On my machine (laptop with a GTX 1050 card) original code could render 1200x800 image with 10 samples per pixels in 18s.
Optimized version can render the same image in less than a second.
The following describes the major changes I made and the effect they had on the renderer performance.

## Add VisualStudio 2019 and CUDA 10.2 project

make a small change to expose M_PI constant. in ve3.h

```
#define _USE_MATH_DEFINES // we need this to get M_PI constant
#include <math.h>
```

## Separate CUDA and CPP code
Render time for 1200x800 10spp is 18.27s

Moving kernel code to a separate file will help us in the long term as we add more features to the renderer. 

Main change is introducing a couple of host functions that will call the kernels. Those functions' declarations need to be prefixed with:

```
extern "C"
```

## Change how we initialize CuRAND
Render time for 1200x800 10spp is 11.47s

Using nvprof.exe to analyse our application we see the following:
```
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   57.20%  11.1503s         1  11.1503s  11.1503s  11.1503s  render(vec3*, int, int, int, camera**, hitable**, curandStateXORWOW*)
                   42.60%  8.30312s         1  8.30312s  8.30312s  8.30312s  render_init(int, int, curandStateXORWOW*)
                    0.12%  22.513ms         1  22.513ms  22.513ms  22.513ms  create_world(hitable**, hitable**, camera**, int, int, curandStateXORWOW*)
                    0.09%  16.661ms         1  16.661ms  16.661ms  16.661ms  free_world(hitable**, hitable**, camera**)
                    0.00%  1.7920us         1  1.7920us  1.7920us  1.7920us  rand_init(curandStateXORWOW*)
```
Turns out `render_init` kernel is taking 42% of total rendering time, which is too much for a kernel that's just initializing cuRand.
Fortunately, the original author already figured out the problemsomeone already figured out the cause:
https://github.com/rogerallen/raytracinginoneweekendincuda/issues/2

Following the same fix we can see that `render` kernel is now taking 99% of rendering time, which is expected:
```
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   99.59%  9.93058s         1  9.93058s  9.93058s  9.93058s  render(vec3*, int, int, int, camera**, hitable**, curandStateXORWOW*)
                    0.23%  22.498ms         1  22.498ms  22.498ms  22.498ms  create_world(hitable**, hitable**, camera**, int, int, curandStateXORWOW*)
                    0.17%  16.786ms         1  16.786ms  16.786ms  16.786ms  free_world(hitable**, hitable**, camera**)
                    0.02%  1.6980ms         1  1.6980ms  1.6980ms  1.6980ms  render_init(int, int, curandStateXORWOW*)
                    0.00%  1.9200us         1  1.9200us  1.9200us  1.9200us  rand_init(curandStateXORWOW*)
```

- 64 registers, occupancy 40%/50%
- kernel is bound by latency, registers are main occupancy limiter
 . memory ~45%
 . compute ~25%
- local memory overhead 92%
- warp execution efficiency 43.8%

## Use compute_61, sm_61
Render time for 1200x800 10spp is 10.65s

- register usage went down to 56!
- occupancy 45.2% / 56.2%
- by enabling verbose PTXAS output (--ptxas-options=v) we see that virtual functions seem to be using local memory, let's remove inheritance from spheres/materials

## Remove virtual functions/inheritance
Render time for 1200x800 10spp is 2.1s
Render time for 1200x800 100spp is 21.1s

- 106 registers, occupancy 19.5%/25%
- no more local memory overhead
- kernel is still bound by latency
	memory	15%
	compute	20%
- local loads/store went down from 1.5B/635M to 0
- device memory total read/write went down from 242M instr to 1M
- given we have a relatively small scenes and it never changes, it may be a good candidate for constant memory
  as a first step we are going to move the scene initialization to the host

## Setup scene and camera on host
Render time for 1200x800 100spp is 13.13s

- 79 registers, occupancy 29.8%/37.5%									
- kernel is bound by latency, registers are main occupancy limiter
 . memory ~35%
 . compute ~33%
- PC sampling: memory dependency 61.54%
- fewer registers used, better occupancy

## Copy spheres to const memory
Render time for 1200x800 100spp is 6.48s

- kernel is bound by latency, registers are main occupancy limiter
 . memory ~15%
 . compute ~55%
  . load/store instruction unit ~35%
- compute is limited by low warp execution efficiency 41.2% (caused by divergence)
- kernel memory
 . L2 cache total went from 78M transactions to 3.5M
 . Unified cache total went from 1B transactions to 5.5M
- biggest change left is using a custom rng.
