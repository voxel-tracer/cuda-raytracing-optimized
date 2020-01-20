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

