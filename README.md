# cuda-raytracing-optimized

## Add VisualStudio 2019 and CUDA 10.2 project

- make a small change to expose M_PI constant. in ve3.h

```
#define _USE_MATH_DEFINES // we need this to get M_PI constant
#include <math.h>
```