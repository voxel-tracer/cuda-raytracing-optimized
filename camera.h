#ifndef CAMERAH
#define CAMERAH

#include <curand_kernel.h>
#include "ray.h"
#include "helper_structs.h"

__device__ vec3 random_in_unit_disk(curandStatePhilox4_32_10_t  *local_rand_state) {
    vec3 p;
    do {
        p = 2.0f*vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),0) - vec3(1,1,0);
    } while (dot(p,p) >= 1.0f);
    return p;
}

__device__ ray get_ray(const camera& c, float s, float t, curandStatePhilox4_32_10_t* local_rand_state) {
    vec3 rd = c.lens_radius * random_in_unit_disk(local_rand_state);
    vec3 offset = c.u * rd.x() + c.v * rd.y();
    return ray(c.origin + offset, c.lower_left_corner + s * c.horizontal + t * c.vertical - c.origin - offset);
}

#endif
