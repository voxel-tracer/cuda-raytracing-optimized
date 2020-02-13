#ifndef CAMERAH
#define CAMERAH

#include "rnd.h"
#include "ray.h"
#include "helper_structs.h"

__device__ ray get_ray(const camera& c, float s, float t, rand_state& state) {
    vec3 rd = c.lens_radius * random_in_unit_disk(state);
    vec3 offset = c.u * rd.x() + c.v * rd.y();
    return ray(c.origin + offset, c.lower_left_corner + s * c.horizontal + t * c.vertical - c.origin - offset);
}

#endif
