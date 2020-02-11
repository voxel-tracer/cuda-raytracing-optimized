#pragma once

#include <math_functions.h>
#include "vec3.h"

typedef unsigned int rand_state;

__device__ unsigned int xor_shift_32(rand_state& state)
{
    unsigned int x = state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 15;
    state = x;
    return x;
}

__device__ float rnd(rand_state& state)
{
    return (xor_shift_32(state) & 0xFFFFFF) / 16777216.0f;
}

__device__ vec3 random_in_unit_disk(rand_state& state) {
    vec3 p;
    do {
        p = 2.0f * vec3(rnd(state), rnd(state), 0) - vec3(1, 1, 0);
    } while (dot(p, p) >= 1.0f);
    return p;
}

/*
* based off http://www.reedbeta.com/blog/quick-and-easy-gpu-random-numbers-in-d3d11/
*/
__device__ rand_state wang_hash(rand_state seed)
{
    seed = (seed ^ 61) ^ (seed >> 16);
    seed *= 9;
    seed = seed ^ (seed >> 4);
    seed *= 0x27d4eb2d;
    seed = seed ^ (seed >> 15);
    return seed;
}

#define RANDVEC3 vec3(rnd(state), rnd(state), rnd(state))

__device__ vec3 random_in_unit_sphere(rand_state& state) {
    float z = rnd(state) * 2.0f - 1.0f;
    float t = rnd(state) * 2.0f * M_PI;
    float r = sqrtf(fmaxf(0.0, 1.0f - z * z));
    float c, s;
    sincosf(t, &s, &c);
    vec3 res = vec3(r * c, r * s, z);
    res *= cbrtf(rnd(state));
    return res;
}
