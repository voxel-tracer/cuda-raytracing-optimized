#pragma once

#include <cuda_runtime.h>
#include "vec3.h"

//#define PATH_DBG

typedef unsigned int rand_state;

struct intersection {
    unsigned int objId; // object that was intersected
    float t;
    vec3 p;
    vec3 normal; // always faces the ray
    bool inside; // true if current path is inside the model
};

struct scatter_info {
    vec3 wi;
    bool specular = false;
    vec3 throughput = vec3(1, 1, 1);
    bool refracted = false;
};

struct path {
    vec3 origin;
    vec3 rayDir;
    vec3 color; // accumulated direct illumination
    bool specular;

    vec3 shadowDir;
    vec3 lightContribution; // direct illumination that will be added to color if this path's shadow ray reaches the light

    uint32_t pixelId;
    uint8_t bounce;
    vec3 attenuation;
    rand_state rng;
    bool inside = false;
#ifdef PATH_DBG
    bool dbg = false;
#endif
};

struct bbox {
    vec3 min;
    vec3 max;

    bbox() {}
    __host__ __device__ bbox(vec3 _min, vec3 _max) :min(_min), max(_max) {}
};

struct grid {
    vec3 size;
    float cellSize;
    uint16_t* C; // C[i] start index of this cell's triangles in L
    uint16_t* L; // triangles indices for all cells

    __host__ __device__ uint16_t sizeC() const {
        return size.x() * size.y() * size.z() + 1;
    }
    __host__ __device__ uint16_t sizeL() const {
        return C[sizeC() - 1];
    }
};

struct mesh {
    vec3* tris;
    uint16_t numTris;
    bbox bounds;

    grid g;
};

struct plane {
    vec3 norm;
    vec3 point; // point in the plane

    plane() {}
    plane(vec3 p, vec3 n) :point(p), norm(unit_vector(n)) {}
};

struct sphere
{
    sphere() {}
    sphere(vec3 c, float r) : center(c), radius(r) {}
    
    vec3 center;
    float radius;
};

struct tri_hit {
    unsigned int triId; // triangle that was intersected
    float u, v;
};

struct camera
{
    camera() {}
    camera(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aspect, float aperture, float focus_dist) { // vfov is top to bottom in degrees
        lens_radius = aperture / 2.0f;
        float theta = vfov * ((float)M_PI) / 180.0f;
        float half_height = tan(theta / 2.0f);
        float half_width = aspect * half_height;
        origin = lookfrom;
        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);
        lower_left_corner = origin - half_width * focus_dist * u - half_height * focus_dist * v - focus_dist * w;
        horizontal = 2.0f * half_width * focus_dist * u;
        vertical = 2.0f * half_height * focus_dist * v;
    }

    vec3 origin;
    vec3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;
    vec3 u, v, w;
    float lens_radius;

};