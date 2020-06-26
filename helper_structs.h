#pragma once

#include <cuda_runtime.h>
#include "vec3.h"

//#define PATH_DBG

typedef unsigned int rand_state;

struct mat3x3 {
    vec3 rows[3];
};

struct intersection {
    unsigned int objId; // object that was intersected
    float t;
    vec3 p;
    vec3 normal; // always faces the ray
    bool inside; // true if current path is inside the model
};

struct scatter_info {
    vec3 wi;
    bool specular;
    vec3 throughput;
    bool refracted;
    float t;

    __device__ scatter_info(const intersection& i) : specular(false), throughput(vec3(1, 1, 1)), refracted(false), t(i.t) {}
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

struct triangle {
    triangle() {}
    triangle(vec3 v0, vec3 v1, vec3 v2, int mID) {
        v[0] = v0;
        v[1] = v1;
        v[2] = v2;
        meshID = mID;
        update();
    }

    void update() {
        center = (v[0] + v[1] + v[2]) / 3;
    }

    vec3 v[3];
    vec3 center;
    unsigned char meshID;
};

struct bvh_node {
    __host__ __device__ bvh_node() {}
    bvh_node(const vec3& A, const vec3& B) :a(A), b(B) {}
    __device__ bvh_node(float x0, float y0, float z0, float x1, float y1, float z1) : a(x0, y0, z0), b(x1, y1, z1) {}

    __device__ vec3 min() const { return a; }
    __device__ vec3 max() const { return b; }

    __host__ __device__ unsigned int split_axis() const { return max_component(b - a); }

    vec3 a;
    vec3 b;
};

struct mesh {
    triangle* tris;
    uint32_t numTris;

    bvh_node* bvh;
    int numBvhNodes;

    bbox bounds;

    ~mesh() {
        delete[] tris;
        delete[] bvh;
    }
};


struct scene {
    char* filename;
    mat3x3 mat;
    float scale;
    vec3 camPos;
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