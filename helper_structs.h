#pragma once

#include <vector>
#include <cuda_runtime.h>
#include "../bvh-builder/geometry.h"

//#define PATH_DBG
//#define BVH_COUNT
//#define COLOR_NUM_NODES

typedef unsigned int rand_state;

struct mat3x3 {
    vec3 rows[3];
};

struct intersection {
    unsigned int objId; // object that was intersected
    unsigned char meshID;
    uint64_t triID;
    float t;
    vec3 p;
    vec3 normal; // always faces the ray
    bool inside; // true if current path is inside the model
    float texCoords[2];
#ifdef BVH_COUNT
    uint64_t traversed = 0;
#endif
#ifdef SAVE_BITSTACK
    unsigned int bitstack = 0;
#endif
#ifdef COLOR_NUM_NODES
    uint64_t numNodes;
#endif // COLOR_NUM_NODES


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

    //uint32_t pixelId;
    uint8_t bounce;
    vec3 attenuation;
    rand_state rng;
    bool inside = false;
    bool done = false;
#ifdef PATH_DBG
    bool dbg = false;
    uint64_t sampleId;
#endif
#ifdef SAVE_BITSTACK
    unsigned int bitstack = 0;
#endif // SAVE_BITSTACK

};

struct mesh {
    std::vector<LinearTriangle> tris;
    std::vector<LinearBVHNode> nodes;
};

enum material_type {
    DIFFUSE,
    METAL,
    GLASS
};

struct material {
    material_type type;
    vec3 color;
    float param; // fuzz
    int texId;
};

struct stexture {
    float* data;
    int width;
    int height;

    stexture() {}
    stexture(float* data, int width, int height) : data(data), width(width), height(height) {}
};

struct scene {
    char* filename;
    mat3x3 mat;
    float scale;
    vec3 camPos;
    material* materials;
    int numMaterials;
    stexture* textures;
    int numTextures;
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
#ifdef BVH_COUNT
    uint64_t traversed;
#endif
#ifdef SAVE_BITSTACK
    unsigned int bitstack = 0;
#endif
#ifdef COLOR_NUM_NODES
    uint64_t numNodes;
#endif // COLOR_NUM_NODES
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

struct kernel_scene {
    mesh* m;
    plane floor;

    material* materials;
    int numMaterials;

    stexture* textures;
    int numTextures;

    int numPrimitivesPerLeaf;
};
