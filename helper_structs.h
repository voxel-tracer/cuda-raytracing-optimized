#pragma once

#include "vec3.h"


struct block {
    uint16_t coords;
    uint64_t voxels;
    uint16_t idx;

    block() {}
    block(uint16_t c, uint64_t v, uint16_t i) : coords(c), voxels(v), idx(i) {}
};

struct voxelModel {
    int numVoxels;
    int numBlocks;
    int numUBlocks;

    block* blocks;
    block* ublocks;
    uint3 center;
};

struct sphere
{
    sphere() {}
    sphere(vec3 c, float r) : center(c), radius(r) {}
    
    vec3 center;
    float radius;
};

enum material_type 
{
    lambertian,
    dielectric,
    metal
};

struct material 
{
    material() {}
    material(vec3 a) : type(material_type::lambertian), albedo(a), fuzz(0) {}
    material(vec3 a, float f) : type(material_type::metal), albedo(a), fuzz(f) {}
    material(float ridx) : type(material_type::dielectric), ref_idx(ridx) {}

    material_type type;
    vec3 albedo;
    union {
        float fuzz;
        float ref_idx;
    };
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