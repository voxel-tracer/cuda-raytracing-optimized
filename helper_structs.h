#pragma once

#include <cuda_runtime.h>
#include "vec3.h"

struct sphere
{
    sphere() {}
    sphere(vec3 c, float r) : center(c), radius(r) {}
    
    vec3 center;
    float radius;
};

enum material_type 
{
    LAMBERTIAN,
    DIELECTRIC,
    METAL,
    COAT,
    CHECKER
};

struct material 
{
    material() {}
    material(material_type t, vec3 a, vec3 b, float f): type(t), albedo(a), albedo2(b), fuzz(f) {}

    material_type type;
    vec3 albedo;
    vec3 albedo2;
    union {
        float fuzz;
        float ref_idx;
        float frequency;
    };
};

material new_lambertian(vec3 albedo);
material new_dielectric(float ref_idx);
material new_metal(vec3 albedo, float fuzz);
material new_coat(vec3 albedo, float ref_idx);
material new_checker(vec3 albedo, vec3 albedo2, float frequency);

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