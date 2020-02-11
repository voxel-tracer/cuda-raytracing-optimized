#ifndef MATERIALH
#define MATERIALH

struct hit_record;

#include "rnd.h"
#include "ray.h"
#include "helper_structs.h"


__device__ float schlick(float cosine, float ref_idx) {
    float r0 = (1.0f-ref_idx) / (1.0f+ref_idx);
    r0 = r0*r0;
    return r0 + (1.0f-r0)*pow((1.0f - cosine),5.0f);
}

__device__ bool refract(const vec3& v, const vec3& n, float ni_over_nt, vec3& refracted) {
    vec3 uv = unit_vector(v);
    float dt = dot(uv, n);
    float discriminant = 1.0f - ni_over_nt*ni_over_nt*(1-dt*dt);
    if (discriminant > 0) {
        refracted = ni_over_nt*(uv - n*dt) - n*sqrt(discriminant);
        return true;
    }
    else
        return false;
}

__device__ vec3 reflect(const vec3& v, const vec3& n) {
     return v - 2.0f*dot(v,n)*n;
}

__device__ bool scatter_lambertian(const vec3& normal, const vec3& albedo, const hit_record& rec, vec3& attenuation, ray& scattered, rand_state& state) {
    vec3 target = rec.p + normal + random_in_unit_sphere(state);
    scattered = ray(rec.p, target - rec.p);
    attenuation = albedo;
    return true;
}

__device__ bool scatter_metal(const vec3& normal, const vec3& albedo, float fuzz, const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, rand_state& state) {
    vec3 reflected = reflect(unit_vector(r_in.direction()), normal);
    scattered = ray(rec.p, reflected + fuzz * random_in_unit_sphere(state));
    attenuation = albedo;
    return (dot(scattered.direction(), normal) > 0.0f);
}

__device__ bool scatter_dielectric(const vec3& normal, float ref_idx, const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, rand_state& state) {
    vec3 outward_normal;
    vec3 reflected = reflect(r_in.direction(), normal);
    float ni_over_nt;
    attenuation = vec3(1.0, 1.0, 1.0);
    vec3 refracted;
    float reflect_prob;
    float cosine;
    if (dot(r_in.direction(), normal) > 0.0f) {
        outward_normal = -normal;
        ni_over_nt = ref_idx;
        cosine = dot(r_in.direction(), normal) / r_in.direction().length();
        cosine = sqrt(1.0f - ref_idx * ref_idx * (1 - cosine * cosine));
    }
    else {
        outward_normal = normal;
        ni_over_nt = 1.0f / ref_idx;
        cosine = -dot(r_in.direction(), normal) / r_in.direction().length();
    }
    if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted))
        reflect_prob = schlick(cosine, ref_idx);
    else
        reflect_prob = 1.0f;
    if (rnd(state) < reflect_prob)
        scattered = ray(rec.p, reflected);
    else
        scattered = ray(rec.p, refracted);
    return true;
}

__device__ bool scatter(const vec3& normal, const material& m, const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, rand_state& state) {
    switch (m.type)
    {
    case lambertian:
        return scatter_lambertian(normal, m.albedo, rec, attenuation, scattered, state);
    case dielectric:
        return scatter_dielectric(normal, m.ref_idx, r_in, rec, attenuation, scattered, state);
    case metal:
        return scatter_metal(normal, m.albedo, m.fuzz, r_in, rec, attenuation, scattered, state);
    default:
        return false;
    }
}

#endif
