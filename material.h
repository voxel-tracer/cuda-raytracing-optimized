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

__device__ bool scatter_lambertian(const vec3& albedo, const hit_record& rec, vec3& attenuation, ray& scattered, rand_state& state, bool& shadow) {
    vec3 target = rec.normal + random_in_unit_sphere(state);
    scattered = ray(rec.p, target);
    attenuation = albedo;
    shadow = true;
    return true;
}

__device__ bool scatter_metal(const vec3& albedo, float fuzz, const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, rand_state& state, bool& shadow) {
    vec3 reflected = reflect(r_in.direction(), rec.normal);
    scattered = ray(rec.p, reflected + fuzz * random_in_unit_sphere(state));
    attenuation = albedo;
    shadow = false;
    return (dot(scattered.direction(), rec.normal) > 0.0f);
}

__device__ bool scatter_dielectric(float ref_idx, const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, rand_state& state, bool& shadow) {
    vec3 outward_normal;
    vec3 reflected = reflect(r_in.direction(), rec.normal);
    float ni_over_nt;
    attenuation = vec3(1.0, 1.0, 1.0);
    shadow = false;
    vec3 refracted;
    float reflect_prob;
    float cosine;
    if (dot(r_in.direction(), rec.normal) > 0.0f) {
        outward_normal = -rec.normal;
        ni_over_nt = ref_idx;
        cosine = dot(r_in.direction(), rec.normal);
        cosine = sqrt(1.0f - ref_idx * ref_idx * (1 - cosine * cosine));
    }
    else {
        outward_normal = rec.normal;
        ni_over_nt = 1.0f / ref_idx;
        cosine = -dot(r_in.direction(), rec.normal);
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

__device__ bool scatter_coat(const vec3& albedo, float ref_idx, const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, rand_state& state, bool& shadow) {
    vec3 outward_normal;
    vec3 reflected = reflect(r_in.direction(), rec.normal);
    float ni_over_nt;
    attenuation = vec3(1.0, 1.0, 1.0);
    shadow = false;
    float reflect_prob;
    float cosine;
    if (dot(r_in.direction(), rec.normal) > 0.0f) {
        outward_normal = -rec.normal;
        ni_over_nt = ref_idx;
        cosine = dot(r_in.direction(), rec.normal);
        cosine = sqrt(1.0f - ref_idx * ref_idx * (1 - cosine * cosine));
    }
    else {
        outward_normal = rec.normal;
        ni_over_nt = 1.0f / ref_idx;
        cosine = -dot(r_in.direction(), rec.normal);
    }
    reflect_prob = schlick(cosine, ni_over_nt);
    if (rnd(state) < reflect_prob)
        scattered = ray(rec.p, reflected);
    else
        scatter_lambertian(albedo, rec, attenuation, scattered, state, shadow);
    return true;
}

__device__ bool scatter(const material& m, const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, rand_state& state, bool& shadow) {
    switch (m.type)
    {
    case LAMBERTIAN:
        return scatter_lambertian(m.albedo, rec, attenuation, scattered, state, shadow);
    case DIELECTRIC:
        return scatter_dielectric(m.ref_idx, r_in, rec, attenuation, scattered, state, shadow);
    case METAL:
        return scatter_metal(m.albedo, m.fuzz, r_in, rec, attenuation, scattered, state, shadow);
    case COAT:
        return scatter_coat(m.albedo, m.ref_idx, r_in, rec, attenuation, scattered, state, shadow);
    default:
        return false;
    }
}

#endif
