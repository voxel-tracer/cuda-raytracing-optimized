#ifndef MATERIALH
#define MATERIALH

struct hit_record;

#include "ray.h"


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

#define RANDVEC3 vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))

__device__ vec3 random_in_unit_sphere(curandState *local_rand_state) {
    vec3 p;
    do {
        p = 2.0f*RANDVEC3 - vec3(1,1,1);
    } while (p.squared_length() >= 1.0f);
    return p;
}

__device__ vec3 reflect(const vec3& v, const vec3& n) {
     return v - 2.0f*dot(v,n)*n;
}

enum material_type {
    lambertian,
    dielectric,
    metal
};

class material  {
    public:
        __device__ material(vec3 a) :albedo(a), type(lambertian) {}
        __device__ material(vec3 a, float f) :albedo(a), fuzz(f), type(metal) {}
        __device__ material(float ridx) : ref_idx(ridx), type(dielectric) {}
        __device__ bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState*local_rand_state) const;

        material_type type;
        vec3 albedo;
        union {
            float fuzz;
            float ref_idx;
        };
};

__device__ bool scatter_lambertian(const vec3& albedo, const hit_record& rec, vec3& attenuation, ray& scattered, curandState* local_rand_state) {
    vec3 target = rec.p + rec.normal + random_in_unit_sphere(local_rand_state);
    scattered = ray(rec.p, target - rec.p);
    attenuation = albedo;
    return true;
}

__device__ bool scatter_metal(const vec3& albedo, float fuzz, const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState* local_rand_state) {
    vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
    scattered = ray(rec.p, reflected + fuzz * random_in_unit_sphere(local_rand_state));
    attenuation = albedo;
    return (dot(scattered.direction(), rec.normal) > 0.0f);
}

__device__ bool scatter_dielectric(float ref_idx, const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState* local_rand_state) {
    vec3 outward_normal;
    vec3 reflected = reflect(r_in.direction(), rec.normal);
    float ni_over_nt;
    attenuation = vec3(1.0, 1.0, 1.0);
    vec3 refracted;
    float reflect_prob;
    float cosine;
    if (dot(r_in.direction(), rec.normal) > 0.0f) {
        outward_normal = -rec.normal;
        ni_over_nt = ref_idx;
        cosine = dot(r_in.direction(), rec.normal) / r_in.direction().length();
        cosine = sqrt(1.0f - ref_idx * ref_idx * (1 - cosine * cosine));
    }
    else {
        outward_normal = rec.normal;
        ni_over_nt = 1.0f / ref_idx;
        cosine = -dot(r_in.direction(), rec.normal) / r_in.direction().length();
    }
    if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted))
        reflect_prob = schlick(cosine, ref_idx);
    else
        reflect_prob = 1.0f;
    if (curand_uniform(local_rand_state) < reflect_prob)
        scattered = ray(rec.p, reflected);
    else
        scattered = ray(rec.p, refracted);
    return true;
}

__device__ bool material::scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState* local_rand_state) const {
    switch (type)
    {
    case lambertian:
        return scatter_lambertian(albedo, rec, attenuation, scattered, local_rand_state);
    case dielectric:
        return scatter_dielectric(ref_idx, r_in, rec, attenuation, scattered, local_rand_state);
    case metal:
        return scatter_metal(albedo, fuzz, r_in, rec, attenuation, scattered, local_rand_state);
    default:
        return false;
    }
}

#endif
