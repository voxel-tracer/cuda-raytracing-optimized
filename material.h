#ifndef MATERIALH
#define MATERIALH

struct hit_record;

#include "rnd.h"
#include "ray.h"
#include "helper_structs.h"


__device__ float schlick(float cosine, float ref_idx) {
    float r0 = (1.0f-ref_idx) / (1.0f+ref_idx);
    r0 = r0*r0;
    return r0 + (1.0f - r0) * pow((1.0f - cosine), 5.0f);
}

__device__ vec3 refract(const vec3& uv, const vec3& n, float etai_over_etat) {
    float cos_theta = fminf(dot(-uv, n), 1.0);
    vec3 r_out_parallel = etai_over_etat * (uv + cos_theta * n);
    vec3 r_out_perp = -sqrt(1.0f - r_out_parallel.squared_length()) * n;
    return r_out_parallel + r_out_perp;
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

// simplified checker that assumes a plane with normal = (0, 0, 1)
__device__ bool scatter_checker(const vec3& albedo1, const vec3& albedo2, float frequency, const hit_record& hit, vec3& attenuation, ray& scattered, rand_state& state, bool& shadow) {
    auto sines = sin(frequency * hit.p.x()) * sin(frequency * hit.p.y()) * sin(frequency * hit.p.z());
    if (sines < 0)
        return scatter_lambertian(albedo1, hit, attenuation, scattered, state, shadow);
    else
        return scatter_lambertian(albedo2, hit, attenuation, scattered, state, shadow);
}

__device__ bool scatter_metal(const vec3& albedo, float fuzz, const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, rand_state& state, bool& shadow) {
    vec3 reflected = reflect(r_in.direction(), rec.normal);
    scattered = ray(rec.p, reflected + fuzz * random_in_unit_sphere(state));
    attenuation = albedo;
    shadow = false;
    return (dot(scattered.direction(), rec.normal) > 0.0f);
}

__device__ bool scatter_dielectric(float ref_idx, const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, rand_state& state, bool& shadow) {
    attenuation = vec3(1.0, 1.0, 1.0);
    shadow = false;
    bool frontFace = dot(r_in.direction(), rec.normal) < 0.0f;
    vec3 hitNormal = frontFace ? rec.normal : -rec.normal;
    float etai_over_etat = frontFace ? (1.0f / ref_idx) : ref_idx;

    float cos_theta = fminf(dot(-r_in.direction(), hitNormal), 1.0f);
    float sin_theta = sqrt(1.0f - cos_theta * cos_theta);
    if (etai_over_etat * sin_theta > 1.0f) {
        vec3 reflected = reflect(r_in.direction(), hitNormal);
        scattered = ray(rec.p, reflected);
        return true;
    }

    float reflect_prob = schlick(cos_theta, etai_over_etat);
    if (rnd(state) < reflect_prob) {
        vec3 reflected = reflect(r_in.direction(), hitNormal);
        scattered = ray(rec.p, reflected);
        return true;
    }

    vec3 refracted = refract(r_in.direction(), hitNormal, etai_over_etat);
    scattered = ray(rec.p, refracted);
    return true;
}

__device__ bool scatter_coat(const vec3& albedo, float ref_idx, const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, rand_state& state, bool& shadow) {
    attenuation = vec3(1.0, 1.0, 1.0);
    shadow = false;
    bool frontFace = dot(r_in.direction(), rec.normal) < 0.0f;
    vec3 hitNormal = frontFace ? rec.normal : -rec.normal;
    float etai_over_etat = frontFace ? (1.0f / ref_idx) : ref_idx;

    float cos_theta = fminf(dot(-r_in.direction(), hitNormal), 1.0f);
    float sin_theta = sqrt(1.0f - cos_theta * cos_theta);
    if (etai_over_etat * sin_theta > 1.0f) {
        vec3 reflected = reflect(r_in.direction(), hitNormal);
        scattered = ray(rec.p, reflected);
        return true;
    }

    float reflect_prob = schlick(cos_theta, etai_over_etat);
    if (rnd(state) < reflect_prob) {
        vec3 reflected = reflect(r_in.direction(), hitNormal);
        scattered = ray(rec.p, reflected);
        return true;
    }

    return scatter_lambertian(albedo, rec, attenuation, scattered, state, shadow);
}

// based off https://computergraphics.stackexchange.com/questions/5214/a-recent-approach-for-subsurface-scattering
__device__ bool scatter_tinted_glass(float ref_idx, const vec3& absorptionCoefficient, const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, rand_state& state, bool& shadow) {
    attenuation = vec3(1.0, 1.0, 1.0);
    shadow = false;
    bool frontFace = dot(r_in.direction(), rec.normal) < 0.0f;
    vec3 hitNormal = frontFace ? rec.normal : -rec.normal;
    float etai_over_etat = frontFace ? (1.0f / ref_idx) : ref_idx;
    if (!frontFace) {
        // ray exiting model, compute absorption. rec.t being the distance travelled inside the model
        attenuation *= exp(-absorptionCoefficient * rec.t);
    }

    float cos_theta = fminf(dot(-r_in.direction(), hitNormal), 1.0f);
    float sin_theta = sqrt(1.0f - cos_theta * cos_theta);
    if (etai_over_etat * sin_theta > 1.0f) {
        vec3 reflected = reflect(r_in.direction(), hitNormal);
        scattered = ray(rec.p, reflected);
        return true;
    }

    float reflect_prob = schlick(cos_theta, etai_over_etat);
    if (rnd(state) < reflect_prob) {
        vec3 reflected = reflect(r_in.direction(), hitNormal);
        scattered = ray(rec.p, reflected);
        return true;
    }

    vec3 refracted = refract(r_in.direction(), hitNormal, etai_over_etat);
    scattered = ray(rec.p, refracted);
    return true;
}
__device__ bool scatter(const material& m, const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, rand_state& state, bool& shadow) {
    switch (m.type)
    {
    case LAMBERTIAN:
        return scatter_lambertian(m.albedo, rec, attenuation, scattered, state, shadow);
    case DIELECTRIC:
        return scatter_dielectric(m.ref_idx, r_in, rec, attenuation, scattered, state, shadow);
    case TINTED_GLASS:
        return scatter_tinted_glass(m.ref_idx, m.absorptionCoefficient, r_in, rec, attenuation, scattered, state, shadow);
    case METAL:
        return scatter_metal(m.albedo, m.fuzz, r_in, rec, attenuation, scattered, state, shadow);
    case COAT:
        return scatter_coat(m.albedo, m.ref_idx, r_in, rec, attenuation, scattered, state, shadow);
    case CHECKER:
        return scatter_checker(m.albedo, m.albedo2, m.frequency, rec, attenuation, scattered, state, shadow);
    default:
        return false;
    }
}

#endif
