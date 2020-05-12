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
    float sqlen = r_out_parallel.squared_length();
    vec3 r_out_perp = sqlen >= 1.0f ? vec3(0, 0, 0) : -sqrt(1.0f - sqlen) * n;
    return r_out_parallel + r_out_perp;
}

__device__ vec3 reflect(const vec3& v, const vec3& n) {
    return v - 2.0f * dot(v, n) * n;
}

__device__ bool scatter_lambertian(const vec3& albedo, path& p, bool& shadow) {
    p.origin += p.hitT * p.rayDir; // TODO should p.origin get updated to the intersection point before we call scatter ?
    p.rayDir = unit_vector(p.hitNormal + random_in_unit_sphere(p.rng));
    p.attenuation *= albedo;
    shadow = true;
    return true;
}

// simplified checker that assumes a plane with normal = (0, 0, 1)
__device__ bool scatter_checker(const vec3& albedo1, const vec3& albedo2, float frequency, path& p, bool& shadow) {
    const vec3 hitP = p.origin + p.hitT * p.rayDir;
    auto sines = sin(frequency * hitP.x()) * sin(frequency * hitP.y()) * sin(frequency * hitP.z());
    if (sines < 0)
        return scatter_lambertian(albedo1, p, shadow);
    else
        return scatter_lambertian(albedo2, p, shadow);
}

__device__ bool scatter_metal(const vec3& albedo, float fuzz, path& p, bool& shadow) {
    vec3 reflected = reflect(p.rayDir, p.hitNormal);
    vec3 scatterDir = reflected + fuzz * random_in_unit_sphere(p.rng);
    if (dot(scatterDir, p.hitNormal) <= 0.0f) return false;
    p.origin += p.hitT * p.rayDir;
    p.rayDir = unit_vector(scatterDir);
    p.attenuation *= albedo;
    shadow = false;
    return true;
}

__device__ bool scatter_dielectric(float ref_idx, path& p, bool& shadow) {
    shadow = false;
    p.origin += p.hitT * p.rayDir;

    bool frontFace = dot(p.rayDir, p.hitNormal) < 0.0f;
    vec3 hitNormal = frontFace ? p.hitNormal : -p.hitNormal;
    float etai_over_etat = frontFace ? (1.0f / ref_idx) : ref_idx;

    float cos_theta = fminf(dot(-p.rayDir, hitNormal), 1.0f);
    float sin_theta = sqrt(1.0f - cos_theta * cos_theta);
    if (etai_over_etat * sin_theta > 1.0f) {
        p.rayDir = unit_vector(reflect(p.rayDir, hitNormal));
        return true;
    }

    float reflect_prob = schlick(cos_theta, etai_over_etat);
    if (rnd(p.rng) < reflect_prob) {
        p.rayDir = unit_vector(reflect(p.rayDir, hitNormal));
        return true;
    }

    p.rayDir = unit_vector(refract(p.rayDir, hitNormal, etai_over_etat));
    return true;
}

__device__ bool scatter_coat(const vec3& albedo, float ref_idx, path& p, bool& shadow) {
    shadow = false;
    bool frontFace = dot(p.rayDir, p.hitNormal) < 0.0f;
    vec3 hitNormal = frontFace ? p.hitNormal : -p.hitNormal;
    float etai_over_etat = frontFace ? (1.0f / ref_idx) : ref_idx;

    float cos_theta = fminf(dot(-p.rayDir, hitNormal), 1.0f);
    float sin_theta = sqrt(1.0f - cos_theta * cos_theta);
    if (etai_over_etat * sin_theta > 1.0f) {
        p.origin += p.hitT * p.rayDir;
        p.rayDir = unit_vector(reflect(p.rayDir, hitNormal));
        return true;
    }

    float reflect_prob = schlick(cos_theta, etai_over_etat);
    if (rnd(p.rng) < reflect_prob) {
        p.origin += p.hitT * p.rayDir;
        p.rayDir = unit_vector(reflect(p.rayDir, hitNormal));
        return true;
    }

    return scatter_lambertian(albedo, p, shadow);
}

// based off https://computergraphics.stackexchange.com/questions/5214/a-recent-approach-for-subsurface-scattering
__device__ bool scatter_tinted_glass(float ref_idx, const vec3& absorptionCoefficient, path& p, bool& shadow) {
    p.origin += p.hitT * p.rayDir;
    shadow = false;

    bool frontFace = dot(p.rayDir, p.hitNormal) < 0.0f;
    vec3 hitNormal = frontFace ? p.hitNormal : -p.hitNormal;
    float etai_over_etat = frontFace ? (1.0f / ref_idx) : ref_idx;

    if (!frontFace) {
        // ray exiting model, compute absorption. rec.t being the distance travelled inside the model
        p.attenuation *= exp(-absorptionCoefficient * p.hitT);
    }

    float cos_theta = fminf(dot(-p.rayDir, hitNormal), 1.0f);
    float sin_theta = sqrt(1.0f - cos_theta * cos_theta);
    if (etai_over_etat * sin_theta > 1.0f) {
        p.rayDir = unit_vector(reflect(p.rayDir, hitNormal));
        return true;
    }

    float reflect_prob = schlick(cos_theta, etai_over_etat);
    if (rnd(p.rng) < reflect_prob) {
        p.rayDir = unit_vector(reflect(p.rayDir, hitNormal));
        return true;
    }

    p.rayDir = unit_vector(refract(p.rayDir, hitNormal, etai_over_etat));
    return true;
}

__device__ bool scatter(const material& m, path& p, bool& shadow) {
    switch (m.type)
    {
    case LAMBERTIAN:
        return scatter_lambertian(m.albedo, p, shadow);
    case DIELECTRIC:
        return scatter_dielectric(m.ref_idx, p, shadow);
    case TINTED_GLASS:
        return scatter_tinted_glass(m.ref_idx, m.absorptionCoefficient, p, shadow);
    case METAL:
        return scatter_metal(m.albedo, m.fuzz, p, shadow);
    case COAT:
        return scatter_coat(m.albedo, m.ref_idx, p, shadow);
    case CHECKER:
        return scatter_checker(m.albedo, m.albedo2, m.frequency, p, shadow);
    default:
        return false;
    }
}

#endif
