#ifndef MATERIALH
#define MATERIALH

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

__device__ void diffuse_bsdf(scatter_info& out, const intersection& i, const vec3& albedo, rand_state& rng) {
    out.wi = unit_vector(i.normal + random_in_unit_sphere(rng));
    out.throughput = albedo;
    out.specular = false;
}

__device__ bool checker_layer(const intersection& i, float frequency) {
    auto sines = sin(frequency * i.p.x()) * sin(frequency * i.p.y()) * sin(frequency * i.p.z());
    return sines < 0;
}

// simplified checker that assumes a plane with normal = (0, 0, 1)
__device__ void scatter_checker(scatter_info & out, const intersection& i, float frequency, const vec3& albedo1, const vec3& albedo2, rand_state& rng) {
    if (checker_layer(i, frequency))
        diffuse_bsdf(out, i, albedo1, rng);
    else
        diffuse_bsdf(out, i, albedo2, rng);
}

__device__ void glossy_bsdf(scatter_info &out, const intersection& i, const vec3& wo, const vec3& tint, float fuzz, rand_state& rng) {
    vec3 reflected = reflect(wo, i.normal);
    if (fuzz > 0.0001f)
        reflected += fuzz * random_in_unit_sphere(rng);
    out.wi = unit_vector(reflected);
    out.throughput *= tint;
    out.specular = true;
}

__device__ bool fresnel_layer(const intersection& i, const vec3& wo, float ior, rand_state& rng) {
    float etai_over_etat = i.inside ? ior : (1.0f / ior);
    float cos_theta = fminf(dot(-wo, i.normal), 1.0f);
    float sin_theta = sqrt(1.0f - cos_theta * cos_theta);
    return (etai_over_etat * sin_theta > 1.0f || rnd(rng) < schlick(cos_theta, etai_over_etat));
}

__device__ void coat_bsdf(scatter_info& out, const intersection& i, const vec3& wo, float layer_ior, const vec3& glossy_tint, float glossy_fuzz, const vec3& diffuse_albedo, rand_state& rng) {
    if (fresnel_layer(i, wo, layer_ior, rng)) {
        // ray will be reflected by the glossy bsdf
        glossy_bsdf(out, i, wo, glossy_tint, glossy_fuzz, rng);
    } else {
        // ray will be reflected by the diffuse bsdf
        diffuse_bsdf(out, i, diffuse_albedo, rng);
    }
}

// based off https://computergraphics.stackexchange.com/questions/5214/a-recent-approach-for-subsurface-scattering
__device__ void dielectric_bsdf(scatter_info& out, const intersection& i, const vec3& wo, float layer_ior, const vec3& glossy_tint, float glossy_fuzz, const vec3& absorptionCoefficient, rand_state& rng) {

    if (i.inside) {
        // ray exiting model, compute absorption. rec.t being the distance travelled inside the model
        out.throughput = exp(-absorptionCoefficient * i.t);
    }

    if (fresnel_layer(i, wo, layer_ior, rng)) {
        // ray will be reflected by the glossy bsdf
        glossy_bsdf(out, i, wo, glossy_tint, glossy_fuzz, rng);
    }
    else {
        // ray will be refracted
        float etai_over_etat = i.inside ? layer_ior : (1.0f / layer_ior);
        out.wi = unit_vector(refract(wo, i.normal, etai_over_etat));
        out.refracted = true;
    }

    out.specular = true;
}

__device__ void subsurface_bsdf(scatter_info& out, const intersection& i, const vec3& wo, const vec3& absorptionCoefficient, float scatteringDistance, rand_state& rng) {
    bool scattered = false;
    if (i.inside) {
        float d = -logf(rnd(rng)) / scatteringDistance;
        if (d < i.t) {
            scattered = true;
            out.t = d;
        }
        out.throughput = exp(-absorptionCoefficient * out.t);
    }

    if (scattered) {
        out.wi = random_in_unit_sphere(rng);
    } else {
        out.wi = wo; // ray doesn't change direction
        out.refracted = true;
    }

    out.specular = true;
}

__device__ void subsurface_dielectric_bsdf(scatter_info& out, const intersection& i, const vec3& wo, float layer_ior, const vec3 &glossy_tint, float glossy_fuzz, const vec3& absorptionCoefficient, float scatteringDistance, rand_state& rng) {
    bool scattered = false;
    if (i.inside) {
        float d = -logf(rnd(rng)) / scatteringDistance;
        if (d < i.t) {
            scattered = true;
            out.t = d;
        }
        out.throughput = exp(-absorptionCoefficient * out.t);
    }

    if (scattered) {
        out.wi = random_in_unit_sphere(rng);
    }
    else {
        if (fresnel_layer(i, wo, layer_ior, rng)) {
            // ray will be reflected by the glossy bsdf
            glossy_bsdf(out, i, wo, glossy_tint, glossy_fuzz, rng);
        }
        else {
            // ray will be refracted
            float etai_over_etat = i.inside ? layer_ior : (1.0f / layer_ior);
            out.wi = unit_vector(refract(wo, i.normal, etai_over_etat));
            out.refracted = true;
        }
    }

    out.specular = true;
}

#endif
