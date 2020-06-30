#pragma once

#include "helper_structs.h"
#include "material.h"

__device__ vec3 hexColor(int hexValue) {
    float r = ((hexValue >> 16) & 0xFF);
    float g = ((hexValue >> 8) & 0xFF);
    float b = ((hexValue) & 0xFF);
    return vec3(r, g, b) / 255.0;
}

__device__ void material_scatter(scatter_info& out, const intersection& i, const vec3& wo, const material& mat, const vec3& color, rand_state& rng) {
    if (mat.type == material_type::DIFFUSE)
        diffuse_bsdf(out, i, color, rng);
    else if (mat.type == material_type::METAL)
        glossy_bsdf(out, i, wo, color, mat.param, rng);
    else // GLASS
        dielectric_bsdf(out, i, wo, mat.param, color, 0.0f, vec3(0, 0, 0), rng);
}

__device__ void floor_coat_scatter(scatter_info& out, const intersection& i, const vec3& wo, rand_state& rng) {
    float ior = 1.5f;
    vec3 glossy_tint(1, 1, 1); // colorless reflections
    float glossy_fuzz = 0.0f;
    vec3 base_color = hexColor(0x511845);
    coat_bsdf(out, i, wo, ior, glossy_tint, glossy_fuzz, base_color, rng);
}

__device__ void floor_diffuse_scatter(scatter_info& out, const intersection& i, const vec3& wo, rand_state& rng) {
    vec3 base_color = hexColor(0x511845);
    diffuse_bsdf(out, i, base_color, rng);
}

__device__ void floor_checker_scatter(scatter_info& out, const intersection& i, const vec3& wo, rand_state& rng) {
    const vec3 color1 = hexColor(0x511845);
    const vec3 color2 = hexColor(0xff5733);
    const float frequency = 0.2f;

    if (checker_layer(i, frequency))
        diffuse_bsdf(out, i, color1, rng);
    else
        diffuse_bsdf(out, i, color2, rng);
}

__device__ void model_coat_scatter(scatter_info& out, const intersection& i, const vec3& wo, rand_state& rng) {
    float ior = 1.1f;
    vec3 glossy_tint(1, 1, 1); // colorless reflections
    float glossy_fuzz = 0.0f;
    vec3 base_color(0.0972942f, 0.0482054f, 0.000273194f);
    coat_bsdf(out, i, wo, ior, glossy_tint, glossy_fuzz, base_color, rng);
}

__device__ void model_diffuse_scatter(scatter_info& out, const intersection& i, const vec3& wo, rand_state& rng) {
    vec3 base_color(0.0972942f, 0.0482054f, 0.000273194f);
    diffuse_bsdf(out, i, base_color, rng);
}

__device__ void model_glossy_scatter(scatter_info& out, const intersection& i, const vec3& wo, rand_state& rng) {
    vec3 glossy_tint(1, 1, 1); // colorless reflections
    float glossy_fuzz = 0.0f;
    glossy_bsdf(out, i, wo, glossy_tint, glossy_fuzz, rng);
}

__device__ void model_glass_scatter(scatter_info& out, const intersection& i, const vec3& wo, rand_state& rng) {
    float ior = 1.1f;
    const vec3 glossy_tint(1, 1, 1); // colorless reflections
    float glossy_fuzz = 0.0f;
    const vec3 absorption(0, 0, 0);
    dielectric_bsdf(out, i, wo, ior, glossy_tint, glossy_fuzz, absorption, rng);
}

__device__ void model_tintedglass_scatter(scatter_info& out, const intersection& i, const vec3& wo, rand_state& rng) {
    float ior = 1.1f;
    const vec3 glossy_tint(1, 1, 1); // colorless reflections
    float glossy_fuzz = 0.0f;
    const vec3 absorptionColor(0.0972942f, 0.0482054f, 0.000273194f);
    float absorptionDistance = 10;
    const vec3 absorption = -log(absorptionColor) / absorptionDistance;
    dielectric_bsdf(out, i, wo, ior, glossy_tint, glossy_fuzz, absorption, rng);
}

__device__ void model_sss_scatter(scatter_info& out, const intersection& i, const vec3& wo, rand_state& rng) {
    float ior = 1.333f;
    const vec3 glossy_tint(1, 1, 1); // colorless reflections
    float glossy_fuzz = 0.0f;
    //const vec3 absorptionColor(0.0972942f, 0.0482054f, 0.000273194f);
    //float absorptionDistance = 1.0f;
    //const vec3 absorption = -log(absorptionColor) / absorptionDistance;
    const vec3 absorption(0.9f, 0.3f, 0.02f);
    float scatteringDistance = 2.0f;
    subsurface_dielectric_bsdf(out, i, wo, ior, glossy_tint, glossy_fuzz, absorption, scatteringDistance, rng);
}
