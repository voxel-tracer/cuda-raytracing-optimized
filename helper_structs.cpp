#include "helper_structs.h"

material new_lambertian(vec3 albedo) {
    return material(LAMBERTIAN, albedo, vec3(), 0.0);
}

material new_dielectric(float ref_idx) {
    return material(DIELECTRIC, vec3(), vec3(), ref_idx);
}

material new_metal(vec3 albedo, float fuzz) {
    return material(METAL, albedo, vec3(), fuzz);
}

material new_coat(vec3 albedo, float ref_idx) {
    return material(COAT, albedo, vec3(), ref_idx);
}

material new_checker(vec3 albedo, vec3 albedo2, float frequency) {
    return material(CHECKER, albedo, albedo2, frequency);
}
