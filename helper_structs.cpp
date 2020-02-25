#include "helper_structs.h"

material new_lambertian(vec3 albedo) {
    return material(LAMBERTIAN, albedo, 0.0);
}

material new_dielectric(float ref_idx) {
    return material(DIELECTRIC, vec3(), ref_idx);
}

material new_metal(vec3 albedo, float fuzz) {
    return material(METAL, albedo, fuzz);
}

material new_coat(vec3 albedo, float ref_idx) {
    return material(COAT, albedo, ref_idx);
}
