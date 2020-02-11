#ifndef SPHEREH
#define SPHEREH

#include "ray.h"
#include "helper_structs.h"

struct hit_record {
    float t;
    vec3 p;
    int hitIdx;
};

__device__ bool sphereHit(const sphere& s, const ray& r, float t_min, float t_max, hit_record& rec) {
    vec3 oc = r.origin() - s.center;
    float a = dot(r.direction(), r.direction());
    float b = dot(oc, r.direction());
    float c = dot(oc, oc) - s.radius * s.radius;
    float discriminant = b*b - a*c;
    if (discriminant > 0) {
        float temp = (-b - sqrt(discriminant))/a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            return true;
        }
        temp = (-b + sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            return true;
        }
    }
    return false;
}


#endif
