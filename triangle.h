#ifndef SPHEREH
#define SPHEREH

#include "ray.h"
#include "helper_structs.h"

struct hit_record {
    float t;
    vec3 p;
    vec3 normal;
    int hitIdx;
};

__device__ bool hit_bbox(const bbox& bb, const ray& r, float t_max) {
    float t_min = 0.001f;
    for (int a = 0; a < 3; a++) {
        float invD = 1.0f / r.direction()[a];
        float t0 = (bb.min[a] - r.origin()[a]) * invD;
        float t1 = (bb.max[a] - r.origin()[a]) * invD;
        if (invD < 0.0f) {
            float tmp = t0; t0 = t1; t1 = tmp;
        }
        t_min = t0 > t_min ? t0 : t_min;
        t_max = t1 < t_max ? t1 : t_max;
        if (t_max <= t_min)
            return false;
    }

    return true;
}

__device__ bool planeHit(const plane& p, const ray& r, float t_min, float t_max, hit_record& rec) {
    float denom = dot(p.norm, r.direction());

    if (denom > -0.000001f) return false;
    vec3 po = p.point - r.origin();
    float t = dot(po, p.norm) / denom;
    if (t < t_min || t > t_max) return false;

    rec.normal = p.norm;
    rec.t = t;
    rec.p = r.point_at_parameter(t);
    return true;
}

__device__ bool triangleHit(const vec3* tri, const ray& r, float t_min, float t_max, hit_record& rec) {
    const float EPSILON = 0.0000001;
    vec3 vertex0 = tri[0];
    vec3 vertex1 = tri[1];
    vec3 vertex2 = tri[2];
    vec3 edge1, edge2, h, s, q;
    float a, f, u, v;
    edge1 = vertex1 - vertex0;
    edge2 = vertex2 - vertex0;
    h = cross(r.direction(), edge2);
    a = dot(edge1, h);
    if (a > -EPSILON && a < EPSILON)
        return false;    // This ray is parallel to this triangle.
    f = 1.0 / a;
    s = r.origin() - vertex0;
    u = f * dot(s, h);
    if (u < 0.0 || u > 1.0)
        return false;
    q = cross(s, edge1);
    v = f * dot(r.direction(), q);
    if (v < 0.0 || u + v > 1.0)
        return false;
    // At this stage we can compute t to find out where the intersection point is on the line.
    float t = f * dot(edge2, q);
    if (t > t_min && t < t_max) // ray intersection
    {
        rec.t = t;
        rec.p = r.point_at_parameter(t);
        rec.normal = unit_vector(cross(edge1, edge2));
        return true;
    }
    else // This means that there is a line intersection but not a ray intersection.
        return false;
}

__device__ bool sphereHit(const sphere& s, const ray& r, float t_min, float t_max, hit_record& rec) {
    vec3 oc = r.origin() - s.center;
    float a = dot(r.direction(), r.direction());
    float b = dot(oc, r.direction());
    float c = dot(oc, oc) - s.radius * s.radius;
    float discriminant = b * b - a * c;
    if (discriminant > 0) {
        float temp = (-b - sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - s.center) / s.radius;
            return true;
        }
        temp = (-b + sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - s.center) / s.radius;
            return true;
        }
    }
    return false;
}

#endif
