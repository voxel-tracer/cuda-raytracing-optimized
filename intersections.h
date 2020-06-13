#ifndef SPHEREH
#define SPHEREH

#include "ray.h"
#include "helper_structs.h"

__device__ bool hit_bbox(const vec3& bmin, const vec3& bmax, const ray& r, float t_max) {
    float t_min = 0.001f;
    for (int a = 0; a < 3; a++) {
        float invD = 1.0f / r.direction()[a];
        float t0 = (bmin[a] - r.origin()[a]) * invD;
        float t1 = (bmax[a] - r.origin()[a]) * invD;
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

__device__ float planeHit(const plane& p, const ray& r, float t_min, float t_max) {
    float denom = dot(p.norm, r.direction());

    if (denom > -0.000001f) return FLT_MAX;
    vec3 po = p.point - r.origin();
    float t = dot(po, p.norm) / denom;
    if (t < t_min || t > t_max) return FLT_MAX;

    return t;
}

__device__ float triangleHit(const vec3* tri, const ray& r, float t_min, float t_max, float & hitU, float &hitV) {
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
        return FLT_MAX;    // This ray is parallel to this triangle.
    f = 1.0 / a;
    s = r.origin() - vertex0;
    u = f * dot(s, h);
    if (u < 0.0 || u > 1.0)
        return FLT_MAX;
    q = cross(s, edge1);
    v = f * dot(r.direction(), q);
    if (v < 0.0 || u + v > 1.0)
        return FLT_MAX;
    // At this stage we can compute t to find out where the intersection point is on the line.
    float t = f * dot(edge2, q);
    if (t > t_min && t < t_max) // ray intersection
    {
        hitU = u;
        hitV = v;
        return t;
    }
    else // This means that there is a line intersection but not a ray intersection.
        return FLT_MAX;
}

__device__ float sphereHit(const sphere& s, const ray& r, float t_min, float t_max) {
    vec3 oc = r.origin() - s.center;
    float a = dot(r.direction(), r.direction());
    float b = dot(oc, r.direction());
    float c = dot(oc, oc) - s.radius * s.radius;
    float discriminant = b * b - a * c;
    if (discriminant > 0) {
        float temp = (-b - sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min) {
            //rec.normal = (rec.p - s.center) / s.radius;
            return temp;
        }
        temp = (-b + sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min) {
            //rec.normal = (rec.p - s.center) / s.radius;
            return temp;
        }
    }
    return FLT_MAX;
}

#endif
