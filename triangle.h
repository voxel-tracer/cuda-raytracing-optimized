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

#endif
