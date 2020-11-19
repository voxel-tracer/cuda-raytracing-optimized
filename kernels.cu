#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "rnd.h"
#include "vec3.h"
#include "camera.h"
#include "intersections.h"
#include "material.h"
#include "scene_materials.h"

#include "kernels.h"

#define STATS
#define RUSSIAN_ROULETTE
#define SHADOW
#define TEXTURES

#define EPSILON 0.01f

//#define DUAL_NODES
//#define BVH_COUNT

//#define USE_BVH_TEXTURE


// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << cudaGetErrorString(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

enum OBJ_ID {
    NONE,
    TRIMESH,
    PLANE,
    LIGHT
};

#ifdef STATS
#define LARGE_LEAF      199
#define LARGE_INTERNAL  1500

#define NUM_RAYS_PRIMARY                0
#define NUM_RAYS_PRIMARY_HIT_MESH       1
#define NUM_RAYS_PRIMARY_NOHITS         2
#define NUM_RAYS_PRIMARY_BBOX_NOHITS    3
#define NUM_RAYS_SECONDARY              4
#define NUM_RAYS_SECONDARY_MESH         5
#define NUM_RAYS_SECONDARY_NOHIT        6
#define NUM_RAYS_SECONDARY_MESH_NOHIT   7
#define NUM_RAYS_SECONDARY_BBOX_NOHIT   8
#define NUM_RAYS_SHADOWS                9
#define NUM_RAYS_SHADOWS_BBOX_NOHITS    10
#define NUM_RAYS_SHADOWS_NOHITS         11
#define NUM_RAYS_LOW_POWER              12
#define NUM_RAYS_EXCEED_MAX_BOUNCE      13
#define NUM_RAYS_RUSSIAN_KILL           14
#define NUM_RAYS_NAN                    15
#define NUM_NODES_BOTH                  16
#define NUM_NODES_SINGLE                17
#define METRIC_NUM_INTERNAL             18
#define METRIC_NUM_LEAVES               19
#define METRIC_NUM_LEAF_HITS            20
#define METRIC_MAX_NUM_INTERNAL         21
#define METRIC_MAX_NUM_LEAVES           22
#define METRIC_MAX_LEAF_HITS            23
#define METRIC_NUM_HIGH_LEAVES          24
#define METRIC_NUM_HIGH_NODES           25
#define NUM_RAYS_SIZE                   26
#endif

struct RenderContext {
    vec3* fb;

    LinearTriangle* tris;

#ifdef USE_BVH_TEXTURE
    float* d_bvh;
    cudaTextureObject_t bvh_tex;
#else
    LinearBVHNode* nodes;
#endif // USE_BVH_TEXTURE

    plane floor;

    int nx;
    int ny;
    int ns;
    int maxDepth;
    camera cam;

    sphere light = sphere(vec3(52.514355, 715.686951, -272.620972), 50);
    vec3 lightColor = vec3(1, 1, 1) * 20;

    material* materials;
#ifdef TEXTURES
    float** tex_data;
    int* tex_width;
    int* tex_height;
#endif

#ifdef STATS
    uint64_t* stats;
    __device__ void incStat(int type) const {
        atomicAdd(stats + type, 1);
    }
    __device__ void incStat(int type, int value) const {
        atomicAdd(stats + type, value);
    }
    __device__ void maxStat(int type, uint64_t value) const {
        atomicMax(stats + type, value);
    }

    void initStats() {
        checkCudaErrors(cudaMallocManaged((void**)&stats, NUM_RAYS_SIZE * sizeof(uint64_t)));
        memset(stats, 0, NUM_RAYS_SIZE * sizeof(uint64_t));
    }
    void printStats() const {
        // total is used to compute average num internal nodes per path
        uint64_t numPrimary = stats[NUM_RAYS_PRIMARY];
        uint64_t numSecondary = stats[NUM_RAYS_SECONDARY];
        uint64_t numShadows = stats[NUM_RAYS_SHADOWS];
        uint64_t total = numPrimary + numSecondary + numShadows;
        // compute total excluding nohits to properly compute average leaf nodes per path
        uint64_t numPrimaryNoHits = stats[NUM_RAYS_PRIMARY_NOHITS];
        uint64_t numSecondaryNoHits = stats[NUM_RAYS_SECONDARY_NOHIT];
        uint64_t numShadowsNoHits = stats[NUM_RAYS_SHADOWS_NOHITS];
        uint64_t totalHits = total - numPrimaryNoHits - numSecondaryNoHits - numShadowsNoHits;

        std::cerr << "num rays:\n";
        std::cerr << " primary                     : " << std::fixed << numPrimary << std::endl;
        std::cerr << " primary hit mesh            : " << std::fixed << stats[NUM_RAYS_PRIMARY_HIT_MESH] << std::endl;
        std::cerr << " primary nohit               : " << std::fixed << numPrimaryNoHits << std::endl;
        std::cerr << " primary bb nohit            : " << std::fixed << stats[NUM_RAYS_PRIMARY_BBOX_NOHITS] << std::endl;
        std::cerr << " secondary                   : " << std::fixed << numSecondary << std::endl;
        std::cerr << " secondary no hit            : " << std::fixed << numSecondaryNoHits << std::endl;
        std::cerr << " secondary bb nohit          : " << std::fixed << stats[NUM_RAYS_SECONDARY_BBOX_NOHIT] << std::endl;
        std::cerr << " secondary mesh              : " << std::fixed << stats[NUM_RAYS_SECONDARY_MESH] << std::endl;
        std::cerr << " secondary mesh nohit        : " << std::fixed << stats[NUM_RAYS_SECONDARY_MESH_NOHIT] << std::endl;
        std::cerr << " shadows                     : " << std::fixed << numShadows << std::endl;
        std::cerr << " shadows nohit               : " << std::fixed << numShadowsNoHits << std::endl;
        std::cerr << " shadows bb nohit            : " << std::fixed << stats[NUM_RAYS_SHADOWS_BBOX_NOHITS] << std::endl;
        std::cerr << " power < 0.01                : " << std::fixed << stats[NUM_RAYS_LOW_POWER] << std::endl;
        std::cerr << " exceeded max bounce         : " << std::fixed << stats[NUM_RAYS_EXCEED_MAX_BOUNCE] << std::endl;
        std::cerr << " russian roulette            : " << std::fixed << stats[NUM_RAYS_RUSSIAN_KILL] << std::endl;
        std::cerr << " both nodes hit              : " << std::fixed << stats[NUM_NODES_BOTH] << std::endl;
        std::cerr << " single node hit             : " << std::fixed << stats[NUM_NODES_SINGLE] << std::endl;

        uint64_t numInternal = stats[METRIC_NUM_INTERNAL];
        uint64_t numLeaves = stats[METRIC_NUM_LEAVES];
        uint64_t numLeafHits = stats[METRIC_NUM_LEAF_HITS];
        uint64_t avgPathSize = numInternal / total;
        uint64_t avgLeafInters = numLeaves / total;
        float avgLeafHits = float(numLeafHits) / totalHits;
        std::cerr << " num internal nodes          : " << std::fixed << numInternal << std::endl;
        std::cerr << " num leaf nodes              : " << std::fixed << numLeaves << std::endl;
        std::cerr << " num leaf hits               : " << std::fixed << numLeafHits << std::endl;
        std::cerr << " avg path size               : " << std::fixed << avgPathSize << std::endl;
        std::cerr << " avg leaf intersections      : " << std::fixed << avgLeafInters << std::endl;
        std::cerr << " avg leaf hits               : " << std::fixed << avgLeafHits << std::endl;
        std::cerr << " max num internal            : " << std::fixed << stats[METRIC_MAX_NUM_INTERNAL] << std::endl;
        std::cerr << " max num leaves              : " << std::fixed << stats[METRIC_MAX_NUM_LEAVES] << std::endl;
        std::cerr << " max leaf hits               : " << std::fixed << stats[METRIC_MAX_LEAF_HITS] << std::endl;
        std::cerr << " num paths with large leaves : " << std::fixed << stats[METRIC_NUM_HIGH_LEAVES] << std::endl;
        std::cerr << " num paths with large nodes  : " << std::fixed << stats[METRIC_NUM_HIGH_NODES] << std::endl;
        if (stats[NUM_RAYS_NAN] > 0)
            std::cerr << "*** " << stats[NUM_RAYS_NAN] << " NaNs detected" << std::endl;
    }
#else
    uint64_t* unique;
    void initStats() {
        checkCudaErrors(cudaMallocManaged((void**)&unique, sizeof(uint64_t)));
        unique[0] = 0;
    }
    void printStats() const {}
#endif
};

RenderContext renderContext;
#ifdef STATS
__device__ void updateBvhStats(const RenderContext& context, uint64_t numNodes, uint64_t numPrimitives, uint64_t numPrimHits) {
    context.incStat(METRIC_NUM_INTERNAL, numNodes);
    context.incStat(METRIC_NUM_LEAVES, numPrimitives);
    context.maxStat(METRIC_MAX_NUM_LEAVES, numPrimitives);
    context.maxStat(METRIC_MAX_NUM_INTERNAL, numNodes);

    context.incStat(METRIC_NUM_LEAF_HITS, numPrimHits);
    context.maxStat(METRIC_MAX_LEAF_HITS, numPrimHits);

    if (numPrimitives > LARGE_LEAF) context.incStat(METRIC_NUM_HIGH_LEAVES);
    if (numNodes > LARGE_INTERNAL) context.incStat(METRIC_NUM_HIGH_NODES);
}
#endif // STATS

#ifdef DUAL_NODES

__device__ void pop_bitstack(unsigned int& bitStack, int& idx) {
    int m = __ffsll(bitStack) - 1;
    bitStack = (bitStack >> m) ^ 1;
    idx = (idx >> m) ^ 1;
}

__device__ float hitBvh(const ray& r, const RenderContext& context, float t_min, float t_max, tri_hit& rec, bool isShadow) {
    int idx = 1;
    float closest = t_max;
    unsigned int bitStack = 1;
#ifdef BVH_COUNT
    uint64_t traversed_single = 0;
    uint64_t traversed_both = 0;
#endif
#ifdef STATS
    uint64_t numLeaves = 0;
    uint64_t numInternal = 0;
    uint64_t numLeafHits = 0;
#endif // STATS

    while (idx) {
        if (idx < context.firstLeafIdx) { // TODO we need to keep a flag if previously loaded children were leaf nodes (nPrimitives == 0)
#ifdef STATS
            numInternal += 2;
#endif // STATS
            // load both children nodes
            int idx2 = idx << 1;
#ifdef USE_BVH_TEXTURE
            int float4_idx = idx * 3;
            float4 bvh_a = tex1Dfetch<float4>(context.bvh_tex, float4_idx);
            float4 bvh_b = tex1Dfetch<float4>(context.bvh_tex, float4_idx + 1);
            float4 bvh_c = tex1Dfetch<float4>(context.bvh_tex, float4_idx + 2);

            bvh_node left(bvh_a.x, bvh_a.y, bvh_a.z, bvh_a.w, bvh_b.x, bvh_b.y);
            bvh_node right(bvh_b.z, bvh_b.w, bvh_c.x, bvh_c.y, bvh_c.z, bvh_c.w);
#else
            LinearBVHNode left = context.bvh[idx2];
            LinearBVHNode right = context.bvh[idx2 + 1];
#endif // USE_BVH_TEXTURE
            float leftHit = hit_bbox_dist(left.bounds.pMin, left.bounds.pMax, r, closest);
            bool traverseLeft = leftHit < closest;
            float rightHit = hit_bbox_dist(right.bounds.pMin, right.bounds.pMax, r, closest);
            bool traverseRight = rightHit < closest;
            bool swap = rightHit < leftHit;
            if (traverseLeft && traverseRight) {
#ifdef BVH_COUNT
                traversed_both++;
#endif
                idx = idx2 + (swap ? 1 : 0);
                bitStack = (bitStack << 1) + 1;
            } else if (traverseLeft || traverseRight) {
#ifdef BVH_COUNT
                traversed_single++;
#endif
                idx = idx2 + (swap ? 1 : 0);
                bitStack = bitStack << 1;
            } else {
                pop_bitstack(bitStack, idx);
            }
        } else { // leaf node
#ifdef STATS
            numLeaves+= context.numPrimitivesPerLeaf;
#endif
            int first = (idx - context.firstLeafIdx) * context.numPrimitivesPerLeaf;
            for (auto i = 0; i < context.numPrimitivesPerLeaf; i++) {
                const LinearTriangle tri = context.tris[first + i];
                if (isinf(tri.v[0].x()))
                    break; // we reached the end of the primitives buffer
                float u, v;
                float hitT = triangleHit(tri, r, t_min, closest, u, v);
                if (hitT < closest) {
#ifdef STATS
                    numLeafHits++;
#endif // STATS
                    if (isShadow) {
#ifdef STATS
                        updateBvhStats(context, numInternal, numLeaves, numLeafHits);
#endif // STATS
                        return 0.0f;
                    }
                    closest = hitT;
                    rec.triId = first + i;
                    rec.u = u;
                    rec.v = v;
                }
            }
            pop_bitstack(bitStack, idx); // TODO reset leaf flag if this call didn't move to sibling node
        }
    }
#ifdef STATS
    updateBvhStats(context, numInternal, numLeaves, numLeafHits);
#endif // STATS
#ifdef BVH_COUNT
    context.addStat(NUM_NODES_BOTH, traversed_both);
    context.addStat(NUM_NODES_SINGLE, traversed_single);
#endif
    return closest;
}
#else

__device__ float hitBvh(const ray& r, const RenderContext& context, float t_min, float t_max, tri_hit& rec, bool isShadow) {
    vec3 invDir(1.0f / r.direction().x(), 1.0f / r.direction().y(), 1.0f / r.direction().z());
    int dirIsNeg[3] = { invDir[0] < 0, invDir[1] < 0, invDir[2] < 0 };

    // Follow ray through BVH nodes to find primitive intersections
    int toVisitOffset = 0, currentNodeIndex = 0;
    int nodesToVisit[64];
    
    float closest = t_max;

#ifdef STATS
    uint64_t numPrimitives = 0;
    uint64_t numNodes = 0;
    uint64_t numPrimHits = 0;
#endif // STATS

    while (true) {
        const LinearBVHNode* node = &context.nodes[currentNodeIndex];
#ifdef STATS
        numNodes++;
#endif // STATS

        // Check ray against BVH node
        if (hit_bbox_dist(node->bounds.pMin, node->bounds.pMax, r, closest) < closest) {
            if (node->nPrimitives > 0) {
                // Intersect ray with primitives in leaf BVH node
                for (int i = 0; i < node->nPrimitives; i++) {
                    float u, v;
                    float hitT = triangleHit(context.tris[node->primitivesOffset + i], r, t_min, closest, u, v);
#ifdef STATS
                    numPrimitives++;
#endif // STATS
                    if (hitT < closest) {
#ifdef STATS
                        numPrimHits++;
#endif // STATS
                        if (isShadow) {
#ifdef STATS
                            updateBvhStats(context, numNodes, numPrimitives, numPrimHits);
#endif // STATS
                            return 0.0f;
                        }
                        closest = hitT;
                        rec.triId = node->primitivesOffset + i;
                        rec.u = u;
                        rec.v = v;
                    }
                }
                if (toVisitOffset == 0) break;
                currentNodeIndex = nodesToVisit[--toVisitOffset];
            } else {
                // Put the far BVH node on nodesToVisit stack, advance to near node
                if (dirIsNeg[node->axis]) {
                    nodesToVisit[toVisitOffset++] = currentNodeIndex + 1;
                    currentNodeIndex = node->secondChildOffset;
                } else {
                    nodesToVisit[toVisitOffset++] = node->secondChildOffset;
                    currentNodeIndex = currentNodeIndex + 1;
                }
            }
        } else {
            if (toVisitOffset == 0) break;
            currentNodeIndex = nodesToVisit[--toVisitOffset];
        }
    }
#ifdef STATS
    updateBvhStats(context, numNodes, numPrimitives, numPrimHits);
#endif // STATS
    return closest;
}
#endif

__device__ float hitMesh(const ray& r, const RenderContext& context, float t_min, float t_max, tri_hit& rec, bool primary, bool isShadow) {
    // TODO not really needed as it is equivalent to intersecting first node of BVH tree
    if (!hit_bbox(context.nodes[0].bounds.pMin, context.nodes[0].bounds.pMax, r, t_max)) {
#ifdef STATS
        if (isShadow) context.incStat(NUM_RAYS_SHADOWS_BBOX_NOHITS);
        else context.incStat(primary ? NUM_RAYS_PRIMARY_BBOX_NOHITS : NUM_RAYS_SECONDARY_BBOX_NOHIT);
#else
        // adding this one line saves 5s of rendering time, no idea why!!!
        atomicAdd(context.unique, 1);
#endif
        return FLT_MAX;
    }

    return hitBvh(r, context, t_min, t_max, rec, isShadow);
}

__device__ bool hit(const RenderContext& context, const path& p, float t_max, bool isShadow, intersection &inters) {
    const ray r = isShadow ? ray(p.origin, p.shadowDir) : ray(p.origin, p.rayDir);
    tri_hit triHit;
    bool primary = p.bounce == 0;
    inters.objId = NONE;
    if ((inters.t = hitMesh(r, context, EPSILON, t_max, triHit, primary, isShadow)) < t_max) {
        if (isShadow) return true; // we don't need to compute the intersection details for shadow rays

        inters.objId = TRIMESH;
        LinearTriangle tri = context.tris[triHit.triId];
        inters.meshID = tri.meshID;
        inters.normal = unit_vector(cross(tri.v[1] - tri.v[0], tri.v[2] - tri.v[0]));
        inters.texCoords[0] = (triHit.u * tri.texCoords[1 * 2 + 0] + triHit.v * tri.texCoords[2 * 2 + 0] + (1 - triHit.u - triHit.v) * tri.texCoords[0 * 2 + 0]);
        inters.texCoords[1] = (triHit.u * tri.texCoords[1 * 2 + 1] + triHit.v * tri.texCoords[2 * 2 + 1] + (1 - triHit.u - triHit.v) * tri.texCoords[0 * 2 + 1]);
    } else {
        if (isShadow) return false; // shadow rays only care about the main triangle mesh
        if (p.specular && sphereHit(context.light, r, EPSILON, t_max) < t_max) { // specular rays should intersect with the light
            inters.objId = LIGHT;
            return true; // we don't need to compute p and update normal to face the ray
        }
    }

    if (inters.objId != NONE) {
        inters.p = r.point_at_parameter(inters.t);
        if (dot(r.direction(), inters.normal) > 0.0f)
            inters.normal = -inters.normal; // ensure normal is always facing the ray
        return true;
    }

    return false;
}

#ifdef SHADOW
__device__ bool generateShadowRay(const RenderContext& context, path& p, const intersection &inters, float &lightDist) {
    // create a random direction towards the light
    // coord system for sampling
    const vec3 sw = unit_vector(context.light.center - p.origin);
    const vec3 su = unit_vector(cross(fabs(sw.x()) > 0.01f ? vec3(0, 1, 0) : vec3(1, 0, 0), sw));
    const vec3 sv = cross(sw, su);

    // sample sphere by solid angle
    const float cosAMax = sqrt(1.0f - context.light.radius * context.light.radius / (p.origin - context.light.center).squared_length());
    if (isnan(cosAMax)) return false; // if the light radius is too big and it reaches the model, this will be null

    const float eps1 = rnd(p.rng);
    const float eps2 = rnd(p.rng);
    const float cosA = 1.0f - eps1 + eps1 * cosAMax;
    const float sinA = sqrt(1.0f - cosA * cosA);
    const float phi = 2 * M_PI * eps2;
    const vec3 l = su * cosf(phi) * sinA + sv * sinf(phi) * sinA + sw * cosA;

    const float dotl = dot(l, inters.normal);
    if (dotl <= 0)
        return false;

    p.shadowDir = unit_vector(l);
    const float omega = 2 * M_PI * (1.0f - cosAMax);
    p.lightContribution = p.attenuation * context.lightColor * dotl * omega / M_PI;

    // compute distance to light so we only intersect triangles in front of the light
    lightDist = (context.light.center - p.origin).length() - context.light.radius;

    return true;
}
#endif

__device__ void color(const RenderContext& context, path& p) {
    p.attenuation = vec3(1.0, 1.0, 1.0);
    p.color = vec3(0, 0, 0);
#ifdef STATS
    bool fromMesh = false;
#endif
    for (p.bounce = 0; p.bounce < context.maxDepth; p.bounce++) {
#ifdef STATS
        bool primary = p.bounce == 0;
        context.incStat(primary ? NUM_RAYS_PRIMARY : NUM_RAYS_SECONDARY);
        if (fromMesh) context.incStat(NUM_RAYS_SECONDARY_MESH);
        if (p.attenuation.length() < 0.01f) context.incStat(NUM_RAYS_LOW_POWER);
#endif
        intersection inters;
        if (!hit(context, p, FLT_MAX, false, inters)) {
#ifdef PATH_DBG
            if (p.dbg) printf("bounce %d: NO_HIT\n", p.bounce);
#endif
#ifdef STATS
            if (primary) context.incStat(NUM_RAYS_PRIMARY_NOHITS);
            else context.incStat(fromMesh ? NUM_RAYS_SECONDARY_MESH_NOHIT : NUM_RAYS_SECONDARY_NOHIT);
#endif
            // sky color
            //float t = 0.5f * (p.rayDir.y() + 1.0f);
            //vec3 c = (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
            //p.color += p.attenuation * c;

            // constant sky color of (0.5, 0.5, 0.5)
            p.color += p.attenuation * vec3(0.5f, 0.5f, 0.5f);
            return;
        }

#ifdef STATS
        fromMesh = (inters.objId == TRIMESH);
        if (primary && !fromMesh) context.incStat(NUM_RAYS_PRIMARY_NOHITS); // primary didn't intersect mesh, only floor
        if (primary && fromMesh) context.incStat(NUM_RAYS_PRIMARY_HIT_MESH);
#endif
        if (inters.objId == LIGHT) {
            // only specular rays can intersect the light

#ifdef PATH_DBG
            if (p.dbg) printf("bounce %d: HIT LIGHT\n", p.bounce);
#endif
            // ray hit the light, compute its contribution and add it to the path's color
#ifdef SHADOW
            // we should uncomment this line, but we need to compute light contribution properly
            // p.color += p.attenuation * context.lightColor;
#else
            p.color += p.attenuation * context.lightColor;
#endif
            return;
        }
#ifdef PATH_DBG
        if (p.dbg) printf("bounce %d: HIT %d at t %f with normal (%f, %f, %f)\n", p.bounce, inters.objId, inters.t, inters.normal.x(), inters.normal.y(), inters.normal.z());
#endif

        inters.inside = p.inside;
        scatter_info scatter(inters);
        if (inters.objId == TRIMESH) {
            const material& mat = context.materials[inters.meshID];
#ifdef TEXTURES
            vec3 albedo;
            if (mat.texId != -1) {
                int texId = mat.texId;
                int width = context.tex_width[texId];
                int height = context.tex_height[texId];
                float tu = inters.texCoords[0];
                tu = tu - floorf(tu);
                float tv = inters.texCoords[1];
                tv = tv - floorf(tv);
                const int tx = (width - 1) * tu;
                const int ty = (height - 1) * tv;
                const int tIdx = ty * width + tx;
                albedo = vec3(
                    context.tex_data[texId][tIdx * 3 + 0],
                    context.tex_data[texId][tIdx * 3 + 1],
                    context.tex_data[texId][tIdx * 3 + 2]);
            }
            else {
                albedo = mat.color;
            }
#else
            vec3 albedo(0.5f, 0.5f, 0.5f);
#endif
            material_scatter(scatter, inters, p.rayDir, context.materials[inters.meshID], albedo, p.rng);
        }
        else 
            floor_diffuse_scatter(scatter, inters, p.rayDir, p.rng);

        p.origin += scatter.t * p.rayDir;
        p.rayDir = scatter.wi;
        p.attenuation *= scatter.throughput;
        p.specular = scatter.specular;
        p.inside = scatter.refracted ? !p.inside : p.inside;
#ifdef SHADOW
        // trace shadow ray for diffuse rays
        float lightDist;
        if (!p.specular && generateShadowRay(context, p, inters, lightDist)) {
#ifdef PATH_DBG
            if (p.dbg) printf("bounce %d: SHADOW\n", p.bounce);
#endif
#ifdef STATS
            context.incStat(NUM_RAYS_SHADOWS);
#endif
            if (!hit(context, p, lightDist, true, inters)) {
#ifdef PATH_DBG
                if (p.dbg) printf("bounce %d: SHADOW NO HIT\n", p.bounce);
#endif
#ifdef STATS
                context.incStat(NUM_RAYS_SHADOWS_NOHITS);
#endif
                // intersection point is illuminated by the light
                p.color += p.lightContribution;
            }
        }
#endif
#ifdef RUSSIAN_ROULETTE
        // russian roulette
        if (p.bounce > 3) {
            float m = max(p.attenuation);
            if (rnd(p.rng) > m) {
#ifdef PATH_DBG
                if (p.dbg) printf("bounce %d: RUSSIAN ROULETTE BREAK\n", p.bounce);
#endif
#ifdef STATS
                context.incStat(NUM_RAYS_RUSSIAN_KILL);
#endif
                return;
            }
            p.attenuation *= 1 / m;
        }
#endif
    }
    // exceeded recursion
#ifdef STATS
    context.incStat(NUM_RAYS_EXCEED_MAX_BOUNCE);
#endif
}

__global__ void render(const RenderContext context) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= context.nx) || (j >= context.ny)) return;

    path p;
    uint64_t pixelId = j * context.nx + i;
    p.rng = (wang_hash(pixelId) * 336343633) | 1;
#ifdef PATH_DBG
    const int dbgId = (context.ny - 308) * context.nx + 164;
    p.dbg = pixelId == dbgId;
#endif
    vec3 col(0, 0, 0); // this is specific to the pixel so it should be stored separately from the path
    for (int s = 0; s < context.ns; s++) {
        float u = float(i + rnd(p.rng)) / float(context.nx);
        float v = float(j + rnd(p.rng)) / float(context.ny);
        ray r = get_ray(context.cam, u, v, p.rng);
        p.origin = r.origin();
        p.rayDir = r.direction();
        p.specular = false;
        p.inside = false;
        color(context, p);
        // once color() is done, p.color will contain all the light received through p
        col += p.color;
#ifdef STATS
        if (isnan(p.color)) context.incStat(NUM_RAYS_NAN);
#endif
    }
    // color is specific to the pixel being traced, 
    //col /= float(context.ns);
    //col[0] = sqrt(col[0]);
    //col[1] = sqrt(col[1]);
    //col[2] = sqrt(col[2]);
    context.fb[pixelId] = col / float(context.ns);
}

extern "C" void
initRenderer(const kernel_scene sc, const camera cam, vec3 * *fb, int nx, int ny, int maxDepth) {
    renderContext.nx = nx;
    renderContext.ny = ny;
    renderContext.floor = sc.floor;
    renderContext.maxDepth = maxDepth;

    size_t fb_size = nx * ny * sizeof(vec3);
    checkCudaErrors(cudaMallocManaged((void**)&(renderContext.fb), fb_size));
    *fb = renderContext.fb;

    checkCudaErrors(cudaMalloc((void**)&renderContext.tris, sc.m->tris.size() * sizeof(LinearTriangle)));
    checkCudaErrors(cudaMemcpy(renderContext.tris, &(sc.m->tris[0]), sc.m->tris.size() * sizeof(LinearTriangle), cudaMemcpyHostToDevice));

#ifdef USE_BVH_TEXTURE
    // copy bvh data to float array
    checkCudaErrors(cudaMalloc((void**)&renderContext.d_bvh, sc.m->numBvhNodes * sizeof(bvh_node)));
    checkCudaErrors(cudaMemcpy(renderContext.d_bvh, sc.m->bvh, sc.m->numBvhNodes * sizeof(bvh_node), cudaMemcpyHostToDevice));

    cudaResourceDesc texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));

    texRes.resType = cudaResourceTypeLinear;
    texRes.res.linear.devPtr = renderContext.d_bvh;
    texRes.res.linear.sizeInBytes = sizeof(bvh_node) * sc.m->numBvhNodes;
    texRes.res.linear.desc = cudaCreateChannelDesc<float4>();

    cudaTextureDesc texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));

    texDescr.normalizedCoords = false; // we access coordinates as is
    texDescr.filterMode = cudaFilterModePoint; // return closest texel
    texDescr.readMode = cudaReadModeElementType;

    checkCudaErrors(cudaCreateTextureObject(&renderContext.bvh_tex, &texRes, &texDescr, NULL));
#else
    checkCudaErrors(cudaMalloc((void**)&renderContext.nodes, sc.m->nodes.size() * sizeof(LinearBVHNode)));
    checkCudaErrors(cudaMemcpy(renderContext.nodes, &(sc.m->nodes[0]), sc.m->nodes.size() * sizeof(LinearBVHNode), cudaMemcpyHostToDevice));
#endif // USE_BVH_TEXTURE

    checkCudaErrors(cudaMalloc((void**)&renderContext.materials, sc.numMaterials * sizeof(material)));
    checkCudaErrors(cudaMemcpy(renderContext.materials, sc.materials, sc.numMaterials * sizeof(material), cudaMemcpyHostToDevice));
#ifdef TEXTURES
    if (sc.numTextures > 0) {
        int* tex_width = new int[sc.numTextures];
        int* tex_height = new int[sc.numTextures];
        float** tex_data = new float* [sc.numTextures];

        for (auto i = 0; i < sc.numTextures; i++) {
            const stexture& tex = sc.textures[i];
            tex_width[i] = tex.width;
            tex_height[i] = tex.height;
            checkCudaErrors(cudaMalloc((void**)&tex_data[i], tex.width * tex.height * 3 * sizeof(float)));
            checkCudaErrors(cudaMemcpy(tex_data[i], tex.data, tex.width * tex.height * 3 * sizeof(float), cudaMemcpyHostToDevice));
        }
        // copy tex_width to device
        checkCudaErrors(cudaMalloc((void**)&renderContext.tex_width, sc.numTextures * sizeof(int)));
        checkCudaErrors(cudaMemcpy(renderContext.tex_width, tex_width, sc.numTextures * sizeof(int), cudaMemcpyHostToDevice));
        // copy tex_height to device
        checkCudaErrors(cudaMalloc((void**)&renderContext.tex_height, sc.numTextures * sizeof(int)));
        checkCudaErrors(cudaMemcpy(renderContext.tex_height, tex_height, sc.numTextures * sizeof(int), cudaMemcpyHostToDevice));
        // copy tex_data to device
        checkCudaErrors(cudaMalloc((void**)&renderContext.tex_data, sc.numTextures * sizeof(float*)));
        checkCudaErrors(cudaMemcpy(renderContext.tex_data, tex_data, sc.numTextures * sizeof(float*), cudaMemcpyHostToDevice));

        delete[] tex_width;
        delete[] tex_height;
        delete[] tex_data;
    }
#endif
    renderContext.cam = cam;
    renderContext.initStats();
}

extern "C" void
runRenderer(int ns, int tx, int ty) {
    renderContext.ns = ns;

    // Render our buffer
    dim3 blocks((renderContext.nx + tx - 1) / tx, (renderContext.ny + ty - 1) / ty);
    dim3 threads(tx, ty);
    render <<<blocks, threads >>> (renderContext);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    renderContext.printStats();
}

extern "C" void
cleanupRenderer() {
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaFree(renderContext.fb));
    checkCudaErrors(cudaFree(renderContext.materials));
#ifdef USE_BVH_TEXTURE
    checkCudaErrors(cudaFree(renderContext.d_bvh));
    checkCudaErrors(cudaDestroyTextureObject(renderContext.bvh_tex));
#else
    checkCudaErrors(cudaFree(renderContext.nodes));
#endif // USE_BVH_TEXTURE


    cudaDeviceReset();
}