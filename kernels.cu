#include <cuda_runtime.h>
#include "rnd.h"
#include "vec3.h"
#include "camera.h"
#include "intersections.h"
#include "material.h"
#include "scene_materials.h"

#define STATS
//#define RUSSIAN_ROULETTE

#define EPSILON 0.01f

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
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
#define NUM_RAYS_SIZE                   16
#endif

struct RenderContext {
    vec3* fb;

    vec3* tris;
    uint32_t numTris;
    bvh_node* bvh;
    uint32_t numBvhNodes;
    uint32_t firstLeafIdx;
    uint32_t numPrimitivesPerLeaf = 5; //TODO load this from bin file
    bbox bounds;

    plane floor;

    int nx;
    int ny;
    int ns;
    camera cam;

    sphere light = sphere(vec3(-2000, 0, 5000), 500);
    vec3 lightColor = vec3(1, 1, 1) * 100;

#ifdef STATS
    uint64_t* numRays;
    __device__ void rayStat(int type) const {
        atomicAdd(numRays + type, 1);
    }
    void initStats() {
        checkCudaErrors(cudaMallocManaged((void**)&numRays, NUM_RAYS_SIZE * sizeof(uint64_t)));
        memset(numRays, 0, NUM_RAYS_SIZE * sizeof(uint64_t));
    }
    void printStats() const {
        std::cerr << "num rays:\n";
        std::cerr << " primary             : " << std::fixed << numRays[NUM_RAYS_PRIMARY] << std::endl;
        std::cerr << " primary hit mesh    : " << std::fixed << numRays[NUM_RAYS_PRIMARY_HIT_MESH] << std::endl;
        std::cerr << " primary nohit       : " << std::fixed << numRays[NUM_RAYS_PRIMARY_NOHITS] << std::endl;
        std::cerr << " primary bb nohit    : " << std::fixed << numRays[NUM_RAYS_PRIMARY_BBOX_NOHITS] << std::endl;
        std::cerr << " secondary           : " << std::fixed << numRays[NUM_RAYS_SECONDARY] << std::endl;
        std::cerr << " secondary no hit    : " << std::fixed << numRays[NUM_RAYS_SECONDARY_NOHIT] << std::endl;
        std::cerr << " secondary bb nohit  : " << std::fixed << numRays[NUM_RAYS_SECONDARY_BBOX_NOHIT] << std::endl;
        std::cerr << " secondary mesh      : " << std::fixed << numRays[NUM_RAYS_SECONDARY_MESH] << std::endl;
        std::cerr << " secondary mesh nohit: " << std::fixed << numRays[NUM_RAYS_SECONDARY_MESH_NOHIT] << std::endl;
        std::cerr << " shadows             : " << std::fixed << numRays[NUM_RAYS_SHADOWS] << std::endl;
        std::cerr << " shadows nohit       : " << std::fixed << numRays[NUM_RAYS_SHADOWS_NOHITS] << std::endl;
        std::cerr << " shadows bb nohit    : " << std::fixed << numRays[NUM_RAYS_SHADOWS_BBOX_NOHITS] << std::endl;
        std::cerr << " power < 0.01        : " << std::fixed << numRays[NUM_RAYS_LOW_POWER] << std::endl;
        std::cerr << " exceeded max bounce : " << std::fixed << numRays[NUM_RAYS_EXCEED_MAX_BOUNCE] << std::endl;
        std::cerr << " russian roulette    : " << std::fixed << numRays[NUM_RAYS_RUSSIAN_KILL] << std::endl;
        if (numRays[NUM_RAYS_NAN] > 0)
            std::cerr << "*** " << numRays[NUM_RAYS_NAN] << " NaNs detected" << std::endl;
    }
#else
    __device__ void rayStat(int type) const {}
    void initStats() {}
    void printStats() const {}
#endif
};

RenderContext renderContext;

__device__ float hitBvh(const ray& r, const RenderContext& context, float t_min, tri_hit& rec, bool isShadow) {
    bool down = true; 
    int idx = 1;
    float closest = FLT_MAX;
    unsigned int bitStack = 0;

    while (true) {
        if (down) {
            bvh_node node = context.bvh[idx];
            if (hit_bbox(node.min(), node.max(), r, closest)) {
                if (idx >= context.firstLeafIdx) { // leaf node
                    int first = (idx - context.firstLeafIdx) * context.numPrimitivesPerLeaf;
                    for (auto i = 0; i < context.numPrimitivesPerLeaf; i++) {
                        if (isinf(context.tris[first * 3].x()))
                            break; // we reached the end of the primitives buffer
                        float u, v;
                        float hitT = triangleHit(context.tris + (first + i) * 3, r, t_min, closest, u, v);
                        if (hitT < FLT_MAX) {
                            if (isShadow) return 0.0f;

                            closest = hitT;
                            rec.triId = first + i;
                            rec.u = u;
                            rec.v = v;
                        }
                    }
                    down = false;
                } else { // internal node
                    // current -> left or right
                    const int childIdx = signbit(r.direction()[node.split_axis()]); // 0=left, 1=right
                    bitStack = (bitStack << 1) + childIdx; // push current child idx in the stack
                    idx = (idx << 1) + childIdx;
                }
            } else { // ray didn't intersect the node, backtrack
                down = false;
            }
        } else if (idx == 1) { // we backtracked up to the root node
            break;
        } else { // back tracking
            const int currentChildIdx = bitStack & 1;
            if ((idx & 1) == currentChildIdx) { // node == current child, visit sibling
                idx += -2 * currentChildIdx + 1; // node = node.sibling
                down = true;
            } else { // we visited both siblings, backtrack
                bitStack = bitStack >> 1;
                idx = idx >> 1; // node = node.parent
            }
        }
    }

    return closest;
}

__device__ float hitMesh(const ray& r, const RenderContext& context, float t_min, float t_max, tri_hit& rec, bool primary, bool isShadow) {
    if (!hit_bbox(context.bounds.min, context.bounds.max, r, t_max)) {
#ifdef STATS
        if (isShadow) context.rayStat(NUM_RAYS_SHADOWS_BBOX_NOHITS);
        else context.rayStat(primary ? NUM_RAYS_PRIMARY_BBOX_NOHITS : NUM_RAYS_SECONDARY_BBOX_NOHIT);
#endif
        return FLT_MAX;
    }

    return hitBvh(r, context, t_min, rec, isShadow);
}

__device__ bool hit(const RenderContext& context, const path& p, bool isShadow, intersection &inters) {
    const ray r = isShadow ? ray(p.origin, p.shadowDir) : ray(p.origin, p.rayDir);
    tri_hit triHit;
    bool primary = p.bounce == 0;
    inters.objId = NONE;
    if ((inters.t = hitMesh(r, context, EPSILON, FLT_MAX, triHit, primary, isShadow)) < FLT_MAX) {
        if (isShadow) return true; // we don't need to compute the intersection details for shadow rays

        inters.objId = TRIMESH;
        vec3 v0 = context.tris[triHit.triId * 3];
        vec3 v1 = context.tris[triHit.triId * 3 + 1];
        vec3 v2 = context.tris[triHit.triId * 3 + 2];

        inters.normal = unit_vector(cross(v1 - v0, v2 - v0));
    } else {
        if (isShadow) return false; // shadow rays only care about the main triangle mesh

        //if ((inters.t = planeHit(context.floor, r, EPSILON, FLT_MAX)) < FLT_MAX) {
        //    inters.objId = PLANE;
        //    inters.normal = context.floor.norm;
        //}
        //else 
        if (p.specular && sphereHit(context.light, r, EPSILON, FLT_MAX) < FLT_MAX) { // specular rays should intersect with the light
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

__device__ bool generateShadowRay(const RenderContext& context, path& p, const intersection &inters) {
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

    return true;
}

__device__ void color(const RenderContext& context, path& p) {
    p.attenuation = vec3(1.0, 1.0, 1.0);
    p.color = vec3(0, 0, 0);
#ifdef STATS
    bool fromMesh = false;
#endif
    for (p.bounce = 0; p.bounce < 10; p.bounce++) {
#ifdef STATS
        bool primary = p.bounce == 0;
        context.rayStat(primary ? NUM_RAYS_PRIMARY : NUM_RAYS_SECONDARY);
        if (fromMesh) context.rayStat(NUM_RAYS_SECONDARY_MESH);
        if (p.attenuation.length() < 0.01f) context.rayStat(NUM_RAYS_LOW_POWER);
#endif
        intersection inters;
        if (!hit(context, p, false, inters)) {
#ifdef PATH_DBG
            if (p.dbg) printf("bounce %d: NO_HIT\n", p.bounce);
#endif
#ifdef STATS
            if (primary) context.rayStat(NUM_RAYS_PRIMARY_NOHITS);
            else context.rayStat(fromMesh ? NUM_RAYS_SECONDARY_MESH_NOHIT : NUM_RAYS_SECONDARY_NOHIT);
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
        if (primary && !fromMesh) context.rayStat(NUM_RAYS_PRIMARY_NOHITS); // primary didn't intersect mesh, only floor
        if (primary && fromMesh) context.rayStat(NUM_RAYS_PRIMARY_HIT_MESH);
#endif
        if (inters.objId == LIGHT) {
#ifdef PATH_DBG
            if (p.dbg) printf("bounce %d: HIT LIGHT\n", p.bounce);
#endif
            // ray hit the light, compute its contribution and add it to the path's color
            p.color += p.attenuation * context.lightColor;
            return;
        }
#ifdef PATH_DBG
        if (p.dbg) printf("bounce %d: HIT %d at t %f with normal (%f, %f, %f)\n", p.bounce, inters.objId, inters.t, inters.normal.x(), inters.normal.y(), inters.normal.z());
#endif

        inters.inside = p.inside;

        scatter_info scatter(inters);
        if (inters.objId == TRIMESH)
            model_diffuse_scatter(scatter, inters, p.rayDir, p.rng);
        else 
            floor_diffuse_scatter(scatter, inters, p.rayDir, p.rng);

        p.origin += scatter.t * p.rayDir;
        p.rayDir = scatter.wi;
        p.attenuation *= scatter.throughput;
        p.specular = scatter.specular;
        p.inside = scatter.refracted ? !p.inside : p.inside;

        // trace shadow ray for diffuse rays
        if (!p.specular && generateShadowRay(context, p, inters)) {
#ifdef PATH_DBG
            if (p.dbg) printf("bounce %d: SHADOW\n", p.bounce);
#endif
#ifdef STATS
            context.rayStat(NUM_RAYS_SHADOWS);
#endif
            if (!hit(context, p, true, inters)) {
#ifdef PATH_DBG
                if (p.dbg) printf("bounce %d: SHADOW NO HIT\n", p.bounce);
#endif
#ifdef STATS
                context.rayStat(NUM_RAYS_SHADOWS_NOHITS);
#endif
                // intersection point is illuminated by the light
                p.color += p.lightContribution;
            }
        }

#ifdef RUSSIAN_ROULETTE
        // russian roulette
        if (p.bounce > 3) {
            float m = max(p.attenuation);
            if (rnd(p.rng) > m) {
#ifdef PATH_DBG
                if (p.dbg) printf("bounce %d: RUSSIAN ROULETTE BREAK\n", p.bounce);
#endif
#ifdef STATS
                context.rayStat(NUM_RAYS_RUSSIAN_KILL);
#endif
                return;
            }
            p.attenuation *= 1 / m;
        }
#endif
    }
    // exceeded recursion
#ifdef STATS
    context.rayStat(NUM_RAYS_EXCEED_MAX_BOUNCE);
#endif
}

__global__ void render(const RenderContext context) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= context.nx) || (j >= context.ny)) return;

    path p;
    p.pixelId = j * context.nx + i;
    p.rng = (wang_hash(p.pixelId) * 336343633) | 1;
#ifdef PATH_DBG
    const int dbgId = (context.ny - 308) * context.nx + 164;
    p.dbg = p.pixelId == dbgId;
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
        if (isnan(p.color)) context.rayStat(NUM_RAYS_NAN);
#endif
    }
    // color is specific to the pixel being traced, 
    col /= float(context.ns);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    context.fb[p.pixelId] = col;
}

extern "C" void
initRenderer(const mesh& m, plane floor, const camera cam, vec3 **fb, int nx, int ny) {
    renderContext.nx = nx;
    renderContext.ny = ny;
    renderContext.floor = floor;

    size_t fb_size = nx * ny * sizeof(vec3);
    checkCudaErrors(cudaMallocManaged((void**)&(renderContext.fb), fb_size));
    *fb = renderContext.fb;

    // copy scene to device
    vec3* vertices = new vec3[m.numTris * 3];
    for (auto t = 0, idx = 0; t < m.numTris; t++) {
        const triangle& tri = m.tris[t];
        for (auto v = 0; v < 3; v++, idx++)
            vertices[idx] = tri.v[v];
    }

    checkCudaErrors(cudaMalloc((void**)&renderContext.tris, m.numTris * 3 * sizeof(vec3)));
    checkCudaErrors(cudaMemcpy(renderContext.tris, vertices, m.numTris * 3 * sizeof(vec3), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void**)&renderContext.bvh, m.numBvhNodes * sizeof(bvh_node)));
    checkCudaErrors(cudaMemcpy(renderContext.bvh, m.bvh, m.numBvhNodes * sizeof(bvh_node), cudaMemcpyHostToDevice));
    renderContext.numTris = m.numTris;
    renderContext.numBvhNodes = m.numBvhNodes;
    renderContext.firstLeafIdx = m.numBvhNodes / 2;
    renderContext.bounds = m.bounds;

    renderContext.cam = cam;

    renderContext.initStats();

    delete[] vertices;
}

extern "C" void
runRenderer(int ns, int tx, int ty) {
    renderContext.ns = ns;

    // Render our buffer
    dim3 blocks(renderContext.nx / tx + 1, renderContext.ny / ty + 1);
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

    cudaDeviceReset();
}