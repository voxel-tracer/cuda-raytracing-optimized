#include <cuda_runtime.h>
#include "rnd.h"
#include "vec3.h"
#include "camera.h"
#include "intersections.h"
#include "material.h"
#include "scene_materials.h"

#define STATS 
#define RUSSIAN_ROULETTE

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

const int kMaxTris = 1000;
const int kMaxC = 1100;
const int kMaxL = 3600;
__device__ __constant__ vec3 d_triangles[kMaxTris * 3];
__device__ __constant__ uint16_t d_gridC[kMaxC];
__device__ __constant__ uint16_t d_gridL[kMaxL];

#ifdef STATS
#define NUM_RAYS_PRIMARY                0
#define NUM_RAYS_PRIMARY_NOHITS         1
#define NUM_RAYS_PRIMARY_BBOX_NOHITS    2
#define NUM_RAYS_SECONDARY              3
#define NUM_RAYS_SECONDARY_MESH         4
#define NUM_RAYS_SECONDARY_NOHIT        5
#define NUM_RAYS_SECONDARY_MESH_NOHIT   6
#define NUM_RAYS_SECONDARY_BBOX_NOHIT   7
#define NUM_RAYS_SHADOWS                8
#define NUM_RAYS_SHADOWS_BBOX_NOHITS    9
#define NUM_RAYS_SHADOWS_NOHITS         10
#define NUM_RAYS_LOW_POWER              11
#define NUM_RAYS_SIZE                   12
#endif

struct RenderContext {
    vec3* fb;
    uint16_t numTris;
    bbox bounds;
    grid g;
    int nx;
    int ny;
    int ns;
    camera cam;
    plane floor;

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
        std::cerr << " power < 0.1         : " << std::fixed << numRays[NUM_RAYS_LOW_POWER] << std::endl;
    }
#else
    __device__ void rayStat(int type) const {}
    void initStats() {}
    void printStats() const {}
#endif
};

RenderContext renderContext;

__device__ float hitMesh(const ray& r, const RenderContext& context, float t_min, float t_max, tri_hit& rec, bool primary, bool isShadow) {
    if (!hit_bbox(context.bounds, r, t_max)) {
#ifdef STATS
        if (isShadow) context.rayStat(NUM_RAYS_SHADOWS_BBOX_NOHITS);
        else context.rayStat(primary ? NUM_RAYS_PRIMARY_BBOX_NOHITS : NUM_RAYS_SECONDARY_BBOX_NOHIT);
#endif
        return FLT_MAX;
    }

    float closest = FLT_MAX;

    // loop through grid cells
    const grid& g = context.g;
    for (uint16_t cz = 0, ci = 0; cz < g.size.z(); cz++) {
        for (uint16_t cy = 0; cy < g.size.y(); cy++) {
            for (uint16_t cx = 0; cx < g.size.x(); cx++, ci++) {
                if (d_gridC[ci] == d_gridC[ci + 1]) continue; // empty cell
                // check if ray intersects cell bounds
                bbox cbounds(
                    vec3(cx, cy, cz) * g.cellSize + context.bounds.min,
                    vec3(cx + 1, cy + 1, cz + 1) * g.cellSize + context.bounds.min
                );
                if (!hit_bbox(cbounds, r, closest)) continue; // ray doesn't intersect with cell's bounds

                // loop through cell's triangles
                for (uint16_t idx = d_gridC[ci]; idx < d_gridC[ci + 1]; idx++) {
                    float u, v;
                    float hitT = triangleHit(d_triangles + d_gridL[idx] * 3, r, t_min, closest, u, v);
                    if (hitT < FLT_MAX) {
                        if (isShadow) return 0.0f;

                        closest = hitT;
                        rec.triId = d_gridL[idx];
                        rec.u = u;
                        rec.v = v;
                    }
                }
            }
        }
    }

    return closest;
}

__device__ bool hit(const RenderContext& context, const path& p, bool isShadow, intersection &inters) {
    const ray r = isShadow ? ray(p.origin, p.shadowDir) : ray(p.origin, p.rayDir);
    tri_hit triHit;
    bool primary = p.bounce == 0;
    inters.objId = NONE;
    if ((inters.t = hitMesh(r, context, EPSILON, FLT_MAX, triHit, primary, isShadow)) < FLT_MAX) {
        if (isShadow) return true; // we don't need to compute the intersection details for shadow rays

        inters.objId = TRIMESH;
        vec3 v0 = d_triangles[triHit.triId * 3];
        vec3 v1 = d_triangles[triHit.triId * 3 + 1];
        vec3 v2 = d_triangles[triHit.triId * 3 + 2];

        inters.normal = unit_vector(cross(v1 - v0, v2 - v0));
    } else {
        if (isShadow) return false; // shadow rays only care about the main triangle mesh

        if ((inters.t = planeHit(context.floor, r, EPSILON, FLT_MAX)) < FLT_MAX) {
            inters.objId = PLANE;
            inters.normal = context.floor.norm;
        }
        else if (p.specular && sphereHit(context.light, r, EPSILON, FLT_MAX) < FLT_MAX) { // specular rays should intersect with the light
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
    for (p.bounce = 0; p.bounce < 50; p.bounce++) {
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
            float t = 0.5f * (p.rayDir.z() + 1.0f);
            vec3 c = (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
            p.color += p.attenuation * c;

            break;
        }

#ifdef STATS
        fromMesh = (inters.objId == TRIMESH);
        if (primary && !fromMesh) context.rayStat(NUM_RAYS_PRIMARY_NOHITS); // primary didn't intersect mesh, only floor
#endif
        if (inters.objId == LIGHT) {
#ifdef PATH_DBG
            if (p.dbg) printf("bounce %d: HIT LIGHT\n", p.bounce);
#endif
            // ray hit the light, compute its contribution and add it to the path's color
            p.color += p.attenuation * context.lightColor;
            break;
        }
#ifdef PATH_DBG
        if (p.dbg) printf("bounce %d: HIT %d at t %f with normal (%f, %f, %f)\n", p.bounce, inters.objId, inters.t, inters.normal.x(), inters.normal.y(), inters.normal.z());
#endif

        inters.inside = p.inside;

        scatter_info scatter;
        if (inters.objId == TRIMESH)
            model_coat_scatter(scatter, inters, p.rayDir, p.rng);
        else 
            floor_coat_scatter(scatter, inters, p.rayDir, p.rng);

        p.origin = inters.p;
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
                break;
            }
            p.attenuation *= 1 / m;
        }
#endif
    }
    // exceeded recursion
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
    }
    // color is specific to the pixel being traced, 
    col /= float(context.ns);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    context.fb[p.pixelId] = col;
}

extern "C" void
initRenderer(const mesh m, plane floor, const camera cam, vec3 **fb, int nx, int ny) {
    renderContext.nx = nx;
    renderContext.ny = ny;
    renderContext.floor = floor;

    size_t fb_size = nx * ny * sizeof(vec3);
    checkCudaErrors(cudaMallocManaged((void**)&(renderContext.fb), fb_size));
    *fb = renderContext.fb;

    checkCudaErrors(cudaMemcpyToSymbol(d_triangles, m.tris, m.numTris * 3 * sizeof(vec3)));
    renderContext.numTris = m.numTris;
    renderContext.bounds = m.bounds;

    // copy grid to gpu
    renderContext.g = m.g;
    checkCudaErrors(cudaMemcpyToSymbol(d_gridC, m.g.C, m.g.sizeC() * sizeof(uint16_t)));
    checkCudaErrors(cudaMemcpyToSymbol(d_gridL, m.g.L, m.g.sizeL() * sizeof(uint16_t)));
    renderContext.cam = cam;

    renderContext.initStats();
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