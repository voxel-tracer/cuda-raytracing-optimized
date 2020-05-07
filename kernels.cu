#include <cuda_runtime.h>
#include "rnd.h"
#include "vec3.h"
#include "camera.h"
#include "triangle.h"
#include "material.h"

//#define STATS

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
    material* materials;
    uint16_t numMats;
    float* hdri = NULL;
    plane floor;
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

__device__ bool hitMesh(const ray& r, const RenderContext& context, float t_min, float t_max, hit_record& rec, bool primary, bool isShadow) {
    if (!hit_bbox(context.bounds, r, t_max)) {
#ifdef STATS
        if (isShadow) context.rayStat(NUM_RAYS_SHADOWS_BBOX_NOHITS);
        else context.rayStat(primary ? NUM_RAYS_PRIMARY_BBOX_NOHITS : NUM_RAYS_SECONDARY_BBOX_NOHIT);
#endif
        return false;
    }

    bool hit_anything = false;
    float closest_so_far = t_max;

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
                if (!hit_bbox(cbounds, r, closest_so_far)) continue; // ray doesn't intersect with cell's bounds

                // loop through cell's triangles
                for (uint16_t idx = d_gridC[ci]; idx < d_gridC[ci + 1]; idx++) {
                    if (triangleHit(d_triangles + d_gridL[idx] * 3, r, t_min, closest_so_far, rec)) {
                        if (isShadow) return true;

                        hit_anything = true;
                        closest_so_far = rec.t;
                    }
                }
            }
        }
    }

    return hit_anything;
}

__device__ bool hit(const ray& r, const RenderContext& context, float t_min, float t_max, hit_record& rec, bool primary, bool isShadow) {
    if (hitMesh(r, context, t_min, t_max, rec, primary, isShadow)) {
        rec.hitIdx = 0;
        return true;
    } else if (planeHit(context.floor, r, t_min, t_max, rec)) {
        rec.hitIdx = 1;
        return true;
    }

    return false;
}

__device__ bool generateShadowRay(const hit_record& hit, ray& shadow, vec3& emitted, rand_state& state) {
    const vec3 lightCenter(-2000, 0, 5000);
    const float lightRadius = 500;
    const vec3 lightColor = vec3(1, 1, 1) * 100;

    // create a random direction towards the light
    // coord system for sampling
    const vec3 sw = unit_vector(lightCenter - hit.p);
    const vec3 su = unit_vector(cross(fabs(sw.x()) > 0.01f ? vec3(0, 1, 0) : vec3(1, 0, 0), sw));
    const vec3 sv = cross(sw, su);

    // sample sphere by solid angle
    const float cosAMax = sqrt(1.0f - lightRadius * lightRadius / (hit.p - lightCenter).squared_length());
    const float eps1 = rnd(state);
    const float eps2 = rnd(state);
    const float cosA = 1.0f - eps1 + eps1 * cosAMax;
    const float sinA = sqrt(1.0f - cosA * cosA);
    const float phi = 2 * M_PI * eps2;
    const vec3 l = unit_vector(su * cosf(phi) * sinA + sv * sinf(phi) * sinA + sw * cosA);

    const float dotl = dot(l, hit.normal);
    if (dotl <= 0)
        return false;

    const float omega = 2 * M_PI * (1.0f - cosAMax);
    shadow = ray(hit.p, l);
    emitted = lightColor * dotl * omega / M_PI;

    return true;
}

// Matching the C++ code would recurse enough into color() calls that
// it was blowing up the stack, so we have to turn this into a
// limited-depth loop instead.  Later code in the book limits to a max
// depth of 50, so we adapt this a few chapters early on the GPU.
__device__ vec3 color(const ray& r, const RenderContext& context, rand_state& state) {
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0, 1.0, 1.0);
    vec3 curColor = vec3(0, 0, 0);
    bool fromMesh = false;
    for (int bounce = 0; bounce < 50; bounce++) {
        bool primary = bounce == 0;
#ifdef STATS
        context.rayStat(primary ? NUM_RAYS_PRIMARY : NUM_RAYS_SECONDARY);
        if (fromMesh) context.rayStat(NUM_RAYS_SECONDARY_MESH);
        if (cur_attenuation.length() < 0.01f) context.rayStat(NUM_RAYS_LOW_POWER);
#endif
        hit_record rec;
        if (hit(cur_ray, context, 0.001f, FLT_MAX, rec, primary, false)) {
            fromMesh = rec.hitIdx == 0;
#ifdef STATS
            if (primary && !fromMesh) context.rayStat(NUM_RAYS_PRIMARY_NOHITS); // primary didn't intersect mesh, only floor
#endif
            ray scattered;
            vec3 attenuation;
            bool hasShadow;
            if (scatter(context.materials[rec.hitIdx], cur_ray, rec, attenuation, scattered, state, hasShadow)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;

                // trace shadow ray if needed
                ray shadow;
                vec3 emitted;
                if (hasShadow && generateShadowRay(rec, shadow, emitted, state)) {
#ifdef STATS
                    context.rayStat(NUM_RAYS_SHADOWS);
#endif
                    if (!hit(shadow, context, 0.001f, FLT_MAX, rec, bounce == 0, true)) {
#ifdef STATS
                        context.rayStat(NUM_RAYS_SHADOWS_NOHITS);
#endif
                        // intersection point is illuminated by the light
                        curColor += emitted * cur_attenuation;
                    }
                }
            }
            else {
                return curColor;
            }
        }
        else {
#ifdef STATS
            if (primary) context.rayStat(NUM_RAYS_PRIMARY_NOHITS);
            else context.rayStat(fromMesh ? NUM_RAYS_SECONDARY_MESH_NOHIT : NUM_RAYS_SECONDARY_NOHIT);
#endif
            if (context.hdri != NULL) {
                // environment map
                vec3 dir = unit_vector(cur_ray.direction());
                uint2 coords = make_uint2(-atan2(dir.x(), dir.y()) * 1024 / (2 * M_PI), acos(dir.z()) * 512 / M_PI);
                vec3 c(
                    context.hdri[(coords.y * 1024 + coords.x)*3],
                    context.hdri[(coords.y * 1024 + coords.x)*3 + 1],
                    context.hdri[(coords.y * 1024 + coords.x)*3 + 2]
                );
                return cur_attenuation * c;
            }
            else {
                // sky color
                vec3 unit_direction = cur_ray.direction();
                float t = 0.5f * (unit_direction.z() + 1.0f);
                vec3 c = (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
                curColor += c * cur_attenuation;
                return curColor;

                // uniform sky color
                //curColor += cur_attenuation; // sky is (1, 1, 1)
                //return curColor;
            }
        }
    }
    return curColor; // exceeded recursion
}

__global__ void render(const RenderContext context) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= context.nx) || (j >= context.ny)) return;
    int pixel_index = j * context.nx + i;
    rand_state state = (wang_hash(pixel_index) * 336343633) | 1;

    vec3 col(0, 0, 0);
    for (int s = 0; s < context.ns; s++) {
        float u = float(i + rnd(state)) / float(context.nx);
        float v = float(j + rnd(state)) / float(context.ny);
        ray r = get_ray(context.cam, u, v, state);
        col += color(r, context, state);
    }
    col /= float(context.ns);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    context.fb[pixel_index] = col;
}

extern "C" void
initRenderer(const mesh m, material* h_materials, uint16_t numMats, plane floor, const camera cam, vec3 **fb, int nx, int ny) {
    renderContext.nx = nx;
    renderContext.ny = ny;
    renderContext.floor = floor;

    size_t fb_size = nx * ny * sizeof(vec3);
    checkCudaErrors(cudaMallocManaged((void**)&(renderContext.fb), fb_size));
    *fb = renderContext.fb;

    // all triangles share the same material
    checkCudaErrors(cudaMalloc((void**)&renderContext.materials, numMats * sizeof(material)));
    checkCudaErrors(cudaMemcpy(renderContext.materials, h_materials, numMats * sizeof(material), cudaMemcpyHostToDevice));
    renderContext.numMats = numMats;

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

extern "C" 
void initHDRi(float* data, int x, int y, int n) {
    checkCudaErrors(cudaMalloc((void**)&renderContext.hdri, x * y * n * sizeof(float)));
    checkCudaErrors(cudaMemcpy(renderContext.hdri, data, x * y * n * sizeof(float), cudaMemcpyHostToDevice));
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
    checkCudaErrors(cudaFree(renderContext.materials));
    checkCudaErrors(cudaFree(renderContext.fb));
    if (renderContext.hdri != NULL) checkCudaErrors(cudaFree(renderContext.hdri));

    cudaDeviceReset();
}