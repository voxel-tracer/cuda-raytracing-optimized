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
#define BVH
#define SHADOW
#define TEXTURES

#define EPSILON 0.01f

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
#define NUM_RAYS_MAX_TRAVERSED_NODES    16
#define NUM_RAYS_SIZE                   17
#endif

struct RenderContext {
    vec3* fb;

    triangle* tris;
    //uint32_t numTris;
    bvh_node* bvh;
    //uint32_t numBvhNodes;
    uint32_t firstLeafIdx;
    uint32_t numPrimitivesPerLeaf = 5; //TODO load this from bin file
    bbox bounds;

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
    __device__ void rayStat(int type) const {
        atomicAdd(stats + type, 1);
    }
    __device__ void maxStat(int type, uint64_t value) const {
        atomicMax(stats + type, value);
    }

    void initStats() {
        checkCudaErrors(cudaMallocManaged((void**)&stats, NUM_RAYS_SIZE * sizeof(uint64_t)));
        memset(stats, 0, NUM_RAYS_SIZE * sizeof(uint64_t));
    }
    void printStats() const {
        std::cerr << "num rays:\n";
        std::cerr << " primary             : " << std::fixed << stats[NUM_RAYS_PRIMARY] << std::endl;
        std::cerr << " primary hit mesh    : " << std::fixed << stats[NUM_RAYS_PRIMARY_HIT_MESH] << std::endl;
        std::cerr << " primary nohit       : " << std::fixed << stats[NUM_RAYS_PRIMARY_NOHITS] << std::endl;
        std::cerr << " primary bb nohit    : " << std::fixed << stats[NUM_RAYS_PRIMARY_BBOX_NOHITS] << std::endl;
        std::cerr << " secondary           : " << std::fixed << stats[NUM_RAYS_SECONDARY] << std::endl;
        std::cerr << " secondary no hit    : " << std::fixed << stats[NUM_RAYS_SECONDARY_NOHIT] << std::endl;
        std::cerr << " secondary bb nohit  : " << std::fixed << stats[NUM_RAYS_SECONDARY_BBOX_NOHIT] << std::endl;
        std::cerr << " secondary mesh      : " << std::fixed << stats[NUM_RAYS_SECONDARY_MESH] << std::endl;
        std::cerr << " secondary mesh nohit: " << std::fixed << stats[NUM_RAYS_SECONDARY_MESH_NOHIT] << std::endl;
        std::cerr << " shadows             : " << std::fixed << stats[NUM_RAYS_SHADOWS] << std::endl;
        std::cerr << " shadows nohit       : " << std::fixed << stats[NUM_RAYS_SHADOWS_NOHITS] << std::endl;
        std::cerr << " shadows bb nohit    : " << std::fixed << stats[NUM_RAYS_SHADOWS_BBOX_NOHITS] << std::endl;
        std::cerr << " power < 0.01        : " << std::fixed << stats[NUM_RAYS_LOW_POWER] << std::endl;
        std::cerr << " exceeded max bounce : " << std::fixed << stats[NUM_RAYS_EXCEED_MAX_BOUNCE] << std::endl;
        std::cerr << " russian roulette    : " << std::fixed << stats[NUM_RAYS_RUSSIAN_KILL] << std::endl;
        std::cerr << " max travers. nodes  : " << std::fixed << stats[NUM_RAYS_MAX_TRAVERSED_NODES] << std::endl;
        if (stats[NUM_RAYS_NAN] > 0)
            std::cerr << "*** " << stats[NUM_RAYS_NAN] << " NaNs detected" << std::endl;
    }
#else
    __device__ void rayStat(int type) const {}
    void initStats() {}
    void printStats() const {}
#endif
};

RenderContext renderContext;

__device__ float hitBvh(const ray& r, const RenderContext& context, float t_min, float t_max, tri_hit& rec, bool isShadow) {
    bool down = true; 
    int idx = 1;
    float closest = t_max;
    unsigned int bitStack = 0;
#ifdef BVH_COUNT
    uint64_t traversed = 0;
#endif
    while (true) {
        if (down) {
            bvh_node node = context.bvh[idx];
#ifdef BVH_COUNT
            traversed++;
#endif
            if (hit_bbox(node.min(), node.max(), r, closest)) {
                if (idx >= context.firstLeafIdx) { // leaf node
                    int first = (idx - context.firstLeafIdx) * context.numPrimitivesPerLeaf;
                    for (auto i = 0; i < context.numPrimitivesPerLeaf; i++) {
                        const triangle tri = context.tris[first + i];
                        if (isinf(tri.v[0].x()))
                            break; // we reached the end of the primitives buffer
                        float u, v;
                        float hitT = triangleHit(tri, r, t_min, closest, u, v);
                        if (hitT < closest) {
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

#ifdef BVH_COUNT
    context.maxStat(NUM_RAYS_MAX_TRAVERSED_NODES, traversed);
    rec.traversed = traversed;
#endif
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

#ifdef BVH
    return hitBvh(r, context, t_min, t_max, rec, isShadow);
#else
    float closest = FLT_MAX;
    for (uint32_t i = 0; i < context.numTris; i++) {
        float u, v;
        float hitT = triangleHit(context.tris[i], r, t_min, closest, u, v);
        if (hitT < FLT_MAX) {
            if (isShadow) return 0.0f;

            closest = hitT;
            rec.triId = i;
            rec.u = u;
            rec.v = v;
        }
    }
    return closest;
#endif // BVH
}

__device__ bool hit(const RenderContext& context, const path& p, float t_max, bool isShadow, intersection &inters) {
    const ray r = isShadow ? ray(p.origin, p.shadowDir) : ray(p.origin, p.rayDir);
    tri_hit triHit;
    bool primary = p.bounce == 0;
    inters.objId = NONE;
    if ((inters.t = hitMesh(r, context, EPSILON, t_max, triHit, primary, isShadow)) < t_max) {
        if (isShadow) return true; // we don't need to compute the intersection details for shadow rays

        inters.objId = TRIMESH;
        triangle tri = context.tris[triHit.triId];
        inters.meshID = tri.meshID;
        inters.normal = unit_vector(cross(tri.v[1] - tri.v[0], tri.v[2] - tri.v[0]));
        inters.texCoords[0] = (triHit.u * tri.texCoords[1 * 2 + 0] + triHit.v * tri.texCoords[2 * 2 + 0] + (1 - triHit.u - triHit.v) * tri.texCoords[0 * 2 + 0]);
        inters.texCoords[1] = (triHit.u * tri.texCoords[1 * 2 + 1] + triHit.v * tri.texCoords[2 * 2 + 1] + (1 - triHit.u - triHit.v) * tri.texCoords[0 * 2 + 1]);
#ifdef BVH_COUNT
        inters.traversed = triHit.traversed;
#endif
    } else {
        if (isShadow) return false; // shadow rays only care about the main triangle mesh
#ifdef BVH_COUNT
        inters.traversed = triHit.traversed;
#endif
        //if ((inters.t = planeHit(context.floor, r, EPSILON, FLT_MAX)) < FLT_MAX) {
        //    inters.objId = PLANE;
        //    inters.normal = context.floor.norm;
        //}
        //else 
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
#ifdef BVH_COUNT
    const uint64_t maxTraversed = 1500;
    const uint64_t targetBounce = 0;
#endif
#ifdef STATS
    bool fromMesh = false;
#endif
    for (p.bounce = 0; p.bounce < context.maxDepth; p.bounce++) {
#ifdef STATS
        bool primary = p.bounce == 0;
        context.rayStat(primary ? NUM_RAYS_PRIMARY : NUM_RAYS_SECONDARY);
        if (fromMesh) context.rayStat(NUM_RAYS_SECONDARY_MESH);
        if (p.attenuation.length() < 0.01f) context.rayStat(NUM_RAYS_LOW_POWER);
#endif
        intersection inters;
        if (!hit(context, p, FLT_MAX, false, inters)) {
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
#ifdef BVH_COUNT
            if (p.bounce == targetBounce) {
                float t = fminf(1.0f, ((float)inters.traversed) / maxTraversed);
                if (t <= 0.5f) {
                    t *= 2; // [0, .5] -> [0, 1]
                    p.color = (1 - t) * vec3(0, 0, 1) + t * vec3(0, 1, 0);
                }
                else {
                    // [0.5, 1] -> [0, 1]
                    t = (t - 0.5f) * 2;
                    p.color = (1 - t) * vec3(0, 1, 0) + t * vec3(1, 0, 0);
                }
            }
#endif // BVH_COUNT
            return;
        }

#ifdef STATS
        fromMesh = (inters.objId == TRIMESH);
        if (primary && !fromMesh) context.rayStat(NUM_RAYS_PRIMARY_NOHITS); // primary didn't intersect mesh, only floor
        if (primary && fromMesh) context.rayStat(NUM_RAYS_PRIMARY_HIT_MESH);
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
#ifdef BVH_COUNT
        if (p.bounce == targetBounce) {
            float t = fminf(1.0f, ((float)inters.traversed) / maxTraversed);
            if (t <= 0.5f) {
                t *= 2; // [0, .5] -> [0, 1]
                p.color = (1 - t) * vec3(0, 0, 1) + t * vec3(0, 1, 0);
            }
            else {
                // [0.5, 1] -> [0, 1]
                t = (t - 0.5f) * 2;
                p.color = (1 - t) * vec3(0, 1, 0) + t * vec3(1, 0, 0);
            }
            return;
        }
#endif
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
            context.rayStat(NUM_RAYS_SHADOWS);
#endif
            if (!hit(context, p, lightDist, true, inters)) {
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
    //col /= float(context.ns);
    //col[0] = sqrt(col[0]);
    //col[1] = sqrt(col[1]);
    //col[2] = sqrt(col[2]);
    context.fb[p.pixelId] = col / float(context.ns);
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

    checkCudaErrors(cudaMalloc((void**)&renderContext.tris, sc.m->numTris * sizeof(triangle)));
    checkCudaErrors(cudaMemcpy(renderContext.tris, sc.m->tris, sc.m->numTris * sizeof(triangle), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((void**)&renderContext.bvh, sc.m->numBvhNodes * sizeof(bvh_node)));
    checkCudaErrors(cudaMemcpy(renderContext.bvh, sc.m->bvh, sc.m->numBvhNodes * sizeof(bvh_node), cudaMemcpyHostToDevice));
    renderContext.firstLeafIdx = sc.m->numBvhNodes / 2;
    renderContext.bounds = sc.m->bounds;

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
    renderContext.numPrimitivesPerLeaf = sc.numPrimitivesPerLeaf;
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

    cudaDeviceReset();
}