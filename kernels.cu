#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "rnd.h"
#include "vec3.h"
#include "camera.h"
#include "sphere.h"
#include "material.h"

#define METRIC

#ifdef METRIC
#include <cooperative_groups.h>
using namespace cooperative_groups;
#endif

//#define UBLOCKS
//#define BLOCKS

const int kMaxBounces = 50;

struct RenderContext {
    vec3* fb;
    int max_x;
    int max_y;
    int ns;
    material* materials;
    int numUBlocks;
    int numBlocks;
    uint3 center;
    uint64_t* numRays;
#ifdef METRIC
    uint64_t* divergence;

    __device__ void mark() const {
        auto g = coalesced_threads();
        if (g.thread_rank() == 0)
            atomicAdd(divergence + (g.size() - 1), 1);
    }
#else
    __device__ void mark() const {}
#endif
};

RenderContext context;
camera d_camera;

const int kMaxBlocks = 2000;
__device__ __constant__ block d_blocks[kMaxBlocks];

const int kMaxUBlocks = 100;
__device__ __constant__ block d_ublocks[kMaxUBlocks];

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

__device__ bool hit_box(const vec3& center, float halfSize, const ray& r, float t_min, float t_max, hit_record& rec) {
    int axis = 0;
    for (int a = 0; a < 3; a++) {
        float invD = 1.0f / r.direction()[a];
        float t0 = (center[a] - halfSize - r.origin()[a]) * invD;
        float t1 = (center[a] + halfSize - r.origin()[a]) * invD;
        if (invD < 0.0f) {
            float tmp = t0; t0 = t1; t1 = tmp;
        }
        if (t0 > t_min) {
            t_min = t0;
            axis = a;
        }
        t_max = t1 < t_max ? t1 : t_max;
        if (t_max <= t_min)
            return false;
    }

    rec.t = t_min;
    vec3 normal(0, 0, 0);
    if (axis == 0) normal[0] = r.direction().x() < 0 ? 1 : -1;
    else if (axis == 1) normal[1] = r.direction().y() < 0 ? 1 : -1;
    else normal[2] = r.direction().z() < 0 ? 1 : -1;
    rec.p = r.point_at_parameter(rec.t);
    rec.normal = normal;

    return true;
}

__device__ bool hit(const ray& r, const RenderContext &context, float t_min, float t_max, hit_record& rec, bool collectMetric) {
    const int blockRes = 32;
    const int ublockRes = 8;

    hit_record temp_rec;
    bool hit_anything = false;
    float closest_so_far = t_max;

    // loop through all ublocks
    for (int ui = 0; ui < context.numUBlocks; ui++) {
        const block& u = d_ublocks[ui];
        // decode ublock coordinates at 8^3 resolution
        int ux = (u.coords % ublockRes);
        int uy = ((u.coords >> 3) % ublockRes);
        int uz = ((u.coords >> 6) % ublockRes);

        // compute ublock center at 128^3, voxel, resolution
        const vec3 ucenter(
            (float)(ux)*16 + 7.5 - context.center.x,
            (float)(uy)*16 + 7.5,
            (float)(uz)*16 + 7.5 - context.center.z);
#ifdef UBLOCKS
        if (hit_box(ucenter * 2, 16, r, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
#else
        if (!hit_box(ucenter * 2, 16, r, t_min, closest_so_far, temp_rec))
            continue;
        const int lastBlockIdx = u.idx + __popcll(u.voxels); // count number of active blocks in current ublock
        // loop through all blocks
        for (int bi = u.idx; bi < lastBlockIdx; bi++) {
            const block& b = d_blocks[bi];
            // decode block coordinates
            int bx = (b.coords % blockRes) << 2;
            int by = ((b.coords >> 5) % blockRes) << 2;
            int bz = ((b.coords >> 10) % blockRes) << 2;

#ifdef BLOCKS
            if (hit_box(vec3(((float)bx - center.x + 1.5f) * 2, (by + 1.5f) * 2, ((float)bz - center.z + 1.5f) * 2), 4, r, t_min, closest_so_far, temp_rec)) {
                hit_anything = true;
                closest_so_far = temp_rec.t;
                rec = temp_rec;
            }
#else
            if (!hit_box(vec3(((float)bx - context.center.x + 1.5f) * 2, (by + 1.5f) * 2, ((float)bz - context.center.z + 1.5f) * 2), 4, r, t_min, closest_so_far, temp_rec))
                continue;

            // loop through all voxels and identify the ones that are set
            for (int xi = 0; xi < 4; xi++) {
                for (int yi = 0; yi < 4; yi++) {
                    for (int zi = 0; zi < 4; zi++) {
                        // compute voxel bit idx
                        int voxelBitIdx = xi + (yi << 2) + (zi << 4);
                        if (b.voxels & (1ULL << voxelBitIdx)) {
                            // compute voxel coordinates, centering the model around the origin
                            int x = bx + xi - context.center.x;
                            int y = by + yi;
                            int z = bz + zi - context.center.z;
                            if (collectMetric) context.mark();

                            if (hit_box(vec3(x, y, z) * 2, 1, r, t_min, closest_so_far, temp_rec)) {
                                hit_anything = true;
                                closest_so_far = temp_rec.t;
                                rec = temp_rec;
                            }
                        }
                    }
                }
            }
#endif // BLOCKS
        }
#endif // UBLOCKS
    }

    // we only have a single material for now
    rec.hitIdx = 0;

    return hit_anything;
}

// Matching the C++ code would recurse enough into color() calls that
// it was blowing up the stack, so we have to turn this into a
// limited-depth loop instead.  Later code in the book limits to a max
// depth of 50, so we adapt this a few chapters early on the GPU.
__device__ vec3 color(const ray& r, const RenderContext &context, rand_state& state) {
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0, 1.0, 1.0);
    for (int i = 0; i < kMaxBounces; i++) {
        atomicAdd(context.numRays, 1);
        hit_record rec;
        if (hit(cur_ray, context, 0.001f, FLT_MAX, rec, i > 0)) {
            ray scattered;
            vec3 attenuation;
            if (scatter(context.materials[rec.hitIdx], cur_ray, rec, attenuation, scattered, state)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            }
            else {
                return vec3(0.0, 0.0, 0.0);
            }
        }
        else {
            vec3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f * (unit_direction.y() + 1.0f);
            vec3 c = (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
            return cur_attenuation * c;
        }
    }
    return vec3(0.0, 0.0, 0.0); // exceeded recursion
}

__global__ void render(const RenderContext context, const camera cam) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= context.max_x) || (j >= context.max_y)) return;
    int pixel_index = j * context.max_x + i;
    rand_state state = (wang_hash(pixel_index) * 336343633) | 1;

    vec3 col(0, 0, 0);
    for (int s = 0; s < context.ns; s++) {
        float u = float(i + rnd(state)) / float(context.max_x);
        float v = float(j + rnd(state)) / float(context.max_y);
        ray r = get_ray(cam, u, v, state);
        col += color(r, context, state);
    }
    col /= float(context.ns);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    context.fb[pixel_index] = col;
}

extern "C" void
initRenderer(const voxelModel &model, material* h_materials, const camera cam, vec3 **fb, int nx, int ny) {
    int num_pixels = nx * ny;
    size_t fb_size = num_pixels * sizeof(vec3);

    checkCudaErrors(cudaMallocManaged((void**)&context.fb, fb_size));
    *fb = context.fb;
#ifdef METRIC
    checkCudaErrors(cudaMallocManaged((void**)&context.divergence, 32 * sizeof(uint64_t)));
    for (auto i = 0; i < 32; i++) context.divergence[i] = 0;
#endif
    checkCudaErrors(cudaMalloc((void**)&context.materials, sizeof(material)));
    checkCudaErrors(cudaMemcpy(context.materials, h_materials, sizeof(material), cudaMemcpyHostToDevice));

    context.center = model.center;
    context.numBlocks = model.numBlocks;
    checkCudaErrors(cudaMemcpyToSymbol(d_blocks, model.blocks, model.numBlocks * sizeof(block)));

    context.numUBlocks = model.numUBlocks;
    checkCudaErrors(cudaMemcpyToSymbol(d_ublocks, model.ublocks, model.numUBlocks * sizeof(block)));

    checkCudaErrors(cudaMallocManaged((void**)&context.numRays, sizeof(uint64_t)));
    context.numRays[0] = 0;

    d_camera = cam;
}

extern "C" void
runRenderer(int nx, int ny, int ns, int tx, int ty) {
    context.max_x = nx;
    context.max_y = ny;
    context.ns = ns;

    // Render our buffer
    dim3 blocks(nx / tx + 1, ny / ty + 1);
    dim3 threads(tx, ty);
    render <<<blocks, threads >>> (context, d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    std::cerr << "total rays traced = " << context.numRays[0] << std::endl;
#ifdef METRIC
    // first count total
    uint64_t total = 0;
    for (auto i = 0; i < 32; i++) total += context.divergence[i];
    // then print for each bucket ratio of warps
    std::cerr << "divergence metric, Total = "<< total << std::endl;
    for (auto i = 0; i < 32; i++) {
        std::cerr << "\t" << (i + 1) << ":\t" << (int)(context.divergence[i] * 100.0 / total) << std::endl;
    }
#endif
}

extern "C" void
cleanupRenderer() {
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaFree(context.materials));
    checkCudaErrors(cudaFree(context.fb));

    cudaDeviceReset();
}