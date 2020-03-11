#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "rnd.h"
#include "vec3.h"
#include "camera.h"
#include "sphere.h"
#include "material.h"

vec3* m_fb;
material* d_materials;
camera d_camera;

const int kMaxBlocks = 2000;
__device__ __constant__ block d_blocks[kMaxBlocks];
int d_numBlocks;
uint3 d_center;

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

__device__ bool hit_box(const vec3& center, const ray& r, float t_min, float t_max, hit_record& rec) {
    int axis = 0;
    for (int a = 0; a < 3; a++) {
        float invD = 1.0f / r.direction()[a];
        float t0 = (center[a] - 1 - r.origin()[a]) * invD;
        float t1 = (center[a] + 1 - r.origin()[a]) * invD;
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
    //normal[axis] = r.direction()[axis] < 0 ? 1 : -1;
    if (axis == 0) normal[0] = r.direction().x() < 0 ? 1 : -1;
    else if (axis == 1) normal[1] = r.direction().y() < 0 ? 1 : -1;
    else normal[2] = r.direction().z() < 0 ? 1 : -1;
    rec.p = r.point_at_parameter(rec.t);
    rec.normal = normal;

    return true;
}

__device__ bool hit(const ray& r, int numBlocks, const uint3 &center, float t_min, float t_max, hit_record& rec) {
    const int coordRes = 128;
    const int blockRes = 32;

    hit_record temp_rec;
    bool hit_anything = false;
    float closest_so_far = t_max;

    // loop through all voxels
    for (int i = 0; i < numBlocks; i++) {
        const block& b = d_blocks[i];
        // decode block coordinates
        int bx = (b.coords % blockRes) << 2;
        int by = ((b.coords >> 5) % blockRes) << 2;
        int bz = ((b.coords >> 10) % blockRes) << 2;

        // loop through all voxels and identify the ones that are set
        for (int xi = 0; xi < 4; xi++) {
            for (int yi = 0; yi < 4; yi++) {
                for (int zi = 0; zi < 4; zi++) {
                    // compute voxel bit idx
                    int voxelBitIdx = xi + (yi << 2) + (zi << 4);
                    if (b.voxels & (1ULL << voxelBitIdx)) {
                        // compute voxel coordinates, centering the model around the origin
                        int x = bx + xi - center.x;
                        int y = by + yi;
                        int z = bz + zi - center.z;

                        if (hit_box(vec3(x, y, z) * 2, r, t_min, closest_so_far, temp_rec)) {
                            hit_anything = true;
                            closest_so_far = temp_rec.t;
                            rec = temp_rec;
                        }
                    }
                }
            }
        }
    }

    // we only have a single material for now
    rec.hitIdx = 0;

    return hit_anything;
}

// Matching the C++ code would recurse enough into color() calls that
// it was blowing up the stack, so we have to turn this into a
// limited-depth loop instead.  Later code in the book limits to a max
// depth of 50, so we adapt this a few chapters early on the GPU.
__device__ vec3 color(const ray& r, int numBlocks, const uint3& center, material* materials, rand_state& state) {
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0, 1.0, 1.0);
    for (int i = 0; i < 50; i++) {
        hit_record rec;
        if (hit(cur_ray, numBlocks, center, 0.001f, FLT_MAX, rec)) {
            ray scattered;
            vec3 attenuation;
            if (scatter(materials[rec.hitIdx], cur_ray, rec, attenuation, scattered, state)) {
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

__global__ void render(vec3* fb, int max_x, int max_y, int ns, const camera cam, material* materials, int numBlocks, uint3 center) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    rand_state state = (wang_hash(pixel_index) * 336343633) | 1;

    vec3 col(0, 0, 0);
    for (int s = 0; s < ns; s++) {
        float u = float(i + rnd(state)) / float(max_x);
        float v = float(j + rnd(state)) / float(max_y);
        ray r = get_ray(cam, u, v, state);
        col += color(r, numBlocks, center, materials, state);
    }
    col /= float(ns);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    fb[pixel_index] = col;
}

extern "C" void
initRenderer(block* h_blocks, int numBlocks, uint3 center, material* h_materials, const camera cam, vec3 **fb, int nx, int ny) {
    int num_pixels = nx * ny;
    size_t fb_size = num_pixels * sizeof(vec3);

    checkCudaErrors(cudaMallocManaged((void**)&m_fb, fb_size));
    *fb = m_fb;

    checkCudaErrors(cudaMalloc((void**)&d_materials, sizeof(material)));
    checkCudaErrors(cudaMemcpy(d_materials, h_materials, sizeof(material), cudaMemcpyHostToDevice));

    d_center = center;
    d_numBlocks = numBlocks;
    checkCudaErrors(cudaMemcpyToSymbol(d_blocks, h_blocks, numBlocks * sizeof(block)));

    d_camera = cam;
}

extern "C" void
runRenderer(int nx, int ny, int ns, int tx, int ty) {
    // Render our buffer
    dim3 blocks(nx / tx + 1, ny / ty + 1);
    dim3 threads(tx, ty);
    render <<<blocks, threads >>> (m_fb, nx, ny, ns, d_camera, d_materials, d_numBlocks, d_center);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

extern "C" void
cleanupRenderer() {
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaFree(d_materials));
    checkCudaErrors(cudaFree(m_fb));

    cudaDeviceReset();
}