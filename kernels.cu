#include <cuda_runtime.h>
#include "rnd.h"
#include "vec3.h"
#include "camera.h"
#include "sphere.h"
#include "material.h"

vec3* m_fb;
material* d_materials;
camera d_camera;

const int kNumHitable = 22 * 22 + 1 + 3;
__device__ __constant__ sphere d_spheres[kNumHitable];

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

__device__ bool hit(const ray& r, float t_min, float t_max, hit_record& rec) {
    hit_record temp_rec;
    bool hit_anything = false;
    float closest_so_far = t_max;
    for (int i = 0; i < kNumHitable; i++) {
        if (sphereHit(d_spheres[i], r, t_min, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
            rec.hitIdx = i;
        }
    }
    return hit_anything;
}

// Matching the C++ code would recurse enough into color() calls that
// it was blowing up the stack, so we have to turn this into a
// limited-depth loop instead.  Later code in the book limits to a max
// depth of 50, so we adapt this a few chapters early on the GPU.
__device__ vec3 color(const ray& r, material* materials, rand_state& state) {
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0, 1.0, 1.0);
    for (int i = 0; i < 50; i++) {
        hit_record rec;
        if (hit(cur_ray, 0.001f, FLT_MAX, rec)) {
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

__global__ void render(vec3* fb, int max_x, int max_y, int ns, const camera cam, material* materials) {
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
        col += color(r, materials, state);
    }
    col /= float(ns);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    fb[pixel_index] = col;
}

extern "C" void
initRenderer(sphere* h_spheres, material* h_materials, const camera cam, vec3 **fb, int nx, int ny) {
    int num_pixels = nx * ny;
    size_t fb_size = num_pixels * sizeof(vec3);

    checkCudaErrors(cudaMallocManaged((void**)&m_fb, fb_size));
    *fb = m_fb;

    checkCudaErrors(cudaMalloc((void**)&d_materials, kNumHitable * sizeof(material)));
    checkCudaErrors(cudaMemcpy(d_materials, h_materials, kNumHitable * sizeof(material), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpyToSymbol(d_spheres, h_spheres, kNumHitable * sizeof(sphere)));

    d_camera = cam;
}

extern "C" void
runRenderer(int nx, int ny, int ns, int tx, int ty) {
    // Render our buffer
    dim3 blocks(nx / tx + 1, ny / ty + 1);
    dim3 threads(tx, ty);
    render <<<blocks, threads >>> (m_fb, nx, ny, ns, d_camera, d_materials);
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