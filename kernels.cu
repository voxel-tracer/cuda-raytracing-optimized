#include <cuda_runtime.h>
#include "rnd.h"
#include "vec3.h"
#include "camera.h"
#include "triangle.h"
#include "material.h"

vec3* m_fb;
material* d_materials;
camera d_camera;
float* d_hdri = NULL;
int hdri_x;
int hdri_y;

const int kMaxTris = 600;
__device__ __constant__ vec3 d_triangles[kMaxTris * 3];

uint16_t d_numTris;
uint16_t d_numMats;

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

__device__ bool hit(const ray& r, uint16_t numTris, float t_min, float t_max, hit_record& rec, bool isShadow) {
    hit_record temp_rec;
    bool hit_anything = false;
    float closest_so_far = t_max;
    for (int i = 0; i < numTris; i++) {
        if (triangleHit(d_triangles + i * 3, r, t_min, closest_so_far, temp_rec)) {
            if (isShadow) return true;

            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
            rec.hitIdx = i;
        }
    }
    rec.hitIdx = rec.hitIdx < (numTris - 2) ? 0 : 1; // last 2 triangles is for the floor
    return hit_anything;
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
__device__ vec3 color(const ray& r, uint16_t numTris, material* materials, const float* hdri, rand_state& state) {
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0, 1.0, 1.0);
    vec3 curColor = vec3(0, 0, 0);
    for (int i = 0; i < 50; i++) {
        hit_record rec;
        if (hit(cur_ray, numTris, 0.001f, FLT_MAX, rec, false)) {
            ray scattered;
            vec3 attenuation;
            bool hasShadow;
            if (scatter(materials[rec.hitIdx], cur_ray, rec, attenuation, scattered, state, hasShadow)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;

                // trace shadow ray if needed
                ray shadow;
                vec3 emitted;
                if (hasShadow && generateShadowRay(rec, shadow, emitted, state) && !hit(shadow, numTris, 0.001f, FLT_MAX, rec, true)) {
                    // intersection point is illuminated by the light
                    curColor += emitted * cur_attenuation;
                }
            }
            else {
                return curColor;
            }
        }
        else {
            //vec3 dir = unit_vector(cur_ray.direction());
            //uint2 coords = make_uint2(-atan2(dir.x(), dir.y()) * 1024 / (2 * M_PI), acos(dir.z()) * 512 / M_PI);
            //vec3 c(
            //    hdri[(coords.y * 1024 + coords.x)*3],
            //    hdri[(coords.y * 1024 + coords.x)*3 + 1],
            //    hdri[(coords.y * 1024 + coords.x)*3 + 2]
            //);
            //return cur_attenuation * c;
            vec3 unit_direction = cur_ray.direction();
            float t = 0.5f * (unit_direction.z() + 1.0f);
            vec3 c = (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
            curColor += c * cur_attenuation;
            return curColor;
        }
    }
    return curColor; // exceeded recursion
}

__global__ void render(vec3* fb, uint16_t numTris, int max_x, int max_y, int ns, const camera cam, material* materials, uint16_t numMats, const float* hdri) {
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
        col += color(r, numTris, materials, hdri, state);
    }
    col /= float(ns);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    fb[pixel_index] = col;
}

extern "C" void
initRenderer(const vec3 *h_triangles, uint16_t numTris, material* h_materials, uint16_t numMats, const camera cam, vec3 **fb, int nx, int ny) {
    int num_pixels = nx * ny;
    size_t fb_size = num_pixels * sizeof(vec3);

    checkCudaErrors(cudaMallocManaged((void**)&m_fb, fb_size));
    *fb = m_fb;

    // all triangles share the same material
    checkCudaErrors(cudaMalloc((void**)&d_materials, numMats * sizeof(material)));
    checkCudaErrors(cudaMemcpy(d_materials, h_materials, numMats * sizeof(material), cudaMemcpyHostToDevice));
    d_numMats = numMats;

    checkCudaErrors(cudaMemcpyToSymbol(d_triangles, h_triangles, numTris * 3 * sizeof(vec3)));
    d_numTris = numTris;

    d_camera = cam;
}

extern "C" 
void initHDRi(float* data, int x, int y, int n) {
    checkCudaErrors(cudaMalloc((void**)&d_hdri, x * y * n * sizeof(float)));
    checkCudaErrors(cudaMemcpy(d_hdri, data, x * y * n * sizeof(float), cudaMemcpyHostToDevice));
}

extern "C" void
runRenderer(int nx, int ny, int ns, int tx, int ty) {
    // Render our buffer
    dim3 blocks(nx / tx + 1, ny / ty + 1);
    dim3 threads(tx, ty);
    render <<<blocks, threads >>> (m_fb, d_numTris, nx, ny, ns, d_camera, d_materials, d_numMats, d_hdri);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

extern "C" void
cleanupRenderer() {
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaFree(d_materials));
    checkCudaErrors(cudaFree(m_fb));
    if (d_hdri != NULL) checkCudaErrors(cudaFree(d_hdri));

    cudaDeviceReset();
}