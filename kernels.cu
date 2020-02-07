#include <curand_kernel.h>
#include "vec3.h"
#include "hitable_list.h"
#include "camera.h"
#include "sphere.h"
#include "material.h"

vec3* m_fb;
curandStatePhilox4_32_10_t * d_rand_state;
curandStatePhilox4_32_10_t * d_rand_state2;
sphere* d_list;
hitable_list** d_world;
camera** d_camera;

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

__global__ void rand_init(curandStatePhilox4_32_10_t * rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1984, 0, 0, rand_state);
    }
}

#define RND (curand_uniform(&local_rand_state))

__global__ void create_world(sphere* d_list, hitable_list** d_world, camera** d_camera, int nx, int ny, curandStatePhilox4_32_10_t * rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curandStatePhilox4_32_10_t  local_rand_state = *rand_state;
        d_list[0] = sphere(vec3(0, -1000.0, -1), 1000,
            new material(vec3(0.5, 0.5, 0.5)));
        int i = 1;
        for (int a = -11; a < 11; a++) {
            for (int b = -11; b < 11; b++) {
                float choose_mat = RND;
                vec3 center(a + RND, 0.2, b + RND);
                if (choose_mat < 0.8f) {
                    d_list[i++] = sphere(center, 0.2,
                        new material(vec3(RND * RND, RND * RND, RND * RND)));
                }
                else if (choose_mat < 0.95f) {
                    d_list[i++] = sphere(center, 0.2,
                        new material(vec3(0.5f * (1.0f + RND), 0.5f * (1.0f + RND), 0.5f * (1.0f + RND)), 0.5f * RND));
                }
                else {
                    d_list[i++] = sphere(center, 0.2, new material(1.5));
                }
            }
        }
        d_list[i++] = sphere(vec3(0, 1, 0), 1.0, new material(1.5));
        d_list[i++] = sphere(vec3(-4, 1, 0), 1.0, new material(vec3(0.4, 0.2, 0.1)));
        d_list[i++] = sphere(vec3(4, 1, 0), 1.0, new material(vec3(0.7, 0.6, 0.5), 0.0));
        *rand_state = local_rand_state;
        *d_world = new hitable_list(d_list, 22 * 22 + 1 + 3);

        vec3 lookfrom(13, 2, 3);
        vec3 lookat(0, 0, 0);
        float dist_to_focus = 10.0; (lookfrom - lookat).length();
        float aperture = 0.1;
        *d_camera = new camera(lookfrom,
            lookat,
            vec3(0, 1, 0),
            30.0,
            float(nx) / float(ny),
            aperture,
            dist_to_focus);
    }
}

__global__ void render_init(int max_x, int max_y, curandStatePhilox4_32_10_t * rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    //Each thread gets same seed, a different sequence number, no offset
    curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

// Matching the C++ code would recurse enough into color() calls that
// it was blowing up the stack, so we have to turn this into a
// limited-depth loop instead.  Later code in the book limits to a max
// depth of 50, so we adapt this a few chapters early on the GPU.
__device__ vec3 color(const ray& r, hitable_list** world, curandStatePhilox4_32_10_t * local_rand_state) {
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0, 1.0, 1.0);
    for (int i = 0; i < 50; i++) {
        hit_record rec;
        if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            ray scattered;
            vec3 attenuation;
            if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
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

__global__ void render(vec3* fb, int max_x, int max_y, int ns, camera** cam, hitable_list** world, curandStatePhilox4_32_10_t * rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j * max_x + i;
    curandStatePhilox4_32_10_t  local_rand_state = rand_state[pixel_index];
    vec3 col(0, 0, 0);
    for (int s = 0; s < ns; s++) {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = (*cam)->get_ray(u, v, &local_rand_state);
        col += color(r, world, &local_rand_state);
    }
    rand_state[pixel_index] = local_rand_state;
    col /= float(ns);
    col[0] = sqrt(col[0]);
    col[1] = sqrt(col[1]);
    col[2] = sqrt(col[2]);
    fb[pixel_index] = col;
}

__global__ void free_world(sphere* d_list, hitable_list** d_world, camera** d_camera) {
    for (int i = 0; i < 22 * 22 + 1 + 3; i++) {
        delete d_list[i].mat_ptr;
    }
    delete* d_world;
    delete* d_camera;
}

extern "C" void
initRenderer(vec3 * *fb, int nx, int ny) {
    int num_pixels = nx * ny;
    size_t fb_size = num_pixels * sizeof(vec3);

    checkCudaErrors(cudaMallocManaged((void**)&m_fb, fb_size));
    *fb = m_fb;

    // allocate random state
    checkCudaErrors(cudaMalloc((void**)&d_rand_state, num_pixels * sizeof(curandStatePhilox4_32_10_t )));
    checkCudaErrors(cudaMalloc((void**)&d_rand_state2, 1 * sizeof(curandStatePhilox4_32_10_t )));

    // we need that 2nd random state to be initialized for the world creation
    rand_init <<<1, 1 >>> (d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // make our world of hitables & the camera
    int num_hitables = 22 * 22 + 1 + 3;
    checkCudaErrors(cudaMalloc((void**)&d_list, num_hitables*sizeof(sphere)));
    checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hitable_list*)));
    checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(camera*)));
    create_world << <1, 1 >> > (d_list, d_world, d_camera, nx, ny, d_rand_state2);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

extern "C" void
runRenderer(int nx, int ny, int ns, int tx, int ty) {
    // Render our buffer
    dim3 blocks(nx / tx + 1, ny / ty + 1);
    dim3 threads(tx, ty);
    render_init <<<blocks, threads >>> (nx, ny, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    render <<<blocks, threads >>> (m_fb, nx, ny, ns, d_camera, d_world, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

extern "C" void
cleanupRenderer() {
    checkCudaErrors(cudaDeviceSynchronize());
    free_world <<<1, 1 >>> (d_list, d_world, d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(d_rand_state2));
    checkCudaErrors(cudaFree(m_fb));

    cudaDeviceReset();
}