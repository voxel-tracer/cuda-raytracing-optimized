#include <iostream>
#include <time.h>
#include <float.h>


// Required to include vec3.h
#include <cuda_runtime.h>
#include "helper_structs.h"

extern "C" void initRenderer(sphere* h_spheres, material* h_materials, camera cam, vec3 * *fb, int nx, int ny);
extern "C" void runRenderer(int nx, int ny, int ns, int tx, int ty);
extern "C" void cleanupRenderer();

float random_float(unsigned int& state) {
    state = (214013 * state + 2531011);
    return (float)((state >> 16) & 0x7FFF) / 32767;
}

#define RND (random_float(rand_state))

camera setup_camera(int nx, int ny) {
    vec3 lookfrom(13, 2, 3);
    vec3 lookat(0, 0, 0);
    float dist_to_focus = 10.0; (lookfrom - lookat).length();
    float aperture = 0.1;
    return camera(lookfrom,
        lookat,
        vec3(0, 1, 0),
        30.0,
        float(nx) / float(ny),
        aperture,
        dist_to_focus);
}

void setup_scene(sphere** h_spheres, material** h_materials) {
    int numHitable = 22 * 22 + 1 + 3;
    sphere* spheres = new sphere[numHitable];
    material* materials = new material[numHitable];

    unsigned int rand_state = 0;

    materials[0] = material(vec3(0.5, 0.5, 0.5));
    spheres[0] = sphere(vec3(0, -1000.0, -1), 1000);
    int i = 1;
    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            float choose_mat = RND;
            vec3 center(a + RND, 0.2, b + RND);
            if (choose_mat < 0.8f) {
                materials[i] = material(vec3(RND * RND, RND * RND, RND * RND));
                spheres[i++] = sphere(center, 0.2);
            }
            else if (choose_mat < 0.95f) {
                materials[i] = material(vec3(0.5f * (1.0f + RND), 0.5f * (1.0f + RND), 0.5f * (1.0f + RND)), 0.5f * RND);
                spheres[i++] = sphere(center, 0.2);
            }
            else {
                materials[i] = material(1.5);
                spheres[i++] = sphere(center, 0.2);
            }
        }
    }
    materials[i] = material(1.5);
    spheres[i++] = sphere(vec3(0, 1, 0), 1.0);
    materials[i] = material(vec3(0.4, 0.2, 0.1));
    spheres[i++] = sphere(vec3(-4, 1, 0), 1.0);
    materials[i] = material(vec3(0.7, 0.6, 0.5), 0);
    spheres[i++] = sphere(vec3(4, 1, 0), 1.0);

    *h_spheres = spheres;
    *h_materials = materials;
}

int main() {
    bool perf = false;
    int nx = !perf ? 1200 : 600;
    int ny = !perf ? 800 : 400;
    int ns = !perf ? 100 : 1;
    int tx = 8;
    int ty = 8;

    std::cerr << "Rendering a " << nx << "x" << ny << " image with " << ns << " samples per pixel ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    // init
    vec3 *fb;
    {
        sphere* spheres;
        material* materials;
        setup_scene(&spheres, &materials);
        camera cam = setup_camera(nx, ny);
        initRenderer(spheres, materials, cam, &fb, nx, ny);
        delete[] spheres;
        delete[] materials;
    }

    clock_t start, stop;
    start = clock();
    runRenderer(nx, ny, ns, tx, ty);
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";

    if (!perf) {
        // Output FB as Image
        std::cout << "P3\n" << nx << " " << ny << "\n255\n";
        for (int j = ny - 1; j >= 0; j--) {
            for (int i = 0; i < nx; i++) {
                size_t pixel_index = j * nx + i;
                int ir = int(255.99 * fb[pixel_index].r());
                int ig = int(255.99 * fb[pixel_index].g());
                int ib = int(255.99 * fb[pixel_index].b());
                std::cout << ir << " " << ig << " " << ib << "\n";
            }
        }
    }

    // clean up
    cleanupRenderer();
}
