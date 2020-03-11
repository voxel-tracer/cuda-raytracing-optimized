#include <iostream>
#include <time.h>
#include <float.h>
#include <fstream>


// Required to include vec3.h
#include <cuda_runtime.h>
#include "helper_structs.h"

extern "C" void initRenderer(block* h_blocks, int numBlocks, uint3 center, material* h_materials, camera cam, vec3 * *fb, int nx, int ny);
extern "C" void runRenderer(int nx, int ny, int ns, int tx, int ty);
extern "C" void cleanupRenderer();

float random_float(unsigned int& state) {
    state = (214013 * state + 2531011);
    return (float)((state >> 16) & 0x7FFF) / 32767;
}

#define RND (random_float(rand_state))

camera setup_camera(int nx, int ny) {
    vec3 lookfrom(256, 200, 256);
    vec3 lookat(0, 0, 0);
    float dist_to_focus = (lookfrom - lookat).length();
    float aperture = 0.1;
    return camera(lookfrom,
        lookat,
        vec3(0, 1, 0),
        30.0,
        float(nx) / float(ny),
        aperture,
        dist_to_focus);
}

void loadFromVxt(const std::string& filepath, block** h_blocks, int &numBlocks, uint3 &center, material** h_materials) {
    const int coordRes = 128;
    const int blockRes = 32;

    std::fstream in(filepath, std::ios::in | std::ios::binary);
    // read magic word and confirm it's a supported file format
    char* MAGIC = "VXT_0.2";
    char magic[sizeof(MAGIC)];
    in.read(magic, sizeof(MAGIC));
    if (strcmp(MAGIC, magic)) {
        std::cerr << "invalid header " << magic << std::endl;
        exit(-1);
    }

    int numVoxels = 0;
    in.read((char*)&numVoxels, sizeof(int));
    in.read((char*)&numBlocks, sizeof(int));
    in.read((char*)&center, sizeof(uint3));
    block* blocks = new block[numBlocks];
    in.read((char*)blocks, numBlocks * sizeof(block));

    material* materials = new material[1];
    materials[0] = material(vec3(0.5, 0.5, 0.5));

    *h_blocks = blocks;
    *h_materials = materials;
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
    bool perf = true;
    int nx = !perf ? 1200 : 600;
    int ny = !perf ? 800 : 400;
    int ns = !perf ? 100 : 1;
    int tx = 8;
    int ty = 8;
    std::string input = "D:\\models\\xyzrgb_dragon_cleaned.v2.vxt";

    std::cerr << "Rendering a " << nx << "x" << ny << " image with " << ns << " samples per pixel ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    // init
    vec3 *fb;
    {
        block* blocks;
        int numBlocks;
        material* materials;
        uint3 center;
        loadFromVxt(input, &blocks, numBlocks, center, &materials);
        camera cam = setup_camera(nx, ny);
        initRenderer(blocks, numBlocks, center, materials, cam, &fb, nx, ny);
        delete[] blocks;
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
