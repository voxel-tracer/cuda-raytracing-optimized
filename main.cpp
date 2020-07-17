#include <iostream>
#include <time.h>
#include <float.h>

//#define CUBE

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define TINYOBJLOADER_IMPLEMENTATION 
#include <tiny_obj_loader.h>

#include "kernels.h"

#include "staircase_scene.h"

float random_float(unsigned int& state) {
    state = (214013 * state + 2531011);
    return (float)((state >> 16) & 0x7FFF) / 32767;
}

#define RND (random_float(rand_state))

int main() {
    bool perf = false;
    bool fast = true;
    int nx = (!perf && !fast) ? 640 : (!perf ? 320 : 160);
    int ny = (!perf && !fast) ? 800 : (!perf ? 400 : 200);
    int ns = (!perf && !fast) ? 1024 : (!perf ? 64 : 4); //!perf ? (fast ? 64 : 1024) : 4;
    int maxDepth = 8;
    int tx = 8;
    int ty = 8;

    std::cerr.imbue(std::locale(""));
    std::cerr << "Rendering a " << nx << "x" << ny << " image with " << ns << " samples per pixel ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    camera cam = setup_camera(nx, ny);

    scene sc;
    if (!load_scene(sc)) {
        return -1;
    }

    kernel_scene ksc;
    if (!setup_kernel_scene(sc, ksc)) {
        std::cerr << "Failed to setup kernel scene" << std::endl;
        return -1;
    }

    vec3 *fb;
    initRenderer(ksc, cam, &fb, nx, ny, maxDepth);

    clock_t start, stop;
    start = clock();
    runRenderer(ns, tx, ty);
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";

    if (!perf) {
        // Output FB as Image
        writePPM(nx, ny, fb);
    }

    // clean up
    cleanupRenderer();
}
