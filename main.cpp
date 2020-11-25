#include <iostream>
#include <time.h>
#include <float.h>
#include <fstream>
#include <sstream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "kernels.h"

#include "staircase_scene.h"

//#define STORE_REFERENCE


float random_float(unsigned int& state) {
    state = (214013 * state + 2531011);
    return (float)((state >> 16) & 0x7FFF) / 32767;
}

#define RND (random_float(rand_state))

#ifdef STORE_REFERENCE
void saveReference(const std::string file, int nx, int ny, const vec3* colors) {
    std::fstream out(file, std::ios::out | std::ios::binary);
    const char* HEADER = "REF_00.01";
    out.write(HEADER, strlen(HEADER) + 1);
    out.write((char*)&nx, sizeof(int));
    out.write((char*)&ny, sizeof(int));
    out.write((char*)colors, sizeof(vec3) * nx * ny);
    out.close();
}
#endif // STORE_REFERENCE

bool loadReference(const std::string file, vec3* reference, int nx, int ny) {
    std::fstream in(file, std::ios::in | std::ios::binary);

    const char* HEADER = "REF_00.01";
    int headerLen = strlen(HEADER) + 1;
    char* header = new char[headerLen];
    in.read(header, headerLen);
    if (strcmp(HEADER, header) != 0) {
        std::cerr << "invalid header " << header << std::endl;
        return false;
    }

    int inNx, inNy;
    in.read((char*)&inNx, sizeof(int));
    in.read((char*)&inNy, sizeof(int));
    if (inNx != nx || inNy != ny) {
        std::cerr << "invalid nx, ny. Found " << inNx << ", " << inNy << ". Expected " << nx << ", " << ny << std::endl;
        return false;
    }

    in.read((char*)reference, sizeof(vec3) * nx * ny);
    in.close();

    return true;
}

int main(int argc, char** argv) {
    bool perf = true;
    bool fast = false;
    int nx = (!perf && !fast) ? 640 : (!perf ? 640 : 160);
    int ny = (!perf && !fast) ? 800 : (!perf ? 800 : 200);
    int ns = (!perf && !fast) ? 1024 : (!perf ? 256 : 4); //!perf ? (fast ? 64 : 1024) : 4;
    int maxDepth = perf ? 8 : 64;
    int tx = 8;
    int ty = 8;
    bool rmse = true;

    if (argc > 1)
        maxDepth = strtol(argv[1], NULL, 10);

    std::cerr.imbue(std::locale(""));
    std::cerr << "Rendering a " << nx << "x" << ny << " image with " << ns << " samples per pixel and max depth " << maxDepth;
    std::cerr << " in " << tx << "x" << ty << " blocks.\n";

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

    if (rmse) {
        std::stringstream file;
        file << "f" << nx << "-" << ny << ".ref";
        vec3* reference = new vec3[nx * ny];
        if (!loadReference(file.str(), reference, nx, ny)) {
            std::cerr << "Failed to load reference image" << std::endl;
            return -1;
        }

        double error = 0.0;
        for (auto i = 0; i < nx*ny; i++) {
            const vec3 f = fb[i];
            const vec3 g = reference[i];
            for (auto c = 0; c < 3; c++) {
                error += (f[c] - g[c]) * (f[c] - g[c]) / 3.0;
            }
        }
        error = sqrt(error / (nx * ny));
        std::cerr << "RMSE = " << error << std::endl;
        delete[] reference;
    }

#ifdef STORE_REFERENCE
    std::stringstream file;
    file << "f" << nx << "-" << ny << ".ref";
    saveReference(file.str(), nx, ny, fb);
#endif // STORE_REFERENCE


    // clean up
    cleanupRenderer();
}
