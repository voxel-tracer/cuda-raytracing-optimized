#include <iostream>
#include <time.h>
#include <float.h>

//#define CUBE

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define TINYOBJLOADER_IMPLEMENTATION 
#include <tiny_obj_loader.h>

#include "kernels.h"

const mat3x3 xUp = {
    vec3(0,-1,0),
    vec3(1,0,0),
    vec3(0,0,1)
};

const mat3x3 yUp = {
    vec3(1,0,0),
    vec3(0,1,0),
    vec3(0,0,1)
};

const mat3x3 zUp = {
    vec3(1,0,0),
    vec3(0,0,1),
    vec3(0,-1,0)
};

float random_float(unsigned int& state) {
    state = (214013 * state + 2531011);
    return (float)((state >> 16) & 0x7FFF) / 32767;
}

#define RND (random_float(rand_state))

camera setup_camera(int nx, int ny, const mesh& m, vec3 lookfrom) {
    lookfrom = vec3(5.555139, 173.679901, 494.515045);
    vec3 lookat(5.555139, 173.679901, 493.515045);
    float dist_to_focus = (lookfrom - lookat).length();
    float aperture = 0.0f;
    return camera(lookfrom,
        lookat,
        vec3(0, 1, 0),
        42.0,
        float(nx) / float(ny),
        aperture,
        dist_to_focus);
}

// http://chilliant.blogspot.com.au/2012/08/srgb-approximations-for-hlsl.html
uint32_t LinearToSRGB(float x)
{
    x = fmaxf(x, 0.0f);
    x = fmaxf(1.055f * powf(x, 0.416666667f) - 0.055f, 0.0f);
    // u = min((uint32_t)(x * 255.9f), 255u)
    uint32_t u = (uint32_t)(x * 255.9f);
    u = u < 255u ? u : 255u;
    return u;
}


bool loadBVH(const char* input, mesh& m, int &numPrimitivesPerLeaf) {
    std::fstream in(input, std::ios::in | std::ios::binary);

    const char* HEADER = "BVH_00.02";
    char* header = new char[sizeof(HEADER)];
    in.read(header, sizeof(HEADER));
    if (!strcmp(HEADER, header))
        return false;

    in.read((char*)&m.numTris, sizeof(int));
    m.tris = new triangle[m.numTris];
    in.read((char*)m.tris, sizeof(triangle) * m.numTris);

    in.read((char*)&m.numBvhNodes, sizeof(int));
    m.bvh = new bvh_node[m.numBvhNodes];
    in.read((char*)m.bvh, sizeof(bvh_node) * m.numBvhNodes);

    in.read((char*)&m.bounds.min, sizeof(vec3));
    in.read((char*)&m.bounds.max, sizeof(vec3));

    in.read((char*)&numPrimitivesPerLeaf, sizeof(int));
}

int main() {
    bool perf = false;
    bool fast = false;
    int nx = (!perf && !fast) ? 640 : 320;
    int ny = (!perf && !fast) ? 800 : 400;
    int ns = !perf ? (fast ? 1 : 128) : 4;
    int tx = 8;
    int ty = 8;

    std::cerr.imbue(std::locale(""));
    std::cerr << "Rendering a " << nx << "x" << ny << " image with " << ns << " samples per pixel ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    material materials[20] = {
        { material_type::DIFFUSE, vec3(0.01, 0.01, 0.01), 0 },              // Black
        { material_type::METAL, vec3(0.27, 0.254, 0.15), 0.01 },            // Brass
        { material_type::METAL, vec3(0.5, 0.5, 0.5), 0 },                   // BrushedAluminium (metal with texture)
        { material_type::DIFFUSE, vec3(1, 1, 1), 0 },                       // Candles
        { material_type::DIFFUSE, vec3(0.117647, 0.054902, 0.0666667), 0 }, // ChairSeat
        { material_type::GLASS, vec3(1, 1, 1), 1.45 },                      // Glass
        { material_type::METAL, vec3(1.0, 0.95, 0.35), 0.05 },              // Gold
        { material_type::DIFFUSE, vec3(0.5, 0.5, 0.5), 0 },                 // Lampshade
        { material_type::DIFFUSE, vec3(0.578596, 0.578596, 0.578596), 0 },  // MagnoliaPaint
        { material_type::DIFFUSE, vec3(0.5, 0.5, 0.5), 0 },                 // Painting1
        { material_type::DIFFUSE, vec3(0.5, 0.5, 0.5), 0 },                 // Painting2
        { material_type::DIFFUSE, vec3(0.5, 0.5, 0.5), 0 },                 // Painting3
        { material_type::METAL, vec3(1.0, 1.0, 1.0), 0.1 },                 // StainlessSteel
        { material_type::DIFFUSE, vec3(0.15, 0.5, 0.5), 0 },                 // wallpaper
        { material_type::DIFFUSE, vec3(0.1578596, 0.578596, 0.1578596), 0 },  // whitePaint
        { material_type::DIFFUSE, vec3(1, 1, 1), 0 },                       // WhitePlastic
        { material_type::DIFFUSE, vec3(0.5, 0.5, 0.5), 0 },                 // WoodChair
        { material_type::DIFFUSE, vec3(0.5, 0.15, 0.15), 0 },                 // woodFloor
        { material_type::DIFFUSE, vec3(0.5, 0.5, 0.5), 0 },                 // WoodLamp
        { material_type::DIFFUSE, vec3(0.5, 0.5, 0.5), 0 },                 // woodstairs
    };
    scene staircase = { "D:\\models\\obj\\staircase.bvh" , yUp, 1, vec3(1,1,1), materials, 20 };

    scene sc = staircase;
    // init
    vec3 *fb;
    {
        plane floor = plane(vec3(0, -0.01, 0), vec3(0, 1, 0));
        mesh m;
        int numPrimitivesPerLeaf = 0;

        if (!loadBVH(sc.filename, m, numPrimitivesPerLeaf)) {
            std::cerr << "Failed to load bvh file" << std::endl;
            std::cerr.flush();
            return -1;
        }

        std::cerr << " there are " << m.numTris << " triangles" << ", and " << m.numBvhNodes << " bvh nodes" << std::endl;
        std::cerr << " bbox.min " << m.bounds.min << "\n bbox.max " << m.bounds.max << std::endl;

        camera cam = setup_camera(nx, ny, m, sc.camPos);

        // setup floor
        initRenderer(m, floor, cam, sc.mats, sc.numMats, &fb, nx, ny, 3);
    }

    clock_t start, stop;
    start = clock();
    runRenderer(ns, tx, ty);
    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";

    if (!perf) {
        // Output FB as Image
        std::cout << "P3\n" << nx << " " << ny << "\n255\n";
        for (int j = ny - 1; j >= 0; j--) {
            for (int i = 0; i < nx; i++) {
                size_t pixel_index = j * nx + i;
                int ir = LinearToSRGB(fb[pixel_index].r());
                int ig = LinearToSRGB(fb[pixel_index].g());
                int ib = LinearToSRGB(fb[pixel_index].b());
                std::cout << ir << " " << ig << " " << ib << "\n";
            }
        }
    }

    // clean up
    cleanupRenderer();
}
