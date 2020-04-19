#include <iostream>
#include <time.h>
#include <float.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define TINYOBJLOADER_IMPLEMENTATION 
#include <tiny_obj_loader.h>

// Required to include vec3.h
#include "helper_structs.h"

extern "C" void initRenderer(sphere* h_spheres, material* h_materials, camera cam, vec3 * *fb, int nx, int ny);
extern "C" void initHDRi(float* data, int x, int y, int n);
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

    materials[0] = new_lambertian(vec3(0.5, 0.5, 0.5));
    spheres[0] = sphere(vec3(0, -1000.0, -1), 1000);
    int i = 1;
    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            float choose_mat = RND;
            vec3 center(a + RND, 0.2, b + RND);
            if (choose_mat < 0.8f) {
                materials[i] = new_coat(vec3(RND * RND, RND * RND, RND * RND), 1.5f);
                spheres[i++] = sphere(center, 0.2);
            }
            else if (choose_mat < 0.95f) {
                materials[i] = new_metal(vec3(0.5f * (1.0f + RND), 0.5f * (1.0f + RND), 0.5f * (1.0f + RND)), 0.5f * RND);
                spheres[i++] = sphere(center, 0.2);
            }
            else {
                materials[i] = new_dielectric(1.5);
                spheres[i++] = sphere(center, 0.2);
            }
        }
    }
    materials[i] = new_dielectric(1.5);
    spheres[i++] = sphere(vec3(0, 1, 0), 1.0);
    materials[i] = new_coat(vec3(0.4, 0.2, 0.1), 1.5f);
    spheres[i++] = sphere(vec3(-4, 1, 0), 1.0);
    materials[i] = new_metal(vec3(0.7, 0.6, 0.5), 0);
    spheres[i++] = sphere(vec3(4, 1, 0), 1.0);

    *h_spheres = spheres;
    *h_materials = materials;
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

void loadObj() {
    std::string inputfile = "D:\\models\\lowpoly\\panter.obj";
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    
    std::string warn;
    std::string err;

    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, inputfile.c_str());

    if (!warn.empty())
        std::cerr << warn << std::endl;
    if (!err.empty())
        std::cerr << err << std::endl;

    if (!ret)
        return;

    std::cerr << " num vertices " << attrib.vertices.size() << std::endl;
    if (!materials.empty())
        std::cerr << " materials size " << materials.size() << std::endl;
    if (!attrib.texcoords.empty())
        std::cerr << " texcoord size " << attrib.texcoords.size() << std::endl;
    if (!attrib.colors.empty())
        std::cerr << " colors size " << attrib.colors.size() << std::endl;

    // loop over shapes
    for (auto s = 0; s < shapes.size(); s++) {
        // loop over faces
        size_t index_offset = 0;
        for (auto f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            int fv = shapes[s].mesh.num_face_vertices[f];
            if (fv != 3)
                std::cerr << "face " << f << " of shape " << s << " has " << fv << " vertices" << std::endl;
            
            // loop over vertices in the face
            for (auto v = 0; v < fv; v++) {
                // access to vertex
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
                tinyobj::real_t vx = attrib.vertices[3 * idx.vertex_index + 0];
                tinyobj::real_t vy = attrib.vertices[3 * idx.vertex_index + 1];
                tinyobj::real_t vz = attrib.vertices[3 * idx.vertex_index + 2];
                tinyobj::real_t nx = attrib.normals[3 * idx.vertex_index + 0];
                tinyobj::real_t ny = attrib.normals[3 * idx.vertex_index + 1];
                tinyobj::real_t nz = attrib.normals[3 * idx.vertex_index + 2];
            }
            index_offset += fv;
        }
    }
}
int main() {
    bool perf = true;
    int nx = !perf ? 1200 : 600;
    int ny = !perf ? 800 : 400;
    int ns = !perf ? 2048 : 1;
    int tx = 8;
    int ty = 8;

    std::cerr << "Rendering a " << nx << "x" << ny << " image with " << ns << " samples per pixel ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    loadObj();

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

        // load hdri
        int x, y, n;
        float* data = stbi_loadf("lebombo_1k.hdr", &x, &y, &n, 0);
        float max = 0;
        for (int i = 0; i < (x * y * n); i++) {
            max = fmaxf(max, data[i]);
        }

        std::cerr << "hdri(x = " << x << ", y = " << y << ", n = " << n << "). max = " << max << std::endl;
        initHDRi(data, x, y, n);
        stbi_image_free(data);
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
