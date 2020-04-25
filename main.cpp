#include <iostream>
#include <time.h>
#include <float.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define TINYOBJLOADER_IMPLEMENTATION 
#include <tiny_obj_loader.h>

// Required to include vec3.h
#include "helper_structs.h"

extern "C" void initRenderer(const vec3 * h_triangles, uint16_t numTris, material * h_materials, uint16_t numMats, const camera cam, vec3 * *fb, int nx, int ny);
extern "C" void initHDRi(float* data, int x, int y, int n);
extern "C" void runRenderer(int nx, int ny, int ns, int tx, int ty);
extern "C" void cleanupRenderer();

float random_float(unsigned int& state) {
    state = (214013 * state + 2531011);
    return (float)((state >> 16) & 0x7FFF) / 32767;
}

#define RND (random_float(rand_state))

camera setup_camera(int nx, int ny) {
    vec3 lookfrom(100, -150, 100);
    vec3 lookat(0, 0, 10);
    float dist_to_focus = (lookfrom - lookat).length();
    float aperture = 0.1;
    return camera(lookfrom,
        lookat,
        vec3(0, 0, 1),
        20.0,
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

vec3 hexColor(int hexValue) {
    float r = ((hexValue >> 16) & 0xFF);
    float g = ((hexValue >> 8) & 0xFF);
    float b = ((hexValue) & 0xFF);
    return vec3(r, g, b) / 255.0;
}

bool loadObj(const char * filename, vec3 ** h_triangles, uint16_t &numTris, material** h_materials, uint16_t &numMats, float floorHalfSize) {
//    std::string inputfile = "D:\\models\\lowpoly\\panter.obj";
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    
    std::string warn;
    std::string err;

    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filename);

    if (!warn.empty())
        std::cerr << warn << std::endl;
    if (!err.empty())
        std::cerr << err << std::endl;

    if (!ret)
        return false;
    
    std::cerr << " num vertices " << attrib.vertices.size() << std::endl;
    if (!materials.empty())
        std::cerr << " materials size " << materials.size() << std::endl;
    if (!attrib.texcoords.empty())
        std::cerr << " texcoord size " << attrib.texcoords.size() << std::endl;
    if (!attrib.colors.empty())
        std::cerr << " colors size " << attrib.colors.size() << std::endl;

    // first count how many triangles we have
    numTris = 0;
    for (auto s = 0; s < shapes.size(); s++) {
        numTris += shapes[s].mesh.num_face_vertices.size();
    }
    numTris += 2; // to account for the floor

    // loop over shapes and copy all triangles to array
    *h_triangles = new vec3[numTris * 3];
    uint16_t vec_index = 0;
    for (auto s = 0; s < shapes.size(); s++) {
        // loop over faces
        size_t index_offset = 0;
        for (auto f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            int fv = shapes[s].mesh.num_face_vertices[f];
            if (fv != 3)
                std::cerr << "face " << f << " of shape " << s << " has " << fv << " vertices" << std::endl;
            
            // loop over vertices in the face
            for (auto v = 0; v < 3; v++) {
                // access to vertex
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
                tinyobj::real_t vx = attrib.vertices[3 * idx.vertex_index + 0];
                tinyobj::real_t vy = attrib.vertices[3 * idx.vertex_index + 1];
                tinyobj::real_t vz = attrib.vertices[3 * idx.vertex_index + 2];
                //tinyobj::real_t nx = attrib.normals[3 * idx.vertex_index + 0];
                //tinyobj::real_t ny = attrib.normals[3 * idx.vertex_index + 1];
                //tinyobj::real_t nz = attrib.normals[3 * idx.vertex_index + 2];
                (*h_triangles)[vec_index++] = vec3(vx, vy, vz);
            }
            index_offset += 3;
        }
    }

    // add a floor at z = 0
    (*h_triangles)[vec_index++] = vec3(1, -1, -0.01) * floorHalfSize;
    (*h_triangles)[vec_index++] = vec3(1, 1, -0.01) * floorHalfSize;
    (*h_triangles)[vec_index++] = vec3(-1, 1, -0.01) * floorHalfSize;
    (*h_triangles)[vec_index++] = vec3(1, -1, -0.01) * floorHalfSize;
    (*h_triangles)[vec_index++] = vec3(-1, 1, -0.01) * floorHalfSize;
    (*h_triangles)[vec_index++] = vec3(-1, -1, -0.01) * floorHalfSize;

    // create a single material for all triangles
    unsigned int rand_state = 0;

    numMats = 2;
    *h_materials = new material[numMats];
    const vec3 modelColor(RND * RND, RND * RND, RND * RND);
    const vec3 floorColor1 = hexColor(0x511845);
    const vec3 floorColor2 = hexColor(0xff5733);

    (*h_materials)[0] = new_dielectric(1);
    //(*h_materials)[0] = new_lambertian(modelColor);
    //(*h_materials)[0] = new_metal(modelColor, 0.2);
    //(*h_materials)[0] = new_coat(modelColor, 1.5f);

    //(*h_materials)[1] = new_lambertian(floorColor1);
    //(*h_materials)[1] = new_metal(floorColor1, 0.2);
    //(*h_materials)[1] = new_coat(floorColor1, 1.5f);
    (*h_materials)[1] = new_checker(floorColor1, floorColor2, 0.2f);

    return true;
}

int main() {
    bool perf = false;
    bool fast = true;
    int nx = (!perf && !fast) ? 1200 : 600;
    int ny = (!perf && !fast) ? 800 : 400;
    int ns = !perf ? (fast ? 40 : 1024) : 1;
    int tx = 8;
    int ty = 8;

    std::cerr << "Rendering a " << nx << "x" << ny << " image with " << ns << " samples per pixel ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    // init
    vec3 *fb;
    {
        vec3* triangles;
        material* materials;
        uint16_t numTris;
        uint16_t numMats;
        if (!loadObj("D:\\models\\lowpoly\\panter.obj", &triangles, numTris, &materials, numMats, 200)) {
            return -1;
        }

        std::cerr << " there are " << numTris << " triangles" << std::endl;

        camera cam = setup_camera(nx, ny);

        initRenderer(triangles, numTris, materials, numMats, cam, &fb, nx, ny);
        delete[] triangles;
        delete[] materials;

        // load hdri
        //{
        //    int x, y, n;
        //    float* data = stbi_loadf("lebombo_1k.hdr", &x, &y, &n, 0);
        //    float max = 0;
        //    for (int i = 0; i < (x * y * n); i++) {
        //        max = fmaxf(max, data[i]);
        //    }

        //    std::cerr << "hdri(x = " << x << ", y = " << y << ", n = " << n << "). max = " << max << std::endl;
        //    initHDRi(data, x, y, n);
        //    stbi_image_free(data);
        //}
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
