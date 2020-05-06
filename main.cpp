#include <iostream>
#include <time.h>
#include <float.h>

//#define CUBE

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define TINYOBJLOADER_IMPLEMENTATION 
#include <tiny_obj_loader.h>

// Required to include vec3.h
#include "helper_structs.h"

extern "C" void initRenderer(const mesh m, material * h_materials, uint16_t numMats, plane floor, const camera cam, vec3 * *fb, int nx, int ny);
extern "C" void initHDRi(float* data, int x, int y, int n);
extern "C" void runRenderer(int ns, int tx, int ty);
extern "C" void cleanupRenderer();

float random_float(unsigned int& state) {
    state = (214013 * state + 2531011);
    return (float)((state >> 16) & 0x7FFF) / 32767;
}

#define RND (random_float(rand_state))

camera setup_camera(int nx, int ny) {
#ifdef CUBE
    vec3 lookfrom(5, -7.5, 5);
    vec3 lookat(0, 0, 0);
#else
    vec3 lookfrom(100, -150, 100);
    vec3 lookat(0, 0, 10);
#endif
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

void buildGrid(mesh& m, float cellSize) {
    // compute grid size in cells
    vec3 gridSize = ceil((m.bounds.max - m.bounds.min) / cellSize);
    std::cerr << "grid size = " << gridSize << std::endl;

    const uint16_t N = gridSize.x() * gridSize.y() * gridSize.z();
    std::vector<std::vector<uint16_t>> cells;
    for (auto i = 0; i < N; i++) {
        std::vector<uint16_t> v;
        cells.push_back(v);
    }

    // loop over all triangles and add them to the corresponding cells
    for (auto i = 0; i < m.numTris; i++) {
        // compute triangle bbox
        vec3 p1 = m.tris[i * 3];
        vec3 p2 = m.tris[i * 3 + 1];
        vec3 p3 = m.tris[i * 3 + 2];
        vec3 bmin = min(p1, min(p2, p3));
        vec3 bmax = max(p1, max(p2, p3));

        vec3 gmin = floor((bmin - m.bounds.min) / cellSize);
        vec3 gmax = floor((bmax - m.bounds.min) / cellSize);

        for (int x = gmin.x(); x <= gmax.x(); x++) {
            for (int y = gmin.y(); y <= gmax.y(); y++) {
                for (int z = gmin.z(); z <= gmax.z(); z++) {
                    // compute cell coordinate
                    uint16_t cellIdx = z * gridSize.x() * gridSize.y() + y * gridSize.x() + x;
                    cells[cellIdx].push_back(i);
                }
            }
        }    
    }

    // now that we constructed cells, we need to convert it to two arrays:
    // L[] contains all triangles indices for all cells in a linear order
    // C[] start index in L[] for each cell

    // let's start with C, N being the total number of cells in the grid, C has a size of N+1
    // C[i] will contain the start index of cell i, so triangles of cell i are between C[i] and C[i+1] exclusive
    uint16_t* C = new uint16_t[N + 1];
    // count num tris in each cell
    for (auto i = 0; i < N; i++)
        C[i] = cells[i].size();
    // compute end index of each cell
    for (auto i = 0; i < N; i++)
        C[i + 1] += C[i];
    // compute start index by shifting cells to the right
    for (auto i = N; i > 0; i--)
        C[i] = C[i - 1];
    C[0] = 0;

    // now let's build L
    uint16_t* L = new uint16_t[C[N]];
    uint16_t idx = 0;
    for (uint16_t i = 0; i < N; i++) {
        for (uint16_t j = 0; j < cells[i].size(); j++) {
            L[idx++] = cells[i][j];
        }
    }
    std::cerr << "check: " << idx << " == " << C[N] << std::endl;

    m.g.size = gridSize;
    m.g.cellSize = cellSize;
    m.g.C = C;
    m.g.L = L;

    // display some stats
    uint16_t numEmpty = 0;
    for (auto i = 0; i < N; i++) {
        if (C[i + 1] == C[i]) numEmpty++;
    }

    std::cerr << "grid C size = " << m.g.sizeC() << std::endl;
    std::cerr << "grid L size = " << m.g.sizeL() << std::endl;
    std::cerr << "num empty cells = " << numEmpty << std::endl;
}

bool setupScene(const char * filename, mesh& m, plane& floor) {
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
    m.numTris = 0;
    for (auto s = 0; s < shapes.size(); s++) {
        m.numTris += shapes[s].mesh.num_face_vertices.size();
    }

    bool first = true;

    // loop over shapes and copy all triangles to array
    m.tris = new vec3[m.numTris * 3];
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

                vec3 tri(vx, vy, vz);
                m.tris[vec_index++] = tri;

                // update bounds
                if (first) {
                    first = false;
                    m.bounds.min = tri;
                    m.bounds.max = tri;
                } else {
                    m.bounds.min = min(m.bounds.min, tri);
                    m.bounds.max = max(m.bounds.max, tri);
                }
            }
            index_offset += 3;
        }
    }

    // setup floor
    floor = plane(vec3(0, 0, -0.001), vec3(0, 0, 1));

    return true;
}

void setupMaterials(material** h_materials, uint16_t& numMats) {
    // create a single material for all triangles
    unsigned int rand_state = 0;

    numMats = 2;
    *h_materials = new material[numMats];
    const vec3 modelColor(RND * RND, RND * RND, RND * RND);
    const vec3 floorColor1 = hexColor(0x511845);
    const vec3 floorColor2 = hexColor(0xff5733);

    //(*h_materials)[0] = new_dielectric(1.5);
    //(*h_materials)[0] = new_lambertian(modelColor);
    //(*h_materials)[0] = new_metal(modelColor, 0.2);
    //(*h_materials)[0] = new_coat(modelColor, 1.5f);
    (*h_materials)[0] = new_tintedGlass(modelColor, 10.0f, 1.1f);

#ifdef CUBE
    (*h_materials)[1] = new_checker(floorColor1, floorColor2, 2.0f);
#else
    //(*h_materials)[1] = new_checker(floorColor1, floorColor2, 0.2f);
    //(*h_materials)[1] = new_lambertian(floorColor1);
    //(*h_materials)[1] = new_metal(floorColor1, 0.2);
    (*h_materials)[1] = new_coat(floorColor1, 1.5f);
#endif // CUBE
}

void loadHDRiEnvMap(const char *filename) {
    int x, y, n;
    float* data = stbi_loadf(filename, &x, &y, &n, 0);
    float max = 0;
    for (int i = 0; i < (x * y * n); i++) {
        max = fmaxf(max, data[i]);
    }

    std::cerr << "hdri(x = " << x << ", y = " << y << ", n = " << n << "). max = " << max << std::endl;
    initHDRi(data, x, y, n);
    stbi_image_free(data);
}

int main() {
    bool perf = false;
    bool fast = false;
    int nx = (!perf && !fast) ? 1200 : 600;
    int ny = (!perf && !fast) ? 800 : 400;
    int ns = !perf ? (fast ? 40 : 4096) : 1;
    int tx = 8;
    int ty = 8;

    std::cerr.imbue(std::locale(""));
    std::cerr << "Rendering a " << nx << "x" << ny << " image with " << ns << " samples per pixel ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    // init
    vec3 *fb;
    {
        plane floor;
        mesh m;
        material* materials;
        uint16_t numMats;
#ifdef CUBE
        if (!setupScene("D:\\models\\lowpoly\\cube.obj", m, floor)) return -1;
#else
        if (!setupScene("D:\\models\\lowpoly\\panter.obj", m, floor)) return -1;
#endif
        std::cerr << " there are " << m.numTris << " triangles" << std::endl;
        std::cerr << " bbox.min " << m.bounds.min << "\n bbox.max " << m.bounds.max << std::endl;

        buildGrid(m, 20);
        setupMaterials(&materials, numMats);

        camera cam = setup_camera(nx, ny);

        // setup floor
        initRenderer(m, materials, numMats, floor, cam, &fb, nx, ny);
        delete[] m.tris;
        delete[] m.g.C;
        delete[] m.g.L;
        delete[] materials;

        //loadHDRiEnvMap("lebombo_1k.hdr");
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
