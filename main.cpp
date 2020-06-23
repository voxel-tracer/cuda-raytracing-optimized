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
    vec3 lookat(0, m.bounds.max.y() / 2, 0);
    float dist_to_focus = (lookfrom - lookat).length();
    float aperture = 0.1;
    return camera(lookfrom,
        lookat,
        vec3(0, 1, 0),
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


void load_from_binary(const char* input, mesh& m) {
    std::fstream in(input, std::ios::in | std::ios::binary);
    in.read((char*)&m.numTris, sizeof(int));
    m.tris = new triangle[m.numTris];
    in.read((char*)m.tris, sizeof(triangle) * m.numTris);

    in.read((char*)&m.numBvhNodes, sizeof(int));
    m.bvh = new bvh_node[m.numBvhNodes];
    in.read((char*)m.bvh, sizeof(bvh_node) * m.numBvhNodes);

    in.read((char*)&m.bounds.min, sizeof(vec3));
    in.read((char*)&m.bounds.max, sizeof(vec3));
}

bool deprecated_setupScene(const char * filename, mesh& m, float scale, const mat3x3& mat) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    
    std::string warn;
    std::string err;

    // note that tinyobj will automatically triangulate non-triangle polygons but it doesn't
    // always do a good job orienting them properly
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
    if (!attrib.normals.empty())
        std::cerr << " normals size " << attrib.normals.size() << std::endl;

    // first count how many triangles we have
    m.numTris = 0;
    for (auto s = 0; s < shapes.size(); s++) {
        m.numTris += shapes[s].mesh.num_face_vertices.size();
    }

    m.bounds.min = vec3(INFINITY, INFINITY, INFINITY);
    m.bounds.max = vec3(-INFINITY, -INFINITY, -INFINITY);

    // loop over shapes and copy all triangles to array
    m.tris = new triangle[m.numTris];
    uint32_t triIdx = 0;
    for (auto s = 0; s < shapes.size(); s++) {
        // loop over faces
        size_t index_offset = 0;
        for (auto f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++, triIdx++) {
            int fv = shapes[s].mesh.num_face_vertices[f];
            if (fv != 3)
                std::cerr << "face " << f << " of shape " << s << " has " << fv << " vertices" << std::endl;
            
            // loop over vertices in the face
            for (auto i = 0; i < 3; i++) {
                // access to vertex
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + i];
                tinyobj::real_t vx = attrib.vertices[3 * idx.vertex_index + 0];
                tinyobj::real_t vy = attrib.vertices[3 * idx.vertex_index + 1];
                tinyobj::real_t vz = attrib.vertices[3 * idx.vertex_index + 2];

                vec3 o(vx, vy, vz);
                // transform vectors as we load them
                vec3 v;
                v[0] = dot(mat.rows[0], o);
                v[1] = dot(mat.rows[1], o);
                v[2] = dot(mat.rows[2], o);
                m.tris[triIdx].v[i] = v;

                // update bounds
                m.bounds.min = min(m.bounds.min, v);
                m.bounds.max = max(m.bounds.max, v);
            }
            index_offset += 3;
        }
    }

    // center scene around origin
    const vec3 mn = m.bounds.min;
    const vec3 mx = m.bounds.max;
    vec3 ctr = (mx + mn) / 2; // this is the model center
    ctr[1] = mn[1]; // make sure we can put the floor at y = 0

    std::cerr << " original model bounds:" << std::endl;
    std::cerr << "  min: " << mn << std::endl;
    std::cerr << "  max: " << mx << std::endl;

    // find max size across all axes
    const float maxSize = max(mx - mn);
    // we want to normalize the model so that its maxSize is scale and centered around ctr
    // for each vertex v:
    //  v = v- ctr // ctr is new origin
    //  v = v / maxSize // scale model to fit in a bbox with maxSize 1
    //  v = v * scale // scale model so that maxSize = scale
    // => v = (v - ctr) * scale/maxSize
    for (int i = 0; i < m.numTris; i++) {
        for (int j = 0; j < 3; j++)
            m.tris[i].v[j] = (m.tris[i].v[j] - ctr) * scale / maxSize;
        m.tris[i].update();
    }

    // update model bounds
    m.bounds.min = vec3(INFINITY, INFINITY, INFINITY);
    m.bounds.max = vec3(-INFINITY, -INFINITY, -INFINITY);
    for (int i = 0; i < m.numTris; i++) {
        for (int j = 0; j < 3; j++) {
            m.bounds.min = min(m.bounds.min, m.tris[i].v[j]);
            m.bounds.max = max(m.bounds.max, m.tris[i].v[j]);
        }
    }

    std::cerr << " updated model bounds:" << std::endl;
    std::cerr << "  min: " << m.bounds.min << std::endl;
    std::cerr << "  max: " << m.bounds.max << std::endl;

    // setup floor

    return true;
}

int main() {
    bool perf = false;
    bool fast = true;
    int nx = (!perf && !fast) ? 512 : 512;
    int ny = (!perf && !fast) ? 512 : 512;
    int ns = !perf ? (fast ? 1 : 512) : 4;
    int tx = 8;
    int ty = 8;

    std::cerr.imbue(std::locale(""));
    std::cerr << "Rendering a " << nx << "x" << ny << " image with " << ns << " samples per pixel ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    scene ball = { "D:\\models\\/lowpoly/icosphere.obj.bin" , yUp, 50, vec3(1,1,1) * 120 };
    scene teapot = { "D:\\models\\obj\\teapot.obj.bin" , yUp, 100, vec3(1,1,1) * 120 };
    scene panter = { "D:\\models\\lowpoly\\panter.obj.bin" , zUp, 100, vec3(1,1,1) * 120 };
    scene bunny = { "D:\\models\\obj\\bunny.obj.bin", yUp, 50, vec3(1,1,1) * 120 };
    scene dragon = { "D:\\models\\obj\\dragon.obj.bin" , yUp, 100, vec3(-1,1,-1) * 200 };
    scene catfolk = { "D:\\models\\lowpoly\\Character Pack 3\\files\\CatfolkRogue.OBJ.bin" , yUp, 100, vec3(1,1,1) * 200 };

    scene sc = dragon;
    // init
    vec3 *fb;
    {
        plane floor = plane(vec3(0, -0.01, 0), vec3(0, 1, 0));
        mesh m;

        load_from_binary(sc.filename, m);
        std::cerr << " there are " << m.numTris << " triangles" << ", and " << m.numBvhNodes << " bvh nodes" << std::endl;
        std::cerr << " bbox.min " << m.bounds.min << "\n bbox.max " << m.bounds.max << std::endl;

        camera cam = setup_camera(nx, ny, m, sc.camPos);

        // setup floor
        initRenderer(m, floor, cam, &fb, nx, ny, 10);
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
