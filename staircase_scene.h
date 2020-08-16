#pragma once

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

void writePPM(int nx, int ny, const vec3* colors) {
    std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny - 1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            size_t pixel_index = j * nx + i;
            int ir = LinearToSRGB(colors[pixel_index].r());
            int ig = LinearToSRGB(colors[pixel_index].g());
            int ib = LinearToSRGB(colors[pixel_index].b());
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }
}

void writePPM(int nx, int ny, int ns, const vec3* colors) {
    std::cout << "P3\n" << nx << " " << ny << "\n255\n";
    for (int j = ny - 1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            vec3 color(0, 0, 0);
            for (auto s = 0; s < ns; s++) {
                color += colors[(j * nx + i) * ns + s];
            }
            color /= float(ns);
            int ir = LinearToSRGB(color.r());
            int ig = LinearToSRGB(color.g());
            int ib = LinearToSRGB(color.b());
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }
}

camera setup_camera(int nx, int ny) {
    const vec3 lookfrom = vec3(5.555139, 173.679901, 494.515045);
    vec3 lookat(5.555139, 173.679901, 493.515045);
    float dist_to_focus = (lookfrom - lookat).length();
    return camera(lookfrom,
        lookat,
        vec3(0, 1, 0),
        42.0,
        float(nx) / float(ny),
        0.0,
        dist_to_focus);
}

bool loadBVH(const char* input, mesh& m, int& numPrimitivesPerLeaf) {
    std::fstream in(input, std::ios::in | std::ios::binary);

    const char* HEADER = "BVH_00.03";
    char* header = new char[sizeof(HEADER)];
    in.read(header, sizeof(HEADER));
    if (!strcmp(HEADER, header)) {
        std::cerr << "invalid header " << header << std::endl;
        return false;
    }

    in.read((char*)&m.numTris, sizeof(int));
    m.tris = new triangle[m.numTris];
    in.read((char*)m.tris, sizeof(triangle) * m.numTris);

    in.read((char*)&m.numBvhNodes, sizeof(int));
    m.bvh = new bvh_node[m.numBvhNodes];
    in.read((char*)m.bvh, sizeof(bvh_node) * m.numBvhNodes);

    in.read((char*)&m.bounds.min, sizeof(vec3));
    in.read((char*)&m.bounds.max, sizeof(vec3));

    in.read((char*)&numPrimitivesPerLeaf, sizeof(int));

    return true;
}

bool loadTexture(const std::string filename, stexture& tex) {
    int width, height, nrChannels;
    // read image as RGB regardless of file format
    unsigned char* data = stbi_load(filename.c_str(), &width, &height, &nrChannels, 3);
    if (!data)
        return false;

    float* fdata = new float[width * height * 3];
    for (auto i = 0; i < width * height * 3; i++) {
        fdata[i] = data[i] / 255.0f;
    }
    tex = stexture(fdata, width, height);

    stbi_image_free(data);
    return true;
}

bool load_scene(scene& sc) {
    stbi_set_flip_vertically_on_load(true);

    stexture* textures = new stexture[9];
    bool success = true;
    success = success && loadTexture("D:\\vstudio\\glsl-pathtracer\\GLSL-PathTracer\\bin\\assets\\staircase\\textures\\WoodFloor.png", textures[0]);
    success = success && loadTexture("D:\\vstudio\\glsl-pathtracer\\GLSL-PathTracer\\bin\\assets\\staircase\\textures\\Wallpaper.png", textures[1]);
    success = success && loadTexture("D:\\vstudio\\glsl-pathtracer\\GLSL-PathTracer\\bin\\assets\\staircase\\textures\\Woodpanel.png", textures[2]);
    success = success && loadTexture("D:\\vstudio\\glsl-pathtracer\\GLSL-PathTracer\\bin\\assets\\staircase\\textures\\Painting1.png", textures[3]);
    success = success && loadTexture("D:\\vstudio\\glsl-pathtracer\\GLSL-PathTracer\\bin\\assets\\staircase\\textures\\Painting2.png", textures[4]);
    success = success && loadTexture("D:\\vstudio\\glsl-pathtracer\\GLSL-PathTracer\\bin\\assets\\staircase\\textures\\Painting3.png", textures[5]);
    success = success && loadTexture("D:\\vstudio\\glsl-pathtracer\\GLSL-PathTracer\\bin\\assets\\staircase\\textures\\WoodChair.png", textures[6]);
    success = success && loadTexture("D:\\vstudio\\glsl-pathtracer\\GLSL-PathTracer\\bin\\assets\\staircase\\textures\\Fabric.png", textures[7]);
    success = success && loadTexture("D:\\vstudio\\glsl-pathtracer\\GLSL-PathTracer\\bin\\assets\\staircase\\textures\\BrushedAluminium.png", textures[8]);
    if (!success) {
        std::cerr << "Failed to load textures" << std::endl;
        return false;
    }

    material* materials = new material[20];
    materials[0] = { material_type::DIFFUSE, vec3(0.01, 0.01, 0.01), 0,-1 };              // Black
    materials[1] = { material_type::METAL, vec3(0.27, 0.254, 0.15), 0.01,-1 };            // Brass
    materials[2] = { material_type::METAL, vec3(), 0, 8 };                                // BrushedAluminium
    materials[3] = { material_type::DIFFUSE, vec3(1, 1, 1), 0,-1 };                       // Candles
    materials[4] = { material_type::DIFFUSE, vec3(0.117647, 0.054902, 0.0666667), 0,-1 }; // ChairSeat
    materials[5] = { material_type::GLASS, vec3(1, 1, 1), 1.45,-1 };                      // Glass
    materials[6] = { material_type::METAL, vec3(1.0, 0.95, 0.35), 0.05,-1 };              // Gold
    materials[7] = { material_type::DIFFUSE, vec3(), 0, 7 };                              // Lampshade
    materials[8] = { material_type::DIFFUSE, vec3(0.578596, 0.578596, 0.578596), 0,-1 };  // MagnoliaPaint
    materials[9] = { material_type::DIFFUSE, vec3(), 0, 3 };                              // Painting1
    materials[10] = { material_type::DIFFUSE, vec3(), 0, 4 };                             // Painting2
    materials[11] = { material_type::DIFFUSE, vec3(), 0, 5 };                             // Painting3
    materials[12] = { material_type::METAL, vec3(1.0, 1.0, 1.0), 0.1,-1 };                // StainlessSteel
    materials[13] = { material_type::DIFFUSE, vec3(), 0, 1 };                             // wallpaper
    materials[14] = { material_type::DIFFUSE, vec3(0.578596, 0.578596, 0.578596), 0,-1 }; // whitePaint
    materials[15] = { material_type::DIFFUSE, vec3(1, 1, 1), 0,-1 };                      // WhitePlastic
    materials[16] = { material_type::DIFFUSE, vec3(), 0, 6 };                             // WoodChair
    materials[17] = { material_type::DIFFUSE, vec3(), 0, 0 };                             // woodFloor
    materials[18] = { material_type::DIFFUSE, vec3(), 0, 6 };                             // WoodLamp
    materials[19] = { material_type::DIFFUSE, vec3(), 0, 2 };                             // woodstairs

    sc = { "D:\\models\\obj\\staircase.bvh" , yUp, 1, vec3(1,1,1), materials, 20, textures, 9 };
    return true;
}

bool setup_kernel_scene(const scene sc, kernel_scene& ksc) {
    //plane floor = plane(vec3(0, -0.01, 0), vec3(0, 1, 0));
    mesh* m = new mesh();
    int numPrimitivesPerLeaf = 0;

    if (!loadBVH(sc.filename, *m, numPrimitivesPerLeaf)) {
        std::cerr << "Failed to load bvh file" << std::endl;
        std::cerr.flush();
        return false;
    }

    std::cerr << " there are " << m->numTris << " triangles" << ", and " << m->numBvhNodes << " bvh nodes" << std::endl;
    std::cerr << " bbox.min " << m->bounds.min << "\n bbox.max " << m->bounds.max << std::endl;

    ksc = { m, plane(), sc.materials, sc.numMaterials, sc.textures, sc.numTextures, numPrimitivesPerLeaf };

    return true;
}
