#pragma once

// Required to include vec3.h
#include "helper_structs.h"

struct kernel_scene {
    mesh & m;
    plane floor;
    
    material* materials;
    int numMaterials;

    stexture* textures;
    int numTextures;
};

extern "C" void initRenderer(const kernel_scene sc, const camera cam, vec3 * *fb, int nx, int ny, int maxDepth, int numPrimitivesPerLeaf);
extern "C" void runRenderer(int ns, int tx, int ty);
extern "C" void cleanupRenderer();
