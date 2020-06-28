#pragma once

// Required to include vec3.h
#include "helper_structs.h"

extern "C" void initRenderer(const mesh &m, plane floor, const camera cam, const material * materials, int numMats, vec3 * *fb, int nx, int ny, int maxDepth, int numPrimitivesPerLeaf);
extern "C" void runRenderer(int ns, int tx, int ty);
extern "C" void cleanupRenderer();
