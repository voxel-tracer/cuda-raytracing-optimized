#pragma once

// Required to include vec3.h
#include "helper_structs.h"

extern "C" void initRenderer(const kernel_scene sc, const camera cam, vec3 * *fb, int nx, int ny, int maxDepth);
extern "C" void runRenderer(int ns, int tx, int ty);
extern "C" void cleanupRenderer();
