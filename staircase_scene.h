#pragma once

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
