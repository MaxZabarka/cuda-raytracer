#pragma once
#include "Sphere.cuh"
#include "Camera.cuh"

struct Scene {
    Sphere* spheres;
    int sphere_count;
    Camera camera;
};