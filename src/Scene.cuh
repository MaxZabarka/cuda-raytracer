#pragma once
#include "Sphere.cuh"
#include "Camera.cuh"

struct Scene {
    Hittable** hittables;
    int sphere_count;
    Camera camera;
};