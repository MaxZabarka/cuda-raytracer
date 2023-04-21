#pragma once
#include "Sphere.cuh"
#include "Camera.cuh"
#include "HittableList.cuh"

struct Scene {
    HittableList hittable_list;
    Camera camera;
};