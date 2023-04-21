#pragma once
#include "Vec3.cuh"
#include "Material.cuh"

struct Hit
{
    float t; // first intersection
             // no intersection = infinity

    Vec3 p; // point of intersection
    void *hittable;
    Vec3 normal;
    Material material;
};