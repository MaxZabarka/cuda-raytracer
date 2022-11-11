#pragma once
#include "Vec3.cuh"

struct Hit
{
    float t; // first intersection
             // no intersection = infinity

    Vec3 p; // point of intersection
    void *hittable;
};