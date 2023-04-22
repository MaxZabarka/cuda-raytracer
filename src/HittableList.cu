#include "HittableList.cuh"
#include <stdio.h>

__device__ __host__ HittableList::HittableList(Hittable **hittables, int size) : hittables{hittables}, size{size}
{
}

__device__ __host__ HittableList::~HittableList()
{
}

__device__ __host__ Hit HittableList::hit(const Ray &ray)
{

    Hit closest_hit = Hit{INFINITY, Vec3(0, 0, 0), nullptr};

    for (int i = 0; i < size; i++)
    {
        Hittable *hittable = *(hittables + i);
        Hit hit = hittable->hit(ray);
        // Fix shadow acne
        if (hit.t < closest_hit.t && hit.t > 0.001)
        {
            closest_hit = hit;
        }
    }

    return closest_hit;
}
