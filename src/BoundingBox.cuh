#pragma once
#include "Ray.cuh"
#include "Hit.cuh"
#include "Hittable.cuh"
#include "Vec3.cuh"
#include "HittableList.cuh"

class BoundingBox: public Hittable
{
private:

public:
   __device__ __host__ BoundingBox(HittableList hittable_list, Point min, Point max);
    __device__ __host__ ~BoundingBox();
    __device__ __host__ virtual Hit hit(const Ray &ray) override;

    HittableList hittable_list;
    Point min;
    Point max;
};