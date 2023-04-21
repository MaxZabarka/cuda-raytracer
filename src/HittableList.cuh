#pragma once
#include "Ray.cuh"
#include "Hit.cuh"
#include "Hittable.cuh"
#include "Vec3.cuh"

class HittableList: public Hittable
{
private:

public:
   __device__ __host__ HittableList(Hittable **hittables, int size);
    __device__ __host__ ~HittableList();
    __device__ __host__ virtual Hit hit(const Ray &ray) override;

    Hittable **hittables;
    int size;
};
