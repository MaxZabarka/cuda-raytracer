#pragma once
#include "Ray.cuh"
#include "Material.cuh"
#include "Hit.cuh"

class Hit;

class Hittable
{
protected:
    Hittable() {} ;

public:
    __device__ __host__ virtual Hit hit(const Ray &ray) = 0;
    __device__ __host__ virtual Material get_material() = 0;
};
