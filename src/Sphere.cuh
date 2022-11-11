#pragma once
#include "Ray.cuh"
#include "Hit.cuh"
#include "Hittable.cuh"
#include "Vec3.cuh"

class Sphere
{
private:

public:
   __device__ __host__ Sphere(Point position = Point{}, float radius = 0.75, Material material = Material{FloatColor{0.5f, 0.5f, 0.5f}});
    __device__ __host__ ~Sphere();
    __device__ __host__ Hit hit( Ray &ray);
    __device__ __host__ Material get_material();
    Point position;
    Material material;
    float radius;
};
