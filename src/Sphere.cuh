#pragma once
#include "Ray.cuh"
#include "Hit.cuh"
#include "Hittable.cuh"
#include "Vec3.cuh"

class Sphere: public Hittable
{
private:

public:
   __device__ __host__ Sphere(Point position = Point{}, float radius = 0.75, Material material = Material{FloatColor{0.5f, 0.5f, 0.5f}});
    __device__ __host__ ~Sphere();
    __device__ __host__ virtual Hit hit(const Ray &ray) override;
    // __device__ __host__ virtual HittableType get_type() override;
    Point position;
    Material material;
    float radius;
};
