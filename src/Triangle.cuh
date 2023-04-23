#pragma once
#include "Ray.cuh"
#include "Hit.cuh"
#include "Hittable.cuh"
#include "Vec3.cuh"

class Triangle : public Hittable
{
private:
public:
    __device__ __host__ Triangle(Point a, Point b, Point c, Material material = Material{FloatColor{0.5f, 0.5f, 0.5f}});
    __device__ __host__ ~Triangle();
    __device__ __host__ virtual Hit hit(const Ray &ray) override;
    Point a, b, c;
    Material material;
};

