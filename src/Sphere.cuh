#pragma once
#include "Ray.cuh"
#include "Hit.cuh"
#include "Hittable.cuh"
#include "Vec3.cuh"

class Sphere
{
private:

public:
    Sphere(Point position = Point{}, float radius = 0.75, Material material = Material{Color{128, 128, 128}});
    ~Sphere();
    __device__ __host__ Hit hit(const Ray &ray);
    __device__ __host__ Material get_material();
    Point position;
    Material material;
    float radius;
};
