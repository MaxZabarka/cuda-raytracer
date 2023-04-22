#pragma once
#include "Ray.cuh"
#include "Hit.cuh"
#include "Hittable.cuh"
#include "Vec3.cuh"
#include "Geometry.cuh"

class Mesh: public Hittable
{
private:

public:
   __device__ __host__ Mesh(Geometry& geometry, Material material = Material{FloatColor{0.5f, 0.5f, 0.5f}});
    __device__ __host__ ~Mesh();
    __device__ __host__ virtual Hit hit(const Ray &ray) override;

    Geometry geometry;
    Material material;
    Triangle *triangles;
};
