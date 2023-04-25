#pragma once
#include "Ray.cuh"
#include "Hit.cuh"
#include "Hittable.cuh"
#include "Vec3.cuh"

struct Vertex
{
    Point position;
    Direction normal;
    Vec3 texcoord;

};
struct TriangleData
{
    Vertex a;
    Vertex b;
    Vertex c;
};

class Triangle : public Hittable
{
private:
public:
    __device__ __host__ Triangle(TriangleData triangle_data, Material material);
    __device__ __host__ ~Triangle();
    __device__ __host__ virtual Hit hit(const Ray &ray) override;
    TriangleData triangle_data;
    Material material;
};
