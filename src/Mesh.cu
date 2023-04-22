#include "Hit.cuh"
#include "Mesh.cuh"
#include <stdio.h>
#include "cuda-wrapper/cuda.cuh"

__host__ Mesh::Mesh(Geometry &geometry, Material material) : geometry{geometry}, material{material}
{
    triangles = (Triangle *)cuda::mallocManaged(sizeof(Triangle) * geometry.num_triangles);
    for (size_t i = 0; i < geometry.num_triangles; i++)
    {
        TriangleData data = geometry.triangles[i];
        triangles[i] = Triangle(data, Material{FloatColor{0.5f, 0.5f, 0.5f}});
    }
}

__device__ __host__ Mesh::~Mesh()
{
}

__device__ __host__ Hit Mesh::hit(const Ray &ray)
{
    Hit closest_hit = Hit{INFINITY, Vec3(0, 0, 0), nullptr};

    for (size_t i = 0; i < geometry.num_triangles; i++)
    {
        Triangle triangle = triangles[i];
        Hit hit = triangle.hit(ray);
        // Fix shadow acne
        if (hit.t < closest_hit.t && hit.t > 0.001)
        {
            closest_hit = hit;
        }
    }

    return closest_hit;
}
