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
        triangles[i] = Triangle(data, material);
        // cuda::fixVirtualPointers << 1, 1 >> (Triangle *)(&triangles[i]);
        
    }
}

__device__ __host__ Mesh::~Mesh()
{
}

__device__ __host__ Hit Mesh::hit(const Ray &ray)
{

    double t_min = -INFINITY;
    double t_max = INFINITY;

    // For each dimension (x, y, z)
    for (int i = 0; i < 3; ++i)
    {
        double inverse_direction, t1, t2;
        double origin_val, min_bounds_val, max_bounds_val;

        switch (i)
        {
        case 0:
            inverse_direction = 1.0 / ray.direction.x;
            origin_val = ray.origin.x;
            min_bounds_val = geometry.min_bounds.x;
            max_bounds_val = geometry.max_bounds.x;
            break;
        case 1:
            inverse_direction = 1.0 / ray.direction.y;
            origin_val = ray.origin.y;
            min_bounds_val = geometry.min_bounds.y;
            max_bounds_val = geometry.max_bounds.y;
            break;
        case 2:
            inverse_direction = 1.0 / ray.direction.z;
            origin_val = ray.origin.z;
            min_bounds_val = geometry.min_bounds.z;
            max_bounds_val = geometry.max_bounds.z;
            break;
        }

        t1 = (min_bounds_val - origin_val) * inverse_direction;
        t2 = (max_bounds_val - origin_val) * inverse_direction;

        if (inverse_direction < 0.0)
        {
            double temp = t1;
            t1 = t2;
            t2 = temp;
        }

        t_min = t_min > t1 ? t_min : t1;
        t_max = t_max < t2 ? t_max : t2;

        if (t_min > t_max || t_max < 0.0)
        {
            return Hit{INFINITY, Vec3(0, 0, 0), nullptr};
        }
    }

    // return Hit{1.0f, Point{0, 0, 0}, this, Point{0, 0, 0}, material};

    Hit closest_hit = Hit{INFINITY, Vec3(0, 0, 0), nullptr};

    for (size_t i = 0; i < geometry.num_triangles; i++)
    {
        Triangle triangle = triangles[i];
        Hit hit = triangle.hit(ray);
        // printf("%f, %f, %f", hit.material.color.x, hit.material.color.y, hit.material.color.z);

        // Fix shadow acne
        if (hit.t < closest_hit.t && hit.t > 0.001)
        {
            closest_hit = hit;
        }
    }

    return closest_hit;
}
