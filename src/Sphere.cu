#include "Sphere.cuh"
#include "Hit.cuh"
#include "Infinity.cuh"
#include <cmath>
#include <iostream>
Sphere::Sphere(Point position, float radius, Material material) : position{position}, radius{radius}, material{material}
{
}

Sphere::~Sphere()
{
}

// int Sphere::test() {
//     return 0;
// }

__device__ __host__ Hit Sphere::hit(const Ray &ray)
{
    Hit hit{INFINITY, this};

    // CO = O - C
    Vec3 sphere_direction = ray.origin - position;

    // < direction, direction >
    float a = ray.direction.dot(ray.direction);

    // 2 <CO, D>
    float b = 2 * ray.direction.dot(sphere_direction);

    // < CO, CO > - r^2
    float c = sphere_direction.dot(sphere_direction) - pow(radius, 2);

    // Solve the quadratic
    float discriminant = b * b - 4 * a * c;

    if (discriminant >= 0)
    {
        hit.t = (-b - sqrt(discriminant)) / (2 * a);
    }

    return hit;
}

__host__ __device__ Material Sphere::get_material()
{
    return material;
}