#include "Sphere.cuh"
#include "Hit.cuh"
#include "Infinity.cuh"
#include <cmath>
#include <iostream>
__device__ __host__ Sphere::Sphere(Point position, float radius, Material material) : position{position}, radius{radius}, material{material}
{
}

__device__ __host__ Sphere::~Sphere()
{
}


__device__ __host__ Hit Sphere::hit(const Ray &ray)
{
    Hit hit{INFINITY, Vec3{}, this, Vec3{}, Material{nullptr}};

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
        hit.p = ray.origin + (ray.direction * hit.t);
    }
    hit.material = material;
    hit.normal = (hit.p - position).normalize();
    return hit;
}

// __device__ __host__ HittableType Sphere::get_type()
// {
//     return HittableType::SPHERE;
// }

