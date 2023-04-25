#include "HittableList.cuh"
#include <stdio.h>
__device__ __host__ HittableList::HittableList(Hittable **hittables, int size) : hittables{hittables}, size{size}
{
}

__device__ __host__ HittableList::~HittableList()
{
}

__device__ __host__ Hit HittableList::hit(const Ray &ray)
{
    Hit closest_hit = Hit{INFINITY, Vec3{}, nullptr, Vec3{}, Material(nullptr)};

    for (int i = 0; i < size; i++)
    {
        Hittable *hittable = *(hittables + i);
        Hit hit = hittable->hit(ray);
        // Fix shadow acne
        if (hit.t < closest_hit.t && hit.t > 0.001)
        {
            closest_hit = hit;
        }
    }


    return closest_hit;

    // Hit hit{INFINITY, Vec3{}, this};

    // // CO = O - C
    // Vec3 sphere_direction = ray.origin - position;

    // // < direction, direction >
    // float a = ray.direction.dot(ray.direction);

    // // 2 <CO, D>
    // float b = 2 * ray.direction.dot(sphere_direction);

    // // < CO, CO > - r^2
    // float c = sphere_direction.dot(sphere_direction) - pow(radius, 2);

    // // Solve the quadratic
    // float discriminant = b * b - 4 * a * c;

    // if (discriminant >= 0)
    // {
    //     hit.t = (-b - sqrt(discriminant)) / (2 * a);
    //     hit.p = ray.origin + (ray.direction * hit.t);
    // }
    // hit.material = material;
    // hit.normal = (hit.p - position).normalize();
    // return hit;
}
