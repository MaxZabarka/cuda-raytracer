#include "BoundingBox.cuh"
#include <stdio.h>

__device__ __host__ BoundingBox::BoundingBox(HittableList hittable_list, Point min, Point max) : hittable_list(hittable_list), min(min), max(max)

{
}

__device__ __host__ BoundingBox::~BoundingBox()
{
}

__device__ __host__ Hit BoundingBox::hit(const Ray &ray)
{


    float t_min = 0;
    float t_max = INFINITY;

    // For each dimension (x, y, z)
    for (int i = 0; i < 3; ++i)
    {
        float inverse_direction, t1, t2;
        float origin_val, min_bounds_val, max_bounds_val;

        switch (i)
        {
        case 0:
            inverse_direction = 1.0 / ray.direction.x;
            origin_val = ray.origin.x;
            min_bounds_val = min.x;
            max_bounds_val = max.x;
            break;
        case 1:
            inverse_direction = 1.0 / ray.direction.y;
            origin_val = ray.origin.y;
            min_bounds_val = min.y;
            max_bounds_val = max.y;
            break;
        case 2:
            inverse_direction = 1.0 / ray.direction.z;
            origin_val = ray.origin.z;
            min_bounds_val = min.z;
            max_bounds_val = max.z;
            break;
        }

        t1 = (min_bounds_val - origin_val) * inverse_direction;
        t2 = (max_bounds_val - origin_val) * inverse_direction;

        if (inverse_direction < 0.0)
        {
            float temp = t1;
            t1 = t2;
            t2 = temp;
        }

        t_min = t_min > t1 ? t_min : t1;
        t_max = t_max < t2 ? t_max : t2;

        if (t_min > t_max)
        {
            return Hit{INFINITY, Vec3(0, 0, 0), nullptr};
        }
    }
    // return Hit{1, Vec3(1, 1, 1), this, Vec3(0, 0, 0), Material{FloatColor{1, 0, 0}}};

    return hittable_list.hit(ray);
}
