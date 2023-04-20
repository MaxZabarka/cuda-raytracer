#include <curand_kernel.h>
#include "Vec3.cuh"

__device__ Point random_in_unit_sphere(curandState *local_rand_state)
{
    while (true)
    {
        Vec3 point = Vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), curand_uniform(local_rand_state)) * 2.0f - Vec3(1, 1, 1);
        if (point.magnitude_squared() >= 1)
            continue;
        return point;
    }
}

__device__ Direction random_unit_vector(curandState *local_rand_state)
{
    return random_in_unit_sphere(local_rand_state).normalize();
}
__device__ Point random_in_hemisphere(const Direction &normal, curandState *local_rand_state)
{
    Point in_unit_sphere = random_in_unit_sphere(local_rand_state);
    if (in_unit_sphere.dot(normal) > 0.0)
    {
        return in_unit_sphere;
    }
    else
    {
        return -in_unit_sphere;
    }
}