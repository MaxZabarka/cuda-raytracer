#pragma once
#include "Vec3.cuh"
#include "Ray.cuh"
#include <curand_kernel.h>

struct Hit;
class Material
{
private:
public:
    __device__ __host__ Material(FloatColor color = FloatColor(0.5f, 0.5f, 0.5f));
    __device__ __host__ ~Material();
    __device__ bool scatter(Ray *ray, Hit *hit, FloatColor *attenuation, curandState &local_rand_state);
    FloatColor color = FloatColor(0.5f, 0.5f, 0.5f);
};
