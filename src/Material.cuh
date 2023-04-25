#pragma once
#include "Vec3.cuh"
#include "Ray.cuh"
#include <curand_kernel.h>
#include "ColorTexture.cuh"

struct Hit;
class Material
{
private:
public:
    __device__ __host__ Material(Texture* color);
    __device__ __host__ Material();
    __device__ __host__ ~Material();
    __device__ bool scatter(Ray *ray, Hit *hit, FloatColor *attenuation, curandState &local_rand_state);
    Texture* color;
};
