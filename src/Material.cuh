#pragma once
#include "Vec3.cuh"
#include "Ray.cuh"

class Material
{
private:
public:
    __device__ __host__ Material(FloatColor color = FloatColor(0.5f, 0.5f, 0.5f));
    __device__ __host__ ~Material();
    __device__ __host__ Ray scatter(Ray &r_in);
    FloatColor color = FloatColor(0.5f, 0.5f, 0.5f);
};
