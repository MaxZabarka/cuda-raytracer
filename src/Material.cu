#include "Material.cuh"

__device__ __host__ Material::Material(FloatColor color) : color{color}

{
}

__device__ __host__ Material::~Material()
{
}

__device__ __host__ Ray Material::scatter(Ray &r_in)
{
    return Ray{r_in.origin, r_in.direction};
}
