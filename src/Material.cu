#include "Material.cuh"
#include "Hit.cuh"
#include "diffuse_formulations.cuh"
#include <curand_kernel.h>
#include "cuda-wrapper/cuda.cuh"

__device__ __host__ Material::Material(Texture *color) : color{color}

{
    if (color == nullptr)
    {

        this->color = (ColorTexture *)cuda::mallocManaged(sizeof(ColorTexture));
        ((ColorTexture *)this->color)->color = FloatColor(0.5f, 0.5f, 0.5f);
        cuda::fixVirtualPointers<<<1, 1>>>((ColorTexture *)(color));
    }
}

__device__ __host__ Material::Material()
{

    this->color = (ColorTexture *)cuda::mallocManaged(sizeof(ColorTexture));
    ((ColorTexture *)this->color)->color = FloatColor(0.5f, 0.5f, 0.5f);
    cuda::fixVirtualPointers<<<1, 1>>>((ColorTexture *)(color));
}

__device__ __host__ Material::~Material()
{
}

// __device__ Direction reflect(Direction normal, Direction incident)
// {
//     return incident - (normal * (incident.dot(normal) * 2));
// }

__device__ Direction reflect(Direction normal, Direction incident)
{
    return incident - (normal * (incident.dot(normal) * 2));
}

__device__ bool Material::scatter(Ray *ray, Hit *hit, FloatColor *attenuation, curandState &local_rand_state)
{

    *attenuation = color->get_color(hit->texcoord) * *attenuation;

    // Direction reflected = reflect(hit->normal, ray->direction);
    // Point target = hit->p + random_in_hemisphere(hit->normal, &local_rand_state);
    float glossy = 0;
    Point target = hit->p + (hit->normal * glossy) + (random_in_hemisphere(hit->normal, &local_rand_state) * (1 - glossy));

    // ray->direction = reflected;
    ray->origin = hit->p;
    ray->direction = target - hit->p;
    return true;
    // return (reflected.dot(hit->normal) > 0); // Highlight artifact
}

// float fuzz = 0.5;
// *attenuation = hit->material.color * *attenuation;
// // Point target = hit->p + random_in_hemisphere(hit->normal, &local_rand_state);

// ray->origin = hit->p;
// // ray->direction = target - hit->p;
// Direction reflected = reflect(hit->normal, ray->direction);
// // ray->direction = reflected + random_in_unit_sphere(&local_rand_state) * fuzz;

// ray->direction = reflected;
// return true;
// // return (scattered.dot(hit->normal) > 0.01);