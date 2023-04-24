#include "Material.cuh"
#include "Hit.cuh"
#include "diffuse_formulations.cuh"
#include <curand_kernel.h>

__device__ __host__ Material::Material(FloatColor color) : color{color}

{
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
    *attenuation = hit->material.color * *attenuation;

    ray->origin = hit->p;

    Direction reflected = reflect(hit->normal, ray->direction);
    ray->direction = reflected;
    return (reflected.dot(hit->normal) > 0); // Highlight artifact
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