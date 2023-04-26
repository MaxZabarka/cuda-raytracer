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
    Point target = hit->p + random_in_hemisphere(hit->normal, &local_rand_state);
    ray->origin = hit->p;
    ray->direction = target - hit->p;
    return true;
}

// *attenuation = color->get_color(hit->texcoord) * *attenuation;
// Point target = hit->p + hit->normal;
// ray->origin = hit->p;
// ray->direction = target - hit->p;

// float roughness = 0.7f;
// float shininess = 2 / pow(roughness, 4) - 2;

// FloatColor diffuseReflection = FloatColor(1, 1, 1) * max(0.0f, hit->normal.dot(ray->direction));

// Vec3 halfVector = (Direction(0.5f, -0.5f, 0.5f) + ray->direction).normalize();
// FloatColor specular = FloatColor(1, 1, 1) * pow(max(0.0f, hit->normal.dot(halfVector)), shininess);

// *attenuation = (specular+diffuseReflection) * *attenuation;

// return true;

// OLD COde

// *attenuation = color->get_color(hit->texcoord) * *attenuation;

// Vec3 tangent;
// Vec3 bitangent;

// if (abs(hit->normal.x) > abs(hit->normal.y))
// {
//     tangent = hit->normal.cross(Vec3(1, 0, 0)).normalize();
// }
// else
// {
//     tangent = hit->normal.cross(Vec3(0, 1, 0)).normalize();
// }
// bitangent = hit->normal.cross(tangent).normalize();

// FloatColor normal_map = normal->get_color(hit->texcoord);

// Vec3 pertrubed_normal = (tangent * normal_map.x) + (bitangent * normal_map.y) + (hit->normal * normal_map.z);

// hit->normal = pertrubed_normal.normalize();

// // Direction reflected = reflect(hit->normal, ray->direction);
// // Point target = hit->p + random_in_hemisphere(hit->normal, &local_rand_state);
// // if (normal)
// // {
// //     hit->normal = hit->normal + normal->get_color(hit->texcoord);
// // }
// float glossy = 0;

// Point target = hit->p + (hit->normal * glossy) + (random_in_hemisphere(hit->normal, &local_rand_state) * (1 - glossy));

// // ray->direction = reflected;
// ray->origin = hit->p;
// ray->direction = target - hit->p;

// return true;

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