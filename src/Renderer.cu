#include "Renderer.cuh"
#include <iostream>
#include "cuda-wrapper/cuda.cuh"

#include "Window.cuh"
#include "Color.cuh"
#include "Camera.cuh"
#include "Ray.cuh"
#include "Hittable.cuh"
#include "Scene.cuh"
#include "Vec3.cuh"
#include "diffuse_formulations.cuh"
#include "map.cuh"

#include <curand_kernel.h>

#define COLOR_NORMALS false

Renderer::Renderer()
{
}

Renderer::~Renderer()
{
}

__device__ FloatColor trace_ray(Ray &ray, Camera &camera, Scene *scene, curandState local_rand_state)
{
    Ray current_ray = Ray{ray.direction, ray.origin};
    FloatColor current_attenuation = FloatColor{1.0f, 1.0f, 1.0f};

    for (int _ = 0; _ < 50; _++)
    {
        Hit closest_hit = Hit{INFINITY, Vec3(0, 0, 0), nullptr};

        for (int i = 0; i < scene->sphere_count; i++)
        {
            Hit hit;
            Sphere *sphere = (scene->spheres + i);
            hit = sphere->hit(current_ray);
                                                            // Fix shadow acne
            if (hit.t < closest_hit.t && hit.t < camera.far && hit.t > 0.001)
            {
                closest_hit = hit;
            }
        }

        if (closest_hit.hittable)
        {
            Sphere sphere = *((Sphere *)closest_hit.hittable);
            Direction normal = (closest_hit.p - sphere.position).normalize();
            normal.z = -normal.z;

            if (COLOR_NORMALS)
            {
                return FloatColor{normal.x + 1, normal.y + 1, normal.z + 1} * 0.5;
            }
            current_attenuation = sphere.get_material().color * current_attenuation;

            // Point target = closest_hit.p + normal + random_in_unit_sphere(&local_rand_state);
            // Point target = closest_hit.p + normal + random_unit_vector(&local_rand_state);
            Point target = closest_hit.p + random_in_hemisphere(normal, &local_rand_state);

            current_ray.origin = closest_hit.p;
            current_ray.direction = target - closest_hit.p;
        }
        else
        {
            Direction ray_unit_direction = current_ray.direction.normalize();
            float y = 0.5 * (ray_unit_direction.y + 1.0);
            FloatColor background_color = FloatColor(1.0, 1.0, 1.0) * (1.0 - y) + FloatColor(0.5, 0.7, 1.0) * y;
            return background_color * current_attenuation;
        }
    }

    // max bounces reached
    return FloatColor{0, 0, 0};
}


// SETUP RENDERING

__global__ void gpuRender(uint32_t *sampled_pixels, int pixel_count, size_t image_width, size_t image_height, Scene *scene, curandState *rand_state, int seed)
{
    int x = (threadIdx.x + blockIdx.x * blockDim.x);
    int y = (threadIdx.y + blockIdx.y * blockDim.y);

    if (x >= image_width || y >= image_height)
        return;

    int index = ((image_height - y - 1) * image_width + (x)) * 3;

    curand_init(seed, index, 0, &rand_state[index / 3]);
    curandState local_rand_state = rand_state[index / 3];
    float random_x = (curand_uniform(&local_rand_state));
    float random_y = (curand_uniform(&local_rand_state));

    Camera camera = scene->camera;
    Direction ray_direction = scene->camera.to_viewport(x + random_x, y + random_y) - camera.position;
    Ray ray = {.direction = ray_direction, .origin = camera.position};

    Color color = trace_ray(ray, camera, scene, local_rand_state).square_root().to_int_color();

    sampled_pixels[index] += color.r;
    sampled_pixels[index + 1] += color.g;
    sampled_pixels[index + 2] += color.b;
}

void Renderer::render(uint32_t *h_sampled_pixels, Scene *scene, Window &window, int seed)
{

    size_t pixels_size = window.image_width * window.image_height * 3 * sizeof(uint32_t);
    size_t pixel_count = window.image_height * window.image_width;

    curandState *d_rand_state = (curandState *)cuda::malloc(pixel_count * sizeof(curandState));

    // Set up device memory
    uint32_t *d_pixels = (uint32_t *)cuda::malloc(pixels_size);
    cuda::copyToDevice(d_pixels, h_sampled_pixels, pixels_size);

    int tx = 16;
    int ty = 16;

    dim3 blocks(window.image_width / tx + 1, window.image_height / ty + 1);
    dim3 threads(tx, ty);
    gpuRender<<<blocks, threads>>>(d_pixels, pixel_count, window.image_width, window.image_height, scene, d_rand_state, seed);
    cuda::synchronize();

    cuda::copyToHost(h_sampled_pixels, d_pixels, pixels_size);
    cuda::free(d_pixels);
    cuda::free(d_rand_state);
}
