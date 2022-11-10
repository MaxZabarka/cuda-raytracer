#include "Renderer.cuh"
#include <iostream>
#include "cuda-wrapper/cuda.cuh"

#include "Window.cuh"
#include "Color.cuh"
#include "Camera.cuh"
#include "Ray.cuh"
#include "Hittable.cuh"
#include "Scene.cuh"

Renderer::Renderer()
{
}

Renderer::~Renderer()
{
}

__device__ Color trace_ray(Ray &ray, Camera &camera, Scene *scene)
{
    float closest_distance = INFINITY;
    Sphere *closest_sphere = nullptr;

    for (int i = 0; i < scene->sphere_count; i++)
    {   
        Sphere *sphere = (scene->spheres+i);
        Hit hit = sphere->hit(ray);

        if (hit.t > 0.001 && hit.t < camera.far && hit.t < closest_distance)
        {
            closest_distance = hit.t;
            closest_sphere = sphere;
        }
    }

    if (closest_sphere)
    {
        return closest_sphere->get_material().color;
    }

    return camera.background;
}

__global__ void gpuRender(uint32_t *sampled_pixels, int pixel_count, size_t image_width, size_t image_height, Scene *scene)
{
    int x = (threadIdx.x + blockIdx.x * blockDim.x);
    int y = (threadIdx.y + blockIdx.y * blockDim.y);

    int index = (y * image_width + x) * 3;

    if (x >= image_width || y >= image_height)
        return;

    // printf("x: %d, y: %d, index: %d\n", x, y, index);
    // sampled_pixels[index] += 255;
    // sampled_pixels[index + 1] += 255;
    // sampled_pixels[index + 2] += 255;
    // scene->camera.to_viewport(x, y);
    // camera.to_viewport(x, y);
    Camera camera = scene->camera;
    Direction ray_direction = scene->camera.to_viewport(x, y) - camera.position;
    Ray ray = {.direction = ray_direction, .origin = camera.position};

    // return;
    Color color = trace_ray(ray, camera, scene);

    
    sampled_pixels[index] += color.r;
    sampled_pixels[index + 1] += color.g;
    sampled_pixels[index + 2] += color.b;
}

void Renderer::render(uint32_t *h_sampled_pixels, Scene *scene, Window &window)
{
    size_t pixels_size = window.image_width * window.image_height * 3 * sizeof(uint32_t);
    size_t pixel_count = window.image_height * window.image_width;

    // Set up device memory
    uint32_t *d_pixels = (uint32_t *)cuda::malloc(pixels_size);
    cuda::copyToDevice(d_pixels, h_sampled_pixels, pixels_size);

    // Scene *d_scene = (Scene *)cuda::malloc(sizeof(Scene));
    // cuda::copyToDevice(d_scene, &scene, sizeof(Scene));


    // Hittable** d_hittables = (Hittable**)cuda::malloc(sizeof(Hittable*) * scene.hittable_count);
    // for (int i = 0; i < scene.hittable_count; i++)
    // {
    //     Hittable* d_hittable = (Hittable*)cuda::malloc(sizeof(Hittable));
    //     cuda::copyToDevice(d_hittable, scene.hittables[i], sizeof(Hittable));
    //     cuda::copyToDevice(d_hittables + i, &d_hittable, sizeof(Hittable*));
    // }

    // return;
    // cuda::copyToDevice(d_hittables, scene.hittables, sizeof(Hittable*) * scene.hittable_count);
    // d_scene->hittables = d_hittables;

    // printf("hittable count: %d\n", d_scene->hittable_count);

    int tx = 16;
    int ty = 16;

    dim3 blocks(window.image_width / tx + 1, window.image_height / ty + 1);
    dim3 threads(tx, ty);
    gpuRender<<<blocks, threads>>>(d_pixels, pixel_count, window.image_width, window.image_height, scene);
    cuda::synchronize();

    cuda::copyToHost(h_sampled_pixels, d_pixels, pixels_size);
    cuda::free(d_pixels);
}
