#pragma once
#include "Window.cuh"
#include "Vec3.cuh"
#include <vector>
#include "Color.cuh"
#include "Camera.cuh"
#include "Scene.cuh"

__global__ void gpuRender(uint32_t *sampled_pixels, int pixel_count, Scene *scene, Camera &camera);
__device__ Color trace_ray();

class Renderer
{
private:
public:
    Renderer();
    ~Renderer();
    void render(uint32_t *sampled_pixels, Scene *scene, Window &window);
};
