#include "Window.cuh"
#include "Scene.cuh"
#include "Sphere.cuh"
#include "Camera.cuh"
#include "Material.cuh"
#include "Vec3.cuh"
#include "cuda-wrapper/cuda.cuh"

template <typename T>
__global__ void fixVirtualPointers(T *other)
{
    T temp = T(*other);
    memcpy(other, &temp, sizeof(T));
}

void printPointerLocation(void *ptr)
{
    cudaPointerAttributes attributes;
    cudaError_t error = cudaPointerGetAttributes(&attributes, ptr);
    if (error != cudaSuccess)
    {
        printf("Error: %s\n", cudaGetErrorString(error));
        return;
    }

    if (attributes.type == cudaMemoryTypeHost)
    {
        printf("Pointer is in host memory\n");
    }
    else if (attributes.type == cudaMemoryTypeDevice)
    {
        printf("Pointer is in device memory\n");
    }
    else if (attributes.type == cudaMemoryTypeManaged)
    {
        printf("Pointer is in unified memory\n");
    }
    else
    {
        printf("Unknown memory type\n");
    }
}

int main()
{

    int image_width = 60;
    int image_height = 60;

    Window window{image_width, image_height};

    Scene *scene;
    cudaMallocManaged(&scene, sizeof(Scene));

    Hittable **host_hittables;
    int hittable_count = 1;

    scene->hittables = (Hittable **)cuda::mallocManaged(sizeof(Hittable *) * hittable_count);

    Sphere test_hittable = Sphere(Point{0.0F, 0.0F, 1.0F}, 0.5, Material{FloatColor{1.0f, 0.0f, 0.0f}});

    scene->hittables[0] = (Hittable *)cuda::mallocManaged(sizeof(Sphere));
    cuda::copyToDevice(scene->hittables[0], &test_hittable, sizeof(Sphere));
    // https://forums.developer.nvidia.com/t/copying-objects-to-device-with-virtual-functions/54927
    fixVirtualPointers<<<1, 1>>>((Sphere *)scene->hittables[0]);


    printf("sizeof sphere: %d\n", sizeof(Sphere));

    printf("radius: %f\n", ((Sphere *)scene->hittables[0])->radius);
    printPointerLocation(&((Sphere *)scene->hittables[0])->radius);

    scene->sphere_count = hittable_count;

    Camera camera{image_width, image_height, Point(0.0f, 0.0f, 0.0f)};
    scene->camera = camera;

    window.draw_scene(scene);
}