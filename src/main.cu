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
    // int image_width = 48;
    // int image_height = 48;

    int image_width = 512;
    int image_height = 288;

    Window window{image_width, image_height};

    Scene *scene;
    cudaMallocManaged(&scene, sizeof(Scene));

    Hittable **host_hittables;
    int num_hittables = 2;
    scene->hittable_list.size = num_hittables;

    scene->hittable_list.hittables = (Hittable **)cuda::mallocManaged(sizeof(Hittable *) * num_hittables);
    

    Sphere test_hittable = Sphere(Point{0.0F, 0.0F, 1.0F}, 0.5, Material{FloatColor{0.5f, 0.5f, 0.5f}});
    scene->hittable_list.hittables[0] = (Hittable *)cuda::mallocManaged(sizeof(Sphere));
    cuda::copyToDevice(scene->hittable_list.hittables[0], &test_hittable, sizeof(Sphere));
    // https://forums.developer.nvidia.com/t/copying-objects-to-device-with-virtual-functions/54927
    fixVirtualPointers<<<1, 1>>>((Sphere *)scene->hittable_list.hittables[0]);

    Sphere test_hittable2 = Sphere(Point{0.0F, -100.5F, 0.0F}, 100, Material{FloatColor{0.5f, 0.5f, 0.5f}});
    scene->hittable_list.hittables[1] = (Hittable *)cuda::mallocManaged(sizeof(Sphere));
    cuda::copyToDevice(scene->hittable_list.hittables[1], &test_hittable2, sizeof(Sphere));
    fixVirtualPointers<<<1, 1>>>((Sphere *)scene->hittable_list.hittables[1]);


    printf("sizeof sphere: %d\n", sizeof(Sphere));

    printf("radius: %f\n", ((Sphere *)scene->hittable_list.hittables[0])->radius);
    printPointerLocation(&((Sphere *)scene->hittable_list.hittables[0])->radius);


    Camera camera{image_width, image_height, Point(0.0f, 0.0f, 0.0f)};
    scene->camera = camera;

    window.draw_scene(scene);
}