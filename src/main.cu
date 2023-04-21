#include "Window.cuh"
#include "Scene.cuh"
#include "Sphere.cuh"
#include "Camera.cuh"
#include "Material.cuh"
#include "Vec3.cuh"
#include "cuda-wrapper/cuda.cuh"

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
    scene->hittables[0] = (Hittable *)cuda::mallocManaged(sizeof(Sphere));

    Sphere test_hittable = Sphere(Point{0.0F, 0.0F, 1.0F}, 0.5, Material{FloatColor{1.0f, 0.0f, 0.0f}});
    cuda::copyToDevice(scene->hittables[0], &test_hittable, sizeof(Sphere));

    ((Sphere *)(scene->hittables[0]))->radius = 5;
    // cudaMemset(scene->hittables[0], 5, sizeof(Sphere));
    // *((void*)(scene->hittables[0])) = 5;

    printPointerLocation(scene->hittables[0]);

    // scene->hittables = host_hittables;
    scene->sphere_count = hittable_count;

    // cudaMemcpy(scene->hittables, host_hittables, hittable_count * sizeof(Hittable *), cudaMemcpyHostToDevice);

    // cudaMallocManaged(&scene->hittables, hittable_count * sizeof(Hittable *));

    // cudaMallocManaged(&host_hittables[0], sizeof(Sphere));
    // ((Hittable*)host_hittables[0])->get_material();

    // host_hittables[0] = new Sphere(Point{0.0F, 0.0F, 1.0F}, 0.5, Material{FloatColor{1.0f, 0.0f, 0.0f}});
    // Create the scene object

    Camera camera{image_width, image_height, Point(0.0f, 0.0f, 0.0f)};
    scene->camera = camera;

    // Scene *scene = (Scene *)cuda::mallocManaged(sizeof(Scene));

    // scene->hittables = (Hittable **)cuda::mallocManaged(sizeof(Hittable *) * 1);
    // cuda::copyToDevice(scene->hittables, &scene->hittables, sizeof(Hittable **));

    // // printPointerLocation();
    // Sphere cpu_sphere = Sphere{Point{0.0F, 0.0F, 1.0F}, 0.5, Material{FloatColor{1.0f, 0.0f, 0.0f}}};
    // Sphere *gpu_sphere = (Sphere *)cuda::mallocManaged(sizeof(Sphere));

    // cuda::copyToDevice(gpu_sphere, &cpu_sphere, sizeof(Sphere));

    // scene->hittables[0] = gpu_sphere;

    // printf("%f\n", scene->hittables[0]->get_material().color.x);

    // gpu_sphere->radius = 5;
    // ((Sphere *)(scene->hittables[0]))->radius = 5;
    // ((Sphere *)scene->hittables[0])->radius = 5;
    // // printPointerLocation(());
    // scene->sphere_count = 1;

    // scene->hittables[0] = (Sphere *)cuda::mallocManaged(sizeof(Sphere));

    // ((Sphere *)(scene->hittables[0]))->position = Point{0.0F, 0.0F, 1.0F};
    // ((Sphere *)scene->hittables[0])->radius = 0.5;
    // ((Sphere *)scene->hittables[0])->material.color = FloatColor{1.0f, 0.0f, 0.0f};

    // scene->hittables[1] = (Hittable *)cuda::mallocManaged(sizeof(Sphere));
    // scene->hittables[2] = (Hittable *)cuda::mallocManaged(sizeof(Sphere));
    // scene->hittables[3] = (Hittable *)cuda::mallocManaged(sizeof(Sphere));

    // Sphere test_sphere = Sphere{Point{0.0F, 0.0F, 1.0F}, 0.5, Material{FloatColor{1.0f, 0.0f, 0.0f}}};
    // printf("%d\n", scene->hittables[0]->get_material().color.x);
    // scene->hittables[0] = (Hittable*)cuda::mallocManaged(sizeof(test_sphere));

    // memcpy(scene->hittables[0], &test_sphere, sizeof(test_sphere));

    // Sphere *test_sphere = (Sphere *)cuda::mallocManaged(sizeof(Sphere));
    // test_sphere->position = Point{0.0F, 0.0F, 1.0F};
    // test_sphere->radius = 0.5;
    // test_sphere->material.color = FloatColor{1.0f, 0.0f, 0.0f};

    // scene->hittables = (Sphere *)cuda::mallocManaged(sizeof(Sphere) * scene->sphere_count);

    // // scene->spheres[0] = Sphere{Point{0.0F, 0.0F, 1.0F}, 0.5, Material{FloatColor{1.0f, 0.0f, 0.0f}}};  // red
    // // scene->spheres[1] = Sphere{Point{0.0F, -100.5F, 1.0F}, 100, Material{FloatColor{0.0f, 1.0f, 0.0f}}};  // green

    // // scene->spheres[0] = Sphere{Point{1.1F, 0.0F, 1.0F}, 0.5, Material{FloatColor{0.5f, 0.0f, 0.0f}}};  // red
    // scene->hittables[0] = Sphere{Point{0.0F, 0.0F, 1.0F}, 0.5, Material{FloatColor{0.5f, 0.5f, 0.5f}}}; // green
    // // scene->spheres[2] = Sphere{Point{-1.1F, 0.0F, 1.0F}, 0.5, Material{FloatColor{0.0f, 0.0f, 0.5f}}}; // blue
    // scene->hittables[3] = &Sphere{Point{0, -100.5f, 0}, 100, Material{FloatColor{0.5f, 0.5f, 0.5f}}}; // ground

    window.draw_scene(scene);
}