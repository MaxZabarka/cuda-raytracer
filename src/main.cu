#include "Window.cuh"
#include "Scene.cuh"
#include "Sphere.cuh"
#include "Camera.cuh"
#include "Material.cuh"
#include "Vec3.cuh"
#include "cuda-wrapper/cuda.cuh"
#include "Triangle.cuh"
#include "OBJParser.cuh"

int main()
{
    OBJParser obj_parser = OBJParser("monkey.obj");
    std::vector<TriangleData> triangle_data = obj_parser.parse();

    // int image_width = 48;
    // int image_height = 48;


    int image_width = 512;
    int image_height = 288;

    Window window{image_width, image_height};

    Scene *scene;
    cudaMallocManaged(&scene, sizeof(Scene));

    int num_hittables = triangle_data.size();
    scene->hittable_list.size = num_hittables;

    scene->hittable_list.hittables = (Hittable **)cuda::mallocManaged(sizeof(Hittable *) * num_hittables);

    for (int i = 0; i < num_hittables; i++)
    {
        TriangleData data = triangle_data[i];
        Triangle triangle = Triangle(data.v1.position, data.v2.position, data.v3.position, Material{FloatColor{0.5f, 0.5f, 0.5f}});
        scene->hittable_list.hittables[i] = (Hittable *)cuda::mallocManaged(sizeof(Triangle));
        cuda::copyToDevice(scene->hittable_list.hittables[i], &triangle, sizeof(Triangle));
        cuda::fixVirtualPointers<<<1, 1>>>((Triangle *)scene->hittable_list.hittables[i]);
        printf("Triangle: %f, %f, %f\n", data.v1.position.x, data.v1.position.y, data.v1.position.z);
    }

    // Triangle test_hittable = Triangle(Point{1.0F, 0.0F, 5.0F}, Point{0.0F, 0.0F, 5.0F}, Point{0.0F, 1.0F, 5.0F}, Material{FloatColor{0.5f, 0.5f, 0.5f}});
    // scene->hittable_list.hittables[0] = (Hittable *)cuda::mallocManaged(sizeof(Triangle));
    // cuda::copyToDevice(scene->hittable_list.hittables[0], &test_hittable, sizeof(Triangle));
    // // https://forums.developer.nvidia.com/t/copying-objects-to-device-with-virtual-functions/54927
    // cuda::fixVirtualPointers<<<1, 1>>>((Triangle *)scene->hittable_list.hittables[0]);

    // Sphere test_hittable = Sphere(Point{0.0F, 1.0F, 2.0F}, 0.1, Material{FloatColor{1.0f, 0.0f, 0.0f}});
    // scene->hittable_list.hittables[0] = (Hittable *)cuda::mallocManaged(sizeof(Sphere));
    // cuda::copyToDevice(scene->hittable_list.hittables[0], &test_hittable, sizeof(Sphere));
    // // https://forums.developer.nvidia.com/t/copying-objects-to-device-with-virtual-functions/54927
    // cuda::fixVirtualPointers<<<1, 1>>>((Sphere *)scene->hittable_list.hittables[0]);

    // Sphere test_hittable2 = Sphere(Point{0.0F, 0.0F, 2.0F}, 0.1, Material{FloatColor{0.5f, 0.5f, 0.5f}});
    // scene->hittable_list.hittables[1] = (Hittable *)cuda::mallocManaged(sizeof(Sphere));
    // cuda::copyToDevice(scene->hittable_list.hittables[1], &test_hittable2, sizeof(Sphere));
    // cuda::fixVirtualPointers<<<1, 1>>>((Sphere *)scene->hittable_list.hittables[1]);

    // Sphere test_hittable3 = Sphere(Point{1.0F, 0.0F, 2.0F}, 0.1, Material{FloatColor{0.0f, 0.0f, 1.0f}});
    // scene->hittable_list.hittables[2] = (Hittable *)cuda::mallocManaged(sizeof(Sphere));
    // cuda::copyToDevice(scene->hittable_list.hittables[2], &test_hittable3, sizeof(Sphere));
    // cuda::fixVirtualPointers<<<1, 1>>>((Sphere *)scene->hittable_list.hittables[2]);

    // Triangle test_hittable4 = Triangle(Point{1.0F, 0.0F, 2.0F}, Point{0.0F, 0.0F, 2.0F}, Point{0.0F, 1.0F, 2.0F}, Material{FloatColor{0.5f, 0.5f, 0.5f}});
    // scene->hittable_list.hittables[3] = (Hittable *)cuda::mallocManaged(sizeof(Triangle));
    // cuda::copyToDevice(scene->hittable_list.hittables[3], &test_hittable4, sizeof(Triangle));
    // cuda::fixVirtualPointers<<<1, 1>>>((Triangle *)scene->hittable_list.hittables[3]);

    Camera camera{image_width, image_height, Point(0.0f, 0.0f, 0.0f)};
    scene->camera = camera;

    window.draw_scene(scene);
}