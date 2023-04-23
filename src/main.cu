#include "Window.cuh"
#include "Scene.cuh"
#include "Sphere.cuh"
#include "Camera.cuh"
#include "Material.cuh"
#include "Vec3.cuh"
#include "cuda-wrapper/cuda.cuh"
#include "Triangle.cuh"
#include "OBJParser.cuh"
#include "Mesh.cuh"

int main()
{
    // OBJParser obj_parser = OBJParser("monkey.obj");
    // Geometry geometry = obj_parser.parse();

    // Mesh mesh = Mesh(geometry, Material{FloatColor{0.5f, 0.5f, 0.5f}});
    // cuda::fixVirtualPointers << 1, 1 >>> (&mesh);

    // int image_width = 48;
    // int image_height = 48;

    int image_width = 512;
    int image_height = 288;

    Window window{image_width, image_height};

    Scene *scene;
    cudaMallocManaged(&scene, sizeof(Scene));

    int num_hittables = 1;
    scene->hittable_list.size = num_hittables;
    scene->hittable_list.hittables = (Hittable **)cuda::mallocManaged(sizeof(Hittable *) * num_hittables);

    // scene->hittable_list.hittables[0] = (Hittable *)cuda::mallocManaged(sizeof(Mesh));
    // cuda::copyToDevice(scene->hittable_list.hittables[0], &mesh, sizeof(Mesh));

    // cuda::fixVirtualPointers<<<1, 1>>>((Mesh *)scene->hittable_list.hittables[0]);


    // printf("%f\n", mesh.geometry.triangles[0].a.position.y);
    // printf("%f\n", ((Mesh *)scene->hittable_list.hittables[0])->geometry.triangles[0].a.position.y);

    // print_pointer_location(((Mesh *)scene->hittable_list.hittables[0])->triangles);

    // for (int i = 0; i < num_hittables; i++)
    // {
    //     TriangleData data = triangle_data[i];
    //     Triangle triangle = Triangle(data, Material{FloatColor{0.5f, 0.5f, 0.5f}});
    //     scene->hittable_list.hittables[i] = (Hittable *)cuda::mallocManaged(sizeof(Triangle));
    //     cuda::copyToDevice(scene->hittable_list.hittables[i], &triangle, sizeof(Triangle));
    //     cuda::fixVirtualPointers<<<1, 1>>>((Triangle *)scene->hittable_list.hittables[i]);
    //     // printf("Triangle: %f, %f, %f\n", data.a.position.x, data.b.position.y, data.c.position.z);
    // }

    // Triangle test_hittable = Triangle(Point{1.0F, 0.0F, 5.0F}, Point{0.0F, 0.0F, 5.0F}, Point{0.0F, 1.0F, 5.0F}, Material{FloatColor{0.5f, 0.5f, 0.5f}});
    // scene->hittable_list.hittables[0] = (Hittable *)cuda::mallocManaged(sizeof(Triangle));
    // cuda::copyToDevice(scene->hittable_list.hittables[0], &test_hittable, sizeof(Triangle));
    // // https://forums.developer.nvidia.com/t/copying-objects-to-device-with-virtual-functions/54927
    // cuda::fixVirtualPointers<<<1, 1>>>((Triangle *)scene->hittable_list.hittables[0]);

    Sphere test_hittable = Sphere(Point{0.0F, 1.0F, 2.0F}, 0.1, Material{FloatColor{1.0f, 0.0f, 0.0f}});
    scene->hittable_list.hittables[0] = (Hittable *)cuda::mallocManaged(sizeof(Sphere));
    cuda::copyToDevice(scene->hittable_list.hittables[0], &test_hittable, sizeof(Sphere));
    // https://forums.developer.nvidia.com/t/copying-objects-to-device-with-virtual-functions/54927
    cuda::fixVirtualPointers<<<1, 1>>>((Sphere *)scene->hittable_list.hittables[0]);

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