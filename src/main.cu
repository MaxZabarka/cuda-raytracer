#include "Window.cuh"
#include "Scene.cuh"
#include "Sphere.cuh"
#include "Camera.cuh"
#include "Material.cuh"
#include "Vec3.cuh"
#include "cuda-wrapper/cuda.cuh"

int main()
{

    int image_width = 512;
    int image_height = 288;

    Window window{512, 288};

    // Create scene
    Camera camera{image_width, image_height, Point(0.0f, 0.0f, 0.0f)};
    Scene *scene = (Scene *)cuda::mallocManaged(sizeof(Scene));

    scene->sphere_count = 4;

    scene->camera = camera;
    scene->spheres = (Sphere *)cuda::mallocManaged(sizeof(Sphere) * scene->sphere_count);

    // scene->spheres[0] = Sphere{Point{0.0F, 0.0F, 1.0F}, 0.5, Material{FloatColor{1.0f, 0.0f, 0.0f}}};  // red
    // scene->spheres[1] = Sphere{Point{0.0F, -100.5F, 1.0F}, 100, Material{FloatColor{0.0f, 1.0f, 0.0f}}};  // green

    // scene->spheres[0] = Sphere{Point{1.1F, 0.0F, 1.0F}, 0.5, Material{FloatColor{0.5f, 0.0f, 0.0f}}};  // red
    scene->spheres[1] = Sphere{Point{0.0F, 0.0F, 1.0F}, 0.5, Material{FloatColor{0.5f, 0.5f, 0.5f}}}; // green
    // scene->spheres[2] = Sphere{Point{-1.1F, 0.0F, 1.0F}, 0.5, Material{FloatColor{0.0f, 0.0f, 0.5f}}}; // blue
    scene->spheres[3] = Sphere{Point{0, -100.5f, 0}, 100, Material{FloatColor{0.5f, 0.5f, 0.5f}}}; // ground

    window.draw_scene(scene);
}