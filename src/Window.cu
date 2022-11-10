#include "Window.cuh"
#include <iostream>
#include <SDL2/SDL.h>
#include <tuple>
#include "Renderer.cuh"
#include "cuda-wrapper/cuda.cuh"
#include <unistd.h>
#include "Hittable.cuh"
#include "Sphere.cuh"
#include "Scene.cuh"
#include "Camera.cuh"

Window::Window(int image_width, int image_height, int window_width, int window_height) : image_width{image_width}, image_height{image_height}
{
    if (window_width == 0 || window_height == 0)
    {
        window_width = 512;
        window_height = window_width * image_height / image_width;
    }
    SDL_Init(SDL_INIT_VIDEO);
    SDL_CreateWindowAndRenderer(window_width, window_height, 0, &window, &renderer);
    SDL_RenderSetLogicalSize(renderer, image_width, image_height);
}

Window::~Window()
{
}

void Window::draw(uint8_t *pixels)
{
    SDL_Texture *buffer = SDL_CreateTexture(renderer,
                                            SDL_PIXELFORMAT_RGB24, SDL_TEXTUREACCESS_STREAMING,
                                            image_width, image_height);

    int pitch = (image_width / 8) * 3;
}

void Window::draw_test()
{

    Renderer gpuRenderer{};
    int pixel_count = image_width * image_height * 3;
    uint8_t *pixels = (uint8_t *)malloc(pixel_count * sizeof(uint8_t));
    uint32_t *sampled_pixels = (uint32_t *)calloc(pixel_count, sizeof(uint32_t));

    SDL_Texture *buffer = SDL_CreateTexture(renderer,
                                            SDL_PIXELFORMAT_RGB24, SDL_TEXTUREACCESS_STREAMING,
                                            image_width, image_height);

    int pitch = (image_width / 8) * 3;
    SDL_Event event;

    bool quit = false;
    uint32_t current_sample = 0;

    // Hittable **hittables = (Hittable **)cuda::mallocManaged(sizeof(Hittable *) * 1);
    // Hittable *hittable = new Sphere{Point{2.2F, 0.0F, 0.0F}, 1, Material{Color{255, 0, 0}}};
    // Hittable *hittable = (Hittable*)cuda::mallocManaged(sizeof(Sphere));

    // *hittable = Sphere{Point{2.2F, 0.0F, 0.0F}, 1, Material{Color{255, 0, 0}}};

    // hittables[0] = new Sphere();
    // hittables[0] = (Hittable *)cuda::mallocManaged(sizeof(Hittable));
    // (hittables[0]) = hittable;

    // (*(hittables[0])).get_material();

    // hittables[0] = Sphere{Vec3{0, 0, -1}, 0.5};

    // *hittables = (Hittable*)cuda::mallocManaged(sizeof(Hittable*));

    // Sphere sphere1 = Sphere{Point{2.2F, 0.0F, 0.0F}, 1, Material{Color{255, 0, 0}}};

    // Sphere* sphere2 = new Sphere{ Point{0.0F, 0.0F, 0.0F}, 1, Material{ Color{0, 255, 0} } };
    // Sphere* sphere3 = new Sphere{ Point{-2.2F, 0.0F, 0.0F}, 1, Material{ Color{0, 0, 255} } };

    // Camera camera = cuda::mallocManaged(sizeof(Camera));

    Camera camera{image_width, image_height, Point(-5.0f, 0, -5.0f)};
    Scene *scene = (Scene *)cuda::mallocManaged(sizeof(Scene));

    // scene->hittables = hittables;

    scene->sphere_count = 3;
    scene->camera = camera;
    scene->spheres = (Sphere *)cuda::mallocManaged(sizeof(Sphere) * scene->sphere_count);
    // scene->spheres[0] = Sphere{Point{0.0F, 0.0F, 5.0F}, 1, Material{Color{255, 0, 0}}};
    scene->spheres[0] = Sphere{Point{2.2F, 0.0F, 0.0F}, 1, Material{Color{255, 0, 0}}};
    scene->spheres[1] = Sphere{Point{0.0F, 0.0F, 0.0F}, 1, Material{Color{0, 255, 0}}};
    scene->spheres[2] = Sphere{Point{-2.2F, 0.0F, 0.0F}, 1, Material{Color{0, 0, 255}}};

    printf("%f\n", scene->spheres[0].radius);
    // test_hittable->get_material();

    // Scene scene{ hittables, 0, .camera = camera };

    while (!quit)
    {
        SDL_PollEvent(&event);
        switch (event.type)
        {
        case SDL_QUIT:
            quit = true;
        }
        SDL_LockTexture(buffer,
                        NULL,
                        (void **)(&pixels),
                        &pitch);

        gpuRenderer.render(sampled_pixels, scene, *this);
        // for (int x = 0; x < image_width; x++)
        // {
        //     for (int y = 0; y < image_height; y++)
        //     {
        //         int index = (y * image_width + (x + offset) % image_width) * 3;
        //         pixels[index] = x * 255 / image_width;
        //         pixels[index + 1] = y * 255 / image_height;
        //         pixels[index + 2] = 65;
        //     }
        // }
        // std::cout << "Sample: " << current_sample << std::endl;
        current_sample++;
        for (int i = 0; i < pixel_count; i++)
        {
            pixels[i] = sampled_pixels[i] / current_sample;
        }

        SDL_UnlockTexture(buffer);
        SDL_RenderCopy(renderer, buffer, NULL, NULL);
        SDL_RenderPresent(renderer);
        // sleep(1);
    }
}

void Window::show()
{
    SDL_RenderPresent(renderer);
    event_loop();
}

std::tuple<int, int> Window::quadrant_to_real(int x, int y)
{
    x = round(image_width / 2) + x;
    y = round(image_height / 2) - y;
    return std::make_tuple(x, y);
}

void Window::event_loop()
{
    SDL_Event event;

    bool quit = false;
    while (!quit)
    {
        SDL_WaitEvent(&event);
        switch (event.type)
        {
        case SDL_QUIT:
            quit = true;
        }
    }
}