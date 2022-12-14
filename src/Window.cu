#include "Window.cuh"
#include <iostream>
#include <SDL2/SDL.h>
#include <time.h>
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
        window_width = 750;
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
    // Setup renderer
    Renderer gpuRenderer{};
    int pixel_size = image_width * image_height * 3;
    uint8_t *pixels = (uint8_t *)malloc(pixel_size * sizeof(uint8_t));
    uint32_t *sampled_pixels = (uint32_t *)calloc(pixel_size, sizeof(uint32_t));
    int pitch = (image_width / 8) * 3;

    SDL_Texture *buffer = SDL_CreateTexture(renderer,
                                            SDL_PIXELFORMAT_RGB24, SDL_TEXTUREACCESS_STREAMING,
                                            image_width, image_height);

    // Create scene
    Camera camera{image_width, image_height, Point(0.0f, 0.0f, -1.0f)};
    Scene *scene = (Scene *)cuda::mallocManaged(sizeof(Scene));

    scene->sphere_count = 2;
    scene->camera = camera;
    scene->spheres = (Sphere *)cuda::mallocManaged(sizeof(Sphere) * scene->sphere_count);

    scene->spheres[0] = Sphere{Point{0.0F, 0.0F, 0.0F}, 0.5, Material{FloatColor{1.0f, 0.0f, 0.0f}}};  // red
    scene->spheres[1] = Sphere{Point{0.0F, -100.5F, 0.0F}, 100, Material{FloatColor{0.0f, 1.0f, 0.0f}}};  // green

    // scene->spheres[0] = Sphere{Point{2.2F, 0.0F, 0.0F}, 1, Material{FloatColor{1.0f, 0.0f, 0.0f}}};  // red
    // scene->spheres[1] = Sphere{Point{0.0F, 0.0F, 0.0F}, 1, Material{FloatColor{0.0f, 1.0f, 0.0f}}};  // green
    // scene->spheres[2] = Sphere{Point{-2.2F, 0.0F, 0.0F}, 1, Material{FloatColor{0.0f, 0.0f, 1.0f}}}; // blue
    // scene->spheres[3] = Sphere{Point{0, -101, 0}, 100, Material{FloatColor{0.5f, 0.5f, 0.5f}}};    // ground

    clock_t lastTick = clock();
    clock_t dt = 0;

    // create loop
    bool rerender = false;
    SDL_Event event;

    bool quit = false;
    uint32_t current_sample = 0;

    while (!quit)
    {
        printf("Sample: %d\n", current_sample);
        clock_t now = clock();
        dt = (now - lastTick);
        lastTick = now;
        const Uint8 *keyboard_state = SDL_GetKeyboardState(NULL);

        SDL_PollEvent(&event);
        switch (event.type)
        {
        case SDL_QUIT:
            quit = true;
        default:
            break;
        }

        if (rerender)
        {
            current_sample = 0;
            for (int i = 0; i < pixel_size; i++)
            {
                sampled_pixels[i] = 0;
            }
            rerender = false;
        }

        float move_speed = 0.00001f * dt;

        if (keyboard_state[SDL_SCANCODE_W])
        {
            scene->camera.position.z += move_speed;
            rerender = true;
        }

        if (keyboard_state[SDL_SCANCODE_S])
        {
            scene->camera.position.z -= move_speed;
            rerender = true;
        }
        if (keyboard_state[SDL_SCANCODE_D])
        {
            scene->camera.position.x += move_speed;
            rerender = true;
        }
        if (keyboard_state[SDL_SCANCODE_A])
        {
            scene->camera.position.x -= move_speed;
            rerender = true;
        }
        if (keyboard_state[SDL_SCANCODE_LSHIFT])
        {
            scene->camera.position.y -= move_speed;
            rerender = true;
        }
        if (keyboard_state[SDL_SCANCODE_SPACE])
        {
            scene->camera.position.y += move_speed;
            rerender = true;
        }

        SDL_LockTexture(buffer,
                        NULL,
                        (void **)(&pixels),
                        &pitch);

        gpuRenderer.render(sampled_pixels, scene, *this, current_sample);
        // return;
        current_sample++;
        for (int i = 0; i < pixel_size; i++)
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