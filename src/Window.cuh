#pragma once
#include <SDL2/SDL.h>
#include "Vec3.cuh"

class Window
{
private:
    SDL_Window *window;
    SDL_Renderer *renderer;

    void event_loop();
    std::tuple<int, int> quadrant_to_real(int x, int y);

public:
    Window(int image_width = 256, int image_height = 256, int window_width = 0, int window_height = 0);
    ~Window();
    void draw_test();
    void draw(u_int8_t* pixels);
    void show();
    int image_width;
    int image_height;
};
