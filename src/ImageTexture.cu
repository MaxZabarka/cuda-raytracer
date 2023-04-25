#include "ImageTexture.cuh"
#include <stdio.h>
#include <SDL2/SDL.h>
#include <SDL2/SDL_image.h>
#include <iostream>

__device__ __host__ ImageTexture::ImageTexture(ImageData image_data) : image_data(image_data)
{
}

__device__ __host__ ImageTexture::~ImageTexture()
{
}

__device__ __host__ FloatColor ImageTexture::get_color(Vec2 coordinate)
{

    int x = static_cast<int>(floor(coordinate.u * image_data.width));
    int y = static_cast<int>(floor(coordinate.v * image_data.height));

    if (x < 0 || x >= image_data.width || y < 0 || y >= image_data.height)
    {
        return FloatColor(0, 0, 0);
    }

    int index = (y * image_data.width + x) ;
    return image_data.data[index];
}
