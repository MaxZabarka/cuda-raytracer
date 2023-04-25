#include "ColorTexture.cuh"
#include <stdio.h>

__device__ __host__ ColorTexture::ColorTexture(FloatColor color) : color{color}
{
}

__device__ __host__ ColorTexture::~ColorTexture()
{
}

__device__ __host__ FloatColor ColorTexture::get_color(Vec2 coordinate)
{
    return color;

    // // printf("coordinates %f %f\n", coordinate.u, coo10rdinate.v);
    // float scale = 10.0f;
    // int ix = static_cast<int>(floor(coordinate.u * scale));
    // int iy = static_cast<int>(floor(coordinate.v * scale));
    // bool isOdd = ((ix + iy) % 2) != 0;

    // return isOdd ? FloatColor(0, 0, 0) : FloatColor(1, 0, 0);
}
