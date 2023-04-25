#include "ColorTexture.cuh"


__device__ __host__ ColorTexture::ColorTexture(FloatColor color) : color{color}
{
}

__device__ __host__ ColorTexture::~ColorTexture()
{
}


__device__ __host__ FloatColor ColorTexture::get_color(Vec2 coordinate)
{
    return color;
}



