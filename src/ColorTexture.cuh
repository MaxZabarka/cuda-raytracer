#pragma once
#include "Texture.cuh"
#include "Vec2.cuh"

class ColorTexture: public Texture 
{
private:

public:
   __device__ __host__ ColorTexture(FloatColor color =FloatColor{});
    __device__ __host__ ~ColorTexture();
    __device__ __host__ virtual FloatColor get_color(Vec2 coordinate) override;
    FloatColor color;
};
