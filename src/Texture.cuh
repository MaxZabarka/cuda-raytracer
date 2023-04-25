#pragma once
#include "Vec2.cuh"
#include "Vec3.cuh"

class Texture
{
protected:
    Texture(){};

public:
    __device__ __host__ virtual FloatColor get_color(Vec2 coordinate) = 0;
};
