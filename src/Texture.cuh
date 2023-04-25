#pragma once
#include "Vec2.cuh"
#include "Vec3.cuh"

struct ImageData {
    int width;
    int height;
    FloatColor* data;
};
class Texture
{
protected:
    Texture(){};

public:
    __device__ __host__ virtual FloatColor get_color(Vec2 coordinate) = 0;
};
