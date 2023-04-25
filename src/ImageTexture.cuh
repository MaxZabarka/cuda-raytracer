#pragma once
#include "Texture.cuh"
#include "Vec2.cuh"
#include <string>

class ImageTexture: public Texture 
{
private:

public:
   __device__ __host__ ImageTexture(ImageData image_data);
    __device__ __host__ ~ImageTexture();
    __device__ __host__ virtual FloatColor get_color(Vec2 coordinate) override;
    ImageData image_data;
};
