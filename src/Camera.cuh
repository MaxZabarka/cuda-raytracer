#pragma once
#include "Color.cuh"
#include "Infinity.cuh"
#include "Vec3.cuh"
class Camera
{
public:
    Camera(
        int image_width,
        int image_height,
        Point position = Point(),
        FloatColor background = FloatColor(0.43f, 0.81f, 1.0f),
        float fov = 75,
        float far = INFINITY);
    __device__ __host__ ~Camera();

    __device__ __host__ Point to_viewport(float x, float y);

public:
    int image_width;
    int image_height;
    FloatColor background;
    Point position;
    float far;
    float fov;
    float viewport_distance;
    float viewport_width;
    float viewport_height;

};
