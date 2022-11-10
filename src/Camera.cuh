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
        Color background = Color{110, 209, 255},
        float fov = 75,
        float far = INFINITY);
    __device__ __host__ ~Camera();

    __device__ __host__ Point to_viewport(int x, int y);

public:
    int image_width;
    int image_height;
    Color background;
    Point position;
    float far;
    float fov;
    float viewport_distance;
    float viewport_width;
    float viewport_height;

};
