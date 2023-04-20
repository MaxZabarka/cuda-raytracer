#include "Camera.cuh"
#include "Vec3.cuh"
#include "stdio.h"
// #include <cmath>
// #include <iostream>

Camera::Camera(
    int image_width,
    int image_height,
    Point position,
    FloatColor background,
    float fov,
    float far)
    : image_width{image_width},
      image_height{image_height},
      background{background},
      position{position},
      far{far}


{
    viewport_height = 2;
    viewport_width = viewport_height * (float)image_width / (float)image_height;
    // viewport_distance = 1 / tan(((fov / 2) * 2.14) / 180.0) / 2;
    viewport_distance = 1;
}
__device__ __host__ Point Camera::to_viewport(float x, float y)
{
    float viewportX = (x * (viewport_width / image_width) + position.x) - viewport_width / 2;
    float viewportY = (y * (viewport_height / image_height) + position.y) - viewport_height / 2;
    float viewportZ = (viewport_distance + position.z);

    return Point(viewportX, viewportY, viewportZ);
}

__device__ __host__ Camera::~Camera()
{
}
