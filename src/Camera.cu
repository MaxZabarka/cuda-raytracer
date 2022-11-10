#include "Camera.cuh"
#include "Vec3.cuh"
#include "stdio.h"
// #include <cmath>
// #include <iostream>

Camera::Camera(
    int image_width,
    int image_height,
    Point position,
    Color background,
    float fov,
    float far)
    : image_width{image_width},
      image_height{image_height},
      background{background},
      position{position},
      far{far}

{
    viewport_height = 1;
    viewport_width = 1 * image_width / image_height;
    // viewport_distance = 1;
    viewport_distance = 1 / tan(((fov / 2) * 2.14) / 180.0) / 2;
}
__device__ __host__ Point Camera::to_viewport(int x, int y)
{

    float viewportX = (x * (viewport_width / image_width) + position.x);
    float viewportY = (y * (viewport_height / image_height) + position.y);
    float viewportZ = (viewport_distance + position.z);



    // return Point();
    // return Point(1.0f, 2.0f, 3.0f);
    return Point(viewportX, viewportY, viewportZ);
}

__device__ __host__ Camera::~Camera()
{
}
