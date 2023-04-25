#pragma once

class Vec2
{

public:
    __host__ __device__ Vec2(float u = 0, float v = 0);
    __host__ __device__ ~Vec2();

    float u;
    float v;
    float z;
};