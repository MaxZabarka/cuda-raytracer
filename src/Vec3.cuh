#pragma once

class Vec3
{

public:
    __host__ __device__ Vec3(float x = 0, float y = 0, float z = 0);
    __host__ __device__ ~Vec3();

    __device__ __host__ Vec3 operator-(const Vec3 &other) const;
    __device__ __host__ Vec3 operator+(const Vec3 &other) const;
    __device__ __host__ float dot(const Vec3 &other) const;

    float x;
    float y;
    float z;
};

using Point = Vec3;
using Direction = Vec3;
