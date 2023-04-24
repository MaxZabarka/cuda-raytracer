#pragma once
#include "Color.cuh"

class Vec3
{

public:
    __host__ __device__ Vec3(float x = 0, float y = 0, float z = 0);
    __host__ __device__ ~Vec3();

    __device__ __host__ Vec3 operator-(const Vec3 &other) const;
    __device__ __host__ Vec3 operator+(const Vec3 &other) const;
    __device__ __host__ Vec3 operator*(const Vec3 &other) const;
    __device__ __host__ Vec3 operator*(float scalar) const;
    
    __device__ __host__ Vec3 operator+(float addend) const;
    __device__ __host__ Vec3 operator-() const;

    __device__ __host__ float dot(const Vec3 &other) const;
    __device__ __host__ float magnitude() const;
    __device__ __host__ float magnitude_squared() const;
    __device__ __host__ Vec3 normalize() const;

    __device__ __host__ Vec3 cross(const Vec3 &other) const;

    __device__ __host__ Color to_int_color() const;
    __device__ __host__ Vec3 square_root() const;

    __device__ __host__ bool near_zero() const;

    float x;
    float y;
    float z;
};

using FloatColor = Vec3;
using Point = Vec3;
using Direction = Vec3;
