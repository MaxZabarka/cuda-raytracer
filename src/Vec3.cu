#include "Vec3.cuh"
#include <iostream>
#include <math.h>
#include "Color.cuh"

__device__ __host__ Vec3::Vec3(float x, float y, float z) : x{x}, y{y}, z{z} {}

__device__ __host__ Vec3::~Vec3()
{
}
__device__ __host__ Vec3 Vec3::operator-(const Vec3 &other) const
{
    return Vec3(x - other.x, y - other.y, z - other.z);
}
__device__ __host__ Vec3 Vec3::operator-() const
{
    return Vec3(-x, -y, -z);
}
__device__ __host__ Vec3 Vec3::operator+(const Vec3 &other) const
{
    return Vec3(x + other.x, y + other.y, z + other.z);
}
__device__ __host__ Vec3 Vec3::operator*(const Vec3 &other) const
{
    return Vec3(x * other.x, y * other.y, z * other.z);
}
__device__ __host__ Vec3 Vec3::operator*(float scalar) const
{
    return Vec3(x * scalar, y * scalar, z * scalar);
}
__device__ __host__ Vec3 Vec3::operator+(float addend) const
{
    return Vec3(x + addend, y + addend, z + addend);
}

__device__ __host__ float Vec3::dot(const Vec3 &other) const
{
    return (x * other.x) + (y * other.y) + (z * other.z);
};
__device__ __host__ float Vec3::magnitude() const
{
    return sqrt(magnitude_squared());
}
__device__ __host__ float Vec3::magnitude_squared() const
{
    return x*x + y*y + z*z;
}
__device__ __host__ Vec3 Vec3::normalize() const
{
    float mag = magnitude();
    return Vec3(x / mag, y / mag, z / mag);
}
__device__ __host__ Color Vec3::to_int_color() const
{
    return Color{(uint8_t)(255.99 * x), (uint8_t)(255.99 * y), (uint8_t)(255.99 * z)};
}

__device__ __host__ Vec3 Vec3::square_root() const
{
    return Vec3(sqrt(x), sqrt(y), sqrt(z));
}

__device__ __host__ Vec3 Vec3::cross(const Vec3 &other) const
{
    return Vec3(y * other.z - z * other.y, z * other.x - x * other.z, x * other.y - y * other.x);
}

__device__ __host__ bool Vec3::near_zero() const
{
    const float s = 1e-8;
    return (fabs(x) < s) && (fabs(y) < s) && (fabs(z) < s);
}