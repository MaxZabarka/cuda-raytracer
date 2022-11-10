#include "Vec3.cuh"
#include <iostream>

__device__ __host__ Vec3::Vec3(float x, float y, float z) : x{x}, y{y}, z{z} {}

__device__ __host__ Vec3::~Vec3()
{
}
__device__ __host__ Vec3 Vec3::operator-(const Vec3 &other) const
{
    return Vec3(x - other.x, y - other.y, z - other.z);
}
__device__ __host__ Vec3 Vec3::operator+(const Vec3 &other) const
{
    return Vec3(x + other.x, y + other.y, z + other.z);
}
__device__ __host__ float Vec3::dot(const Vec3 &other) const
{
    return (x * other.x) + (y * other.y) + (z * other.z);
};
