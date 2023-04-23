#include "Triangle.cuh"
#include "Hit.cuh"
#include "Infinity.cuh"
#include <cmath>
#include <iostream>

__device__ __host__ Triangle::Triangle(Point a, Point b, Point c, Material material) : a{a}, b{b}, c{c}, material{material}
{
}

__device__ __host__ Triangle::~Triangle()
{
}

__device__ __host__ Hit Triangle::hit(const Ray &ray)
{
    Hit result;
    result.t = INFINITY;
    result.hittable = nullptr;

    // Edge vectors
    Vec3 edge1 = b - a;
    Vec3 edge2 = c - a;

    // Compute the determinant
    Vec3 P = ray.direction.cross(edge2);
    float det = edge1.dot(P);

    // Check if the ray is parallel to the triangle (det ~= 0)
    const float EPSILON = 1e-7;
    if (std::fabs(det) < EPSILON)
    {
        return result; // No intersection
    }

    float inv_det = 1.0 / det;

    // Calculate vector from vertex A to ray origin
    Vec3 T = ray.origin - a;

    // Calculate u coordinate
    float u = T.dot(P) * inv_det;
    if (u < 0.0 || u > 1.0)
    {
        return result; // No intersection
    }

    // Prepare to test v coordinate
    Vec3 Q = T.cross(edge1);

    // Calculate v coordinate
    float v = ray.direction.dot(Q) * inv_det;
    if (v < 0.0 || u + v > 1.0)
    {
        return result; // No intersection
    }

    // Calculate t (intersection distance)
    float t = edge2.dot(Q) * inv_det;

    // If t > 0, the intersection is valid
    if (t > 0.0)
    {
        result.t = t;
        result.p = ray.origin + ray.direction * t;
        result.hittable = (void *)this;
        result.normal = edge1.cross(edge2).normalize();
        result.material = material;
    }

    return result;
}
