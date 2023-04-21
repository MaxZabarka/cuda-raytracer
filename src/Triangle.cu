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
    // Vec3 position = Vec3({0, 0, 5});
    // float radius = 1;
    // Hit hit{INFINITY, Vec3{}, this};

    // // CO = O - C
    // Vec3 sphere_direction = ray.origin - position;

    // // < direction, direction >
    // float a = ray.direction.dot(ray.direction);

    // // 2 <CO, D>
    // float b = 2 * ray.direction.dot(sphere_direction);

    // // < CO, CO > - r^2
    // float c = sphere_direction.dot(sphere_direction) - pow(radius, 2);

    // // Solve the quadratic
    // float discriminant = b * b - 4 * a * c;

    // if (discriminant >= 0)
    // {
    //     hit.t = (-b - sqrt(discriminant)) / (2 * a);
    //     hit.p = ray.origin + (ray.direction * hit.t);
    // }
    // hit.material = material;
    // hit.normal = (hit.p - position).normalize();
    // return hit;

    // Hit hit{INFINITY, Vec3{}, this};
    // Vec3 edgeAB = b - a;
    // Vec3 edgeAC = c - a;
    // Vec3 normal = edgeAB.cross(edgeAC);
    // Vec3 ao = ray.origin - a;
    // Vec3 dao = ray.direction.cross(ao);

    // float det = -ray.direction.dot(normal);
    // float invDet = 1 / det;

    // float dst = ao.dot(normal) * invDet;
    // float u = edgeAC.dot(dao) * invDet;
    // float v = -edgeAB.dot(dao) * invDet;
    // float w = 1 - u - v;

    // if (dst > 0 && u >= 0 && v >= 0 && w >= 0)
    // {
    //     hit.hittable = this;
    //     hit.normal = normal;
    //     hit.material = material;
    // }

    // return hit;

    // const float EPSILON = 0.0000001;

    // Vec3 edge1 = b - a;
    // Vec3 edge2 = c - a;
    // Vec3 h = edge2.cross(ray.direction);
    // float a_dot_edge1 = edge1.dot(h);

    // if (a_dot_edge1 > -EPSILON && a_dot_edge1 < EPSILON)
    // {

    //     return Hit(); // Ray and triangle are parallel, no intersection
    // }

    // float f = 1.0f / a_dot_edge1;
    // Vec3 s = ray.origin - a;
    // float u = f * s.dot(h);

    // if (u < 0.0f || u > 1.0f)
    // {

    //     return Hit(); // Intersection is outside of the triangle
    // }

    // Vec3 q = s.cross(edge1);
    // float v = f * q.dot(ray.direction);

    // if (v < 0.0f || u + v > 1.0f)
    // {
    //     return Hit(); // Intersection is outside of the triangle
    // }

    // float t = f * edge2.dot(q);
    // if (t > EPSILON)
    // {
    //     Vec3 intersection_point = ray.origin + ray.direction * t;
    //     Vec3 triangle_normal = edge1.cross(edge2).normalize();
    //     return Hit({t, intersection_point, this, triangle_normal, material});
    // }

    // return Hit(); // Intersection is behind the ray's origin

     Hit result;
    result.t = std::numeric_limits<float>::max(); // Initialize with maximum possible value
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
