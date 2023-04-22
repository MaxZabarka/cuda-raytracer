#pragma once
#include "Vec3.cuh"
#include "Triangle.cuh"
#include "Infinity.cuh"

struct Geometry
{
    TriangleData *triangles;
    size_t num_triangles;
    Point min_bounds = Point{INFINITY, INFINITY, INFINITY};
    Point max_bounds = Point{-INFINITY, -INFINITY, -INFINITY};
};
