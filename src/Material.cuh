#pragma once
#include "Vec3.cuh"

// class Material
// {
// public:
//     virtual bool scatter(const Ray &r_in, const Hit &hit, Vec3 &attenuation, Ray &scattered) const = 0;
// };

struct Material
{
    FloatColor color;
};
