#pragma once
class Hittable;

struct Hit
{
    float t; // first intersection
             // no intersection = infinity

    void *hittable;
};