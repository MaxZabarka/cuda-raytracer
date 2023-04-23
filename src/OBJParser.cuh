#pragma once
#include <string>
#include "Vec3.cuh"
#include <vector>
#include "Triangle.cuh"
#include "HittableList.cuh"

struct VertexIndices
{
    size_t position_index;
    size_t texturecoord_index;
    size_t normal_index;
};


class OBJParser
{
public:
    OBJParser(std::string file_path);
    ~OBJParser();
    HittableList parse();
    std::string file_path;
    float parse_float(std::string line, size_t *line_index);
    int parse_int(std::string line, size_t *line_index);
    VertexIndices parse_vertex_indices(std::string line, size_t *line_index);
};
