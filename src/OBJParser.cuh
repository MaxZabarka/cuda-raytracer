#pragma once
#include <string>
#include "Vec3.cuh"
#include <vector>
<<<<<<< HEAD
#include "Triangle.cuh"
#include "Geometry.cuh"
=======
>>>>>>> parent of af5e36a (smooth shading)

struct VertexIndices
{
    size_t position_index;
    size_t texturecoord_index;
    size_t normal_index;
};

struct Vertex
{
    Point position;
    Vec3 texcoord;
    Direction normal;
};

struct TriangleData {
    Vertex v1;
    Vertex v2;
    Vertex v3;
};

class OBJParser
{
public:
    OBJParser(std::string file_path);
    ~OBJParser();
    Geometry parse();
    std::string file_path;
    float parse_float(std::string line, size_t *line_index);
    int parse_int(std::string line, size_t *line_index);
    VertexIndices parse_vertex_indices(std::string line, size_t *line_index);
};
