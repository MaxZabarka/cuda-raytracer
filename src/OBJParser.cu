#include "OBJParser.cuh"
#include <fstream>
#include <iostream>
#include "Vec3.cuh"
#include <vector>
#include "cuda-wrapper/cuda.cuh"
#include "BoundingBox.cuh"

OBJParser::OBJParser(
    std::string file_path)
    : file_path{file_path}
{
}

OBJParser::~OBJParser()
{
}

float OBJParser::parse_float(std::string line, size_t *line_index)
{
    std::string current_token;
    while (line[*line_index] == ' ')
    {
        (*line_index)++;
    }
    while (line[*line_index] != ' ' && *line_index < line.size())
    {
        current_token += line[*line_index];
        (*line_index)++;
    }
    return std::stof(current_token);
}
int OBJParser::parse_int(std::string line, size_t *line_index)
{
    std::string current_token;
    while (line[*line_index] == ' ' or line[*line_index] == '/')
    {
        (*line_index)++;
    }
    while (line[*line_index] != ' ' && line[*line_index] != '/' && *line_index < line.size())
    {
        current_token += line[*line_index];
        (*line_index)++;
    }
    if (current_token == "")
    {
        return -1;
    }
    return std::stoi(current_token);
}

VertexIndices OBJParser::parse_vertex_indices(std::string line, size_t *line_index)
{
    VertexIndices vertex_indices;
    vertex_indices.position_index = parse_int(line, line_index) - 1;
    vertex_indices.texturecoord_index = parse_int(line, line_index) - 1;
    vertex_indices.normal_index = parse_int(line, line_index) - 1;
    return vertex_indices;
}

HittableList OBJParser::parse()
{
    std::ifstream file(file_path);
    if (!file.is_open())
    {
        throw std::runtime_error("Could not open file: " + file_path);
    }

    std::vector<Triangle> triangles;
    std::vector<Point> vertex_positions;
    std::vector<Direction> vertex_normals;
    std::vector<Vec3> vertex_texture_coords;

    Point min_bounds = Point{INFINITY, INFINITY, INFINITY};
    Point max_bounds = Point{-INFINITY, -INFINITY, -INFINITY};

    std::string line;
    while (std::getline(file, line))
    {

        if (line[0] == '#')
        {
            continue;
        }
        std::string line_directive;
        size_t line_index = 0;

        while (line[line_index] != ' ')
        {
            line_directive += line[line_index];
            line_index++;
        }

        if (line_directive == "v")
        {
            float x = parse_float(line, &line_index);
            float y = parse_float(line, &line_index);
            float z = parse_float(line, &line_index);

            vertex_positions.push_back(Point(x, y, z));

            if (x > max_bounds.x)
            {
                max_bounds.x = x;
            }
            if (y > max_bounds.y)
            {
                max_bounds.y = y;
            }
            if (z > max_bounds.z)
            {
                max_bounds.z = z;
            }

            if (x < min_bounds.x)
            {
                min_bounds.x = x;
            }
            if (y < min_bounds.y)
            {
                min_bounds.y = y;
            }
            if (z < min_bounds.z)
            {
                min_bounds.z = z;
            }
        }
        else if (line_directive == "vn")
        {
            float x = parse_float(line, &line_index);
            float y = parse_float(line, &line_index);
            float z = parse_float(line, &line_index);

            vertex_normals.push_back(Direction(x, y, z));
        }
        else if (line_directive == "f")
        {
            VertexIndices vi_1 = parse_vertex_indices(line, &line_index);
            VertexIndices vi_2 = parse_vertex_indices(line, &line_index);
            VertexIndices vi_3 = parse_vertex_indices(line, &line_index);
            Vertex v1 = Vertex{vertex_positions[vi_1.position_index], vertex_normals[vi_1.normal_index]};
            Vertex v2 = Vertex{vertex_positions[vi_2.position_index], vertex_normals[vi_2.normal_index]};
            Vertex v3 = Vertex{vertex_positions[vi_3.position_index], vertex_normals[vi_3.normal_index]};

            TriangleData triangle_data = TriangleData{v1, v2, v3};
            Triangle triangle = Triangle{triangle_data, Material{FloatColor{0.5, 0.5, 0.5}}};

            triangles.push_back(triangle);
        }
    }

    Hittable **triangle_hittables = (Hittable **)cuda::mallocManaged(sizeof(Hittable *) * triangles.size());
    HittableList triangle_hittable_list = HittableList{triangle_hittables, triangles.size()};
    for (int i = 0; i < triangles.size(); i++)
    {
        triangle_hittables[i] = (Hittable *)cuda::mallocManaged(sizeof(Triangle));
        cuda::copyToDevice(triangle_hittables[i], &triangles[i], sizeof(Triangle));
        cuda::fixVirtualPointers<<<1, 1>>>((Triangle *)triangle_hittables[i]);
    }

    // print min bounds
    std::cout << "min bounds: " << min_bounds.x << ", " << min_bounds.y << ", " << min_bounds.z << std::endl;

    // print max bounds
    std::cout << "max bounds: " << max_bounds.x << ", " << max_bounds.y << ", " << max_bounds.z << std::endl;

    Hittable **hittables = (Hittable **)cuda::mallocManaged(sizeof(Hittable *) * 1);
    HittableList hittable_list = HittableList{hittables, 1};
    BoundingBox bounding_box = BoundingBox{triangle_hittable_list, min_bounds, max_bounds};

    hittables[0] = (Hittable *)cuda::mallocManaged(sizeof(BoundingBox));
    cuda::copyToDevice(hittables[0], &bounding_box, sizeof(BoundingBox));
    cuda::fixVirtualPointers<<<1, 1>>>((BoundingBox *)hittables[0]);

    // hittables[0] = bounding_box

    return hittable_list;
}