#include "OBJParser.cuh"
#include <fstream>
#include <iostream>
#include "Vec3.cuh"
#include <vector>
#include "Geometry.cuh"
#include "cuda-wrapper/cuda.cuh"

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

Geometry OBJParser::parse()
{
    std::ifstream file(file_path);
    if (!file.is_open())
    {
        throw std::runtime_error("Could not open file: " + file_path);
    }

    Geometry geometry = Geometry();

    std::vector<TriangleData> triangles;
    std::vector<Point> vertex_positions;
    std::vector<Direction> vertex_normals;
    std::vector<Vec3> vertex_texture_coords;

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

            if (x < geometry.min_bounds.x)
            {
                geometry.min_bounds.x = x;
            }
            if (y < geometry.min_bounds.y)
            {
                geometry.min_bounds.y = y;
            }
            if (z < geometry.min_bounds.z)
            {
                geometry.min_bounds.z = z;
            }

            if (x > geometry.max_bounds.x)
            {
                geometry.max_bounds.x = x;
            }
            if (y > geometry.max_bounds.y)
            {
                geometry.max_bounds.y = y;
            }
            if (z > geometry.max_bounds.z)
            {
                geometry.max_bounds.z = z;
            }

            vertex_positions.push_back(Point(x, y, z));
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

            triangles.push_back(triangle_data);
        }
    }

    TriangleData* arr_triangles = (TriangleData*)cuda::mallocManaged(triangles.size() * sizeof(TriangleData));
    std::copy(triangles.begin(), triangles.end(), arr_triangles);
    geometry.triangles = arr_triangles;
    geometry.num_triangles = triangles.size();

    // Print bounds
    std:: cout << "Min bounds: " << geometry.min_bounds.x << ", " << geometry.min_bounds.y << ", " << geometry.min_bounds.z << std::endl;
    std:: cout << "Max bounds: " << geometry.max_bounds.x << ", " << geometry.max_bounds.y << ", " << geometry.max_bounds.z << std::endl;

    // // Print first triangle data
    // std:: cout << "Triangle 1: " << std::endl;
    // std:: cout << "Vertex 1: " << std::endl;
    // std:: cout << "Position: " << triangles[0].v1.position.x << ", " << triangles[0].v1.position.y << ", " << triangles[0].v1.position.z << std::endl;
    // std:: cout << "Vertex 2: " << std::endl;
    // std:: cout << "Position: " << triangles[0].v2.position.x << ", " << triangles[0].v2.position.y << ", " << triangles[0].v2.position.z << std::endl;
    // std:: cout << "Vertex 3: " << std::endl;
    // std:: cout << "Position: " << triangles[0].v3.position.x << ", " << triangles[0].v3.position.y << ", " << triangles[0].v3.position.z << std::endl;

    // Print the vertex normals of the first triangle
    // std::cout << "Triangle 1: " << std::endl;
    // std::cout << "Vertex 1: " << std::endl;
    // std::cout << "Normal: " << triangles[0].a.normal.x << ", " << triangles[0].a.normal.y << ", " << triangles[0].a.normal.z << std::endl;
    // std::cout << "Vertex 2: " << std::endl;
    // std::cout << "Normal: " << triangles[0].b.normal.x << ", " << triangles[0].b.normal.y << ", " << triangles[0].b.normal.z << std::endl;
    // std::cout << "Vertex 3: " << std::endl;
    // std::cout << "Normal: " << triangles[0].c.normal.x << ", " << triangles[0].c.normal.y << ", " << triangles[0].c.normal.z << std::endl;
    
    
    return geometry;
}