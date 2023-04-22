#include "OBJParser.cuh"
#include <fstream>
#include <iostream>
#include "Vec3.cuh"
#include <vector>

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

std::vector<TriangleData> OBJParser::parse()
{
    std::ifstream file(file_path);
    if (!file.is_open())
    {
        throw std::runtime_error("Could not open file: " + file_path);
    }

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

            vertex_positions.push_back(Point(x, y, z));

        }
        else if (line_directive == "f")
        {
            VertexIndices vi_1 = parse_vertex_indices(line, &line_index);
            VertexIndices vi_2 = parse_vertex_indices(line, &line_index);
            VertexIndices vi_3 = parse_vertex_indices(line, &line_index);
            Vertex v1 = Vertex{vertex_positions[vi_1.position_index]};
            Vertex v2 = Vertex{vertex_positions[vi_2.position_index]};
            Vertex v3 = Vertex{vertex_positions[vi_3.position_index]};

            TriangleData triangle_data = TriangleData{v1, v2, v3};

            triangles.push_back(triangle_data);

        }
    }
    return triangles;



    // // Print first triangle data
    // std:: cout << "Triangle 1: " << std::endl;
    // std:: cout << "Vertex 1: " << std::endl;
    // std:: cout << "Position: " << triangles[0].v1.position.x << ", " << triangles[0].v1.position.y << ", " << triangles[0].v1.position.z << std::endl;
    // std:: cout << "Vertex 2: " << std::endl;
    // std:: cout << "Position: " << triangles[0].v2.position.x << ", " << triangles[0].v2.position.y << ", " << triangles[0].v2.position.z << std::endl;
    // std:: cout << "Vertex 3: " << std::endl;
    // std:: cout << "Position: " << triangles[0].v3.position.x << ", " << triangles[0].v3.position.y << ", " << triangles[0].v3.position.z << std::endl;
    
}