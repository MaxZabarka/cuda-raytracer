#include "OBJParser.cuh"
#include <fstream>
#include <iostream>
#include "Vec3.cuh"
#include <vector>
#include "cuda-wrapper/cuda.cuh"
#include "BoundingBox.cuh"
#include "ColorTexture.cuh"
#include "Material.cuh"
#include <SDL2/SDL_image.h>
#include <filesystem>
#include "ImageTexture.cuh"

namespace fs = std::filesystem;

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

ImageData OBJParser::load_image(std::string image_path)
{
    SDL_Surface *image_surface = IMG_Load(image_path.c_str());
    if (image_surface == NULL)
    {
        throw std::runtime_error("Unable to load image! SDL_image Error: " + std::string(IMG_GetError()));
    }

    ImageData image_data;
    image_data.width = image_surface->w;
    image_data.height = image_surface->h;
    SDL_PixelFormat *pixel_format = image_surface->format;
    Uint8 bpp = pixel_format->BytesPerPixel;

    image_data.data = (FloatColor *)cuda::mallocManaged(sizeof(FloatColor) * image_data.width * image_data.height);
    for (int y = 0; y < image_data.height; y++)
    {
        for (int x = 0; x < image_data.width; x++)
        {
            Uint8 *pixel = (Uint8 *)image_surface->pixels + y * image_surface->pitch + x * bpp;
            Uint32 pixel_color;

            switch (bpp)
            {
            case 1:
                pixel_color = *pixel;
                break;
            case 2:
                pixel_color = *(Uint16 *)pixel;
                break;
            case 3:
                if (SDL_BYTEORDER == SDL_BIG_ENDIAN)
                {
                    pixel_color = pixel[0] << 16 | pixel[1] << 8 | pixel[2];
                }
                else
                {
                    pixel_color = pixel[0] | pixel[1] << 8 | pixel[2] << 16;
                }
                break;
            case 4:
                pixel_color = *(Uint32 *)pixel;
                break;
            default:
                pixel_color = 0; // Should not happen, but avoids warnings
                break;
            }

            // Extract RGBA values
            Uint8 r, g, b, a;
            SDL_GetRGBA(pixel_color, pixel_format, &r, &g, &b, &a);

            image_data.data[y * image_data.width + x] = FloatColor(r / 255.0f, g / 255.0f, b / 255.0f);
        }
    }
    return image_data;

    SDL_FreeSurface(image_surface);
}

void OBJParser::parse_materials(std::string file_path)
{
    std::ifstream file(file_path);
    if (!file.is_open())
    {
        throw std::runtime_error("Could not open file: " + file_path);
    }

    fs::path folder_path = fs::absolute(file_path);
    folder_path.remove_filename();

    std::string line;
    std::string current_material = "";

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

        if (line_directive == "Kd")
        {
            std::cout << "Kd" << std::endl;
            std::cout << "material: " << current_material << std::endl;

            float x = parse_float(line, &line_index);
            float y = parse_float(line, &line_index);
            float z = parse_float(line, &line_index);

            ColorTexture *texture = (ColorTexture *)cuda::mallocManaged(sizeof(ColorTexture));
            texture->color = FloatColor(x, y, z);
            cuda::fixVirtualPointers<<<1, 1>>>(texture);
            materials[current_material].color = texture;
        }
        if (line_directive == "map_Kd")
        {
            while (line[line_index] == ' ')
            {
                line_index++;
            }
            ImageData image_data = load_image(folder_path.string() + line.substr(line_index, line.size() - line_index));
            ImageTexture *texture = (ImageTexture *)cuda::mallocManaged(sizeof(ImageTexture));
            texture->image_data = image_data;
            cuda::fixVirtualPointers<<<1, 1>>>(texture);
            materials[current_material].color = texture;
        }

        if (line_directive == "map_Bump") {
            while (line[line_index] == ' ')
            {
                line_index++;
            }
            ImageData image_data = load_image(folder_path.string() + line.substr(line_index, line.size() - line_index));
            ImageTexture *texture = (ImageTexture *)cuda::mallocManaged(sizeof(ImageTexture));
            texture->image_data = image_data;
            cuda::fixVirtualPointers<<<1, 1>>>(texture);
            materials[current_material].normal = texture;

        }

        if (line_directive == "newmtl")
        {
            current_material = "";
            while (line[line_index] == ' ')
            {
                line_index++;
            }
            while (line[line_index] != ' ' && line_index < line.size())
            {
                current_material += line[line_index];
                line_index++;
            }
        }
    }
    // // print materials

    // for (auto const &material : materials)
    // {
    //     std::cout << "material: " << material.first << std::endl;
    //     std::cout << "color: " << material.second.color.x << std::endl;
    // }
}

Material OBJParser::safe_get_material(std::string material_name)
{
    if (materials.find(material_name) == materials.end())
    {
        return Material();
    }
    return materials[material_name];
}

HittableList OBJParser::parse()
{
    parse_materials(file_path + ".mtl");
    std::ifstream file(file_path + ".obj");

    if (!file.is_open())
    {
        throw std::runtime_error("Could not open file: " + file_path);
    }

    std::vector<Triangle> triangles;
    std::vector<Point> vertex_positions;
    std::vector<Direction> vertex_normals;
    std::vector<Vec2> vertex_texture_coords;

    Point min_bounds = Point{INFINITY, INFINITY, INFINITY};
    Point max_bounds = Point{-INFINITY, -INFINITY, -INFINITY};

    std::string line;
    std::string current_material;

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
        if (line_directive == "usemtl")
        {
            current_material = "";
            while (line[line_index] == ' ')
            {
                line_index++;
            }
            while (line[line_index] != ' ' && line_index < line.size())
            {
                current_material += line[line_index];
                line_index++;
            }
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

            vertex_normals.push_back(Direction(x, y, z).normalize());
        }
        else if (line_directive == "vt")
        {
            float u = parse_float(line, &line_index);
            float v = parse_float(line, &line_index);

            vertex_texture_coords.push_back(Vec2(u, v));
        }
        else if (line_directive == "f")
        {
            VertexIndices vi_1 = parse_vertex_indices(line, &line_index);
            VertexIndices vi_2 = parse_vertex_indices(line, &line_index);
            VertexIndices vi_3 = parse_vertex_indices(line, &line_index);
            Vertex v1 = Vertex{vertex_positions[vi_1.position_index], vertex_normals[vi_1.normal_index], vertex_texture_coords[vi_1.texturecoord_index]};
            Vertex v2 = Vertex{vertex_positions[vi_2.position_index], vertex_normals[vi_2.normal_index], vertex_texture_coords[vi_2.texturecoord_index]};
            Vertex v3 = Vertex{vertex_positions[vi_3.position_index], vertex_normals[vi_3.normal_index], vertex_texture_coords[vi_3.texturecoord_index]};

            TriangleData triangle_data = TriangleData{v1, v2, v3};
            // std::cout << "Material: " << current_material << std::endl;
            Triangle triangle = Triangle{triangle_data, safe_get_material(current_material)};

            triangles.push_back(triangle);
        }
    }

    for (auto const &texture_coord : vertex_texture_coords)
    {
        std::cout << "texture coord: " << texture_coord.u << ", " << texture_coord.v << std::endl;
    }

    Hittable **triangle_hittables = (Hittable **)cuda::mallocManaged(sizeof(Hittable *) * triangles.size());
    HittableList triangle_hittable_list = HittableList{triangle_hittables, triangles.size()};
    for (int i = 0; i < triangles.size(); i++)
    {
        triangle_hittables[i] = (Hittable *)cuda::mallocManaged(sizeof(Triangle));
        cuda::copyToDevice(triangle_hittables[i], &triangles[i], sizeof(Triangle));
        cuda::fixVirtualPointers<<<1, 1>>>((Triangle *)triangle_hittables[i]);
    }

    Hittable **hittables = (Hittable **)cuda::mallocManaged(sizeof(Hittable *) * 1);
    HittableList hittable_list = HittableList{hittables, 1};
    BoundingBox bounding_box = BoundingBox{triangle_hittable_list, min_bounds, max_bounds};

    hittables[0] = (Hittable *)cuda::mallocManaged(sizeof(BoundingBox));
    cuda::copyToDevice(hittables[0], &bounding_box, sizeof(BoundingBox));
    cuda::fixVirtualPointers<<<1, 1>>>((BoundingBox *)hittables[0]);

    // hittables[0] = bounding_box

    return hittable_list;
}