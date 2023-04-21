#pragma once

void print_pointer_location(void *ptr)
{
    cudaPointerAttributes attributes;
    cudaError_t error = cudaPointerGetAttributes(&attributes, ptr);
    if (error != cudaSuccess)
    {
        printf("Error: %s\n", cudaGetErrorString(error));
        return;
    }

    if (attributes.type == cudaMemoryTypeHost)
    {
        printf("Pointer is in host memory\n");
    }
    else if (attributes.type == cudaMemoryTypeDevice)
    {
        printf("Pointer is in device memory\n");
    }
    else if (attributes.type == cudaMemoryTypeManaged)
    {
        printf("Pointer is in unified memory\n");
    }
    else
    {
        printf("Unknown memory type\n");
    }
}