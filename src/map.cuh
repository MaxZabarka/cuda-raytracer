__device__ float map(float input, float input_start, float input_end, float output_start, float output_end)
{
    return output_start + (output_end - output_start) * ((input - input_start) / (input_end - input_start));
}