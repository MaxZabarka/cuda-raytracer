#include <iostream>
#include <math.h>
#include <stdio.h>
#include "cuda-wrapper/cuda.cuh"

__global__
void add(float x[], float y[])
{

  int i = threadIdx.x + blockIdx.x * blockDim.x;  
  y[i] = 500.0f;
  // *(y+i) = 5.0f;
  // *(x+i) = ;
}

int test(void)
{
  int N = 1 << 10;

  float *x = (float *)cuda::mallocManaged(N * sizeof(float));
  float *y = (float *)cuda::mallocManaged(N * sizeof(float));

  for (int i = 0; i < N; i++)
  {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }


  int threadsPerBlock = 256;
  int blocksPerGrid =(N + threadsPerBlock - 1) / threadsPerBlock;

  add<<<blocksPerGrid, threadsPerBlock>>>(x, y);

  cudaDeviceSynchronize();
  // Verify that the result vector is correct
  for (int i = 0; i < N; i++) {
    if (fabs(y[i] - 3.0f) > 1e-5) {
      std::cout << "Error: " << y[i] << " " << 3.0f << std::endl;
      // break;
    }
  }

  cuda::free(x);
  cuda::free(y);

  return 0;

}