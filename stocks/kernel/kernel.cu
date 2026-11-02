#pragma once
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>
#include <iostream>

#ifndef CHECK_CUDA
#define CHECK_CUDA(err) \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
    }
#endif

__global__ void stocks(
  float *d_paths,
  float S0,
  float drift,
  float vol,
  int steps,
  unsigned long long seed,
  int num_paths
)
{
  int path_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (path_idx >= num_paths) return;

  curandState state;
  curand_init(seed, path_idx, 0, &state);

  float current_price = S0;

  int offset = path_idx * (steps + 1);
  d_paths[offset + 0] = current_price;

  for (int t = 1; t <= steps; ++t) {
    float Z = curand_normal(&state);

    current_price = current_price * expf(drift + vol * Z);

    d_paths[offset + t] = current_price;
  }
}

template <int threads_per_block = 256>
void launchPaths(float *d_paths, // paths x (steps + 1) matrix (including start day)
    int num_paths, int num_steps, // matrix dims
    float S0, float mu, float sigma, float T, unsigned long long seed // market parameters
)
{
    float dt = T / (float)num_steps;
    float drift = (mu - (sigma * sigma) / 2.0f) * dt;
    float vol = sigma * sqrtf(dt);

    int blocks_per_grid = (num_paths + threads_per_block - 1) / threads_per_block;
    stocks<<<blocks_per_grid, threads_per_block>>>(d_paths, S0, drift, vol, num_steps, seed, num_paths);
    CHECK_CUDA(cudaGetLastError());
}