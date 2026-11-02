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


void launchPaths(float *d_paths, // paths x (steps + 1) matrix (including start day)
    int num_paths, int num_steps, // matrix dims
    float S0, float mu, float sigma, float T, unsigned long long seed // market parameters
);
