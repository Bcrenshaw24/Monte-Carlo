#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <math_constants.h>
#include <cmath>


__global__ void esitmatePi(
  unsigned long long *d_inside_count,
  unsigned long long trials_per_thread,
  unsigned long long seed
  )
  {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    curandState state;
    curand_init(seed, tid, 0, &state);

    unsigned long long local_inside = 0;

    for (unsigned long long i = 0; i < trials_per_thread; ++i) {
      float x = curand_uniform(&state);
      float y = curand_uniform(&state);

      if ((x * x + y * y) <= 1.0f) {
        local_inside++;
      }
    }
    atomicAdd(d_inside_count, local_inside);
  }

int main() {
  int threads_per_block = 256;
  int blocks_per_grid = 4096;
  unsigned long long trials_per_thread = 10000;

  unsigned long long total_threads = threads_per_block * blocks_per_grid;
  unsigned long long total_trials = total_threads * trials_per_thread;

  std::cout << "Launching " << total_threads << " concurrent threads..." << std::endl;
  std::cout << "Running a total of " << total_trials << " Monte Carlo trials." << std::endl;

  unsigned long long *d_inside_count;
  cudaMalloc((void**)&d_inside_count, sizeof(unsigned long long));
  cudaMemset(d_inside_count, 0, sizeof(unsigned long long));

  unsigned long long seed = 1234ULL;

  esitmatePi<<<blocks_per_grid, threads_per_block>>>(d_inside_count, trials_per_thread, seed);

  cudaDeviceSynchronize();

  unsigned long long h_inside_count = 0;
  cudaMemcpy(&h_inside_count, d_inside_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

  double estimate = 4.0 * (double)h_inside_count / (double)total_trials;

  double compare = M_PIf64;

  double diff = std::max(compare, estimate) - std::min(compare, estimate);
  double error = (diff / compare) * 100.0;

  std::cout << "Points inside circle: " << h_inside_count << "\n";
  std::cout << "Estimated PI Value: " << estimate << "\n";
  std::cout << "Error: " << error << "%\n";

  cudaFree(d_inside_count);
  return 0;
}