#include <iostream>
#include <string>
#include <nlohmann/json.hpp>
#include <fstream>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <vector>
#include "kernel.cuh"

struct Stock
{
    float S0; 
    float mu; 
    float sigma;
};

struct ProjectStock 
{ 
    Stock* stock;
    int num_paths;
    int num_steps;
    float T;
    float* paths;
};

struct Bar {
    std::time_t ts;
    double open, high, low, close, volume;
};


Stock* generate() { 
  

    Stock stock = {S0: S0, mu: mu, sigma: sigma};
    return &stock;
}



void run_paths(int num_paths, int num_steps, float T, std::string stock_name = "AAPL", unsigned long long seed = 42ULL) { 
    std::ifstream f("params.json");
    nlohmann::json j = nlohmann::json::parse(f);
    float S0    = j["S0"];
    float mu    = j["mu"];
    float sigma = j["sigma"];

    int num_elements = num_paths * (num_steps + 1);
    size_t bytes = (size_t)num_elements * sizeof(float);
    float* h_paths = (float*)malloc(bytes);

    float* d_paths; 
    cudaMalloc((void**)&d_paths, bytes);

    launchPaths(&d_paths, num_paths, num_steps, S0, mu, sigma, T, seed);

    cudaMemcpy(h_paths, d_paths, bytes, cudaMemcpyDeviceToHost);

    project->paths = h_paths;
    std::string file = "data/" + stock_name + ".csv";
    writeCsv(file, &h_paths, num_paths, num_steps + 1);
}


void writeCsv(const std::string& file, const float* data, int paths, int days) {
    std::ofstream f(file);
    if (!f) throw std::runtime_error("cannot open " + file);
    f << std::setprecision(8);

    f << "path";
    for (int j = 0; j < days; ++j) f << ",day " << j;
    f << '\n';

    for (int i = 0; i < paths; ++i) {
        f << "path " << i;
        for (int j = 0; j < days; ++j) f << ',' << data[i * days + j];
        f << '\n';
    }
}
