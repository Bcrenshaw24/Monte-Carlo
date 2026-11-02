#include "stocks.cpp"

int main() {
  run_paths(10000, 252, 1.0);
  std::cout << "Simulation pipeline completed successfully." << std::endl;
  return 0;
}