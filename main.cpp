#include "edge_detector.h"
#include <iostream>
#include <vector>

void printKernel(const std::vector<std::vector<double>> &kernel) {
  double sum = 0;
  for (int i = 0; i < kernel.size(); ++i) {
    for (int j = 0; j < kernel.size(); ++j) {
      std::cout << kernel[i][j] << " ";
      sum += kernel[i][j];
    }
    std::cout << "\n";
  }

  std::cout << "\nsum: " << sum;
}

int main() {

  int size = 5;
  double sigma = 1.0;

  CannyEdgeDetector edgeDetector("something", 10, 20);

  std::vector<std::vector<double>> kernel =
      edgeDetector.createGaussianFilter(size, sigma);

  printKernel(kernel);

  return 0;
}
