#include "edge_detector.h"
#include <iostream>
#include <vector>

int main()
{

  int size = 5;
  double sigma = 1.0;

  CannyEdgeDetector edge{"../images/cat.jpg", 12, 12};
  // CannyEdgeDetector edgeDetector("./images/cat.jpg", 10, 20);

  // edgeDetector.applyGaussianBlur();
  edge.applyGaussianBlur();
  std::cout << "Test";
  return 0;
}