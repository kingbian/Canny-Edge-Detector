#include "edge_detector.h"
#include <iostream>
#include <vector>

int main()
{

  int size = 5;
  double sigma = 1.0;

  CannyEdgeDetector edge{"../images/before/lizard.jpg", 76, 230};

  return 0;
}