#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cstdlib>
// #define PI 3.14159265358979323846
class CannyEdgeDetector
{

public:
    CannyEdgeDetector(const std::string &imagePath, int lowThreshold, int upperThreshold);

    cv::Mat loadImage(const std::string &imagePath);
    void applyGaussianBlur();
    void findGradient();
    void displayImage();
    void applyNonMaxSuppression();
    std::vector<std::vector<double>> createGaussianFilter(int size, double sigma);

private:
    cv::Mat srcImage;
    int lowThreshold, upperThreshold;
};
