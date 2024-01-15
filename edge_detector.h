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
    void findGradient(cv::Mat &dstImage);
    void displayImage(const cv::Mat &dstImage);
    void applyNonMaxSuppression(std::vector<std::vector<float>> &magnitudes,
                                std::vector<std::vector<float>> &gradientX,
                                std::vector<std::vector<float>> &gradientY,
                                float &maxMagnitude, int rows, int cols);
    std::vector<std::vector<float>> convolution(const cv::Mat &image, const float sobelOperator[][3], int rows, int cols);

    std::vector<std::vector<double>> createGaussianFilter(int size, double sigma);

private:
    cv::Mat srcImage, dstImage;
    std::string imagePath;
    int lowThreshold, upperThreshold;
};
