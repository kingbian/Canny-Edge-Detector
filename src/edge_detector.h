#pragma once
#include <cstdlib>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <queue>
#include <vector>

typedef struct {
    int upper;
    int lower;
} threshold;

// #define PI 3.14159265358979323846
class CannyEdgeDetector {
   public:
    CannyEdgeDetector(const std::string &imagePath);

    cv::Mat loadImage(const std::string &imagePath);
    void applyGaussianBlur();
    void findGradient(cv::Mat &dstImage);
    void displayImage(const cv::Mat &dstImage);
    void applyNonMaxSuppression(std::vector<std::vector<float>> &magnitudes,
                                std::vector<std::vector<float>> &gradientX,
                                std::vector<std::vector<float>> &gradientY,
                                float &maxMagnitude, int rows, int cols);
    std::vector<std::vector<float>> convolution(const cv::Mat &image,
                                                const float sobelOperator[][3],
                                                int rows, int cols);

    std::vector<std::vector<double>> createGaussianFilter(int size, double sigma);

    void applyHysteresis(std::vector<std::vector<float>> &magnitudes, int rows,
                         int cols);
    void trackEdge(std::vector<std::vector<float>> &magnitudes, int i, int j,
                   int rows, int cols);
    // cv::Mat convertToImage(const std::vector<std::vector<float>> &vec);
    void computeHistogram(const std::vector<std::vector<float>> &magnitudes);

   private:
    cv::Mat srcImage, dstImage;
    std::string imagePath;
    double lowThreshold = 76, upperThreshold = 230;
    // double lowThreshold = 0, upperThreshold = 0;

    int weakEdge = 100, strongEdge = 250, histogram[256];
};
