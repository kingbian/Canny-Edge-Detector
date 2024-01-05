#include "edge_detector.h"

CannyEdgeDetector::CannyEdgeDetector(const std::string &imagePath, int lowThreshold, int upperThreshold)
{

    // srcImage = loadImage(imagePath);
}

cv::Mat CannyEdgeDetector::loadImage(const std::string &imagePath)
{

    cv::Mat image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);

    if (image.empty())
    {
        std::cout << "Unable to load image from: " << imagePath << "\nEnter a key to continue\n";
        std::cin.get(); // wait for input before preceding

        exit(EXIT_FAILURE);
    }

    return image;
}

// create a 2D Gaussian kernel
std::vector<std::vector<double>> CannyEdgeDetector::createGaussianFilter(int size, double sigma)
{
    // create a kernel of size
    std::vector<std::vector<double>> kernel(size, std::vector<double>(size, 0.0));

    double sum = 0.0;
    for (int i = -size / 2; i <= size / 2; ++i)
    { // ensure the center position of the kernel

        for (int j = -size / 2; j <= size / 2; ++j)
        {
            // calculate the gaussian distribution
            kernel[i + size / 2][j + size / 2] = std::exp(-((std::pow(i, 2) + std::pow(j, 2)) / (2 * std::pow(sigma, 2))) / (2 * PI * std::pow(sigma, 2)));

            sum += kernel[i + size / 2][j + size / 2];
        }
    }

    // normalize the kernel

    /**
     * why is this important?
     * when converting the gaussian distribution function into elements inside the kernel,
     * the sum of kernel element values will be different from 1
     * this can cause darkening or bright spots int image which is not ideal
     * and since this a probability distribution function the total probability must be 1
     * Normalization ensures this happens
     */

    for (int i = 0; i < size; ++i)
    {

        for (int j = 0; j < size; ++j)
        {
            kernel[i][j] /= sum;
        }
    }

    std::cout << "\nsum: " << sum << "\n";

    // if(std::abs(sum -1.0) > 1e-6){
    //     std::cout<<"Error occurred\n";
    // }

    return kernel;
}
