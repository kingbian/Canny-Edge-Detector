#include "edge_detector.h"
#include <math.h>

CannyEdgeDetector::CannyEdgeDetector(const std::string &imagePath,
                                     int lowThreshold, int upperThreshold)
    : lowThreshold(lowThreshold), upperThreshold(upperThreshold),
      srcImage(loadImage(imagePath))
{

  // srcImage = loadImage(imagePath);
}

cv::Mat CannyEdgeDetector::loadImage(const std::string &imagePath)
{
  std::cout << imagePath << "\n";
  std::cout << "about to read image \n";
  cv::Mat image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);

  if (image.empty())
  {
    std::cout << "Unable to load image from: " << imagePath
              << "\nEnter a key to continue\n";
    std::cin.get(); // wait for input before preceding

    exit(EXIT_FAILURE);
  }

  return image;
}

// create a 2D Gaussian kernel
std::vector<std::vector<double>>
CannyEdgeDetector::createGaussianFilter(int size, double sigma)
{

  // create a kernel of size
  std::vector<std::vector<double>> kernel(size, std::vector<double>(size, 0.0));

  double sum = 0.0;
  for (int i = -size / 2; i <= size / 2;
       ++i)
  { // ensure the center position of the kernel

    for (int j = -size / 2; j <= size / 2; ++j)
    {
      // calculate the gaussian distribution
      kernel[i + size / 2][j + size / 2] =
          (1 / (2 * M_PI * std::pow(sigma, 2))) *
          std::exp(
              -((std::pow(i, 2) + std::pow(j, 2)) / (2 * std::pow(sigma, 2))));

      sum += kernel[i + size / 2][j + size / 2];
    }
  }

  // normalize the kernel

  /**
   * why is this important?
   * when converting the gaussian distribution function into elements inside the
   * kernel, the sum of kernel element values will be different from 1 this can
   * cause darkening or bright spots int image which is not ideal and since this
   * a probability distribution function the total probability must be 1
   * Normalization ensures this happens
   */

  for (int i = 0; i < size; ++i)
  {

    for (int j = 0; j < size; ++j)
    {
      kernel[i][j] /= sum;
    }
  }

  // if(std::abs(sum -1.0) > 1e-6){
  //     std::cout<<"Error occurred\n";
  // }
  return kernel;
}

void display(const cv::Mat &image)
{

  cv::imshow("Display window", image);

  int input = cv::waitKey(0); // wait for input

  if (input == 's')
  {
    std::cout << "writing image\n";
  }
}

void CannyEdgeDetector::applyGaussianBlur()
{

  std::cout << "applying blur\n";
  std::vector<std::vector<double>> kernel = createGaussianFilter(7, 3.0);

  int x, y;
  x = y = 1;

  // cv::Mat dstImage;
  dstImage.create(srcImage.rows, srcImage.cols, srcImage.type());

  for (int i = 0; i < srcImage.rows; ++i)
  {

    for (int j = 0; j < srcImage.cols; ++j)
    {

      double convolvedPixel = 0.0;
      // iterate over the kernel rows
      for (int k = 0; k < kernel.size(); ++k)
      {

        // iterate over kernel cols
        for (int l = 0; l < kernel.size(); ++l)
        {

          // check to make sure were not out of bounds
          x = j - 1 + l;
          y = i + k - 1;
          if (x >= 0 && x < srcImage.cols && y >= 0 && y < srcImage.rows)
          {

            uchar temp = srcImage.at<uchar>(y, x);
            // std::cout << "current src image pixel: " << temp << "\n";
            convolvedPixel += temp * kernel[k][l];

            // std::cout << "convolved pixel: " << convolvedPixel << "\n";
          }
          else
          {
            // the kernel is out bounds and hanging of the image
            convolvedPixel += 0 * kernel[k][l];
          }
        }
        // add dis image here
        // std::cout << "final convolved pixel: " << convolvedPixel << "\n";

        if (convolvedPixel < 0)
          convolvedPixel = 0;
        // std::cout << "Intermediate convolved pixel at (" << i << ", " << j <<
        // "): " << convolvedPixel << "\n";

        uchar test =
            static_cast<uchar>(std::min(std::max(convolvedPixel, 0.0), 255.0));

        dstImage.at<uchar>(i, j) = test;
      }
    }
  }

  findGradient(dstImage);
  // cv::imwrite("personBlurred.jpg", dstImage);
  // if (input == 's')
  // {
  // }

  // display(dstImage);
}

void CannyEdgeDetector::findGradient(cv::Mat &dstImage)
{

  // cv::Mat dstImage = cv::imread("../images/cat.jpg", cv::IMREAD_GRAYSCALE);

  std::cout << "getting gradient\n";
  float sobel_X[3][3] = {{1, 0, -1},
                         {2, 0, -2},
                         {1, 0, -1}};

  float sobel_Y[3][3] = {{1, 2, 1},
                         {0, 0, 0},
                         {-1, -2, -1}};
  // get the gradient magnitudes

  int rows = dstImage.rows;
  int cols = dstImage.cols;

  std::vector<std::vector<float>> gradientX(rows, std::vector<float>(cols, 0));
  std::vector<std::vector<float>> gradientY(rows, std::vector<float>(cols, 0));
  std::vector<std::vector<float>> magnitudes(rows, std::vector<float>(cols, 0));

  // get the image intensity in the x direction
  for (int i = 1; i < rows - 1; ++i)
  {
    for (int j = 1; j < cols - 1; ++j)
    {
      // convolution here
      float sumX = 0;
      for (int k = -1; k <= 1; ++k)
      { // go through the rows of sobel x

        for (int l = -1; l <= 1; ++l)
        {

          sumX += dstImage.at<uchar>(i + k, j + l) * sobel_X[k + 1][l + 1];
        }
      }

      gradientX[i][j] = sumX;
    }
  }
  // get the y direction

  std::cout << "getting y directions gradient\n";
  for (int i = 1; i < rows - 1; ++i)
  {
    for (int j = 1; j < cols - 1; ++j)
    {
      // convolution here
      float sumY = 0;
      for (int k = -1; k <= 1; ++k)
      { // go through the rows of sobel x

        for (int l = -1; l <= 1; ++l)
        {

          sumY += dstImage.at<uchar>(i + k, j + l) * sobel_Y[k + 1][l + 1];
        }
      }

      gradientY[i][j] = sumY;
    }
  }

  std::cout << "Y DONE\n";

  float maxMagnitude = 0.0;
  for (int i = 0; i < rows; ++i)
  {

    for (int j = 0; j < cols; ++j)
    {

      float gradientXValue = gradientX[i][j];
      float gradientYValue = gradientY[i][j];

      float magnitude = std::sqrt(std::pow(gradientXValue, 2) + std::pow(gradientYValue, 2));

      if (magnitude > maxMagnitude)
        maxMagnitude = magnitude;

      magnitudes[i][j] = magnitude;
      // dstImage.at<uchar>(i, j) = static_cast<uchar>(std::min(255.0f, std::max(0.0f, magnitude)));
    }
  }

  for (int i = 0; i < rows; ++i)
  {

    for (int j = 0; j < cols; ++j)
    {

      float normalizedMag = magnitudes[i][j] / maxMagnitude * 255.0f;

      dstImage.at<uchar>(i, j) = static_cast<uchar>(normalizedMag);
    }
  }
  cv::imwrite("personBlurred1.jpg", dstImage);
}

// void CannyEdgeDetector::displayImage()
// {

//     cv::imshow("Display window", dstImage);

//     int input = cv::waitKey(0); // wait for input

//     if (input == 's')
//     {
//         std::cout << "writing image\n";
//     }
// }

void printKernel(const std::vector<std::vector<double>> &kernel)
{
  double sum = 0;
  for (int i = 0; i < kernel.size(); ++i)
  {
    for (int j = 0; j < kernel.size(); ++j)
    {
      std::cout << kernel[i][j] << " ";
      sum += kernel[i][j];
    }
    std::cout << "\n";
  }

  std::cout << "\nsum: " << sum;
}
