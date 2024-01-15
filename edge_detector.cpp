#include "edge_detector.h"
#include <math.h>

CannyEdgeDetector::CannyEdgeDetector(const std::string &imagePath,
                                     int lowThreshold, int upperThreshold)
    : lowThreshold(lowThreshold), upperThreshold(upperThreshold), imagePath(imagePath),
      srcImage(loadImage(imagePath))
{

  // srcImage = loadImage(imagePath);
  applyGaussianBlur();
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

  return kernel;
}

void CannyEdgeDetector::displayImage(const cv::Mat &image)
{

  cv::imshow("Display window", image);

  // int input = cv::waitKey(0); // wait for input
  cv::imwrite("../images/test.jpg", image);
  // if (input == 's')
  // {
  //   std::cout << "writing image\n";
  // }
}

void CannyEdgeDetector::applyGaussianBlur()
{

  std::vector<std::vector<double>> kernel = createGaussianFilter(3, 1.4);

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

            convolvedPixel += temp * kernel[k][l];
          }
          else
          {

            convolvedPixel += 0 * kernel[k][l];
          }
        }

        if (convolvedPixel < 0)
          convolvedPixel = 0;

        dstImage.at<uchar>(i, j) = static_cast<uchar>(std::min(std::max(convolvedPixel, 0.0), 255.0));
      }
    }
  }

  findGradient(dstImage);
}

void CannyEdgeDetector::findGradient(cv::Mat &dstImage)
{

  float sobel_X[3][3] = {{1, 0, -1},
                         {2, 0, -2},
                         {1, 0, -1}};

  float sobel_Y[3][3] = {{1, 2, 1},
                         {0, 0, 0},
                         {-1, -2, -1}};
  // get the gradient magnitudes

  int rows = dstImage.rows;
  int cols = dstImage.cols;

  std::vector<std::vector<float>> gradientX = convolution(dstImage, sobel_X, rows, cols);
  std::vector<std::vector<float>> gradientY = convolution(dstImage, sobel_Y, rows, cols);
  std::vector<std::vector<float>> magnitudes(rows, std::vector<float>(cols, 0));

  float maxMagnitude = 0.0;

  applyNonMaxSuppression(magnitudes, gradientX, gradientY, maxMagnitude, rows, cols);

  // normalize the gradient magnitude
  for (int i = 0; i < rows; ++i)
  {

    for (int j = 0; j < cols; ++j)
    {
      float normalizedMag = magnitudes[i][j] / maxMagnitude * 255.0f;

      dstImage.at<uchar>(i, j) = static_cast<uchar>(normalizedMag);
    }
  }

  displayImage(dstImage);
}

/**
 * calculate convolution NOTE: can be optimized
 */
std::vector<std::vector<float>> CannyEdgeDetector::convolution(const cv::Mat &image, const float sobelOperator[][3], int rows, int cols)
{

  std::vector<std::vector<float>> result(rows, std::vector<float>(cols, 0));

  // start 1 and subtract 1  to avoid going out of bounds
  for (int i = 1; i < rows - 1; ++i)
  {

    for (int j = 1; j < cols - 1; ++j)
    {
      float sum = 0;

      for (int k = -1; k <= 1; ++k)
      {

        for (int l = -1; l <= 1; ++l)
        {

          sum += image.at<uchar>(i + k, j + l) * sobelOperator[k + 1][l + 1];
        }
      }

      result[i][j] = sum;
    }
  }

  return result;
}

/**
 * apply Non-Maximum Suppression
 */
void CannyEdgeDetector::applyNonMaxSuppression(std::vector<std::vector<float>> &magnitudes,
                                               std::vector<std::vector<float>> &gradientX, std::vector<std::vector<float>> &gradientY,
                                               float &maxMagnitude, int rows, int cols)
{
  const float angleThreshold = 2.0; // account for numerical imprecisions that may occur while rounding gradientAngle

  for (int i = 1; i < rows - 1; ++i)
  {
    for (int j = 1; j < cols - 1; ++j)
    {

      float gradientXValue = gradientX[i][j];
      float gradientYValue = gradientY[i][j];

      float magnitude = std::sqrt(std::pow(gradientXValue, 2) + std::pow(gradientYValue, 2)); // find the gradient magnitude

      // find the gradient direction in degrees
      float gradientAngle = std::atan2(gradientYValue, gradientXValue) * 180 / M_PI;
      gradientAngle = std::round(gradientAngle); // round the result angle

      // get the neighboring pixels magnitudes
      float northWestDirection = magnitudes[i - 1][j - 1]; // north- west direction
      float northDirection = magnitudes[i - 1][j];         // north direction
      float northEastDirection = magnitudes[i - 1][j + 1]; // north-east direction
      float westDirection = magnitudes[i][j - 1];          // west in direction
      float eastDirection = magnitudes[i][j + 1];          // east in direction
      float southWestDirection = magnitudes[i + 1][j - 1]; // south-west direction
      float southDirection = magnitudes[i + 1][j];         // south direction
      float southEastDirection = magnitudes[i + 1][j + 1]; // south-east direction

      if ((gradientAngle == 0 || gradientAngle == 180) && (magnitude > westDirection && magnitude > eastDirection))
      {

        magnitudes[i][j] = magnitude;
      }
      else if ((std::abs((gradientAngle - 90) < angleThreshold)) && (magnitude > northDirection && magnitude > southDirection))
      {

        magnitudes[i][j] = magnitude;
      }
      else if ((std::abs(gradientAngle - 135) < angleThreshold) && (magnitude > northWestDirection && magnitude > southEastDirection))
      {

        magnitudes[i][j] = magnitude;
      }
      else if ((std::abs(gradientAngle - 45) < angleThreshold) && (magnitude > northEastDirection && magnitude > southWestDirection))
      {

        magnitudes[i][j] = magnitude;
      }
      else
      {

        magnitudes[i][j] = 0;
      }

      if (magnitude > maxMagnitude)
        maxMagnitude = magnitude; // get the max for normalization
    }
  }
}
