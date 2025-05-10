#include "edge_detector.h"

// CannyEdgeDetector::CannyEdgeDetector(const std::string &imagePath,
//                                      double lowThreshold, double
//                                      upperThreshold)
//     : lowThreshold(lowThreshold), upperThreshold(upperThreshold),
//       imagePath(imagePath), srcImage(loadImage(imagePath)) {
//
//   // srcImage = loadImage(imagePath);
//   applyGaussianBlur();
// }
//

/**
 * The constructor
 * initialize's srcImage variable
 */
CannyEdgeDetector::CannyEdgeDetector(const std::string &imagePath)
    : srcImage(loadImage(imagePath)) {
    applyGaussianBlur();
}
void CannyEdgeDetector::displayImage(const cv::Mat &image) {
    cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
    cv::moveWindow("Display window", 100, 100);
    cv::imshow("Display window", image);

    int input = cv::waitKey(0);  // wait for input
                                 // cv::imwrite("../images/test3.jpg", image);
                                 // if (input == 's')
                                 // {
                                 //   std::cout << "writing image\n";
                                 // }
}

cv::Mat CannyEdgeDetector::loadImage(const std::string &imagePath) {
    std::cout << imagePath << "\n";
    std::cout << "about to read image \n";
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);

    if (image.empty()) {
        std::cout << "Unable to load image from: " << imagePath
                  << "\nEnter a key to continue\n";
        std::cin.get();  // wait for input before preceding

        exit(EXIT_FAILURE);
    }

    std::cout << "image read";
    return image;
}

// create a 2D Gaussian kernel
std::vector<std::vector<double>>
CannyEdgeDetector::createGaussianFilter(int size, double sigma) {
    // create a kernel of size
    std::vector<std::vector<double>> kernel(size, std::vector<double>(size, 0.0));

    double sum = 0.0;
    for (int i = -size / 2; i <= size / 2;
         ++i) {  // ensure the center position of the kernel

        for (int j = -size / 2; j <= size / 2; ++j) {
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

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            kernel[i][j] /= sum;
        }
    }

    return kernel;
}

void CannyEdgeDetector::applyGaussianBlur() {
    std::cout << "applyin blur\n";
    std::vector<std::vector<double>> kernel = createGaussianFilter(5, 1.4);

    int x, y;
    x = y = 1;

    // cv::Mat dstImage;
    dstImage.create(srcImage.rows, srcImage.cols, srcImage.type());

    for (int i = 0; i < srcImage.rows; ++i) {
        for (int j = 0; j < srcImage.cols; ++j) {
            double convolvedPixel = 0.0;
            // iterate over the kernel rows
            for (int k = 0; k < kernel.size(); ++k) {
                // iterate over kernel cols
                for (int l = 0; l < kernel.size(); ++l) {
                    // check to make sure were not out of bounds
                    x = j - 1 + l;
                    y = i + k - 1;
                    if (x >= 0 && x < srcImage.cols && y >= 0 && y < srcImage.rows) {
                        uchar temp = srcImage.at<uchar>(y, x);

                        convolvedPixel += temp * kernel[k][l];
                    } else {
                        convolvedPixel += 0 * kernel[k][l];
                    }
                }
            }

            if (convolvedPixel < 0)
                convolvedPixel = 0;

            dstImage.at<uchar>(i, j) =
                static_cast<uchar>(std::min(std::max(convolvedPixel, 0.0), 255.0));
        }
    }
    std::cout << "blur done\n";

    findGradient(dstImage);
}

// Function to normalize and convert the 2D vector to a cv::Mat
cv::Mat convertToImage(const std::vector<std::vector<float>> &vec) {
    int rows = vec.size();
    int cols = vec[0].size();

    // Find the min and max values
    float minVal = vec[0][0], maxVal = vec[0][0];
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (vec[i][j] < minVal)
                minVal = vec[i][j];
            if (vec[i][j] > maxVal)
                maxVal = vec[i][j];
        }
    }

    // Create a Mat object
    cv::Mat img(rows, cols, CV_32F);

    // Copy data to Mat and normalize
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            img.at<float>(i, j) = (vec[i][j] - minVal) / (maxVal - minVal) * 255.0f;
        }
    }

    // Convert the float image to 8-bit image
    img.convertTo(img, CV_8U);
    // displayImage(img);
    return img;
}

void CannyEdgeDetector::findGradient(cv::Mat &dstImage) {
    std::cout << "finding gradient \n";
    float sobel_X[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};

    float sobel_Y[3][3] = {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};
    // get the gradient magnitudes

    int rows = dstImage.rows;
    int cols = dstImage.cols;

    std::vector<std::vector<float>> gradientX =
        convolution(dstImage, sobel_X, rows, cols);
    // displayImage(dstImage);
    std::vector<std::vector<float>> gradientY =
        convolution(dstImage, sobel_Y, rows, cols);
    std::vector<std::vector<float>> magnitudes(rows, std::vector<float>(cols, 0));

    float maxMagnitude = 0.0;

    std::cout << "starting non-max sup\n";
    applyNonMaxSuppression(magnitudes, gradientX, gradientY, maxMagnitude, rows,
                           cols);

    // make sure that maxMagnitude is valid
    if (maxMagnitude <= 0) {
        std::cerr << "Error occured: max  magnitude is zero, unable to normalize\n";
        return;
    }

    std::cout << "The max mag is: " << maxMagnitude << std::endl;
    std::cout << "Done non-max sup\n";
    // normalize the gradient magnitude
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            magnitudes[i][j] /= maxMagnitude;
            magnitudes[i][j] *= 255.0f;  // scale to be in 0-255
                                         // dstImage.at<uchar>(i, j) = static_cast<uchar>(normalizedMag);
        }
    }

    std::cout << "starting hystersis\n";
    computeHistogram(magnitudes);
    applyHysteresis(magnitudes, rows, cols);

    std::cout << "done hystersis\n";
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            dstImage.at<uchar>(i, j) = static_cast<uchar>(magnitudes[i][j]);
        }
    }

    std::cout << "done normalizing\n displaying image";
    displayImage(dstImage);
}

/**
 * calculate convolution NOTE: can be optimized
 */
std::vector<std::vector<float>> CannyEdgeDetector::convolution(
    const cv::Mat &image, const float sobelOperator[][3], int rows, int cols) {
    std::vector<std::vector<float>> result(rows, std::vector<float>(cols, 0));

    // start 1 and subtract 1  to avoid going out of bounds
    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            float sum = 0;

            for (int k = -1; k <= 1; ++k) {
                for (int l = -1; l <= 1; ++l) {
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
 * this is done to remove noise
 */
void CannyEdgeDetector::applyNonMaxSuppression(
    std::vector<std::vector<float>> &magnitudes,
    std::vector<std::vector<float>> &gradientX,
    std::vector<std::vector<float>> &gradientY, float &maxMagnitude, int rows,
    int cols) {
    // account for numerical imprecisions that
    // may occur while rounding gradientAngle
    constexpr const float angleThreshold = 3.0;

    for (int i = 1; i < rows - 1; ++i) {
        for (int j = 1; j < cols - 1; ++j) {
            float gradientXValue = gradientX[i][j];
            float gradientYValue = gradientY[i][j];

            /**
             * To find the gradient magnitude of each pixel use the law of
             Pythagoras
             */
            float magnitude =
                std::sqrt(std::pow(gradientXValue, 2) +
                          std::pow(gradientYValue, 2));  // find the gradient
                                                         // magnitude

            // find the gradient direction in degrees using the inverse tangent of
            // Gx/Gy *(180/PI)
            float gradientAngle =
                std::atan2(gradientYValue, gradientXValue) * 180 / M_PI;

            gradientAngle = std::round(gradientAngle);  // round the result angle

            // get the neighboring pixels magnitudes
            float northWestDirection =
                magnitudes[i - 1][j - 1];                 // north- west direction
            float northDirection = magnitudes[i - 1][j];  // north direction
            float northEastDirection =
                magnitudes[i - 1][j + 1];                // north-east direction
            float westDirection = magnitudes[i][j - 1];  // west  direction
            float eastDirection = magnitudes[i][j + 1];  // east  direction
            float southWestDirection =
                magnitudes[i + 1][j - 1];                 // south-west direction
            float southDirection = magnitudes[i + 1][j];  // south direction
            float southEastDirection =
                magnitudes[i + 1][j + 1];  // south-east direction

            if ((gradientAngle == 0 || gradientAngle == 180) &&
                (magnitude > westDirection && magnitude > eastDirection)) {
                magnitudes[i][j] = magnitude;
            } else if ((((gradientAngle - 90) < angleThreshold)) &&
                       (magnitude > northDirection && magnitude > southDirection)) {
                magnitudes[i][j] = magnitude;
            } else if ((std::abs(gradientAngle - 135) < angleThreshold) &&
                       (magnitude > northWestDirection &&
                        magnitude > southEastDirection)) {
                magnitudes[i][j] = magnitude;
            } else if ((std::abs(gradientAngle - 45) < angleThreshold) &&
                       (magnitude > northEastDirection &&
                        magnitude > southWestDirection)) {
                magnitudes[i][j] = magnitude;
            } else {
                magnitudes[i][j] = 0.0;
            }

            if (magnitude > maxMagnitude)
                maxMagnitude = magnitude;  // get the max for normalization

            // std::cout << "Magnitude: " << magnitude << " Angle: " <<
            // gradientAngle
            //           << std::endl;
        }
    }
}

/*
 * this function computes the histogram of the magnitudes vector
 * */
void CannyEdgeDetector::computeHistogram(
    const std::vector<std::vector<float>> &magnitudes) {
    for (const auto &row : magnitudes) {
        for (const auto &value : row) {
            int intensity = static_cast<int>(value);

            if (intensity >= 0 && intensity <= 256) {
                // std::cout << "intensity val: " << intensity << std::endl;
                histogram[intensity]++;
            }
        }
    }

    int totalPixles = magnitudes.size() * magnitudes[0].size();
    std::cout << "Total pixels in the image: " << totalPixles << std::endl;

    // use percentile to determine the threshold,  90th percentile for
    // upperThreshold
    int highCount = totalPixles * 0.1;
    int cumulativeCount = 0;

    int peakValue, peakIndx;
    peakIndx = peakValue = 0;

    for (int i = 256; i >= 0; --i) {
        cumulativeCount += histogram[i];

        /* if (histogram[i] > peakValue) {
          peakValue = histogram[i];
          peakIndx = i;
        } */

        if (cumulativeCount >= highCount) {
            upperThreshold = i;
            break;
        }
    }

    // upperThreshold = peakValue * 0.8;
    lowThreshold = upperThreshold * 0.5;
    std::cout << "Selected upperThreshold: " << upperThreshold << std::endl;
    std::cout << "Selected lowThreshold: " << lowThreshold << std::endl;
}

/**
 * apply double threshold and hysteresis
 */

void CannyEdgeDetector::applyHysteresis(
    std::vector<std::vector<float>> &magnitudes, int rows, int cols) {
    std::cout << "upper is: " << upperThreshold << " lower is: " << lowThreshold
              << "\n";
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (magnitudes[i][j] >= upperThreshold)

            {  // strong edge

                magnitudes[i][j] = 255;

                trackEdge(magnitudes, i, j, rows, cols);
            } else if (magnitudes[i][j] < upperThreshold &&
                       magnitudes[i][j] >= lowThreshold) {
                magnitudes[i][j] = 100;
            } else {
                magnitudes[i][j] = 0;
            }
        }
    }
}

void CannyEdgeDetector::trackEdge(std::vector<std::vector<float>> &magnitudes,
                                  int startPosI, int startPosY, int rows,
                                  int cols) {
    // std::vector<std::vector<bool>> visited;

    std::queue<std::pair<int, int>> queue;  // store pixel positions

    queue.push({startPosI, startPosY});  // this is the starting point

    while (!queue.empty()) {  // continue until the queue is empty

        // get the current pixel's position
        std::pair<int, int> currentPixelPos = queue.front();
        queue.pop();  // remove the position we just got

        int pixelPosAti = currentPixelPos.first;   // get the row
        int pixelPosAtj = currentPixelPos.second;  // get the col

        /**
         * check bounds and also check if
         * the current pixel is a weak edge
         * if so skip it
         */
        if (pixelPosAti < 0 || pixelPosAti >= rows || pixelPosAtj < 0 ||
            pixelPosAtj >= cols || magnitudes[pixelPosAti][pixelPosAtj] != 100) {
            continue;
        }

        magnitudes[pixelPosAti][pixelPosAtj] =
            255;  // update the pixel to be a strong edge

        // check for neighboring pixels that

        for (int i = -1; i <= 1; ++i) {
            for (int j = -1; j <= 1; ++j) {
                if (i == 0 && j == 0)
                    continue;  // skip the current pixel be processed

                int newPosI = pixelPosAti + i;
                int newPosY = pixelPosAtj + j;

                /**
                 * check if the new pixel (neighboring ) is a weak edge
                 * if it is added to queue to link to a strong edge
                 */
                if (newPosI >= 0 && newPosI < rows && newPosY >= 0 && newPosY < cols &&
                    magnitudes[newPosI][newPosY] == 100) {
                    queue.push({newPosI, newPosY});
                }
            }
        }
    }
}
