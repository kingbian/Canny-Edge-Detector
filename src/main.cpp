#include "edge_detector.h"

void showOptions() {
    std::cout << "\nUsage: program [options]: \n"
              << "\t-h\tshow help menu\n"
              << "\t-p <path to image> \tfind edges ina given image\n"
              << "\t-v <path to video>\tfind edges in a given video\n";
}

int main(int argc, char* argv[]) {
    for (int i = 0; i < argc; i++) {
        std::cout << "[" << i << "]:" << argv[i] << "\n";
    }

    if (argc < 2) {
        std::cout << "\nNot enough arguments provided\n";
        showOptions();
        exit(EXIT_FAILURE);
    }
    for (int i = 1; i < argc; ++i) {
        std::string opt = argv[i];

        if (opt == "-h") {
            showOptions();
            exit(EXIT_SUCCESS);
        } else if (opt == "-p") {
            std::cout << "Image option is selected\n";
            if (argv[i + 1] != nullptr) {
                std::string imagePath = argv[i + 1];

                CannyEdgeDetector edge{imagePath};
            } else {
                std::cout << "Please enter a valid image path\n";
                exit(EXIT_FAILURE);
            }
        } else if (opt == "-v") {
            std::cout << "Video mode selected\n";

            if (argv[i + 1] != nullptr) {
                std::string vidPath = argv[i + 1];

                CannyEdgeDetector edge{vidPath};
            } else {
                std::cout << "Please enter a valid image path\n";
                exit(EXIT_FAILURE);
            }
        }
    }

    // CannyEdgeDetector edge{"../images/before/lizard.jpg"};
    // CannyEdgeDetector edge{"../images/before/person.jpg", 30, 130};

    return 0;
}
