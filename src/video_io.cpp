#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, char** argv) {
    // Use default camera (index 0)
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera." << std::endl;
        return -1;
    }

    // Optionally set resolution
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    cv::Mat frame;
    while (true) {
        // Grab frame
        cap >> frame;
        if (frame.empty()) break;

        // Show frame
        cv::imshow("Webcam Feed", frame);

        // Exit on ESC key
        char c = (char)cv::waitKey(1);
        if (c == 27) break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
