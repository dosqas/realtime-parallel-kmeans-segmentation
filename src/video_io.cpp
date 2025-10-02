#include "video_io.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>

void showWebcamFeed() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera." << std::endl;
        return;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        cv::imshow("Webcam Feed", frame);
        char c = (char)cv::waitKey(1);
        if (c == 27) break;
    }

    cap.release();
    cv::destroyAllWindows();
}