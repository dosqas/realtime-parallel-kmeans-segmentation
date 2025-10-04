#include "video_io.hpp"
#include <opencv2/opencv.hpp>
#include "clustering.hpp"
#include <iostream>
#include <string>

#ifdef _WIN32
#include <windows.h>
#endif

void showWebcamFeed() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera." << std::endl;
        return;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    cv::Mat frame;

    int k_trackbar = 5;
    const int k_min = 1;
    const int k_max = 12;
    const int sample = 2000;
    const float color_scale = 1.0f;
    const float spatial_scale = 0.5f;

    Backend backend = BACKEND_SEQ;

    const std::string windowName = "Realtime Segmentation";
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::createTrackbar("k", windowName, &k_trackbar, k_max);
    cv::setTrackbarMin("k", windowName, k_min);

#ifdef _WIN32
    // Disable window resizing and maximize button using Windows API
    HWND hwnd = FindWindow(NULL, windowName.c_str());
    if (hwnd) {
        LONG style = GetWindowLong(hwnd, GWL_STYLE);
        style &= ~(WS_SIZEBOX | WS_MAXIMIZEBOX);
        style |= WS_BORDER;
        SetWindowLong(hwnd, GWL_STYLE, style);
        SetWindowPos(hwnd, NULL, 0, 0, 0, 0, 
                     SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_FRAMECHANGED);
    }
#endif

    int64 lastTick = cv::getTickCount();
    double fps = 0.0;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        if (cv::getWindowProperty(windowName, cv::WND_PROP_VISIBLE) < 1) {
            break; // Window was closed by the user
        }

        int k = (std::max)(k_min, (std::min)(k_max, k_trackbar));

        cv::Mat seg;
        seg = segmentFrameWithKMeans(frame, k, sample, backend, color_scale, spatial_scale);

        if (seg.size() != frame.size()) {
            cv::resize(seg, seg, frame.size(), 0, 0, cv::INTER_NEAREST);
        }

        cv::Mat combined(frame.rows, frame.cols * 2, frame.type());
        frame.copyTo(combined(cv::Rect(0, 0, frame.cols, frame.rows)));
        seg.copyTo(combined(cv::Rect(frame.cols, 0, frame.cols, frame.rows)));

        int64 now = cv::getTickCount();
        double dt = (now - lastTick) / cv::getTickFrequency();
        lastTick = now;
        if (dt > 0) fps = 1.0 / dt;

        std::string backendName = (backend == BACKEND_SEQ ? "SEQ" : backend == BACKEND_CUDA ? "CUDA" : backend == BACKEND_THR ? "THR" : "MPI");
        std::string overlay = "k=" + std::to_string(k) + "  backend=" + backendName + "  FPS=" + cv::format("%.1f", fps);
        cv::putText(combined, overlay, cv::Point(12, 28), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
        cv::putText(combined, overlay, cv::Point(12, 28), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

        cv::putText(combined, "Original", cv::Point(12, frame.rows - 12), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        cv::putText(combined, "Segmented", cv::Point(frame.cols + 12, frame.rows - 12), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

        cv::imshow(windowName, combined);

        char c = (char)cv::waitKey(1);
        if (c == 27) break;
        if (c == '1') backend = BACKEND_SEQ;
        if (c == '2') backend = BACKEND_THR;
        if (c == '3') backend = BACKEND_MPI;
        if (c == '4') backend = BACKEND_CUDA;
    }

    cap.release();
    cv::destroyAllWindows();
}