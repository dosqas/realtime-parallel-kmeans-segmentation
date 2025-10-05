#include "video_io.hpp"
#include <opencv2/opencv.hpp>
#include "clustering.hpp"
#include <iostream>
#include <string>
#include <deque>

#ifdef _WIN32 // If on Windows, include Windows.h for window style manipulation
#include <windows.h>
#endif

// Display webcam feed with real-time k-means segmentation and an adjustable 'K' parameter slider
void showWebcamFeed() 
{
	cv::VideoCapture cap(0); // Open the default camera
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera." << std::endl;
        return;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    cv::Mat frame;

    int k_trackbar = 5;
	int last_k_trackbar = k_trackbar;
    const int k_min = 2;
    const int k_max = 12;
    const int sample = 2000;
    const float color_scale = 1.0f;
    const float spatial_scale = 0.5f;

    const std::string windowName = "Realtime Segmentation";
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::createTrackbar("k", windowName, &k_trackbar, k_max);
    cv::setTrackbarMin("k", windowName, k_min);

#ifdef _WIN32
    // Disable window resizing and maximize button using Windows API
	// (OpenCV does not provide direct functionality for this)
	HWND hwnd = FindWindow(NULL, windowName.c_str()); // Get the window handle
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

    std::deque<std::pair<double, double>> fpsHistory; // (timestamp, fps)
    double minFps = 0.0, maxFps = 0.0;
    Backend backend = BACKEND_SEQ;
    Backend lastBackend = backend;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        if (cv::getWindowProperty(windowName, cv::WND_PROP_VISIBLE) < 1) {
			break; // Exit if window is closed
        }

        int k = (std::max)(k_min, (std::min)(k_max, k_trackbar));

        cv::Mat seg;
        seg = segmentFrameWithKMeans(frame, k, sample, backend, color_scale, spatial_scale);

        if (seg.size() != frame.size()) {
			cv::resize(seg, seg, frame.size(), 0, 0, cv::INTER_NEAREST); // Resize segmented image to match original frame size
        }

        cv::Mat combined(frame.rows, frame.cols * 2, frame.type());
		frame.copyTo(combined(cv::Rect(0, 0, frame.cols, frame.rows))); // Original on the left
		seg.copyTo(combined(cv::Rect(frame.cols, 0, frame.cols, frame.rows))); // Segmented on the right

        int64 now = cv::getTickCount();
        double nowSec = now / cv::getTickFrequency();
        double dt = (now - lastTick) / cv::getTickFrequency();
        lastTick = now;
        if (dt > 0) fps = 1.0 / dt;

        // Reset FPS history if backend changed or if K changed
        if (backend != lastBackend || k_trackbar != last_k_trackbar) {
            fpsHistory.clear();
            minFps = maxFps = fps;
            lastBackend = backend;
			last_k_trackbar = k_trackbar;
        }

        // Add current FPS to history
        fpsHistory.emplace_back(nowSec, fps);

        // Remove old FPS values (older than 3 seconds)
        while (!fpsHistory.empty() && nowSec - fpsHistory.front().first > 3.0) {
            fpsHistory.pop_front();
        }

        // Compute min/max FPS in the last 3 seconds
        minFps = maxFps = fps;
        for (const auto& p : fpsHistory) {
            if (p.second < minFps) minFps = p.second;
            if (p.second > maxFps) maxFps = p.second;
        }

        std::string backendName = (backend == BACKEND_SEQ ? "SEQ" : backend == BACKEND_CUDA ? "CUDA" : backend == BACKEND_THR ? "THR" : "MPI");
        std::string overlay = "k=" + std::to_string(k) +
            "  backend=" + backendName +
            "  FPS=" + cv::format("%.1f", fps) +
            "  min=" + cv::format("%.1f", minFps) +
            "  max=" + cv::format("%.1f", maxFps);
        cv::putText(combined, overlay, cv::Point(12, 28), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
        cv::putText(combined, overlay, cv::Point(12, 28), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

        cv::putText(combined, "Original", cv::Point(12, frame.rows - 12), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        cv::putText(combined, "Segmented", cv::Point(frame.cols + 12, frame.rows - 12), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

        cv::imshow(windowName, combined);

        char c = (char)cv::waitKey(1);
		if (c == 27) break; // Exit on ESC key
		if (c == '1') backend = BACKEND_SEQ;  // Switch to sequential backend if '1' is pressed
		if (c == '2') backend = BACKEND_MPI;  // Switch to MPI backend if '2' is pressed
		if (c == '3') backend = BACKEND_THR;  // Switch to threaded backend if '3' is pressed
		if (c == '4') backend = BACKEND_CUDA; // Switch to CUDA backend if '4' is pressed
    }

	cap.release(); // Clean up camera
	cv::destroyAllWindows(); // Close all OpenCV windows
}