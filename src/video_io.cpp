#include "video_io.hpp"
#include <opencv2/opencv.hpp>
#include "clustering.hpp"
#include <iostream>
#include <string>
#include <deque>
#include <mpi.h>

#ifdef _WIN32 // If on Windows, include Windows.h for window style manipulation
#include <windows.h>
#endif

// Display webcam feed with real-time k-means segmentation and an adjustable 'K' parameter slider
void showWebcamFeed()
{
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    cv::VideoCapture cap;
    if (rank == 0) {
        cap.open(0); // Open webcam only on rank 0
        if (!cap.isOpened()) {
            std::cerr << "Error: Could not open camera." << std::endl;
            return;
        }

        cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    }

    cv::Mat frame;
    int k_trackbar = 5;
    const int k_min = 2;
    const int k_max = 12;
    const int sample = 2000;
    const float color_scale = 1.0f;
    const float spatial_scale = 0.5f;

    const std::string windowName = "Realtime Segmentation";

    // GUI initialization only on rank 0
    if (rank == 0) {
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
    }

    int64 lastTick = cv::getTickCount();
    double fps = 0.0;
    std::deque<std::pair<double, double>> fpsHistory; // (timestamp, fps)
    double minFps = 0.0, maxFps = 0.0;

    Backend backend = BACKEND_SEQ;
    Backend lastBackend = backend;
    int last_k_trackbar = k_trackbar;

    while (true) {
        if (rank == 0) {
            cap >> frame;
            if (frame.empty()) break; // Exit if window is closed
        }

        // Broadcast frame dimensions
        int rows = (rank == 0 ? frame.rows : 0);
        int cols = (rank == 0 ? frame.cols : 0);
        int type = (rank == 0 ? frame.type() : 0);
		// Bcast rows, cols, and type to all ranks
		// The root (in our case 0), is the one sending the data to all the other processes
		// Other ranks will receive the data
		// All the processes have to call Bcast so that MPI knows who is participating in the communication
        MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&type, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (rows == 0 || cols == 0) break;

        if (rank != 0) frame.create(rows, cols, type);
        MPI_Bcast(frame.data, rows * frame.step, MPI_BYTE, 0, MPI_COMM_WORLD);

        // Get current K from trackbar (only on rank 0)
        int k = k_trackbar;
        MPI_Bcast(&k, 1, MPI_INT, 0, MPI_COMM_WORLD);

        cv::Mat seg = segmentFrameWithKMeans(frame, k, sample, backend, color_scale, spatial_scale);

        if (rank == 0) {
            if (seg.size() != frame.size()) {
                cv::resize(seg, seg, frame.size(), 0, 0, cv::INTER_NEAREST);
            }

            cv::Mat combined(frame.rows, frame.cols * 2, frame.type());
            frame.copyTo(combined(cv::Rect(0, 0, frame.cols, frame.rows)));
            seg.copyTo(combined(cv::Rect(frame.cols, 0, frame.cols, frame.rows)));

            // FPS tracking
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

            // Update FPS history
            fpsHistory.emplace_back(nowSec, fps);

            // Remove old FPS values (older than 3 seconds)
            while (!fpsHistory.empty() && nowSec - fpsHistory.front().first > 3.0)
                fpsHistory.pop_front();

            minFps = maxFps = fps;
            for (const auto& p : fpsHistory) {
                if (p.second < minFps) minFps = p.second;
                if (p.second > maxFps) maxFps = p.second;
            }

            std::string backendName =
                (backend == BACKEND_SEQ ? "SEQ" : backend == BACKEND_CUDA ? "CUDA"
                    : backend == BACKEND_THR ? "THR" : "MPI");

            std::string overlay = "k=" + std::to_string(k) +
                "  backend=" + backendName +
                "  FPS=" + cv::format("%.1f", fps) +
                "  min=" + cv::format("%.1f", minFps) +
                "  max=" + cv::format("%.1f", maxFps);

            cv::putText(combined, overlay, cv::Point(12, 28),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);
            cv::putText(combined, overlay, cv::Point(12, 28),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

            cv::putText(combined, "Original", cv::Point(12, frame.rows - 12),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
            cv::putText(combined, "Segmented", cv::Point(frame.cols + 12, frame.rows - 12),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

            cv::imshow(windowName, combined);

            char c = (char)cv::waitKey(1);
            if (c == 27) break; // ESC
            if (c == '1') backend = BACKEND_SEQ;
            if (c == '2') backend = BACKEND_THR;
            if (c == '3') backend = BACKEND_MPI;
            if (c == '4') backend = BACKEND_CUDA;
        }

        // Broadcast backend and break state
        MPI_Bcast(&backend, 1, MPI_INT, 0, MPI_COMM_WORLD);
        int stopFlag = 0;
        if (rank == 0 && (cv::getWindowProperty(windowName, cv::WND_PROP_VISIBLE) < 1))
            stopFlag = 1;
        MPI_Bcast(&stopFlag, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (stopFlag) break;
    }

    if (rank == 0) {
        cap.release(); // Clean up camera
        cv::destroyAllWindows(); // Close all OpenCV windows
    }
}
