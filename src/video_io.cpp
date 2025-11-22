#include "video_io.hpp"
#include <opencv2/opencv.hpp>
#include "clustering.hpp"
#include "utils.hpp"
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
        cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
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

    bool useThreadPool = false; // Toggle flag for thread pool

    while (true) {
        if (rank == 0) {
            cap >> frame;
            if (frame.empty()) break; // Exit if window is closed
        }

        int k = k_trackbar;

        cv::Mat localFrame;
        cv::Mat seg;
        int totalRows = 0, totalCols = 0;

        if (backend == BACKEND_MPI)
        {
            // Broadcast frame dimensions
            totalRows = (rank == 0 ? frame.rows : 0);
            totalCols = (rank == 0 ? frame.cols : 0);
            int type = (rank == 0 ? frame.type() : 0);

            MPI_Bcast(&totalRows, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&totalCols, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&type, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&k, 1, MPI_INT, 0, MPI_COMM_WORLD);

            if (totalRows == 0 || totalCols == 0)
                break;

            // Compute KMeans centers on rank 0
            // We do it here since we need the centers based on the whole frame,
            // so we cannot include it in the MPI's segment frame with K means function
            std::vector<float> flatCenters(k * 5);

            if (rank == 0) {
                auto centers = computeKMeansCenters(frame, k, sample, color_scale, spatial_scale);

                for (int i = 0; i < k; ++i)
                    for (int d = 0; d < 5; ++d)
                        flatCenters[static_cast<std::vector<float, std::allocator<float>>::size_type>(i) * 5 + d] = centers[i][d];
            }

            // Broadcast the centers to all ranks
            MPI_Bcast(flatCenters.data(), k * 5, MPI_FLOAT, 0, MPI_COMM_WORLD);

            // Scatter local rows to each rank
            int rowsPerRank = totalRows / size;
            std::vector<int> sendCounts(size), displs(size);

            // Prepare arrays for Scatterv: number of elements and offsets per rank
            for (int i = 0; i < size; ++i) {
                int sRow = i * rowsPerRank;
                int eRow = (i == size - 1 ? totalRows : sRow + rowsPerRank);
                sendCounts[i] = (eRow - sRow) * totalCols * 3;
                displs[i] = sRow * totalCols * 3;
            }

            // Calculate number of rows per rank
            int startRow = rank * rowsPerRank;
            int endRow = (rank == size - 1 ? totalRows : startRow + rowsPerRank);
            int localRows = endRow - startRow;

            localFrame.create(localRows, totalCols, type);

            // Scatter frame chunks from rank 0 to all ranks
            MPI_Scatterv(
                rank == 0 ? frame.data : nullptr,  // send buffer on root
                sendCounts.data(),                 // elements per rank
                displs.data(),                     // offsets per rank
                MPI_UNSIGNED_CHAR,                 // data type
                localFrame.data,                   // receive buffer
                sendCounts[rank],                  // elements received by this rank
                MPI_UNSIGNED_CHAR,                 // data type
                0,                                 // root rank
                MPI_COMM_WORLD
            );

            // Do segmentation using localFrame + received centers
            seg = segmentFrameWithKMeans(
                localFrame,
                k,
                backend,
                sample,
                color_scale,
                spatial_scale,
                flatCenters,
                totalRows,
                totalCols
            );
        }
        else {
            // Only rank 0 runs segmentation for non-MPI backends
            if (rank == 0) {
                seg = segmentFrameWithKMeans(frame, k, backend, sample, color_scale, spatial_scale);
            }
        }

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

            // Reset FPS history if backend changed
            if (backend != lastBackend) {
                fpsHistory.clear();
                minFps = maxFps = fps;

                // If we were using THRPOOL and now switched to something that's not THR, reset useThreadPool
                if (lastBackend == BACKEND_THRPOOL && backend != BACKEND_THR) {
                    useThreadPool = false;
                }

                lastBackend = backend;
                last_k_trackbar = k_trackbar;
            }


            // Reset FPS history if K changed
            if (k_trackbar != last_k_trackbar) {
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
                (backend == BACKEND_SEQ ? "SEQ" : backend == BACKEND_THR ? "THR"
                    : backend == BACKEND_MPI ? "MPI" : backend == BACKEND_CUDA ? "CUDA" : "THRPOOL");

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
            if (c == '2' && backend != BACKEND_THRPOOL) backend = BACKEND_THR;
            if (c == '3') backend = BACKEND_MPI;
            if (c == '4') backend = BACKEND_CUDA;

            if (c == '*' && (backend == BACKEND_THR || backend == BACKEND_THRPOOL)) {
                useThreadPool = !useThreadPool;
                useThreadPool ? backend = BACKEND_THRPOOL : backend = BACKEND_THR;
            }
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
